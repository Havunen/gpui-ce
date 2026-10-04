//! Bounded native-backend path storage. Backends own textures and completion fences.
use crate::{
    gpu_policy::{RetentionBudget, RetentionLease, WINDOW_LAYER_BYTES},
    path_plan::{self, PixelRect},
    path_types::{self, PathRasterizationVertex},
};
use gpui::{Path, PrimitiveBatch, RenderCommand, ScaledPixels, Scene};
use std::sync::Arc;

pub struct Entry<T> {
    pub resource: T,
    pub bounds: PixelRect,
    _lease: RetentionLease,
    key: u64,
    vertices: Vec<PathRasterizationVertex>,
}

pub struct PathCache<T> {
    entries: Vec<Arc<Entry<T>>>,
    budget: RetentionBudget,
    warmed: usize,
    pub hits: u64,
    pub misses: u64,
}

pub enum Source<T> {
    Inline,
    Invisible,
    Cached(Arc<Entry<T>>),
    Packed { world: PixelRect, tile: PixelRect },
}

pub fn plan<T>(
    scene: &Scene,
    extent: (u32, u32),
    cache: &mut PathCache<T>,
    options: crate::gpu_policy::GpuOptions,
) -> Vec<Source<T>> {
    if !options.cached_layers && !options.batched_paths {
        return Vec::new();
    }
    let mut misses = Vec::new();
    let mut result: Vec<_> = scene
        .render_commands()
        .iter()
        .enumerate()
        .map(|(index, command)| {
            if let RenderCommand::Batch(PrimitiveBatch::Paths {
                range,
                rasterization_vertex_count,
                ..
            }) = command
            {
                if *rasterization_vertex_count == 0 {
                    return Source::Invisible;
                }
                let paths = &scene.paths[range.clone()];
                let Some(world) =
                    path_plan::visible_rect(paths.iter().map(Path::clipped_bounds), extent)
                else {
                    return Source::Invisible;
                };
                if options.cached_layers {
                    if let Some(entry) = cache.get(paths) {
                        return Source::Cached(entry);
                    }
                }
                misses.push((index, world));
            }
            Source::Inline
        })
        .collect();
    if options.batched_paths && !scene.requires_offscreen_rendering() {
        if let Some(tiles) = path_plan::pack(
            &misses.iter().map(|(_, world)| *world).collect::<Vec<_>>(),
            extent,
        ) {
            for ((index, world), tile) in misses.into_iter().zip(tiles) {
                result[index] = Source::Packed { world, tile };
            }
        }
    }
    result
}

impl<T> PathCache<T> {
    pub fn new(budget: RetentionBudget) -> Self {
        Self {
            entries: Vec::new(),
            budget,
            warmed: 0,
            hits: 0,
            misses: 0,
        }
    }
    pub fn begin(&mut self) {
        self.warmed = 0;
        self.hits = 0;
        self.misses = 0;
    }
    pub fn bytes(&self) -> u64 {
        self.entries.iter().map(|entry| entry.bounds.bytes()).sum()
    }
    pub fn clear(&mut self) {
        self.entries.clear();
    }
    pub fn get(&mut self, paths: &[Path<ScaledPixels>]) -> Option<Arc<Entry<T>>> {
        let key = path_plan::fingerprint(paths);
        let found = self.entries.iter().position(|entry| {
            entry.key == key
                && entry.vertices.len() == path_types::rasterization_vertex_count(paths)
                && entry
                    .vertices
                    .iter()
                    .cloned()
                    .eq(path_types::rasterization_vertices(paths))
        });
        match found {
            Some(index) => {
                self.hits += 1;
                let entry = self.entries.remove(index);
                self.entries.push(entry.clone());
                Some(entry)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }
    /// Admit before creating a texture. Callers retain returned entries until GPU completion,
    /// including an entry evicted by another capture in the same command submission.
    pub fn capture<E>(
        &mut self,
        paths: &[Path<ScaledPixels>],
        bounds: PixelRect,
        create: impl FnOnce() -> Result<T, E>,
    ) -> Result<Option<Arc<Entry<T>>>, E> {
        const GEOMETRY_BYTES: usize = 1024 * 1024;
        let geometry = path_types::rasterization_vertex_count(paths)
            .saturating_mul(std::mem::size_of::<PathRasterizationVertex>());
        if self.warmed >= 2
            || bounds.width == 0
            || bounds.height == 0
            || bounds.bytes() > WINDOW_LAYER_BYTES
            || geometry > GEOMETRY_BYTES
        {
            return Ok(None);
        }
        while !self.entries.is_empty()
            && (self.bytes().saturating_add(bounds.bytes()) > WINDOW_LAYER_BYTES
                || self
                    .entries
                    .iter()
                    .map(|e| std::mem::size_of_val(e.vertices.as_slice()))
                    .sum::<usize>()
                    .saturating_add(geometry)
                    > GEOMETRY_BYTES)
        {
            self.entries.remove(0);
        }
        let Some(lease) = self.budget.try_acquire(bounds.bytes()) else {
            return Ok(None);
        };
        let resource = create()?;
        let entry = Arc::new(Entry {
            resource,
            bounds,
            _lease: lease,
            key: path_plan::fingerprint(paths),
            vertices: path_types::rasterization_vertices(paths).collect(),
        });
        self.entries.push(entry.clone());
        self.warmed += 1;
        Ok(Some(entry))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_is_bounded_and_inflight_leases_survive_eviction() {
        let budget = RetentionBudget::new(128);
        let mut cache = PathCache::new(budget.clone());
        let bounds = PixelRect {
            width: 4,
            height: 4,
            ..Default::default()
        };
        let a = cache
            .capture(&[], bounds, || Ok::<_, ()>(1))
            .unwrap()
            .unwrap();
        cache
            .capture(&[], bounds, || Ok::<_, ()>(2))
            .unwrap()
            .unwrap();
        assert!(
            cache
                .capture(&[], bounds, || -> Result<i32, ()> { panic!("warm limit") })
                .unwrap()
                .is_none()
        );
        assert_eq!(budget.used(), 128);
        cache.clear();
        assert_eq!(budget.used(), 64);
        drop(a);
        assert_eq!(budget.used(), 0);
        cache.begin();
        assert!(cache.capture(&[], bounds, || Err(())).is_err());
        assert_eq!(budget.used(), 0);
    }
}
