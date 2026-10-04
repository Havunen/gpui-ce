//! Bounded storage for rasterized path batches, reused while their geometry is unchanged.
//! Backends own the textures and the fences that tell when the GPU is done with them.
use crate::{
    gpu_policy::{RetentionBudget, RetentionLease, WINDOW_LAYER_BYTES},
    path_plan::{self, PixelRect},
    path_types::{self, PathRasterizationVertex},
};
use gpui::{Path, PrimitiveBatch, RenderCommand, ScaledPixels, Scene};
use std::sync::Arc;

/// Cached geometry is compared vertex by vertex on lookup, so bound it independently
/// of the texture budget: a tiny raster may still hold arbitrarily complex tessellation.
const GEOMETRY_BYTES: u64 = 1024 * 1024;

pub struct Entry<T> {
    pub resource: T,
    pub bounds: PixelRect,
    _lease: RetentionLease,
    _window_lease: RetentionLease,
    _geometry_lease: RetentionLease,
    key: u64,
    vertices: Vec<PathRasterizationVertex>,
}

pub struct PathCache<T> {
    entries: Vec<Arc<Entry<T>>>,
    budget: RetentionBudget,
    window_budget: RetentionBudget,
    geometry_budget: RetentionBudget,
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
            window_budget: RetentionBudget::new(WINDOW_LAYER_BYTES),
            geometry_budget: RetentionBudget::new(GEOMETRY_BYTES),
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
        self.window_budget.used()
    }
    pub fn clear(&mut self) {
        self.entries.clear();
    }
    /// The cached entries, least recently used first.
    pub fn entries(&self) -> impl Iterator<Item = &Arc<Entry<T>>> {
        self.entries.iter()
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
        let geometry = (path_types::rasterization_vertex_count(paths) as u64)
            .saturating_mul(std::mem::size_of::<PathRasterizationVertex>() as u64);
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
                || self.geometry_budget.used().saturating_add(geometry) > GEOMETRY_BYTES)
        {
            self.entries.remove(0);
        }
        let Some(lease) = self.budget.try_acquire(bounds.bytes()) else {
            return Ok(None);
        };
        let Some(window_lease) = self.window_budget.try_acquire(bounds.bytes()) else {
            return Ok(None);
        };
        let Some(geometry_lease) = self.geometry_budget.try_acquire(geometry) else {
            return Ok(None);
        };
        let resource = create()?;
        let mut vertices = Vec::with_capacity(path_types::rasterization_vertex_count(paths));
        vertices.extend(path_types::rasterization_vertices(paths));
        let entry = Arc::new(Entry {
            resource,
            bounds,
            _lease: lease,
            _window_lease: window_lease,
            _geometry_lease: geometry_lease,
            key: path_plan::fingerprint(paths),
            vertices,
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
    fn window_limit_includes_evicted_entries_awaiting_gpu_completion() {
        let budget = RetentionBudget::default();
        let mut cache = PathCache::new(budget.clone());
        let bounds = PixelRect {
            width: 1600,
            height: 1000,
            ..Default::default()
        };
        let inflight = cache
            .capture(&[], bounds, || Ok::<_, ()>(1))
            .unwrap()
            .unwrap();
        cache.clear();
        cache.begin();
        assert_eq!(cache.bytes(), bounds.bytes());
        assert!(
            cache
                .capture(&[], bounds, || -> Result<i32, ()> {
                    panic!("window budget exceeded")
                })
                .unwrap()
                .is_none()
        );
        drop(inflight);
        assert_eq!(cache.bytes(), 0);
        assert!(
            cache
                .capture(&[], bounds, || Ok::<_, ()>(2))
                .unwrap()
                .is_some()
        );
        cache.clear();
        assert_eq!(budget.used(), 0);
    }

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
