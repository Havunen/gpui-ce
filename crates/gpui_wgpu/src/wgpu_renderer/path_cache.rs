use gpui::{Path, ScaledPixels};
use gpui_render::{
    gpu_policy::{RetentionBudget, RetentionLease, WINDOW_LAYER_BYTES},
    path_plan::{self, PixelRect},
    path_types::{self, PathRasterizationVertex},
};
use std::sync::Arc;

pub(super) struct CachedPath {
    _texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
    pub(super) bounds: PixelRect,
    _lease: RetentionLease,
    key: u64,
    vertices: Vec<PathRasterizationVertex>,
}
pub(super) struct PathCache {
    entries: Vec<Arc<CachedPath>>,
    warmed: usize,
    budget: RetentionBudget,
    pub(super) hits: u64,
    pub(super) misses: u64,
}
impl PathCache {
    pub(super) fn new(budget: RetentionBudget) -> Self {
        Self {
            entries: Vec::new(),
            warmed: 0,
            budget,
            hits: 0,
            misses: 0,
        }
    }
    pub(super) fn begin(&mut self) {
        self.warmed = 0;
        self.hits = 0;
        self.misses = 0;
    }
    pub(super) fn bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bounds.bytes()).sum()
    }
    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }
    pub(super) fn leases(&self) -> Vec<Arc<CachedPath>> {
        self.entries.clone()
    }
    pub(super) fn views(&self) -> Vec<wgpu::TextureView> {
        self.entries.iter().map(|e| e.view.clone()).collect()
    }
    pub(super) fn get(&mut self, paths: &[Path<ScaledPixels>]) -> Option<Arc<CachedPath>> {
        let key = path_plan::fingerprint(paths);
        let found = self.entries.iter().position(|e| {
            e.key == key
                && e.vertices.len() == path_types::rasterization_vertex_count(paths)
                && e.vertices
                    .iter()
                    .cloned()
                    .eq(path_types::rasterization_vertices(paths))
        });
        if let Some(index) = found {
            self.hits += 1;
            let entry = self.entries.remove(index);
            self.entries.push(entry.clone());
            Some(entry)
        } else {
            self.misses += 1;
            None
        }
    }
    pub(super) fn capture(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        source_bounds: PixelRect,
        world_bounds: PixelRect,
        paths: &[Path<ScaledPixels>],
    ) {
        if self.warmed >= 2
            || world_bounds.bytes() > WINDOW_LAYER_BYTES
            || world_bounds.width == 0
            || world_bounds.height == 0
        {
            return;
        }
        // A tiny raster may still contain arbitrarily complex tessellation.
        // Bound copied geometry independently of the GPU texture budget.
        const GEOMETRY_BYTES: usize = 1024 * 1024;
        let geometry_bytes = path_types::rasterization_vertex_count(paths)
            .saturating_mul(std::mem::size_of::<PathRasterizationVertex>());
        if geometry_bytes > GEOMETRY_BYTES {
            return;
        }
        while !self.entries.is_empty()
            && (self.bytes().saturating_add(world_bounds.bytes()) > WINDOW_LAYER_BYTES
                || self
                    .entries
                    .iter()
                    .map(|entry| {
                        entry.vertices.len() * std::mem::size_of::<PathRasterizationVertex>()
                    })
                    .sum::<usize>()
                    .saturating_add(geometry_bytes)
                    > GEOMETRY_BYTES)
        {
            self.entries.remove(0);
        }
        let Some(lease) = self.budget.try_acquire(world_bounds.bytes()) else {
            return;
        };
        self.warmed += 1;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gpui_cached_path_batch"),
            size: wgpu::Extent3d {
                width: world_bounds.width,
                height: world_bounds.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: source.format(),
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                origin: wgpu::Origin3d {
                    x: source_bounds.x,
                    y: source_bounds.y,
                    z: 0,
                },
                ..source.as_image_copy()
            },
            texture.as_image_copy(),
            texture.size(),
        );
        let view = texture.create_view(&Default::default());
        self.entries.push(Arc::new(CachedPath {
            _texture: texture,
            view,
            bounds: world_bounds,
            _lease: lease,
            key: path_plan::fingerprint(paths),
            vertices: path_types::rasterization_vertices(paths).collect(),
        }));
    }
}
