//! Device-scoped weak caches. Dropping the final renderer releases the cache.
use super::pipelines::{WgpuBindGroupLayouts, WgpuPipelines};
use crate::RendererTier;
use gpui_render::{gpu_policy::RetentionBudget, scratch_pool::Pool, sharing::Registry};
use std::{
    cell::RefCell,
    sync::{Arc, Weak},
};

type Key = (
    wgpu::TextureFormat,
    wgpu::CompositeAlphaMode,
    u32,
    bool,
    RendererTier,
);
struct Entry {
    device: Weak<wgpu::Device>,
    key: Key,
    layouts: Weak<WgpuBindGroupLayouts>,
    pipelines: Weak<WgpuPipelines>,
}
thread_local! {
    static CACHE: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
    static RETENTION: Registry<DeviceRetention> = const { Registry::new() };
}

type TextureKey = (
    wgpu::Extent3d,
    wgpu::TextureFormat,
    u32,
    wgpu::TextureUsages,
);
pub(super) type ScratchPool = Pool<TextureKey, wgpu::Texture>;

/// Optional resources retained beyond a frame (path caches, retained frames, pooled
/// scratch textures), charged to one budget per device rather than per window.
pub(super) struct DeviceRetention {
    device: Arc<wgpu::Device>,
    pub(super) pool: ScratchPool,
}

impl DeviceRetention {
    pub(super) fn for_device(device: &Arc<wgpu::Device>) -> Arc<Self> {
        RETENTION.with(|retention| {
            retention.get_or_insert_with(
                |retention| Arc::ptr_eq(&retention.device, device),
                || Self {
                    device: device.clone(),
                    pool: Pool::new(RetentionBudget::default()),
                },
            )
        })
    }

    pub(super) fn budget(&self) -> &RetentionBudget {
        self.pool.budget()
    }

    /// Offers `texture` for reuse once the queue finishes the work submitted so far.
    pub(super) fn retire(&self, queue: &wgpu::Queue, texture: wgpu::Texture) {
        let key = (
            texture.size(),
            texture.format(),
            texture.sample_count(),
            texture.usage(),
        );
        let bytes = u64::from(texture.width())
            * u64::from(texture.height())
            * 4
            * u64::from(texture.sample_count());
        if let Some(pending) = self.pool.retire(key, bytes, texture) {
            queue.on_submitted_work_done(move || ScratchPool::complete(&pending));
        }
    }
}

/// A texture for `descriptor`: an idle one from `pool` when it has one, otherwise new.
pub(super) fn scratch_texture(
    device: &wgpu::Device,
    pool: Option<&ScratchPool>,
    descriptor: &wgpu::TextureDescriptor<'_>,
) -> wgpu::Texture {
    let key = (
        descriptor.size,
        descriptor.format,
        descriptor.sample_count,
        descriptor.usage,
    );
    pool.and_then(|pool| pool.take(&key))
        .unwrap_or_else(|| device.create_texture(descriptor))
}

pub(super) fn pipelines(
    device: &Arc<wgpu::Device>,
    format: wgpu::TextureFormat,
    alpha: wgpu::CompositeAlphaMode,
    samples: u32,
    dual_source: bool,
    tier: RendererTier,
    shared: bool,
) -> (Arc<WgpuBindGroupLayouts>, Arc<WgpuPipelines>) {
    let key = (format, alpha, samples, dual_source, tier);
    CACHE.with_borrow_mut(|cache| {
        cache.retain(|e| e.device.strong_count() > 0 && e.layouts.strong_count() > 0);
        let same_device = |entry: &&Entry| entry.device.as_ptr() == Arc::as_ptr(device);
        if shared {
            if let Some((layouts, pipelines)) = cache
                .iter()
                .filter(same_device)
                .find(|e| e.key == key)
                .and_then(|e| Some((e.layouts.upgrade()?, e.pipelines.upgrade()?)))
            {
                return (layouts, pipelines);
            }
        }
        let layouts = shared
            .then(|| {
                cache
                    .iter()
                    .filter(same_device)
                    .find(|e| e.key.4 == tier)
                    .and_then(|e| e.layouts.upgrade())
            })
            .flatten()
            .unwrap_or_else(|| Arc::new(WgpuBindGroupLayouts::new(device, tier)));
        let pipelines = Arc::new(WgpuPipelines::new(
            device,
            &layouts,
            format,
            alpha,
            samples,
            dual_source,
            tier,
        ));
        if shared {
            cache.retain(|e| !(e.device.as_ptr() == Arc::as_ptr(device) && e.key == key));
            cache.push(Entry {
                device: Arc::downgrade(device),
                key,
                layouts: Arc::downgrade(&layouts),
                pipelines: Arc::downgrade(&pipelines),
            });
        }
        (layouts, pipelines)
    })
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;

    #[test]
    fn scratch_reuse_waits_for_completion_and_releases_budget_on_checkout() {
        let context = crate::WgpuContext::new_headless(None).unwrap();
        let retention = DeviceRetention::for_device(&context.device);
        let descriptor = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };
        let submission = context.queue.submit([]);
        retention.retire(&context.queue, context.device.create_texture(&descriptor));
        assert_eq!(retention.budget().used(), 1024);
        assert_eq!(
            retention.pool.bytes(),
            (0, 1024),
            "in flight until the queue finishes"
        );
        context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
        assert_eq!(retention.pool.bytes(), (1024, 0));

        let multisampled = wgpu::TextureDescriptor {
            sample_count: 4,
            ..descriptor.clone()
        };
        scratch_texture(&context.device, Some(&retention.pool), &multisampled);
        assert_eq!(
            retention.pool.bytes(),
            (1024, 0),
            "only an identical texture is reused"
        );
        let reused = scratch_texture(&context.device, Some(&retention.pool), &descriptor);
        assert_eq!(retention.pool.bytes(), (0, 0));
        assert_eq!(retention.budget().used(), 0, "checkout releases the budget");

        let budget = retention.budget().clone();
        retention.retire(&context.queue, reused);
        assert_eq!(budget.used(), 1024);
        let weak = Arc::downgrade(&retention);
        drop(retention);
        assert!(weak.upgrade().is_none());
        assert_eq!(
            budget.used(),
            0,
            "the last window closing releases unfinished retirements"
        );
    }
    #[test]
    fn device_pipelines_share_variants_without_retaining_the_last_window() {
        let context = crate::WgpuContext::new_headless(None).unwrap();
        let make = |alpha| {
            pipelines(
                &context.device,
                wgpu::TextureFormat::Bgra8Unorm,
                alpha,
                4,
                false,
                context.renderer_tier(),
                true,
            )
        };
        let a = make(wgpu::CompositeAlphaMode::Opaque);
        let b = make(wgpu::CompositeAlphaMode::Opaque);
        assert!(Arc::ptr_eq(&a.1, &b.1));
        let transparent = make(wgpu::CompositeAlphaMode::PreMultiplied);
        assert!(Arc::ptr_eq(&a.0, &transparent.0));
        assert!(!Arc::ptr_eq(&a.1, &transparent.1));
        let weak = Arc::downgrade(&a.1);
        drop(a);
        assert!(weak.upgrade().is_some());
        drop(b);
        assert!(weak.upgrade().is_none());
        drop(transparent);
    }
}
