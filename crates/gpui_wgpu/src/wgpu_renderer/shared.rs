//! Device-scoped weak caches. Dropping the final renderer releases the cache.
use super::pipelines::{WgpuBindGroupLayouts, WgpuPipelines};
use crate::RendererTier;
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
thread_local! { static CACHE: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) }; }
thread_local! {
    static BUDGETS: RefCell<Vec<(Weak<wgpu::Device>, gpui_render::gpu_policy::RetentionBudget)>> = const { RefCell::new(Vec::new()) };
}
pub(super) fn retention_budget(
    device: &Arc<wgpu::Device>,
) -> gpui_render::gpu_policy::RetentionBudget {
    BUDGETS.with_borrow_mut(|budgets| {
        budgets.retain(|(device, _)| device.strong_count() > 0);
        if let Some((_, budget)) = budgets
            .iter()
            .find(|(d, _)| d.as_ptr() == Arc::as_ptr(device))
        {
            return budget.clone();
        }
        let budget = gpui_render::gpu_policy::RetentionBudget::default();
        budgets.push((Arc::downgrade(device), budget.clone()));
        budget
    })
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
