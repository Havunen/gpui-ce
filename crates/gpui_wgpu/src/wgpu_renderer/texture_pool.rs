//! Optional scratch reuse. Textures become reusable only after the submitting queue finishes.
use gpui_render::gpu_policy::{RetentionBudget, RetentionLease};
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicU64, Ordering},
};

struct Idle {
    texture: wgpu::Texture,
    _lease: RetentionLease,
}
pub(super) struct TexturePool {
    idle: Mutex<Vec<Idle>>,
    retiring: Mutex<Vec<Arc<Mutex<Option<Idle>>>>>,
    budget: RetentionBudget,
    pending: AtomicU64,
}
thread_local! { static POOLS: std::cell::RefCell<Vec<(Weak<wgpu::Device>, Weak<TexturePool>)>> = const { std::cell::RefCell::new(Vec::new()) }; }
impl TexturePool {
    pub(super) fn for_device(device: &Arc<wgpu::Device>) -> Arc<Self> {
        POOLS.with_borrow_mut(|pools| {
            pools.retain(|(d, p)| d.strong_count() > 0 && p.strong_count() > 0);
            if let Some(pool) = pools
                .iter()
                .find(|(d, _)| d.as_ptr() == Arc::as_ptr(device))
                .and_then(|(_, p)| p.upgrade())
            {
                return pool;
            }
            let pool = Arc::new(Self {
                idle: Mutex::new(Vec::new()),
                retiring: Mutex::new(Vec::new()),
                budget: super::shared::retention_budget(device),
                pending: AtomicU64::new(0),
            });
            pools.push((Arc::downgrade(device), Arc::downgrade(&pool)));
            pool
        })
    }
    pub(super) fn take(&self, descriptor: &wgpu::TextureDescriptor<'_>) -> Option<wgpu::Texture> {
        let mut idle = self.idle.lock().unwrap();
        let i = idle.iter().position(|entry| {
            entry.texture.size() == descriptor.size
                && entry.texture.format() == descriptor.format
                && entry.texture.sample_count() == descriptor.sample_count
                && entry.texture.usage() == descriptor.usage
        })?;
        Some(idle.swap_remove(i).texture)
    }
    pub(super) fn retire(self: &Arc<Self>, queue: &wgpu::Queue, texture: wgpu::Texture) {
        let bytes = bytes(&texture);
        let Some(lease) = self.budget.try_acquire(bytes) else {
            return;
        };
        self.pending.fetch_add(bytes, Ordering::Relaxed);
        let entry = Arc::new(Mutex::new(Some(Idle {
            texture,
            _lease: lease,
        })));
        let weak_entry = Arc::downgrade(&entry);
        {
            let mut retiring = self.retiring.lock().unwrap();
            retiring.retain(|entry| entry.lock().unwrap().is_some());
            retiring.push(entry);
        }
        let pool = Arc::downgrade(self);
        // Weak references also release resources if the last window closes
        // before another poll. The GPU itself retains in-flight textures.
        queue.on_submitted_work_done(move || {
            if let (Some(pool), Some(entry)) = (pool.upgrade(), weak_entry.upgrade()) {
                if let Some(entry) = entry.lock().unwrap().take() {
                    pool.pending.fetch_sub(bytes, Ordering::Relaxed);
                    pool.idle.lock().unwrap().push(entry);
                }
            }
        });
    }

    pub(super) fn bytes(&self) -> (u64, u64) {
        (
            self.idle
                .lock()
                .unwrap()
                .iter()
                .map(|e| bytes(&e.texture))
                .sum(),
            self.pending.load(Ordering::Relaxed),
        )
    }
}
fn bytes(texture: &wgpu::Texture) -> u64 {
    u64::from(texture.width()) * u64::from(texture.height()) * 4 * u64::from(texture.sample_count())
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;
    #[test]
    fn scratch_reuse_waits_for_completion_and_releases_budget_on_checkout() {
        let context = crate::WgpuContext::new_headless(None).unwrap();
        let pool = TexturePool::for_device(&context.device);
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
        let texture = context.device.create_texture(&descriptor);
        let submission = context.queue.submit([]);
        pool.retire(&context.queue, texture);
        assert_eq!(pool.budget.used(), 1024);
        context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
        assert_eq!(pool.bytes(), (1024, 0));
        assert!(
            pool.take(&wgpu::TextureDescriptor {
                sample_count: 4,
                ..descriptor.clone()
            })
            .is_none()
        );
        assert!(pool.take(&descriptor).is_some());
        assert_eq!(pool.budget.used(), 0);
        let budget = pool.budget.clone();
        pool.retire(&context.queue, context.device.create_texture(&descriptor));
        assert_eq!(budget.used(), 1024);
        let weak = Arc::downgrade(&pool);
        drop(pool);
        assert!(weak.upgrade().is_none());
        assert_eq!(
            budget.used(),
            0,
            "last-window close must release unpolled retirements"
        );
    }
}
