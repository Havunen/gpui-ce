//! Frame context and device budget shared by the retained renderers.
use std::{
    cell::Cell,
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

/// Window state passed to the platform renderer on this thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuFrameContext {
    /// Whether the submitting window is active.
    pub active: bool,
    /// Current painted scene revision.
    pub scene_revision: u64,
}
thread_local! {
    static CONTEXT: Cell<GpuFrameContext> = const {
        Cell::new(GpuFrameContext { active: false, scene_revision: 0 })
    };
}

/// Restores the previous context, including when drawing unwinds.
pub struct GpuFrameContextGuard(GpuFrameContext);
impl Drop for GpuFrameContextGuard {
    fn drop(&mut self) {
        CONTEXT.with(|context| context.set(self.0));
    }
}
/// Install a scoped renderer context, restoring the previous value when the guard drops.
pub fn set_gpu_frame_context(active: bool, scene_revision: u64) -> GpuFrameContextGuard {
    CONTEXT.with(|context| {
        GpuFrameContextGuard(context.replace(GpuFrameContext {
            active,
            scene_revision,
        }))
    })
}
/// Return the current thread's renderer context.
pub fn gpu_frame_context() -> GpuFrameContext {
    CONTEXT.with(Cell::get)
}
/// Whether renderers need the frame context; only retained rendering reads it.
pub fn frame_context_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| experiment_enabled("partial-redraw"))
}
/// Whether an experiment flag was explicitly requested by the environment.
pub fn experiment_enabled(name: &str) -> bool {
    std::env::var("GPUI_GPU_EXPERIMENTS")
        .is_ok_and(|value| value.split(',').any(|flag| flag.trim() == name))
}

/// Optional retained rendering allocations share this limit on each device.
pub const RETAINED_DEVICE_LIMIT: u64 = 32 * 1024 * 1024;
struct DeviceBudget(AtomicU64);
/// A retained-allocation budget reservation, released when dropped.
pub struct GpuMemoryReservation {
    budget: Arc<DeviceBudget>,
    /// Number of bytes reserved against the device budget.
    pub bytes: u64,
}
impl Drop for GpuMemoryReservation {
    fn drop(&mut self) {
        self.budget.0.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
/// Reserve additional retained bytes on one device, failing when its shared budget is exceeded.
pub fn reserve_retained_bytes(
    backend: &str,
    device_key: usize,
    bytes: u64,
) -> Option<GpuMemoryReservation> {
    type Registry = HashMap<(String, usize), Weak<DeviceBudget>>;
    static BUDGETS: OnceLock<Mutex<Registry>> = OnceLock::new();
    let mut budgets = BUDGETS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    budgets.retain(|_, budget| budget.strong_count() != 0);
    let key = (backend.to_owned(), device_key);
    let budget = budgets
        .get(&key)
        .and_then(Weak::upgrade)
        .unwrap_or_else(|| {
            let budget = Arc::new(DeviceBudget(AtomicU64::new(0)));
            budgets.insert(key, Arc::downgrade(&budget));
            budget
        });
    budget
        .0
        .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
            used.checked_add(bytes)
                .filter(|total| *total <= RETAINED_DEVICE_LIMIT)
        })
        .ok()?;
    Some(GpuMemoryReservation { budget, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_budget_is_device_wide_and_released() {
        let first = reserve_retained_bytes("test", 1, RETAINED_DEVICE_LIMIT - 1).unwrap();
        assert!(reserve_retained_bytes("test", 1, 2).is_none());
        assert!(reserve_retained_bytes("test", 2, 2).is_some());
        drop(first);
        assert!(reserve_retained_bytes("test", 1, RETAINED_DEVICE_LIMIT).is_some());
    }
    #[test]
    fn scoped_context_restores_even_nested_calls() {
        let _outer = set_gpu_frame_context(true, 4);
        {
            let _inner = set_gpu_frame_context(false, 9);
            assert_eq!(gpu_frame_context().scene_revision, 9);
        }
        assert_eq!(gpu_frame_context().scene_revision, 4);
        assert!(gpu_frame_context().active);
    }
}
