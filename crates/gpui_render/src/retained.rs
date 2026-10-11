//! Device memory budget for retained frame targets.
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

/// Retained targets on one device share this limit: two 4K BGRA8 targets, so a
/// second window or a resize keeps retaining.
pub const RETAINED_DEVICE_LIMIT: u64 = 64 * 1024 * 1024;
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
    fn two_4k_targets_fit_one_device() {
        let target = 3840 * 2160 * 4;
        let first = reserve_retained_bytes("test-4k", 1, target).unwrap();
        let second = reserve_retained_bytes("test-4k", 1, target);
        assert!(second.is_some());
        assert!(reserve_retained_bytes("test-4k", 1, target).is_none());
        drop((first, second));
    }
}
