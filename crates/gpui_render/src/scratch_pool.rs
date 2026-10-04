//! Reuse of scratch resources with an identical configuration, once the GPU is done with
//! them. Completion callbacks hold weak references only, so dropping the pool releases
//! everything it retains even while those callbacks are still pending.
use crate::gpu_policy::{RetentionBudget, RetentionLease};
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

/// Frames an idle resource stays pooled without being reused: about two seconds of
/// continuous rendering. Idle resources hold budget that path caches and retained
/// frames on the same device would otherwise use, so a size that is never requested
/// again (after a resize, say) must not keep it.
pub const IDLE_FRAMES: u64 = 120;

pub struct Pending<K, T> {
    key: K,
    value: Mutex<Option<T>>,
    ready: AtomicBool,
    bytes: u64,
    retired_frame: u64,
    _lease: RetentionLease,
}

pub struct Pool<K, T> {
    /// Oldest retirement first.
    entries: Mutex<Vec<Arc<Pending<K, T>>>>,
    budget: RetentionBudget,
    frame: AtomicU64,
}
impl<K: PartialEq, T> Pool<K, T> {
    pub fn new(budget: RetentionBudget) -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            budget,
            frame: AtomicU64::new(0),
        }
    }
    /// The budget charged for retained resources, shared with other device caches.
    pub fn budget(&self) -> &RetentionBudget {
        &self.budget
    }
    pub fn take(&self, key: &K) -> Option<T> {
        let mut entries = self.entries.lock().unwrap();
        let index = entries
            .iter()
            .position(|entry| entry.key == *key && entry.ready.load(Ordering::Acquire))?;
        let entry = entries.remove(index);
        let value = entry.value.lock().unwrap().take();
        value
    }
    /// Pools `value` until the GPU is done with it, when the caller [completes](Self::complete)
    /// the returned handle. Recently retired resources are the likeliest to be reused,
    /// so the oldest idle ones make room for it; in-flight ones never do.
    pub fn retire(&self, key: K, bytes: u64, value: T) -> Option<Weak<Pending<K, T>>> {
        if bytes > self.budget.limit() {
            return None;
        }
        let mut entries = self.entries.lock().unwrap();
        let lease = loop {
            if let Some(lease) = self.budget.try_acquire(bytes) {
                break lease;
            }
            let oldest_idle = entries
                .iter()
                .position(|entry| entry.ready.load(Ordering::Acquire))?;
            entries.remove(oldest_idle);
        };
        let entry = Arc::new(Pending {
            key,
            value: Mutex::new(Some(value)),
            bytes,
            ready: AtomicBool::new(false),
            retired_frame: self.frame.load(Ordering::Relaxed),
            _lease: lease,
        });
        let weak = Arc::downgrade(&entry);
        entries.push(entry);
        Some(weak)
    }
    /// Advances the pool's clock, releasing resources idle for [`IDLE_FRAMES`]. Every
    /// renderer sharing the pool calls this once per frame.
    pub fn end_frame(&self) {
        let frame = self.frame.fetch_add(1, Ordering::Relaxed) + 1;
        self.entries.lock().unwrap().retain(|entry| {
            !entry.ready.load(Ordering::Acquire) || frame - entry.retired_frame <= IDLE_FRAMES
        });
    }
    pub fn complete(pending: &Weak<Pending<K, T>>) {
        if let Some(entry) = pending.upgrade() {
            entry.ready.store(true, Ordering::Release);
        }
    }
    pub fn bytes(&self) -> (u64, u64) {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .fold((0, 0), |(idle, pending), e| {
                if e.ready.load(Ordering::Acquire) {
                    (idle + e.bytes, pending)
                } else {
                    (idle, pending + e.bytes)
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_matching_waits_for_completion_and_final_owner_releases_pending_resources() {
        let budget = RetentionBudget::new(100);
        let pool = Pool::new(budget.clone());
        let pending = pool.retire((100, 200, 4), 60, 7).unwrap();
        assert!(pool.take(&(100, 200, 4)).is_none());
        assert!(pool.retire((1, 2, 4), 41, 8).is_none());
        assert_eq!(pool.bytes(), (0, 60));
        Pool::complete(&pending);
        assert!(pool.take(&(100, 200, 1)).is_none());
        assert_eq!(pool.take(&(100, 200, 4)), Some(7));
        assert_eq!(budget.used(), 0);
        let pending = pool.retire((1, 2, 4), 100, 8).unwrap();
        drop(pool);
        assert_eq!(budget.used(), 0);
        assert!(pending.upgrade().is_none());
    }

    #[test]
    fn idle_resources_make_room_for_newer_ones_and_expire() {
        let budget = RetentionBudget::new(100);
        let pool = Pool::new(budget.clone());
        // A resize retires the previous size; nothing will ask for it again.
        Pool::complete(&pool.retire(1, 60, 'a').unwrap());
        let newer = pool.retire(2, 60, 'b').expect("the stale entry makes room");
        assert_eq!(pool.take(&1), None);
        assert!(
            pool.retire(3, 60, 'c').is_none(),
            "resources the GPU may still use are never evicted"
        );
        assert!(pool.retire(4, 101, 'd').is_none(), "over budget on its own");
        Pool::complete(&newer);
        assert_eq!(pool.bytes(), (60, 0));

        // Unused resources expire, returning their budget to the device's caches.
        for _ in 0..IDLE_FRAMES {
            pool.end_frame();
        }
        assert_eq!(pool.bytes(), (60, 0), "still within its grace period");
        pool.end_frame();
        assert_eq!(pool.bytes(), (0, 0));
        assert_eq!(budget.used(), 0);

        // In-flight resources outlive the grace period until they complete.
        let slow = pool.retire(5, 10, 'e').unwrap();
        for _ in 0..=IDLE_FRAMES {
            pool.end_frame();
        }
        assert_eq!(pool.bytes(), (0, 10));
        Pool::complete(&slow);
        pool.end_frame();
        assert_eq!(budget.used(), 0);
    }
}
