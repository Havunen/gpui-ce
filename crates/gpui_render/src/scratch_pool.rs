//! Reuse of scratch resources with an identical configuration, once the GPU is done with
//! them. Completion callbacks hold weak references only, so dropping the pool releases
//! everything it retains even while those callbacks are still pending.
use crate::gpu_policy::{RetentionBudget, RetentionLease};
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, Ordering},
};

pub struct Pending<K, T> {
    key: K,
    value: Mutex<Option<T>>,
    ready: AtomicBool,
    bytes: u64,
    _lease: RetentionLease,
}

pub struct Pool<K, T> {
    entries: Mutex<Vec<Arc<Pending<K, T>>>>,
    budget: RetentionBudget,
}
impl<K: PartialEq, T> Pool<K, T> {
    pub fn new(budget: RetentionBudget) -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            budget,
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
    pub fn retire(&self, key: K, bytes: u64, value: T) -> Option<Weak<Pending<K, T>>> {
        let lease = self.budget.try_acquire(bytes)?;
        let entry = Arc::new(Pending {
            key,
            value: Mutex::new(Some(value)),
            bytes,
            ready: AtomicBool::new(false),
            _lease: lease,
        });
        let weak = Arc::downgrade(&entry);
        self.entries.lock().unwrap().push(entry);
        Some(weak)
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
}
