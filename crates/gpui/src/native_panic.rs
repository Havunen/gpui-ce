use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;

type Observer = fn(&'static str, &(dyn Any + Send));
static OBSERVER: OnceLock<Observer> = OnceLock::new();

/// Registers a process-wide observer for panics escaping a guarded native callback.
/// The first registration wins. The callback's thread invokes the observer before aborting.
pub fn set_native_panic_observer(observer: fn(&'static str, &(dyn Any + Send))) {
    let _ = OBSERVER.set(observer);
}

#[doc(hidden)]
pub fn with_native_callback<R>(context: &'static str, callback: impl FnOnce() -> R) -> R {
    match catch_unwind(AssertUnwindSafe(callback)) {
        Ok(result) => result,
        Err(payload) => {
            if let Some(observer) = OBSERVER.get() {
                // An observer's panic payload can itself panic on drop.
                std::mem::forget(catch_unwind(AssertUnwindSafe(|| {
                    observer(context, &*payload)
                })));
            }
            std::process::abort()
        }
    }
}
