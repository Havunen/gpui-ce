//! Opt-in renderer experiments and backend-independent allocation policy.
use gpui::{Bounds, ScaledPixels};
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicU64, Ordering},
};

/// Experiments are disabled until native backend validation is complete.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuOptions {
    pub cropped_paths: bool,
    pub shared_resources: bool,
    pub pooled_targets: bool,
    pub cached_layers: bool,
    pub batched_paths: bool,
    pub partial_redraw: bool,
}

impl GpuOptions {
    pub fn from_env() -> Self {
        static OPTIONS: OnceLock<GpuOptions> = OnceLock::new();
        *OPTIONS
            .get_or_init(|| Self::parse(&std::env::var("GPUI_GPU_EXPERIMENTS").unwrap_or_default()))
    }

    pub fn parse(value: &str) -> Self {
        let mut options = Self::default();
        for flag in value.split(',').map(str::trim) {
            match flag {
                "cropped-paths" => options.cropped_paths = true,
                "shared-resources" => options.shared_resources = true,
                "pooled-targets" => options.pooled_targets = true,
                "cached-layers" => options.cached_layers = true,
                "batched-paths" => options.batched_paths = true,
                "partial-redraw" => options.partial_redraw = true,
                _ => {}
            }
        }
        options
    }
}

/// Keep window-space coordinates; omit unused right and bottom pixels only.
pub fn path_target_extent(
    bounds: impl Iterator<Item = Bounds<ScaledPixels>>,
    viewport: (u32, u32),
    previous: (u32, u32),
    cropped: bool,
) -> (u32, u32) {
    if !cropped {
        return (viewport.0.max(1), viewport.1.max(1));
    }
    let (right, bottom) = bounds.fold((0.0_f32, 0.0_f32), |(right, bottom), b| {
        if b.right().0 <= 0.0
            || b.bottom().0 <= 0.0
            || b.left().0 >= viewport.0 as f32
            || b.top().0 >= viewport.1 as f32
        {
            return (right, bottom);
        }
        (right.max(b.right().0), bottom.max(b.bottom().0))
    });
    let grow = |required: f32, old: u32, limit: u32| {
        let needed = (required.ceil() as u32).max(1).min(limit.max(1));
        if old >= needed {
            return old.min(limit.max(1));
        }
        needed.max(old.saturating_add(old / 2)).saturating_add(63) / 64 * 64
    };
    let width = grow(right, previous.0, viewport.0).min(viewport.0.max(1));
    // Once the graph occupies the viewport, reserving its height avoids a second
    // large allocation when the bottom rows become visible after loading.
    let height = if bottom >= viewport.1 as f32 / 2.0 {
        viewport.1.max(1)
    } else {
        grow(bottom, previous.1, viewport.1).min(viewport.1.max(1))
    };
    (width, height)
}

pub const DEVICE_RETENTION_BYTES: u64 = 32 * 1024 * 1024;
pub const WINDOW_LAYER_BYTES: u64 = 8 * 1024 * 1024;

/// A device-local budget for optional retained resources, not required live uploads.
#[derive(Clone, Debug)]
pub struct RetentionBudget(Arc<BudgetState>);
#[derive(Debug)]
struct BudgetState {
    used: AtomicU64,
    limit: u64,
}
#[derive(Debug)]
pub struct RetentionLease {
    budget: RetentionBudget,
    bytes: u64,
}

impl Default for RetentionBudget {
    fn default() -> Self {
        Self::new(DEVICE_RETENTION_BYTES)
    }
}
impl RetentionBudget {
    pub fn new(limit: u64) -> Self {
        Self(Arc::new(BudgetState {
            used: AtomicU64::new(0),
            limit,
        }))
    }
    pub fn used(&self) -> u64 {
        self.0.used.load(Ordering::Relaxed)
    }
    pub fn limit(&self) -> u64 {
        self.0.limit
    }
    pub fn try_acquire(&self, bytes: u64) -> Option<RetentionLease> {
        self.0
            .used
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(bytes).filter(|next| *next <= self.0.limit)
            })
            .ok()?;
        Some(RetentionLease {
            budget: self.clone(),
            bytes,
        })
    }
}
impl Drop for RetentionLease {
    fn drop(&mut self) {
        self.budget.0.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{point, size};
    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(w), ScaledPixels(h)),
        )
    }
    #[test]
    fn experiments_require_explicit_names() {
        assert_eq!(GpuOptions::parse(""), GpuOptions::default());
        assert_eq!(GpuOptions::parse("all,typo"), GpuOptions::default());
        let options = GpuOptions::parse("cropped-paths, cached-layers");
        assert!(options.cropped_paths && options.cached_layers && !options.partial_redraw);
    }
    #[test]
    fn cropped_growth_is_clipped_and_reserves_graph_height() {
        let viewport = (2560, 1440);
        let extent = |b, previous| path_target_extent([b].into_iter(), viewport, previous, true);
        assert_eq!(extent(bounds(10., 10., 100., 20.), (0, 0)), (128, 64));
        assert_eq!(
            extent(bounds(0., 100., 490., 1052.), (128, 64)),
            (512, 1440)
        );
        assert_eq!(
            extent(bounds(0., 100., 490., 1300.), (512, 1440)),
            (512, 1440)
        );
        assert_eq!(extent(bounds(0., 0., 50., 20.), (512, 1440)), (512, 1440));
        assert_eq!(extent(bounds(-200., -200., 10., 10.), (0, 0)), (64, 64));
        assert_eq!(extent(bounds(3000., 100., 10., 10.), (0, 0)), (64, 64));
        assert_eq!(extent(bounds(2500., 1400., 1000., 1000.), (0, 0)), viewport);
        assert_eq!(
            path_target_extent(
                [bounds(0., 0., 100., 100.)].into_iter(),
                (80, 30),
                (0, 0),
                true
            ),
            (80, 30)
        );
    }
    #[test]
    fn budgets_are_shared_checked_and_released() {
        let a = RetentionBudget::new(100);
        let b = a.clone();
        let lease = a.try_acquire(60).unwrap();
        assert!(b.try_acquire(41).is_none());
        assert!(b.try_acquire(u64::MAX).is_none());
        assert_eq!(b.used(), 60);
        drop(lease);
        let _lease = b.try_acquire(100).unwrap();
        assert!(a.try_acquire(1).is_none());
    }
}
