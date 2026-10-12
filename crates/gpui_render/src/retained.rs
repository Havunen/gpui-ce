//! Retained frame state shared by the renderers, and its device memory budget.
use crate::damage::{Damage, SceneSnapshot};
use gpui::Scene;
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

/// The last frame a retained target holds, which decides how much of the next
/// frame to repaint. The renderer owns the target itself.
#[derive(Default)]
pub struct RetainedScene {
    snapshot: Option<SceneSnapshot>,
    revision: Option<u64>,
    /// Tiles of a replaced snapshot, reused by the next capture.
    spare: Vec<u64>,
}

impl RetainedScene {
    /// Forget the retained frame, so the next one repaints everything.
    pub fn forget(&mut self) {
        if let Some(snapshot) = self.snapshot.take() {
            self.spare = snapshot.tiles;
        }
        self.revision = None;
    }

    /// The damage for drawing `scene` over the retained frame, and the snapshot
    /// to keep once it succeeds: none when the frame repeats the retained scene.
    /// `None` when the scene can't be retained.
    pub fn plan(
        &mut self,
        scene: &Scene,
        revision: u64,
        atlas_generation: u64,
        width: u32,
        height: u32,
    ) -> Option<(Damage, Option<SceneSnapshot>)> {
        if self.revision == Some(revision)
            && (self.snapshot.as_ref())
                .is_some_and(|kept| kept.atlas_generation == atlas_generation)
        {
            return Some((Damage::Reuse, None));
        }
        let tiles = std::mem::take(&mut self.spare);
        let snapshot = SceneSnapshot::capture(scene, atlas_generation, width, height, tiles)?;
        let damage = match &self.snapshot {
            Some(previous) => snapshot.compare(previous),
            None => Damage::Full,
        };
        Some((damage, Some(snapshot)))
    }

    /// Record a frame drawn from [`Self::plan`] as the retained one.
    pub fn complete(&mut self, snapshot: Option<SceneSnapshot>, revision: u64) {
        if let Some(replaced) = snapshot.and_then(|snapshot| self.snapshot.replace(snapshot)) {
            self.spare = replaced.tiles;
        }
        self.revision = Some(revision);
    }
}

/// Retained targets on one device share this limit: two 4K BGRA8 targets, so a
/// second window or a resize keeps retaining.
pub const RETAINED_DEVICE_LIMIT: u64 = 64 * 1024 * 1024;
struct DeviceBudget(AtomicU64);
/// A retained-allocation budget reservation, released when dropped.
pub struct GpuMemoryReservation {
    budget: Arc<DeviceBudget>,
    /// Number of bytes reserved against the device budget.
    bytes: u64,
}
impl Drop for GpuMemoryReservation {
    fn drop(&mut self) {
        self.budget.0.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
/// Reserve additional retained bytes on one device, failing when its shared budget is exceeded.
pub fn reserve_retained_bytes(
    backend: &'static str,
    device_key: usize,
    bytes: u64,
) -> Option<GpuMemoryReservation> {
    type Registry = HashMap<(&'static str, usize), Weak<DeviceBudget>>;
    static BUDGETS: OnceLock<Mutex<Registry>> = OnceLock::new();
    let mut budgets = BUDGETS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    budgets.retain(|_, budget| budget.strong_count() != 0);
    let key = (backend, device_key);
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
    use gpui::{Bounds, ContentMask, Hsla, Quad, ScaledPixels, point, size};

    fn scene(color: Hsla) -> Scene {
        let bounds = |x, y, width, height| Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(width), ScaledPixels(height)),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            bounds: bounds(0.0, 10.0, 200.0, 20.0),
            content_mask: ContentMask {
                bounds: bounds(0.0, 0.0, 200.0, 200.0),
                ..Default::default()
            },
            background: color.into(),
            ..Default::default()
        });
        scene.finish();
        scene
    }

    #[test]
    fn repeated_revisions_reuse_and_changed_scenes_repaint_their_damage() {
        let mut retained = RetainedScene::default();
        let red = scene(gpui::red());
        let (damage, snapshot) = retained.plan(&red, 1, 1, 200, 200).unwrap();
        assert_eq!(damage, Damage::Full);
        retained.complete(snapshot, 1);
        // The same revision reuses the target without capturing the scene again.
        assert!(matches!(
            retained.plan(&red, 1, 1, 200, 200),
            Some((Damage::Reuse, None))
        ));
        // A new revision is compared, so only the changed quad repaints.
        let (damage, snapshot) = retained.plan(&scene(gpui::blue()), 2, 1, 200, 200).unwrap();
        assert!(matches!(damage, Damage::Partial(_)));
        assert!(snapshot.is_some());
        // An atlas change, a scene that can't be captured, or forgetting all repaint in full.
        assert_eq!(retained.plan(&red, 1, 2, 200, 200).unwrap().0, Damage::Full);
        assert!(retained.plan(&red, 3, 1, 0, 200).is_none());
        retained.forget();
        assert_eq!(retained.plan(&red, 1, 1, 200, 200).unwrap().0, Damage::Full);
    }

    #[test]
    fn a_reused_frame_keeps_the_retained_snapshot() {
        let mut retained = RetainedScene::default();
        let red = scene(gpui::red());
        let (_, snapshot) = retained.plan(&red, 1, 1, 200, 200).unwrap();
        retained.complete(snapshot, 1);
        retained.complete(None, 1);
        // An identical scene under a new revision still matches the kept snapshot.
        assert!(matches!(
            retained.plan(&red, 2, 1, 200, 200),
            Some((Damage::Reuse, Some(_)))
        ));
    }

    #[test]
    fn a_replaced_snapshot_lends_its_tiles_to_the_next_capture() {
        let mut retained = RetainedScene::default();
        let (_, first) = retained.plan(&scene(gpui::red()), 1, 1, 200, 200).unwrap();
        let first_tiles = first.as_ref().unwrap().tiles.as_ptr();
        retained.complete(first, 1);
        let (_, second) = retained.plan(&scene(gpui::blue()), 2, 1, 200, 200).unwrap();
        retained.complete(second, 2);
        // The second frame replaced the first, whose buffer holds the third capture.
        let (_, third) = retained.plan(&scene(gpui::red()), 3, 1, 200, 200).unwrap();
        assert_eq!(third.unwrap().tiles.as_ptr(), first_tiles);
    }

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
