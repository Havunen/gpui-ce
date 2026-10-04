//! Resources that several windows' renderers share, typically per device.
use collections::{FxHashMap, FxHashSet};
use gpui::AtlasKey;
use std::{
    cell::RefCell,
    sync::{Arc, Weak},
};

/// Values shared by the renderers on one thread, held weakly: the last renderer to
/// drop a value releases it, along with whatever it retains, such as its device.
pub struct Registry<T>(RefCell<Vec<Weak<T>>>);

impl<T> Registry<T> {
    pub const fn new() -> Self {
        Self(RefCell::new(Vec::new()))
    }

    /// The live value that `matches`, or a new one from `create`.
    pub fn get_or_insert_with(
        &self,
        matches: impl Fn(&T) -> bool,
        create: impl FnOnce() -> T,
    ) -> Arc<T> {
        if let Some(value) = self
            .0
            .borrow()
            .iter()
            .filter_map(Weak::upgrade)
            .find(|value| matches(value))
        {
            return value;
        }
        let value = Arc::new(create());
        let mut values = self.0.borrow_mut();
        values.retain(|value| value.strong_count() > 0);
        values.push(Arc::downgrade(&value));
        value
    }
}

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Which windows use each tile of an atlas, so that one window removing a tile frees
/// it only once no other window sharing the atlas uses it. An unshared atlas has a
/// single window and frees tiles as soon as it removes them.
pub struct TileOwners {
    shared: bool,
    next_owner: u64,
    /// The tiles each window uses.
    owners: FxHashMap<OwnerId, FxHashSet<AtlasKey>>,
    /// How many windows use each tile.
    counts: FxHashMap<AtlasKey, usize>,
}

/// A window using an atlas, as [`TileOwners::add_owner`] registered it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OwnerId(u64);

impl TileOwners {
    pub fn new(shared: bool) -> Self {
        Self {
            shared,
            next_owner: 0,
            owners: FxHashMap::default(),
            counts: FxHashMap::default(),
        }
    }

    pub fn add_owner(&mut self) -> OwnerId {
        let owner = OwnerId(self.next_owner);
        self.next_owner += 1;
        if self.shared {
            self.owners.insert(owner, FxHashSet::default());
        }
        owner
    }

    /// Records that `owner` uses the tile for `key`.
    pub fn retain(&mut self, owner: OwnerId, key: &AtlasKey) {
        if !self.shared {
            return;
        }
        if let Some(keys) = self.owners.get_mut(&owner)
            && keys.insert(key.clone())
        {
            *self.counts.entry(key.clone()).or_default() += 1;
        }
    }

    /// Releases `owner`'s use of `key`. True when no window uses the tile any more, so
    /// the atlas should free it. Repeated releases, and releases of tiles the window
    /// never used, are ignored.
    pub fn release(&mut self, owner: OwnerId, key: &AtlasKey) -> bool {
        if !self.shared {
            return true;
        }
        let released = self
            .owners
            .get_mut(&owner)
            .is_some_and(|keys| keys.remove(key));
        released && release_one(&mut self.counts, key)
    }

    /// Unregisters `owner`, returning the tiles no window uses any more.
    pub fn remove_owner(&mut self, owner: OwnerId) -> Vec<AtlasKey> {
        let Some(keys) = self.owners.remove(&owner) else {
            return Vec::new();
        };
        keys.into_iter()
            .filter(|key| release_one(&mut self.counts, key))
            .collect()
    }

    /// Forgets every use after the atlas dropped all of its tiles (cleared, or device
    /// lost). Windows stay registered and own the tiles they use from now on.
    pub fn reset(&mut self) {
        self.counts.clear();
        for keys in self.owners.values_mut() {
            keys.clear();
        }
    }
}

fn release_one(counts: &mut FxHashMap<AtlasKey, usize>, key: &AtlasKey) -> bool {
    let Some(count) = counts.get_mut(key) else {
        return false;
    };
    *count -= 1;
    if *count > 0 {
        return false;
    }
    counts.remove(key);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(id: usize) -> AtlasKey {
        AtlasKey::Image(gpui::RenderImageParams {
            image_id: gpui::ImageId(id),
            frame_index: 0,
        })
    }

    #[test]
    fn registry_shares_live_values_and_forgets_released_ones() {
        let registry = Registry::new();
        let first = registry.get_or_insert_with(|value: &(u32, u32)| value.0 == 1, || (1, 10));
        let shared = registry.get_or_insert_with(|value| value.0 == 1, || panic!("not shared"));
        assert!(Arc::ptr_eq(&first, &shared));
        let other = registry.get_or_insert_with(|value| value.0 == 2, || (2, 20));
        assert!(!Arc::ptr_eq(&first, &other));

        let weak = Arc::downgrade(&first);
        drop((first, shared));
        assert!(weak.upgrade().is_none(), "the registry must not own values");
        let recreated = registry.get_or_insert_with(|value| value.0 == 1, || (1, 30));
        assert_eq!(*recreated, (1, 30));
        assert_eq!(registry.0.borrow().len(), 2, "released entries are pruned");
    }

    #[test]
    fn shared_tiles_are_freed_by_their_last_window() {
        let mut owners = TileOwners::new(true);
        let first = owners.add_owner();
        let second = owners.add_owner();
        owners.retain(first, &key(1));
        owners.retain(first, &key(1));
        owners.retain(second, &key(1));
        owners.retain(second, &key(2));

        assert!(!owners.release(first, &key(1)), "the second window uses it");
        assert!(
            !owners.release(first, &key(1)),
            "a repeated release is ignored"
        );
        assert!(
            !owners.release(first, &key(2)),
            "the first window never used it"
        );
        assert!(owners.release(second, &key(1)));

        owners.retain(first, &key(2));
        assert!(
            owners.remove_owner(second).is_empty(),
            "the first window uses tile 2"
        );
        assert!(
            owners.remove_owner(first) == [key(2)],
            "the last window frees it"
        );
        assert!(owners.counts.is_empty() && owners.owners.is_empty());
        assert!(owners.remove_owner(first).is_empty());
    }

    #[test]
    fn reset_tiles_are_owned_anew() {
        let mut owners = TileOwners::new(true);
        let first = owners.add_owner();
        let second = owners.add_owner();
        owners.retain(first, &key(1));
        owners.retain(second, &key(1));

        owners.reset();
        assert!(
            !owners.release(first, &key(1)),
            "the tile was dropped by the reset"
        );
        // The tile is rebuilt by the second window alone.
        owners.retain(second, &key(1));
        assert!(owners.remove_owner(first).is_empty());
        assert!(owners.release(second, &key(1)));
    }

    #[test]
    fn unshared_tiles_are_freed_on_removal() {
        let mut owners = TileOwners::new(false);
        let owner = owners.add_owner();
        owners.retain(owner, &key(1));
        assert!(owners.release(owner, &key(1)));
        assert!(
            owners.release(owner, &key(1)),
            "the atlas ignores unknown keys"
        );
        assert!(owners.remove_owner(owner).is_empty());
        assert!(owners.counts.is_empty() && owners.owners.is_empty());
    }
}
