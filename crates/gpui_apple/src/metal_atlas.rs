use anyhow::{Context as _, Result};
use collections::FxHashMap;
use derive_more::{Deref, DerefMut};
use etagere::BucketedAtlasAllocator;
use foreign_types::ForeignTypeRef;
use gpui::{
    AtlasKey, AtlasTextureId, AtlasTextureKind, AtlasTextureList, AtlasTile, Bounds, DevicePixels,
    PlatformAtlas, Point, Size,
};
use gpui_render::sharing::{OwnerId, Registry, TileOwners};
use metal::Device;
use parking_lot::Mutex;
use std::borrow::Cow;
use std::sync::Arc;

/// A window's view of an atlas. With shared resources, every window on a device uses
/// one atlas state, and a tile lives until the last window using it removes it.
pub struct MetalAtlas {
    state: Arc<Mutex<MetalAtlasState>>,
    owner: OwnerId,
}

thread_local! {
    static SHARED_ATLASES: Registry<Mutex<MetalAtlasState>> = const { Registry::new() };
}

impl MetalAtlas {
    pub(crate) fn new(device: Device, is_apple_gpu: bool) -> Self {
        let shared = gpui_render::gpu_policy::GpuOptions::from_env().shared_resources;
        let state = if shared {
            SHARED_ATLASES.with(|atlases| {
                atlases.get_or_insert_with(
                    |state| state.lock().device.as_ptr() == device.as_ptr(),
                    || MetalAtlasState::new(device.clone(), is_apple_gpu, true),
                )
            })
        } else {
            Arc::new(MetalAtlasState::new(device, is_apple_gpu, false))
        };
        Self::with_state(state)
    }

    fn with_state(state: Arc<Mutex<MetalAtlasState>>) -> Self {
        let owner = state.lock().owners.add_owner();
        Self { state, owner }
    }

    pub(crate) fn allocated_bytes(&self) -> u64 {
        let state = self.state.lock();
        [&state.monochrome_textures, &state.polychrome_textures]
            .into_iter()
            .flat_map(|list| list.textures.iter().flatten())
            .map(|tile| {
                tile.metal_texture.width()
                    * tile.metal_texture.height()
                    * if tile.id.kind == AtlasTextureKind::Monochrome {
                        1
                    } else {
                        4
                    }
            })
            .sum()
    }

    pub(crate) fn metal_texture(&self, id: AtlasTextureId) -> metal::Texture {
        self.state.lock().texture(id).metal_texture.clone()
    }
}

struct MetalAtlasState {
    device: AssertSend<Device>,
    is_apple_gpu: bool,
    monochrome_textures: AtlasTextureList<MetalAtlasTexture>,
    polychrome_textures: AtlasTextureList<MetalAtlasTexture>,
    tiles_by_key: FxHashMap<AtlasKey, AtlasTile>,
    generation: u64,
    owners: TileOwners,
}

impl PlatformAtlas for MetalAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        let mut lock = self.state.lock();
        let tile = match lock.tiles_by_key.get(key) {
            Some(tile) => *tile,
            None => {
                let Some((size, bytes)) = build()? else {
                    return Ok(None);
                };
                let tile = lock
                    .allocate(size, key.texture_kind())
                    .context("failed to allocate")?;
                let texture = lock.texture(tile.texture_id);
                texture.upload(tile.bounds, &bytes);
                lock.tiles_by_key.insert(key.clone(), tile);
                tile
            }
        };
        lock.owners.retain(self.owner, key);
        Ok(Some(tile))
    }

    fn remove(&self, key: &AtlasKey) {
        let mut lock = self.state.lock();
        if lock.owners.release(self.owner, key) {
            lock.free_tile(key);
        }
    }

    fn generation(&self) -> u64 {
        self.state.lock().generation
    }
}

impl MetalAtlasState {
    fn new(device: Device, is_apple_gpu: bool, shared: bool) -> Mutex<Self> {
        Mutex::new(Self {
            device: AssertSend(device),
            is_apple_gpu,
            monochrome_textures: Default::default(),
            polychrome_textures: Default::default(),
            tiles_by_key: Default::default(),
            generation: 0,
            owners: TileOwners::new(shared),
        })
    }

    fn free_tile(&mut self, key: &AtlasKey) {
        let Some(tile) = self.tiles_by_key.remove(key) else {
            return;
        };
        let id = tile.texture_id;

        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };

        let Some(texture_slot) = textures
            .textures
            .iter_mut()
            .find(|texture| texture.as_ref().is_some_and(|v| v.id == id))
        else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                textures.free_list.push(id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
            self.generation = self.generation.wrapping_add(1);
        }
    }

    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = match texture_kind {
                AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
                AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
                AtlasTextureKind::Subpixel => unreachable!(),
            };

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut MetalAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        // Max texture size on all modern Apple GPUs. Anything bigger than that crashes in validateWithDevice.
        const MAX_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(16384),
            height: DevicePixels(16384),
        };
        let size = min_size.min(&MAX_ATLAS_SIZE).max(&DEFAULT_ATLAS_SIZE);
        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.into());
        texture_descriptor.set_height(size.height.into());
        let pixel_format;
        let usage;
        match kind {
            AtlasTextureKind::Monochrome => {
                // Shared shaders read coverage from red, like WGPU's R8 atlas; A8 samples
                // coverage through alpha and made every glyph transparent.
                pixel_format = metal::MTLPixelFormat::R8Unorm;
                usage = metal::MTLTextureUsage::ShaderRead;
            }
            AtlasTextureKind::Polychrome => {
                pixel_format = metal::MTLPixelFormat::BGRA8Unorm;
                usage = metal::MTLTextureUsage::ShaderRead;
            }
            AtlasTextureKind::Subpixel => unreachable!(),
        }
        texture_descriptor.set_pixel_format(pixel_format);
        texture_descriptor.set_usage(usage);
        // Shared memory mode can be used only on Apple GPU families
        // https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared
        texture_descriptor.set_storage_mode(if self.is_apple_gpu {
            metal::MTLStorageMode::Shared
        } else {
            metal::MTLStorageMode::Managed
        });
        let metal_texture = self.device.new_texture(&texture_descriptor);

        let texture_list = match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };

        let index = texture_list.free_list.pop();

        let atlas_texture = MetalAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            allocator: etagere::BucketedAtlasAllocator::new(size_to_etagere(size)),
            metal_texture: AssertSend(metal_texture),
            live_atlas_keys: 0,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix)
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut()
        }
        .unwrap()
        .as_mut()
        .unwrap()
    }

    fn texture(&self, id: AtlasTextureId) -> &MetalAtlasTexture {
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };
        textures[id.index as usize].as_ref().unwrap()
    }
}

struct MetalAtlasTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    metal_texture: AssertSend<metal::Texture>,
    live_atlas_keys: u32,
}

impl MetalAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            bounds: Bounds {
                origin: point_from_etagere(allocation.rectangle.min),
                size,
            },
            padding: 0,
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn upload(&self, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        let region = metal::MTLRegion::new_2d(
            bounds.origin.x.into(),
            bounds.origin.y.into(),
            bounds.size.width.into(),
            bounds.size.height.into(),
        );
        self.metal_texture.replace_region(
            region,
            0,
            bytes.as_ptr() as *const _,
            bounds.size.width.to_bytes(self.bytes_per_pixel()) as u64,
        );
    }

    fn bytes_per_pixel(&self) -> u8 {
        use metal::MTLPixelFormat::*;
        match self.metal_texture.pixel_format() {
            A8Unorm | R8Unorm => 1,
            RGBA8Unorm | BGRA8Unorm => 4,
            _ => unimplemented!(),
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&mut self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    etagere::Size::new(size.width.into(), size.height.into())
}

fn point_from_etagere(value: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels::from(value.x),
        y: DevicePixels::from(value.y),
    }
}

#[derive(Deref, DerefMut)]
struct AssertSend<T>(T);

unsafe impl<T> Send for AssertSend<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::PlatformAtlas;
    use std::borrow::Cow;

    fn create_atlas() -> Option<MetalAtlas> {
        let device = metal::Device::system_default()?;
        Some(MetalAtlas::new(device, true))
    }

    fn make_image_key(image_id: usize, frame_index: usize) -> AtlasKey {
        AtlasKey::Image(gpui::RenderImageParams {
            image_id: gpui::ImageId(image_id),
            frame_index,
        })
    }

    fn insert_tile(atlas: &MetalAtlas, key: &AtlasKey, size: Size<DevicePixels>) -> AtlasTile {
        atlas
            .get_or_insert_with(key, &mut || {
                let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
                Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
            })
            .expect("allocation should succeed")
            .expect("callback returns Some")
    }

    #[test]
    fn test_remove_clears_stale_keys_from_tiles_by_key() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };

        let key_a = make_image_key(1, 0);
        let key_b = make_image_key(2, 0);
        let key_c = make_image_key(3, 0);

        let tile_a = insert_tile(&atlas, &key_a, small);
        let tile_b = insert_tile(&atlas, &key_b, small);
        let tile_c = insert_tile(&atlas, &key_c, small);

        assert_eq!(tile_a.texture_id, tile_b.texture_id);
        assert_eq!(tile_b.texture_id, tile_c.texture_id);

        // Remove A: texture still has B and C, so it stays.
        // The key for A must be removed from tiles_by_key.
        atlas.remove(&key_a);

        // Remove B: texture still has C.
        atlas.remove(&key_b);

        // Remove C: texture becomes unreferenced and is deleted.
        atlas.remove(&key_c);

        // Re-inserting A must allocate a fresh tile on a new texture,
        // NOT return a stale tile referencing the deleted texture.
        let tile_a2 = insert_tile(&atlas, &key_a, small);

        // The texture must actually exist — this would panic before the fix.
        let _texture = atlas.metal_texture(tile_a2.texture_id);
    }

    #[test]
    fn test_remove_deallocates_tile_space_for_reuse() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let big = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };

        let keeper_key = make_image_key(1, 0);
        let big_key_a = make_image_key(2, 0);
        let big_key_b = make_image_key(3, 0);

        let keeper_tile = insert_tile(&atlas, &keeper_key, small);
        let tile_a = insert_tile(&atlas, &big_key_a, big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);
        let tile_b = insert_tile(&atlas, &big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
    }

    #[test]
    fn test_remove_nonexistent_key_is_noop() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let key = make_image_key(999, 0);
        atlas.remove(&key);
    }

    #[test]
    fn monochrome_atlas_uses_red_channel_coverage() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let key = AtlasKey::Svg(gpui::RenderSvgParams {
            path: "monochrome-format-test".into(),
            size: Size {
                width: DevicePixels(8),
                height: DevicePixels(8),
            },
        });
        let tile = atlas
            .get_or_insert_with(&key, &mut || {
                Ok(Some((
                    Size {
                        width: DevicePixels(8),
                        height: DevicePixels(8),
                    },
                    Cow::Owned(vec![255; 64]),
                )))
            })
            .unwrap()
            .unwrap();

        assert_eq!(
            atlas.metal_texture(tile.texture_id).pixel_format(),
            metal::MTLPixelFormat::R8Unorm
        );
    }
}

impl Drop for MetalAtlas {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        for key in state.owners.remove_owner(self.owner) {
            state.free_tile(&key);
        }
    }
}

#[cfg(test)]
mod shared_atlas_tests {
    use super::*;

    #[test]
    fn shared_atlas_survives_one_window_closing_and_releases_the_final_lease() -> anyhow::Result<()>
    {
        let device = metal::Device::system_default().expect("Metal device required");
        let is_apple_gpu = device.supports_family(metal::MTLGPUFamily::Apple1);
        let state = Arc::new(MetalAtlasState::new(device, is_apple_gpu, true));
        let first = MetalAtlas::with_state(state.clone());
        let second = MetalAtlas::with_state(state);
        let key = AtlasKey::Image(gpui::RenderImageParams {
            image_id: gpui::ImageId(8),
            frame_index: 0,
        });
        let mut build = || {
            Ok(Some((
                Size {
                    width: DevicePixels(1),
                    height: DevicePixels(1),
                },
                std::borrow::Cow::Owned(vec![0, 0, 0, 255]),
            )))
        };
        let tile = first.get_or_insert_with(&key, &mut build)?.unwrap();
        assert_eq!(
            second
                .get_or_insert_with(&key, &mut || panic!("shared tile rebuilt"))?
                .unwrap(),
            tile
        );
        first.remove(&key);
        first.remove(&key);
        assert!(second.state.lock().tiles_by_key.contains_key(&key));
        first.get_or_insert_with(&key, &mut build)?;
        drop(first);
        assert!(
            second.state.lock().tiles_by_key.contains_key(&key),
            "a closing window keeps tiles another window uses"
        );
        second.remove(&key);
        assert_eq!(second.allocated_bytes(), 0);
        let weak = Arc::downgrade(&second.state);
        drop(second);
        assert!(weak.upgrade().is_none());
        Ok(())
    }
}
