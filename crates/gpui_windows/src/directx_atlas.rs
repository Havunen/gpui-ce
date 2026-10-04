use crate::bindings::Windows::Win32::{
    D3D11_BIND_SHADER_RESOURCE, D3D11_BOX, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, ID3D11Device,
    ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D, *,
};
use collections::{FxHashMap, FxHashSet};
use etagere::BucketedAtlasAllocator;
use parking_lot::Mutex;
use std::sync::Arc;

use gpui::{
    AtlasKey, AtlasTextureId, AtlasTextureKind, AtlasTextureList, AtlasTile, Bounds, DevicePixels,
    PlatformAtlas, Point, Size,
};

pub(crate) struct DirectXAtlas {
    state: Arc<Mutex<DirectXAtlasState>>,
    owned: Mutex<(u64, FxHashSet<AtlasKey>)>,
}
thread_local! { static SHARED: std::cell::RefCell<Vec<std::sync::Weak<Mutex<DirectXAtlasState>>>> = const { std::cell::RefCell::new(Vec::new()) }; }

struct DirectXAtlasState {
    shared: bool,
    device: ID3D11Device,
    device_context: ID3D11DeviceContext,
    monochrome_textures: AtlasTextureList<DirectXAtlasTexture>,
    polychrome_textures: AtlasTextureList<DirectXAtlasTexture>,
    subpixel_textures: AtlasTextureList<DirectXAtlasTexture>,
    tiles_by_key: FxHashMap<AtlasKey, AtlasTile>,
    generation: u64,
    epoch: u64,
    owners: FxHashMap<AtlasKey, usize>,
}

struct DirectXAtlasTexture {
    id: AtlasTextureId,
    bytes_per_pixel: u32,
    allocator: BucketedAtlasAllocator,
    texture: ID3D11Texture2D,
    view: [Option<ID3D11ShaderResourceView>; 1],
    live_atlas_keys: u32,
}

impl DirectXAtlas {
    pub(crate) fn new(device: &ID3D11Device, device_context: &ID3D11DeviceContext) -> Self {
        let make = || Self {
            state: Arc::new(Mutex::new(DirectXAtlasState {
                shared: gpui_render::gpu_policy::GpuOptions::from_env().shared_resources,
                device: device.clone(),
                device_context: device_context.clone(),
                monochrome_textures: Default::default(),
                polychrome_textures: Default::default(),
                subpixel_textures: Default::default(),
                tiles_by_key: Default::default(),
                generation: 0,
                epoch: 0,
                owners: FxHashMap::default(),
            })),
            owned: Mutex::default(),
        };
        if !gpui_render::gpu_policy::GpuOptions::from_env().shared_resources {
            return make();
        }
        SHARED.with_borrow_mut(|cache| {
            cache.retain(|s| s.strong_count() > 0);
            for shared in cache.iter().filter_map(std::sync::Weak::upgrade) {
                let lock = shared.lock();
                if lock.device == *device {
                    let epoch = lock.epoch;
                    drop(lock);
                    return Self {
                        state: shared,
                        owned: Mutex::new((epoch, FxHashSet::default())),
                    };
                }
            }
            let atlas = make();
            cache.push(Arc::downgrade(&atlas.state));
            atlas
        })
    }

    pub(crate) fn allocated_bytes(&self) -> u64 {
        let state = self.state.lock();
        [
            &state.monochrome_textures,
            &state.polychrome_textures,
            &state.subpixel_textures,
        ]
        .into_iter()
        .flat_map(|list| list.textures.iter().flatten())
        .map(|tile| {
            let mut desc = D3D11_TEXTURE2D_DESC::default();
            unsafe {
                tile.texture.GetDesc(&mut desc);
            }
            u64::from(desc.Width) * u64::from(desc.Height) * u64::from(tile.bytes_per_pixel)
        })
        .sum()
    }

    pub(crate) fn get_texture_view(
        &self,
        id: AtlasTextureId,
    ) -> anyhow::Result<[Option<ID3D11ShaderResourceView>; 1]> {
        let lock = self.state.lock();
        let tex = lock
            .texture(id)
            .ok_or_else(|| anyhow::anyhow!("missing DirectX atlas texture {id:?}"))?;
        Ok(tex.view.clone())
    }

    pub(crate) fn handle_device_lost(
        &self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
    ) {
        let mut lock = self.state.lock();
        lock.device = device.clone();
        lock.device_context = device_context.clone();
        lock.monochrome_textures = AtlasTextureList::default();
        lock.polychrome_textures = AtlasTextureList::default();
        lock.subpixel_textures = AtlasTextureList::default();
        lock.tiles_by_key.clear();
        lock.owners.clear();
        lock.epoch = lock.epoch.wrapping_add(1);
        lock.generation = lock.generation.wrapping_add(1);
    }
}

impl PlatformAtlas for DirectXAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> anyhow::Result<
            Option<(Size<DevicePixels>, std::borrow::Cow<'a, [u8]>)>,
        >,
    ) -> anyhow::Result<Option<AtlasTile>> {
        let mut lock = self.state.lock();
        if let Some(tile) = lock.tiles_by_key.get(key).copied() {
            self.retain_key(&mut lock, key);
            Ok(Some(tile))
        } else {
            let Some((size, bytes)) = build()? else {
                return Ok(None);
            };
            // Validate before allocation: a rejected bitmap must never leave a cached,
            // uninitialized tile that every later glyph/SVG lookup treats as successful.
            key.texture_kind().validate_upload(size, &bytes)?;
            anyhow::ensure!(
                size.width.0 <= 16384 && size.height.0 <= 16384,
                "atlas tile {size:?} exceeds the Direct3D 11 texture limit"
            );
            let tile = lock
                .allocate(size, key.texture_kind())
                .ok_or_else(|| anyhow::anyhow!("failed to allocate"))?;
            let texture = lock.texture(tile.texture_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "missing newly allocated DirectX atlas texture {:?}",
                    tile.texture_id
                )
            })?;
            texture.upload(&lock.device_context, tile.bounds, &bytes);
            lock.tiles_by_key.insert(key.clone(), tile);
            self.retain_key(&mut lock, key);
            Ok(Some(tile))
        }
    }

    fn remove(&self, key: &AtlasKey) {
        let mut lock = self.state.lock();

        if !lock.shared {
            lock.release_key(key);
            return;
        }
        let mut owned = self.owned.lock();
        if owned.0 != lock.epoch || !owned.1.remove(key) {
            return;
        }
        lock.release_key(key);
    }

    fn generation(&self) -> u64 {
        self.state.lock().generation
    }
}

impl DirectXAtlasState {
    fn release_key(&mut self, key: &AtlasKey) {
        if self.shared {
            let Some(owners) = self.owners.get_mut(key) else {
                return;
            };
            *owners -= 1;
            if *owners > 0 {
                return;
            }
            self.owners.remove(key);
        }
        let Some(tile) = self.tiles_by_key.remove(key) else {
            return;
        };
        let id = tile.texture_id;

        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
        };

        let Some(texture_slot) = textures.textures.get_mut(id.index as usize) else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                textures.free_list.push(texture.id.index as usize);
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
                AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
            };

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind)?;
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> Option<&mut DirectXAtlasTexture> {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        // Max texture size for DirectX. See:
        // https://learn.microsoft.com/en-us/windows/win32/direct3d11/overviews-direct3d-11-resources-limits
        const MAX_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(16384),
            height: DevicePixels(16384),
        };
        let size = min_size.min(&MAX_ATLAS_SIZE).max(&DEFAULT_ATLAS_SIZE);
        let pixel_format;
        let bind_flag;
        let bytes_per_pixel;
        match kind {
            AtlasTextureKind::Monochrome => {
                pixel_format = DXGI_FORMAT_R8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 1;
            }
            AtlasTextureKind::Polychrome => {
                pixel_format = DXGI_FORMAT_B8G8R8A8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 4;
            }
            AtlasTextureKind::Subpixel => {
                pixel_format = DXGI_FORMAT_R8G8B8A8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 4;
            }
        }
        let texture_desc = D3D11_TEXTURE2D_DESC {
            Width: size.width.0 as u32,
            Height: size.height.0 as u32,
            MipLevels: 1,
            ArraySize: 1,
            Format: pixel_format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: bind_flag as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture: Option<ID3D11Texture2D> = None;
        unsafe {
            // This only returns None if the device is lost, which we will recreate later.
            // So it's ok to return None here.
            self.device
                .CreateTexture2D(&texture_desc, None, Some(&mut texture))
                .ok()
                .ok()?;
        }
        let texture = texture.unwrap();

        let texture_list = match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
        };
        let index = texture_list.free_list.pop();
        let view = unsafe {
            let mut view = None;
            self.device
                .CreateShaderResourceView(&texture, None, Some(&mut view))
                .ok()
                .ok()?;
            [view]
        };
        let atlas_texture = DirectXAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            bytes_per_pixel,
            allocator: etagere::BucketedAtlasAllocator::new(device_size_to_etagere(size)),
            texture,
            view,
            live_atlas_keys: 0,
        };
        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix).unwrap().as_mut()
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut().unwrap().as_mut()
        }
    }

    fn texture(&self, id: AtlasTextureId) -> Option<&DirectXAtlasTexture> {
        match id.kind {
            AtlasTextureKind::Monochrome => {
                self.monochrome_textures.textures.get(id.index as usize)
            }
            AtlasTextureKind::Polychrome => {
                self.polychrome_textures.textures.get(id.index as usize)
            }
            AtlasTextureKind::Subpixel => self.subpixel_textures.textures.get(id.index as usize),
        }
        .and_then(|texture| texture.as_ref())
    }
}

impl DirectXAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(device_size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            bounds: Bounds {
                origin: etagere_point_to_device(allocation.rectangle.min),
                size,
            },
            padding: 0,
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn upload(
        &self,
        device_context: &ID3D11DeviceContext,
        bounds: Bounds<DevicePixels>,
        bytes: &[u8],
    ) {
        // The insertion boundary validates the exact byte count before allocating this tile.
        debug_assert!(self.id.kind.validate_upload(bounds.size, bytes).is_ok());
        unsafe {
            device_context.UpdateSubresource(
                &self.texture,
                0,
                Some(&D3D11_BOX {
                    left: bounds.left().0 as u32,
                    top: bounds.top().0 as u32,
                    front: 0,
                    right: bounds.right().0 as u32,
                    bottom: bounds.bottom().0 as u32,
                    back: 1,
                }),
                bytes.as_ptr() as _,
                bounds.size.width.to_bytes(self.bytes_per_pixel as u8),
                0,
            );
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&mut self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn device_size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    etagere::Size::new(size.width.into(), size.height.into())
}

fn etagere_point_to_device(value: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels::from(value.x),
        y: DevicePixels::from(value.y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::Windows::Win32::{
        D3D_DRIVER_TYPE_WARP, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION,
        D3D11CreateDevice, HMODULE,
    };
    use gpui::{ImageId, RenderImageParams};
    use std::borrow::Cow;

    fn create_atlas() -> Option<DirectXAtlas> {
        let mut device: Option<ID3D11Device> = None;
        let mut device_context: Option<ID3D11DeviceContext> = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_WARP,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT as u32,
                None,
                D3D11_SDK_VERSION as u32,
                Some(&mut device),
                None,
                Some(&mut device_context),
            )
            .ok()
        }
        .ok()?;
        Some(DirectXAtlas::new(&device?, &device_context?))
    }

    fn make_image_key(image_id: usize) -> AtlasKey {
        AtlasKey::Image(RenderImageParams {
            image_id: ImageId(image_id),
            frame_index: 0,
        })
    }

    fn insert_tile(atlas: &DirectXAtlas, key: &AtlasKey, size: Size<DevicePixels>) -> AtlasTile {
        atlas
            .get_or_insert_with(key, &mut || {
                let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
                Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
            })
            .expect("allocation should succeed")
            .expect("callback returns Some")
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

        let keeper_key = make_image_key(1);
        let big_key_a = make_image_key(2);
        let big_key_b = make_image_key(3);

        let keeper_tile = insert_tile(&atlas, &keeper_key, small);
        let tile_a = insert_tile(&atlas, &big_key_a, big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);

        let tile_b = insert_tile(&atlas, &big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
    }
}

impl DirectXAtlas {
    fn retain_key(&self, lock: &mut DirectXAtlasState, key: &AtlasKey) {
        if !lock.shared {
            return;
        }
        let mut owned = self.owned.lock();
        if owned.0 != lock.epoch {
            owned.0 = lock.epoch;
            owned.1.clear();
        }
        if owned.1.insert(key.clone()) {
            *lock.owners.entry(key.clone()).or_default() += 1;
        }
    }
}
impl Drop for DirectXAtlas {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        if !state.shared {
            return;
        }
        let owned = self.owned.get_mut();
        if owned.0 == state.epoch {
            for key in owned.1.drain() {
                state.release_key(&key);
            }
        }
    }
}

#[cfg(test)]
mod shared_atlas_tests {
    use super::*;
    #[test]
    fn shared_atlas_survives_one_window_closing_and_releases_the_final_lease() -> anyhow::Result<()>
    {
        let devices = crate::directx_devices::DirectXDevices::new()?;
        let first = DirectXAtlas::new(&devices.device, &devices.device_context);
        first.state.lock().shared = true;
        let second = DirectXAtlas {
            state: first.state.clone(),
            owned: Mutex::default(),
        };
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
        assert_eq!(second.state.lock().owners[&key], 1);
        first.get_or_insert_with(&key, &mut build)?;
        drop(first);
        assert_eq!(second.state.lock().owners[&key], 1);
        second.remove(&key);
        assert_eq!(second.allocated_bytes(), 0);
        let weak = Arc::downgrade(&second.state);
        drop(second);
        assert!(weak.upgrade().is_none());
        Ok(())
    }
}
