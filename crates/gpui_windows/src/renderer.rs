//! Process-wide Windows renderer selection. Storage belongs to the application.
use crate::{DirectXDevices, DirectXRenderer, HWND, LazyDirectXDevices, SharedDirectXDevices};
use anyhow::{Context as _, Result, bail};
use gpui::{DevicePixels, GpuSpecs, PlatformAtlas, Scene, Size, WindowBackgroundAppearance};
use std::{cell::Cell, rc::Rc, sync::Arc};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WindowsRendererPreference {
    Auto,
    #[default]
    Dx11,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowsRendererBackend {
    Dx11,
    Dx12,
}

#[derive(Clone, Debug)]
pub enum WindowsRendererEvent {
    Selected {
        backend: WindowsRendererBackend,
        overridden: bool,
    },
    /// Called on the foreground thread, outside GPU locks, before falling back to DX11.
    /// A `runtime` failure switches the failing window to DX11 in place; later windows
    /// also start on DX11, and a `Selected` event follows once the window has switched.
    Failed { message: String, runtime: bool },
}

#[derive(Clone, Default)]
pub struct WindowsRendererOptions {
    pub preference: WindowsRendererPreference,
    pub on_event: Option<Rc<dyn Fn(WindowsRendererEvent)>>,
}

#[derive(Clone)]
pub(crate) struct RendererContext(Rc<RendererState>);

struct RendererState {
    options: WindowsRendererOptions,
    backend: Cell<Option<WindowsRendererBackend>>,
    overridden: bool,
    failure_reported: Cell<bool>,
    #[cfg(feature = "windows-wgpu")]
    failed_recoveries: Cell<u8>,
    #[cfg(feature = "windows-wgpu")]
    gpu: gpui_wgpu::GpuContext,
}

impl RendererContext {
    pub(crate) fn new(options: WindowsRendererOptions) -> Result<Self> {
        Self::with_override(
            options,
            std::env::var("GPUI_WINDOWS_RENDERER").ok().as_deref(),
        )
    }

    fn with_override(options: WindowsRendererOptions, value: Option<&str>) -> Result<Self> {
        let backend = match value {
            Some("dx11") => Some(WindowsRendererBackend::Dx11),
            #[cfg(feature = "windows-wgpu")]
            Some("wgpu-dx12") => Some(WindowsRendererBackend::Dx12),
            Some(value) => bail!(
                "Invalid or unavailable GPUI_WINDOWS_RENDERER={value:?}; expected dx11 or wgpu-dx12"
            ),
            None if options.preference == WindowsRendererPreference::Dx11 => {
                Some(WindowsRendererBackend::Dx11)
            }
            None => None,
        };
        Ok(Self(Rc::new(RendererState {
            options,
            backend: Cell::new(backend),
            overridden: value.is_some(),
            failure_reported: Cell::new(false),
            #[cfg(feature = "windows-wgpu")]
            failed_recoveries: Cell::new(0),
            #[cfg(feature = "windows-wgpu")]
            gpu: Default::default(),
        })))
    }

    fn emit(&self, event: WindowsRendererEvent) {
        if let Some(callback) = &self.0.options.on_event {
            callback(event);
        }
    }

    /// After a DX12 window fell back at runtime, later windows start on DX11.
    #[cfg(feature = "windows-wgpu")]
    fn use_dx11_for_new_windows(&self) {
        self.0.backend.set(Some(WindowsRendererBackend::Dx11));
    }

    fn failed(&self, message: String, runtime: bool) {
        if !self.0.overridden && !self.0.failure_reported.replace(true) {
            log::warn!("DX12 unavailable: {message}");
            self.emit(WindowsRendererEvent::Failed { message, runtime });
        }
    }

    #[cfg(feature = "windows-wgpu")]
    fn failed_recovery(&self, window_failures: &mut u8, error: &anyhow::Error) -> bool {
        // Stabilization waits never enter this function. Bound device-wide attempts and
        // keep one window's successful recovery from resetting another window's failures.
        let attempts = self.0.failed_recoveries.get() + 1;
        self.0.failed_recoveries.set(attempts);
        *window_failures += 1;
        if attempts >= 3 || *window_failures >= 3 {
            self.failed(
                format!("DX12 recovery failed after three attempts: {error:#}"),
                true,
            );
            return true;
        }
        log::warn!("DX12 recovery attempt failed: {error:#}");
        false
    }

    fn create_window(
        &self,
        hwnd: HWND,
        devices: &SharedDirectXDevices,
        disable_composition: bool,
    ) -> Result<Backend> {
        self.create_with_devices(
            devices,
            || DirectXDevices::new().context("Creating DirectX devices"),
            dx12_supported,
            || {
                #[cfg(feature = "windows-wgpu")]
                {
                    Dx12Renderer::new(hwnd, self.clone())
                        .map(|renderer| Backend::Dx12(Box::new(renderer)))
                }
                #[cfg(not(feature = "windows-wgpu"))]
                {
                    bail!("DX12 renderer is unavailable in this build")
                }
            },
            |devices| DirectXRenderer::new(hwnd, devices, disable_composition).map(Backend::Dx11),
        )
    }

    fn create_with_devices<D: Clone, T>(
        &self,
        devices: &LazyDirectXDevices<D>,
        create_devices: impl FnOnce() -> Result<D>,
        probe: impl FnOnce() -> Result<bool>,
        dx12: impl FnOnce() -> Result<T>,
        dx11: impl FnOnce(&D) -> Result<T>,
    ) -> Result<T> {
        self.create_with(probe, dx12, || {
            // Successful DX12 selection never acquires a DX11 device. Release
            // the slot lock before creating a renderer or emitting callbacks.
            let devices = devices.get_or_try_init(create_devices)?;
            dx11(&devices)
        })
    }

    fn create_with<T>(
        &self,
        probe: impl FnOnce() -> Result<bool>,
        dx12: impl FnOnce() -> Result<T>,
        dx11: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let first_auto = self.0.backend.get().is_none();
        if first_auto {
            match probe() {
                Ok(true) => self.0.backend.set(Some(WindowsRendererBackend::Dx12)),
                result => {
                    let message = match result {
                        Ok(false) => "DirectX 12 is not supported by any hardware adapter".into(),
                        Err(error) => format!("DX12 support check failed: {error:#}"),
                        Ok(true) => unreachable!(),
                    };
                    self.failed(message, false);
                    self.0.backend.set(Some(WindowsRendererBackend::Dx11));
                }
            }
        }
        if self.0.backend.get() == Some(WindowsRendererBackend::Dx12) {
            match dx12() {
                Ok(renderer) => {
                    self.emit(WindowsRendererEvent::Selected {
                        backend: WindowsRendererBackend::Dx12,
                        overridden: self.0.overridden,
                    });
                    return Ok(renderer);
                }
                Err(error) => {
                    self.failed(
                        format!("DX12 initialization failed: {error:#}"),
                        !first_auto,
                    );
                    if !first_auto {
                        return Err(error);
                    }
                    // The failed renderer has dropped its surface/resources. Release the shared device too.
                    #[cfg(feature = "windows-wgpu")]
                    {
                        *self.0.gpu.borrow_mut() = None;
                    }
                    self.0.backend.set(Some(WindowsRendererBackend::Dx11));
                }
            }
        }
        let renderer = dx11()?;
        self.emit(WindowsRendererEvent::Selected {
            backend: WindowsRendererBackend::Dx11,
            overridden: self.0.overridden,
        });
        Ok(renderer)
    }
}

#[cfg(not(feature = "windows-wgpu"))]
fn dx12_supported() -> Result<bool> {
    Ok(false)
}

#[cfg(feature = "windows-wgpu")]
fn dx12_supported() -> Result<bool> {
    use crate::bindings::Windows::Win32::*;
    use windows_core::{GUID, HRESULT, Interface};
    // System32-only loading avoids an application-local DLL influencing the support probe.
    windows_core::link!("kernel32.dll" "system" fn LoadLibraryExW(name: *const u16, file: HMODULE, flags: u32) -> HMODULE);
    let module = unsafe {
        LoadLibraryExW(
            windows_core::w!("d3d12.dll").as_ptr(),
            HMODULE::default(),
            0x800,
        )
    };
    if module.0.is_null() {
        return Ok(false);
    }
    let result = (|| {
        let Some(raw) = (unsafe { GetProcAddress(module, windows_core::s!("D3D12CreateDevice")) })
        else {
            return Ok(false);
        };
        type Probe = unsafe extern "system" fn(
            *mut std::ffi::c_void,
            i32,
            *const GUID,
            *mut *mut std::ffi::c_void,
        ) -> HRESULT;
        let probe: Probe = unsafe { std::mem::transmute(raw) };
        let iid = GUID::from_u128(0x189819f1_1db6_4b57_be54_1821339b85f7);
        // Capability probing requires a DXGI factory, not a D3D11 device/context.
        let factory = crate::directx_devices::get_dxgi_factory(false)?;
        for index in 0.. {
            let adapter: IDXGIAdapter1 = match unsafe { factory.EnumAdapters(index) } {
                Ok(adapter) => adapter.cast()?,
                Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(error) => return Err(error.into()),
            };
            let mut desc = DXGI_ADAPTER_DESC1::default();
            unsafe {
                adapter.GetDesc1(&mut desc).ok()?;
            }
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE as u32 != 0 {
                continue;
            }
            // DX12 can run at feature level 11_0; the API version and feature level are distinct.
            if unsafe {
                probe(
                    adapter.as_raw(),
                    D3D_FEATURE_LEVEL_11_0,
                    &iid,
                    std::ptr::null_mut(),
                )
            }
            .is_ok()
            {
                return Ok(true);
            }
        }
        Ok(false)
    })();
    unsafe {
        let _ = FreeLibrary(module);
    }
    result
}

/// A window's renderer. A DX12 window whose renderer fails at runtime switches to DX11
/// in place instead of taking the process down.
pub(crate) struct WindowRenderer {
    backend: Backend,
    #[cfg(feature = "windows-wgpu")]
    fallback: Dx11Fallback,
    /// gpui keeps the atlas a window returned when it opened. A DX12 window returns this
    /// one, which follows the window to DX11.
    #[cfg(feature = "windows-wgpu")]
    atlas: Option<Arc<SwitchableAtlas>>,
    /// Set by a fallback: the next frame must render everything into the DX11 atlas.
    #[cfg(feature = "windows-wgpu")]
    force_redraw: bool,
}

enum Backend {
    Dx11(DirectXRenderer),
    #[cfg(feature = "windows-wgpu")]
    Dx12(Box<Dx12Renderer>),
    /// DX12 failed and DX11 could not replace it: the window no longer draws.
    #[cfg(feature = "windows-wgpu")]
    Unavailable {
        atlas: Arc<dyn PlatformAtlas>,
    },
}

/// What a DX12 window needs to create its DX11 replacement.
#[cfg(feature = "windows-wgpu")]
struct Dx11Fallback {
    hwnd: HWND,
    devices: SharedDirectXDevices,
    disable_composition: bool,
    /// The last size requested for the window, which a new renderer must adopt.
    size: Option<Size<DevicePixels>>,
}

#[cfg(feature = "windows-wgpu")]
impl Dx11Fallback {
    fn create(&self) -> Result<DirectXRenderer> {
        let devices = self
            .devices
            .get_or_try_init(|| DirectXDevices::new().context("Creating DirectX devices"))?;
        let mut renderer = DirectXRenderer::new(self.hwnd, &devices, self.disable_composition)?;
        if let Some(size) = self.size {
            renderer.resize(size)?;
        }
        Ok(renderer)
    }
}

impl WindowRenderer {
    pub(crate) fn new(
        hwnd: HWND,
        devices: &SharedDirectXDevices,
        disable_composition: bool,
        context: &RendererContext,
    ) -> Result<Self> {
        let backend = context.create_window(hwnd, devices, disable_composition)?;
        Ok(Self {
            #[cfg(feature = "windows-wgpu")]
            atlas: match &backend {
                Backend::Dx12(renderer) => Some(SwitchableAtlas::new(
                    renderer.renderer.sprite_atlas().clone(),
                )),
                _ => None,
            },
            backend,
            #[cfg(feature = "windows-wgpu")]
            fallback: Dx11Fallback {
                hwnd,
                devices: devices.clone(),
                disable_composition,
                size: None,
            },
            #[cfg(feature = "windows-wgpu")]
            force_redraw: false,
        })
    }

    pub(crate) fn release_retained_frame(&mut self) {
        match &mut self.backend {
            Backend::Dx11(renderer) => renderer.release_retained_frame(),
            #[cfg(feature = "windows-wgpu")]
            Backend::Dx12(renderer) => renderer.renderer.release_retained_frame(),
            #[cfg(feature = "windows-wgpu")]
            Backend::Unavailable { .. } => {}
        }
    }

    pub(crate) fn draw(
        &mut self,
        scene: &Scene,
        appearance: WindowBackgroundAppearance,
    ) -> Result<()> {
        match &mut self.backend {
            Backend::Dx11(renderer) => renderer.draw(scene, appearance),
            // DX12 windows are always opaque. A non-opaque appearance would reconfigure
            // the swap chain and rebuild every pipeline when the window first draws.
            #[cfg(feature = "windows-wgpu")]
            Backend::Dx12(renderer) => {
                if let Err(error) = renderer.draw(scene, appearance) {
                    self.fall_back_to_dx11(error);
                }
                Ok(())
            }
            #[cfg(feature = "windows-wgpu")]
            Backend::Unavailable { .. } => Ok(()),
        }
    }

    pub(crate) fn resize(&mut self, size: Size<DevicePixels>) -> Result<()> {
        #[cfg(feature = "windows-wgpu")]
        {
            self.fallback.size = Some(size);
        }
        match &mut self.backend {
            Backend::Dx11(renderer) => renderer.resize(size),
            #[cfg(feature = "windows-wgpu")]
            Backend::Dx12(renderer) => {
                renderer.resize(size);
                Ok(())
            }
            #[cfg(feature = "windows-wgpu")]
            Backend::Unavailable { .. } => Ok(()),
        }
    }

    pub(crate) fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        #[cfg(feature = "windows-wgpu")]
        if let Some(atlas) = &self.atlas {
            return atlas.clone();
        }
        match &self.backend {
            Backend::Dx11(renderer) => renderer.sprite_atlas(),
            #[cfg(feature = "windows-wgpu")]
            Backend::Dx12(renderer) => renderer.renderer.sprite_atlas().clone(),
            #[cfg(feature = "windows-wgpu")]
            Backend::Unavailable { atlas } => atlas.clone(),
        }
    }

    pub(crate) fn gpu_specs(&self) -> Result<GpuSpecs> {
        match &self.backend {
            Backend::Dx11(renderer) => renderer.gpu_specs(),
            #[cfg(feature = "windows-wgpu")]
            Backend::Dx12(renderer) => Ok(renderer.renderer.gpu_specs()),
            #[cfg(feature = "windows-wgpu")]
            Backend::Unavailable { .. } => bail!("this window's renderer is unavailable"),
        }
    }

    pub(crate) fn handle_device_lost(&mut self, devices: &DirectXDevices) -> Result<()> {
        match &mut self.backend {
            Backend::Dx11(renderer) => renderer.handle_device_lost(devices),
            // DX11 support-device recovery does not invalidate DX12 textures.
            #[cfg(feature = "windows-wgpu")]
            Backend::Dx12(_) | Backend::Unavailable { .. } => Ok(()),
        }
    }

    pub(crate) fn mark_drawable(&mut self) {
        match &mut self.backend {
            Backend::Dx11(renderer) => renderer.mark_drawable(),
            #[cfg(feature = "windows-wgpu")]
            Backend::Dx12(_) | Backend::Unavailable { .. } => {}
        }
    }

    pub(crate) fn take_force_redraw(&mut self) -> bool {
        match &mut self.backend {
            #[cfg(not(feature = "windows-wgpu"))]
            Backend::Dx11(_) => false,
            #[cfg(feature = "windows-wgpu")]
            Backend::Dx11(_) => std::mem::take(&mut self.force_redraw),
            #[cfg(feature = "windows-wgpu")]
            Backend::Dx12(renderer) => match renderer.take_force_redraw() {
                Ok(force_redraw) => force_redraw,
                Err(error) => {
                    self.fall_back_to_dx11(error);
                    std::mem::take(&mut self.force_redraw)
                }
            },
            #[cfg(feature = "windows-wgpu")]
            Backend::Unavailable { .. } => false,
        }
    }

    #[cfg(feature = "windows-wgpu")]
    pub(crate) fn gpu_context_info(&self) -> Option<gpui_wgpu::WgpuContextHandle> {
        match &self.backend {
            Backend::Dx12(renderer) => renderer.renderer.gpu_context_info(),
            Backend::Dx11(_) | Backend::Unavailable { .. } => None,
        }
    }

    #[cfg(feature = "windows-wgpu")]
    pub(crate) fn destroy(&mut self) {
        if let Backend::Dx12(renderer) = &mut self.backend {
            // Closing a hidden window must also save a confirmed failure that has not
            // reached the foreground frame callback yet.
            if let Some(error) = renderer.renderer.terminal_error() {
                renderer.context.failed(error, true);
            }
            renderer.renderer.destroy();
        }
    }

    /// Replaces a failed DX12 renderer with DX11 for the rest of this window's life, and
    /// makes later windows start on DX11. The application hears about the failure through
    /// [`WindowsRendererEvent::Failed`] and can persist the DX11 preference.
    #[cfg(feature = "windows-wgpu")]
    fn fall_back_to_dx11(&mut self, error: String) {
        let atlas = self.sprite_atlas();
        let Backend::Dx12(mut dx12) =
            std::mem::replace(&mut self.backend, Backend::Unavailable { atlas })
        else {
            return;
        };
        log::error!("DX12 rendering failed; switching this window to DX11: {error}");
        let context = dx12.context.clone();
        context.failed(error, true);
        context.use_dx11_for_new_windows();
        // DX11 creates its own swap chain and DirectComposition target for this window,
        // so DX12 must release both first.
        dx12.renderer.destroy();
        drop(dx12);
        match self.fallback.create() {
            Ok(mut renderer) => {
                if let Some(atlas) = &self.atlas {
                    atlas.replace(renderer.sprite_atlas());
                }
                renderer.mark_drawable();
                context.emit(WindowsRendererEvent::Selected {
                    backend: WindowsRendererBackend::Dx11,
                    overridden: context.0.overridden,
                });
                self.backend = Backend::Dx11(renderer);
                self.force_redraw = true;
            }
            Err(error) => {
                log::error!(
                    "DX11 could not replace DX12; this window can no longer draw: {error:#}"
                )
            }
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn render_to_image(
        &mut self,
        scene: &Scene,
        appearance: WindowBackgroundAppearance,
    ) -> Result<image::RgbaImage> {
        match &mut self.backend {
            Backend::Dx11(renderer) => renderer.render_to_image(scene, appearance),
            #[cfg(all(feature = "windows-wgpu", feature = "test-support"))]
            Backend::Dx12(renderer) => {
                anyhow::ensure!(
                    renderer.prepare_frame().map_err(anyhow::Error::msg)?,
                    "render_to_image unavailable while recovering from a lost device"
                );
                renderer.renderer.render_to_image(scene)
            }
            #[cfg(all(feature = "windows-wgpu", not(feature = "test-support")))]
            Backend::Dx12(_) => bail!("WGPU readback requires test-support"),
            #[cfg(feature = "windows-wgpu")]
            Backend::Unavailable { .. } => bail!("this window's renderer is unavailable"),
        }
    }
}

/// The atlas a DX12 window gives gpui. When the window falls back to DX11 it forwards to
/// the DX11 atlas, and its generation jumps past every earlier one, so gpui discards
/// frames built from DX12 tiles and rasterizes again.
#[cfg(feature = "windows-wgpu")]
pub(crate) struct SwitchableAtlas {
    current: parking_lot::RwLock<Arc<dyn PlatformAtlas>>,
    /// Generations already reported by replaced atlases.
    generation_base: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "windows-wgpu")]
impl SwitchableAtlas {
    fn new(atlas: Arc<dyn PlatformAtlas>) -> Arc<Self> {
        // PlatformWindow's atlas API requires Arc. Like the atlases it forwards to, this
        // one stays on the UI thread.
        #[allow(clippy::arc_with_non_send_sync)]
        Arc::new(Self {
            current: parking_lot::RwLock::new(atlas),
            generation_base: Default::default(),
        })
    }

    fn replace(&self, atlas: Arc<dyn PlatformAtlas>) {
        use std::sync::atomic::Ordering;
        let mut current = self.current.write();
        let base = self.generation_base.load(Ordering::Relaxed) + current.generation() + 1;
        self.generation_base.store(base, Ordering::Relaxed);
        *current = atlas;
    }
}

#[cfg(feature = "windows-wgpu")]
impl PlatformAtlas for SwitchableAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &gpui::AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, std::borrow::Cow<'a, [u8]>)>>,
    ) -> Result<Option<gpui::AtlasTile>> {
        self.current.read().get_or_insert_with(key, build)
    }

    fn get_or_insert_glyph_with(
        &self,
        params: &gpui::RenderGlyphParams,
        build: &mut dyn FnMut() -> Result<gpui::ValidatedRasterizedGlyph>,
    ) -> Result<gpui::GlyphAtlasEntry> {
        self.current.read().get_or_insert_glyph_with(params, build)
    }

    fn remove(&self, key: &gpui::AtlasKey) {
        self.current.read().remove(key)
    }

    fn generation(&self) -> u64 {
        let current = self.current.read();
        self.generation_base
            .load(std::sync::atomic::Ordering::Relaxed)
            + current.generation()
    }

    #[cfg(any(test, feature = "test-support"))]
    fn contains(&self, key: &gpui::AtlasKey) -> bool {
        self.current.read().contains(key)
    }
}

#[cfg(feature = "windows-wgpu")]
pub(crate) struct Dx12Renderer {
    renderer: gpui_wgpu::WgpuRenderer,
    startup_resize: StartupResize,
    window: crate::SafeHwnd,
    force_redraw: bool,
    context: RendererContext,
    // A different window recovering successfully must not grant this window unlimited retries.
    failed_window_recoveries: u8,
    warned_non_opaque: bool,
}

/// Native window creation can deliver several sizes before GPUI has a scene.
/// Keep the checked initial surface configuration, then configure only the
/// latest size when rendering starts. This state outlives GPU resource recovery.
#[cfg(feature = "windows-wgpu")]
#[derive(Default)]
struct StartupResize {
    started: bool,
    pending_size: Option<Size<DevicePixels>>,
}

#[cfg(feature = "windows-wgpu")]
impl StartupResize {
    fn resize(&mut self, size: Size<DevicePixels>, apply: impl FnOnce(Size<DevicePixels>)) {
        if self.started {
            apply(size);
        } else {
            self.pending_size = Some(size);
        }
    }

    /// Starts rendering, returning the size to configure first, if any.
    fn start(&mut self) -> Option<Size<DevicePixels>> {
        self.started = true;
        self.pending_size.take()
    }
}

#[cfg(feature = "windows-wgpu")]
impl Dx12Renderer {
    /// Draws a frame, or returns why DX12 can no longer render this window.
    fn draw(
        &mut self,
        scene: &Scene,
        appearance: WindowBackgroundAppearance,
    ) -> Result<(), String> {
        if !appearance.is_opaque() && !std::mem::replace(&mut self.warned_non_opaque, true) {
            log::warn!("DX12 windows are opaque; ignoring the {appearance:?} background");
        }
        if !self.prepare_frame()? {
            // Recovery runs before the next frame's layout/paint, where
            // rebuilding cached scenes can use the replacement atlas.
            return Ok(());
        }
        self.renderer.draw(scene);
        self.force_redraw |= self.renderer.needs_redraw();
        Ok(())
    }

    fn resize(&mut self, size: Size<DevicePixels>) {
        self.startup_resize
            .resize(size, |size| self.renderer.update_drawable_size(size));
    }

    /// Configures the latest startup size once the device can render: a lost device is
    /// never configured, and failed or pending recovery leaves the size queued.
    fn prepare_frame(&mut self) -> Result<bool, String> {
        if !self.check_device()? {
            return Ok(false);
        }
        match self.startup_resize.start() {
            Some(size) => {
                self.renderer.update_drawable_size(size);
                // Configuration can itself report a loss or a terminal error. The
                // target already stores the final size for a subsequent recovery.
                self.check_device()
            }
            None => Ok(true),
        }
    }

    /// `Ok(false)` while the device is lost, `Err` once DX12 can no longer render.
    fn check_device(&self) -> Result<bool, String> {
        if self.renderer.device_lost() {
            return Ok(false);
        }
        match self.renderer.terminal_error() {
            Some(error) => Err(error),
            None => Ok(true),
        }
    }

    fn take_force_redraw(&mut self) -> Result<bool, String> {
        if let Some(error) = self.renderer.terminal_error() {
            return Err(error);
        }
        if self.renderer.device_lost() {
            if let Err(error) = self.renderer.recover(&self.window) {
                if !error.is::<gpui_wgpu::RecoveryPending>()
                    && self
                        .context
                        .failed_recovery(&mut self.failed_window_recoveries, &error)
                {
                    return Err(format!(
                        "DX12 recovery failed after three attempts: {error:#}"
                    ));
                }
                return Ok(false);
            }
            self.context.0.failed_recoveries.set(0);
            self.failed_window_recoveries = 0;
            self.force_redraw = true;
        }
        Ok(std::mem::take(&mut self.force_redraw))
    }

    fn new(hwnd: HWND, context: RendererContext) -> Result<Self> {
        use crate::bindings::Windows::Win32::*;
        use gpui_wgpu::{
            FontRasterizationSettings, SubpixelOrder, WgpuRenderer, WgpuSurfaceConfig, wgpu,
        };
        use windows_core::Interface as _;
        let window = crate::SafeHwnd::from(hwnd);
        let mut renderer = WgpuRenderer::new_dx12(
            context.0.gpu.clone(),
            &window,
            WgpuSurfaceConfig {
                size: gpui::size(DevicePixels(1), DevicePixels(1)),
                transparent: false,
                preferred_present_mode: Some(wgpu::PresentMode::Immediate),
            },
            None,
            None,
        )?;
        let info = renderer
            .gpu_context_info()
            .expect("new renderer has context");
        if info.backend() != gpui_wgpu::WgpuBackend::Native(wgpu::Backend::Dx12)
            || info.adapter_info().device_type == wgpu::DeviceType::Cpu
        {
            bail!(
                "wgpu-dx12 requires a hardware DX12 adapter; got {:?}",
                info.adapter_info()
            );
        }
        // Match the native renderer's DirectWrite gamma and subpixel settings.
        let factory: IDWriteFactory5 = unsafe { DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)? };
        let params: IDWriteRenderingParams1 = unsafe { factory.CreateRenderingParams()? }.cast()?;
        renderer.set_font_rasterization_settings(FontRasterizationSettings::new(
            unsafe { params.GetGamma() },
            unsafe { params.GetGrayscaleEnhancedContrast() },
            unsafe { params.GetEnhancedContrast() },
            if unsafe { params.GetPixelGeometry() } == DWRITE_PIXEL_GEOMETRY_BGR {
                SubpixelOrder::BlueGreenRed
            } else {
                SubpixelOrder::RedGreenBlue
            },
        ));
        log::info!(
            "Windows renderer: WGPU DX12, adapter={:?}; DX11 devices deferred until needed",
            info.adapter_info()
        );
        Ok(Self {
            renderer,
            startup_resize: StartupResize::default(),
            window,
            force_redraw: false,
            context,
            failed_window_recoveries: 0,
            warned_non_opaque: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn context(
        preference: WindowsRendererPreference,
    ) -> (RendererContext, Rc<RefCell<Vec<WindowsRendererEvent>>>) {
        let events = Rc::new(RefCell::new(Vec::new()));
        let captured = events.clone();
        let context = RendererContext::with_override(
            WindowsRendererOptions {
                preference,
                on_event: Some(Rc::new(move |event| captured.borrow_mut().push(event))),
            },
            None,
        )
        .unwrap();
        (context, events)
    }

    #[cfg(feature = "windows-wgpu")]
    #[test]
    fn startup_resizes_configure_only_the_latest_size_then_resize_immediately() {
        let mut resize = StartupResize::default();
        let mut configured = Vec::new();
        let first = gpui::size(DevicePixels(800), DevicePixels(600));
        let final_size = gpui::size(DevicePixels(1200), DevicePixels(900));
        for size in [first, final_size, final_size] {
            resize.resize(size, |size| configured.push(size));
        }
        assert!(configured.is_empty());
        assert_eq!(resize.start(), Some(final_size));
        assert_eq!(resize.start(), None, "the startup size is configured once");

        // Once rendering starts, keep the existing immediate runtime semantics,
        // including requests whose equality/clamping is handled by WGPU.
        for size in [first, first, final_size] {
            resize.resize(size, |size| configured.push(size));
        }
        assert_eq!(configured, [first, first, final_size]);
    }

    #[cfg(feature = "windows-wgpu")]
    #[test]
    fn first_frame_without_a_resize_keeps_the_checked_initial_configuration() {
        let mut resize = StartupResize::default();
        assert_eq!(resize.start(), None);
        let requested = gpui::size(DevicePixels(320), DevicePixels(240));
        let mut configured = None;
        resize.resize(requested, |size| configured = Some(size));
        assert_eq!(configured, Some(requested));
    }

    #[cfg(feature = "windows-wgpu")]
    mod switchable_atlas {
        use super::*;
        use gpui::{AtlasKey, AtlasTile, GlyphAtlasEntry, RenderGlyphParams};
        use std::borrow::Cow;

        /// Records lookups and reports a settable generation.
        #[derive(Default)]
        struct FakeAtlas {
            generation: Cell<u64>,
            lookups: Cell<usize>,
        }

        impl PlatformAtlas for FakeAtlas {
            fn get_or_insert_with<'a>(
                &self,
                _key: &AtlasKey,
                _build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
            ) -> Result<Option<AtlasTile>> {
                self.lookups.set(self.lookups.get() + 1);
                Ok(None)
            }

            fn get_or_insert_glyph_with(
                &self,
                _params: &RenderGlyphParams,
                _build: &mut dyn FnMut() -> Result<gpui::ValidatedRasterizedGlyph>,
            ) -> Result<GlyphAtlasEntry> {
                bail!("unused")
            }

            fn remove(&self, _key: &AtlasKey) {}

            fn generation(&self) -> u64 {
                self.generation.get()
            }
        }

        fn key() -> AtlasKey {
            AtlasKey::Image(gpui::RenderImageParams {
                image_id: gpui::ImageId(1),
                frame_index: 0,
            })
        }

        #[test]
        #[allow(clippy::arc_with_non_send_sync)]
        fn forwards_to_the_current_atlas_and_invalidates_every_older_generation() {
            let dx12 = Arc::new(FakeAtlas::default());
            dx12.generation.set(41);
            let atlas = SwitchableAtlas::new(dx12.clone());
            atlas.get_or_insert_with(&key(), &mut || Ok(None)).unwrap();
            assert_eq!(dx12.lookups.get(), 1);
            assert_eq!(atlas.generation(), 41);

            // A fresh DX11 atlas starts at generation 0, below what gpui has seen.
            let dx11 = Arc::new(FakeAtlas::default());
            atlas.replace(dx11.clone());
            let switched = atlas.generation();
            assert!(
                switched > 41,
                "gpui must discard frames built from DX12 tiles"
            );
            atlas.get_or_insert_with(&key(), &mut || Ok(None)).unwrap();
            assert_eq!((dx12.lookups.get(), dx11.lookups.get()), (1, 1));

            dx11.generation.set(3);
            assert_eq!(atlas.generation(), switched + 3);
            atlas.replace(Arc::new(FakeAtlas::default()));
            assert!(atlas.generation() > switched + 3);
        }
    }

    #[cfg(all(feature = "windows-wgpu", feature = "test-support"))]
    struct HiddenDx12Window(HWND);

    #[cfg(all(feature = "windows-wgpu", feature = "test-support"))]
    impl HiddenDx12Window {
        fn new() -> Result<Self> {
            use crate::bindings::Windows::Win32::{CreateWindowExW, WS_OVERLAPPED};
            Ok(Self(
                unsafe {
                    CreateWindowExW(
                        0,
                        windows_core::w!("STATIC"),
                        windows_core::w!("gpui dx12 renderer test"),
                        WS_OVERLAPPED as u32,
                        0,
                        0,
                        200,
                        100,
                        None,
                        None,
                        None,
                        None,
                    )
                }
                .ok()?,
            ))
        }

        /// A DX12 window renderer with the same DX11 fallback `WindowRenderer::new` uses.
        fn renderer(&self, context: &RendererContext) -> Result<WindowRenderer> {
            let renderer = Dx12Renderer::new(self.0, context.clone())?;
            Ok(WindowRenderer {
                atlas: Some(SwitchableAtlas::new(
                    renderer.renderer.sprite_atlas().clone(),
                )),
                backend: Backend::Dx12(Box::new(renderer)),
                fallback: Dx11Fallback {
                    hwnd: self.0,
                    devices: Default::default(),
                    disable_composition: false,
                    size: None,
                },
                force_redraw: false,
            })
        }

        fn destroy(&mut self) {
            if !self.0.is_invalid() {
                let _ = unsafe { crate::bindings::Windows::Win32::DestroyWindow(self.0) }.ok();
                self.0 = HWND::default();
            }
        }
    }

    #[cfg(all(feature = "windows-wgpu", feature = "test-support"))]
    impl Drop for HiddenDx12Window {
        fn drop(&mut self) {
            self.destroy();
        }
    }

    #[cfg(all(feature = "windows-wgpu", feature = "test-support"))]
    fn dx12(renderer: &mut WindowRenderer) -> &mut Dx12Renderer {
        match &mut renderer.backend {
            Backend::Dx12(renderer) => renderer,
            _ => panic!("expected a DX12 renderer"),
        }
    }

    #[cfg(all(feature = "windows-wgpu", feature = "test-support"))]
    #[test]
    fn native_dx12_first_draw_and_readback_use_latest_startup_size() -> Result<()> {
        if !dx12_supported()? {
            eprintln!("Skipping native DX12 resize test: no hardware DX12 adapter");
            return Ok(());
        }
        let (context, _) = context(WindowsRendererPreference::Auto);
        for first_use_is_draw in [true, false] {
            let window = HiddenDx12Window::new()?;
            let mut renderer = window.renderer(&context)?;
            let mut scene = Scene::default();
            scene.finish();
            renderer.resize(gpui::size(DevicePixels(48), DevicePixels(32)))?;
            renderer.resize(gpui::size(DevicePixels(96), DevicePixels(64)))?;
            // Bypass only the wrapper to inspect the unchanged checked target.
            // Intermediate WM_SIZE requests must not configure the actual WGPU target.
            let inner = dx12(&mut renderer);
            assert_eq!(inner.renderer.render_to_image(&scene)?.dimensions(), (1, 1));
            assert!(!inner.startup_resize.started);

            if first_use_is_draw {
                renderer.draw(&scene, WindowBackgroundAppearance::Opaque)?;
            }
            assert_eq!(
                renderer
                    .render_to_image(&scene, WindowBackgroundAppearance::Opaque)?
                    .dimensions(),
                (96, 64)
            );
            renderer.resize(gpui::size(DevicePixels(80), DevicePixels(40)))?;
            let inner = dx12(&mut renderer);
            assert!(inner.startup_resize.started);
            // Read the inner target directly: runtime resize must already have applied.
            assert_eq!(
                inner.renderer.render_to_image(&scene)?.dimensions(),
                (80, 40)
            );
        }
        Ok(())
    }

    #[cfg(all(feature = "windows-wgpu", feature = "test-support"))]
    #[test]
    fn native_dx12_startup_resize_survives_pending_recovery_and_resource_replacement() -> Result<()>
    {
        if !dx12_supported()? {
            eprintln!("Skipping native DX12 recovery test: no hardware DX12 adapter");
            return Ok(());
        }
        let (context, _) = context(WindowsRendererPreference::Auto);
        let window = HiddenDx12Window::new()?;
        let mut renderer = window.renderer(&context)?;
        renderer.resize(gpui::size(DevicePixels(48), DevicePixels(32)))?;
        let inner = dx12(&mut renderer);
        // An absent shared context exercises the real stabilization wait without
        // destroying a device or depending on a hardware driver reset.
        let shared_context = context.0.gpu.borrow_mut().take();
        let recovery = inner.renderer.recover(&inner.window);
        *context.0.gpu.borrow_mut() = shared_context;
        assert!(recovery.unwrap_err().is::<gpui_wgpu::RecoveryPending>());
        assert!(!inner.startup_resize.started);
        assert_eq!(
            inner.startup_resize.pending_size,
            Some(gpui::size(DevicePixels(48), DevicePixels(32)))
        );
        renderer.resize(gpui::size(DevicePixels(112), DevicePixels(72)))?;
        let inner = dx12(&mut renderer);
        inner.renderer.recover(&inner.window)?;
        assert!(!inner.startup_resize.started);
        let mut scene = Scene::default();
        scene.finish();
        assert_eq!(
            renderer
                .render_to_image(&scene, WindowBackgroundAppearance::Opaque)?
                .dimensions(),
            (112, 72)
        );
        Ok(())
    }

    #[cfg(all(feature = "windows-wgpu", feature = "test-support"))]
    fn white_quad_scene() -> Scene {
        let bounds = gpui::Bounds::new(
            gpui::point(gpui::ScaledPixels(8.0), gpui::ScaledPixels(8.0)),
            gpui::size(gpui::ScaledPixels(32.0), gpui::ScaledPixels(24.0)),
        );
        let mut scene = Scene::default();
        scene.insert_primitive(gpui::Quad {
            bounds,
            content_mask: gpui::ContentMask {
                bounds,
                ..Default::default()
            },
            background: gpui::white().into(),
            ..Default::default()
        });
        scene.finish();
        scene
    }

    /// A DX12 failure that used to panic now replaces the window's renderer with DX11
    /// in place: same atlas object for gpui, a newer atlas generation, one forced frame,
    /// the application told, and later windows starting on DX11.
    #[cfg(all(feature = "windows-wgpu", feature = "test-support"))]
    #[test]
    fn native_dx12_terminal_errors_switch_the_window_to_dx11_without_panicking() -> Result<()> {
        if !dx12_supported()? {
            eprintln!("Skipping native DX12 fallback test: no hardware DX12 adapter");
            return Ok(());
        }
        for failure_seen_by_draw in [true, false] {
            let (context, events) = context(WindowsRendererPreference::Auto);
            let window = HiddenDx12Window::new()?;
            let mut renderer = window.renderer(&context)?;
            renderer.resize(gpui::size(DevicePixels(64), DevicePixels(48)))?;
            renderer.draw(&white_quad_scene(), WindowBackgroundAppearance::Opaque)?;
            let atlas = renderer.sprite_atlas();
            let generation = atlas.generation();

            dx12(&mut renderer)
                .renderer
                .inject_terminal_error_for_tests("injected DX12 failure");
            if failure_seen_by_draw {
                renderer.draw(&white_quad_scene(), WindowBackgroundAppearance::Opaque)?;
                assert!(
                    renderer.take_force_redraw(),
                    "the switch forces one full frame"
                );
            } else {
                assert!(
                    renderer.take_force_redraw(),
                    "the switch forces one full frame"
                );
            }
            assert!(!renderer.take_force_redraw());

            assert!(matches!(renderer.backend, Backend::Dx11(_)));
            assert!(Arc::ptr_eq(&atlas, &renderer.sprite_atlas()));
            assert!(renderer.sprite_atlas().generation() > generation);
            assert_eq!(
                context.0.backend.get(),
                Some(WindowsRendererBackend::Dx11),
                "later windows must start on DX11"
            );
            let events = events.borrow();
            assert!(events.iter().any(|event| matches!(
                event,
                WindowsRendererEvent::Failed { message, runtime: true } if message.contains("injected DX12 failure")
            )));
            assert!(matches!(
                events.last(),
                Some(WindowsRendererEvent::Selected {
                    backend: WindowsRendererBackend::Dx11,
                    ..
                })
            ));
            drop(events);

            // The replacement draws at the window's size.
            renderer.draw(&white_quad_scene(), WindowBackgroundAppearance::Opaque)?;
            let image = renderer
                .render_to_image(&white_quad_scene(), WindowBackgroundAppearance::Opaque)?;
            assert_eq!(image.dimensions(), (64, 48));
            assert_eq!(image.get_pixel(20, 16).0, [255; 4]);
            assert!(renderer.gpu_specs().is_ok());
        }
        Ok(())
    }

    /// If DX11 cannot replace DX12 either, the window stops drawing but nothing panics.
    #[cfg(all(feature = "windows-wgpu", feature = "test-support"))]
    #[test]
    fn native_failed_dx11_fallback_leaves_an_inert_window_without_panicking() -> Result<()> {
        if !dx12_supported()? {
            eprintln!("Skipping native DX12 fallback test: no hardware DX12 adapter");
            return Ok(());
        }
        let (context, _) = context(WindowsRendererPreference::Auto);
        let mut window = HiddenDx12Window::new()?;
        let mut renderer = window.renderer(&context)?;
        let atlas = renderer.sprite_atlas();
        dx12(&mut renderer)
            .renderer
            .inject_terminal_error_for_tests("injected DX12 failure");
        // DX11 cannot create a swap chain for a destroyed window.
        window.destroy();
        renderer.draw(&white_quad_scene(), WindowBackgroundAppearance::Opaque)?;
        assert!(matches!(renderer.backend, Backend::Unavailable { .. }));
        renderer.draw(&white_quad_scene(), WindowBackgroundAppearance::Opaque)?;
        renderer.resize(gpui::size(DevicePixels(32), DevicePixels(32)))?;
        assert!(!renderer.take_force_redraw());
        assert!(renderer.gpu_specs().is_err());
        assert!(Arc::ptr_eq(&atlas, &renderer.sprite_atlas()));
        assert!(
            renderer
                .render_to_image(&white_quad_scene(), WindowBackgroundAppearance::Opaque)
                .is_err()
        );
        Ok(())
    }

    /// DX12 windows stay opaque: a transparent background must neither reconfigure the
    /// surface nor rebuild pipelines.
    #[cfg(all(feature = "windows-wgpu", feature = "test-support"))]
    #[test]
    fn native_dx12_windows_ignore_non_opaque_backgrounds() -> Result<()> {
        if !dx12_supported()? {
            eprintln!("Skipping native DX12 opacity test: no hardware DX12 adapter");
            return Ok(());
        }
        let (context, _) = context(WindowsRendererPreference::Auto);
        let window = HiddenDx12Window::new()?;
        let mut renderer = window.renderer(&context)?;
        renderer.resize(gpui::size(DevicePixels(64), DevicePixels(48)))?;
        renderer.draw(&white_quad_scene(), WindowBackgroundAppearance::Opaque)?;
        let opaque =
            renderer.render_to_image(&white_quad_scene(), WindowBackgroundAppearance::Opaque)?;
        for appearance in [
            WindowBackgroundAppearance::Transparent,
            WindowBackgroundAppearance::Blurred,
        ] {
            renderer.draw(&white_quad_scene(), appearance)?;
            assert!(matches!(renderer.backend, Backend::Dx12(_)));
            assert_eq!(
                renderer.render_to_image(&white_quad_scene(), appearance)?,
                opaque,
                "{appearance:?} must render exactly like an opaque window"
            );
        }
        Ok(())
    }

    #[test]
    fn dx12_windows_never_initialize_dx11_devices() {
        let (context, _) = context(WindowsRendererPreference::Auto);
        let devices = LazyDirectXDevices::<u8>::default();
        for _ in 0..2 {
            assert_eq!(
                context
                    .create_with_devices(
                        &devices,
                        || panic!("DX11 device creation on a DX12 path"),
                        || Ok(true),
                        || Ok(12),
                        |_| panic!("DX11 renderer on a DX12 path"),
                    )
                    .unwrap(),
                12
            );
            assert!(devices.get().is_none());
        }
    }

    #[cfg(feature = "windows-wgpu")]
    #[test]
    fn forced_dx12_failure_does_not_initialize_dx11_devices() {
        let context =
            RendererContext::with_override(WindowsRendererOptions::default(), Some("wgpu-dx12"))
                .unwrap();
        let devices = LazyDirectXDevices::<u8>::default();
        assert!(
            context
                .create_with_devices(
                    &devices,
                    || panic!("forced DX12 must not create fallback devices"),
                    || panic!("forced DX12 must not probe"),
                    || Err::<(), _>(anyhow::anyhow!("DX12 initialization failed")),
                    |_| panic!("forced DX12 must not fall back"),
                )
                .is_err()
        );
        assert!(devices.get().is_none());
    }

    #[test]
    fn fallback_device_creation_is_retryable_and_shared_across_windows() {
        let (context, _) = context(WindowsRendererPreference::Auto);
        let devices = LazyDirectXDevices::<u8>::default();
        let attempts = Cell::new(0);
        let create_devices = || {
            attempts.set(attempts.get() + 1);
            if attempts.get() == 1 {
                bail!("DX11 device initialization failed");
            }
            Ok(11)
        };
        assert!(
            context
                .create_with_devices(
                    &devices,
                    create_devices,
                    || Ok(true),
                    || Err::<u8, _>(anyhow::anyhow!("DX12 surface failed")),
                    |device| Ok(*device),
                )
                .is_err()
        );
        assert!(devices.get().is_none());
        assert_eq!(attempts.get(), 1);

        // A failed factory is not cached. Once it succeeds, every subsequent
        // window uses the shared device without probing or creating another.
        for _ in 0..2 {
            assert_eq!(
                context
                    .create_with_devices(
                        &devices,
                        create_devices,
                        || panic!("fallback choice should be shared"),
                        || panic!("fallback must remain DX11"),
                        |device| Ok(*device),
                    )
                    .unwrap(),
                11
            );
        }
        assert_eq!(attempts.get(), 2);
    }

    #[test]
    fn native_dx11_device_is_created_on_demand_and_reused() -> Result<()> {
        use windows_core::Interface;

        // Exercise both explicit DX11 and automatic fallback with real devices.
        for preference in [
            WindowsRendererPreference::Dx11,
            WindowsRendererPreference::Auto,
        ] {
            let (context, _) = context(preference);
            let devices = LazyDirectXDevices::default();
            let attempts = Cell::new(0);
            let mut first = None;
            for _ in 0..2 {
                let device = context.create_with_devices(
                    &devices,
                    || {
                        attempts.set(attempts.get() + 1);
                        DirectXDevices::with_debug_layer(false)
                    },
                    || Ok(false),
                    || panic!("DX12 should not be selected"),
                    |devices| {
                        unsafe { devices.device.GetDeviceRemovedReason().ok()? };
                        Ok(devices.device.clone())
                    },
                )?;
                if let Some(first) = first {
                    assert_eq!(device.as_raw(), first);
                } else {
                    first = Some(device.as_raw());
                }
            }
            assert_eq!(attempts.get(), 1);
        }
        Ok(())
    }

    #[test]
    fn saved_dx11_skips_every_dx12_step() {
        let (context, _) = context(WindowsRendererPreference::Dx11);
        assert_eq!(
            context
                .create_with(
                    || panic!("probe"),
                    || panic!("DX12 initialization"),
                    || Ok(11)
                )
                .unwrap(),
            11
        );
    }

    #[test]
    fn strict_dx11_override_skips_dx12_and_never_reports_persistable_failures() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let captured = events.clone();
        let context = RendererContext::with_override(
            WindowsRendererOptions {
                preference: WindowsRendererPreference::Auto,
                on_event: Some(Rc::new(move |event| captured.borrow_mut().push(event))),
            },
            Some("dx11"),
        )
        .unwrap();
        assert_eq!(
            context
                .create_with(|| panic!("DX12 probe"), || panic!("DX12 device"), || Ok(11),)
                .unwrap(),
            11
        );
        context.failed("injected error".into(), true);
        assert!(matches!(
            events.borrow().as_slice(),
            [WindowsRendererEvent::Selected {
                backend: WindowsRendererBackend::Dx11,
                overridden: true,
            }]
        ));
    }

    #[test]
    fn invalid_overrides_fail_without_using_automatic_policy() {
        assert!(
            RendererContext::with_override(WindowsRendererOptions::default(), Some("vulkan"))
                .is_err()
        );
    }

    #[cfg(feature = "windows-wgpu")]
    #[test]
    fn native_support_probe_does_not_construct_a_wgpu_context() {
        let (context, _) = context(WindowsRendererPreference::Auto);
        dx12_supported().unwrap();
        assert!(context.0.gpu.borrow().is_none());
    }
    #[test]
    fn unsupported_falls_back_once_and_shares_choice_across_windows() {
        let (context, events) = context(WindowsRendererPreference::Auto);
        assert_eq!(
            context
                .create_with(|| Ok(false), || panic!("unsupported DX12"), || Ok(11))
                .unwrap(),
            11
        );
        assert_eq!(
            context
                .create_with(
                    || panic!("repeated probe"),
                    || panic!("repeated DX12"),
                    || Ok(11)
                )
                .unwrap(),
            11
        );
        assert_eq!(
            events
                .borrow()
                .iter()
                .filter(|event| matches!(event, WindowsRendererEvent::Failed { .. }))
                .count(),
            1
        );
    }
    #[test]
    fn initialization_failures_notify_before_dx11_including_when_both_fail() {
        for stage in ["device", "surface", "pipeline"] {
            let (context, events) = context(WindowsRendererPreference::Auto);
            let result: Result<()> = context.create_with(
                || Ok(true),
                || bail!("{stage} failure"),
                || {
                    assert!(matches!(
                        &events.borrow()[0],
                        WindowsRendererEvent::Failed { runtime: false, .. }
                    ));
                    bail!("DX11 failure")
                },
            );
            assert!(result.is_err());
            assert_eq!(context.0.backend.get(), Some(WindowsRendererBackend::Dx11));
        }
    }
    #[test]
    fn dx12_success_reuses_selection_and_runtime_failure_is_notified_once() {
        let (context, events) = context(WindowsRendererPreference::Auto);
        assert_eq!(
            context
                .create_with(|| Ok(true), || Ok(12), || panic!("DX11"))
                .unwrap(),
            12
        );
        assert_eq!(
            context
                .create_with(|| panic!("second probe"), || Ok(12), || panic!("DX11"))
                .unwrap(),
            12
        );
        context.failed("terminal render error".into(), true);
        context.failed("duplicate".into(), true);
        assert_eq!(
            events
                .borrow()
                .iter()
                .filter(|event| matches!(event, WindowsRendererEvent::Failed { runtime: true, .. }))
                .count(),
            1
        );
    }
    #[cfg(feature = "windows-wgpu")]
    #[test]
    fn forced_dx12_has_no_fallback_or_persistence() {
        let events = Rc::new(Cell::new(0));
        let captured = events.clone();
        let context = RendererContext::with_override(
            WindowsRendererOptions {
                preference: WindowsRendererPreference::Auto,
                on_event: Some(Rc::new(move |_| captured.set(captured.get() + 1))),
            },
            Some("wgpu-dx12"),
        )
        .unwrap();
        assert!(
            context
                .create_with(
                    || panic!("probe"),
                    || Err::<u8, _>(anyhow::anyhow!("DX12 failed")),
                    || panic!("DX11")
                )
                .is_err()
        );
        assert_eq!(events.get(), 0);
    }

    #[cfg(feature = "windows-wgpu")]
    #[test]
    fn recovery_is_bounded_across_windows_and_after_another_window_recovers() {
        let error = anyhow::anyhow!("recreation failed");
        let (context, events) = context(WindowsRendererPreference::Auto);
        let mut windows = [0; 4];
        for (index, failures) in windows.iter_mut().take(3).enumerate() {
            assert_eq!(context.failed_recovery(failures, &error), index == 2);
        }
        assert!(matches!(
            events.borrow().last(),
            Some(WindowsRendererEvent::Failed { runtime: true, .. })
        ));

        let (context, events) = self::context(WindowsRendererPreference::Auto);
        let mut broken_window = 0;
        for attempt in 1..=3 {
            assert_eq!(
                context.failed_recovery(&mut broken_window, &error),
                attempt == 3
            );
            // Another window successfully recovers the shared device between attempts.
            context.0.failed_recoveries.set(0);
        }
        assert_eq!(events.borrow().len(), 1);
    }
}
