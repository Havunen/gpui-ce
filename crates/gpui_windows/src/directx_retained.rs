//! Opt-in retained scene target. Retired allocations stay charged until a GPU fence completes.
use crate::bindings::Windows::Win32::*;
use gpui::profiler::gpu;
use gpui::{PlatformAtlas, Scene, WindowBackgroundAppearance};
use gpui_render::damage::{Damage, SceneSnapshot};
use gpui_render::retained::{GpuMemoryReservation, RetainedScene, reserve_retained_bytes};
use std::{cell::RefCell, ffi::c_void};
use windows_core::Interface;

struct Target {
    _texture: ID3D11Texture2D,
    rtv: Option<ID3D11RenderTargetView>,
    srv: Option<ID3D11ShaderResourceView>,
    width: u32,
    height: u32,
    appearance: WindowBackgroundAppearance,
    // Held for its drop, which releases the device budget.
    _reservation: GpuMemoryReservation,
}
struct Retired {
    device_key: usize,
    context: ID3D11DeviceContext,
    fence: Option<ID3D11Asynchronous>,
    _target: Target,
}
thread_local! {static RETIRED:RefCell<Vec<Retired>>=const {RefCell::new(Vec::new())};}
pub(super) fn poll_retired() {
    RETIRED.with(|pending| {
        pending.borrow_mut().retain(|item| {
            // A query failure keeps its reservation charged until device recovery.
            !item
                .fence
                .as_ref()
                .is_some_and(|fence| fence_passed(&item.context, fence))
        })
    });
}
fn retire(target: Target, device: &ID3D11Device, context: &ID3D11DeviceContext) {
    let fence = event_query(device).ok();
    if let Some(fence) = &fence {
        end_query(context, fence);
    }
    RETIRED.with(|pending| {
        pending.borrow_mut().push(Retired {
            device_key: device.as_raw() as usize,
            context: context.clone(),
            fence,
            _target: target,
        })
    });
}
// Generated bindings omit these methods. The stable COM ABI uses device slot 24
// for CreateQuery and immediate-context slot 28 for End. ID3D11Query derives
// from ID3D11Asynchronous, so its interface pointer is a valid upcast.
fn event_query(device: &ID3D11Device) -> anyhow::Result<ID3D11Asynchronous> {
    #[repr(C)]
    struct QueryDesc {
        query: i32,
        misc_flags: u32,
    }
    type Create = unsafe extern "system" fn(
        *mut c_void,
        *const QueryDesc,
        *mut *mut c_void,
    ) -> windows_core::HRESULT;
    let mut output = std::ptr::null_mut();
    unsafe {
        let table = Interface::vtable(device) as *const _ as *const *const c_void;
        let create: Create = std::mem::transmute(*table.add(24));
        // D3D11_QUERY_EVENT
        let desc = QueryDesc {
            query: 0,
            misc_flags: 0,
        };
        create(device.as_raw(), &desc, &mut output).ok()?;
        anyhow::ensure!(!output.is_null(), "CreateQuery returned a null interface");
        Ok(ID3D11Asynchronous::from_raw(output))
    }
}
fn end_query(context: &ID3D11DeviceContext, query: &ID3D11Asynchronous) {
    type End = unsafe extern "system" fn(*mut c_void, *mut c_void);
    unsafe {
        let table = Interface::vtable(context) as *const _ as *const *const c_void;
        let end: End = std::mem::transmute(*table.add(28));
        end(context.as_raw(), query.as_raw());
    }
}
/// Whether the GPU has passed `fence`, polled without flushing the context.
fn fence_passed(context: &ID3D11DeviceContext, fence: &ID3D11Asynchronous) -> bool {
    let mut passed = 0i32;
    let result = unsafe {
        context.GetData(
            fence,
            Some(&mut passed as *mut i32 as *mut c_void),
            std::mem::size_of::<i32>() as u32,
            1, // D3D11_ASYNC_GETDATA_DONOTFLUSH
        )
    };
    result.0 == 0 && passed != 0
}
pub(super) struct RetainedFrame {
    enabled: bool,
    target: Option<Target>,
    scene: RetainedScene,
    devices: Option<(ID3D11Device, ID3D11DeviceContext)>,
    #[cfg(test)]
    last_damage: &'static str,
}
pub(super) struct RetainedPlan {
    pub rtv: Option<ID3D11RenderTargetView>,
    pub srv: Option<ID3D11ShaderResourceView>,
    pub damage: Damage,
    pub snapshot: Option<SceneSnapshot>,
    revision: u64,
    atlas: u64,
}
impl RetainedFrame {
    pub fn new() -> Self {
        Self {
            enabled: gpu::experiment_enabled("partial-redraw"),
            target: None,
            scene: RetainedScene::default(),
            devices: None,
            #[cfg(test)]
            last_damage: "full",
        }
    }
    pub fn enabled(&self) -> bool {
        self.enabled
    }
    #[cfg(test)]
    pub fn enable_for_tests(&mut self) {
        self.enabled = true;
    }
    #[cfg(test)]
    pub fn disable_for_tests(&mut self) {
        self.enabled = false;
        self.invalidate();
    }
    #[cfg(test)]
    pub fn last_damage(&self) -> &'static str {
        self.last_damage
    }
    pub fn invalidate(&mut self) {
        self.scene.forget();
        if let Some(target) = self.target.take() {
            if let Some((device, context)) = &self.devices {
                retire(target, device, context);
            }
        }
    }
    pub fn device_lost(&mut self) {
        self.target = None;
        self.scene.forget();
        if let Some((device, _)) = &self.devices {
            let key = device.as_raw() as usize;
            RETIRED.with(|pending| pending.borrow_mut().retain(|item| item.device_key != key));
        }
        self.devices = None;
    }
    pub fn prepare(
        &mut self,
        scene: &Scene,
        atlas: &dyn PlatformAtlas,
        width: u32,
        height: u32,
        appearance: WindowBackgroundAppearance,
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
    ) -> anyhow::Result<RetainedPlan> {
        let frame = gpu::gpu_frame_context();
        let generation = atlas.generation();
        let full = || RetainedPlan {
            rtv: None,
            srv: None,
            damage: Damage::Full,
            snapshot: None,
            revision: frame.scene_revision,
            atlas: generation,
        };
        if !self.enabled {
            return Ok(full());
        }
        poll_retired();
        if !frame.active {
            self.invalidate();
            return Ok(full());
        }
        self.devices = Some((device.clone(), context.clone()));
        if self.target.as_ref().is_some_and(|target| {
            target.width != width || target.height != height || target.appearance != appearance
        }) {
            self.invalidate();
        }
        let Some((mut damage, snapshot)) =
            self.scene
                .plan(scene, frame.scene_revision, generation, width, height)
        else {
            self.invalidate();
            return Ok(full());
        };
        if self.target.is_none() {
            let bytes = width as u64 * height as u64 * 4;
            let Some(reservation) =
                reserve_retained_bytes("d3d11", device.as_raw() as usize, bytes)
            else {
                self.invalidate();
                return Ok(full());
            };
            let desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: crate::directx_renderer::RENDER_TARGET_FORMAT,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: (D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE) as u32,
                ..Default::default()
            };
            let mut texture = None;
            let mut rtv = None;
            let mut srv = None;
            unsafe {
                device
                    .CreateTexture2D(&desc, None, Some(&mut texture))
                    .ok()?;
                let texture = texture.as_ref().unwrap();
                device
                    .CreateRenderTargetView(texture, None, Some(&mut rtv))
                    .ok()?;
                device
                    .CreateShaderResourceView(texture, None, Some(&mut srv))
                    .ok()?;
            }
            self.target = Some(Target {
                _texture: texture.unwrap(),
                rtv,
                srv,
                width,
                height,
                appearance,
                _reservation: reservation,
            });
            // A new target holds nothing yet.
            damage = Damage::Full;
        }
        let target = self.target.as_ref().unwrap();
        Ok(RetainedPlan {
            rtv: target.rtv.clone(),
            srv: target.srv.clone(),
            damage,
            snapshot,
            revision: frame.scene_revision,
            atlas: generation,
        })
    }
    pub fn complete(&mut self, plan: RetainedPlan, success: bool) {
        #[cfg(test)]
        {
            self.last_damage = plan.damage.kind();
        }
        if success {
            self.scene
                .complete(plan.snapshot, plan.revision, plan.atlas);
        } else {
            self.invalidate();
        }
    }
}
impl Drop for RetainedFrame {
    fn drop(&mut self) {
        self.invalidate();
        poll_retired();
    }
}
