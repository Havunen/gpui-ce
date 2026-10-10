//! Opt-in retained scene target. Retired allocations stay charged until a GPU fence completes.
use crate::bindings::Windows::Win32::*;
use crate::directx_gpu_timing::{end_query, query, ready};
use gpui::profiler::gpu::{self, GpuMemoryReservation};
use gpui::{PlatformAtlas, Scene, WindowBackgroundAppearance};
use gpui_render::damage::{Damage, SceneSnapshot};
use std::cell::RefCell;
use windows_core::Interface;

struct Target {
    _texture: ID3D11Texture2D,
    rtv: Option<ID3D11RenderTargetView>,
    srv: Option<ID3D11ShaderResourceView>,
    width: u32,
    height: u32,
    appearance: WindowBackgroundAppearance,
    reservation: GpuMemoryReservation,
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
            !item.fence.as_ref().is_some_and(
                |fence| matches!(ready::<i32>(&item.context,fence),Ok(Some(value)) if value!=0),
            )
        })
    });
}
fn retire(target: Target, device: &ID3D11Device, context: &ID3D11DeviceContext) {
    let fence = query(device, 0).ok();
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
pub(super) struct RetainedFrame {
    enabled: bool,
    target: Option<Target>,
    snapshot: Option<SceneSnapshot>,
    scene_revision: Option<u64>,
    atlas_generation: u64,
    devices: Option<(ID3D11Device, ID3D11DeviceContext)>,
    #[cfg(test)]
    last_damage: &'static str,
}
pub(super) struct RetainedPlan {
    pub rtv: Option<ID3D11RenderTargetView>,
    pub srv: Option<ID3D11ShaderResourceView>,
    pub damage: Damage,
    pub snapshot: Option<SceneSnapshot>,
    pub bytes: u64,
    pub fallback_reason: Option<&'static str>,
    pub diff_detail: Option<String>,
    revision: u64,
    atlas: u64,
    reuse_snapshot: bool,
}
impl RetainedFrame {
    pub fn new() -> Self {
        Self {
            enabled: gpu::experiment_enabled("partial-redraw"),
            target: None,
            snapshot: None,
            scene_revision: None,
            atlas_generation: 0,
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
        self.snapshot = None;
        self.scene_revision = None;
        if let Some(target) = self.target.take() {
            if let Some((device, context)) = &self.devices {
                retire(target, device, context);
            }
        }
    }
    pub fn device_lost(&mut self) {
        self.target = None;
        self.snapshot = None;
        self.scene_revision = None;
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
        eligible: bool,
        appearance: WindowBackgroundAppearance,
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
    ) -> anyhow::Result<RetainedPlan> {
        if !self.enabled {
            return Ok(RetainedPlan {
                rtv: None,
                srv: None,
                damage: Damage::Full,
                snapshot: None,
                bytes: 0,
                revision: 0,
                atlas: 0,
                reuse_snapshot: false,
                fallback_reason: None,
                diff_detail: None,
            });
        }
        let frame = gpu::gpu_frame_context();
        let generation = atlas.generation();
        let full = |reason| RetainedPlan {
            rtv: None,
            srv: None,
            damage: Damage::Full,
            snapshot: None,
            bytes: 0,
            revision: frame.scene_revision,
            atlas: generation,
            reuse_snapshot: false,
            fallback_reason: Some(reason),
            diff_detail: None,
        };
        poll_retired();
        if !eligible || !frame.active {
            self.invalidate();
            return Ok(full(if !eligible {
                "diagnostic-ablation"
            } else {
                "inactive-window"
            }));
        }
        self.devices = Some((device.clone(), context.clone()));
        let mut target_change = None;
        if self.target.as_ref().is_some_and(|target| {
            target.width != width || target.height != height || target.appearance != appearance
        }) {
            target_change = Some("target-configuration-change");
            self.invalidate();
        }
        if self.scene_revision == Some(frame.scene_revision)
            && self.atlas_generation == generation
            && self.snapshot.is_some()
        {
            if let Some(target) = &self.target {
                return Ok(RetainedPlan {
                    rtv: target.rtv.clone(),
                    srv: target.srv.clone(),
                    damage: Damage::Reuse,
                    snapshot: None,
                    bytes: target.reservation.bytes,
                    revision: frame.scene_revision,
                    atlas: generation,
                    reuse_snapshot: true,
                    fallback_reason: None,
                    diff_detail: None,
                });
            }
        }
        let snapshot = match SceneSnapshot::capture_checked(scene, generation, width, height) {
            Ok(snapshot) => snapshot,
            Err(reason) => {
                self.invalidate();
                return Ok(full(reason.as_str()));
            }
        };
        if self.target.is_none() {
            let bytes = width as u64 * height as u64 * 4;
            let Some(reservation) =
                gpu::reserve_retained_bytes("d3d11", device.as_raw() as usize, bytes)
            else {
                return Ok(full("retained-device-budget"));
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
                reservation,
            });
        }
        let (damage, comparison_reason, diff_detail) = match self.snapshot.as_ref() {
            Some(previous) => {
                let (comparison, detail) = if gpu::timings_enabled() {
                    snapshot.compare_checked_diagnostic(previous)
                } else {
                    (snapshot.compare_checked(previous), None)
                };
                match comparison {
                    Ok(damage) => (damage, None, detail),
                    Err(reason) => (Damage::Full, Some(reason), detail),
                }
            }
            None => (Damage::Full, None, None),
        };
        let target = self.target.as_ref().unwrap();
        let fallback_reason = if damage == Damage::Full {
            Some(
                target_change
                    .or(comparison_reason)
                    .unwrap_or(if self.snapshot.is_some() {
                        "scene-change"
                    } else {
                        "initial-scene"
                    }),
            )
        } else {
            None
        };
        Ok(RetainedPlan {
            rtv: target.rtv.clone(),
            srv: target.srv.clone(),
            damage,
            snapshot: Some(snapshot),
            bytes: target.reservation.bytes,
            revision: frame.scene_revision,
            atlas: generation,
            reuse_snapshot: false,
            fallback_reason,
            diff_detail,
        })
    }
    pub fn complete(&mut self, plan: RetainedPlan, success: bool) {
        #[cfg(test)]
        {
            self.last_damage = plan.damage.kind();
        }
        if !success {
            self.invalidate();
            return;
        }
        if !plan.reuse_snapshot {
            self.snapshot = plan.snapshot;
        }
        self.scene_revision = Some(plan.revision);
        self.atlas_generation = plan.atlas;
    }
}
impl Drop for RetainedFrame {
    fn drop(&mut self) {
        self.invalidate();
        poll_retired();
    }
}
