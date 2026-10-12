//! The retained scene target for partial redraw.
use crate::bindings::Windows::Win32::*;
use gpui::profiler::gpu;
use gpui::{PlatformAtlas, Scene, WindowBackgroundAppearance};
use gpui_render::damage::{Damage, SceneSnapshot};
use gpui_render::retained::{GpuMemoryReservation, RetainedScene, reserve_retained_bytes};
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
pub(super) struct RetainedFrame {
    /// Always on; tests switch it off to render full-redraw references.
    enabled: bool,
    target: Option<Target>,
    scene: RetainedScene,
    #[cfg(test)]
    last_damage: &'static str,
}
pub(super) struct RetainedPlan {
    pub rtv: Option<ID3D11RenderTargetView>,
    pub srv: Option<ID3D11ShaderResourceView>,
    pub damage: Damage,
    pub snapshot: Option<SceneSnapshot>,
    revision: u64,
}
impl RetainedFrame {
    pub fn new() -> Self {
        Self {
            enabled: true,
            target: None,
            scene: RetainedScene::default(),
            #[cfg(test)]
            last_damage: "full",
        }
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
    /// Forget the retained frame and free its target and budget at once: D3D11
    /// keeps a texture alive while queued GPU work still uses it.
    pub fn invalidate(&mut self) {
        self.scene.forget();
        self.target = None;
    }
    pub fn prepare(
        &mut self,
        scene: &Scene,
        atlas: &dyn PlatformAtlas,
        width: u32,
        height: u32,
        appearance: WindowBackgroundAppearance,
        device: &ID3D11Device,
    ) -> RetainedPlan {
        let frame = gpu::gpu_frame_context();
        let generation = atlas.generation();
        let full = || RetainedPlan {
            rtv: None,
            srv: None,
            damage: Damage::Full,
            snapshot: None,
            revision: frame.scene_revision,
        };
        if !self.enabled {
            return full();
        }
        if !frame.interactive {
            self.invalidate();
            return full();
        }
        if self.target.as_ref().is_some_and(|target| {
            target.width != width || target.height != height || target.appearance != appearance
        }) {
            self.invalidate();
        }
        // Reserve before hashing, so a window the budget can't hold doesn't
        // capture its scene every frame only to draw in full.
        let reservation = if self.target.is_none() {
            let bytes = width as u64 * height as u64 * 4;
            let reservation = reserve_retained_bytes("d3d11", device.as_raw() as usize, bytes);
            if reservation.is_none() {
                self.invalidate();
                return full();
            }
            reservation
        } else {
            None
        };
        let Some((mut damage, snapshot)) =
            self.scene
                .plan(scene, frame.scene_revision, generation, width, height)
        else {
            self.invalidate();
            return full();
        };
        if let Some(reservation) = reservation {
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
            let created = unsafe {
                device
                    .CreateTexture2D(&desc, None, Some(&mut texture))
                    .ok()
                    .and_then(|()| {
                        let texture = texture.as_ref().unwrap();
                        device
                            .CreateRenderTargetView(texture, None, Some(&mut rtv))
                            .ok()?;
                        device
                            .CreateShaderResourceView(texture, None, Some(&mut srv))
                            .ok()
                    })
            };
            // The target is optional: without one, draw in full, as with an
            // exhausted budget, instead of failing the frame.
            if created.is_err() {
                self.invalidate();
                return full();
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
        RetainedPlan {
            rtv: target.rtv.clone(),
            srv: target.srv.clone(),
            damage,
            snapshot,
            revision: frame.scene_revision,
        }
    }
    pub fn complete(&mut self, plan: RetainedPlan, success: bool) {
        #[cfg(test)]
        {
            self.last_damage = plan.damage.kind();
        }
        if success {
            self.scene.complete(plan.snapshot, plan.revision);
        } else {
            self.invalidate();
        }
    }
}
