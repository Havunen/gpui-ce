use gpui::Scene;
use gpui::profiler::gpu::{self, GpuMemoryReservation};
use gpui_render::damage::{Damage, SceneSnapshot};
use std::sync::Arc;

pub(super) struct RetainedFrame {
    enabled: bool,
    target: Option<Target>,
    snapshot: Option<SceneSnapshot>,
    scene_revision: Option<u64>,
    atlas_generation: u64,
    queue: Option<Arc<wgpu::Queue>>,
    #[cfg(test)]
    last_damage: &'static str,
}
struct Target {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    alpha_mode: wgpu::CompositeAlphaMode,
    clear_color: wgpu::Color,
    reservation: GpuMemoryReservation,
}
pub(super) struct RetainedPlan {
    pub view: Option<wgpu::TextureView>,
    pub damage: Damage,
    pub snapshot: Option<SceneSnapshot>,
    pub bytes: u64,
    pub scene_revision: u64,
    pub atlas_generation: u64,
    pub reuse_snapshot: bool,
    pub fallback_reason: Option<&'static str>,
    pub diff_detail: Option<String>,
}
impl RetainedFrame {
    pub fn new() -> Self {
        Self {
            enabled: cfg!(target_os = "windows") && gpu::experiment_enabled("partial-redraw"),
            target: None,
            snapshot: None,
            scene_revision: None,
            atlas_generation: 0,
            queue: None,
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
        if let Some(queue) = self.queue.clone() {
            self.invalidate(&queue);
        }
    }
    #[cfg(test)]
    pub fn last_damage(&self) -> &'static str {
        self.last_damage
    }
    pub fn invalidate(&mut self, queue: &wgpu::Queue) {
        self.snapshot = None;
        self.scene_revision = None;
        if let Some(target) = self.target.take() {
            queue.on_submitted_work_done(move || drop(target));
        }
    }
    pub fn prepare(
        &mut self,
        scene: &Scene,
        atlas: u64,
        width: u32,
        height: u32,
        eligible: bool,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
        clear_color: wgpu::Color,
    ) -> RetainedPlan {
        if !self.enabled {
            return RetainedPlan {
                view: None,
                damage: Damage::Full,
                snapshot: None,
                bytes: 0,
                scene_revision: 0,
                atlas_generation: 0,
                reuse_snapshot: false,
                fallback_reason: None,
                diff_detail: None,
            };
        }
        let context = gpu::gpu_frame_context();
        let full = |reason| RetainedPlan {
            view: None,
            damage: Damage::Full,
            snapshot: None,
            bytes: 0,
            scene_revision: context.scene_revision,
            atlas_generation: atlas,
            reuse_snapshot: false,
            fallback_reason: Some(reason),
            diff_detail: None,
        };
        if !eligible || !context.active {
            self.invalidate(queue);
            return full(if !eligible {
                "diagnostic-ablation"
            } else {
                "inactive-window"
            });
        }
        if !matches!(
            alpha_mode,
            wgpu::CompositeAlphaMode::Opaque | wgpu::CompositeAlphaMode::PreMultiplied
        ) {
            self.invalidate(queue);
            return full("unsupported-alpha-mode");
        }
        // The damage-clear Quad must reproduce the raw target clear exactly.
        // Colored translucent clears would be premultiplied again by its shader;
        // arbitrary opaque colors also lack an exact HSL round-trip proof.
        if !retained_clear_supported(clear_color) {
            self.invalidate(queue);
            return full("unsupported-clear-color");
        }
        if retained_bytes(format, width, height).is_none() {
            self.invalidate(queue);
            return full("unsupported-target-format-or-size");
        }
        self.queue = Some(queue.clone());
        let mut target_change = None;
        if self.target.as_ref().is_some_and(|target| {
            target.width != width
                || target.height != height
                || target.format != format
                || target.alpha_mode != alpha_mode
                || target.clear_color != clear_color
        }) {
            target_change = Some("target-configuration-change");
            self.invalidate(queue);
        }
        // Keepalive submissions reuse the same finished scene. Avoid cloning or comparing
        // primitives on that path; atlas changes still invalidate it independently.
        if self.scene_revision == Some(context.scene_revision)
            && self.atlas_generation == atlas
            && self.snapshot.is_some()
        {
            if let Some(target) = &self.target {
                return RetainedPlan {
                    view: Some(target.view.clone()),
                    damage: Damage::Reuse,
                    snapshot: None,
                    bytes: target.reservation.bytes,
                    scene_revision: context.scene_revision,
                    atlas_generation: atlas,
                    reuse_snapshot: true,
                    fallback_reason: None,
                    diff_detail: None,
                };
            }
        }
        let snapshot = match SceneSnapshot::capture_checked(scene, atlas, width, height) {
            Ok(snapshot) => snapshot,
            Err(reason) => {
                self.invalidate(queue);
                return full(reason.as_str());
            }
        };
        if self.target.is_none() {
            let Some(bytes) = retained_bytes(format, width, height) else {
                return full("unsupported-target-format-or-size");
            };
            let Some(reservation) =
                gpu::reserve_retained_bytes("wgpu", Arc::as_ptr(device) as usize, bytes)
            else {
                return full("retained-device-budget");
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gpui_retained_frame"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            self.target = Some(Target {
                _texture: texture,
                view,
                width,
                height,
                format,
                alpha_mode,
                clear_color,
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
                    // Exact local painter sequences handle disjoint global-rank
                    // permutations without weakening the strict fast path.
                    Err(_) => match snapshot.compare_tiled_checked(previous) {
                        Ok(damage) => (damage, None, detail),
                        Err(reason) => (Damage::Full, Some(reason), detail),
                    },
                }
            }
            None => (Damage::Full, None, None),
        };
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
        let target = self.target.as_ref().unwrap();
        RetainedPlan {
            view: Some(target.view.clone()),
            damage,
            snapshot: Some(snapshot),
            bytes: target.reservation.bytes,
            scene_revision: context.scene_revision,
            atlas_generation: atlas,
            reuse_snapshot: false,
            fallback_reason,
            diff_detail,
        }
    }
    pub fn complete(
        &mut self,
        snapshot: Option<SceneSnapshot>,
        revision: u64,
        atlas: u64,
        reuse_snapshot: bool,
        damage_kind: &'static str,
        success: bool,
        queue: &wgpu::Queue,
    ) {
        #[cfg(test)]
        {
            self.last_damage = damage_kind;
        }
        #[cfg(not(test))]
        let _ = damage_kind;
        if success {
            if !reuse_snapshot {
                self.snapshot = snapshot;
            }
            self.scene_revision = Some(revision);
            self.atlas_generation = atlas;
        } else {
            self.invalidate(queue);
        }
    }
}
impl Drop for RetainedFrame {
    fn drop(&mut self) {
        if let Some(queue) = self.queue.clone() {
            self.invalidate(&queue);
        }
    }
}

fn retained_clear_supported(clear: wgpu::Color) -> bool {
    // Both linear and sRGB targets preserve the black/white endpoints exactly.
    // Black RGB stays black through either premultiplied or straight-alpha output.
    let black = clear.r == 0.0 && clear.g == 0.0 && clear.b == 0.0;
    let opaque_white = clear.r == 1.0 && clear.g == 1.0 && clear.b == 1.0 && clear.a == 1.0;
    (black && clear.a.is_finite() && (0.0..=1.0).contains(&clear.a)) || opaque_white
}

fn retained_bytes(format: wgpu::TextureFormat, width: u32, height: u32) -> Option<u64> {
    if width == 0
        || height == 0
        || !matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm
                | wgpu::TextureFormat::Bgra8UnormSrgb
                | wgpu::TextureFormat::Rgba8Unorm
                | wgpu::TextureFormat::Rgba8UnormSrgb
        )
    {
        return None;
    }
    (width as u64).checked_mul(height as u64)?.checked_mul(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_clear_requires_exact_shader_endpoint_representation() {
        assert!(retained_clear_supported(wgpu::Color::TRANSPARENT));
        assert!(retained_clear_supported(wgpu::Color::BLACK));
        assert!(retained_clear_supported(wgpu::Color::WHITE));
        assert!(retained_clear_supported(wgpu::Color {
            a: 0.25,
            ..wgpu::Color::BLACK
        }));
        for alpha in [0.0, 0.25, 1.0] {
            assert!(!retained_clear_supported(wgpu::Color {
                r: 0.04,
                g: 0.03,
                b: 0.02,
                a: alpha,
            }));
        }
        assert!(!retained_clear_supported(wgpu::Color {
            a: f64::NAN,
            ..wgpu::Color::BLACK
        }));
        assert!(!retained_clear_supported(wgpu::Color {
            a: -0.25,
            ..wgpu::Color::BLACK
        }));
    }

    #[test]
    fn retained_budget_rejects_hdr_unknown_formats_zero_extent_and_overflow() {
        assert_eq!(
            retained_bytes(wgpu::TextureFormat::Bgra8Unorm, 4096, 2048),
            Some(gpu::RETAINED_DEVICE_LIMIT)
        );
        assert_eq!(
            retained_bytes(wgpu::TextureFormat::Rgba16Float, 4096, 2048),
            None
        );
        assert_eq!(retained_bytes(wgpu::TextureFormat::Rgba8Unorm, 0, 8), None);
        assert_eq!(
            retained_bytes(wgpu::TextureFormat::Rgba8Unorm, u32::MAX, u32::MAX),
            None
        );
    }
}
