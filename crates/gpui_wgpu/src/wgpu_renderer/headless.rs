use std::sync::Arc;

use gpui::{DevicePixels, Scene, Size};

use crate::{WgpuAtlas, WgpuContext};

use super::{WgpuRenderer, WgpuSurfaceConfig};

struct OffscreenTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    padded_bytes_per_row: u32,
    size: Size<DevicePixels>,
}

impl WgpuRenderer {
    pub(super) fn new_headless(
        context: &WgpuContext,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<Self> {
        Self::new_internal(
            None,
            context,
            None,
            WgpuSurfaceConfig {
                size,
                transparent: false,
                preferred_present_mode: None,
            },
            None,
            None,
            Arc::new(WgpuAtlas::from_context(context)),
        )
    }

    fn create_offscreen_target(&self) -> OffscreenTarget {
        let width = self.target.width();
        let height = self.target.height();
        let padded_bytes_per_row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let texture = self
            .resources()
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("gpui_offscreen_target"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.target.format(),
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let readback = self
            .resources()
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpui_offscreen_readback"),
                size: u64::from(padded_bytes_per_row) * u64::from(height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
        OffscreenTarget {
            texture,
            view,
            readback,
            padded_bytes_per_row,
            size: self.viewport_size(),
        }
    }

    fn read_offscreen_target(
        &self,
        target: &OffscreenTarget,
        submission: wgpu::SubmissionIndex,
    ) -> anyhow::Result<image::RgbaImage> {
        let width = target.size.width.0 as u32;
        let height = target.size.height.0 as u32;
        let bytes_per_row = width * 4;
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        target
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        self.resources()
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| anyhow::anyhow!("failed to poll offscreen readback: {error}"))?;
        receiver
            .recv()
            .map_err(|error| anyhow::anyhow!("offscreen readback callback dropped: {error}"))?
            .map_err(|error| anyhow::anyhow!("failed to map offscreen readback: {error}"))?;

        let mapped = target.readback.slice(..).get_mapped_range();
        let mut pixels = Vec::with_capacity(bytes_per_row as usize * height as usize);
        for row in mapped.chunks_exact(target.padded_bytes_per_row as usize) {
            pixels.extend_from_slice(&row[..bytes_per_row as usize]);
        }
        drop(mapped);
        target.readback.unmap();
        if self.target.format() == wgpu::TextureFormat::Bgra8Unorm {
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        image::RgbaImage::from_raw(width, height, pixels)
            .ok_or_else(|| anyhow::anyhow!("offscreen readback dimensions did not match its data"))
    }

    /// Renders through the normal scene path and reads back without presenting.
    pub fn render_to_image(&mut self, scene: &Scene) -> anyhow::Result<image::RgbaImage> {
        let target = self.create_offscreen_target();
        let submission = self
            .render_to_view_with_readback(scene, &target.view, target.readback_copy())
            .ok_or_else(|| anyhow::anyhow!("failed to render scene into the offscreen target"))?;
        self.read_offscreen_target(&target, submission)
    }
}

impl OffscreenTarget {
    fn readback_copy(&self) -> super::frame::ReadbackCopy<'_> {
        super::frame::ReadbackCopy {
            texture: &self.texture,
            buffer: &self.readback,
            bytes_per_row: self.padded_bytes_per_row,
            width: self.size.width.0 as u32,
            height: self.size.height.0 as u32,
        }
    }
}

/// Surface-free renderer used by GPUI visual tests and benchmarks.
pub struct WgpuHeadlessRenderer {
    renderer: WgpuRenderer,
    target: Option<OffscreenTarget>,
}

impl WgpuHeadlessRenderer {
    pub fn new() -> anyhow::Result<Self> {
        let context = WgpuContext::new_headless(None)?;
        let renderer = WgpuRenderer::new_headless(
            &context,
            Size {
                width: DevicePixels(1),
                height: DevicePixels(1),
            },
        )?;
        Ok(Self {
            renderer,
            target: None,
        })
    }

    fn ensure_target(&mut self, size: Size<DevicePixels>) -> anyhow::Result<()> {
        anyhow::ensure!(
            size.width.0 > 0 && size.height.0 > 0,
            "headless render target must have positive dimensions"
        );
        if self
            .target
            .as_ref()
            .is_some_and(|target| target.size == size)
        {
            return Ok(());
        }
        self.renderer.update_drawable_size(size);
        self.target = Some(self.renderer.create_offscreen_target());
        Ok(())
    }

    /// Renders through the normal submission path and waits for that work to finish.
    pub fn render_scene_and_wait(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<()> {
        self.ensure_target(size)?;
        let target = self.target.as_ref().expect("target was just ensured");
        let submission =
            super::frame::render_to_view(&mut self.renderer, scene, &target.view, None)
                .ok_or_else(|| anyhow::anyhow!("failed to render headless scene"))?;
        self.renderer
            .resources()
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| anyhow::anyhow!("failed to wait for headless render: {error}"))?;
        Ok(())
    }
}

impl gpui::PlatformHeadlessRenderer for WgpuHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.ensure_target(size)?;
        let target = self.target.as_ref().expect("target was just ensured");
        let submission = self
            .renderer
            .render_to_view_with_readback(scene, &target.view, target.readback_copy())
            .ok_or_else(|| anyhow::anyhow!("failed to render headless scene"))?;
        self.renderer.read_offscreen_target(target, submission)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.ensure_target(size)?;
        let target = self.target.as_ref().expect("target was just ensured");
        anyhow::ensure!(
            self.renderer.render_to_view(scene, &target.view),
            "failed to render headless scene"
        );
        Ok(())
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}

#[cfg(test)]
mod path_target_tests {
    use super::*;
    use gpui::{
        Bounds, ContentMask, PathBuilder, Quad, ScaledPixels, point, px, size, solid_background,
    };

    fn triangle(
        x: f32,
        y: f32,
        edge: f32,
        mask: ContentMask<ScaledPixels>,
    ) -> gpui::Path<ScaledPixels> {
        let mut builder = PathBuilder::fill();
        builder.move_to(point(px(x), px(y)));
        builder.line_to(point(px(x + edge), px(y)));
        builder.line_to(point(px(x), px(y + edge)));
        builder.close();
        let mut path = builder.build().unwrap().scale(1.0);
        path.content_mask = mask;
        path.color = gpui::linear_gradient(
            90.0,
            gpui::linear_color_stop(gpui::rgb(0xff0000), 0.0),
            gpui::linear_color_stop(gpui::rgb(0x00ff00), 1.0),
        );
        path
    }

    #[test]
    fn cropped_path_targets_match_full_viewport_pixels_when_growing_and_resizing() {
        let context = WgpuContext::new_headless(None).expect("hardware or software GPU");
        let target = size(DevicePixels(512), DevicePixels(320));
        let mask = ContentMask {
            bounds: Bounds::new(
                point(ScaledPixels(0.0), ScaledPixels(0.0)),
                size(ScaledPixels(1024.0), ScaledPixels(640.0)),
            ),
            ..Default::default()
        };
        let mut full = WgpuRenderer::new_headless(&context, target).unwrap();
        let mut cropped = WgpuRenderer::new_headless(&context, target).unwrap();
        cropped.options.cropped_paths = true;
        full.options.cropped_paths = false;
        for (x, y, target) in [
            (16.0, 16.0, target),
            (180.0, 120.0, target),
            (16.0, 16.0, target),
            (40.0, 35.0, size(DevicePixels(360), DevicePixels(240))),
        ] {
            full.update_drawable_size(target);
            cropped.update_drawable_size(target);
            let mut force_full = Scene::default();
            force_full.paths.push(triangle(
                0.0,
                0.0,
                target.width.0.max(target.height.0) as f32,
                mask,
            ));
            full.ensure_path_textures(&force_full);

            let mut scene = Scene::default();
            scene.insert_primitive(Quad {
                bounds: mask.bounds,
                content_mask: mask,
                background: solid_background(gpui::black()),
                ..Default::default()
            });
            scene.insert_primitive(triangle(x, y, 47.5, mask));
            // A later path beyond the viewport makes the combined sprite span
            // empty space past the cropped target. That space must stay black.
            scene.insert_primitive(Quad {
                bounds: Bounds::new(
                    point(ScaledPixels(0.0), ScaledPixels(0.0)),
                    size(ScaledPixels(1.0), ScaledPixels(1.0)),
                ),
                content_mask: mask,
                background: solid_background(gpui::black()),
                ..Default::default()
            });
            scene.insert_primitive(triangle(800.0, 10.0, 40.0, mask));
            // Negative geometry and a clipped curve exercise coordinate/clip
            // preservation independently of the viewport's dimensions.
            let mut curve = PathBuilder::stroke(px(1.6));
            curve.move_to(point(px(-10.0), px(30.0)));
            curve.cubic_bezier_to(
                point(px(40.0), px(70.0)),
                point(px(24.0), px(22.0)),
                point(px(5.0), px(60.0)),
            );
            let mut curve = curve.build().unwrap().scale(1.0);
            curve.content_mask = ContentMask {
                bounds: Bounds::new(
                    point(ScaledPixels(0.0), ScaledPixels(0.0)),
                    size(ScaledPixels(30.0), ScaledPixels(60.0)),
                ),
                ..Default::default()
            };
            curve.color = solid_background(gpui::white());
            scene.insert_primitive(curve);
            scene.finish();
            let expected = full.render_to_image(&scene).expect("full target frame");
            let actual = cropped
                .render_to_image(&scene)
                .expect("cropped target frame");
            assert!(
                actual.get_pixel(x as u32 + 8, y as u32 + 8)[0] > 100,
                "triangle must render"
            );
            assert_eq!(
                &actual.get_pixel(300, 200).0[..3],
                &[0, 0, 0],
                "no clamped edge smear"
            );
            let max_diff = actual
                .as_raw()
                .iter()
                .zip(expected.as_raw())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            assert!(
                max_diff <= 1,
                "cropped rendering changed a pixel by {max_diff}"
            );
            let texture = cropped
                .resources()
                .path_intermediate_texture
                .as_ref()
                .unwrap();
            assert!(
                texture.width() < target.width.0 as u32,
                "narrow scene must keep a narrow target"
            );
            assert!(
                texture.height() <= target.height.0 as u32,
                "height reservation stays within the viewport"
            );
            if let Ok(directory) = std::env::var("GPUI_PATH_TEST_IMAGES") {
                actual
                    .save(
                        std::path::Path::new(&directory)
                            .join(format!("cropped-{x}-{y}-{}.png", target.width.0)),
                    )
                    .unwrap();
                expected
                    .save(
                        std::path::Path::new(&directory)
                            .join(format!("full-{x}-{y}-{}.png", target.width.0)),
                    )
                    .unwrap();
            }
        }
    }

    #[test]
    fn cached_and_packed_paths_preserve_pixels_and_invalidate_changed_geometry() {
        let context = WgpuContext::new_headless(None).unwrap();
        let viewport = size(DevicePixels(512), DevicePixels(320));
        let mask = ContentMask {
            bounds: Bounds::new(
                point(ScaledPixels(-20.0), ScaledPixels(-20.0)),
                size(ScaledPixels(600.0), ScaledPixels(400.0)),
            ),
            ..Default::default()
        };
        for (cache, packed) in [(true, false), (false, true), (true, true)] {
            let mut reference = WgpuRenderer::new_headless(&context, viewport).unwrap();
            reference.options = Default::default();
            let mut candidate = WgpuRenderer::new_headless(&context, viewport).unwrap();
            candidate.options.cropped_paths = true;
            candidate.options.cached_layers = cache;
            candidate.options.batched_paths = packed;
            for frame in 0..8 {
                let mut scene = Scene::default();
                for row in 0..4 {
                    let mut path = triangle(
                        if row == 0 { -5.25 } else { 16.25 },
                        12.25 + row as f32 * 50.0,
                        24.5,
                        mask,
                    );
                    path.order = row * 2 + 1;
                    if frame >= 5 && row == 1 {
                        path.color = gpui::rgba(0x00ffff88).into();
                    }
                    scene.paths.push(path);
                    // A live row decoration between batches changes during hover.
                    scene.quads.push(Quad {
                        order: row * 2 + 2,
                        bounds: Bounds::new(
                            point(ScaledPixels(10.0), ScaledPixels(30.0 + row as f32 * 50.0)),
                            size(ScaledPixels(32.0), ScaledPixels(10.0)),
                        ),
                        content_mask: mask,
                        background: solid_background(if frame % 2 == 0 {
                            gpui::rgba(0xffffff80)
                        } else {
                            gpui::rgba(0x0000ff80)
                        }),
                        ..Default::default()
                    });
                }
                let mut invisible = triangle(700.0, 60.0, 20.0, mask);
                invisible.order = 4;
                scene.paths.push(invisible);
                scene.finish();
                let expected = reference.render_to_image(&scene).unwrap();
                let actual = candidate.render_to_image(&scene).unwrap();
                let difference = actual
                    .as_raw()
                    .iter()
                    .zip(expected.as_raw())
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap();
                assert!(
                    difference <= 1,
                    "cache={cache} packed={packed} frame={frame} difference={difference}"
                );
                if cache && frame == 4 {
                    assert!(candidate.resources().path_cache.borrow().hits >= 4);
                }
                if cache && frame == 5 {
                    assert!(candidate.resources().path_cache.borrow().misses >= 1);
                }
                assert!(
                    candidate.resources().path_cache.borrow().bytes()
                        <= gpui_render::gpu_policy::WINDOW_LAYER_BYTES
                );
            }
        }
    }
    #[test]
    fn retained_colour_damage_matches_full_redraw_and_recovers_after_fallbacks() {
        let _ = env_logger::try_init();
        for transparent in [false, true] {
            let context = WgpuContext::new_headless(None).unwrap();
            let viewport = size(DevicePixels(512), DevicePixels(320));
            let mut reference = WgpuRenderer::new_headless(&context, viewport).unwrap();
            reference.options = Default::default();
            reference.update_transparency(transparent);
            let mut candidate = WgpuRenderer::new_headless(&context, viewport).unwrap();
            candidate.update_transparency(transparent);
            candidate.options.partial_redraw = true;
            candidate.options.cached_layers = true;
            candidate.options.batched_paths = true;
            candidate.options.cropped_paths = true;
            let mut observed_rect = false;
            for frame in 0..16 {
                let resized = if frame >= 12 {
                    size(DevicePixels(450), DevicePixels(280))
                } else {
                    viewport
                };
                reference.update_drawable_size(resized);
                candidate.update_drawable_size(resized);
                let mask = ContentMask {
                    bounds: Bounds::new(
                        point(ScaledPixels(0.), ScaledPixels(0.)),
                        size(
                            ScaledPixels(resized.width.0 as f32),
                            ScaledPixels(resized.height.0 as f32),
                        ),
                    ),
                    ..Default::default()
                };
                let mut scene = Scene::default();
                scene.quads.push(Quad {
                    order: 0,
                    bounds: mask.bounds,
                    content_mask: mask,
                    background: gpui::rgba(0x12345688).into(),
                    ..Default::default()
                });
                let y = if frame == 7 { 140. } else { 50. };
                scene.quads.push(Quad {
                    order: 1,
                    bounds: Bounds::new(
                        point(ScaledPixels(10.), ScaledPixels(y)),
                        size(ScaledPixels(100.), ScaledPixels(20.)),
                    ),
                    content_mask: mask,
                    background: gpui::rgba(if frame % 2 == 0 {
                        0xff000088
                    } else {
                        0x0000ff40
                    })
                    .into(),
                    ..Default::default()
                });
                scene.shadows.push(gpui::Shadow {
                    order: 2,
                    bounds: Bounds::new(
                        point(ScaledPixels(15.), ScaledPixels(40.)),
                        size(ScaledPixels(100.), ScaledPixels(30.)),
                    ),
                    content_mask: mask,
                    blur_radius: ScaledPixels(6.),
                    color: gpui::rgba(0x00880088).into(),
                    corner_radii: Default::default(),
                    element_bounds: mask.bounds,
                    element_corner_radii: Default::default(),
                    inset: gpui::ShaderBool::Disabled,
                    corner_smoothing: 0.,
                });
                let mut path = triangle(25., 55., 40., mask);
                path.order = 2;
                scene.paths.push(path);
                if frame == 9 {
                    // A popup appears, then disappears.
                    scene.quads.push(Quad {
                        order: 3,
                        bounds: Bounds::new(
                            point(ScaledPixels(20.), ScaledPixels(60.)),
                            size(ScaledPixels(100.), ScaledPixels(80.)),
                        ),
                        content_mask: mask,
                        background: gpui::white().into(),
                        ..Default::default()
                    });
                }
                scene.finish();
                if let Some(retained) = candidate.resources().retained.borrow().as_ref() {
                    if let Some(previous) = retained.snapshot.as_ref() {
                        let snapshot = gpui_render::damage::Snapshot::capture(&scene, 0).unwrap();
                        observed_rect |= matches!(
                            snapshot.compare(
                                previous,
                                (resized.width.0 as u32, resized.height.0 as u32)
                            ),
                            gpui_render::damage::Damage::Rect(_)
                        );
                    }
                }
                let expected = reference.render_to_image(&scene).unwrap();
                let actual = candidate.render_to_image(&scene).unwrap();
                let difference = actual
                    .as_raw()
                    .iter()
                    .zip(expected.as_raw())
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap();
                assert!(difference <= 1, "frame={frame} difference={difference}");
            }
            assert!(
                observed_rect,
                "must exercise incremental redraws, not just full fallback"
            );
            assert!(candidate.resources().retained.borrow().is_some());
            candidate.update_transparency(!transparent);
            assert!(candidate.resources().retained.borrow().is_none());
        }
    }
}
