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
        let mut renderer = Self::new_internal(
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
        )?;

        renderer.target.clear_color = wgpu::Color::BLACK;

        Ok(renderer)
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
        // Frame preparation can consume an asynchronous validation error.
        // Retain its text for test diagnostics without changing retry policy.
        let pending_before = self.faults.pending_error.lock().unwrap().clone();
        let submission = self
            .render_to_view_with_readback(scene, &target.view, target.readback_copy())
            .ok_or_else(|| anyhow::anyhow!(
                "failed to render scene into the offscreen target; pending_before={pending_before:?}; pending_after={:?}; terminal={:?}; failed_frames={}",
                self.faults.pending_error.lock().unwrap(), self.terminal_error(), self.faults.consecutive_failed_frames))?;
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
        AtlasKey, AtlasTile, Bounds, ContentMask, FontId, GlyphId, GlyphRenderMode,
        MonochromeSprite, PathBuilder, PlatformAtlas, PreparedRasterStyle, Quad,
        RasterizedGlyphFormat, RenderGlyphParams, ScaledPixels, Shadow, point, px, size,
        solid_background,
    };
    use std::borrow::Cow;

    fn scaled(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(width), ScaledPixels(height)),
        )
    }
    fn full_mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: scaled(0.0, 0.0, 200.0, 100.0),
            ..Default::default()
        }
    }

    fn clipped_quad_scene(fraction: f32, smoothing: f32, paint: usize) -> Scene {
        let bounds = scaled(7.25 + fraction, 8.5 + fraction, 180.25, 83.25);
        let x = bounds.origin.x.0;
        let y = bounds.origin.y.0;
        let w = bounds.size.width.0;
        let h = bounds.size.height.0;
        let gradient = gpui::linear_gradient(
            37.0,
            gpui::linear_color_stop(gpui::hsla(0.01, 0.85, 0.62, 0.8), 0.0),
            gpui::linear_color_stop(gpui::hsla(0.62, 0.7, 0.35, 0.55), 1.0),
        );
        let background = match paint {
            0 => gpui::Background::default(), // Actual border-only split shape.
            1 => gradient.color_space(gpui::ColorSpace::Srgb),
            2 => gradient.color_space(gpui::ColorSpace::Oklab),
            _ => gpui::pattern_slash(gpui::hsla(0.6, 0.8, 0.6, 0.7), 1.5, 4.25),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            bounds: scaled(0.0, 0.0, 200.0, 100.0),
            content_mask: full_mask(),
            background: solid_background(gpui::rgb(0x18212b)),
            ..Default::default()
        });
        for (index, clip) in [
            scaled(x, y, w, 2.5),
            scaled(x + w - 3.0, y, 3.0, h),
            scaled(x, y + h - 4.0, w, 4.0),
            scaled(x, y, 1.5, h),
            scaled(x + 45.25, y + 25.5, 8.75, 7.25),
            scaled(x + 250.0, y + 140.0, 1.0, 1.0), // Empty intersection.
        ]
        .into_iter()
        .enumerate()
        {
            scene.insert_primitive(Quad {
                bounds,
                content_mask: ContentMask {
                    bounds: clip,
                    fade_out: if index == 4 {
                        gpui::Edges::all(ScaledPixels(1.25))
                    } else {
                        gpui::Edges::default()
                    },
                },
                background,
                border_color: gradient,
                border_style: if paint % 2 == 0 {
                    gpui::BorderStyle::Solid
                } else {
                    gpui::BorderStyle::Dashed
                },
                border_dashed_length: 3.75,
                border_dashed_gap: 1.25,
                corner_radii: gpui::Corners {
                    top_left: ScaledPixels(11.5),
                    top_right: ScaledPixels(16.25),
                    bottom_right: ScaledPixels(8.75),
                    bottom_left: ScaledPixels(14.5),
                },
                border_widths: gpui::Edges {
                    top: ScaledPixels(2.5),
                    right: ScaledPixels(3.0),
                    bottom: ScaledPixels(4.0),
                    left: ScaledPixels(1.5),
                },
                corner_smoothing: smoothing,
                ..Default::default()
            });
        }
        // Later translucent paint makes accidental clear/reordering visible.
        scene.insert_primitive(Quad {
            bounds: scaled(40.5, 9.5, 18.5, 70.5),
            content_mask: full_mask(),
            background: solid_background(gpui::hsla(0.02, 0.8, 0.6, 0.4)),
            corner_radii: gpui::Corners::all(ScaledPixels(4.25)),
            ..Default::default()
        });
        scene.finish();
        scene
    }

    #[test]
    fn clipped_quads_match_full_pixels_for_fractional_masks_borders_gradients_fades()
    -> anyhow::Result<()> {
        let _gpu_test_guard = crate::test_gpu::guard();
        let context = WgpuContext::new_headless(None)?;
        let mut renderer =
            WgpuRenderer::new_headless(&context, size(DevicePixels(200), DevicePixels(100)))?;
        renderer.retained.disable_for_tests();
        renderer.timings.diagnostic_ablations = gpui::profiler::gpu::DiagnosticAblations::default();
        assert!(
            renderer
                .profile_capabilities()
                .implemented_experiments
                .iter()
                .any(|name| name == "clipped-quads")
        );
        for fraction in [0.0, 0.25, 0.5, 0.75] {
            for smoothing in [0.0, 0.65] {
                for paint in 0..4 {
                    let scene = clipped_quad_scene(fraction, smoothing, paint);
                    renderer.timings.clipped_quads = false;
                    assert!(
                        !renderer
                            .profile_capabilities()
                            .active_experiments
                            .iter()
                            .any(|name| name == "clipped-quads")
                    );
                    let full = renderer.render_to_image(&scene)?;
                    renderer.timings.clipped_quads = true;
                    assert!(
                        renderer
                            .profile_capabilities()
                            .active_experiments
                            .iter()
                            .any(|name| name == "clipped-quads")
                    );
                    let clipped = renderer.render_to_image(&scene)?;
                    let difference = full
                        .pixels()
                        .zip(clipped.pixels())
                        .enumerate()
                        .find(|(_, (a, b))| a != b);
                    assert!(
                        difference.is_none(),
                        "fraction={fraction} smoothing={smoothing} paint={paint} first differing pixel={difference:?}"
                    );
                }
            }
        }
        Ok(())
    }

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

    fn hover_scene(row: usize, resting_fills: bool) -> Scene {
        let mask = ContentMask {
            bounds: Bounds::new(
                point(ScaledPixels(0.0), ScaledPixels(0.0)),
                size(ScaledPixels(200.0), ScaledPixels(100.0)),
            ),
            ..Default::default()
        };
        let mut scene = Scene::default();
        // The headless target clears to black. Leave uncovered pixels around each
        // translucent row so a wrong retained clear color is observable.
        for index in 0..3 {
            if !resting_fills && index != row {
                continue;
            }
            scene.insert_primitive(Quad {
                bounds: Bounds::new(
                    point(ScaledPixels(4.0), ScaledPixels(10.0 + index as f32 * 25.0)),
                    size(ScaledPixels(192.0), ScaledPixels(14.0)),
                ),
                content_mask: mask,
                background: solid_background(if row == index {
                    gpui::hsla(0.6, 0.4, 0.4, 0.7)
                } else {
                    gpui::hsla(0.6, 0.1, 0.15, 0.7)
                }),
                ..Default::default()
            });
        }
        let mut graph = triangle(
            20.0,
            8.0,
            18.0,
            ContentMask {
                bounds: Bounds::new(
                    point(ScaledPixels(20.0), ScaledPixels(10.0)),
                    size(ScaledPixels(18.0), ScaledPixels(14.0)),
                ),
                ..Default::default()
            },
        );
        graph.color = solid_background(if row == 0 {
            gpui::hsla(0.6, 0.4, 0.4, 0.7)
        } else {
            gpui::black()
        });
        scene.insert_primitive(graph);
        scene.insert_primitive(Quad {
            bounds: Bounds::new(
                point(ScaledPixels(80.0), ScaledPixels(5.0)),
                size(ScaledPixels(15.0), ScaledPixels(70.0)),
            ),
            content_mask: ContentMask {
                bounds: Bounds::new(
                    point(ScaledPixels(82.0), ScaledPixels(0.0)),
                    size(ScaledPixels(10.0), ScaledPixels(100.0)),
                ),
                ..Default::default()
            },
            background: solid_background(gpui::hsla(0.02, 0.8, 0.6, 0.4)),
            corner_radii: gpui::Corners::all(ScaledPixels(4.0)),
            ..Default::default()
        });
        scene.finish();
        scene
    }

    #[test]
    fn diagnostic_ablation_clears_and_batch_omissions_match_empty_frame() -> anyhow::Result<()> {
        let _gpu_test_guard = crate::test_gpu::guard();
        let context = WgpuContext::new_headless(None)?;
        let target = size(DevicePixels(200), DevicePixels(100));
        let mut renderer = WgpuRenderer::new_headless(&context, target)?;
        renderer.retained.enable_for_tests();
        let mut reference = WgpuRenderer::new_headless(&context, target)?;
        reference.retained.disable_for_tests();
        let mut empty = Scene::default();
        empty.finish();
        let expected = reference.render_to_image(&empty)?;
        let scene = hover_scene(0, true);
        let _scope = gpui::profiler::gpu::set_gpu_frame_context(
            Some("diagnostic-wgpu-test".into()),
            true,
            1,
            1,
        );
        assert_ne!(renderer.render_to_image(&scene)?, expected);
        for name in ["clear-only", "no-quads,no-paths"] {
            renderer.timings.diagnostic_ablations =
                gpui::profiler::gpu::DiagnosticAblations::parse(name)
                    .map_err(anyhow::Error::msg)?;
            assert_eq!(
                renderer.render_to_image(&scene)?,
                expected,
                "diagnostic {name}"
            );
            assert_eq!(renderer.retained.last_damage(), "full");
            assert_eq!(
                renderer.profile_capabilities().active_diagnostic_ablations,
                renderer.timings.diagnostic_ablations.active_names()
            );
        }
        Ok(())
    }

    #[test]
    fn retained_hover_matches_full_pixels_for_rows_graph_clip_and_invalidation()
    -> anyhow::Result<()> {
        let _gpu_test_guard = crate::test_gpu::guard();
        let context = WgpuContext::new_headless(None)?;
        for (crop_paths, resting_fills) in
            [(false, true), (true, true), (false, false), (true, false)]
        {
            let target = size(DevicePixels(200), DevicePixels(100));
            let mut full = WgpuRenderer::new_headless(&context, target)?;
            full.retained.disable_for_tests();
            let mut retained = WgpuRenderer::new_headless(&context, target)?;
            full.crop_path_targets = crop_paths;
            retained.crop_path_targets = crop_paths;
            retained.retained.enable_for_tests();
            let mut revision = 0;
            let mut previous = None;
            for row in [0, 0, 1, 2, 0] {
                if previous != Some(row) {
                    revision += 1;
                }
                let scene = hover_scene(row, resting_fills);
                let _scope = gpui::profiler::gpu::set_gpu_frame_context(
                    Some("retained-wgpu-test".into()),
                    true,
                    revision,
                    revision,
                );
                let expected = full.render_to_image(&scene)?;
                let actual = retained.render_to_image(&scene)?;
                assert_eq!(
                    actual, expected,
                    "retained pixels differ for row {row}, revision {revision}, cropped paths {crop_paths}"
                );
                assert_eq!(
                    retained.retained.last_damage(),
                    if previous.is_none() {
                        "full"
                    } else if previous == Some(row) {
                        "reuse"
                    } else {
                        "partial"
                    },
                    "damage selection for row {row}, previous {previous:?}, revision {revision}, cropped paths {crop_paths}, resting fills {resting_fills}"
                );
                previous = Some(row);
            }
            let scene = hover_scene(0, resting_fills);
            let _scope = gpui::profiler::gpu::set_gpu_frame_context(
                Some("retained-wgpu-test".into()),
                true,
                revision,
                revision,
            );
            retained.atlas.clear();
            assert_eq!(
                retained.render_to_image(&scene)?,
                full.render_to_image(&scene)?
            );
            assert_eq!(
                retained.retained.last_damage(),
                "full",
                "atlas invalidates unchanged scene revision"
            );
            let target = size(DevicePixels(190), DevicePixels(100));
            retained.update_drawable_size(target);
            full.update_drawable_size(target);
            assert_eq!(
                retained.render_to_image(&scene)?,
                full.render_to_image(&scene)?
            );
            assert_eq!(
                retained.retained.last_damage(),
                "full",
                "resize invalidates unchanged scene revision"
            );
            retained.update_transparency(true);
            full.update_transparency(true);
            assert_eq!(
                retained.render_to_image(&scene)?,
                full.render_to_image(&scene)?
            );
            assert_eq!(
                retained.retained.last_damage(),
                "full",
                "appearance transition invalidates the previous opaque target"
            );
            assert_eq!(
                retained.render_to_image(&scene)?,
                full.render_to_image(&scene)?
            );
            assert_eq!(retained.retained.last_damage(), "reuse");
        }
        Ok(())
    }

    fn assert_retained_rgba(
        full: &mut WgpuRenderer,
        retained: &mut WgpuRenderer,
        scene: &Scene,
        revision: u64,
        damage: &str,
        context: &str,
    ) -> anyhow::Result<image::RgbaImage> {
        let _scope = gpui::profiler::gpu::set_gpu_frame_context(
            Some("transparent-retained-wgpu-test".into()),
            true,
            revision,
            revision,
        );
        let expected = full
            .render_to_image(scene)
            .map_err(|error| anyhow::anyhow!("{context}: full renderer: {error:#}"))?;
        let actual = retained
            .render_to_image(scene)
            .map_err(|error| anyhow::anyhow!("{context}: retained renderer: {error:#}"))?;
        assert_eq!(actual.dimensions(), expected.dimensions(), "{context}");
        let first_difference = actual
            .as_raw()
            .iter()
            .zip(expected.as_raw())
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected);
        assert!(
            first_difference.is_none(),
            "{context}: exact RGBA comparison failed at byte {first_difference:?}"
        );
        assert_eq!(
            retained.retained.last_damage(),
            damage,
            "{context}: the requested retained path must actually execute"
        );
        Ok(actual)
    }

    fn empty_transparent_hover_scene(row: Option<usize>) -> Scene {
        let mut scene = Scene::default();
        if let Some(row) = row {
            scene.insert_primitive(Quad {
                bounds: scaled(12.25, 10.5 + row as f32 * 25.0, 120.5, 13.25),
                content_mask: full_mask(),
                background: gpui::rgba(0x4488cc70).into(),
                corner_radii: gpui::Corners::all(ScaledPixels(3.25)),
                ..Default::default()
            });
        }
        // A zero-alpha anchor preserves an exact following painter slot when the
        // only visible primitive moves; every resting-frame pixel is still clear.
        scene.insert_primitive(Quad {
            bounds: scaled(190.0, 90.0, 2.0, 2.0),
            content_mask: full_mask(),
            background: gpui::rgba(0x00000000).into(),
            ..Default::default()
        });
        scene.finish();
        scene
    }

    #[test]
    fn retained_transparent_clear_insert_move_erase_is_exact_rgba() -> anyhow::Result<()> {
        let _gpu_test_guard = crate::test_gpu::guard();
        let context = WgpuContext::new_headless(None)?;
        let target = size(DevicePixels(200), DevicePixels(100));
        let mut full = WgpuRenderer::new_headless(&context, target)?;
        let mut retained = WgpuRenderer::new_headless(&context, target)?;
        full.retained.disable_for_tests();
        retained.retained.enable_for_tests();
        full.update_transparency(true);
        retained.update_transparency(true);
        full.timings.diagnostic_ablations = Default::default();
        retained.timings.diagnostic_ablations = Default::default();
        let mut revision = 0;
        let mut previous = None;
        for (index, row) in [None, None, Some(0), Some(0), Some(1), Some(1), None, None]
            .into_iter()
            .enumerate()
        {
            if index == 0 || row != previous {
                revision += 1;
            }
            let image = assert_retained_rgba(
                &mut full,
                &mut retained,
                &empty_transparent_hover_scene(row),
                revision,
                if index == 0 {
                    "full"
                } else if row == previous {
                    "reuse"
                } else {
                    "partial"
                },
                &format!("empty transparent target, step {index}, row {row:?}"),
            )?;
            if row.is_none() {
                assert!(
                    image.as_raw().iter().all(|byte| *byte == 0),
                    "initial clear and erased hover must leave transparent black RGBA"
                );
            } else {
                assert!(image.pixels().any(|pixel| pixel[3] > 0 && pixel[3] < 255));
            }
            previous = row;
        }
        Ok(())
    }

    #[test]
    fn retained_raw_clear_endpoints_and_colored_fallback_match_full_rgba() -> anyhow::Result<()> {
        let _gpu_test_guard = crate::test_gpu::guard();
        let context = WgpuContext::new_headless(None)?;
        let target = size(DevicePixels(200), DevicePixels(100));
        let mut full = WgpuRenderer::new_headless(&context, target)?;
        let mut retained = WgpuRenderer::new_headless(&context, target)?;
        full.retained.disable_for_tests();
        retained.retained.enable_for_tests();
        full.timings.diagnostic_ablations = Default::default();
        retained.timings.diagnostic_ablations = Default::default();
        for transparent in [false, true] {
            full.update_transparency(transparent);
            retained.update_transparency(transparent);
            for (clear, supported, label) in [
                (wgpu::Color::TRANSPARENT, true, "transparent black"),
                (
                    wgpu::Color {
                        a: 0.25,
                        ..wgpu::Color::BLACK
                    },
                    true,
                    "translucent black",
                ),
                (wgpu::Color::BLACK, true, "opaque black"),
                (wgpu::Color::WHITE, true, "opaque white"),
                (
                    wgpu::Color {
                        r: 0.04,
                        g: 0.03,
                        b: 0.02,
                        a: 0.25,
                    },
                    false,
                    "translucent colored clear",
                ),
                (
                    wgpu::Color {
                        r: 0.04,
                        g: 0.03,
                        b: 0.02,
                        a: 0.0,
                    },
                    false,
                    "zero-alpha colored raw clear",
                ),
                (
                    wgpu::Color {
                        r: 0.04,
                        g: 0.03,
                        b: 0.02,
                        a: 1.0,
                    },
                    false,
                    "opaque colored raw clear",
                ),
            ] {
                full.target.clear_color = clear;
                retained.target.clear_color = clear;
                let queue = retained.resources().queue.clone();
                retained.retained.invalidate(&queue);
                for (index, row) in [None, Some(0), Some(1), None].into_iter().enumerate() {
                    let scene = empty_transparent_hover_scene(row);
                    let revision = index as u64 + 1;
                    let label = format!("alpha={transparent}, clear={label}, step={index}");
                    assert_retained_rgba(
                        &mut full,
                        &mut retained,
                        &scene,
                        revision,
                        if supported && index != 0 {
                            "partial"
                        } else {
                            "full"
                        },
                        &label,
                    )?;
                    // Unsupported clears must never retain faulty pixels, including
                    // an unchanged repeat immediately after an inserted/erased fill.
                    assert_retained_rgba(
                        &mut full,
                        &mut retained,
                        &scene,
                        revision,
                        if supported { "reuse" } else { "full" },
                        &format!("repeat: {label}"),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn retained_glyph_tile(renderer: &WgpuRenderer) -> anyhow::Result<AtlasTile> {
        let key = AtlasKey::Glyph {
            params: RenderGlyphParams {
                font_id: FontId(0),
                glyph_id: GlyphId(65),
                font_size: px(8.0),
                subpixel_variant: point(0, 0),
                scale_factor: 1.0,
                raster_style: PreparedRasterStyle::independent(GlyphRenderMode::Grayscale),
            },
            format: RasterizedGlyphFormat::AlphaMask,
        };
        // Deterministic glyph-shaped coverage exercises the real monochrome atlas
        // pipeline without depending on a machine's font rasterizer or font files.
        let coverage = [
            0, 0, 96, 255, 255, 96, 0, 0, 0, 64, 255, 96, 96, 255, 64, 0, 0, 128, 255, 0, 0, 255,
            128, 0, 32, 224, 160, 0, 0, 160, 224, 32, 96, 255, 255, 255, 255, 255, 255, 96, 160,
            224, 64, 0, 0, 64, 224, 160, 224, 160, 0, 0, 0, 0, 160, 224, 255, 96, 0, 0, 0, 0, 96,
            255,
        ];
        renderer
            .atlas
            .get_or_insert_with(&key, &mut || {
                Ok(Some((
                    size(DevicePixels(8), DevicePixels(8)),
                    Cow::Borrowed(&coverage),
                )))
            })?
            .ok_or_else(|| anyhow::anyhow!("retained glyph fixture must produce an atlas tile"))
    }

    fn retained_tiled_glyph(
        renderer: &WgpuRenderer,
        mode: GlyphRenderMode,
    ) -> anyhow::Result<AtlasTile> {
        let key = AtlasKey::Glyph {
            params: RenderGlyphParams {
                font_id: FontId(0),
                glyph_id: GlyphId(90),
                font_size: px(8.0),
                subpixel_variant: point(0, 0),
                scale_factor: 1.0,
                raster_style: PreparedRasterStyle::independent(mode),
            },
            format: mode.rasterized_format(),
        };
        let coverage: Vec<u8> = (0..64)
            .map(|index| match (index / 8, index % 8) {
                (4, 1..=6) | (1..=6, 1 | 6) => 255,
                (_, 2 | 5) => 96,
                _ => 0,
            })
            .collect();
        let bytes = if mode == GlyphRenderMode::Subpixel {
            coverage
                .iter()
                .flat_map(|&value| [value / 2, value, value.saturating_add(24), 255])
                .collect()
        } else {
            coverage
        };
        renderer
            .atlas
            .get_or_insert_with(&key, &mut || {
                Ok(Some((
                    size(DevicePixels(8), DevicePixels(8)),
                    Cow::Borrowed(&bytes),
                )))
            })?
            .ok_or_else(|| anyhow::anyhow!("tiled glyph fixture must allocate an atlas tile"))
    }

    fn tiled_history_scene(
        hover: Option<usize>,
        fraction: f32,
        glyph: AtlasTile,
        mode: GlyphRenderMode,
    ) -> Scene {
        let mut scene = Scene::default();
        let pane = scaled(20.0, 12.0, 460.0, 840.0);
        let mask = ContentMask {
            bounds: pane,
            ..Default::default()
        };
        scene.insert_primitive(Shadow {
            order: 0,
            blur_radius: ScaledPixels(2.25),
            bounds: pane,
            corner_radii: gpui::Corners::all(ScaledPixels(8.25)),
            content_mask: ContentMask {
                bounds: scaled(0.0, 0.0, 512.0, 1024.0),
                ..Default::default()
            },
            color: gpui::rgba(0x17243850).into(),
            element_bounds: pane,
            element_corner_radii: gpui::Corners::all(ScaledPixels(8.25)),
            inset: false.into(),
            corner_smoothing: 0.65,
        });
        scene.insert_primitive(Quad {
            bounds: pane,
            content_mask: mask,
            background: gpui::rgba(0x223344bb).into(),
            corner_radii: gpui::Corners::all(ScaledPixels(8.25)),
            corner_smoothing: 0.65,
            ..Default::default()
        });
        // Stable graph raster groups precede ordinary bounds-tree row paints.
        // No per-row push_layer/raise_order_floor masks the live rank permutation.
        for row in [0, 5, 10, 19] {
            scene.insert_primitive(triangle(
                24.25 + fraction,
                18.25 + row as f32 * 40.0,
                10.5,
                mask,
            ));
        }
        let paint_glyph = |scene: &mut Scene, x: f32, y: f32, color: gpui::Hsla| {
            if mode == GlyphRenderMode::Subpixel {
                scene.insert_primitive(gpui::SubpixelSprite {
                    order: 0,
                    padding: 0,
                    bounds: scaled(x, y, 8.5, 10.25),
                    content_mask: mask,
                    color: color.into(),
                    tile: glyph,
                    transformation: Default::default(),
                });
            } else {
                scene.insert_primitive(MonochromeSprite {
                    order: 0,
                    padding: 0,
                    bounds: scaled(x, y, 8.5, 10.25),
                    content_mask: mask,
                    color: color.into(),
                    tile: glyph,
                    transformation: Default::default(),
                });
            }
        };
        for row in 0..20 {
            let y = 16.25 + fraction + row as f32 * 40.0;
            if hover == Some(row) {
                scene.insert_primitive(Quad {
                    bounds: scaled(20.0, y, 460.0, 28.5),
                    content_mask: mask,
                    background: gpui::rgba(0x55779944).into(),
                    ..Default::default()
                });
            }
            // Live graph cutouts have unchanged geometry but hover-dependent
            // backing colors. Color changes must not consume structural edits.
            for (x, width) in [(134.0, 66.0), (200.0, 4.0), (410.0, 24.0)] {
                scene.insert_primitive(Quad {
                    bounds: scaled(x, y, width, 28.5),
                    content_mask: mask,
                    background: if hover == Some(row) {
                        gpui::rgba(0x55779999)
                    } else {
                        gpui::rgba(0x22334499)
                    }
                    .into(),
                    ..Default::default()
                });
            }
            for index in 0..20 {
                paint_glyph(
                    &mut scene,
                    32.25 + fraction + index as f32 * 18.5,
                    y + 3.25,
                    gpui::hsla(0.12, 0.6, 0.65, 0.85),
                );
            }
            if hover == Some(row) {
                for index in 0..7 + row % 3 {
                    paint_glyph(
                        &mut scene,
                        282.25 + fraction + index as f32 * 7.0,
                        y + 15.0,
                        gpui::hsla(0.6, 0.4, 0.6, 0.6),
                    );
                }
            }
        }
        scene.insert_primitive(Quad {
            bounds: scaled(250.25 + fraction, 14.5, 8.5, 832.25),
            content_mask: mask,
            background: gpui::rgba(0xda624d60).into(),
            ..Default::default()
        });
        scene.finish();
        scene
    }

    #[test]
    fn retained_tiled_history_bounds_tree_matches_exact_rgba_for_rank_permutations()
    -> anyhow::Result<()> {
        let _gpu_test_guard = crate::test_gpu::guard();
        let context = WgpuContext::new_headless(None)?;
        for mode in [GlyphRenderMode::Grayscale, GlyphRenderMode::Subpixel] {
            for (transparent, fraction, clipped) in
                [(false, 0.0, false), (true, 0.25, false), (true, 0.75, true)]
            {
                let target = size(DevicePixels(512), DevicePixels(1024));
                let mut full = WgpuRenderer::new_headless(&context, target)?;
                let mut retained = WgpuRenderer::new_headless(&context, target)?;
                full.retained.disable_for_tests();
                retained.retained.enable_for_tests();
                full.update_transparency(transparent);
                retained.update_transparency(transparent);
                full.timings.clipped_quads = clipped;
                retained.timings.clipped_quads = clipped;
                full.timings.diagnostic_ablations = Default::default();
                retained.timings.diagnostic_ablations = Default::default();
                let glyph = retained_tiled_glyph(&full, mode)?;
                assert_eq!(glyph, retained_tiled_glyph(&retained, mode)?);
                let first = tiled_history_scene(Some(0), fraction, glyph, mode);
                let last = tiled_history_scene(Some(19), fraction, glyph, mode);
                let first_snapshot =
                    gpui_render::damage::SceneSnapshot::capture(&first, 1, 512, 1024).unwrap();
                let last_snapshot =
                    gpui_render::damage::SceneSnapshot::capture(&last, 1, 512, 1024).unwrap();
                assert!(
                    last_snapshot.compare_checked(&first_snapshot).is_err(),
                    "fixture must exercise strict global rank rejection"
                );
                assert!(
                    matches!(
                        last_snapshot.compare_tiled_checked(&first_snapshot),
                        Ok(gpui_render::damage::Damage::Partial(_))
                    ),
                    "fixture must qualify by exact tile-local painter order"
                );
                let mut previous = None;
                let mut revision = 0;
                for (index, row) in [
                    None,
                    None,
                    Some(0),
                    Some(0),
                    Some(19),
                    Some(18),
                    Some(1),
                    Some(1),
                    None,
                    None,
                ]
                .into_iter()
                .enumerate()
                {
                    if index == 0 || row != previous {
                        revision += 1;
                    }
                    let scene = tiled_history_scene(row, fraction, glyph, mode);
                    assert_retained_rgba(
                        &mut full,
                        &mut retained,
                        &scene,
                        revision,
                        if index == 0 {
                            "full"
                        } else if row == previous {
                            "reuse"
                        } else {
                            "partial"
                        },
                        &format!(
                            "tiled history mode={mode:?}, transparent={transparent}, fraction={fraction}, clipped={clipped}, step={index}, row={row:?}"
                        ),
                    )?;
                    previous = row;
                }

                // Identical overlapping descriptors around a third translucent
                // paint must change pixels, rather than being mistaken for Reuse.
                let mask = ContentMask {
                    bounds: scaled(0.0, 0.0, 512.0, 1024.0),
                    ..Default::default()
                };
                let a = Quad {
                    bounds: scaled(62.25, 62.25, 24.5, 24.5),
                    content_mask: mask,
                    background: gpui::rgba(0xff000080).into(),
                    ..Default::default()
                };
                let b = Quad {
                    background: gpui::rgba(0x0000ff80).into(),
                    ..a
                };
                let mut overlap = Scene::default();
                for quad in [a, a, b] {
                    overlap.insert_primitive(quad);
                }
                overlap.finish();
                // Recreate targets to ensure this independent order proof starts
                // with an initial Full rather than an unrelated previous scene.
                let queue = retained.resources().queue.clone();
                retained.retained.invalidate(&queue);
                revision += 1;
                let original = assert_retained_rgba(
                    &mut full,
                    &mut retained,
                    &overlap,
                    revision,
                    "full",
                    "tiled overlap initial",
                )?;
                overlap.quads.swap(1, 2); // Counts/pipelines stay unchanged.
                revision += 1;
                let changed = assert_retained_rgba(
                    &mut full,
                    &mut retained,
                    &overlap,
                    revision,
                    "partial",
                    "tiled overlap reordered",
                )?;
                assert_ne!(
                    original.as_raw(),
                    changed.as_raw(),
                    "overlapping painter reorder must visibly change RGBA"
                );
                assert_retained_rgba(
                    &mut full,
                    &mut retained,
                    &overlap,
                    revision,
                    "reuse",
                    "tiled overlap repeated",
                )?;
            }
        }
        Ok(())
    }

    fn transparent_composite_hover_scene(
        row: Option<usize>,
        fraction: f32,
        smoothing: f32,
        backing_alpha: f32,
        glyph: AtlasTile,
        badge_before_graph: Option<bool>,
    ) -> Scene {
        let mut scene = Scene::default();
        let backing_bounds = scaled(4.25 + fraction, 3.5 + fraction, 190.5, 90.25);
        let radii = gpui::Corners::all(ScaledPixels(9.25));
        scene.insert_primitive(Shadow {
            order: 0,
            blur_radius: ScaledPixels(2.5),
            bounds: backing_bounds,
            corner_radii: radii,
            content_mask: full_mask(),
            color: gpui::rgba(0x17243850).into(),
            element_bounds: backing_bounds,
            element_corner_radii: radii,
            inset: false.into(),
            corner_smoothing: smoothing,
        });
        scene.raise_order_floor();
        scene.insert_primitive(Quad {
            bounds: backing_bounds,
            content_mask: full_mask(),
            background: solid_background(gpui::hsla(0.62, 0.3, 0.35, backing_alpha)),
            corner_radii: radii,
            corner_smoothing: smoothing,
            ..Default::default()
        });
        scene.raise_order_floor();
        // A shifted/shrunk inset hole is far from the first hovered row. Its
        // actual raster geometry is the whole element, so pruning by the hole
        // bounds would lose static inset pixels when that row is damage-cleared.
        scene.insert_primitive(Shadow {
            order: 0,
            blur_radius: ScaledPixels(0.5),
            bounds: scaled(140.25 + fraction, 62.5, 26.5, 14.0),
            corner_radii: gpui::Corners::all(ScaledPixels(2.0)),
            content_mask: full_mask(),
            color: gpui::rgba(0x10172870).into(),
            element_bounds: backing_bounds,
            element_corner_radii: radii,
            inset: true.into(),
            corner_smoothing: smoothing,
        });
        scene.raise_order_floor();
        if let Some(row) = row {
            scene.insert_primitive(Quad {
                bounds: scaled(
                    10.25 + fraction,
                    12.5 + row as f32 * 24.0,
                    165.5 - row as f32 * 0.75,
                    13.25,
                ),
                content_mask: ContentMask {
                    bounds: scaled(
                        11.5 + fraction,
                        12.75 + row as f32 * 24.0,
                        162.5 - row as f32 * 0.25,
                        12.5,
                    ),
                    fade_out: gpui::Edges {
                        left: ScaledPixels(1.25),
                        right: ScaledPixels(1.75),
                        ..Default::default()
                    },
                },
                background: gpui::rgba(0x4f98df70).into(),
                corner_radii: gpui::Corners::all(ScaledPixels(3.5 + row as f32 * 0.25)),
                corner_smoothing: smoothing,
                ..Default::default()
            });
        }
        let paint_badge = |scene: &mut Scene| {
            if let Some(row) = row {
                // The live history row adds an entire lane-name badge only while
                // hovered. Variable run lengths and distant row moves must repaint
                // both the old and new badge without losing the static text anchor.
                for index in 0..7 + row {
                    scene.insert_primitive(MonochromeSprite {
                        order: 0,
                        padding: 0,
                        bounds: scaled(
                            112.25 + fraction + index as f32 * 6.5,
                            14.5 + row as f32 * 24.0,
                            6.5,
                            10.75,
                        ),
                        content_mask: full_mask(),
                        color: gpui::hsla(0.6, 0.4, 0.6, 0.6).into(),
                        tile: glyph,
                        transformation: Default::default(),
                    });
                }
            }
        };
        scene.raise_order_floor();
        if badge_before_graph == Some(true) {
            paint_badge(&mut scene);
            scene.raise_order_floor();
        }
        scene.insert_primitive(triangle(
            17.25 + fraction,
            10.5,
            17.5,
            ContentMask {
                bounds: scaled(19.5 + fraction, 13.25, 13.75, 12.5),
                fade_out: gpui::Edges::all(ScaledPixels(0.75)),
            },
        ));
        scene.raise_order_floor();
        if badge_before_graph == Some(false) {
            paint_badge(&mut scene);
        }
        scene.raise_order_floor();
        scene.insert_primitive(MonochromeSprite {
            order: 0,
            padding: 0,
            bounds: scaled(44.25 + fraction, 14.5, 12.5, 10.75),
            content_mask: ContentMask {
                bounds: scaled(45.0 + fraction, 14.25, 11.5, 11.5),
                fade_out: gpui::Edges::all(ScaledPixels(0.75)),
            },
            color: gpui::hsla(0.12, 0.8, 0.65, 0.85).into(),
            tile: glyph,
            transformation: Default::default(),
        });
        scene.raise_order_floor();
        scene.insert_primitive(Quad {
            bounds: scaled(80.25 + fraction, 6.5, 16.5, 78.25),
            content_mask: full_mask(),
            background: gpui::rgba(0xda624d60).into(),
            corner_radii: gpui::Corners::all(ScaledPixels(4.25)),
            corner_smoothing: smoothing,
            ..Default::default()
        });
        scene.finish();
        scene
    }

    #[test]
    fn retained_transparent_composite_hover_matches_exact_rgba_and_invalidates_appearance()
    -> anyhow::Result<()> {
        let _gpu_test_guard = crate::test_gpu::guard();
        let context = WgpuContext::new_headless(None)?;
        for (crop_paths, badge_before_graph) in [
            (false, None),
            (true, None),
            (false, Some(false)),
            (true, Some(false)),
            (false, Some(true)),
            (true, Some(true)),
        ] {
            for (fraction, smoothing, backing_alpha) in [
                (0.0, 0.0, 0.45),
                (0.25, 0.65, 1.0),
                (0.5, 0.0, 1.0),
                (0.75, 0.65, 0.45),
            ] {
                let target = size(DevicePixels(200), DevicePixels(100));
                let mut full = WgpuRenderer::new_headless(&context, target)?;
                let mut retained = WgpuRenderer::new_headless(&context, target)?;
                full.retained.disable_for_tests();
                retained.retained.enable_for_tests();
                full.update_transparency(true);
                retained.update_transparency(true);
                full.crop_path_targets = crop_paths;
                retained.crop_path_targets = crop_paths;
                full.timings.diagnostic_ablations = Default::default();
                retained.timings.diagnostic_ablations = Default::default();
                let glyph = retained_glyph_tile(&full)?;
                assert_eq!(glyph, retained_glyph_tile(&retained)?);
                let label = format!(
                    "fraction={fraction}, smoothing={smoothing}, backing={backing_alpha}, crop={crop_paths}, badge_before_graph={badge_before_graph:?}"
                );
                let mut revision = 0;
                let mut previous = None;
                let resting = transparent_composite_hover_scene(
                    None,
                    fraction,
                    smoothing,
                    backing_alpha,
                    glyph,
                    badge_before_graph,
                );
                let mut resting_pixels = None;
                for (index, row) in [
                    None,
                    None,
                    Some(0),
                    Some(0),
                    Some(2),
                    Some(0),
                    Some(1),
                    Some(2),
                    None,
                    None,
                ]
                .into_iter()
                .enumerate()
                {
                    if index == 0 || row != previous {
                        revision += 1;
                    }
                    let scene = transparent_composite_hover_scene(
                        row,
                        fraction,
                        smoothing,
                        backing_alpha,
                        glyph,
                        badge_before_graph,
                    );
                    if badge_before_graph == Some(true)
                        && let Some(row) = row
                    {
                        assert!(scene.render_commands().windows(2).any(|commands|
                            matches!((&commands[0], &commands[1]),
                                (gpui::RenderCommand::Batch(gpui::PrimitiveBatch::Quads { range: quad, .. }),
                                 gpui::RenderCommand::Batch(gpui::PrimitiveBatch::MonochromeSprites { range: glyphs, .. }))
                                if quad.len() == 1 && glyphs.len() == 7 + row)),
                            "fixture must prove fill directly precedes the changed badge: {label}");
                    }
                    let image = assert_retained_rgba(
                        &mut full,
                        &mut retained,
                        &scene,
                        revision,
                        if index == 0 {
                            "full"
                        } else if row == previous {
                            "reuse"
                        } else {
                            "partial"
                        },
                        &format!("{label}, step={index}, row={row:?}"),
                    )?;
                    assert!(image.pixels().any(|pixel| pixel[3] > 0 && pixel[3] < 255));
                    assert!(
                        image.get_pixel(0, 0)[3] < 255,
                        "rounded outside corner: {label}"
                    );
                    if backing_alpha == 1.0 {
                        assert_eq!(image.get_pixel(150, 85)[3], 255, "opaque interior: {label}");
                    }
                    if index == 0 {
                        resting_pixels = Some(image.clone());
                    } else if row.is_none() {
                        assert_eq!(
                            Some(&image),
                            resting_pixels.as_ref(),
                            "erase restores every byte: {label}"
                        );
                    } else {
                        assert_ne!(
                            Some(&image),
                            resting_pixels.as_ref(),
                            "hover must visibly alter the fixture: {label}"
                        );
                    }
                    previous = row;
                }
                // Invalidate independently of scene revision: resize, alpha mode,
                // clear color, and atlas generation must never reuse stale pixels.
                let resized = size(DevicePixels(193), DevicePixels(97));
                full.update_drawable_size(resized);
                retained.update_drawable_size(resized);
                assert_retained_rgba(
                    &mut full,
                    &mut retained,
                    &resting,
                    revision,
                    "full",
                    &format!("resize: {label}"),
                )?;
                assert_retained_rgba(
                    &mut full,
                    &mut retained,
                    &resting,
                    revision,
                    "reuse",
                    &format!("resized reuse: {label}"),
                )?;
                for transparent in [false, true] {
                    full.update_transparency(transparent);
                    retained.update_transparency(transparent);
                    assert_retained_rgba(
                        &mut full,
                        &mut retained,
                        &resting,
                        revision,
                        "full",
                        &format!("alpha transition to {transparent}: {label}"),
                    )?;
                    assert_retained_rgba(
                        &mut full,
                        &mut retained,
                        &resting,
                        revision,
                        "reuse",
                        &format!("alpha reuse {transparent}: {label}"),
                    )?;
                }
                // A direct clear-color change must be detected by the retained
                // appearance key even when no public alpha toggle fires.
                full.target.clear_color = wgpu::Color {
                    r: 0.04,
                    g: 0.03,
                    b: 0.02,
                    a: 0.25,
                };
                retained.target.clear_color = full.target.clear_color;
                assert_retained_rgba(
                    &mut full,
                    &mut retained,
                    &resting,
                    revision,
                    "full",
                    &format!("clear-color transition: {label}"),
                )?;
                assert_retained_rgba(
                    &mut full,
                    &mut retained,
                    &resting,
                    revision,
                    "full",
                    &format!("unsupported clear-color fallback: {label}"),
                )?;
                // Restore an exact endpoint so atlas invalidation still tests
                // a qualified retained target rather than the unsupported clear.
                full.target.clear_color = wgpu::Color::TRANSPARENT;
                retained.target.clear_color = full.target.clear_color;
                full.atlas.clear();
                retained.atlas.clear();
                let glyph = retained_glyph_tile(&full)?;
                assert_eq!(glyph, retained_glyph_tile(&retained)?);
                let scene = transparent_composite_hover_scene(
                    None,
                    fraction,
                    smoothing,
                    backing_alpha,
                    glyph,
                    badge_before_graph,
                );
                assert_retained_rgba(
                    &mut full,
                    &mut retained,
                    &scene,
                    revision,
                    "full",
                    &format!("atlas invalidation: {label}"),
                )?;
                assert_retained_rgba(
                    &mut full,
                    &mut retained,
                    &scene,
                    revision,
                    "reuse",
                    &format!("atlas reuse: {label}"),
                )?;
            }
        }
        Ok(())
    }

    #[test]
    fn cropped_path_targets_match_full_viewport_pixels_when_growing_and_resizing() {
        let _gpu_test_guard = crate::test_gpu::guard();
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
        full.crop_path_targets = false;
        cropped.crop_path_targets = true;
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
            assert!(texture.height() < target.height.0 as u32);
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
}
