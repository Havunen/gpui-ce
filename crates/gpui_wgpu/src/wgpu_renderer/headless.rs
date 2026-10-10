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
                "transparent alpha uses full rendering"
            );
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
