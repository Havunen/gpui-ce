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
    use gpui::{Bounds, ContentMask, PathBuilder, Quad, ScaledPixels, point, px, size};

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
    fn failed_frames_discard_unsubmitted_path_captures_and_report_failure() {
        use gpui_render::optimization_fixture as fixture;
        let context = WgpuContext::new_headless(None).unwrap();
        for packed in [false, true] {
            let viewport = fixture::viewport(0);
            let mut candidate = WgpuRenderer::new_headless(&context, viewport).unwrap();
            candidate.options.cached_layers = true;
            candidate.options.batched_paths = packed;
            let mut failed = fixture::scene(0);
            let bounds = Bounds::new(
                point(ScaledPixels(0.), ScaledPixels(0.)),
                size(ScaledPixels(40.), ScaledPixels(40.)),
            );
            failed.insert_primitive(gpui::PaintSurface {
                order: 1000,
                bounds,
                content_mask: gpui::ContentMask {
                    bounds,
                    ..Default::default()
                },
                source: gpui::SurfaceSource::Unsupported(size(DevicePixels(40), DevicePixels(40))),
            });
            failed.surfaces[0].order = 1000; // Fail after paths have encoded their captures.
            failed.finish();
            assert!(candidate.render_to_image(&failed).is_err());
            assert_eq!(candidate.resources().path_cache.borrow().bytes(), 0);
            if gpui::gpu_profiler::enabled() {
                let frames = gpui::gpu_profiler::GpuFrameCollector::default().collect_unseen();
                assert!(frames.iter().any(|f| f.renderer_id == candidate.renderer_id
                    && matches!(f.status, "render_failed" | "unsupported")
                    && f.gpu_duration_ns.is_none()));
            }
            let mut reference = WgpuRenderer::new_headless(&context, viewport).unwrap();
            reference.options = Default::default();
            let scene = fixture::scene(0);
            let expected = reference.render_to_image(&scene).unwrap();
            for _ in 0..3 {
                let actual = candidate.render_to_image(&scene).unwrap();
                assert!(fixture::max_difference(actual.as_raw(), expected.as_raw()) <= 1);
            }
        }
    }

    #[test]
    fn native_gpu_fixture_covers_cached_packed_and_retained_rendering() {
        use gpui_render::{gpu_policy, optimization_fixture as fixture};
        let context = WgpuContext::new_headless(None).unwrap();
        for transparent in [false, true] {
            for options in fixture::modes() {
                let mut full = WgpuRenderer::new_headless(&context, fixture::viewport(0)).unwrap();
                let mut candidate =
                    WgpuRenderer::new_headless(&context, fixture::viewport(0)).unwrap();
                full.options = Default::default();
                candidate.options = options;
                full.update_transparency(transparent);
                candidate.update_transparency(transparent);
                for frame in 0..12 {
                    let scene = fixture::scene(frame);
                    full.update_drawable_size(fixture::viewport(frame));
                    candidate.update_drawable_size(fixture::viewport(frame));
                    let expected = full.render_to_image(&scene).unwrap();
                    fixture::assert_paths_painted(expected.as_raw(), expected.width(), frame);
                    for repeat in 0..2 {
                        let actual = candidate.render_to_image(&scene).unwrap();
                        let difference =
                            fixture::max_difference(actual.as_raw(), expected.as_raw());
                        assert!(
                            difference <= 1,
                            "{options:?}, transparent={transparent}, frame={frame}, repeat={repeat}, difference={difference}"
                        );
                        let resources = candidate.resources();
                        let cache = resources.path_cache.borrow();
                        // Frame 4 repeats every path of frame 3; frame 5 recolors one.
                        if options.cached_layers && !options.partial_redraw && repeat == 0 {
                            if frame == 4 {
                                assert!(cache.hits >= 4, "{options:?}");
                            }
                            if frame == 5 {
                                assert!(cache.misses >= 1, "{options:?}");
                            }
                        }
                        assert!(cache.bytes() <= gpu_policy::WINDOW_LAYER_BYTES);
                        assert!(
                            resources.retention.budget().used()
                                <= gpu_policy::DEVICE_RETENTION_BYTES
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cropped_path_targets_match_full_viewport_pixels() {
        use gpui_render::optimization_fixture as fixture;
        let context = WgpuContext::new_headless(None).expect("hardware or software GPU");
        let (_, _, viewport) = fixture::CROPPED_PATH_CASES[0];
        let mut full = WgpuRenderer::new_headless(&context, viewport).unwrap();
        let mut cropped = WgpuRenderer::new_headless(&context, viewport).unwrap();
        full.options.cropped_paths = false;
        cropped.options.cropped_paths = true;
        for (x, y, viewport) in fixture::CROPPED_PATH_CASES {
            full.update_drawable_size(viewport);
            cropped.update_drawable_size(viewport);
            let scene = fixture::cropped_path_scene(x, y);
            let expected = full.render_to_image(&scene).expect("full target frame");
            let actual = cropped
                .render_to_image(&scene)
                .expect("cropped target frame");
            fixture::assert_cropped_paths_painted(actual.as_raw(), actual.width(), x, y);
            let difference = fixture::max_difference(actual.as_raw(), expected.as_raw());
            assert!(
                difference <= 1,
                "cropped rendering changed a pixel by {difference}"
            );
            let resources = cropped.resources();
            let target = resources.path_intermediate_texture.as_ref().unwrap();
            assert!(
                target.width() < viewport.width.0 as u32,
                "narrow scene must keep a narrow target"
            );
            assert!(
                target.height() <= viewport.height.0 as u32,
                "height reservation stays within the viewport"
            );
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
                let difference = gpui_render::optimization_fixture::max_difference(
                    actual.as_raw(),
                    expected.as_raw(),
                );
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
