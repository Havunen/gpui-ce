use super::{
    WgpuRenderer, begin_color_render_pass,
    buffers::{InstanceTransport, InstanceUpload},
    filters::{FILTER_UNIFORMS_PER_COMPOSITE, FrameUniformRequirements},
    path_types,
};
use gpui::{
    FilterRenderTarget, MAX_FILTER_GROUP_DEPTH, MonochromeSprite, PolychromeSprite, PrimitiveBatch,
    Quad, RenderCommand, Scene, Shadow, SubpixelSprite, Underline,
};
use gpui_render::blur::{FilterCompositeClip, FilterCompositeParameters};
use gpui_render::damage::{Damage, DamageRect, batch_intersects, clear_quad};
use gpui_render::shaders::{
    common::{FontRasterizationUniforms, GlobalUniforms, ShaderBool},
    interface as shader_interface,
};

pub(super) fn render_to_view(
    renderer: &mut WgpuRenderer,
    scene: &Scene,
    frame_view: &wgpu::TextureView,
    frame_texture: Option<&wgpu::Texture>,
    readback: Option<ReadbackCopy<'_>>,
) -> Option<wgpu::SubmissionIndex> {
    let mut targets = PreparedTargets::prepare(renderer, scene, frame_view)?;
    let device = renderer.resources().device.clone();
    let queue = renderer.resources().queue.clone();
    let atlas_generation = if renderer.retained.enabled() {
        gpui::PlatformAtlas::generation(renderer.atlas.as_ref())
    } else {
        0
    };
    let retained = renderer.retained.prepare(
        scene,
        atlas_generation,
        renderer.target.width(),
        renderer.target.height(),
        renderer.target.retained_copy_enabled(),
        &device,
        &queue,
        renderer.target.format(),
        renderer.target.configuration().alpha_mode,
        renderer.target.clear_color(),
    );
    let retained_copy = if renderer.target.retained_copy_enabled() && targets.offscreen.is_none() {
        retained
            .copy_texture
            .as_ref()
            .zip(frame_texture)
            .filter(|(source, destination)| {
                super::retained_copy::compatible(
                    source,
                    destination,
                    renderer.target.format(),
                    renderer.target.width(),
                    renderer.target.height(),
                    renderer.target.alpha_mode(),
                )
            })
            .map(|(source, destination)| (source.clone(), destination.clone()))
    } else {
        None
    };
    if let Some(view) = &retained.view {
        targets.active = view.clone();
        // The sampled presentation fallback needs a BlurUniforms slot even
        // without filters. A texture copy has no shader-uniform upload.
        if retained_copy.is_none() {
            let mut uniforms =
                FrameRequirements::for_scene(scene, renderer.resources().instances.transport())
                    .uniforms;
            uniforms.filter_count = uniforms.filter_count.saturating_add(1);
            if !renderer.ensure_uniform_capacity(uniforms) {
                renderer.retained.invalidate(&queue);
                return None;
            }
        }
    }
    // Create the overwrite-clear PSO while a qualified retained Full frame is
    // being initialized. Its first use must not compile a pipeline on hover.
    // Ineligible scenes and exhausted retained budgets have no retained view.
    if retained.view.is_some() && retained.damage == Damage::Full {
        renderer
            .resources_mut()
            .pipelines
            .ensure_damage_clear(&device);
    }
    // Partial regions replay batches independently. Reserve bounded duplicate uploads and clear quads.
    if let Damage::Partial(regions) = &retained.damage {
        let device = renderer.resources().device.clone();
        renderer
            .resources_mut()
            .pipelines
            .ensure_damage_clear(&device);
        let transport = renderer.resources().instances.transport();
        let requirements = FrameRequirements::for_partial_scene(scene, regions, transport);
        let bytes = requirements.storage_bytes;
        let batches = requirements.instance_batches;
        let device = renderer.resources().device.clone();
        let resources = renderer.resources_mut();
        if !resources.instances.ensure_capacity(
            &device,
            &resources.bind_group_layouts,
            bytes,
            batches,
        ) {
            return None;
        }
        let resources = renderer.resources();
        let instances = resources
            .instances
            .begin_upload(&resources.queue, bytes, batches)?;
        targets.instances = instances;
    }
    // Prune once against the complete scene. Doing this inside each draw batch would evict a
    // texture that reappears after an intervening primitive and recreate its platform view.
    renderer.retain_surface_cache(&scene.surfaces);

    let retained_blit = retained.view.is_some() && retained_copy.is_none();
    #[cfg(test)]
    {
        renderer.last_retained_presentation = if retained_copy.is_some() {
            "copy"
        } else if retained_blit {
            "blit"
        } else {
            "direct"
        };
    }
    let damage_kind = retained.damage.kind();
    let result = FrameEncoder::new(
        renderer,
        scene,
        targets,
        retained.damage,
        retained_blit,
        retained_copy,
    )
    .encode(readback);
    renderer.retained.complete(
        retained.snapshot,
        retained.scene_revision,
        retained.atlas_generation,
        damage_kind,
        result.is_ok(),
        &queue,
    );
    match result {
        Ok(command_buffer) => {
            let submission = renderer.resources().queue.submit([command_buffer]);
            // A cleanup skipped a frame. Only a real submission without a new validation
            // error can reset that retry budget; merely starting the next frame cannot.
            if renderer.faults.cleaned_up_failed_frames
                && renderer.faults.pending_error.lock().unwrap().is_none()
            {
                renderer.faults.consecutive_failed_frames = 0;
                renderer.faults.cleaned_up_failed_frames = false;
            }
            Some(submission)
        }
        Err(DrawError::ExternalSurface) => None,
        Err(DrawError::CapacityPlanningInvariant) => {
            log::error!("frame storage exceeded its precomputed capacity");
            None
        }
        Err(DrawError::MissingIntermediateTarget) => {
            log::error!("frame preparation did not create a required intermediate target");
            None
        }
    }
}

pub(super) struct ReadbackCopy<'a> {
    pub(super) texture: &'a wgpu::Texture,
    pub(super) buffer: &'a wgpu::Buffer,
    pub(super) bytes_per_row: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

struct PreparedTargets {
    active: wgpu::TextureView,
    presentation: wgpu::TextureView,
    offscreen: Option<wgpu::TextureView>,
    instances: InstanceUpload,
}

impl PreparedTargets {
    fn prepare(
        renderer: &mut WgpuRenderer,
        scene: &Scene,
        frame_view: &wgpu::TextureView,
    ) -> Option<Self> {
        if !begin_frame(renderer) {
            return None;
        }
        let transport = renderer.resources().instances.transport();
        let requirements = FrameRequirements::for_scene(scene, transport);
        let device = renderer.resources().device.clone();
        let resources = renderer.resources_mut();
        if !resources.instances.ensure_capacity(
            &device,
            &resources.bind_group_layouts,
            requirements.storage_bytes,
            requirements.instance_batches,
        ) {
            return None;
        }
        let instances = {
            let resources = renderer.resources();
            resources.instances.begin_upload(
                &resources.queue,
                requirements.storage_bytes,
                requirements.instance_batches,
            )?
        };
        if !renderer.ensure_uniform_capacity(requirements.uniforms) {
            return None;
        }

        if requirements.uses_path_target {
            renderer.ensure_path_textures(scene);
        }
        if requirements.uses_offscreen_target {
            renderer.ensure_filter_textures(requirements.isolated_target_count);
        }
        write_shader_globals(renderer);

        if requirements.uses_offscreen_target {
            let resources = renderer.resources();
            let offscreen = resources
                .scene_color_view
                .as_ref()
                .expect("blur texture preparation must create a scene target")
                .clone();
            Some(Self {
                active: offscreen.clone(),
                presentation: frame_view.clone(),
                offscreen: Some(offscreen),
                instances,
            })
        } else {
            Some(Self {
                active: frame_view.clone(),
                presentation: frame_view.clone(),
                offscreen: None,
                instances,
            })
        }
    }
}

fn begin_frame(renderer: &mut WgpuRenderer) -> bool {
    // Release the callback mutex before atlas uploads can report another GPU error.
    let pending_error = renderer.faults.pending_error.lock().unwrap().take();
    let Some(error) = pending_error else {
        if !renderer.faults.cleaned_up_failed_frames {
            renderer.faults.consecutive_failed_frames = 0;
        }
        renderer.atlas.before_frame();
        return true;
    };

    handle_gpu_error(renderer, error)
}

pub(super) fn handle_gpu_error(renderer: &mut WgpuRenderer, error: String) -> bool {
    renderer.release_retained_frame();
    renderer.faults.consecutive_failed_frames += 1;
    log::error!(
        "GPU error during frame (failure {} of 10): {error}",
        renderer.faults.consecutive_failed_frames
    );
    if renderer.faults.consecutive_failed_frames > 10 {
        if renderer.reports_terminal_errors() {
            renderer.faults.terminal_error = Some(format!(
                "too many consecutive GPU errors; last error: {error}"
            ));
            return false;
        }
        panic!("too many consecutive GPU errors; last error: {error}");
    }
    if renderer.faults.consecutive_failed_frames > 5 && !renderer.faults.cleaned_up_failed_frames {
        if let Some(resources) = renderer.resources.as_mut() {
            resources.invalidate_intermediate_textures();
        }
        renderer.atlas.clear();
        renderer.target.request_redraw();
        renderer.faults.cleaned_up_failed_frames = true;
        if !renderer.reports_terminal_errors() {
            // Preserve the existing retry policy for other platform integrations.
            renderer.faults.consecutive_failed_frames = 0;
            renderer.faults.cleaned_up_failed_frames = false;
        }
        return false;
    }

    renderer.atlas.before_frame();
    true
}

#[cfg(all(test, feature = "test-support", target_os = "windows"))]
mod failure_tests {
    use super::*;

    #[test]
    fn persistent_dx12_errors_become_terminal_and_a_healthy_frame_resets_the_count() {
        let context = crate::WgpuContext::new_headless(None).unwrap();
        let mut renderer = WgpuRenderer::new_headless(
            &context,
            gpui::size(gpui::DevicePixels(16), gpui::DevicePixels(16)),
        )
        .unwrap();
        for _ in 0..18 {
            handle_gpu_error(&mut renderer, "injected validation failure".into());
        }
        assert_eq!(renderer.faults.consecutive_failed_frames, 0);
        assert!(renderer.terminal_error().is_none());
        renderer.native_backend = Some(crate::NativeBackend::Dx12);
        for _ in 0..6 {
            handle_gpu_error(&mut renderer, "injected validation failure".into());
        }
        assert!(renderer.faults.cleaned_up_failed_frames);
        assert!(begin_frame(&mut renderer));
        assert_eq!(renderer.faults.consecutive_failed_frames, 6);
        let mut scene = Scene::default();
        scene.finish();
        renderer.render_to_image(&scene).unwrap();
        assert_eq!(renderer.faults.consecutive_failed_frames, 0);
        for _ in 0..10 {
            handle_gpu_error(&mut renderer, "injected validation failure".into());
        }
        assert!(renderer.terminal_error().is_none());
        assert!(!handle_gpu_error(
            &mut renderer,
            "injected validation failure".into()
        ));
        assert!(
            renderer
                .terminal_error()
                .unwrap()
                .contains("validation failure")
        );
        // The error can be consumed without holding a callback/GPU mutex across application notification.
        assert!(renderer.faults.fatal_error.try_lock().is_ok());
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(super) struct GlobalUniformState {
    globals: GlobalUniforms,
    path_globals: GlobalUniforms,
    font_rasterization: FontRasterizationUniforms,
}

fn write_shader_globals(renderer: &mut WgpuRenderer) {
    let font = renderer.rendering_params.font_rasterization;
    let font_rasterization = FontRasterizationUniforms {
        gamma_ratios: wgsl_rs::std::vec4f(
            font.gamma_ratios[0],
            font.gamma_ratios[1],
            font.gamma_ratios[2],
            font.gamma_ratios[3],
        ),
        grayscale_enhanced_contrast: font.grayscale_enhanced_contrast,
        subpixel_enhanced_contrast: font.subpixel_enhanced_contrast,
        uses_blue_green_red_subpixel_order: ShaderBool::from(
            renderer.subpixel_order == super::SubpixelOrder::BlueGreenRed,
        ),
        padding: 0,
    };
    let globals = GlobalUniforms {
        viewport_size: wgsl_rs::std::vec2f(
            renderer.target.width() as f32,
            renderer.target.height() as f32,
        ),
        premultiplied_alpha: ShaderBool::from(
            renderer.target.alpha_mode() == wgpu::CompositeAlphaMode::PreMultiplied,
        ),
        padding: 0,
    };
    let path_globals = GlobalUniforms {
        viewport_size: renderer
            .resources()
            .path_intermediate_texture
            .as_ref()
            .map(|texture| wgsl_rs::std::vec2f(texture.width() as f32, texture.height() as f32))
            .unwrap_or(globals.viewport_size),
        premultiplied_alpha: ShaderBool::Disabled,
        ..globals
    };
    let state = GlobalUniformState {
        globals,
        path_globals,
        font_rasterization,
    };
    if renderer.uploaded_globals == Some(state) {
        return;
    }

    let resources = renderer.resources();
    let globals_size = std::mem::size_of::<GlobalUniforms>();
    let font_size = std::mem::size_of::<FontRasterizationUniforms>();
    let upload_size = renderer.globals.font_offset + font_size as u64;
    let mut upload = resources
        .queue
        .write_buffer_with(
            &resources.globals_buffer,
            0,
            std::num::NonZeroU64::new(upload_size).expect("global uniforms are non-empty"),
        )
        .expect("global uniform upload must fit its buffer");
    upload.slice(..).fill(0);
    upload
        .slice(..globals_size)
        .copy_from_slice(shader_interface::bytes_of(&globals));
    upload
        .slice(
            renderer.globals.path_offset as usize
                ..renderer.globals.path_offset as usize + globals_size,
        )
        .copy_from_slice(shader_interface::bytes_of(&path_globals));
    upload
        .slice(
            renderer.globals.font_offset as usize
                ..renderer.globals.font_offset as usize + font_size,
        )
        .copy_from_slice(shader_interface::bytes_of(&font_rasterization));
    drop(upload);
    renderer.uploaded_globals = Some(state);
}

#[derive(Clone, Copy, Default)]
pub(super) struct FrameRequirements {
    storage_bytes: u64,
    /// Instance batches this frame; one downlevel range-uniform slot per batch.
    instance_batches: u64,
    pub(super) uniforms: FrameUniformRequirements,
    isolated_target_count: usize,
    uses_path_target: bool,
    uses_offscreen_target: bool,
}

impl FrameRequirements {
    pub(super) fn for_scene(scene: &Scene, transport: InstanceTransport) -> Self {
        let planned = scene.render_plan().requirements();
        let mut requirements = Self::scene_metadata(scene);

        for command in scene.render_commands() {
            let RenderCommand::Batch(batch) = command else {
                continue;
            };
            requirements.reserve_batch(batch, transport);
        }
        debug_assert_eq!(
            requirements.instance_batches as usize,
            planned.instance_batch_count
        );

        requirements
    }

    fn scene_metadata(scene: &Scene) -> Self {
        let planned = scene.render_plan().requirements();
        Self {
            uniforms: FrameUniformRequirements {
                filter_count: FILTER_UNIFORMS_PER_COMPOSITE
                    * (planned.backdrop_filter_count + planned.isolated_filter_count) as u64
                    + u64::from(planned.uses_offscreen_target),
                surface_count: planned.surface_count as u64,
            },
            isolated_target_count: planned.isolated_target_count,
            uses_path_target: planned.uses_path_target,
            uses_offscreen_target: planned.uses_offscreen_target,
            ..Default::default()
        }
    }

    fn for_partial_scene(
        scene: &Scene,
        regions: &[DamageRect],
        transport: InstanceTransport,
    ) -> Self {
        let mut requirements = Self::scene_metadata(scene);
        // Encoding prepends a clear and skips nonintersecting batches. Either
        // change alters the alignment of every following type, and regions share
        // one arena cursor. An origin-zero full-frame estimate cannot be scaled.
        for &region in regions {
            requirements.reserve(std::mem::size_of::<Quad>(), 1, transport);
            for command in scene.render_commands() {
                if let RenderCommand::Batch(batch) = command
                    && batch_intersects(scene, batch, region)
                {
                    requirements.reserve_batch(batch, transport);
                }
            }
        }
        requirements
    }

    fn reserve(&mut self, element_size: usize, count: usize, transport: InstanceTransport) {
        if count > 0 {
            let stride = element_size as u64;
            self.storage_bytes = self
                .storage_bytes
                .next_multiple_of(transport.batch_alignment(stride));
            self.storage_bytes = self
                .storage_bytes
                .saturating_add(stride.saturating_mul(count as u64));
            self.instance_batches += 1;
        }
    }

    fn reserve_batch(&mut self, batch: &PrimitiveBatch, transport: InstanceTransport) {
        match batch {
            PrimitiveBatch::Shadows { range, .. } => {
                self.reserve(std::mem::size_of::<Shadow>(), range.len(), transport)
            }
            PrimitiveBatch::Quads { range, .. } => {
                self.reserve(std::mem::size_of::<Quad>(), range.len(), transport)
            }
            PrimitiveBatch::Paths {
                rasterization_vertex_count,
                sprite_count,
                ..
            } if *rasterization_vertex_count > 0 => {
                self.reserve(
                    std::mem::size_of::<path_types::PathRasterizationVertex>(),
                    *rasterization_vertex_count,
                    transport,
                );
                self.reserve(
                    std::mem::size_of::<path_types::PathSprite>(),
                    *sprite_count,
                    transport,
                );
            }
            PrimitiveBatch::Underlines(range) => {
                self.reserve(std::mem::size_of::<Underline>(), range.len(), transport)
            }
            PrimitiveBatch::MonochromeSprites { range, .. } => self.reserve(
                std::mem::size_of::<MonochromeSprite>(),
                range.len(),
                transport,
            ),
            PrimitiveBatch::SubpixelSprites { range, .. } => self.reserve(
                std::mem::size_of::<SubpixelSprite>(),
                range.len(),
                transport,
            ),
            PrimitiveBatch::PolychromeSprites { range, .. } => self.reserve(
                std::mem::size_of::<PolychromeSprite>(),
                range.len(),
                transport,
            ),
            PrimitiveBatch::Paths { .. }
            | PrimitiveBatch::Surfaces(_)
            | PrimitiveBatch::BackdropFilters(_)
            | PrimitiveBatch::FilterBoundary(_) => {}
        }
    }
}

#[cfg(test)]
mod planning_tests {
    use super::*;

    #[test]
    fn partial_alignment_includes_clear_and_shared_region_cursor() {
        let transport = InstanceTransport::StorageBuffer;
        let sequence = [
            (std::mem::size_of::<Shadow>(), 1),
            (std::mem::size_of::<Quad>(), 1),
            (std::mem::size_of::<Shadow>(), 1),
            (std::mem::size_of::<Quad>(), 1),
            (
                std::mem::size_of::<path_types::PathRasterizationVertex>(),
                3,
            ),
            (std::mem::size_of::<path_types::PathSprite>(), 1),
            (std::mem::size_of::<MonochromeSprite>(), 1),
            (std::mem::size_of::<Quad>(), 1),
        ];
        let mut full = FrameRequirements::default();
        for &(stride, count) in &sequence {
            full.reserve(stride, count, transport);
        }
        assert_eq!(full.storage_bytes, 1984);
        let old_estimate = full.storage_bytes + 2 * std::mem::size_of::<Quad>() as u64;
        assert_eq!(old_estimate, 2480);
        let mut partial = FrameRequirements::default();
        partial.reserve(std::mem::size_of::<Quad>(), 1, transport);
        for &(stride, count) in &sequence {
            partial.reserve(stride, count, transport);
        }
        assert_eq!(partial.storage_bytes, 2728);
        assert_eq!(partial.instance_batches, 9);
        assert!(partial.storage_bytes > old_estimate);
        // A second region starts at the first region's actual ending offset.
        // It cannot restart an origin-zero estimate or merely double it.
        partial.reserve(std::mem::size_of::<Quad>(), 1, transport);
        for &(stride, count) in &sequence {
            partial.reserve(stride, count, transport);
        }
        assert_eq!(partial.instance_batches, 18);
        assert_eq!(partial.storage_bytes, 5456);
    }

    #[test]
    fn partial_planner_counts_only_intersecting_batches_and_clear_slots() {
        use gpui::{Bounds, ContentMask, ScaledPixels, point, size};
        let bounds = |x, y| {
            Bounds::new(
                point(ScaledPixels(x), ScaledPixels(y)),
                size(ScaledPixels(10.0), ScaledPixels(10.0)),
            )
        };
        let mask = ContentMask {
            bounds: Bounds::new(
                point(ScaledPixels(0.0), ScaledPixels(0.0)),
                size(ScaledPixels(100.0), ScaledPixels(100.0)),
            ),
            ..Default::default()
        };
        let mut scene = Scene::default();
        for (index, (x, y)) in [(0.0, 0.0), (20.0, 20.0)].into_iter().enumerate() {
            scene.insert_primitive(Quad {
                bounds: bounds(x, y),
                content_mask: mask,
                corner_smoothing: if index == 0 { 0.0 } else { 0.65 },
                ..Default::default()
            });
        }
        scene.finish();
        let regions = [
            DamageRect {
                x: 0,
                y: 0,
                width: 5,
                height: 5,
            },
            DamageRect {
                x: 24,
                y: 24,
                width: 5,
                height: 5,
            },
            DamageRect {
                x: 60,
                y: 60,
                width: 5,
                height: 5,
            },
        ];
        for transport in [
            InstanceTransport::StorageBuffer,
            InstanceTransport::DataTexture,
        ] {
            let requirements = FrameRequirements::for_partial_scene(&scene, &regions, transport);
            assert_eq!(
                requirements.instance_batches, 5,
                "three clears plus two intersecting batches"
            );
            let expected_bytes = match transport {
                InstanceTransport::StorageBuffer => 1240,
                InstanceTransport::DataTexture => 1272,
            };
            assert_eq!(requirements.storage_bytes, expected_bytes);
            assert!(!requirements.uses_path_target);
            assert!(!requirements.uses_offscreen_target);
        }
    }
}

struct FrameEncoder<'a> {
    renderer: &'a WgpuRenderer,
    scene: &'a Scene,
    encoder: wgpu::CommandEncoder,
    targets: TargetStack,
    offscreen: Option<wgpu::TextureView>,
    presentation: wgpu::TextureView,
    instances: InstanceUpload,
    damage: Damage,
    retained_blit: bool,
    retained_copy: Option<(wgpu::Texture, wgpu::Texture)>,
}

impl<'a> FrameEncoder<'a> {
    fn new(
        renderer: &'a WgpuRenderer,
        scene: &'a Scene,
        targets: PreparedTargets,
        damage: Damage,
        retained_blit: bool,
        retained_copy: Option<(wgpu::Texture, wgpu::Texture)>,
    ) -> Self {
        let encoder =
            renderer
                .resources()
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gpui_frame"),
                });
        Self {
            renderer,
            scene,
            encoder,
            targets: TargetStack::new(targets.active),
            offscreen: targets.offscreen,
            presentation: targets.presentation,
            instances: targets.instances,
            damage,
            retained_blit,
            retained_copy,
        }
    }

    fn encode(
        mut self,
        readback: Option<ReadbackCopy<'_>>,
    ) -> Result<wgpu::CommandBuffer, DrawError> {
        let result = self.encode_commands();
        if result.is_ok() {
            if let Some((source, destination)) = &self.retained_copy {
                self.encoder.copy_texture_to_texture(
                    source.as_image_copy(),
                    destination.as_image_copy(),
                    wgpu::Extent3d {
                        width: self.renderer.target.width(),
                        height: self.renderer.target.height(),
                        depth_or_array_layers: 1,
                    },
                );
            }
            if self.retained_blit {
                self.renderer.blit_to_frame(
                    &mut self.encoder,
                    self.targets.current(),
                    &self.presentation,
                );
            }
            if let Some(offscreen) = &self.offscreen {
                self.renderer
                    .blit_to_frame(&mut self.encoder, offscreen, &self.presentation);
            }
            if let Some(readback) = readback {
                self.encoder.copy_texture_to_buffer(
                    readback.texture.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: readback.buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(readback.bytes_per_row),
                            rows_per_image: Some(readback.height),
                        },
                    },
                    wgpu::Extent3d {
                        width: readback.width,
                        height: readback.height,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
        self.instances.finish(&mut self.encoder);
        self.renderer.resources().finish_frame_uploads();
        let command_buffer = self.encoder.finish();
        result.map(|()| command_buffer)
    }

    fn encode_commands(&mut self) -> DrawResult {
        match std::mem::replace(&mut self.damage, Damage::Reuse) {
            Damage::Reuse => Ok(()),
            Damage::Partial(regions) => {
                for region in regions {
                    self.encode_region(Some(region))?;
                }
                Ok(())
            }
            Damage::Full => self.encode_region(None),
        }
    }
    fn encode_region(&mut self, region: Option<DamageRect>) -> DrawResult {
        let mut pass = begin_scene_render_pass(
            self.renderer,
            &mut self.encoder,
            "main_pass",
            self.targets.current(),
            if region.is_some() {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(self.renderer.target.clear_color())
            },
            region,
        );
        if let Some(region) = region {
            let clear = self.renderer.target.clear_color();
            let clear = [clear.r, clear.g, clear.b, clear.a].map(|channel| channel as f32);
            self.renderer.draw_damage_clear(
                &[clear_quad(region, clear)],
                &mut self.instances,
                &mut pass,
            )?;
        }

        for command in self.scene.render_commands() {
            if let (Some(region), RenderCommand::Batch(batch)) = (region, command) {
                if !batch_intersects(self.scene, batch, region) {
                    continue;
                }
            }
            match command {
                RenderCommand::Batch(PrimitiveBatch::Paths {
                    range,
                    rasterization_vertex_count,
                    ..
                }) => {
                    if *rasterization_vertex_count == 0 {
                        continue;
                    }
                    let paths = &self.scene.paths[range.clone()];
                    drop(pass);
                    let rasterized = self.renderer.draw_paths_to_intermediate(
                        &mut self.encoder,
                        paths,
                        &mut self.instances,
                    );
                    pass = begin_scene_render_pass(
                        self.renderer,
                        &mut self.encoder,
                        "after_paths",
                        self.targets.current(),
                        wgpu::LoadOp::Load,
                        region,
                    );
                    rasterized?;
                    self.renderer.draw_paths_from_intermediate(
                        paths,
                        &mut self.instances,
                        &mut pass,
                    )?;
                }
                RenderCommand::Batch(PrimitiveBatch::BackdropFilters(range)) => {
                    drop(pass);
                    for filter in &self.scene.backdrop_filters[range.clone()] {
                        self.renderer.draw_backdrop_filter(
                            &mut self.encoder,
                            filter,
                            self.targets.current(),
                        );
                    }
                    pass = begin_scene_render_pass(
                        self.renderer,
                        &mut self.encoder,
                        "after_backdrop_filter",
                        self.targets.current(),
                        wgpu::LoadOp::Load,
                        region,
                    );
                }
                RenderCommand::Batch(PrimitiveBatch::FilterBoundary(_)) => {
                    unreachable!("filter boundaries must be compiled into render commands")
                }
                RenderCommand::Batch(batch) => encode_inline_batch(
                    self.renderer,
                    self.scene,
                    batch,
                    &mut self.instances,
                    &mut pass,
                )?,
                RenderCommand::BeginFilter {
                    target: FilterRenderTarget::Isolated(index),
                    ..
                } => {
                    drop(pass);
                    let target =
                        self.renderer.resources().filter_group_views[index.as_usize()].clone();
                    self.targets.enter(target);
                    pass = begin_scene_render_pass(
                        self.renderer,
                        &mut self.encoder,
                        "filter_group",
                        self.targets.current(),
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        region,
                    );
                }
                RenderCommand::EndFilter {
                    boundary_index,
                    target: FilterRenderTarget::Isolated(_),
                    ..
                } => {
                    drop(pass);
                    let (filtered, parent) = self.targets.exit();
                    let boundary = &self.scene.filter_boundaries[*boundary_index];
                    self.renderer.blur_and_composite(
                        &mut self.encoder,
                        &filtered,
                        parent,
                        FilterCompositeParameters {
                            bounds: boundary.bounds,
                            content_mask: boundary.content_mask.bounds,
                            corner_radii: boundary.corner_radii,
                            corner_smoothing: boundary.corner_smoothing,
                            blur_radius: boundary.max_blur_radius(),
                            opacity: boundary.opacity,
                            clip: FilterCompositeClip::ContentShape,
                        },
                    );
                    pass = begin_scene_render_pass(
                        self.renderer,
                        &mut self.encoder,
                        "after_content_filter",
                        self.targets.current(),
                        wgpu::LoadOp::Load,
                        region,
                    );
                }
                RenderCommand::BeginFilter {
                    target: FilterRenderTarget::Inline,
                    ..
                }
                | RenderCommand::EndFilter {
                    target: FilterRenderTarget::Inline,
                    ..
                } => {}
            }
        }
        drop(pass);
        self.targets.assert_balanced();
        Ok(())
    }
}

fn begin_scene_render_pass<'a>(
    renderer: &'a WgpuRenderer,
    encoder: &'a mut wgpu::CommandEncoder,
    label: &'a str,
    target: &'a wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    region: Option<DamageRect>,
) -> wgpu::RenderPass<'a> {
    let mut pass = begin_color_render_pass(encoder, label, target, load);
    if let Some(region) = region {
        pass.set_scissor_rect(region.x, region.y, region.width, region.height);
    }
    pass.set_bind_group(
        shader_interface::GLOBAL_BIND_GROUP,
        &renderer.resources().globals_bind_group,
        &[],
    );
    pass
}

struct TargetStack {
    current: wgpu::TextureView,
    parents: smallvec::SmallVec<[wgpu::TextureView; MAX_FILTER_GROUP_DEPTH]>,
}

impl TargetStack {
    fn new(root: wgpu::TextureView) -> Self {
        Self {
            current: root,
            parents: smallvec::SmallVec::new(),
        }
    }

    fn current(&self) -> &wgpu::TextureView {
        &self.current
    }

    fn enter(&mut self, next: wgpu::TextureView) {
        self.parents
            .push(std::mem::replace(&mut self.current, next));
    }

    fn exit(&mut self) -> (wgpu::TextureView, &wgpu::TextureView) {
        let parent = self
            .parents
            .pop()
            .expect("render plan ended an isolated filter without beginning one");
        let filtered = std::mem::replace(&mut self.current, parent);
        (filtered, &self.current)
    }

    fn assert_balanced(&self) {
        assert!(
            self.parents.is_empty(),
            "render plan left an isolated filter group open"
        );
    }
}

#[derive(Debug)]
pub(super) enum DrawError {
    CapacityPlanningInvariant,
    ExternalSurface,
    MissingIntermediateTarget,
}

pub(super) type DrawResult = Result<(), DrawError>;

fn encode_inline_batch(
    renderer: &WgpuRenderer,
    scene: &Scene,
    batch: &PrimitiveBatch,
    instances: &mut InstanceUpload,
    pass: &mut wgpu::RenderPass<'_>,
) -> DrawResult {
    match batch {
        PrimitiveBatch::Quads { range, smoothed } => {
            renderer.draw_quads(&scene.quads[range.clone()], *smoothed, instances, pass)
        }
        PrimitiveBatch::Shadows { range, smoothed } => {
            renderer.draw_shadows(&scene.shadows[range.clone()], *smoothed, instances, pass)
        }
        PrimitiveBatch::Underlines(range) => {
            renderer.draw_underlines(&scene.underlines[range.clone()], instances, pass)
        }
        PrimitiveBatch::MonochromeSprites { texture_id, range } => renderer
            .draw_monochrome_sprites(
                &scene.monochrome_sprites[range.clone()],
                *texture_id,
                instances,
                pass,
            ),
        PrimitiveBatch::SubpixelSprites { texture_id, range } => renderer.draw_subpixel_sprites(
            &scene.subpixel_sprites[range.clone()],
            *texture_id,
            instances,
            pass,
        ),
        PrimitiveBatch::PolychromeSprites {
            texture_id,
            range,
            smoothed,
        } => renderer.draw_polychrome_sprites(
            &scene.polychrome_sprites[range.clone()],
            *texture_id,
            *smoothed,
            instances,
            pass,
        ),
        PrimitiveBatch::Surfaces(range) => renderer.draw_surfaces(
            &scene.surfaces[range.clone()],
            &scene.surface_opacities()[range.clone()],
            pass,
        ),
        PrimitiveBatch::Paths { .. }
        | PrimitiveBatch::BackdropFilters(_)
        | PrimitiveBatch::FilterBoundary(_) => {
            unreachable!("pass-interrupting batches are handled by FrameEncoder")
        }
    }
}
