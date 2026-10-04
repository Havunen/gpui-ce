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
use gpui_render::damage::Damage;
use gpui_render::shaders::{
    common::{FontRasterizationUniforms, GlobalUniforms, ShaderBool},
    interface as shader_interface,
};

pub(super) fn render_to_view(
    renderer: &mut WgpuRenderer,
    scene: &Scene,
    frame_view: &wgpu::TextureView,
    readback: Option<ReadbackCopy<'_>>,
) -> Option<wgpu::SubmissionIndex> {
    let Some(targets) = PreparedTargets::prepare(renderer, scene, frame_view) else {
        return None;
    };
    // Prune once against the complete scene. Doing this inside each draw batch would evict a
    // texture that reappears after an intervening primitive and recreate its platform view.
    renderer.retain_surface_cache(&scene.surfaces);

    renderer.submission_id += 1;
    if let Some(trace) = renderer.resources().gpu_trace.as_ref() {
        let mut metrics = gpui::gpu_profiler::GpuFrameMetrics::new(
            scene,
            renderer.renderer_id,
            renderer.submission_id,
            "wgpu",
        );
        let r = renderer.resources();
        let bytes = |texture: &wgpu::Texture| {
            u64::from(texture.width())
                * u64::from(texture.height())
                * 4
                * u64::from(texture.sample_count())
        };
        metrics.memory.atlas_bytes = Some(renderer.atlas.allocated_bytes());
        metrics.memory.upload_bytes = Some(
            r.instances.allocated_bytes()
                + r.globals_buffer.size()
                + r.surface_uniforms.buffer.size()
                + r.filter_uniforms.buffer.size(),
        );
        metrics.memory.retained_frame_bytes = Some(
            r.retained
                .borrow()
                .as_ref()
                .map(|f| bytes(&f._texture))
                .unwrap_or(0),
        );
        let (pooled, pending) = r
            .texture_pool
            .as_ref()
            .map(|p| p.bytes())
            .unwrap_or_default();
        metrics.memory.pooled_bytes = Some(pooled);
        metrics.memory.pending_release_bytes = Some(pending);
        metrics.memory.device_retention_bytes =
            Some(super::shared::retention_budget(&r.device).used());
        metrics.path_target = r
            .path_intermediate_texture
            .as_ref()
            .map(|t| (t.width(), t.height()))
            .unwrap_or_default();
        metrics.memory.path_bytes = r
            .path_intermediate_texture
            .iter()
            .chain(r.path_msaa_texture.iter())
            .map(bytes)
            .sum();
        metrics.memory.filter_bytes = Some(
            r.scene_color_texture
                .iter()
                .chain(r.blur_ping_texture.iter())
                .chain(r.blur_pong_texture.iter())
                .chain(r.filter_group_textures.iter())
                .map(bytes)
                .sum(),
        );
        trace.begin(metrics);
    }
    match FrameEncoder::new(renderer, scene, targets).encode(readback) {
        Ok((command_buffer, mut used_paths)) => {
            let submission = renderer.resources().queue.submit([command_buffer]);
            let resources = renderer.resources();
            if renderer.options.cached_layers || renderer.options.partial_redraw {
                used_paths.extend(resources.path_cache.borrow().leases());
                let retained = resources
                    .retained
                    .borrow()
                    .as_ref()
                    .map(|frame| frame.lease.clone());
                resources
                    .queue
                    .on_submitted_work_done(move || drop((used_paths, retained)));
            }
            if let Some(trace) = &resources.gpu_trace {
                trace.submitted(submission.clone());
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
    damage: Damage,
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
        let mut requirements = {
            let transport = renderer.resources().instances.transport();
            FrameRequirements::for_scene(scene, transport)
        };
        if renderer.options.cached_layers
            || renderer.options.batched_paths
            || renderer.options.partial_redraw
        {
            requirements.storage_bytes = requirements.worst_storage_bytes;
        }
        if renderer.options.partial_redraw {
            requirements.storage_bytes += 2 * std::mem::size_of::<Quad>() as u64;
            requirements.instance_batches += 1;
            requirements.uniforms.filter_count += 1;
        }
        requirements.storage_bytes = requirements
            .storage_bytes
            .next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
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

        if let Some((view, damage)) = renderer.prepare_retained_frame(scene) {
            Some(Self {
                active: view.clone(),
                presentation: frame_view.clone(),
                offscreen: Some(view),
                instances,
                damage,
            })
        } else if requirements.uses_offscreen_target {
            let resources = renderer.resources();
            let offscreen = resources
                .scene_color_view
                .as_ref()
                .expect("blur texture preparation must create a scene target")
                .clone();
            Some(Self {
                damage: Damage::Full,
                active: offscreen.clone(),
                presentation: frame_view.clone(),
                offscreen: Some(offscreen),
                instances,
            })
        } else {
            Some(Self {
                damage: Damage::Full,
                active: frame_view.clone(),
                presentation: frame_view.clone(),
                offscreen: None,
                instances,
            })
        }
    }
}

fn begin_frame(renderer: &mut WgpuRenderer) -> bool {
    let Some(error) = renderer.faults.pending_error.lock().unwrap().take() else {
        renderer.faults.consecutive_failed_frames = 0;
        if renderer.options.cached_layers
            || renderer.options.partial_redraw
            || renderer.options.pooled_targets
        {
            let _ = renderer.resources().device.poll(wgpu::PollType::Poll);
        }
        renderer.atlas.before_frame();
        return true;
    };

    renderer.faults.consecutive_failed_frames += 1;
    log::error!(
        "GPU error during frame (failure {} of 10): {error}",
        renderer.faults.consecutive_failed_frames
    );
    if renderer.faults.consecutive_failed_frames > 10 {
        panic!("too many consecutive GPU errors; last error: {error}");
    }
    if renderer.faults.consecutive_failed_frames > 5 {
        if let Some(resources) = renderer.resources.as_mut() {
            resources.invalidate_intermediate_textures();
        }
        renderer.atlas.clear();
        renderer.target.request_redraw();
        renderer.faults.consecutive_failed_frames = 0;
        return false;
    }

    renderer.atlas.before_frame();
    true
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
    worst_storage_bytes: u64,
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
        let mut storage_bytes = 0_u64;
        let mut worst_storage_bytes = 0_u64;
        let mut instance_batches = 0_u64;
        let mut reserve = |element_size: usize, count: usize| {
            if count > 0 {
                let stride = element_size as u64;
                worst_storage_bytes = worst_storage_bytes
                    .saturating_add(stride.saturating_mul(count as u64))
                    .saturating_add(transport.batch_alignment(stride) - 1);
                storage_bytes = storage_bytes.next_multiple_of(transport.batch_alignment(stride));
                storage_bytes = storage_bytes.saturating_add(stride.saturating_mul(count as u64));
                instance_batches += 1;
            }
        };

        for command in scene.render_commands() {
            let RenderCommand::Batch(batch) = command else {
                continue;
            };
            match batch {
                PrimitiveBatch::Shadows { range, .. } => {
                    reserve(std::mem::size_of::<Shadow>(), range.len())
                }
                PrimitiveBatch::Quads { range, .. } => {
                    reserve(std::mem::size_of::<Quad>(), range.len())
                }
                PrimitiveBatch::Paths {
                    rasterization_vertex_count,
                    sprite_count,
                    ..
                } if *rasterization_vertex_count > 0 => {
                    reserve(
                        std::mem::size_of::<path_types::PathRasterizationVertex>(),
                        *rasterization_vertex_count,
                    );
                    reserve(std::mem::size_of::<path_types::PathSprite>(), *sprite_count);
                }
                PrimitiveBatch::Underlines(range) => {
                    reserve(std::mem::size_of::<Underline>(), range.len())
                }
                PrimitiveBatch::MonochromeSprites { range, .. } => {
                    reserve(std::mem::size_of::<MonochromeSprite>(), range.len())
                }
                PrimitiveBatch::SubpixelSprites { range, .. } => {
                    reserve(std::mem::size_of::<SubpixelSprite>(), range.len())
                }
                PrimitiveBatch::PolychromeSprites { range, .. } => {
                    reserve(std::mem::size_of::<PolychromeSprite>(), range.len())
                }
                PrimitiveBatch::Paths { .. }
                | PrimitiveBatch::Surfaces(_)
                | PrimitiveBatch::BackdropFilters(_)
                | PrimitiveBatch::FilterBoundary(_) => {}
            }
        }
        debug_assert_eq!(instance_batches as usize, planned.instance_batch_count);

        Self {
            storage_bytes,
            worst_storage_bytes,
            instance_batches,
            uniforms: FrameUniformRequirements {
                filter_count: FILTER_UNIFORMS_PER_COMPOSITE
                    * (planned.backdrop_filter_count + planned.isolated_filter_count) as u64
                    + u64::from(planned.uses_offscreen_target),
                surface_count: planned.surface_count as u64,
            },
            isolated_target_count: planned.isolated_target_count,
            uses_path_target: planned.uses_path_target,
            uses_offscreen_target: planned.uses_offscreen_target,
        }
    }
}

struct FrameEncoder<'a> {
    damage: Damage,
    renderer: &'a WgpuRenderer,
    scene: &'a Scene,
    encoder: wgpu::CommandEncoder,
    targets: TargetStack,
    offscreen: Option<wgpu::TextureView>,
    presentation: wgpu::TextureView,
    instances: InstanceUpload,
    paths: Vec<PathSource>,
}

enum PathSource {
    Inline,
    Cached(std::sync::Arc<super::path_cache::CachedPath>),
    Packed {
        world: gpui_render::path_plan::PixelRect,
        tile: gpui_render::path_plan::PixelRect,
    },
}

fn plan_paths(renderer: &WgpuRenderer, scene: &Scene) -> Vec<PathSource> {
    if !renderer.options.cached_layers && !renderer.options.batched_paths {
        return Vec::new();
    }
    use gpui_render::path_plan;
    let r = renderer.resources();
    let mut cache = r.path_cache.borrow_mut();
    cache.begin();
    let mut plan: Vec<_> = scene
        .render_commands()
        .iter()
        .map(|command| {
            if renderer.options.cached_layers {
                if let RenderCommand::Batch(PrimitiveBatch::Paths { range, .. }) = command {
                    if let Some(cached) = cache.get(&scene.paths[range.clone()]) {
                        return PathSource::Cached(cached);
                    }
                }
            }
            PathSource::Inline
        })
        .collect();
    if renderer.options.batched_paths && !scene.requires_offscreen_rendering() {
        if let Some(target) = &r.path_intermediate_texture {
            let extent = (target.width(), target.height());
            let misses: Vec<_> = scene
                .render_commands()
                .iter()
                .enumerate()
                .filter_map(|(i, command)| {
                    if matches!(plan[i], PathSource::Cached(_)) {
                        return None;
                    }
                    if let RenderCommand::Batch(PrimitiveBatch::Paths {
                        range,
                        rasterization_vertex_count,
                        ..
                    }) = command
                    {
                        if *rasterization_vertex_count == 0 {
                            return None;
                        }
                        let bounds = path_plan::visible_rect(
                            scene.paths[range.clone()]
                                .iter()
                                .map(gpui::Path::clipped_bounds),
                            extent,
                        )?;
                        Some((i, bounds))
                    } else {
                        None
                    }
                })
                .collect();
            if let Some(tiles) = path_plan::pack(
                &misses.iter().map(|(_, rect)| *rect).collect::<Vec<_>>(),
                extent,
            ) {
                for ((i, world), tile) in misses.into_iter().zip(tiles) {
                    plan[i] = PathSource::Packed { world, tile };
                }
            }
        }
    }
    let mut active = cache.views();
    active.extend(r.path_intermediate_view.iter().cloned());
    r.instances.retain_path_bindings(&active);
    plan
}

impl<'a> FrameEncoder<'a> {
    fn new(renderer: &'a WgpuRenderer, scene: &'a Scene, targets: PreparedTargets) -> Self {
        let paths = plan_paths(renderer, scene);
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
            damage: targets.damage,
            targets: TargetStack::new(targets.active),
            offscreen: targets.offscreen,
            presentation: targets.presentation,
            instances: targets.instances,
            paths,
        }
    }

    fn encode(
        mut self,
        readback: Option<ReadbackCopy<'_>>,
    ) -> Result<
        (
            wgpu::CommandBuffer,
            Vec<std::sync::Arc<super::path_cache::CachedPath>>,
        ),
        DrawError,
    > {
        let result = self.encode_commands();
        if result.is_ok() {
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
        if result.is_err() {
            if let Some(retained) = self.renderer.resources().retained.borrow_mut().as_mut() {
                retained.snapshot = None;
            }
        }
        self.instances.finish(&mut self.encoder);
        self.renderer.resources().finish_frame_uploads();
        if let Some(trace) = &self.renderer.resources().gpu_trace {
            if result.is_ok() {
                let cache = self.renderer.resources().path_cache.borrow();
                trace.cache_stats(cache.bytes(), cache.hits, cache.misses);
                trace.resolve(&mut self.encoder);
            } else {
                trace.cancel();
            }
        }
        let command_buffer = self.encoder.finish();
        let used_paths = self
            .paths
            .into_iter()
            .filter_map(|path| match path {
                PathSource::Cached(cached) => Some(cached),
                _ => None,
            })
            .collect();
        result.map(|()| (command_buffer, used_paths))
    }

    fn encode_commands(&mut self) -> DrawResult {
        if self.damage == Damage::Unchanged {
            return Ok(());
        }
        let packed: Vec<_> = self
            .scene
            .render_commands()
            .iter()
            .enumerate()
            .filter_map(|(i, command)| {
                if let (
                    PathSource::Packed { world, tile },
                    RenderCommand::Batch(PrimitiveBatch::Paths { range, .. }),
                ) = (self.paths.get(i).unwrap_or(&PathSource::Inline), command)
                {
                    Some((
                        &self.scene.paths[range.clone()][..],
                        tile.origin() - world.origin(),
                        Some(*tile),
                    ))
                } else {
                    None
                }
            })
            .collect();
        if !packed.is_empty() {
            self.renderer.draw_path_batches(
                &mut self.encoder,
                packed.into_iter(),
                &mut self.instances,
            )?;
            for (i, command) in self.scene.render_commands().iter().enumerate() {
                if let (
                    PathSource::Packed { world, tile },
                    RenderCommand::Batch(PrimitiveBatch::Paths { range, .. }),
                ) = (self.paths.get(i).unwrap_or(&PathSource::Inline), command)
                {
                    capture_path(
                        self.renderer,
                        &mut self.encoder,
                        *tile,
                        *world,
                        &self.scene.paths[range.clone()],
                    );
                }
            }
        }
        let mut pass = begin_scene_render_pass(
            self.renderer,
            &mut self.encoder,
            "main_pass",
            self.targets.current(),
            if matches!(self.damage, Damage::Rect(_)) {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(self.renderer.target.clear_color())
            },
            self.damage,
        );
        if matches!(self.damage, Damage::Rect(_)) {
            self.renderer.clear_damage(&mut self.instances, &mut pass)?;
        }

        for (command_index, command) in self.scene.render_commands().iter().enumerate() {
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
                    match self.paths.get(command_index).unwrap_or(&PathSource::Inline) {
                        PathSource::Cached(cached) => {
                            self.renderer.draw_paths_from_texture(
                                paths,
                                cached.bounds.origin(),
                                Some(cached.bounds),
                                &cached.view,
                                &mut self.instances,
                                &mut pass,
                            )?;
                            continue;
                        }
                        PathSource::Packed { world, tile } => {
                            let view = self
                                .renderer
                                .resources()
                                .path_intermediate_view
                                .as_ref()
                                .ok_or(DrawError::MissingIntermediateTarget)?;
                            self.renderer.draw_paths_from_texture(
                                paths,
                                world.origin() - tile.origin(),
                                Some(*world),
                                view,
                                &mut self.instances,
                                &mut pass,
                            )?;
                            continue;
                        }
                        PathSource::Inline => {}
                    }
                    if gpui_render::path_plan::visible_rect(
                        paths.iter().map(gpui::Path::clipped_bounds),
                        (self.renderer.target.width(), self.renderer.target.height()),
                    )
                    .is_none()
                    {
                        continue;
                    }
                    drop(pass);
                    let rasterized = self.renderer.draw_paths_to_intermediate(
                        &mut self.encoder,
                        paths,
                        &mut self.instances,
                    );
                    if rasterized.is_ok() {
                        if let Some(target) =
                            self.renderer.resources().path_intermediate_texture.as_ref()
                        {
                            if let Some(bounds) = gpui_render::path_plan::visible_rect(
                                paths.iter().map(gpui::Path::clipped_bounds),
                                (target.width(), target.height()),
                            ) {
                                capture_path(
                                    self.renderer,
                                    &mut self.encoder,
                                    bounds,
                                    bounds,
                                    paths,
                                );
                            }
                        }
                    }
                    pass = begin_scene_render_pass(
                        self.renderer,
                        &mut self.encoder,
                        "after_paths",
                        self.targets.current(),
                        wgpu::LoadOp::Load,
                        self.damage,
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
                        self.damage,
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
                        Damage::Full,
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
                        self.damage,
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

fn capture_path(
    renderer: &WgpuRenderer,
    encoder: &mut wgpu::CommandEncoder,
    source: gpui_render::path_plan::PixelRect,
    world: gpui_render::path_plan::PixelRect,
    paths: &[gpui::Path<gpui::ScaledPixels>],
) {
    if !renderer.options.cached_layers {
        return;
    }
    let r = renderer.resources();
    if let Some(texture) = &r.path_intermediate_texture {
        r.path_cache
            .borrow_mut()
            .capture(&r.device, encoder, texture, source, world, paths);
    }
}

fn begin_scene_render_pass<'a>(
    renderer: &'a WgpuRenderer,
    encoder: &'a mut wgpu::CommandEncoder,
    label: &'static str,
    target: &'a wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    damage: Damage,
) -> wgpu::RenderPass<'a> {
    let mut pass = begin_color_render_pass(
        encoder,
        label,
        target,
        load,
        renderer.resources().gpu_trace.as_ref(),
    );
    if let Damage::Rect(rect) = damage {
        pass.set_scissor_rect(rect.x, rect.y, rect.width, rect.height);
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
