use gpui::{
    AtlasTextureId, MonochromeSprite, Path, PolychromeSprite, Quad, ScaledPixels, Shadow,
    SubpixelSprite, Underline,
};

use crate::WgpuTextureInfo;
use gpui_render::shaders::interface::{self as shader_interface, BufferData};

use super::{
    WgpuRenderer,
    buffers::{InstanceSlice, InstanceUpload},
    frame, path_types, pipelines,
};

impl WgpuRenderer {
    pub(super) fn clear_damage(
        &self,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let r = self.resources();
        let bounds = gpui::Bounds::new(
            Default::default(),
            gpui::size(
                ScaledPixels(self.target.width() as f32),
                ScaledPixels(self.target.height() as f32),
            ),
        );
        let color = self.target.clear_color();
        let quad = Quad {
            bounds,
            content_mask: gpui::ContentMask {
                bounds,
                ..Default::default()
            },
            background: gpui::Rgba::new(
                color.r as f32,
                color.g as f32,
                color.b as f32,
                color.a as f32,
            )
            .into(),
            ..Default::default()
        };
        self.draw_instances(
            &[quad],
            r.pipelines.clear_quads(
                &r.device,
                &r.bind_group_layouts,
                self.target.format(),
                r.renderer_tier,
            ),
            instances,
            pass,
        )
    }
    pub(super) fn draw_quads(
        &self,
        quads: &[Quad],
        smoothed: bool,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let pipelines = &self.resources().pipelines;
        let pipeline = if smoothed {
            &pipelines.smoothed_quads
        } else {
            &pipelines.quads
        };
        self.draw_instances(quads, pipeline, instances, pass)
    }

    pub(super) fn draw_shadows(
        &self,
        shadows: &[Shadow],
        smoothed: bool,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let pipelines = &self.resources().pipelines;
        let pipeline = if smoothed {
            &pipelines.smoothed_shadows
        } else {
            &pipelines.shadows
        };
        self.draw_instances(shadows, pipeline, instances, pass)
    }

    pub(super) fn draw_underlines(
        &self,
        underlines: &[Underline],
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        self.draw_instances(
            underlines,
            &self.resources().pipelines.underlines,
            instances,
            pass,
        )
    }

    pub(super) fn draw_monochrome_sprites(
        &self,
        sprites: &[MonochromeSprite],
        texture_id: AtlasTextureId,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let texture = self.atlas.get_texture_info(texture_id);
        self.draw_instances_with_texture(
            sprites,
            texture_id,
            &texture,
            &self.resources().pipelines.monochrome_sprites,
            instances,
            pass,
        )
    }

    pub(super) fn draw_subpixel_sprites(
        &self,
        sprites: &[SubpixelSprite],
        texture_id: AtlasTextureId,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let texture = self.atlas.get_texture_info(texture_id);
        let resources = self.resources();
        let pipeline = resources
            .pipelines
            .subpixel_sprites
            .as_ref()
            .unwrap_or(&resources.pipelines.monochrome_sprites);
        self.draw_instances_with_texture(sprites, texture_id, &texture, pipeline, instances, pass)
    }

    pub(super) fn draw_polychrome_sprites(
        &self,
        sprites: &[PolychromeSprite],
        texture_id: AtlasTextureId,
        smoothed: bool,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let texture = self.atlas.get_texture_info(texture_id);
        let pipelines = &self.resources().pipelines;
        let pipeline = if smoothed {
            &pipelines.smoothed_polychrome_sprites
        } else {
            &pipelines.polychrome_sprites
        };
        self.draw_instances_with_texture(sprites, texture_id, &texture, pipeline, instances, pass)
    }

    fn draw_instances<T: BufferData>(
        &self,
        values: &[T],
        pipeline: &pipelines::WgpuRenderPipeline,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        if values.is_empty() {
            return Ok(());
        }
        let resources = self.resources();
        self.draw_bound_instances(
            values,
            pipeline,
            resources.instances.bind_group(),
            instances,
            pass,
        )
    }

    fn draw_bound_instances<T: BufferData>(
        &self,
        values: &[T],
        pipeline: &pipelines::WgpuRenderPipeline,
        bind_group: &wgpu::BindGroup,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let Some(slice) = instances.write(values) else {
            return Err(frame::DrawError::CapacityPlanningInvariant);
        };
        self.draw_bound_slice(pipeline, bind_group, &slice, pass);
        Ok(())
    }

    fn draw_bound_slice<T>(
        &self,
        pipeline: &pipelines::WgpuRenderPipeline,
        bind_group: &wgpu::BindGroup,
        slice: &InstanceSlice<T>,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        pass.set_pipeline(pipeline);
        slice.set_data_bind_group(pass, bind_group);
        pass.draw(0..pipeline.fixed_vertex_count(), slice.range());
    }

    fn draw_instances_with_texture<T: BufferData>(
        &self,
        values: &[T],
        texture_id: AtlasTextureId,
        texture: &WgpuTextureInfo,
        pipeline: &pipelines::WgpuRenderPipeline,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        if values.is_empty() {
            return Ok(());
        }
        let resources = self.resources();
        let bind_group = resources.instances.textured_bind_group(
            &resources.device,
            &resources.bind_group_layouts,
            pipeline.data_layout(),
            texture_id,
            texture.identity,
            &texture.view,
            &resources.atlas_sampler,
        );
        self.draw_bound_instances(values, pipeline, &bind_group, instances, pass)
    }

    pub(super) fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let Some(view) = self.resources().path_intermediate_view.as_ref() else {
            return Err(frame::DrawError::MissingIntermediateTarget);
        };
        self.draw_paths_from_texture(paths, Default::default(), None, view, instances, pass)
    }

    pub(super) fn draw_paths_from_texture(
        &self,
        paths: &[Path<ScaledPixels>],
        origin: gpui::Point<ScaledPixels>,
        clip: Option<gpui_render::path_plan::PixelRect>,
        view: &wgpu::TextureView,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let sprite_count = path_types::sprite_count(paths);
        let sprites = path_types::sprites(paths).map(|mut sprite| {
            sprite.texture_origin = origin;
            if let Some(clip) = clip {
                sprite.bounds = sprite.bounds.intersect(&gpui::Bounds::new(
                    clip.origin(),
                    gpui::size(
                        ScaledPixels(clip.width as f32),
                        ScaledPixels(clip.height as f32),
                    ),
                ));
            }
            sprite
        });
        let Some(sprite_slice) = instances.write_iter(sprite_count, sprites) else {
            return Err(frame::DrawError::CapacityPlanningInvariant);
        };
        let resources = self.resources();
        let bind_group = resources.instances.path_bind_group(
            &resources.device,
            &resources.bind_group_layouts,
            resources.pipelines.paths.data_layout(),
            view,
            &resources.atlas_sampler,
        );
        self.draw_bound_slice(&resources.pipelines.paths, &bind_group, &sprite_slice, pass);
        Ok(())
    }

    pub(super) fn draw_paths_to_intermediate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        paths: &[Path<ScaledPixels>],
        instances: &mut InstanceUpload,
    ) -> frame::DrawResult {
        self.draw_path_batches(
            encoder,
            std::iter::once((paths, Default::default(), None)),
            instances,
        )
    }

    pub(super) fn draw_path_batches<'p>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        batches: impl Iterator<
            Item = (
                &'p [Path<ScaledPixels>],
                gpui::Point<ScaledPixels>,
                Option<gpui_render::path_plan::PixelRect>,
            ),
        >,
        instances: &mut InstanceUpload,
    ) -> frame::DrawResult {
        let resources = self.resources();
        let Some(path_intermediate_view) = resources.path_intermediate_view.as_ref() else {
            return Err(frame::DrawError::MissingIntermediateTarget);
        };
        let (target_view, resolve_target) = if let Some(msaa_view) = &resources.path_msaa_view {
            (msaa_view, Some(path_intermediate_view))
        } else {
            (path_intermediate_view, None)
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("path_rasterization_pass"),
            timestamp_writes: resources
                .gpu_trace
                .as_ref()
                .and_then(|trace| trace.timestamps("path_rasterization")),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    // Only the resolved texture is sampled later. The MSAA
                    // attachment is transient and cannot use StoreOp::Store.
                    store: if resolve_target.is_some() {
                        wgpu::StoreOp::Discard
                    } else {
                        wgpu::StoreOp::Store
                    },
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&resources.pipelines.path_rasterization);
        pass.set_bind_group(
            shader_interface::GLOBAL_BIND_GROUP,
            &resources.path_globals_bind_group,
            &[],
        );
        for (paths, offset, clip) in batches {
            let count = path_types::rasterization_vertex_count(paths);
            if count == 0 {
                continue;
            }
            let vertices = path_types::rasterization_vertices(paths).map(|mut v| {
                v.raster_offset = offset;
                v
            });
            let Some(slice) = instances.write_iter(count, vertices) else {
                return Err(frame::DrawError::CapacityPlanningInvariant);
            };
            if let Some(clip) = clip {
                pass.set_scissor_rect(clip.x, clip.y, clip.width, clip.height);
            }
            slice.set_data_bind_group(&mut pass, resources.instances.bind_group());
            pass.draw(slice.range(), 0..1);
        }
        Ok(())
    }
}
