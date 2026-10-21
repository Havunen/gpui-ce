//! Shared DX12/Vulkan shader comparisons.

#[cfg(feature = "test-support")]
use super::pipelines::PipelineShaders;
use super::pipelines::WgpuPipelines;

/// Every WGPU pipeline, labelled, as `WgpuPipelines` holds them.
pub(super) fn every_pipeline(pipelines: &WgpuPipelines) -> Vec<(&'static str, bool)> {
    let mut all = vec![
        &pipelines.quads,
        &pipelines.smoothed_quads,
        &pipelines.shadows,
        &pipelines.smoothed_shadows,
        &pipelines.path_rasterization,
        &pipelines.paths,
        &pipelines.underlines,
        &pipelines.monochrome_sprites,
        &pipelines.polychrome_sprites,
        &pipelines.smoothed_polychrome_sprites,
        &pipelines.surfaces,
        &pipelines.blur_downsample,
        &pipelines.blur,
        &pipelines.blur_composite,
        &pipelines.smoothed_blur_composite,
    ];
    all.extend(pipelines.subpixel_sprites.as_ref());
    all.extend(pipelines.damage_clear_quads.as_ref());
    all.into_iter()
        .map(|pipeline| (pipeline.label(), pipeline.is_precompiled()))
        .collect()
}

#[cfg(feature = "test-support")]
pub(super) use rendering::*;

#[cfg(feature = "test-support")]
mod rendering {
    use super::{PipelineShaders, WgpuPipelines, every_pipeline};
    use crate::{WgpuContext, wgpu_renderer::WgpuRenderer};
    use gpui::{
        AtlasKey, AtlasTile, BackdropFilter, BorderStyle, Bounds, ContentMask, Corners,
        DevicePixels, Edges, FilterBoundary, FontId, GlyphId, GlyphRenderMode, ImageId,
        MonochromeSprite, PathBuilder, PlatformAtlas, PolychromeSprite, PreparedRasterStyle, Quad,
        RenderGlyphParams, RenderImageParams, ScaledFilter, ScaledPixels, Scene, Shadow,
        SubpixelSprite, Underline, point, px, size,
    };
    use std::borrow::Cow;

    const SIZE: i32 = 160;

    pub(crate) fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(width), ScaledPixels(height)),
        )
    }

    pub(crate) fn mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: bounds(0.0, 0.0, SIZE as f32, SIZE as f32),
            ..Default::default()
        }
    }

    /// One renderer per shader source, sharing a device.
    pub(crate) fn renderers(context: &WgpuContext) -> anyhow::Result<(WgpuRenderer, WgpuRenderer)> {
        let target = size(DevicePixels(SIZE), DevicePixels(SIZE));
        let mut precompiled = WgpuRenderer::new_headless(context, target)?;
        let mut compiled = WgpuRenderer::new_headless(context, target)?;
        for renderer in [&mut precompiled, &mut compiled] {
            renderer.retained.disable_for_tests();
        }
        let config = compiled.target.configuration().clone();
        let samples = compiled.rendering_params.path_sample_count;
        let dual_source_blending = compiled.dual_source_blending;
        let resources = compiled.resources_mut();
        resources.pipelines = WgpuPipelines::with_shaders(
            &resources.device,
            &resources.bind_group_layouts,
            config.format,
            config.alpha_mode,
            samples,
            dual_source_blending,
            resources.renderer_tier,
            PipelineShaders::new(&[], resources.renderer_tier),
        );
        let precompiled_pipelines = every_pipeline(&precompiled.resources().pipelines);
        assert!(
            precompiled_pipelines
                .iter()
                .all(|(_, precompiled)| *precompiled),
            "the default renderer must use precompiled bytecode: {precompiled_pipelines:?}"
        );
        assert!(
            every_pipeline(&compiled.resources().pipelines)
                .iter()
                .all(|(_, precompiled)| !precompiled)
        );
        Ok((precompiled, compiled))
    }

    fn glyph_tile(renderer: &WgpuRenderer, mode: GlyphRenderMode) -> anyhow::Result<AtlasTile> {
        let key = AtlasKey::Glyph {
            params: RenderGlyphParams {
                font_id: FontId(0),
                glyph_id: GlyphId(71),
                font_size: px(10.0),
                subpixel_variant: point(0, 0),
                scale_factor: 1.0,
                raster_style: PreparedRasterStyle::independent(mode),
            },
            format: mode.rasterized_format(),
        };
        let coverage = (0..100u32)
            .map(|index| ((index * 37 + index / 10 * 11) % 256) as u8)
            .collect::<Vec<_>>();
        let bytes = if mode == GlyphRenderMode::Subpixel {
            coverage
                .iter()
                .flat_map(|&value| [value / 3, value, value.saturating_add(40), 255])
                .collect()
        } else {
            coverage
        };
        renderer
            .atlas
            .get_or_insert_with(&key, &mut || {
                Ok(Some((
                    size(DevicePixels(10), DevicePixels(10)),
                    Cow::Borrowed(&bytes),
                )))
            })?
            .ok_or_else(|| anyhow::anyhow!("glyph fixture must allocate a tile"))
    }

    fn image_tile(renderer: &WgpuRenderer) -> anyhow::Result<AtlasTile> {
        let pixels = (0..16 * 16)
            .flat_map(|index: u32| {
                let (x, y) = (index % 16, index / 16);
                [(x * 16) as u8, (y * 16) as u8, ((x ^ y) * 16) as u8, 255]
            })
            .collect::<Vec<_>>();
        renderer
            .atlas
            .get_or_insert_with(
                &AtlasKey::Image(RenderImageParams {
                    image_id: ImageId(4821),
                    frame_index: 0,
                }),
                &mut || {
                    Ok(Some((
                        size(DevicePixels(16), DevicePixels(16)),
                        Cow::Borrowed(&pixels),
                    )))
                },
            )?
            .ok_or_else(|| anyhow::anyhow!("image fixture must allocate a tile"))
    }

    pub(crate) struct Tiles {
        pub(crate) monochrome: AtlasTile,
        pub(crate) subpixel: Option<AtlasTile>,
        pub(crate) image: AtlasTile,
    }

    pub(crate) fn atlas_tiles(renderer: &WgpuRenderer) -> anyhow::Result<Tiles> {
        Ok(Tiles {
            monochrome: glyph_tile(renderer, GlyphRenderMode::Grayscale)?,
            subpixel: renderer
                .dual_source_blending
                .then(|| glyph_tile(renderer, GlyphRenderMode::Subpixel))
                .transpose()?,
            image: image_tile(renderer)?,
        })
    }

    pub(crate) fn backdrop(scene: &mut Scene) {
        scene.insert_primitive(Quad {
            bounds: bounds(0.0, 0.0, SIZE as f32, SIZE as f32),
            content_mask: mask(),
            background: gpui::checkerboard(gpui::rgb(0x3355aa), 3.0),
            ..Default::default()
        });
        scene.raise_order_floor();
    }

    /// Scenes that together draw with every pipeline: each smoothing variant, borders,
    /// paths, wavy underlines, every sprite kind, backdrop and isolated filters with and
    /// without blur, and a backdrop inside an opacity group (the snapshot composite).
    pub(crate) fn scenes(tiles: &Tiles) -> Vec<(&'static str, Scene)> {
        let radii = Corners::all(ScaledPixels(14.0));
        let element = bounds(24.0, 28.0, 96.0, 80.0);
        let mut scenes = Vec::new();
        let mut add = |name: &'static str, build: &dyn Fn(&mut Scene)| {
            let mut scene = Scene::default();
            backdrop(&mut scene);
            build(&mut scene);
            scene.finish();
            scenes.push((name, scene));
        };
        for smoothing in [0.0, 0.7] {
            let smoothed = smoothing > 0.0;
            add(
                if smoothed { "smoothed quads" } else { "quads" },
                &|scene| {
                    for (index, style) in [BorderStyle::Solid, BorderStyle::Dashed]
                        .into_iter()
                        .enumerate()
                    {
                        scene.insert_primitive(Quad {
                            bounds: bounds(12.0 + index as f32 * 70.0, 20.0, 60.0, 110.0),
                            content_mask: mask(),
                            background: gpui::linear_gradient(
                                35.0,
                                gpui::linear_color_stop(gpui::rgba(0xe0405080), 0.0),
                                gpui::linear_color_stop(gpui::rgba(0x40e0a0ff), 1.0),
                            ),
                            border_color: gpui::rgba(0xffd040ff).into(),
                            border_widths: Edges::all(ScaledPixels(3.5)),
                            border_style: style,
                            border_dashed_length: 7.0,
                            border_dashed_gap: 4.0,
                            corner_radii: radii,
                            corner_smoothing: smoothing,
                            ..Default::default()
                        });
                    }
                },
            );
            add(
                if smoothed {
                    "smoothed shadows"
                } else {
                    "shadows"
                },
                &|scene| {
                    for (index, inset) in [false, true].into_iter().enumerate() {
                        let element = bounds(18.0 + index as f32 * 72.0, 30.0, 56.0, 90.0);
                        scene.insert_primitive(Shadow {
                            order: 0,
                            blur_radius: ScaledPixels(7.5),
                            bounds: element,
                            corner_radii: radii,
                            content_mask: mask(),
                            color: gpui::hsla(0.8, 0.6, 0.3, 0.8).into(),
                            element_bounds: element,
                            element_corner_radii: radii,
                            inset: inset.into(),
                            corner_smoothing: smoothing,
                        });
                    }
                },
            );
            add(
                if smoothed {
                    "smoothed images"
                } else {
                    "images"
                },
                &|scene| {
                    for (index, grayscale) in [false, true].into_iter().enumerate() {
                        scene.insert_primitive(PolychromeSprite {
                            order: 0,
                            bounds: bounds(10.0 + index as f32 * 74.0, 26.0, 66.0, 96.0),
                            content_mask: mask(),
                            corner_radii: radii,
                            corner_smoothing: smoothing,
                            grayscale: grayscale.into(),
                            opacity: 0.85,
                            tile: tiles.image,
                        });
                    }
                },
            );
            add(
                if smoothed {
                    "smoothed backdrop blur"
                } else {
                    "backdrop blur"
                },
                &|scene| {
                    scene.insert_primitive(BackdropFilter {
                        order: 0,
                        bounds: element,
                        content_mask: mask(),
                        corner_radii: radii,
                        corner_smoothing: smoothing,
                        filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(6.0))],
                        opacity: 0.9,
                    });
                },
            );
            for blur in [0.0, 5.0] {
                add(
                    match (smoothed, blur > 0.0) {
                        (false, false) => "opacity group",
                        (false, true) => "isolated blur",
                        (true, false) => "smoothed opacity group",
                        (true, true) => "smoothed isolated blur",
                    },
                    &|scene| {
                        let boundary = FilterBoundary {
                            order: 0,
                            bounds: element,
                            content_mask: mask(),
                            corner_radii: radii,
                            corner_smoothing: smoothing,
                            filters: if blur > 0.0 {
                                smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(blur))]
                            } else {
                                smallvec::smallvec![]
                            },
                            opacity: 0.6,
                            is_start: true,
                        };
                        scene.insert_primitive(boundary.clone());
                        scene.insert_primitive(Quad {
                            bounds: bounds(36.0, 40.0, 72.0, 56.0),
                            content_mask: mask(),
                            background: gpui::rgba(0xf0f0f0ff).into(),
                            corner_radii: radii,
                            ..Default::default()
                        });
                        // Inside a group without filters, a backdrop composites a snapshot.
                        if blur == 0.0 {
                            scene.insert_primitive(BackdropFilter {
                                order: 0,
                                bounds: bounds(50.0, 50.0, 50.0, 40.0),
                                content_mask: mask(),
                                corner_radii: radii,
                                corner_smoothing: smoothing,
                                filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(4.0))],
                                opacity: 1.0,
                            });
                        }
                        scene.insert_primitive(FilterBoundary {
                            is_start: false,
                            ..boundary
                        });
                    },
                );
            }
        }
        add("paths and underlines", &|scene| {
            let mut builder = PathBuilder::fill();
            builder.move_to(point(px(12.0), px(140.0)));
            builder.curve_to(point(px(140.0), px(130.0)), point(px(60.0), px(10.0)));
            builder.line_to(point(px(30.0), px(40.0)));
            builder.close();
            let mut path = builder.build().unwrap().scale(1.0);
            path.content_mask = mask();
            path.color = gpui::linear_gradient(
                120.0,
                gpui::linear_color_stop(gpui::rgb(0xff2040), 0.0),
                gpui::linear_color_stop(gpui::rgb(0x20ffa0), 1.0),
            );
            scene.insert_primitive(path);
            for (index, wavy) in [false, true].into_iter().enumerate() {
                scene.insert_primitive(Underline {
                    order: 0,
                    padding: 0,
                    bounds: bounds(14.0, 120.0 + index as f32 * 16.0, 130.0, 6.0),
                    content_mask: mask(),
                    color: gpui::hsla(0.15, 0.9, 0.6, 1.0).into(),
                    thickness: ScaledPixels(2.0),
                    wavy: wavy.into(),
                });
            }
        });
        add("glyphs", &|scene| {
            for row in 0..6 {
                for column in 0..12 {
                    let glyph_bounds = bounds(
                        8.0 + column as f32 * 12.25,
                        10.0 + row as f32 * 24.5,
                        11.5,
                        13.0,
                    );
                    let color = gpui::hsla(column as f32 / 12.0, 0.7, 0.6, 0.9);
                    match (row % 2, tiles.subpixel) {
                        (1, Some(tile)) => scene.insert_primitive(SubpixelSprite {
                            order: 0,
                            padding: 0,
                            bounds: glyph_bounds,
                            content_mask: mask(),
                            color: color.into(),
                            tile,
                            transformation: Default::default(),
                        }),
                        _ => scene.insert_primitive(MonochromeSprite {
                            order: 0,
                            padding: 0,
                            bounds: glyph_bounds,
                            content_mask: mask(),
                            color: color.into(),
                            tile: tiles.monochrome,
                            transformation: Default::default(),
                        }),
                    }
                }
            }
        });
        scenes
    }

    pub(crate) fn assert_identical(
        name: &str,
        precompiled: &image::RgbaImage,
        compiled: &image::RgbaImage,
    ) {
        if let Some((x, y, pixel)) = precompiled
            .enumerate_pixels()
            .find(|(x, y, pixel)| *pixel != compiled.get_pixel(*x, *y))
        {
            panic!(
                "{name}: precompiled pixel ({x}, {y}) is {:?}, runtime-compiled is {:?}",
                pixel.0,
                compiled.get_pixel(x, y).0
            );
        }
    }

    pub(crate) fn compare_rendering(context: &WgpuContext) -> anyhow::Result<()> {
        let (mut precompiled, mut compiled) = renderers(context)?;
        let tiles = atlas_tiles(&precompiled)?;
        let compiled_tiles = atlas_tiles(&compiled)?;
        assert_eq!(
            (tiles.monochrome, tiles.subpixel, tiles.image),
            (
                compiled_tiles.monochrome,
                compiled_tiles.subpixel,
                compiled_tiles.image
            ),
            "both atlases must place the fixtures identically"
        );
        let mut empty = Scene::default();
        backdrop(&mut empty);
        empty.finish();
        let background = precompiled.render_to_image(&empty)?;
        for (name, scene) in scenes(&tiles) {
            let image = precompiled.render_to_image(&scene)?;
            assert!(image != background, "{name} must draw something");
            assert_identical(name, &image, &compiled.render_to_image(&scene)?);
        }
        assert!(context.uncaptured_error_slot().lock().unwrap().is_none());
        Ok(())
    }

    /// Retained frames add the damage-clear and texture-blit pipelines: a partial frame
    /// clears with the Quad shaders, and a target that cannot be copied is blitted.
    pub(crate) fn compare_retained(context: &WgpuContext) -> anyhow::Result<()> {
        let (mut precompiled, mut compiled) = renderers(context)?;
        let _frame = gpui::profiler::gpu::set_gpu_frame_context(true, 1);
        let frame = |row: usize| {
            let mut scene = Scene::default();
            for index in 0..6 {
                scene.insert_primitive(Quad {
                    bounds: bounds(10.0, 10.0 + index as f32 * 24.0, 140.0, 20.0),
                    content_mask: mask(),
                    background: if index == row {
                        gpui::rgba(0x5080c0ff)
                    } else {
                        gpui::rgba(0x203040ff)
                    }
                    .into(),
                    corner_radii: Corners::all(ScaledPixels(4.0)),
                    ..Default::default()
                });
            }
            scene.finish();
            scene
        };
        let backend_can_copy = precompiled.target.retained_copy_enabled();
        for copy_supported in [true, false] {
            for renderer in [&mut precompiled, &mut compiled] {
                renderer.retained.enable_for_tests();
                if !copy_supported {
                    renderer
                        .target
                        .update_surface_copy_support(wgpu::TextureUsages::RENDER_ATTACHMENT, &[]);
                }
            }
            for row in [0, 3, 3, 5] {
                let scene = frame(row);
                let image = precompiled.render_to_image(&scene)?;
                assert_identical(
                    &format!("retained row {row}, copy={copy_supported}"),
                    &image,
                    &compiled.render_to_image(&scene)?,
                );
            }
            assert_eq!(
                precompiled.last_retained_presentation,
                if copy_supported && backend_can_copy {
                    "copy"
                } else {
                    "blit"
                }
            );
        }
        let damage_clear = precompiled
            .resources()
            .pipelines
            .damage_clear_quads
            .as_ref()
            .expect("partial frames create the damage-clear pipeline");
        assert!(damage_clear.is_precompiled());
        assert!(context.uncaptured_error_slot().lock().unwrap().is_none());
        Ok(())
    }
}
