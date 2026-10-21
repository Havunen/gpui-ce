//! DX12 pipelines are created from bytecode compiled at build time (see
//! `gpui_render/build.rs`). That bytecode is only valid for the exact HLSL, and so the
//! exact root signature, that this WGPU version generates for each pipeline layout.
//! These tests pin both: the artifacts must be WGPU's own HLSL, every pipeline must
//! actually use them, they must render exactly like WGPU's runtime compilation, and
//! anything the device rejects must fall back to that compilation.

use super::pipelines::{PipelineShaders, WgpuBindGroupLayouts, WgpuPipelines};
use crate::RendererTier;
use gpui_render::{
    artifacts::{DX12_SHADER_HLSL, DX12_SHADERS, Dx12Shader},
    shaders::interface as shader,
};

/// A DX12 instance whose WGPU compiles exactly as the build does: with FXC, so its HLSL
/// targets the artifacts' shader model, and without `InstanceFlags::DEBUG`, which debug
/// builds enable and which adds `D3DCOMPILE_SKIP_OPTIMIZATION`. Its WGSL pipelines then
/// run the same optimized bytecode as the artifacts, so a pixel difference between the
/// two is a binding or layout mismatch, never two compilations rounding differently.
fn fxc_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::default() - wgpu::InstanceFlags::DEBUG,
        backend_options: wgpu::BackendOptions {
            dx12: wgpu::Dx12BackendOptions {
                shader_compiler: wgpu::Dx12Compiler::Fxc,
                ..Default::default()
            },
            ..Default::default()
        },
        memory_budget_thresholds: Default::default(),
        display: None,
    })
}

fn fxc_device(passthrough: bool) -> anyhow::Result<(wgpu::Adapter, wgpu::Device)> {
    let instance = fxc_instance();
    let adapter = gpui::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))?;
    let mut required_features = wgpu::Features::empty();
    if adapter
        .features()
        .contains(wgpu::Features::DUAL_SOURCE_BLENDING)
    {
        required_features |= wgpu::Features::DUAL_SOURCE_BLENDING;
    }
    if passthrough {
        required_features |= wgpu::Features::PASSTHROUGH_SHADERS;
    }
    let (device, _queue) = gpui::block_on(
        adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("precompiled_shader_tests"),
            required_features,
            required_limits: wgpu::Limits::downlevel_defaults()
                .using_resolution(adapter.limits())
                .using_alignment(adapter.limits()),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }),
    )?;
    Ok((adapter, device))
}

fn build_pipelines(
    device: &wgpu::Device,
    dual_source_blending: bool,
    shaders: PipelineShaders,
) -> WgpuPipelines {
    let layouts = WgpuBindGroupLayouts::new(device, RendererTier::Modern);
    WgpuPipelines::with_shaders(
        device,
        &layouts,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::CompositeAlphaMode::Opaque,
        1,
        dual_source_blending,
        RendererTier::Modern,
        shaders,
    )
}

/// Every WGPU pipeline, labelled, as `WgpuPipelines` holds them.
fn every_pipeline(pipelines: &WgpuPipelines) -> Vec<(&'static str, bool)> {
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

mod hlsl_capture {
    //! Captures the HLSL wgpu-hal logs (at debug level) as it compiles each stage.

    use std::sync::{Mutex, OnceLock};
    use std::thread::ThreadId;

    struct Capture;

    static CAPTURED: Mutex<Option<(ThreadId, Vec<String>)>> = Mutex::new(None);

    impl log::Log for Capture {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.target().starts_with("wgpu_hal")
        }

        fn log(&self, record: &log::Record<'_>) {
            if !self.enabled(record.metadata()) {
                return;
            }
            let mut captured = CAPTURED.lock().unwrap();
            if let Some((thread, messages)) = captured.as_mut()
                && *thread == std::thread::current().id()
            {
                messages.push(record.args().to_string());
            }
        }

        fn flush(&self) {}
    }

    /// Runs `f`, returning `(stage, entry point) -> HLSL` for each stage wgpu-hal
    /// generated on this thread meanwhile.
    pub(super) fn capture(f: impl FnOnce()) -> Vec<(String, String, String)> {
        static INSTALLED: OnceLock<bool> = OnceLock::new();
        assert!(
            *INSTALLED.get_or_init(|| log::set_logger(&Capture).is_ok()),
            "another logger owns this test process; wgpu-hal's HLSL cannot be captured"
        );
        /// Restores the log level and stops capturing even if `f` panics, so later
        /// tests neither format every debug log nor feed a stale capture.
        struct Armed(log::LevelFilter);
        impl Drop for Armed {
            fn drop(&mut self) {
                log::set_max_level(self.0);
                CAPTURED
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take();
            }
        }

        *CAPTURED.lock().unwrap() = Some((std::thread::current().id(), Vec::new()));
        let armed = Armed(log::max_level());
        log::set_max_level(log::LevelFilter::Debug);
        f();
        let messages = CAPTURED
            .lock()
            .unwrap()
            .as_mut()
            .map(|(_, messages)| std::mem::take(messages))
            .unwrap_or_default();
        drop(armed);
        messages
            .into_iter()
            .filter_map(|message| {
                // `Naga generated shader for {entry_point:?} at {naga_stage:?}:\n{source}`
                let rest = message.strip_prefix("Naga generated shader for \"")?;
                let (entry_point, rest) = rest.split_once("\" at ")?;
                let (stage, source) = rest.split_once(":\n")?;
                Some((
                    stage.to_string(),
                    entry_point.to_string(),
                    source.to_string(),
                ))
            })
            .collect()
    }
}

fn first_difference(expected: &str, actual: &str) -> String {
    expected
        .lines()
        .zip(actual.lines())
        .enumerate()
        .find(|(_, (expected, actual))| expected != actual)
        .map(|(line, (expected, actual))| {
            format!(
                "line {}:\n  build:  {expected}\n  wgpu:   {actual}",
                line + 1
            )
        })
        .unwrap_or_else(|| {
            format!(
                "one is a prefix of the other ({} vs {} lines)",
                expected.lines().count(),
                actual.lines().count()
            )
        })
}

#[test]
fn build_has_dx12_bytecode_for_every_wgpu_pipeline() {
    let labels = |shaders: &mut dyn Iterator<Item = &'static str>| shaders.collect::<Vec<_>>();
    let expected = labels(&mut shader::ALL.iter().map(|pipeline| pipeline.label));
    assert_eq!(
        labels(&mut DX12_SHADERS.iter().map(|shader| shader.label)),
        expected,
        "a Windows build must precompile every WGPU pipeline"
    );
    assert_eq!(
        labels(&mut DX12_SHADER_HLSL.iter().map(|shader| shader.label)),
        expected
    );
    for (shader, hlsl) in DX12_SHADERS.iter().zip(DX12_SHADER_HLSL) {
        for (stage, bytecode, entry, hlsl_entry) in [
            (
                "vertex",
                shader.vertex.bytecode,
                shader.vertex.entry_point,
                hlsl.vertex.entry_point,
            ),
            (
                "fragment",
                shader.fragment.bytecode,
                shader.fragment.entry_point,
                hlsl.fragment.entry_point,
            ),
        ] {
            assert!(
                bytecode.len() > 32 && bytecode.starts_with(b"DXBC"),
                "{} {stage}: not a DXBC container",
                shader.label
            );
            assert_eq!(entry, hlsl_entry, "{} {stage} entry point", shader.label);
        }
    }
}

/// The artifacts must be exactly the HLSL this WGPU generates for gpui's layouts. A WGPU,
/// Naga or layout change that alters WGPU's DX12 HLSL fails here, not on a user's GPU.
#[test]
fn precompiled_hlsl_is_exactly_what_wgpu_generates() -> anyhow::Result<()> {
    let _gpu_test_guard = crate::test_gpu::guard();
    let (adapter, device) = fxc_device(false)?;
    let dual_source_blending = adapter
        .features()
        .contains(wgpu::Features::DUAL_SOURCE_BLENDING);
    let generated = hlsl_capture::capture(|| {
        let mut pipelines = build_pipelines(
            &device,
            dual_source_blending,
            PipelineShaders::new(&[], RendererTier::Modern),
        );
        pipelines.ensure_damage_clear(&device);
        assert!(
            every_pipeline(&pipelines)
                .iter()
                .all(|(_, precompiled)| !precompiled)
        );
    });
    let mut compared = 0;
    let mut failures = Vec::new();
    for artifact in DX12_SHADER_HLSL {
        if artifact.label == shader::SUBPIXEL_SPRITES.label && !dual_source_blending {
            continue;
        }
        for (stage, expected) in [
            ("Vertex", &artifact.vertex),
            ("Fragment", &artifact.fragment),
        ] {
            let Some((_, _, actual)) = generated.iter().find(|(generated_stage, entry, _)| {
                generated_stage == stage && entry == expected.entry_point
            }) else {
                failures.push(format!(
                    "{} {stage}: WGPU generated no HLSL for {}",
                    artifact.label, expected.entry_point
                ));
                continue;
            };
            compared += 1;
            if actual != expected.source {
                failures.push(format!(
                    "{} {stage} differs from WGPU's HLSL at {}",
                    artifact.label,
                    first_difference(expected.source, actual)
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    assert!(
        compared >= 2 * (DX12_SHADER_HLSL.len() - 1),
        "only {compared} stages were compared"
    );
    Ok(())
}

#[test]
fn every_dx12_pipeline_is_created_from_precompiled_bytecode() -> anyhow::Result<()> {
    let _gpu_test_guard = crate::test_gpu::guard();
    let (adapter, device) = fxc_device(true)?;
    let dual_source_blending = adapter
        .features()
        .contains(wgpu::Features::DUAL_SOURCE_BLENDING);
    let shaders = PipelineShaders::for_device(&device, wgpu::Backend::Dx12, RendererTier::Modern);
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut pipelines = build_pipelines(&device, dual_source_blending, shaders);
    pipelines.ensure_damage_clear(&device);
    assert!(gpui::block_on(scope.pop()).is_none());
    let fallbacks = every_pipeline(&pipelines)
        .into_iter()
        .filter(|(_, precompiled)| !precompiled)
        .collect::<Vec<_>>();
    assert!(fallbacks.is_empty(), "compiled at runtime: {fallbacks:?}");
    assert!(
        !pipelines.compiled_wgsl(),
        "no WGSL module may be created when every pipeline is precompiled"
    );
    // A rebuild for a new alpha mode or surface stays on the fast path.
    let rebuilt = build_pipelines(
        &device,
        dual_source_blending,
        PipelineShaders::for_device(&device, wgpu::Backend::Dx12, RendererTier::Modern),
    );
    assert!(
        every_pipeline(&rebuilt)
            .iter()
            .all(|(_, precompiled)| *precompiled)
    );
    Ok(())
}

#[test]
fn precompiled_bytecode_needs_a_dx12_passthrough_device() -> anyhow::Result<()> {
    let _gpu_test_guard = crate::test_gpu::guard();
    let (_, without_passthrough) = fxc_device(false)?;
    let (_, with_passthrough) = fxc_device(true)?;
    for (device, backend, expected) in [
        (&without_passthrough, wgpu::Backend::Dx12, false),
        // DXBC is meaningless to any other backend, whatever features it reports.
        (&with_passthrough, wgpu::Backend::Vulkan, false),
        (&with_passthrough, wgpu::Backend::Dx12, true),
    ] {
        assert_eq!(
            PipelineShaders::for_device(device, backend, RendererTier::Modern).is_precompiled(),
            expected,
            "{backend:?}, passthrough={}",
            device
                .features()
                .contains(wgpu::Features::PASSTHROUGH_SHADERS)
        );
    }
    assert!(
        !PipelineShaders::for_device(&with_passthrough, wgpu::Backend::Dx12, RendererTier::WebGl2)
            .is_precompiled(),
        "the downlevel tier uses another shader dialect"
    );
    let pipelines = build_pipelines(
        &without_passthrough,
        false,
        PipelineShaders::for_device(
            &without_passthrough,
            wgpu::Backend::Dx12,
            RendererTier::Modern,
        ),
    );
    assert!(
        every_pipeline(&pipelines)
            .iter()
            .all(|(_, precompiled)| !precompiled)
    );
    Ok(())
}

/// Bytecode the device rejects must cost only that pipeline its fast path: it is compiled
/// from WGSL instead, the error never reaches the device, and every other pipeline keeps
/// its bytecode.
#[test]
fn rejected_bytecode_falls_back_to_wgsl_for_that_pipeline_only() -> anyhow::Result<()> {
    let _gpu_test_guard = crate::test_gpu::guard();
    let (_, device) = fxc_device(true)?;
    let mut corrupted = DX12_SHADERS.to_vec();
    for shader in &mut corrupted {
        if shader.label == shader::QUADS.label {
            // A pixel shader in the vertex slot: D3D12 rejects the pipeline state.
            std::mem::swap(&mut shader.vertex, &mut shader.fragment);
        }
        if shader.label == shader::SMOOTHED_SHADOWS.label {
            // Not a shader container at all.
            shader.fragment.bytecode = &b"DXBC but not really"[..];
        }
    }
    let corrupted: &'static [Dx12Shader] = Box::leak(corrupted.into_boxed_slice());
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let mut pipelines = build_pipelines(
        &device,
        false,
        PipelineShaders::new(corrupted, RendererTier::Modern),
    );
    pipelines.ensure_damage_clear(&device);
    assert!(gpui::block_on(internal.pop()).is_none());
    assert!(gpui::block_on(errors.pop()).is_none());
    for (label, precompiled) in every_pipeline(&pipelines) {
        let rejected = label == shader::QUADS.label || label == shader::SMOOTHED_SHADOWS.label;
        assert_eq!(precompiled, !rejected, "{label}");
    }
    assert!(pipelines.compiled_wgsl());
    Ok(())
}

#[cfg(feature = "test-support")]
mod rendering {
    //! Precompiled pipelines must draw exactly what WGPU's runtime-compiled ones draw.

    use super::*;
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

    fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(width), ScaledPixels(height)),
        )
    }

    fn mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: bounds(0.0, 0.0, SIZE as f32, SIZE as f32),
            ..Default::default()
        }
    }

    /// A headless context created as `WgpuContext::new_headless` creates one, on the
    /// instance that compiles as the build does (see `fxc_instance`).
    fn context() -> anyhow::Result<WgpuContext> {
        let instance = fxc_instance();
        let adapter = gpui::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))?;
        let device = gpui::block_on(WgpuContext::create_device(&adapter, None))?;
        let context = WgpuContext::from_created_device(instance, adapter, device)?;
        assert!(
            context
                .device
                .features()
                .contains(wgpu::Features::PASSTHROUGH_SHADERS),
            "WgpuContext must request passthrough shaders on DX12"
        );
        Ok(context)
    }

    /// One renderer per shader source, sharing a device.
    fn renderers(context: &WgpuContext) -> anyhow::Result<(WgpuRenderer, WgpuRenderer)> {
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
            "the default DX12 renderer must use precompiled bytecode: {precompiled_pipelines:?}"
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

    struct Tiles {
        monochrome: AtlasTile,
        subpixel: Option<AtlasTile>,
        image: AtlasTile,
    }

    fn atlas_tiles(renderer: &WgpuRenderer) -> anyhow::Result<Tiles> {
        Ok(Tiles {
            monochrome: glyph_tile(renderer, GlyphRenderMode::Grayscale)?,
            subpixel: renderer
                .dual_source_blending
                .then(|| glyph_tile(renderer, GlyphRenderMode::Subpixel))
                .transpose()?,
            image: image_tile(renderer)?,
        })
    }

    fn backdrop(scene: &mut Scene) {
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
    fn scenes(tiles: &Tiles) -> Vec<(&'static str, Scene)> {
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

    fn assert_identical(name: &str, precompiled: &image::RgbaImage, compiled: &image::RgbaImage) {
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

    #[test]
    fn precompiled_pipelines_render_exactly_like_runtime_compiled_ones() -> anyhow::Result<()> {
        let _gpu_test_guard = crate::test_gpu::guard();
        let context = context()?;
        let (mut precompiled, mut compiled) = renderers(&context)?;
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
    #[test]
    fn precompiled_retained_partial_frames_and_blits_match_runtime_compiled() -> anyhow::Result<()>
    {
        let _gpu_test_guard = crate::test_gpu::guard();
        let context = context()?;
        let (mut precompiled, mut compiled) = renderers(&context)?;
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
                if copy_supported { "copy" } else { "blit" }
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

    /// A renderer whose bytecode was rejected still renders correctly from WGSL.
    #[test]
    fn rejected_bytecode_renders_through_the_wgsl_fallback() -> anyhow::Result<()> {
        let _gpu_test_guard = crate::test_gpu::guard();
        let context = context()?;
        let (mut precompiled, mut fallback) = renderers(&context)?;
        let mut rejected = DX12_SHADERS.to_vec();
        for shader in &mut rejected {
            std::mem::swap(&mut shader.vertex, &mut shader.fragment);
        }
        let rejected: &'static [Dx12Shader] = Box::leak(rejected.into_boxed_slice());
        let config = fallback.target.configuration().clone();
        let samples = fallback.rendering_params.path_sample_count;
        let dual_source_blending = fallback.dual_source_blending;
        let resources = fallback.resources_mut();
        resources.pipelines = WgpuPipelines::with_shaders(
            &resources.device,
            &resources.bind_group_layouts,
            config.format,
            config.alpha_mode,
            samples,
            dual_source_blending,
            resources.renderer_tier,
            PipelineShaders::new(rejected, resources.renderer_tier),
        );
        assert!(
            every_pipeline(&fallback.resources().pipelines)
                .iter()
                .all(|(_, precompiled)| !precompiled)
        );
        let tiles = atlas_tiles(&precompiled)?;
        atlas_tiles(&fallback)?;
        for (name, scene) in scenes(&tiles).into_iter().take(4) {
            assert_identical(
                name,
                &precompiled.render_to_image(&scene)?,
                &fallback.render_to_image(&scene)?,
            );
        }
        assert!(context.uncaptured_error_slot().lock().unwrap().is_none());
        Ok(())
    }
}
