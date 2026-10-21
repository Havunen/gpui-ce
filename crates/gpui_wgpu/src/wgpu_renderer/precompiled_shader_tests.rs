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
/// targets the artifacts' shader model, and with gpui's default instance flags, which
/// `gpui_render/build.rs` mirrors. Its WGSL pipelines then run the same bytecode as the
/// artifacts, so a pixel difference between the two is a binding or layout mismatch,
/// never two compilations rounding differently.
fn fxc_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::default(),
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

use super::precompiled_shader_test_support::every_pipeline;

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
            // WGPU compiles with `D3DCOMPILE_DEBUG`, which embeds an `SPDB` chunk, exactly
            // when debug assertions are on. Bytecode compiled otherwise runs differently
            // than WGPU's own would in this build.
            assert_eq!(
                dxbc_chunks(bytecode).iter().any(|chunk| *chunk == b"SPDB"),
                cfg!(debug_assertions),
                "{} {stage}: compiled with other FXC flags than WGPU uses in this build",
                shader.label
            );
        }
    }
}

/// The four-character codes of a DXBC container's chunks.
fn dxbc_chunks(bytecode: &[u8]) -> Vec<&[u8]> {
    let u32_at = |offset: usize| {
        u32::from_le_bytes(bytecode[offset..offset + 4].try_into().unwrap()) as usize
    };
    (0..u32_at(28))
        .map(|chunk| u32_at(32 + 4 * chunk))
        .map(|offset| &bytecode[offset..offset + 4])
        .collect()
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
        // A backend without artifacts cannot use DXBC, whatever features it reports.
        (&with_passthrough, wgpu::Backend::Metal, false),
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

    use super::super::precompiled_shader_test_support::*;
    use super::*;
    use crate::WgpuContext;

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

    #[test]
    fn precompiled_pipelines_render_exactly_like_runtime_compiled_ones() -> anyhow::Result<()> {
        let _gpu_test_guard = crate::test_gpu::guard();
        compare_rendering(&context()?)
    }

    #[test]
    fn precompiled_retained_partial_frames_and_blits_match_runtime_compiled() -> anyhow::Result<()>
    {
        let _gpu_test_guard = crate::test_gpu::guard();
        compare_retained(&context()?)
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
