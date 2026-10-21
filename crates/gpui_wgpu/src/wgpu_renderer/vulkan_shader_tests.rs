//! Exercise the actual Vulkan backend even on Windows, where the default is DX12.

use super::{
    pipelines::{PipelineShaders, WgpuBindGroupLayouts, WgpuPipelines},
    precompiled_shader_test_support::every_pipeline,
};
use crate::{RendererTier, WgpuContext};
use gpui_render::{artifacts::VULKAN_SHADERS, shaders::interface as shader};

fn instance() -> wgpu::Instance {
    crate::wgpu_context::NativeBackend::Vulkan
        .instance(None)
        .raw
}

fn adapter(instance: &wgpu::Instance) -> anyhow::Result<wgpu::Adapter> {
    Ok(gpui::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
    )?)
}

pub(super) fn context() -> anyhow::Result<Option<WgpuContext>> {
    let instance = instance();
    let adapter = match adapter(&instance) {
        Ok(adapter) => adapter,
        // GL-only developer machines and Windows CI may have no Vulkan ICD.
        // The dedicated Vulkan CI step explicitly requires its installed Mesa driver.
        Err(error) if std::env::var_os("GPUI_REQUIRE_VULKAN_TESTS").is_some() => return Err(error),
        Err(error) => {
            eprintln!("skipping Vulkan GPU test: no Vulkan adapter: {error:#}");
            return Ok(None);
        }
    };
    let created = gpui::block_on(WgpuContext::create_device(&adapter, None))?;
    let context = WgpuContext::from_created_device(instance, adapter, created)?;
    assert!(
        context
            .device
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS)
    );
    Ok(Some(context))
}

fn pipelines(device: &wgpu::Device, shaders: PipelineShaders) -> WgpuPipelines {
    WgpuPipelines::with_shaders(
        device,
        &WgpuBindGroupLayouts::new(device, RendererTier::Modern),
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::CompositeAlphaMode::Opaque,
        1,
        device
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING),
        RendererTier::Modern,
        shaders,
    )
}

#[test]
fn every_vulkan_pipeline_uses_precompiled_spirv() -> anyhow::Result<()> {
    let _guard = crate::test_gpu::guard();
    assert_eq!(VULKAN_SHADERS.len(), shader::ALL.len());
    for spec in shader::ALL {
        let artifact = VULKAN_SHADERS
            .iter()
            .find(|artifact| artifact.label == spec.label)
            .unwrap();
        assert_eq!(artifact.vertex.entry_point, spec.vertex_entry);
        assert_eq!(artifact.fragment.entry_point, spec.fragment_entry);
        for stage in [artifact.vertex, artifact.fragment] {
            assert_eq!(&stage.words[..2], &[0x07230203, 0x00010000]);
        }
    }
    let Some(context) = context()? else {
        return Ok(());
    };
    let device = &context.device;
    let shaders = PipelineShaders::for_device(device, wgpu::Backend::Vulkan, RendererTier::Modern);
    assert!(shaders.is_precompiled());
    let mut pipelines = pipelines(device, shaders);
    pipelines.ensure_damage_clear(device);
    assert!(!pipelines.compiled_wgsl());
    for (label, precompiled) in every_pipeline(&pipelines) {
        assert!(precompiled, "{label} fell back to WGSL");
    }
    assert!(context.uncaptured_error_slot().lock().unwrap().is_none());
    Ok(())
}

#[test]
fn vulkan_passthrough_requires_matching_backend_tier_and_device_feature() -> anyhow::Result<()> {
    let _guard = crate::test_gpu::guard();
    let Some(context) = context()? else {
        return Ok(());
    };
    for backend in [
        wgpu::Backend::Gl,
        wgpu::Backend::Metal,
        wgpu::Backend::BrowserWebGpu,
    ] {
        assert!(
            !PipelineShaders::for_device(&context.device, backend, RendererTier::Modern)
                .is_precompiled()
        );
    }
    assert!(
        !PipelineShaders::for_device(&context.device, wgpu::Backend::Vulkan, RendererTier::WebGl2)
            .is_precompiled()
    );
    let instance = instance();
    let adapter = adapter(&instance)?;
    let (device, _queue) = gpui::block_on(
        adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits::downlevel_defaults()
                .using_resolution(adapter.limits())
                .using_alignment(adapter.limits()),
            ..Default::default()
        }),
    )?;
    let shaders = PipelineShaders::for_device(&device, wgpu::Backend::Vulkan, RendererTier::Modern);
    assert!(!shaders.is_precompiled());
    let pipelines = pipelines(&device, shaders);
    assert!(pipelines.compiled_wgsl());
    assert!(
        every_pipeline(&pipelines)
            .iter()
            .all(|(_, precompiled)| !precompiled)
    );
    Ok(())
}

#[cfg(feature = "test-support")]
#[test]
fn vulkan_precompiled_pipelines_render_like_runtime_wgsl() -> anyhow::Result<()> {
    let _guard = crate::test_gpu::guard();
    let Some(context) = context()? else {
        return Ok(());
    };
    super::precompiled_shader_test_support::compare_rendering(&context)
}

#[cfg(feature = "test-support")]
#[test]
fn vulkan_precompiled_retained_frames_and_blits_match_runtime_wgsl() -> anyhow::Result<()> {
    let _guard = crate::test_gpu::guard();
    let Some(context) = context()? else {
        return Ok(());
    };
    super::precompiled_shader_test_support::compare_retained(&context)
}

// Exercise the surface shader directly: Windows' native capture import uses DX11,
// while Linux uses WGPU textures. Both share this pipeline and dynamic uniform ABI.
#[cfg(any(target_os = "linux", target_os = "freebsd", feature = "wgpu-surfaces"))]
#[test]
fn vulkan_surface_bindings_and_dynamic_offsets_match_runtime_wgsl() -> anyhow::Result<()> {
    use gpui_render::shaders::{common::SurfaceColorFormat, surface::SurfaceUniforms};
    use wgpu::util::DeviceExt;

    let _guard = crate::test_gpu::guard();
    let Some(context) = context()? else {
        return Ok(());
    };
    let device = &context.device;
    let queue = &context.queue;
    let layouts = WgpuBindGroupLayouts::new(device, RendererTier::Modern);
    let precompiled = pipelines(
        device,
        PipelineShaders::for_device(device, wgpu::Backend::Vulkan, RendererTier::Modern),
    );
    let runtime = pipelines(device, PipelineShaders::new(&[], RendererTier::Modern));
    let uniform_buffer = |bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        })
    };
    let globals = uniform_buffer(bytemuck::cast_slice(&[64.0f32, 8.0, 0.0, 0.0]));
    let font = uniform_buffer(&[0; 32]);
    let globals = layouts.create_globals(
        device,
        "surface_test_globals",
        globals.as_entire_buffer_binding(),
        font.as_entire_buffer_binding(),
    );
    let stride = std::mem::size_of::<SurfaceUniforms>()
        .next_multiple_of(device.limits().min_uniform_buffer_offset_alignment as usize);
    let bounds = |x, width| {
        gpui::Bounds::new(
            gpui::point(gpui::ScaledPixels(x), gpui::ScaledPixels(0.0)),
            gpui::size(gpui::ScaledPixels(width), gpui::ScaledPixels(8.0)),
        )
    };
    let mut bytes = vec![0; stride * 3 + std::mem::size_of::<SurfaceUniforms>()];
    for index in 0..3 {
        let value = SurfaceUniforms {
            bounds: bounds(index as f32 * 20.0, 16.0).into(),
            content_mask: gpui::ContentMask {
                bounds: bounds(0.0, 64.0),
                ..Default::default()
            }
            .into(),
            color_format: if index == 1 {
                SurfaceColorFormat::Yuv
            } else {
                SurfaceColorFormat::Rgba
            },
            opacity: 0.75,
            padding0: 0,
            padding1: 0,
            padding2: 0,
            padding3: 0,
            padding4: 0,
            padding5: 0,
        };
        let value = shader::bytes_of(&value);
        bytes[index * stride..index * stride + value.len()].copy_from_slice(value);
    }
    let uniforms = uniform_buffer(&bytes);
    let texture_desc = |size, usage| wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage,
        view_formats: &[],
    };
    let input_size = wgpu::Extent3d {
        width: 2,
        height: 2,
        depth_or_array_layers: 1,
    };
    let input_desc = texture_desc(input_size, wgpu::TextureUsages::TEXTURE_BINDING);
    let color = device.create_texture_with_data(
        queue,
        &input_desc,
        wgpu::util::TextureDataOrder::LayerMajor,
        &[
            20, 80, 160, 255, 180, 60, 30, 255, 40, 160, 80, 255, 200, 220, 40, 255,
        ],
    );
    let chroma = device.create_texture_with_data(
        queue,
        &input_desc,
        wgpu::util::TextureDataOrder::LayerMajor,
        &[
            100, 140, 0, 255, 150, 80, 0, 255, 120, 160, 0, 255, 180, 130, 0, 255,
        ],
    );
    let sampler = device.create_sampler(&Default::default());
    let data = layouts.create_surface(
        device,
        wgpu::BufferBinding {
            buffer: &uniforms,
            offset: 0,
            size: std::num::NonZeroU64::new(std::mem::size_of::<SurfaceUniforms>() as u64),
        },
        &color.create_view(&Default::default()),
        &chroma.create_view(&Default::default()),
        &sampler,
    );
    let render = |pipelines: &WgpuPipelines| -> anyhow::Result<Vec<u8>> {
        let size = wgpu::Extent3d {
            width: 64,
            height: 8,
            depth_or_array_layers: 1,
        };
        let output = device.create_texture(&texture_desc(
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        ));
        let view = output.create_view(&Default::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64 * 8 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipelines.surfaces);
            pass.set_bind_group(0, &globals, &[]);
            for index in 0..3 {
                pass.set_bind_group(1, &data, &[(index * stride) as u32]);
                pass.draw(0..pipelines.surfaces.fixed_vertex_count(), 0..1);
            }
        }
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            size,
        );
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let pixels = readback.slice(..).get_mapped_range()?.to_vec();
        readback.unmap();
        Ok(pixels)
    };
    let expected = render(&runtime)?;
    assert!(expected.iter().any(|byte| *byte != 0));
    assert_eq!(render(&precompiled)?, expected);
    assert!(context.uncaptured_error_slot().lock().unwrap().is_none());
    Ok(())
}
