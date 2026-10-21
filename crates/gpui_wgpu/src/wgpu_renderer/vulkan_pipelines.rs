use super::*;
use gpui_render::artifacts::VulkanShader;

/// Driver or pipeline creation errors fall back to WGSL. Unlike DX12, Vulkan does
/// not validate arbitrary bytecode; the safety contract must hold before this call.
pub(super) fn create_pipeline(
    device: &wgpu::Device,
    specification: shader::Pipeline,
    layout: &wgpu::PipelineLayout,
    color_target: &wgpu::ColorTargetState,
    sample_count: u32,
    shader: &VulkanShader,
) -> Option<WgpuRenderPipeline> {
    let scopes = [
        device.push_error_scope(wgpu::ErrorFilter::Validation),
        device.push_error_scope(wgpu::ErrorFilter::Internal),
    ];
    let module = |words: &'static [u32], entry_point: &'static str| {
        // SAFETY: these are validated, build-generated Naga shaders using the same
        // descriptor layout and coordinate convention as wgpu-hal 30's Vulkan backend.
        // The build emits portable SPIR-V with bounds checks and bounded loops. Tests
        // compare every pipeline's output with WGPU's normal WGSL translation path.
        unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(specification.label),
                entry_points: std::borrow::Cow::Borrowed(&[wgpu::PassthroughShaderEntryPoint {
                    name: std::borrow::Cow::Borrowed(entry_point),
                    workgroup_size: (0, 0, 0),
                }]),
                spirv: Some(std::borrow::Cow::Borrowed(words)),
                ..Default::default()
            })
        }
    };
    let vertex = module(shader.vertex.words, shader.vertex.entry_point);
    let fragment = module(shader.fragment.words, shader.fragment.entry_point);
    let pipeline = create_render_pipeline(
        device,
        specification,
        layout,
        color_target,
        sample_count,
        PipelineStages {
            vertex: (&vertex, shader.vertex.entry_point),
            fragment: (&fragment, shader.fragment.entry_point),
        },
        true,
    );
    let mut rejected = None;
    for scope in scopes.into_iter().rev() {
        if let Some(error) = gpui::block_on(scope.pop()) {
            rejected.get_or_insert(error);
        }
    }
    match rejected {
        None => Some(pipeline),
        Some(error) => {
            log::warn!(
                "precompiled Vulkan shaders for {} were rejected; compiling WGSL instead: {error}",
                specification.label
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_pipeline_errors_are_scoped_and_missing_artifacts_fall_back() -> anyhow::Result<()> {
        let _guard = crate::test_gpu::guard();
        let Some(context) = crate::wgpu_renderer::vulkan_shader_tests::context()? else {
            return Ok(());
        };
        let device = &context.device;
        let layouts = WgpuBindGroupLayouts::new(device, RendererTier::Modern);
        let layout =
            create_pipeline_layout(device, "vulkan_error_test", &layouts, &layouts.instances);
        let shader = gpui_render::artifacts::VULKAN_SHADERS
            .iter()
            .find(|shader| shader.label == shader::QUADS.label)
            .unwrap();
        // Test rejection with valid SPIR-V and an invalid color target. Feeding broken
        // SPIR-V to passthrough would itself violate the unsafe API's contract.
        let outer = device.push_error_scope(wgpu::ErrorFilter::Validation);
        assert!(
            create_pipeline(
                device,
                shader::QUADS,
                &layout,
                &color_target(wgpu::TextureFormat::Depth32Float, None),
                1,
                shader,
            )
            .is_none()
        );
        assert!(gpui::block_on(outer.pop()).is_none());
        let mut shaders =
            PipelineShaders::for_device(device, wgpu::Backend::Vulkan, RendererTier::Modern);
        shaders.vulkan = &[];
        let pipeline = shaders.create(
            device,
            shader::QUADS,
            &layout,
            &color_target(wgpu::TextureFormat::Bgra8Unorm, None),
            1,
        );
        assert!(!pipeline.is_precompiled());
        assert!(shaders.base.get().is_some());
        assert!(context.uncaptured_error_slot().lock().unwrap().is_none());
        Ok(())
    }
}
