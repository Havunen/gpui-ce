/// One stage compiled to portable SPIR-V at build time for WGPU's Vulkan backend.
#[derive(Clone, Copy, Debug)]
pub struct VulkanShaderStage {
    pub entry_point: &'static str,
    pub words: &'static [u32],
}

/// Precompiled stages for one modern WGPU pipeline. The table is empty on Apple
/// and wasm targets, where GPUI does not use Vulkan.
#[derive(Clone, Copy, Debug)]
pub struct VulkanShader {
    pub label: &'static str,
    pub vertex: VulkanShaderStage,
    pub fragment: VulkanShaderStage,
}

// Decode at compile time to guarantee u32 alignment and native endianness without
// allocating, retaining duplicate byte arrays, or casting unaligned include_bytes!.
#[allow(dead_code)] // No artifacts on Apple/wasm.
const fn spirv_words<const N: usize>(bytes: &[u8]) -> [u32; N] {
    assert!(bytes.len() == N * 4);
    let mut words = [0; N];
    let mut i = 0;
    while i < N {
        let offset = i * 4;
        words[i] = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        i += 1;
    }
    words
}

include!(concat!(env!("OUT_DIR"), "/vulkan_shaders.rs"));

#[cfg(all(test, not(target_family = "wasm"), not(target_vendor = "apple")))]
mod tests {
    use super::*;

    #[test]
    fn vulkan_spirv_round_trips_through_naga_validation() {
        assert_eq!(VULKAN_SHADERS.len(), crate::shaders::interface::ALL.len());
        for shader in VULKAN_SHADERS {
            for (stage, expected) in [
                (shader.vertex, naga::ShaderStage::Vertex),
                (shader.fragment, naga::ShaderStage::Fragment),
            ] {
                let module = naga::front::spv::Frontend::new(
                    stage.words.iter().copied(),
                    &naga::front::spv::Options::default(),
                )
                .parse()
                .unwrap_or_else(|error| panic!("{}: {error}", stage.entry_point));
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap_or_else(|error| panic!("{}: {error:?}", stage.entry_point));
                assert_eq!(module.entry_points.len(), 1);
                assert_eq!(module.entry_points[0].name, stage.entry_point);
                assert_eq!(module.entry_points[0].stage, expected);
            }
        }
    }
}
