//! WGSL linked at Cargo build time from the Rust-authored shader modules.

use std::marker::PhantomData;

pub const BASE_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/gpui_base.wgsl"));
pub const SUBPIXEL_DUAL_SOURCE_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/gpui_subpixel_dual_source.wgsl"));
/// Downlevel (WebGL2/GLES) dialect: scene arrays travel via an `rgba32uint` data texture.
pub const BASE_DOWNLEVEL_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/gpui_base_downlevel.wgsl"));

#[derive(Clone, Copy)]
pub struct GeneratedBinding {
    pub binding: u32,
    pub visibility: u32,
    pub kind: GeneratedBindingKind,
}

#[derive(Clone, Copy)]
pub enum GeneratedBindingKind {
    Uniform(u64),
    StorageRead(u64),
    Texture2dFloat,
    FilteringSampler,
    /// Downlevel only: the `rgba32uint` scene-data texture.
    DataTexture,
    /// Downlevel only: the per-draw batch base uniform, bound with a dynamic offset.
    RangeUniform,
}

include!(concat!(env!("OUT_DIR"), "/shader_interface.rs"));

/// D3D11 bytecode generated from HLSL at build time.
///
/// Keeping both stages together makes it impossible to pass textual HLSL to a
/// D3D11 creation API by accident.
#[derive(Clone, Copy)]
pub struct Dx11Bytecode {
    pub vertex: &'static [u8],
    pub fragment: &'static [u8],
    /// Present for every pipeline whose vertex shader reads `instance_index`.
    ///
    /// Direct3D 11 does not add `StartInstanceLocation` to `SV_InstanceID`, so the
    /// generated vertex shader adds [`Dx11DrawConstants::first_instance`] itself. The
    /// renderer must upload a [`Dx11DrawConstants`] to this register before every draw.
    pub draw_constants: Option<Dx11DrawConstantsBinding>,
}

/// Where an instanced pipeline expects its [`Dx11DrawConstants`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dx11DrawConstantsBinding {
    /// Constant-buffer register (`b<register>`), bound to the vertex stage.
    pub register: u32,
}

/// The per-draw cbuffer Naga's HLSL backend declares as `NagaConstants`.
///
/// Layout mirrors the generated `struct NagaConstants { int first_vertex; int
/// first_instance; uint other; }`, padded to the 16-byte cbuffer granule.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dx11DrawConstants {
    /// Added to `SV_VertexID` to form `vertex_index`.
    pub first_vertex: i32,
    /// Added to `SV_InstanceID` to form `instance_index`.
    pub first_instance: i32,
    /// Unused by graphics pipelines; Naga reserves it for compute dispatch sizes.
    pub other: u32,
    pub padding: u32,
}

impl Dx11DrawConstants {
    /// Constants for a draw whose instances start at `first_instance` in the bound buffer.
    pub fn for_instances(first_instance: u32) -> Self {
        Self {
            first_vertex: 0,
            first_instance: i32::try_from(first_instance)
                .expect("instance bases are bounded by the D3D11 buffer limit"),
            other: 0,
            padding: 0,
        }
    }
}

const _: () = assert!(std::mem::size_of::<Dx11DrawConstants>() == 16);

/// The DX11 artifact state for this build host.
///
/// `D3DCompile` exists only on Windows. A Windows-targeting cross-build from a
/// non-Windows host is rejected by the build script, with no runtime-compiler
/// escape hatch.
#[derive(Clone, Copy)]
pub enum Dx11Shader {
    Sm50(Dx11Bytecode),
    NativeWindowsBuildRequired,
}

/// One stage of a GLSL program generated from the downlevel shader dialect.
///
/// No in-tree renderer consumes this source directly. `gpui_wgpu` provides
/// WGSL to WGPU, which owns the final binding assignment for each GL context.
/// These artifacts describe the requested GLSL profile; driver compilation
/// and program linking still belong to a future raw-GL consumer.
pub struct GlslShaderStage {
    pub source: &'static str,
}

/// Desktop core GLSL 3.30 profile marker.
pub enum Glsl330 {}

/// OpenGL ES 3.00 profile marker.
pub enum Gles300 {}

/// A pair of GLSL stages generated from validated WGSL for one profile.
pub struct GlslShader<Profile> {
    pub vertex: GlslShaderStage,
    pub fragment: GlslShaderStage,
    _profile: PhantomData<fn() -> Profile>,
}

impl<Profile> GlslShader<Profile> {
    pub const fn new(vertex: GlslShaderStage, fragment: GlslShaderStage) -> Self {
        Self {
            vertex,
            fragment,
            _profile: PhantomData,
        }
    }
}

pub struct NativeShader {
    pub label: &'static str,
    /// The shared pipeline description this shader was generated for: topology, vertex
    /// count and data layout are the same on every backend.
    pub pipeline: &'static crate::shaders::interface::Pipeline,
    pub vertex_entry: &'static str,
    pub fragment_entry: &'static str,
    pub dx11: Dx11Shader,
    /// GLSL 3.30 core, using the data-texture downlevel transport.
    pub glsl_330: GlslShader<Glsl330>,
    /// GLSL ES 3.00, using the same downlevel transport.
    pub gles_300: GlslShader<Gles300>,
    pub msl: &'static str,
    /// `msl`, precompiled when the build had Apple's Metal Toolchain. Loading it
    /// skips Metal's runtime source compiler.
    pub metallib: Option<&'static [u8]>,
}

include!(concat!(env!("OUT_DIR"), "/native_shaders.rs"));

/// One stage of a [`Dx12Shader`].
#[derive(Clone, Copy, Debug)]
pub struct Dx12ShaderStage {
    /// The HLSL entry point the bytecode was compiled from.
    pub entry_point: &'static str,
    /// Shader model 5.1 DXBC.
    pub bytecode: &'static [u8],
}

/// DX12 bytecode for one WGPU pipeline, compiled at build time from the HLSL that
/// WGPU's DX12 backend generates for this pipeline's layout.
///
/// [`DX12_SHADERS`] is empty unless the build targeted Windows from a Windows host,
/// in which case WGPU compiles its DX12 shaders at runtime.
#[derive(Clone, Copy, Debug)]
pub struct Dx12Shader {
    /// The [`crate::shaders::interface::Pipeline`] label this pipeline implements.
    pub label: &'static str,
    pub vertex: Dx12ShaderStage,
    pub fragment: Dx12ShaderStage,
}

/// One stage of a [`Dx12ShaderHlsl`].
#[derive(Clone, Copy, Debug)]
pub struct Dx12HlslStage {
    pub entry_point: &'static str,
    pub source: &'static str,
}

/// The HLSL each [`Dx12Shader`] was compiled from. Tests compare it with the HLSL
/// WGPU generates at runtime; nothing else should need it.
#[derive(Clone, Copy, Debug)]
pub struct Dx12ShaderHlsl {
    pub label: &'static str,
    pub vertex: Dx12HlslStage,
    pub fragment: Dx12HlslStage,
}

include!(concat!(env!("OUT_DIR"), "/dx12_shaders.rs"));

#[path = "vulkan_artifacts.rs"]
mod vulkan;
pub use vulkan::{VULKAN_SHADERS, VulkanShader, VulkanShaderStage};

/// The precompiled DX12 shader for the pipeline with `label`, if this build has one.
pub fn dx12_shader(label: &str) -> Option<&'static Dx12Shader> {
    DX12_SHADERS.iter().find(|shader| shader.label == label)
}

#[cfg(test)]
mod tests {
    use super::BASE_DOWNLEVEL_WGSL;

    fn generated_function<'a>(source: &'a str, name: &str) -> &'a str {
        let marker = format!("fn {name}");

        source
            .split_once(&marker)
            .and_then(|(_, source)| source.split_once("\n}"))
            .map(|(source, _)| source)
            .unwrap_or_else(|| panic!("generated {name} function must exist"))
    }

    #[test]
    fn downlevel_background_array_uses_host_element_stride() {
        let decoder = generated_function(BASE_DOWNLEVEL_WGSL, "dl_load_Background_impl");
        let element_stride = std::mem::size_of::<gpui::LinearColorStop>() / 4;
        let second_element_offset = 7 + element_stride;

        assert!(
            decoder.contains("base + 7u"),
            "Background decoder must load its first color stop at word 7"
        );
        assert!(
            decoder.contains(&format!("base + {second_element_offset}u")),
            "Background decoder must use the host LinearColorStop stride"
        );
        assert!(
            !decoder.contains("base + 17u)), dl_scene_word"),
            "Background decoder must not read padding as the second color stop"
        );
    }

    #[test]
    fn downlevel_quad_decoder_uses_host_dash_offsets_and_stride() {
        let decoder = generated_function(BASE_DOWNLEVEL_WGSL, "dl_load_Quad_impl");
        let loader = generated_function(BASE_DOWNLEVEL_WGSL, "dl_load_QUADS");
        let dash_length_word = std::mem::offset_of!(gpui::Quad, border_dashed_length) / 4;
        let dash_gap_word = std::mem::offset_of!(gpui::Quad, border_dashed_gap) / 4;
        let stride_words = std::mem::size_of::<gpui::Quad>() / 4;

        assert!(
            decoder.contains(&format!("base, {dash_length_word}u")),
            "Quad decoder must load border_dashed_length from its host offset"
        );
        assert!(
            decoder.contains(&format!("base, {dash_gap_word}u")),
            "Quad decoder must load border_dashed_gap from its host offset"
        );
        assert!(
            loader.contains(&format!("i * {stride_words}u")),
            "Quad decoder must advance by the host Quad stride"
        );
    }
}
