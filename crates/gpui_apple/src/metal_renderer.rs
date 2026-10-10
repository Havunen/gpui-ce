use crate::metal_atlas::MetalAtlas;
use anyhow::Result;
use block::ConcreteBlock;
use core_graphics::geometry::CGSize;
use gpui::{
    AtlasTextureId, Bounds, Corners, DevicePixels, FilterRenderTarget, MAX_FILTER_GROUP_DEPTH,
    MonochromeSprite, PaintSurface, Path, PolychromeSprite, PrimitiveBatch, Quad, RenderCommand,
    ScaledPixels, Scene, Shadow, Size, SurfaceSource, Underline, size,
};
use gpui_render::{
    artifacts::{NATIVE_SHADERS, NativeShader},
    blur::{
        BlurAxis, BlurKernel, BlurUniforms, GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS, ScissorRectangle,
        downsampled_dimension,
    },
    path_types::{self, PathRasterizationVertex},
    shaders::{
        common::{FontRasterizationUniforms, GlobalUniforms, ShaderBool, SurfaceColorFormat},
        surface::SurfaceUniforms,
    },
};
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use image::RgbaImage;

use core_foundation::base::TCFType;
use core_video::{
    image_buffer::TCVImageBuffer as _, metal_texture::CVMetalTextureGetTexture,
    metal_texture_cache::CVMetalTextureCache,
    pixel_buffer::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CAMetalLayer, CommandQueue, MTLGPUFamily, MTLPixelFormat, MTLResourceOptions, MTLScissorRect,
    NSRange, NSUInteger, RenderPassColorAttachmentDescriptorRef, SamplerDescriptor,
};
use objc2_quartz_core::{CAAutoresizingMask, CAMetalLayer as Objc2CAMetalLayer};
use parking_lot::Mutex;
use smallvec::SmallVec;
use wgsl_rs::std::{vec2f, vec4f};

use std::{cell::Cell, mem, ptr, sync::Arc};

/// Renderer options (`GPUI_METAL_EXPERIMENTS`, comma separated).
///
/// On by default, each with an opt-out for comparisons:
/// - path prepass (`no-path-prepass`): every path batch of a frame is rasterized
///   in at most [`MAX_PATH_LAYERS`] cropped passes before the main pass, instead
///   of ending the main pass around each batch with a full-window MSAA pass.
/// - opaque quads (`no-opaque-quads`): opaque rectangles are drawn without
///   blending, so the tile hardware skips fragments a later opaque quad hides.
///
/// Measurement only (render incorrectly): `skip-paths`, `skip-shadows`,
/// `skip-quads`, `skip-sprites` drop a primitive kind to attribute GPU time.
#[derive(Clone, Copy, Debug, Default)]
struct Experiments {
    path_prepass: bool,
    opaque_quads: bool,
    skip_paths: bool,
    // Measurement only: drop a primitive kind to attribute GPU time.
    skip_shadows: bool,
    skip_quads: bool,
    skip_sprites: bool,
}

fn experiments() -> Experiments {
    static EXPERIMENTS: std::sync::OnceLock<Experiments> = std::sync::OnceLock::new();
    *EXPERIMENTS.get_or_init(|| {
        let value = std::env::var("GPUI_METAL_EXPERIMENTS").unwrap_or_default();
        let flags: Vec<&str> = value.split(',').map(str::trim).collect();
        Experiments {
            path_prepass: !flags.contains(&"no-path-prepass"),
            opaque_quads: !flags.contains(&"no-opaque-quads"),
            skip_paths: flags.contains(&"skip-paths"),
            skip_shadows: flags.contains(&"skip-shadows"),
            skip_quads: flags.contains(&"skip-quads"),
            skip_sprites: flags.contains(&"skip-sprites"),
        }
    })
}

fn dump_frame_prefix() -> Option<&'static std::path::Path> {
    static PREFIX: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    PREFIX
        .get_or_init(|| {
            std::env::var_os("GPUI_DUMP_FRAME")
                .filter(|value| !value.is_empty())
                .map(std::path::PathBuf::from)
        })
        .as_deref()
}

/// Path batches of one frame are grouped into layers whose sampled regions are
/// pairwise disjoint; each layer is one rasterization pass into its own target.
const MAX_PATH_LAYERS: usize = 3;

struct PathLayerPlan {
    /// Right and bottom edge of the layer's geometry, in device pixels.
    extent: (u32, u32),
    /// Regions compositing samples (sprite bounds), grown by one pixel.
    regions: Vec<Bounds<ScaledPixels>>,
    union: Option<Bounds<ScaledPixels>>,
    /// Indices into the scene's render commands.
    batches: SmallVec<[usize; 8]>,
}

struct PathPrepassPlan {
    /// The layer of each render command, for path batches with geometry.
    layer_of_command: Vec<Option<u8>>,
    layers: SmallVec<[PathLayerPlan; MAX_PATH_LAYERS]>,
}

fn grown(bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
    Bounds {
        origin: gpui::point(
            ScaledPixels(bounds.origin.x.0 - 1.0),
            ScaledPixels(bounds.origin.y.0 - 1.0),
        ),
        size: size(
            ScaledPixels(bounds.size.width.0 + 2.0),
            ScaledPixels(bounds.size.height.0 + 2.0),
        ),
    }
}

fn overlaps(a: &Bounds<ScaledPixels>, b: &Bounds<ScaledPixels>) -> bool {
    a.origin.x.0 < b.origin.x.0 + b.size.width.0
        && b.origin.x.0 < a.origin.x.0 + a.size.width.0
        && a.origin.y.0 < b.origin.y.0 + b.size.height.0
        && b.origin.y.0 < a.origin.y.0 + a.size.height.0
}

/// Plans the path prepass for a frame, or `None` when the batches need more
/// layers than allowed (the frame then renders paths the original way).
fn plan_path_prepass(scene: &Scene, viewport: Size<DevicePixels>) -> Option<PathPrepassPlan> {
    let commands = scene.render_commands();
    let mut plan = PathPrepassPlan {
        layer_of_command: vec![None; commands.len()],
        layers: SmallVec::new(),
    };
    for (index, command) in commands.iter().enumerate() {
        let RenderCommand::Batch(PrimitiveBatch::Paths {
            range,
            rasterization_vertex_count,
            ..
        }) = command
        else {
            continue;
        };
        if *rasterization_vertex_count == 0 {
            continue;
        }
        let paths = &scene.paths[range.clone()];
        let regions: SmallVec<[Bounds<ScaledPixels>; 8]> = path_types::sprites(paths)
            .map(|sprite| grown(sprite.bounds))
            .collect();
        let union = regions.iter().copied().reduce(|a, b| a.union(&b));
        let (mut right, mut bottom) = (0.0f32, 0.0f32);
        for path in paths {
            let bounds = path.clipped_bounds();
            right = right.max(bounds.origin.x.0 + bounds.size.width.0);
            bottom = bottom.max(bounds.origin.y.0 + bounds.size.height.0);
        }
        let fits = |layer: &PathLayerPlan| {
            let (Some(union), Some(layer_union)) = (union, layer.union) else {
                return true;
            };
            !overlaps(&union, &layer_union)
                || !regions
                    .iter()
                    .any(|a| layer.regions.iter().any(|b| overlaps(a, b)))
        };
        let layer_index = match plan.layers.iter().position(fits) {
            Some(index) => index,
            None if plan.layers.len() < MAX_PATH_LAYERS => {
                plan.layers.push(PathLayerPlan {
                    extent: (0, 0),
                    regions: Vec::new(),
                    union: None,
                    batches: SmallVec::new(),
                });
                plan.layers.len() - 1
            }
            None => return None,
        };
        let layer = &mut plan.layers[layer_index];
        layer.regions.extend(regions.iter().copied());
        layer.union = match (layer.union, union) {
            (Some(a), Some(b)) => Some(a.union(&b)),
            (a, b) => a.or(b),
        };
        let clamp = |value: f32, limit: i32| -> u32 {
            // Whole 32-pixel tiles; never beyond the target.
            let value = (value.max(0.0).ceil() as u32).saturating_add(31) / 32 * 32;
            value.clamp(1, limit.max(1) as u32)
        };
        layer.extent.0 = layer.extent.0.max(clamp(right, viewport.width.0));
        layer.extent.1 = layer.extent.1.max(clamp(bottom, viewport.height.0));
        layer.batches.push(index);
        plan.layer_of_command[index] = Some(layer_index as u8);
    }
    Some(plan)
}

pub(crate) fn quad_is_opaque(quad: &Quad) -> bool {
    let zero = |value: ScaledPixels| value.0 == 0.0;
    let corners = &quad.corner_radii;
    let borders = &quad.border_widths;
    let fade = &quad.content_mask.fade_out;
    let mask = &quad.content_mask.bounds;
    let bounds = &quad.bounds;
    zero(corners.top_left)
        && zero(corners.top_right)
        && zero(corners.bottom_right)
        && zero(corners.bottom_left)
        && zero(borders.top)
        && zero(borders.right)
        && zero(borders.bottom)
        && zero(borders.left)
        && zero(fade.top)
        && zero(fade.right)
        && zero(fade.bottom)
        && zero(fade.left)
        && mask.origin.x.0 <= bounds.origin.x.0
        && mask.origin.y.0 <= bounds.origin.y.0
        && mask.origin.x.0 + mask.size.width.0 >= bounds.origin.x.0 + bounds.size.width.0
        && mask.origin.y.0 + mask.size.height.0 >= bounds.origin.y.0 + bounds.size.height.0
        && quad
            .background
            .as_solid()
            .is_some_and(|color| color.alpha >= 1.0)
}

// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;

// Buffer slots declared by the generated MSL. Group 0 globals land at 0/1, the group 1 data
// binding at 2, Naga's runtime-array sizes buffer at 3.
const GLOBALS_SLOT: u64 = 0;
const FONT_SLOT: u64 = 1;
const DATA_SLOT: u64 = 2;
const SIZES_SLOT: u64 = 3;
const PRIMARY_TEXTURE_SLOT: u64 = 0;
const SECONDARY_TEXTURE_SLOT: u64 = 1;
const SAMPLER_SLOT: u64 = 0;

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

/// Per-frame global uniforms bound by every pipeline.
struct SceneUniforms {
    globals: GlobalUniforms,
    font: FontRasterizationUniforms,
}

impl SceneUniforms {
    fn new(viewport_size: Size<DevicePixels>) -> Self {
        Self {
            globals: GlobalUniforms {
                viewport_size: vec2f(
                    i32::from(viewport_size.width) as f32,
                    i32::from(viewport_size.height) as f32,
                ),
                // Metal composites straight-alpha; paths premultiply in-shader, matching
                // the neutral (disabled) shader behavior.
                premultiplied_alpha: ShaderBool::Disabled,
                padding: 0,
            },
            // Metal text is gamma-corrected grayscale; font corrections stay neutral.
            font: FontRasterizationUniforms {
                gamma_ratios: vec4f(0.0, 0.0, 0.0, 0.0),
                grayscale_enhanced_contrast: 0.0,
                subpixel_enhanced_contrast: 0.0,
                uses_blue_green_red_subpixel_order: ShaderBool::Disabled,
                padding: 0,
            },
        }
    }
}

fn native_shader(label: &str) -> &'static NativeShader {
    NATIVE_SHADERS
        .iter()
        .find(|shader| shader.label == label)
        .unwrap_or_else(|| panic!("missing generated native shader {label}"))
}

fn bind_scene_uniforms(encoder: &metal::RenderCommandEncoderRef, uniforms: &SceneUniforms) {
    encoder.set_vertex_bytes(
        GLOBALS_SLOT,
        mem::size_of::<GlobalUniforms>() as u64,
        &uniforms.globals as *const GlobalUniforms as *const _,
    );
    encoder.set_fragment_bytes(
        GLOBALS_SLOT,
        mem::size_of::<GlobalUniforms>() as u64,
        &uniforms.globals as *const GlobalUniforms as *const _,
    );
    encoder.set_vertex_bytes(
        FONT_SLOT,
        mem::size_of::<FontRasterizationUniforms>() as u64,
        &uniforms.font as *const FontRasterizationUniforms as *const _,
    );
    encoder.set_fragment_bytes(
        FONT_SLOT,
        mem::size_of::<FontRasterizationUniforms>() as u64,
        &uniforms.font as *const FontRasterizationUniforms as *const _,
    );
}

/// Binds an instance slice and the byte length used by Naga's runtime-array size ABI.
fn bind_instances<T>(
    encoder: &metal::RenderCommandEncoderRef,
    buffer: &metal::BufferRef,
    offset: usize,
    instances: &[T],
) {
    bind_instance_bytes(encoder, buffer, offset, mem::size_of_val(instances));
}

fn bind_instance_bytes(
    encoder: &metal::RenderCommandEncoderRef,
    buffer: &metal::BufferRef,
    offset: usize,
    byte_len: usize,
) {
    let byte_len = u32::try_from(byte_len).expect("Metal instance binding exceeds 4 GiB");
    encoder.set_vertex_buffer(DATA_SLOT, Some(buffer), offset as u64);
    encoder.set_fragment_buffer(DATA_SLOT, Some(buffer), offset as u64);
    encoder.set_vertex_bytes(SIZES_SLOT, 4, &byte_len as *const u32 as *const _);
    encoder.set_fragment_bytes(SIZES_SLOT, 4, &byte_len as *const u32 as *const _);
}

/// Creates the CAMetalLayer that backs a window's view. The renderer drawing
/// into it sets its device and pixel format.
pub fn new_window_layer(transparent: bool) -> metal::MetalLayer {
    let layer = metal::MetalLayer::new();
    // Support direct-to-display rendering if the window is not transparent
    // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
    layer.set_opaque(!transparent);
    // `metal::MetalLayer` is a CAMetalLayer retained by the Metal crate.
    // Reborrow its Objective-C object as the generated objc2 class to keep
    // selector encodings and the autoresizing mask type checked here.
    let layer_object = unsafe { &*(layer.as_ptr() as *const Objc2CAMetalLayer) };
    layer_object.setAllowsNextDrawableTimeout(false);
    layer_object.setNeedsDisplayOnBoundsChange(true);
    layer_object.setAutoresizingMask(
        CAAutoresizingMask::LayerWidthSizable | CAAutoresizingMask::LayerHeightSizable,
    );
    layer
}

pub fn new_renderer(
    context: self::Context,
    _bounds: gpui::Size<f32>,
    transparent: bool,
) -> Result<Renderer> {
    Ok(MetalRenderer::new(context, transparent))
}

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(
        &mut self,
        device: &metal::Device,
        unified_memory: bool,
        minimum_size: usize,
    ) -> InstanceBuffer {
        if minimum_size > self.buffer_size {
            self.reset(
                minimum_size
                    .checked_next_power_of_two()
                    .unwrap_or(minimum_size),
            );
        }
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            let options = if unified_memory {
                MTLResourceOptions::StorageModeShared
                    // Buffers are write only which can benefit from the combined cache
                    // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            };

            device.new_buffer(self.buffer_size as u64, options)
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

pub struct MetalRenderer {
    device: metal::Device,
    layer: Option<metal::MetalLayer>,
    is_apple_gpu: bool,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    /// For headless rendering, tracks whether output should be opaque
    opaque: bool,
    command_queue: CommandQueue,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    smoothed_shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    /// The quad pipeline without blending, for opaque rectangles (`opaque-quads`).
    opaque_quads_pipeline_state: metal::RenderPipelineState,
    smoothed_quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    smoothed_polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    // Blur pipelines: downsample (no blend, also used for the final blit), separable gaussian
    // (no blend), and composite (alpha blend into a rounded rect), from shared shader sources.
    blur_downsample_pipeline_state: metal::RenderPipelineState,
    blur_pipeline_state: metal::RenderPipelineState,
    blur_composite_pipeline_state: metal::RenderPipelineState,
    smoothed_blur_composite_pipeline_state: metal::RenderPipelineState,
    sampler: metal::SamplerState,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: core_video::metal_texture_cache::CVMetalTextureCache,
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    /// Resolve targets for path prepass layers after the first (`path-prepass`).
    path_layer_textures: Vec<metal::Texture>,
    experiments: Experiments,
    // Offscreen scene target (the scene is rendered here, then blitted to the drawable, so blur
    // passes can sample already-painted content), the half-res ping/pong blur targets, and a
    // full-res target for content-filter groups.
    scene_color_texture: Option<metal::Texture>,
    blur_ping_texture: Option<metal::Texture>,
    blur_pong_texture: Option<metal::Texture>,
    /// Full-resolution offscreen targets a content-filter (`filter`) group renders into before
    /// being blurred and composited back. One per nesting level (indexed by isolation depth) so
    /// nested content blurs isolate consistently with [`MAX_FILTER_GROUP_DEPTH`]; deeper nests render
    /// inline.
    group_textures: Vec<metal::Texture>,
    intermediate_texture_size: Option<Size<DevicePixels>>,
    path_sample_count: u32,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    headless_render_target: Option<metal::Texture>,
}

impl MetalRenderer {
    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>, transparent: bool) -> Self {
        let device = Self::create_device();

        let layer = new_window_layer(transparent);
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_maximum_drawable_count(3);
        // Allow texture reading for visual tests (captures screenshots without ScreenCaptureKit)
        #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
        layer.set_framebuffer_only(false);

        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn new_headless(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Self {
        let device = Self::create_device();
        Self::new_internal(device, None, true, instance_buffer_pool)
    }

    fn create_device() -> metal::Device {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        if let Some(d) = metal::Device::all()
            .into_iter()
            .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
        {
            d
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            metal::Device::system_default().unwrap_or_else(|| {
                log::error!("unable to access a compatible graphics device");
                std::process::exit(1);
            })
        }
    }

    fn new_internal(
        device: metal::Device,
        layer: Option<metal::MetalLayer>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Self {
        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        let is_unified_memory = device.has_unified_memory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supports_family(MTLGPUFamily::Apple1);

        // Load each Naga-generated module once: precompiled when the build had Apple's
        // Metal Toolchain, otherwise (or if this OS rejects the library) compiled from
        // MSL by the device's runtime compiler. Deduplicated per source.
        let mut libraries: Vec<(&'static str, metal::Library)> = Vec::new();
        let mut library_for = |shader: &'static NativeShader| -> metal::Library {
            if let Some((_, library)) = libraries
                .iter()
                .find(|(registered, _)| *registered == shader.msl)
            {
                return library.clone();
            }
            let precompiled = shader.metallib.and_then(|bytes| {
                device
                    .new_library_with_data(bytes)
                    // A library that loads can still lack a function this OS
                    // accepts; the pipeline builders would then panic.
                    .and_then(|library| {
                        for entry in [shader.vertex_entry, shader.fragment_entry] {
                            library.get_function(entry, None)?;
                        }
                        Ok(library)
                    })
                    .inspect_err(|error| {
                        log::warn!(
                            "precompiled metal library {} is unusable, compiling MSL: {error}",
                            shader.label
                        )
                    })
                    .ok()
            });
            let library = precompiled.unwrap_or_else(|| {
                device
                    .new_library_with_source(shader.msl, &metal::CompileOptions::new())
                    .unwrap_or_else(|error| panic!("error building metal library: {error}"))
            });
            libraries.push((shader.msl, library.clone()));
            library
        };

        let mut pipeline = |label: &str| -> (&'static NativeShader, metal::Library) {
            let shader = native_shader(label);
            (shader, library_for(shader))
        };

        let (path_rasterization_shader, path_rasterization_library) =
            pipeline("path_rasterization");
        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &path_rasterization_library,
            path_rasterization_shader,
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let (paths_shader, paths_library) = pipeline("paths");
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &paths_library,
            paths_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (shadows_shader, shadows_library) = pipeline("shadows");
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &shadows_library,
            shadows_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (smoothed_shadows_shader, smoothed_shadows_library) = pipeline("smoothed_shadows");
        let smoothed_shadows_pipeline_state = build_pipeline_state(
            &device,
            &smoothed_shadows_library,
            smoothed_shadows_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (quads_shader, quads_library) = pipeline("quads");
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &quads_library,
            quads_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let opaque_quads_pipeline_state = build_opaque_pipeline_state(
            &device,
            &quads_library,
            quads_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (smoothed_quads_shader, smoothed_quads_library) = pipeline("smoothed_quads");
        let smoothed_quads_pipeline_state = build_pipeline_state(
            &device,
            &smoothed_quads_library,
            smoothed_quads_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (underlines_shader, underlines_library) = pipeline("underlines");
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &underlines_library,
            underlines_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (monochrome_shader, monochrome_library) = pipeline("monochrome_sprites");
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &monochrome_library,
            monochrome_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (polychrome_shader, polychrome_library) = pipeline("polychrome_sprites");
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &polychrome_library,
            polychrome_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (smoothed_polychrome_shader, smoothed_polychrome_library) =
            pipeline("smoothed_polychrome_sprites");
        let smoothed_polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &smoothed_polychrome_library,
            smoothed_polychrome_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (surfaces_shader, surfaces_library) = pipeline("surfaces");
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &surfaces_library,
            surfaces_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (blur_downsample_shader, blur_downsample_library) = pipeline("blur_downsample");
        let blur_downsample_pipeline_state = build_blur_pipeline_state(
            &device,
            &blur_downsample_library,
            blur_downsample_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (blur_shader, blur_library) = pipeline("blur");
        let blur_pipeline_state = build_blur_pipeline_state(
            &device,
            &blur_library,
            blur_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        // Premultiplied blend (One / OneMinusSourceAlpha) — the composite outputs a premultiplied
        // blurred sample; straight-alpha blending would darken the faded edges.
        let (blur_composite_shader, blur_composite_library) = pipeline("blur_composite");
        let blur_composite_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &blur_composite_library,
            blur_composite_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (smoothed_blur_composite_shader, smoothed_blur_composite_library) =
            pipeline("smoothed_blur_composite");
        let smoothed_blur_composite_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &smoothed_blur_composite_library,
            smoothed_blur_composite_shader,
            MTLPixelFormat::BGRA8Unorm,
        );

        let sampler_descriptor = SamplerDescriptor::new();
        sampler_descriptor.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        sampler_descriptor.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        let sampler = device.new_sampler(&sampler_descriptor);

        let command_queue = device.new_command_queue();
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), is_apple_gpu));
        let core_video_texture_cache =
            CVMetalTextureCache::new(None, device.clone(), None).unwrap();

        Self {
            device,
            layer,
            presents_with_transaction: false,
            is_apple_gpu,
            is_unified_memory,
            opaque,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            smoothed_shadows_pipeline_state,
            quads_pipeline_state,
            opaque_quads_pipeline_state,
            smoothed_quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            smoothed_polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            blur_downsample_pipeline_state,
            blur_pipeline_state,
            blur_composite_pipeline_state,
            smoothed_blur_composite_pipeline_state,
            sampler,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            path_layer_textures: Vec::new(),
            experiments: experiments(),
            scene_color_texture: None,
            blur_ping_texture: None,
            blur_pong_texture: None,
            group_textures: Vec::new(),
            intermediate_texture_size: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
            headless_render_target: None,
        }
    }

    pub fn layer(&self) -> Option<&metal::MetalLayerRef> {
        self.layer.as_ref().map(|l| l.as_ref())
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|l| l.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.set_presents_with_transaction(presents_with_transaction);
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            layer.set_drawable_size(CGSize::new(size.width.0 as f64, size.height.0 as f64));
        }
        self.update_intermediate_texture_size(size);
    }

    fn update_intermediate_texture_size(&mut self, size: Size<DevicePixels>) {
        if self.intermediate_texture_size == Some(size) {
            return;
        }
        self.path_intermediate_texture = None;
        self.path_intermediate_msaa_texture = None;
        self.path_layer_textures.clear();
        self.scene_color_texture = None;
        self.blur_ping_texture = None;
        self.blur_pong_texture = None;
        self.group_textures.clear();
        self.intermediate_texture_size = (size.width.0 > 0 && size.height.0 > 0).then_some(size);
    }

    fn prepare_intermediate_textures(&mut self, scene: &Scene, size: Size<DevicePixels>) {
        self.update_intermediate_texture_size(size);
        let Some(size) = self.intermediate_texture_size else {
            return;
        };
        let requirements = scene.render_plan().requirements();
        let full_w = size.width.0 as u64;
        let full_h = size.height.0 as u64;

        let make_color_texture = |width: u64, height: u64| {
            let descriptor = metal::TextureDescriptor::new();
            descriptor.set_width(width.max(1));
            descriptor.set_height(height.max(1));
            descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
            descriptor.set_storage_mode(metal::MTLStorageMode::Private);
            descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            self.device.new_texture(&descriptor)
        };

        if requirements.uses_path_target && self.path_intermediate_texture.is_none() {
            let texture_descriptor = metal::TextureDescriptor::new();
            texture_descriptor.set_width(full_w);
            texture_descriptor.set_height(full_h);
            texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
            texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
            texture_descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            self.path_intermediate_texture = Some(self.device.new_texture(&texture_descriptor));

            // Storage mode guidance:
            // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
            // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
            if self.path_sample_count > 1 {
                let storage_mode = if self.is_apple_gpu {
                    metal::MTLStorageMode::Memoryless
                } else {
                    metal::MTLStorageMode::Private
                };
                let msaa_descriptor = texture_descriptor;
                msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
                msaa_descriptor.set_storage_mode(storage_mode);
                msaa_descriptor.set_sample_count(self.path_sample_count as _);
                self.path_intermediate_msaa_texture =
                    Some(self.device.new_texture(&msaa_descriptor));
            }
        }

        if requirements.uses_offscreen_target {
            self.scene_color_texture
                .get_or_insert_with(|| make_color_texture(full_w, full_h));
            let blur_width = u64::from(downsampled_dimension(full_w as u32));
            let blur_height = u64::from(downsampled_dimension(full_h as u32));
            self.blur_ping_texture
                .get_or_insert_with(|| make_color_texture(blur_width, blur_height));
            self.blur_pong_texture
                .get_or_insert_with(|| make_color_texture(blur_width, blur_height));
            while self.group_textures.len() < requirements.isolated_target_count {
                self.group_textures.push(make_color_texture(full_w, full_h));
            }
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.set_opaque(!transparent);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn draw(&mut self, scene: &Scene) {
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!(
                    "draw() called on headless renderer - use render_scene_to_image() instead"
                );
                return;
            }
        };
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return;
        };

        if crate::scene_stats::level() > 0 {
            crate::scene_stats::log(scene, viewport_size);
        }
        if let Some(prefix) = dump_frame_prefix() {
            // Diagnostics: the same scene rendered offscreen and read back, so
            // renderer variants can be compared pixel for pixel without screen
            // capture. Each frame overwrites the file; the last one remains.
            if let Err(error) = self.dump_scene(scene, viewport_size, prefix) {
                log::error!("GPUI_DUMP_FRAME: {error}");
            }
        }
        let mut instance_buffer = self.acquire_instance_buffer(scene);
        let command_buffer =
            match self.draw_primitives(scene, &mut instance_buffer, drawable, viewport_size) {
                Ok(command_buffer) => command_buffer,
                Err(error) => {
                    log::error!("failed to render pre-sized scene: {error}");
                    return;
                }
            };
        self.release_instance_buffer_when_complete(&command_buffer, instance_buffer);

        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.wait_until_scheduled();
            drawable.present();
        } else {
            command_buffer.present_drawable(drawable);
            command_buffer.commit();
        }
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    ///
    /// Note: This requires a layer-backed renderer. For headless rendering,
    /// use `render_scene_to_image()` instead.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_image requires a layer-backed renderer"))?;
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = layer
            .next_drawable()
            .ok_or_else(|| anyhow::anyhow!("Failed to get drawable for render_to_image"))?;

        let mut instance_buffer = self.acquire_instance_buffer(scene);
        let command_buffer =
            self.draw_primitives(scene, &mut instance_buffer, drawable, viewport_size)?;

        command_buffer.commit();
        command_buffer.wait_until_completed();
        self.instance_buffer_pool.lock().release(instance_buffer);

        let texture = drawable.texture();
        let width = texture.width() as u32;
        let height = texture.height() as u32;
        let bytes_per_row = width as usize * 4;
        let buffer_size = height as usize * bytes_per_row;
        let mut pixels = vec![0u8; buffer_size];
        let region = metal::MTLRegion {
            origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
            size: metal::MTLSize {
                width: width as u64,
                height: height as u64,
                depth: 1,
            },
        };
        texture.get_bytes(
            pixels.as_mut_ptr() as *mut std::ffi::c_void,
            bytes_per_row as u64,
            region,
            0,
        );
        for chunk in pixels.chunks_exact_mut(4) {
            chunk.swap(0, 2);
        }
        RgbaImage::from_raw(width, height, pixels)
            .ok_or_else(|| anyhow::anyhow!("Failed to create RgbaImage from pixel data"))
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Create an offscreen texture as render target
        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Managed);
        let target_texture = self.device.new_texture(&texture_descriptor);

        let mut instance_buffer = self.acquire_instance_buffer(scene);
        let command_buffer =
            self.draw_primitives_to_texture(scene, &mut instance_buffer, &target_texture, size)?;

        if !self.is_unified_memory {
            let blit = command_buffer.new_blit_command_encoder();
            blit.synchronize_resource(&target_texture);
            blit.end_encoding();
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        self.instance_buffer_pool.lock().release(instance_buffer);

        let width = size.width.0 as u32;
        let height = size.height.0 as u32;
        let bytes_per_row = width as usize * 4;
        let buffer_size = height as usize * bytes_per_row;
        let mut pixels = vec![0u8; buffer_size];
        let region = metal::MTLRegion {
            origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
            size: metal::MTLSize {
                width: width as u64,
                height: height as u64,
                depth: 1,
            },
        };
        target_texture.get_bytes(
            pixels.as_mut_ptr() as *mut std::ffi::c_void,
            bytes_per_row as u64,
            region,
            0,
        );
        for chunk in pixels.chunks_exact_mut(4) {
            chunk.swap(0, 2);
        }
        RgbaImage::from_raw(width, height, pixels)
            .ok_or_else(|| anyhow::anyhow!("Failed to create RgbaImage from pixel data"))
    }

    /// Renders a scene to a reused offscreen texture without reading pixels
    /// back or blocking on GPU completion.
    ///
    /// This mirrors the CPU cost of presenting a frame to a window (scene
    /// encoding, instance buffer writes, command submission) and is used by
    /// headless benchmark rendering, where the produced pixels are never
    /// inspected.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene: {:?}", size);
        }

        let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
            texture.width() != size.width.0 as u64 || texture.height() != size.height.0 as u64
        });
        if needs_new_target {
            let texture_descriptor = metal::TextureDescriptor::new();
            texture_descriptor.set_width(size.width.0 as u64);
            texture_descriptor.set_height(size.height.0 as u64);
            texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            texture_descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
            self.headless_render_target = Some(self.device.new_texture(&texture_descriptor));
        }
        let target_texture = self
            .headless_render_target
            .clone()
            .expect("just ensured the render target exists");

        let mut instance_buffer = self.acquire_instance_buffer(scene);
        let command_buffer =
            self.draw_primitives_to_texture(scene, &mut instance_buffer, &target_texture, size)?;
        self.release_instance_buffer_when_complete(&command_buffer, instance_buffer);
        command_buffer.commit();
        Ok(())
    }

    fn acquire_instance_buffer(&self, scene: &Scene) -> InstanceBuffer {
        self.instance_buffer_pool.lock().acquire(
            &self.device,
            self.is_unified_memory,
            required_instance_buffer_size(scene),
        )
    }

    fn release_instance_buffer_when_complete(
        &self,
        command_buffer: &metal::CommandBufferRef,
        instance_buffer: InstanceBuffer,
    ) {
        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let instance_buffer = Cell::new(Some(instance_buffer));
        let block = ConcreteBlock::new(move |_| {
            if let Some(instance_buffer) = instance_buffer.take() {
                instance_buffer_pool.lock().release(instance_buffer);
            }
        });
        command_buffer.add_completed_handler(&block.copy());
    }

    fn draw_primitives(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        drawable: &metal::MetalDrawableRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        self.draw_primitives_to_texture(scene, instance_buffer, drawable.texture(), viewport_size)
    }

    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        self.prepare_intermediate_textures(scene, viewport_size);
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        let alpha = if self.opaque { 1. } else { 0. };
        let mut instance_offset = 0;
        let scene_uniforms = SceneUniforms::new(viewport_size);

        // Render the scene into an offscreen color texture (so filters can sample it), then
        // blit it to `texture`. Owned clones keep the textures borrowable without borrowing
        // `self` across the batch loop (which calls `&mut self` methods like `draw_surfaces`).
        // Only route through the offscreen scene texture when the scene actually contains blur
        // filters; otherwise render straight to `texture` exactly as before (no regression, no
        // extra blit for the common case).
        let use_offscreen = scene.requires_offscreen_rendering();
        let scene_color_owned = self.scene_color_texture.clone();
        let blur_ping_owned = self.blur_ping_texture.clone();
        let blur_pong_owned = self.blur_pong_texture.clone();
        let group_owned = self
            .group_textures
            .iter()
            .cloned()
            .collect::<SmallVec<[metal::Texture; MAX_FILTER_GROUP_DEPTH]>>();
        let scene_color: &metal::TextureRef = if use_offscreen {
            scene_color_owned.as_deref().unwrap_or(texture)
        } else {
            texture
        };
        // The active render target; switches to the group texture inside a content-filter group.
        let mut current_target: &metal::TextureRef = scene_color;
        let mut filter_stack = SmallVec::<[&metal::TextureRef; MAX_FILTER_GROUP_DEPTH]>::new();

        let experiments = self.experiments;
        let path_plan = if experiments.path_prepass
            && !experiments.skip_paths
            && scene.render_plan().requirements().uses_path_target
        {
            plan_path_prepass(scene, viewport_size)
        } else {
            None
        };
        if let Some(plan) = &path_plan {
            self.ensure_path_layer_textures(plan.layers.len(), viewport_size);
            for (layer_index, layer) in plan.layers.iter().enumerate() {
                if !self.rasterize_path_layer(
                    scene,
                    layer,
                    layer_index,
                    instance_buffer,
                    &mut instance_offset,
                    command_buffer,
                ) {
                    anyhow::bail!("scene too large: {} paths", scene.paths.len());
                }
            }
        }

        let mut command_encoder = new_command_encoder_for_texture(
            command_buffer,
            current_target,
            viewport_size,
            |color_attachment| {
                color_attachment.set_load_action(metal::MTLLoadAction::Clear);
                color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., alpha));
            },
        );

        for (command_index, command) in scene.render_commands().iter().enumerate() {
            let skipped = match command {
                RenderCommand::Batch(PrimitiveBatch::Shadows { .. }) => experiments.skip_shadows,
                RenderCommand::Batch(PrimitiveBatch::Quads { .. }) => experiments.skip_quads,
                RenderCommand::Batch(
                    PrimitiveBatch::MonochromeSprites { .. }
                    | PrimitiveBatch::PolychromeSprites { .. },
                ) => experiments.skip_sprites,
                _ => false,
            };
            if skipped {
                continue;
            }
            let ok = match command {
                RenderCommand::Batch(PrimitiveBatch::Shadows { range, smoothed }) => self
                    .draw_shadows(
                        &scene.shadows[range.clone()],
                        *smoothed,
                        instance_buffer,
                        &mut instance_offset,
                        &scene_uniforms,
                        command_encoder,
                    ),
                RenderCommand::Batch(PrimitiveBatch::Quads { range, smoothed }) => self.draw_quads(
                    &scene.quads[range.clone()],
                    *smoothed,
                    instance_buffer,
                    &mut instance_offset,
                    &scene_uniforms,
                    command_encoder,
                ),
                RenderCommand::Batch(PrimitiveBatch::Paths {
                    range,
                    rasterization_vertex_count,
                    sprite_count,
                }) => {
                    if *rasterization_vertex_count == 0 || experiments.skip_paths {
                        continue;
                    }
                    let paths = &scene.paths[range.clone()];
                    if let Some(layer) = path_plan
                        .as_ref()
                        .and_then(|plan| plan.layer_of_command[command_index])
                    {
                        let texture = if layer == 0 {
                            self.path_intermediate_texture.clone()
                        } else {
                            self.path_layer_textures.get(layer as usize - 1).cloned()
                        };
                        let Some(texture) = texture else {
                            continue;
                        };
                        self.draw_paths_from_texture(
                            paths,
                            *sprite_count,
                            &texture,
                            instance_buffer,
                            &mut instance_offset,
                            &scene_uniforms,
                            command_encoder,
                        )
                    } else {
                        command_encoder.end_encoding();

                        let did_draw = self.draw_paths_to_intermediate(
                            paths,
                            *rasterization_vertex_count,
                            instance_buffer,
                            &mut instance_offset,
                            &scene_uniforms,
                            command_buffer,
                        );

                        command_encoder = new_command_encoder_for_texture(
                            command_buffer,
                            current_target,
                            viewport_size,
                            |color_attachment| {
                                color_attachment.set_load_action(metal::MTLLoadAction::Load);
                            },
                        );

                        if did_draw {
                            self.draw_paths_from_intermediate(
                                paths,
                                *sprite_count,
                                instance_buffer,
                                &mut instance_offset,
                                &scene_uniforms,
                                command_encoder,
                            )
                        } else {
                            false
                        }
                    }
                }
                RenderCommand::Batch(PrimitiveBatch::Underlines(range)) => self.draw_underlines(
                    &scene.underlines[range.clone()],
                    instance_buffer,
                    &mut instance_offset,
                    &scene_uniforms,
                    command_encoder,
                ),
                RenderCommand::Batch(PrimitiveBatch::MonochromeSprites { texture_id, range }) => {
                    self.draw_monochrome_sprites(
                        *texture_id,
                        &scene.monochrome_sprites[range.clone()],
                        instance_buffer,
                        &mut instance_offset,
                        &scene_uniforms,
                        command_encoder,
                    )
                }
                RenderCommand::Batch(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range,
                    smoothed,
                }) => self.draw_polychrome_sprites(
                    *texture_id,
                    &scene.polychrome_sprites[range.clone()],
                    *smoothed,
                    instance_buffer,
                    &mut instance_offset,
                    &scene_uniforms,
                    command_encoder,
                ),
                RenderCommand::Batch(PrimitiveBatch::Surfaces(range)) => self.draw_surfaces(
                    &scene.surfaces[range.clone()],
                    &scene.surface_opacities()[range.clone()],
                    &scene_uniforms,
                    command_encoder,
                ),
                RenderCommand::Batch(PrimitiveBatch::BackdropFilters(range)) => {
                    command_encoder.end_encoding();
                    if let (Some(ping), Some(pong)) =
                        (blur_ping_owned.as_deref(), blur_pong_owned.as_deref())
                    {
                        for filter in &scene.backdrop_filters[range.clone()] {
                            self.metal_blur_and_composite(
                                command_buffer,
                                &scene_uniforms,
                                current_target,
                                current_target,
                                ping,
                                pong,
                                viewport_size,
                                filter.bounds,
                                filter.content_mask.bounds,
                                filter.corner_radii,
                                filter.corner_smoothing,
                                filter.max_blur_radius(),
                                filter.opacity,
                                true,
                            );
                        }
                    }
                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        current_target,
                        viewport_size,
                        |color_attachment| {
                            color_attachment.set_load_action(metal::MTLLoadAction::Load);
                        },
                    );
                    true
                }
                RenderCommand::BeginFilter {
                    target: FilterRenderTarget::Isolated(target_index),
                    ..
                } => {
                    command_encoder.end_encoding();
                    filter_stack.push(current_target);
                    current_target = group_owned[target_index.as_usize()].as_ref();
                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        current_target,
                        viewport_size,
                        |color_attachment| {
                            color_attachment.set_load_action(metal::MTLLoadAction::Clear);
                            color_attachment
                                .set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));
                        },
                    );
                    true
                }
                RenderCommand::EndFilter {
                    boundary_index,
                    target: FilterRenderTarget::Isolated(_),
                    ..
                } => {
                    let boundary = &scene.filter_boundaries[*boundary_index];
                    let parent = filter_stack
                        .pop()
                        .expect("render plan emitted an unmatched isolated filter end");
                    command_encoder.end_encoding();
                    if let (Some(ping), Some(pong)) =
                        (blur_ping_owned.as_deref(), blur_pong_owned.as_deref())
                    {
                        self.metal_blur_and_composite(
                            command_buffer,
                            &scene_uniforms,
                            current_target,
                            parent,
                            ping,
                            pong,
                            viewport_size,
                            boundary.bounds,
                            boundary.content_mask.bounds,
                            boundary.corner_radii,
                            boundary.corner_smoothing,
                            boundary.max_blur_radius(),
                            boundary.opacity,
                            false,
                        );
                    }
                    current_target = parent;
                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        current_target,
                        viewport_size,
                        |color_attachment| {
                            color_attachment.set_load_action(metal::MTLLoadAction::Load);
                        },
                    );
                    true
                }
                RenderCommand::BeginFilter {
                    target: FilterRenderTarget::Inline,
                    ..
                }
                | RenderCommand::EndFilter {
                    target: FilterRenderTarget::Inline,
                    ..
                } => true,
                RenderCommand::Batch(PrimitiveBatch::SubpixelSprites { .. }) => unreachable!(),
                RenderCommand::Batch(PrimitiveBatch::FilterBoundary(_)) => {
                    unreachable!("filter boundaries are resolved by the render plan")
                }
            };
            if !ok {
                command_encoder.end_encoding();
                anyhow::bail!(
                    "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                );
            }
        }

        command_encoder.end_encoding();

        // Present the offscreen scene by copying it into the drawable/target texture.
        if use_offscreen && scene_color_owned.is_some() {
            self.run_metal_blur_pass(
                command_buffer,
                &self.blur_downsample_pipeline_state,
                texture,
                scene_color,
                viewport_size,
                &scene_uniforms,
                BlurUniforms::copy([
                    i32::from(viewport_size.width) as f32,
                    i32::from(viewport_size.height) as f32,
                ]),
                ScissorRectangle {
                    x: 0,
                    y: 0,
                    width: i32::from(viewport_size.width).max(0) as u32,
                    height: i32::from(viewport_size.height).max(0) as u32,
                },
                metal::MTLPrimitiveType::Triangle,
                3,
                false,
            );
        }

        if !self.is_unified_memory {
            // Sync the instance buffer to the GPU
            instance_buffer.metal_buffer.did_modify_range(NSRange {
                location: 0,
                length: instance_offset as NSUInteger,
            });
        }

        Ok(command_buffer.to_owned())
    }

    /// Run a single blur pass: draw a full-screen (or composite) quad sampling `source` into
    /// `target`. `params` is supplied to both shader stages; `load` keeps existing target
    /// contents (used by the composite), otherwise the target is cleared.
    #[allow(clippy::too_many_arguments)]
    fn run_metal_blur_pass(
        &self,
        command_buffer: &metal::CommandBufferRef,
        pipeline: &metal::RenderPipelineState,
        target: &metal::TextureRef,
        source: &metal::TextureRef,
        target_viewport: Size<DevicePixels>,
        scene_uniforms: &SceneUniforms,
        params: BlurUniforms,
        scissor: ScissorRectangle,
        primitive: metal::MTLPrimitiveType,
        vertex_count: u64,
        load: bool,
    ) {
        let encoder = new_command_encoder_for_texture(
            command_buffer,
            target,
            target_viewport,
            |color_attachment| {
                if load {
                    color_attachment.set_load_action(metal::MTLLoadAction::Load);
                } else {
                    color_attachment.set_load_action(metal::MTLLoadAction::Clear);
                    color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));
                }
            },
        );
        encoder.set_scissor_rect(MTLScissorRect {
            x: scissor.x as u64,
            y: scissor.y as u64,
            width: scissor.width as u64,
            height: scissor.height as u64,
        });
        encoder.set_render_pipeline_state(pipeline);
        bind_scene_uniforms(encoder, scene_uniforms);
        encoder.set_vertex_bytes(
            DATA_SLOT,
            mem::size_of::<BlurUniforms>() as u64,
            &params as *const BlurUniforms as *const _,
        );
        encoder.set_fragment_bytes(
            DATA_SLOT,
            mem::size_of::<BlurUniforms>() as u64,
            &params as *const BlurUniforms as *const _,
        );
        encoder.set_fragment_texture(PRIMARY_TEXTURE_SLOT, Some(source));
        encoder.set_fragment_sampler_state(SAMPLER_SLOT, Some(&self.sampler));
        encoder.draw_primitives(primitive, 0, vertex_count);
        encoder.end_encoding();
    }

    /// Blur `source` (full-resolution) using the half-res ping/pong textures and composite the
    /// result into `target`, clipped to `bounds`/`corner_radii`/`content_mask` and modulated by
    /// `opacity`. Shared by the backdrop and content-filter paths, mirroring the wgpu
    /// backend's pass structure via the shared `gpui_render::blur` contracts.
    #[allow(clippy::too_many_arguments)]
    fn metal_blur_and_composite(
        &self,
        command_buffer: &metal::CommandBufferRef,
        scene_uniforms: &SceneUniforms,
        source: &metal::TextureRef,
        target: &metal::TextureRef,
        ping: &metal::TextureRef,
        pong: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        bounds: Bounds<ScaledPixels>,
        content_mask: Bounds<ScaledPixels>,
        corner_radii: Corners<ScaledPixels>,
        corner_smoothing: f32,
        blur_radius: f32,
        opacity: f32,
        // Backdrop clips to the rounded rect; content (`filter`) bleeds past its bounds.
        clip_rounded: bool,
    ) {
        let Some(kernel) = BlurKernel::for_radius(blur_radius) else {
            return;
        };
        let full_width = i32::from(viewport_size.width).max(0) as u32;
        let full_height = i32::from(viewport_size.height).max(0) as u32;
        let blur_size = [
            downsampled_dimension(full_width) as f32,
            downsampled_dimension(full_height) as f32,
        ];
        let blur_viewport_size = Size {
            width: DevicePixels(blur_size[0] as i32),
            height: DevicePixels(blur_size[1] as i32),
        };
        let dilation = GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * blur_radius;
        let scissor =
            ScissorRectangle::for_blurred_bounds(bounds, dilation, full_width, full_height);
        if scissor.is_empty() {
            return;
        }
        let clip = if clip_rounded {
            gpui_render::blur::FilterCompositeClip::RoundedBounds
        } else {
            gpui_render::blur::FilterCompositeClip::ContentShape
        };

        // Downsample source -> ping, then separable gaussian ping -> pong -> ping.
        self.run_metal_blur_pass(
            command_buffer,
            &self.blur_downsample_pipeline_state,
            ping,
            source,
            blur_viewport_size,
            scene_uniforms,
            BlurUniforms::downsample([full_width as f32, full_height as f32], blur_size),
            scissor,
            metal::MTLPrimitiveType::Triangle,
            3,
            false,
        );
        self.run_metal_blur_pass(
            command_buffer,
            &self.blur_pipeline_state,
            pong,
            ping,
            blur_viewport_size,
            scene_uniforms,
            BlurUniforms::gaussian(BlurAxis::Horizontal, blur_size, kernel),
            scissor,
            metal::MTLPrimitiveType::Triangle,
            3,
            false,
        );
        self.run_metal_blur_pass(
            command_buffer,
            &self.blur_pipeline_state,
            ping,
            pong,
            blur_viewport_size,
            scene_uniforms,
            BlurUniforms::gaussian(BlurAxis::Vertical, blur_size, kernel),
            scissor,
            metal::MTLPrimitiveType::Triangle,
            3,
            false,
        );

        // Composite the blurred result into the target (preserving its contents).
        let composite_bounds = if clip_rounded {
            bounds
        } else {
            bounds.dilate(ScaledPixels(dilation))
        };
        let composite_uniforms = BlurUniforms::composite(
            composite_bounds,
            content_mask,
            corner_radii,
            corner_smoothing,
            opacity,
            clip,
            blur_size,
            [full_width as f32, full_height as f32],
        );
        let encoder = new_command_encoder_for_texture(
            command_buffer,
            target,
            viewport_size,
            |color_attachment| {
                color_attachment.set_load_action(metal::MTLLoadAction::Load);
            },
        );
        let pipeline = if composite_uniforms.corner_smoothing > 0.0 {
            &self.smoothed_blur_composite_pipeline_state
        } else {
            &self.blur_composite_pipeline_state
        };
        encoder.set_render_pipeline_state(pipeline);
        bind_scene_uniforms(encoder, scene_uniforms);
        encoder.set_vertex_bytes(
            DATA_SLOT,
            mem::size_of::<BlurUniforms>() as u64,
            &composite_uniforms as *const BlurUniforms as *const _,
        );
        encoder.set_fragment_bytes(
            DATA_SLOT,
            mem::size_of::<BlurUniforms>() as u64,
            &composite_uniforms as *const BlurUniforms as *const _,
        );
        encoder.set_fragment_texture(PRIMARY_TEXTURE_SLOT, Some(ping));
        encoder.set_fragment_sampler_state(SAMPLER_SLOT, Some(&self.sampler));
        encoder.draw_primitives(metal::MTLPrimitiveType::TriangleStrip, 0, 4);
        encoder.end_encoding();
    }

    fn dump_scene(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
        prefix: &std::path::Path,
    ) -> Result<()> {
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(size.width.0.max(1) as u64);
        descriptor.set_height(size.height.0.max(1) as u64);
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        descriptor.set_storage_mode(if self.is_unified_memory {
            metal::MTLStorageMode::Shared
        } else {
            metal::MTLStorageMode::Managed
        });
        let target = self.device.new_texture(&descriptor);
        let mut instance_buffer = self.acquire_instance_buffer(scene);
        let command_buffer =
            self.draw_primitives_to_texture(scene, &mut instance_buffer, &target, size)?;
        if !self.is_unified_memory {
            let blit = command_buffer.new_blit_command_encoder();
            blit.synchronize_resource(&target);
            blit.end_encoding();
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        self.instance_buffer_pool.lock().release(instance_buffer);
        let (width, height) = (size.width.0.max(1) as u32, size.height.0.max(1) as u32);
        let bytes_per_row = width as usize * 4;
        let mut pixels = vec![0u8; height as usize * bytes_per_row];
        target.get_bytes(
            pixels.as_mut_ptr() as *mut std::ffi::c_void,
            bytes_per_row as u64,
            metal::MTLRegion {
                origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
                size: metal::MTLSize {
                    width: width as u64,
                    height: height as u64,
                    depth: 1,
                },
            },
            0,
        );
        for chunk in pixels.chunks_exact_mut(4) {
            chunk.swap(0, 2);
        }
        let image = image::RgbaImage::from_raw(width, height, pixels)
            .ok_or_else(|| anyhow::anyhow!("frame buffer size mismatch"))?;
        let path = prefix.with_extension("png");
        let partial = prefix.with_extension("partial.png");
        image.save(&partial)?;
        std::fs::rename(&partial, &path)?;
        Ok(())
    }

    /// Full-size resolve targets for prepass layers after the first.
    fn ensure_path_layer_textures(&mut self, layers: usize, size: Size<DevicePixels>) {
        while self.path_layer_textures.len() + 1 < layers {
            let descriptor = metal::TextureDescriptor::new();
            descriptor.set_width(size.width.0.max(1) as u64);
            descriptor.set_height(size.height.0.max(1) as u64);
            descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
            descriptor.set_storage_mode(metal::MTLStorageMode::Private);
            descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            self.path_layer_textures
                .push(self.device.new_texture(&descriptor));
        }
    }

    /// One MSAA pass rasterizing every batch of a prepass layer, cropped to the
    /// layer's right/bottom extent so untouched tiles are neither cleared nor
    /// resolved. Sampling divides by the full texture size, so texels keep
    /// their window positions.
    fn rasterize_path_layer(
        &self,
        scene: &Scene,
        layer: &PathLayerPlan,
        layer_index: usize,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        command_buffer: &metal::CommandBufferRef,
    ) -> bool {
        let target = if layer_index == 0 {
            self.path_intermediate_texture.as_deref()
        } else {
            self.path_layer_textures
                .get(layer_index - 1)
                .map(|texture| texture.as_ref())
        };
        let Some(target) = target else {
            return false;
        };
        let commands = scene.render_commands();
        let mut vertex_count = 0;
        for &index in &layer.batches {
            if let RenderCommand::Batch(PrimitiveBatch::Paths {
                rasterization_vertex_count,
                ..
            }) = &commands[index]
            {
                vertex_count += rasterization_vertex_count;
            }
        }
        align_offset(instance_offset);
        let bytes_len = mem::size_of::<PathRasterizationVertex>() * vertex_count;
        let next_offset = *instance_offset + bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) }
                as *mut PathRasterizationVertex;
        let mut written = 0;
        for &index in &layer.batches {
            if let RenderCommand::Batch(PrimitiveBatch::Paths { range, .. }) = &commands[index] {
                for vertex in path_types::rasterization_vertices(&scene.paths[range.clone()]) {
                    if written == vertex_count {
                        return false;
                    }
                    unsafe { buffer_contents.add(written).write(vertex) };
                    written += 1;
                }
            }
        }
        if written != vertex_count {
            return false;
        }

        let (width, height) = layer.extent;
        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        render_pass_descriptor.set_render_target_width(width as u64);
        render_pass_descriptor.set_render_target_height(height as u64);
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));
        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(target));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(target));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }
        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_viewport(metal::MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: width as f64,
            height: height as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);
        // Clip space spans the cropped extent, so window coordinates land on
        // the same texels as in a full-size pass.
        let uniforms = SceneUniforms::new(size(
            DevicePixels(width as i32),
            DevicePixels(height as i32),
        ));
        bind_scene_uniforms(command_encoder, &uniforms);
        bind_instance_bytes(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            bytes_len,
        );
        command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, vertex_count as u64);
        command_encoder.end_encoding();
        *instance_offset = next_offset;
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_paths_from_texture(
        &self,
        paths: &[Path<ScaledPixels>],
        sprite_count: usize,
        texture: &metal::TextureRef,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if paths.is_empty() {
            return true;
        }
        align_offset(instance_offset);
        let sprite_bytes_len = mem::size_of::<path_types::PathSprite>() * sprite_count;
        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }
        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        bind_scene_uniforms(command_encoder, scene_uniforms);
        command_encoder.set_fragment_texture(PRIMARY_TEXTURE_SLOT, Some(texture));
        command_encoder.set_fragment_sampler_state(SAMPLER_SLOT, Some(&self.sampler));
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) }
                as *mut path_types::PathSprite;
        let mut sprites = path_types::sprites(paths);
        for index in 0..sprite_count {
            let Some(sprite) = sprites.next() else {
                return false;
            };
            unsafe { buffer_contents.add(index).write(sprite) };
        }
        if sprites.next().is_some() {
            return false;
        }
        bind_instance_bytes(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            sprite_bytes_len,
        );
        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::TriangleStrip,
            0,
            4,
            sprite_count as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        rasterization_vertex_count: usize,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_buffer: &metal::CommandBufferRef,
    ) -> bool {
        if paths.is_empty() {
            return true;
        }
        let Some(intermediate_texture) = &self.path_intermediate_texture else {
            return false;
        };

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        align_offset(instance_offset);
        let vertices_bytes_len =
            mem::size_of::<PathRasterizationVertex>() * rasterization_vertex_count;
        let next_offset = *instance_offset + vertices_bytes_len;
        if next_offset > instance_buffer.size {
            command_encoder.end_encoding();
            return false;
        }
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) }
                as *mut PathRasterizationVertex;
        let mut vertices = path_types::rasterization_vertices(paths);
        for index in 0..rasterization_vertex_count {
            let Some(vertex) = vertices.next() else {
                command_encoder.end_encoding();
                return false;
            };
            unsafe { buffer_contents.add(index).write(vertex) };
        }
        if vertices.next().is_some() {
            command_encoder.end_encoding();
            return false;
        }
        bind_instance_bytes(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            vertices_bytes_len,
        );
        command_encoder.draw_primitives(
            metal::MTLPrimitiveType::Triangle,
            0,
            rasterization_vertex_count as u64,
        );
        *instance_offset = next_offset;

        command_encoder.end_encoding();
        true
    }

    fn draw_shadows(
        &self,
        shadows: &[Shadow],
        smoothed: bool,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if shadows.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let shadow_bytes_len = mem::size_of_val(shadows);
        let next_offset = *instance_offset + shadow_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let pipeline = if smoothed {
            &self.smoothed_shadows_pipeline_state
        } else {
            &self.shadows_pipeline_state
        };
        command_encoder.set_render_pipeline_state(pipeline);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };
        unsafe {
            ptr::copy_nonoverlapping(
                shadows.as_ptr() as *const u8,
                buffer_contents,
                shadow_bytes_len,
            );
        }
        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            shadows,
        );

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::TriangleStrip,
            0,
            4,
            shadows.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_quads(
        &self,
        quads: &[Quad],
        smoothed: bool,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if quads.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let quad_bytes_len = mem::size_of_val(quads);
        let next_offset = *instance_offset + quad_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let pipeline = if smoothed {
            &self.smoothed_quads_pipeline_state
        } else {
            &self.quads_pipeline_state
        };
        command_encoder.set_render_pipeline_state(pipeline);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };
        unsafe {
            ptr::copy_nonoverlapping(quads.as_ptr() as *const u8, buffer_contents, quad_bytes_len);
        }

        if !smoothed && self.experiments.opaque_quads {
            // Runs of opaque and blended quads, in order. Opaque fragments
            // overwrite instead of blending, which lets the tile hardware drop
            // fragments a later opaque quad hides.
            let mut start = 0;
            while start < quads.len() {
                let opaque = quad_is_opaque(&quads[start]);
                let mut end = start + 1;
                while end < quads.len() && quad_is_opaque(&quads[end]) == opaque {
                    end += 1;
                }
                command_encoder.set_render_pipeline_state(if opaque {
                    &self.opaque_quads_pipeline_state
                } else {
                    &self.quads_pipeline_state
                });
                bind_instances(
                    command_encoder,
                    &instance_buffer.metal_buffer,
                    *instance_offset + start * mem::size_of::<Quad>(),
                    &quads[start..end],
                );
                command_encoder.draw_primitives_instanced(
                    metal::MTLPrimitiveType::TriangleStrip,
                    0,
                    4,
                    (end - start) as u64,
                );
                start = end;
            }
            *instance_offset = next_offset;
            return true;
        }

        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            quads,
        );

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::TriangleStrip,
            0,
            4,
            quads.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        sprite_count: usize,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if paths.is_empty() {
            return true;
        }
        let Some(intermediate_texture) = self.path_intermediate_texture.as_deref() else {
            return false;
        };
        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        self.draw_paths_from_texture(
            paths,
            sprite_count,
            intermediate_texture,
            instance_buffer,
            instance_offset,
            scene_uniforms,
            command_encoder,
        )
    }

    fn draw_underlines(
        &self,
        underlines: &[Underline],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if underlines.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let underline_bytes_len = mem::size_of_val(underlines);
        let next_offset = *instance_offset + underline_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };
        unsafe {
            ptr::copy_nonoverlapping(
                underlines.as_ptr() as *const u8,
                buffer_contents,
                underline_bytes_len,
            );
        }
        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            underlines,
        );

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::TriangleStrip,
            0,
            4,
            underlines.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[MonochromeSprite],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let sprite_bytes_len = mem::size_of_val(sprites);
        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let texture = self.sprite_atlas.metal_texture(texture_id);
        command_encoder.set_render_pipeline_state(&self.monochrome_sprites_pipeline_state);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };
        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }
        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            sprites,
        );
        // Generated native shaders derive tile coordinates from the atlas size in the
        // vertex stage; bind the same texture to both stages, as WGPU and DirectX do.
        command_encoder.set_vertex_texture(PRIMARY_TEXTURE_SLOT, Some(&texture));
        command_encoder.set_fragment_texture(PRIMARY_TEXTURE_SLOT, Some(&texture));
        command_encoder.set_fragment_sampler_state(SAMPLER_SLOT, Some(&self.sampler));

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::TriangleStrip,
            0,
            4,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[PolychromeSprite],
        smoothed: bool,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let sprite_bytes_len = mem::size_of_val(sprites);
        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let pipeline = if smoothed {
            &self.smoothed_polychrome_sprites_pipeline_state
        } else {
            &self.polychrome_sprites_pipeline_state
        };
        command_encoder.set_render_pipeline_state(pipeline);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };
        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }
        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            sprites,
        );
        command_encoder.set_vertex_texture(PRIMARY_TEXTURE_SLOT, Some(&texture));
        command_encoder.set_fragment_texture(PRIMARY_TEXTURE_SLOT, Some(&texture));
        command_encoder.set_fragment_sampler_state(SAMPLER_SLOT, Some(&self.sampler));

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::TriangleStrip,
            0,
            4,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        opacities: &[f32],
        scene_uniforms: &SceneUniforms,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
        bind_scene_uniforms(command_encoder, scene_uniforms);
        command_encoder.set_fragment_sampler_state(SAMPLER_SLOT, Some(&self.sampler));

        for (index, surface) in surfaces.iter().enumerate() {
            let image_buffer = match &surface.source {
                SurfaceSource::Surface(image_buffer) => image_buffer,
                SurfaceSource::Unsupported(size) => {
                    log::error!("Metal cannot draw unsupported surface source with size {size:?}");
                    continue;
                }
                // WGPU textures exist when gpui's `custom-gpu` feature is on; only
                // the WGPU renderer (gpui_macos's `wgpu` feature) can draw them.
                #[allow(unreachable_patterns)]
                _ => {
                    log::error!("Metal cannot draw WGPU texture surfaces");
                    continue;
                }
            };

            assert_eq!(
                image_buffer.get_pixel_format(),
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            );

            let y_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    &image_buffer.as_image_buffer(),
                    None,
                    MTLPixelFormat::R8Unorm,
                    image_buffer.get_width_of_plane(0),
                    image_buffer.get_height_of_plane(0),
                    0,
                )
                .unwrap();
            let cb_cr_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    &image_buffer.as_image_buffer(),
                    None,
                    MTLPixelFormat::RG8Unorm,
                    image_buffer.get_width_of_plane(1),
                    image_buffer.get_height_of_plane(1),
                    1,
                )
                .unwrap();

            let surface_uniforms = SurfaceUniforms {
                bounds: surface.bounds.into(),
                content_mask: surface.content_mask.into(),
                color_format: SurfaceColorFormat::Yuv,
                opacity: opacities.get(index).copied().unwrap_or(1.0),
                padding0: 0,
                padding1: 0,
                padding2: 0,
                padding3: 0,
                padding4: 0,
                padding5: 0,
            };
            command_encoder.set_vertex_bytes(
                DATA_SLOT,
                mem::size_of::<SurfaceUniforms>() as u64,
                &surface_uniforms as *const SurfaceUniforms as *const _,
            );
            command_encoder.set_fragment_bytes(
                DATA_SLOT,
                mem::size_of::<SurfaceUniforms>() as u64,
                &surface_uniforms as *const SurfaceUniforms as *const _,
            );
            command_encoder.set_fragment_texture(PRIMARY_TEXTURE_SLOT, unsafe {
                let texture = CVMetalTextureGetTexture(y_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });
            command_encoder.set_fragment_texture(SECONDARY_TEXTURE_SLOT, unsafe {
                let texture = CVMetalTextureGetTexture(cb_cr_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });

            command_encoder.draw_primitives(metal::MTLPrimitiveType::TriangleStrip, 0, 4);
        }
        true
    }
}

fn new_command_encoder_for_texture<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    texture: &'a metal::TextureRef,
    viewport_size: Size<DevicePixels>,
    configure_color_attachment: impl Fn(&RenderPassColorAttachmentDescriptorRef),
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(texture));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    configure_color_attachment(color_attachment);

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    shader: &NativeShader,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(shader.vertex_entry, None)
        .unwrap_or_else(|error| panic!("error locating {}: {error}", shader.vertex_entry));
    let fragment_fn = library
        .get_function(shader.fragment_entry, None)
        .unwrap_or_else(|error| panic!("error locating {}: {error}", shader.fragment_entry));

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(shader.label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

/// `build_pipeline_state` without blending: fragments overwrite the target.
fn build_opaque_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    shader: &NativeShader,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(shader.vertex_entry, None)
        .unwrap_or_else(|error| panic!("error locating {}: {error}", shader.vertex_entry));
    let fragment_fn = library
        .get_function(shader.fragment_entry, None)
        .unwrap_or_else(|error| panic!("error locating {}: {error}", shader.fragment_entry));

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(shader.label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(false);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    shader: &NativeShader,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(shader.vertex_entry, None)
        .unwrap_or_else(|error| panic!("error locating {}: {error}", shader.vertex_entry));
    let fragment_fn = library
        .get_function(shader.fragment_entry, None)
        .unwrap_or_else(|error| panic!("error locating {}: {error}", shader.fragment_entry));

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(shader.label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    shader: &NativeShader,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(shader.vertex_entry, None)
        .unwrap_or_else(|error| panic!("error locating {}: {error}", shader.vertex_entry));
    let fragment_fn = library
        .get_function(shader.fragment_entry, None)
        .unwrap_or_else(|error| panic!("error locating {}: {error}", shader.fragment_entry));

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(shader.label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

// Blur downsample/gaussian passes overwrite their target (no blending). The composite pass
// uses the normal alpha-blending pipeline (`build_path_sprite_pipeline_state`) instead.
fn build_blur_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    shader: &NativeShader,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(shader.vertex_entry, None)
        .unwrap_or_else(|error| panic!("error locating {}: {error}", shader.vertex_entry));
    let fragment_fn = library
        .get_function(shader.fragment_entry, None)
        .unwrap_or_else(|error| panic!("error locating {}: {error}", shader.fragment_entry));

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(shader.label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(false);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn required_instance_buffer_size(scene: &Scene) -> usize {
    let mut required = 0;
    let mut reserve = |element_size: usize, count: usize| {
        if count > 0 {
            align_offset(&mut required);
            required = required.saturating_add(element_size.saturating_mul(count));
        }
    };

    for command in scene.render_commands() {
        let RenderCommand::Batch(batch) = command else {
            continue;
        };
        match batch {
            PrimitiveBatch::Shadows { range, .. } => reserve(mem::size_of::<Shadow>(), range.len()),
            PrimitiveBatch::Quads { range, .. } => reserve(mem::size_of::<Quad>(), range.len()),
            PrimitiveBatch::Paths {
                rasterization_vertex_count,
                sprite_count,
                ..
            } if *rasterization_vertex_count > 0 => {
                reserve(
                    mem::size_of::<path_types::PathRasterizationVertex>(),
                    *rasterization_vertex_count,
                );
                reserve(mem::size_of::<path_types::PathSprite>(), *sprite_count);
            }
            PrimitiveBatch::Underlines(range) => reserve(mem::size_of::<Underline>(), range.len()),
            PrimitiveBatch::MonochromeSprites { range, .. } => {
                reserve(mem::size_of::<MonochromeSprite>(), range.len())
            }
            PrimitiveBatch::PolychromeSprites { range, .. } => {
                reserve(mem::size_of::<PolychromeSprite>(), range.len())
            }
            PrimitiveBatch::Paths { .. }
            | PrimitiveBatch::SubpixelSprites { .. }
            | PrimitiveBatch::Surfaces(_)
            | PrimitiveBatch::BackdropFilters(_)
            | PrimitiveBatch::FilterBoundary(_) => {}
        }
    }
    required
}

// Align to multiples of 256 make Metal happy.
fn align_offset(offset: &mut usize) {
    *offset = (*offset).div_ceil(256) * 256;
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Self {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool);
        Self { renderer }
    }
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for MetalHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_scene_to_image(scene, size)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.render_scene(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{
        AtlasKey, BackdropFilter, BorderStyle, ContentMask, Edges, ImageId, Path, PlatformAtlas,
        PlatformHeadlessRenderer, RenderImageParams, RenderSvgParams, ScaledFilter,
        TransformationMatrix, checkerboard, hsla, linear_color_stop, linear_gradient,
        pattern_slash, px, solid_background, white,
    };
    use std::borrow::Cow;

    #[test]
    fn precompiled_libraries_load_with_their_entry_points() {
        let device = metal::Device::system_default().expect("Metal device");
        // Builds without the Metal Toolchain embed no libraries and compile MSL.
        for shader in NATIVE_SHADERS {
            let Some(bytes) = shader.metallib else {
                continue;
            };
            let library = device
                .new_library_with_data(bytes)
                .unwrap_or_else(|error| panic!("{}: {error}", shader.label));
            for entry in [shader.vertex_entry, shader.fragment_entry] {
                library
                    .get_function(entry, None)
                    .unwrap_or_else(|error| panic!("{} {entry}: {error}", shader.label));
            }
        }
    }

    #[test]
    fn intermediate_textures_follow_scene_requirements() {
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut renderer = MetalRenderer::new_headless(pool);
        let target_size = size(DevicePixels(5), DevicePixels(5));
        let mut empty_scene = Scene::default();
        empty_scene.finish();

        renderer.prepare_intermediate_textures(&empty_scene, target_size);
        assert!(renderer.path_intermediate_texture.is_none());
        assert!(renderer.scene_color_texture.is_none());
        assert!(renderer.blur_ping_texture.is_none());
        assert!(renderer.group_textures.is_empty());

        let bounds = Bounds {
            origin: gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size: size(ScaledPixels(5.0), ScaledPixels(5.0)),
        };
        let mut filtered_scene = Scene::default();
        filtered_scene.insert_primitive(BackdropFilter {
            bounds,
            content_mask: ContentMask {
                bounds,
                ..Default::default()
            },
            filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(1.0))],
            opacity: 1.0,
            ..Default::default()
        });
        filtered_scene.finish();

        renderer.prepare_intermediate_textures(&filtered_scene, target_size);
        assert!(renderer.path_intermediate_texture.is_none());
        assert!(renderer.scene_color_texture.is_some());
        assert!(renderer.blur_ping_texture.is_some());
        assert!(renderer.blur_pong_texture.is_some());
        assert!(renderer.group_textures.is_empty());
    }

    #[test]
    fn generated_sprite_shader_preserves_monochrome_coverage() {
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut renderer = MetalRenderer::new_headless(pool);
        let tile_size = Size {
            width: DevicePixels(8),
            height: DevicePixels(8),
        };
        let key = AtlasKey::Svg(RenderSvgParams {
            path: "generated-sprite-coverage-test".into(),
            size: tile_size,
        });
        let tile = renderer
            .sprite_atlas()
            .get_or_insert_with(&key, &mut || {
                Ok(Some((tile_size, Cow::Owned(vec![64; 64]))))
            })
            .unwrap()
            .unwrap();
        let bounds = Bounds {
            origin: gpui::point(ScaledPixels(8.0), ScaledPixels(8.0)),
            size: Size {
                width: ScaledPixels(16.0),
                height: ScaledPixels(16.0),
            },
        };
        let mut scene = Scene::default();
        scene.insert_primitive(MonochromeSprite {
            order: 0,
            padding: 0,
            bounds,
            content_mask: ContentMask {
                bounds,
                ..Default::default()
            },
            color: white().into(),
            tile,
            transformation: TransformationMatrix::unit(),
        });
        scene.finish();

        let image = renderer
            .render_scene_to_image(
                &scene,
                Size {
                    width: DevicePixels(32),
                    height: DevicePixels(32),
                },
            )
            .unwrap();
        let center = image.get_pixel(16, 16).0;
        assert!(
            center[0] >= 63 && center[0] <= 65 && center[1] == center[0] && center[2] == center[0],
            "monochrome coverage was altered before blending: {center:?}"
        );
    }

    /// Pixel spot-check helpers for the generated-shader contract tests below.
    ///
    /// Headless Metal runs on whichever Apple GPU the host provides, and the
    /// shared shaders exercise `sin`/`pow`/`exp`/MSAA paths whose last-ulp
    /// rounding varies across GPU generations and Metal compiler versions.
    /// These tests therefore pin exact bytes only where the math is bit-exact
    /// (solid colors, cleared background) and use small tolerances plus
    /// structural checks (coverage ramps, dash gaps, blur falloff) everywhere
    /// else. Deterministic byte-exact parity for quad backgrounds is covered
    /// by `quad_backgrounds_match_legacy_metal_pixels` in `gpui_wgpu`, which
    /// runs on a software renderer.
    fn pixel(image: &RgbaImage, x: u32, y: u32) -> [u8; 4] {
        image.get_pixel(x, y).0
    }

    fn assert_pixel_close(
        image: &RgbaImage,
        x: u32,
        y: u32,
        expected: [u8; 4],
        tolerance: u8,
        what: &str,
    ) {
        let actual = pixel(image, x, y);
        for (channel, (observed, reference)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                observed.abs_diff(*reference) <= tolerance,
                "{what} pixel ({x}, {y}) channel {channel}: got {actual:?}, expected {expected:?} ± {tolerance}",
            );
        }
    }

    /// Renders `scene` headlessly and checks the output size. The pixel
    /// contracts below only hold where Metal actually executes, so non-macOS
    /// builds stop after the dimension check.
    fn render_for_contracts(
        renderer: &mut MetalHeadlessRenderer,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Option<RgbaImage> {
        let image = renderer.render_scene_to_image(scene, size).unwrap();
        assert_eq!(
            image.dimensions(),
            (size.width.0 as u32, size.height.0 as u32)
        );
        if cfg!(target_os = "macos") {
            Some(image)
        } else {
            None
        }
    }

    #[test]
    fn generated_quad_shaders_render_background_contracts() {
        let mut renderer = MetalHeadlessRenderer::new();
        let bounds = |x, y| Bounds {
            origin: gpui::point(ScaledPixels(x), ScaledPixels(y)),
            size: Size {
                width: ScaledPixels(2.0),
                height: ScaledPixels(2.0),
            },
        };
        let mut quads = Scene::default();
        for (bounds, background) in [
            (bounds(0.0, 0.0), solid_background(hsla(0.0, 1.0, 0.5, 1.0))),
            (
                bounds(2.0, 0.0),
                checkerboard(hsla(0.6, 0.7, 0.5, 1.0), 1.0),
            ),
            (
                bounds(0.0, 2.0),
                pattern_slash(hsla(0.3, 0.8, 0.5, 1.0), 1.0, 1.0),
            ),
            (
                bounds(2.0, 2.0),
                linear_gradient(
                    45.0,
                    linear_color_stop(hsla(0.8, 0.9, 0.4, 1.0), 0.0),
                    linear_color_stop(hsla(0.1, 0.8, 0.6, 1.0), 1.0),
                ),
            ),
        ] {
            quads.insert_primitive(Quad {
                bounds,
                content_mask: ContentMask {
                    bounds,
                    ..Default::default()
                },
                background,
                ..Default::default()
            });
        }
        quads.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &quads,
            Size {
                width: DevicePixels(4),
                height: DevicePixels(4),
            },
        ) else {
            return;
        };

        // Solid red is exactly representable, so it must round-trip untouched.
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            assert_pixel_close(&image, x, y, [255, 0, 0, 255], 0, "solid background");
        }
        // The checkerboard alternates cleared black with the pattern color.
        // The pattern color is pure ALU math (no transcendentals), so it is tight.
        assert_pixel_close(&image, 2, 0, [0, 0, 0, 255], 1, "checkerboard off-square");
        assert_pixel_close(
            &image,
            3,
            0,
            [38, 110, 217, 255],
            2,
            "checkerboard on-square",
        );
        assert_pixel_close(
            &image,
            2,
            1,
            [38, 110, 217, 255],
            2,
            "checkerboard on-square",
        );
        assert_pixel_close(&image, 3, 1, [0, 0, 0, 255], 1, "checkerboard off-square");
        // Slash coverage runs through trig (`sin`) and derivatives, so assert
        // hue structure instead of exact bytes: green-dominant, lit, varying.
        let slash = [
            pixel(&image, 0, 2),
            pixel(&image, 1, 2),
            pixel(&image, 0, 3),
            pixel(&image, 1, 3),
        ];
        for (i, texel) in slash.iter().enumerate() {
            assert!(
                texel[1] > texel[0] && texel[0] > texel[2] && texel[1] > 20,
                "slash texel {i} keeps its green hue: {texel:?}",
            );
        }
        let mut slash_distinct = 0;
        for (i, texel) in slash.iter().enumerate() {
            if slash[..i].iter().all(|other| other != texel) {
                slash_distinct += 1;
            }
        }
        assert!(
            slash_distinct >= 2,
            "slash pattern varies across the quadrant: {slash:?}",
        );
        // The gradient interpolates in Oklab with per-pixel dither, so its
        // texels must be shaded (never flat red/black) and almost all distinct.
        let gradient = [
            pixel(&image, 2, 2),
            pixel(&image, 3, 2),
            pixel(&image, 2, 3),
            pixel(&image, 3, 3),
        ];
        for (i, texel) in gradient.iter().enumerate() {
            assert_ne!(*texel, [255, 0, 0, 255], "gradient texel {i} is shaded");
            assert_ne!(*texel, [0, 0, 0, 255], "gradient texel {i} is shaded");
        }
        let mut gradient_distinct = 0;
        for (i, texel) in gradient.iter().enumerate() {
            if gradient[..i].iter().all(|other| other != texel) {
                gradient_distinct += 1;
            }
        }
        assert!(
            gradient_distinct >= 3,
            "gradient interpolates across the quadrant: {gradient:?}",
        );
    }

    #[test]
    fn generated_underline_shaders_render_line_contracts() {
        let mut renderer = MetalHeadlessRenderer::new();
        let underline_bounds = Bounds {
            origin: gpui::point(ScaledPixels(1.5), ScaledPixels(3.5)),
            size: Size {
                width: ScaledPixels(5.0),
                height: ScaledPixels(1.5),
            },
        };
        let underline_frame = Bounds {
            origin: gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size: Size {
                width: ScaledPixels(8.0),
                height: ScaledPixels(8.0),
            },
        };
        let mut underline = Scene::default();
        underline.insert_primitive(Quad {
            bounds: underline_frame,
            content_mask: ContentMask {
                bounds: underline_frame,
                ..Default::default()
            },
            background: solid_background(hsla(0.0, 0.0, 0.1, 1.0)),
            ..Default::default()
        });
        underline.insert_primitive(Underline {
            order: 0,
            padding: 0,
            bounds: underline_bounds,
            content_mask: ContentMask {
                bounds: underline_bounds,
                ..Default::default()
            },
            color: hsla(0.6, 0.8, 0.6, 0.65).into(),
            thickness: ScaledPixels(1.0),
            wavy: gpui::ShaderBool::Disabled,
        });
        underline.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &underline,
            Size {
                width: DevicePixels(8),
                height: DevicePixels(8),
            },
        ) else {
            return;
        };
        assert_pixel_close(&image, 0, 0, [26, 26, 26, 255], 1, "underline backdrop");
        assert_pixel_close(&image, 7, 7, [26, 26, 26, 255], 1, "underline backdrop");
        assert_pixel_close(&image, 3, 3, [56, 98, 162, 255], 3, "straight underline");
        assert_pixel_close(&image, 3, 4, [56, 98, 162, 255], 3, "straight underline");

        let mut wavy = Scene::default();
        wavy.insert_primitive(Underline {
            order: 0,
            padding: 0,
            bounds: underline_bounds,
            content_mask: ContentMask {
                bounds: underline_bounds,
                ..Default::default()
            },
            color: hsla(0.1, 0.9, 0.55, 1.0).into(),
            thickness: ScaledPixels(1.0),
            wavy: gpui::ShaderBool::Enabled,
        });
        wavy.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &wavy,
            Size {
                width: DevicePixels(8),
                height: DevicePixels(8),
            },
        ) else {
            return;
        };
        assert_pixel_close(
            &image,
            0,
            0,
            [0, 0, 0, 255],
            1,
            "wavy underline clears outside the line",
        );
        // The wave must actually oscillate instead of drawing a straight band,
        // and its warm line color must show up somewhere.
        let raw = image.as_raw();
        assert_ne!(
            &raw[3 * 8 * 4..4 * 8 * 4],
            &raw[4 * 8 * 4..5 * 8 * 4],
            "wavy underline oscillates between rows",
        );
        assert!(
            image
                .pixels()
                .any(|texel| texel.0[0] > 100 && texel.0[0] > texel.0[2]),
            "wavy underline draws its warm line color",
        );
    }

    #[test]
    fn generated_border_shader_renders_dashed_rounded_rect() {
        let mut renderer = MetalHeadlessRenderer::new();
        let full = Bounds {
            origin: gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size: Size {
                width: ScaledPixels(16.0),
                height: ScaledPixels(16.0),
            },
        };
        let box_bounds = Bounds {
            origin: gpui::point(ScaledPixels(4.0), ScaledPixels(4.0)),
            size: Size {
                width: ScaledPixels(8.0),
                height: ScaledPixels(8.0),
            },
        };
        let mut border = Scene::default();
        border.insert_primitive(Quad {
            bounds: box_bounds,
            content_mask: ContentMask {
                bounds: full,
                ..Default::default()
            },
            background: solid_background(hsla(0.05, 0.8, 0.45, 1.0)),
            border_style: BorderStyle::Dashed,
            border_color: hsla(0.6, 0.9, 0.7, 1.0).into(),
            corner_radii: Corners::all(ScaledPixels(2.0)),
            border_widths: Edges::all(ScaledPixels(1.0)),
            ..Default::default()
        });
        border.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &border,
            Size {
                width: DevicePixels(16),
                height: DevicePixels(16),
            },
        ) else {
            return;
        };
        assert_pixel_close(
            &image,
            0,
            0,
            [0, 0, 0, 255],
            1,
            "outside the rounded rect stays clear",
        );
        assert_pixel_close(&image, 8, 8, [207, 78, 23, 255], 3, "interior fill");
        // Dashes: along the top edge some texels carry the border hue while
        // others fall back to gaps. A dash period spans 3px over an 8px edge,
        // so both must appear regardless of dash phase.
        let edge: Vec<[u8; 4]> = (4..12).map(|x| pixel(&image, x, 4)).collect();
        let border_like = edge
            .iter()
            .filter(|texel| texel[2] > 150 && texel[1] > 100)
            .count();
        assert!(
            (1..8).contains(&border_like),
            "top edge mixes dash and gap texels: {edge:?}",
        );

        let mut custom_border = Scene::default();
        custom_border.insert_primitive(Quad {
            bounds: box_bounds,
            content_mask: ContentMask {
                bounds: full,
                ..Default::default()
            },
            background: solid_background(hsla(0.05, 0.8, 0.45, 1.0)),
            border_style: BorderStyle::Dashed,
            border_dashed_length: 4.0,
            border_dashed_gap: 0.5,
            border_color: hsla(0.6, 0.9, 0.7, 1.0).into(),
            corner_radii: Corners::all(ScaledPixels(2.0)),
            border_widths: Edges::all(ScaledPixels(1.0)),
            ..Default::default()
        });

        custom_border.finish();
        let Some(custom_image) = render_for_contracts(
            &mut renderer,
            &custom_border,
            Size {
                width: DevicePixels(16),
                height: DevicePixels(16),
            },
        ) else {
            return;
        };

        assert_ne!(
            image.as_raw(),
            custom_image.as_raw(),
            "custom dash length and gap change the rendered border",
        );
    }

    #[test]
    fn generated_shadow_shader_renders_soft_falloff() {
        let mut renderer = MetalHeadlessRenderer::new();
        let full = Bounds {
            origin: gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size: Size {
                width: ScaledPixels(16.0),
                height: ScaledPixels(16.0),
            },
        };
        let box_bounds = Bounds {
            origin: gpui::point(ScaledPixels(4.0), ScaledPixels(4.0)),
            size: Size {
                width: ScaledPixels(8.0),
                height: ScaledPixels(8.0),
            },
        };
        let mut shadow = Scene::default();
        shadow.insert_primitive(Shadow {
            order: 0,
            blur_radius: ScaledPixels(2.0),
            bounds: box_bounds,
            content_mask: ContentMask {
                bounds: full,
                ..Default::default()
            },
            corner_radii: Corners::all(ScaledPixels(2.0)),
            color: hsla(0.7, 0.8, 0.3, 0.7).into(),
            element_bounds: box_bounds,
            element_corner_radii: Corners::all(ScaledPixels(2.0)),
            inset: gpui::ShaderBool::Disabled,
            corner_smoothing: 0.0,
        });
        shadow.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &shadow,
            Size {
                width: DevicePixels(16),
                height: DevicePixels(16),
            },
        ) else {
            return;
        };
        let center = pixel(&image, 8, 8);
        let corner = pixel(&image, 0, 0);
        assert!(
            center[0] > 10,
            "shadow covers the element center: {center:?}",
        );
        assert!(
            corner[0] < 5,
            "shadow falls off to clear at the corner: {corner:?}",
        );
        assert!(
            center[0] > corner[0],
            "shadow has a coverage ramp: center {center:?}, corner {corner:?}",
        );
        // The opaque headless target composites over black, so alpha stays 1.
        assert_eq!(center[3], 255, "shadow center alpha: {center:?}");
        assert_eq!(corner[3], 255, "shadow corner alpha: {corner:?}");
    }

    #[test]
    fn generated_blur_shader_softens_backdrop() {
        let mut renderer = MetalHeadlessRenderer::new();
        let full = Bounds {
            origin: gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size: Size {
                width: ScaledPixels(16.0),
                height: ScaledPixels(16.0),
            },
        };
        let box_bounds = Bounds {
            origin: gpui::point(ScaledPixels(4.0), ScaledPixels(4.0)),
            size: Size {
                width: ScaledPixels(8.0),
                height: ScaledPixels(8.0),
            },
        };
        let mut filter = Scene::default();
        filter.insert_primitive(Quad {
            bounds: full,
            content_mask: ContentMask {
                bounds: full,
                ..Default::default()
            },
            background: checkerboard(hsla(0.2, 0.7, 0.5, 1.0), 2.0),
            ..Default::default()
        });
        filter.insert_primitive(BackdropFilter {
            bounds: box_bounds,
            content_mask: ContentMask {
                bounds: full,
                ..Default::default()
            },
            corner_radii: Corners::all(ScaledPixels(2.0)),
            filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(1.0))],
            opacity: 0.8,
            ..Default::default()
        });
        filter.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &filter,
            Size {
                width: DevicePixels(16),
                height: DevicePixels(16),
            },
        ) else {
            return;
        };
        // Outside the filter region the checkerboard is untouched.
        assert_pixel_close(&image, 0, 0, [0, 0, 0, 255], 1, "unfiltered off-square");
        assert_pixel_close(&image, 2, 0, [181, 217, 38, 255], 2, "unfiltered on-square");
        // Inside, the blur must mix neighbors: neither sharp nor clear.
        let soft = pixel(&image, 8, 8);
        let sharp = [181, 217, 38, 255];
        assert_ne!(
            soft,
            [0, 0, 0, 255],
            "blurred texel mixes content: {soft:?}"
        );
        let drift = soft
            .iter()
            .zip(sharp.iter())
            .map(|(observed, reference)| observed.abs_diff(*reference))
            .max()
            .unwrap_or(0);
        assert!(
            drift >= 15,
            "blur mixes neighboring texels instead of passing through: {soft:?}",
        );
    }

    #[test]
    fn generated_sprite_shader_blends_opacity() {
        let mut renderer = MetalHeadlessRenderer::new();
        let key = AtlasKey::Image(RenderImageParams {
            image_id: ImageId(0),
            frame_index: 0,
        });
        let tile = renderer
            .sprite_atlas()
            .get_or_insert_with(&key, &mut || {
                Ok(Some((
                    Size {
                        width: DevicePixels(2),
                        height: DevicePixels(2),
                    },
                    Cow::Borrowed(&[
                        0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 0, 255, 255, 255, 255, 255,
                    ]),
                )))
            })
            .unwrap()
            .unwrap();
        let sprite_bounds = Bounds {
            origin: gpui::point(ScaledPixels(1.0), ScaledPixels(1.0)),
            size: Size {
                width: ScaledPixels(6.0),
                height: ScaledPixels(6.0),
            },
        };
        let mut sprite = Scene::default();
        sprite.insert_primitive(PolychromeSprite {
            order: 0,
            grayscale: gpui::ShaderBool::Disabled,
            opacity: 0.75,
            corner_smoothing: 0.0,
            bounds: sprite_bounds,
            content_mask: ContentMask {
                bounds: sprite_bounds,
                ..Default::default()
            },
            corner_radii: Corners::all(ScaledPixels(1.0)),
            tile,
        });
        sprite.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &sprite,
            Size {
                width: DevicePixels(8),
                height: DevicePixels(8),
            },
        ) else {
            return;
        };
        assert_pixel_close(
            &image,
            0,
            0,
            [0, 0, 0, 255],
            1,
            "outside the sprite stays clear",
        );
        // Texel centers sample exactly, so the red and white texels must land
        // at 75% opacity (255 * 0.75 = 191.25 -> 191).
        assert_pixel_close(
            &image,
            2,
            2,
            [191, 0, 0, 255],
            2,
            "red texel at 75% opacity",
        );
        assert_pixel_close(
            &image,
            5,
            5,
            [191, 191, 191, 255],
            2,
            "white texel at 75% opacity",
        );
        // The 2x2 tile holds saturated texels, so the sprite interior must be
        // lit but capped by the 75% opacity (a full white texel lands at 191).
        let mut sum = [0u64; 3];
        let mut count = 0u64;
        let mut brightest = 0u8;
        for y in 2..6 {
            for x in 2..6 {
                let texel = pixel(&image, x, y);
                for channel in 0..3 {
                    sum[channel] += u64::from(texel[channel]);
                    brightest = brightest.max(texel[channel]);
                }
                count += 1;
            }
        }
        let mean = [sum[0] / count, sum[1] / count, sum[2] / count];
        for (channel, level) in mean.iter().enumerate() {
            assert!(
                (40..=200).contains(level),
                "sprite interior channel {channel} looks sampled and blended, not degenerate: {mean:?}",
            );
        }
        assert!(
            (150..=195).contains(&brightest),
            "75% opacity caps the brightest texel near 191, got {brightest}",
        );
    }

    #[test]
    fn generated_path_shader_fills_and_antialiases() {
        let mut renderer = MetalHeadlessRenderer::new();
        let mut path = Path::new(gpui::point(px(1.0), px(1.0)));
        path.line_to(gpui::point(px(7.0), px(1.0)));
        path.line_to(gpui::point(px(4.0), px(7.0)));
        path.line_to(gpui::point(px(1.0), px(1.0)));
        path.content_mask = ContentMask {
            bounds: Bounds {
                origin: gpui::point(px(0.0), px(0.0)),
                size: Size {
                    width: px(8.0),
                    height: px(8.0),
                },
            },
            ..Default::default()
        };
        path.color = solid_background(hsla(0.45, 0.9, 0.45, 1.0));
        let mut path_scene = Scene::default();
        path_scene.insert_primitive(path.scale(1.0));
        path_scene.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &path_scene,
            Size {
                width: DevicePixels(8),
                height: DevicePixels(8),
            },
        ) else {
            return;
        };
        assert_pixel_close(
            &image,
            4,
            2,
            [11, 218, 156, 255],
            2,
            "triangle interior fill",
        );
        assert_pixel_close(
            &image,
            0,
            7,
            [0, 0, 0, 255],
            1,
            "outside the triangle stays clear",
        );
        // MSAA edges must produce partially covered texels: strictly between
        // clear and the fill color along the green channel.
        assert!(
            image.pixels().any(|texel| {
                let g = texel.0[1];
                g > 20 && g < 200 && texel.0[0] < 11
            }),
            "triangle edges are antialiased",
        );
    }

    fn experiment_renderer(experiments: Experiments) -> MetalRenderer {
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut renderer = MetalRenderer::new_headless(pool);
        renderer.experiments = experiments;
        renderer
    }

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: gpui::point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(width), ScaledPixels(height)),
        }
    }

    fn triangle(x: f32, y: f32, extent: f32, hue: f32, alpha: f32) -> Path<ScaledPixels> {
        let mut path = Path::new(gpui::point(px(x), px(y)));
        path.line_to(gpui::point(px(x + extent), px(y + extent * 0.15)));
        path.line_to(gpui::point(px(x + extent * 0.4), px(y + extent)));
        path.line_to(gpui::point(px(x), px(y)));
        path.content_mask = ContentMask {
            bounds: Bounds {
                origin: gpui::point(px(0.0), px(0.0)),
                size: size(px(4096.0), px(4096.0)),
            },
            ..Default::default()
        };
        path.color = solid_background(hsla(hue, 0.8, 0.5, alpha));
        path.scale(1.0)
    }

    fn quad(bounds: Bounds<ScaledPixels>, hue: f32, alpha: f32) -> Quad {
        Quad {
            bounds,
            content_mask: ContentMask {
                bounds,
                ..Default::default()
            },
            background: solid_background(hsla(hue, 0.6, 0.6, alpha)),
            ..Default::default()
        }
    }

    fn assert_same_pixels(expected: &RgbaImage, actual: &RgbaImage, what: &str) {
        assert_eq!(expected.dimensions(), actual.dimensions());
        let differing: Vec<_> = expected
            .enumerate_pixels()
            .filter(|(x, y, pixel)| actual.get_pixel(*x, *y) != *pixel)
            .take(5)
            .map(|(x, y, pixel)| (x, y, pixel.0, actual.get_pixel(x, y).0))
            .collect();
        assert!(
            differing.is_empty(),
            "{what}: pixels differ, e.g. {differing:?}"
        );
    }

    /// Path batches separated by other primitives, both overlapping (one
    /// layer each) and disjoint (sharing a layer), plus more overlapping
    /// batches than layers so the frame falls back to per-batch passes.
    fn path_layer_scenes() -> Vec<(&'static str, Scene)> {
        let mut scenes = Vec::new();

        // Fill, a quad over it, then an outline over both (the repo tab).
        let mut scene = Scene::default();
        scene.insert_primitive(triangle(4.0, 4.0, 40.0, 0.1, 1.0));
        scene.insert_primitive(quad(rect(14.0, 14.0, 12.0, 8.0), 0.6, 1.0));
        scene.insert_primitive(triangle(10.0, 10.0, 36.0, 0.3, 0.7));
        // Disjoint batches far away, each over its own quad.
        for row in 0..4 {
            let y = 60.0 + row as f32 * 22.0;
            scene.insert_primitive(quad(rect(70.0, y, 30.0, 20.0), 0.5, 1.0));
            scene.insert_primitive(triangle(72.0, y + 2.0, 16.0, 0.05 * row as f32, 0.9));
            scene.insert_primitive(quad(rect(80.0, y + 4.0, 6.0, 6.0), 0.9, 0.5));
        }
        scene.finish();
        scenes.push(("overlapping and disjoint batches", scene));

        // Five stacked batches over the same spot exceed the layer limit.
        let mut scene = Scene::default();
        for index in 0..5 {
            let offset = index as f32 * 3.0;
            scene.insert_primitive(triangle(
                8.0 + offset,
                8.0 + offset,
                40.0,
                0.15 * index as f32,
                0.8,
            ));
            scene.insert_primitive(quad(rect(20.0 + offset, 20.0 + offset, 8.0, 8.0), 0.7, 0.6));
        }
        scene.finish();
        scenes.push(("more overlapping batches than layers", scene));
        scenes
    }

    #[test]
    fn path_prepass_matches_per_batch_passes() {
        let size = size(DevicePixels(160), DevicePixels(160));
        for (what, scene) in path_layer_scenes() {
            let legacy = experiment_renderer(Experiments::default())
                .render_scene_to_image(&scene, size)
                .unwrap();
            let prepass = experiment_renderer(Experiments {
                path_prepass: true,
                ..Default::default()
            })
            .render_scene_to_image(&scene, size)
            .unwrap();
            assert_same_pixels(&legacy, &prepass, what);
        }
    }

    #[test]
    fn path_prepass_plans_layers_by_sampled_region() {
        let size = size(DevicePixels(160), DevicePixels(160));
        let scenes = path_layer_scenes();
        let plan = plan_path_prepass(&scenes[0].1, size).expect("fits in the layer limit");
        assert_eq!(plan.layers.len(), 2, "the outline overlaps the fill");
        let batches: usize = plan.layers.iter().map(|layer| layer.batches.len()).sum();
        assert!(batches >= 3, "disjoint rows share layers: {batches}");
        for layer in &plan.layers {
            assert!(layer.extent.0 <= 160 && layer.extent.1 <= 160);
            assert_eq!(layer.extent.0 % 32, 0);
        }
        assert!(
            plan_path_prepass(&scenes[1].1, size).is_none(),
            "five stacked batches fall back"
        );
    }

    #[test]
    fn opaque_quads_match_blended_quads() {
        let size = size(DevicePixels(96), DevicePixels(96));
        let mut scene = Scene::default();
        // Opaque background, then quads that must keep blending: clipped by
        // their content mask, faded mask edges, rounded, bordered, translucent.
        scene.insert_primitive(quad(rect(0.0, 0.0, 96.0, 96.0), 0.1, 1.0));
        let mut clipped = quad(rect(10.0, 10.0, 40.0, 40.0), 0.3, 1.0);
        clipped.content_mask.bounds = rect(20.0, 20.0, 40.0, 10.0);
        scene.insert_primitive(clipped);
        let mut faded = quad(rect(50.0, 10.0, 30.0, 30.0), 0.4, 1.0);
        faded.content_mask.fade_out = Edges {
            top: ScaledPixels(0.0),
            right: ScaledPixels(8.0),
            bottom: ScaledPixels(0.0),
            left: ScaledPixels(0.0),
        };
        scene.insert_primitive(faded);
        let mut rounded = quad(rect(10.0, 55.0, 30.0, 30.0), 0.5, 1.0);
        rounded.corner_radii = Corners::all(ScaledPixels(9.0));
        scene.insert_primitive(rounded);
        let mut bordered = quad(rect(50.0, 50.0, 30.0, 30.0), 0.6, 1.0);
        bordered.border_widths = Edges::all(ScaledPixels(2.0));
        bordered.border_color = solid_background(hsla(0.0, 0.0, 0.1, 1.0));
        scene.insert_primitive(bordered);
        scene.insert_primitive(quad(rect(30.0, 30.0, 40.0, 40.0), 0.8, 0.5));
        // Opaque quads stacked over the translucent one and each other.
        scene.insert_primitive(quad(rect(35.0, 35.0, 10.0, 10.0), 0.9, 1.0));
        scene.insert_primitive(quad(rect(40.5, 40.25, 10.0, 10.0), 0.2, 1.0));
        scene.finish();

        let blended = experiment_renderer(Experiments::default())
            .render_scene_to_image(&scene, size)
            .unwrap();
        let opaque = experiment_renderer(Experiments {
            opaque_quads: true,
            ..Default::default()
        })
        .render_scene_to_image(&scene, size)
        .unwrap();
        assert_same_pixels(&blended, &opaque, "opaque quads");

        let mut both = experiment_renderer(Experiments {
            opaque_quads: true,
            path_prepass: true,
            ..Default::default()
        });
        let path_size = gpui::size(DevicePixels(160), DevicePixels(160));
        for (what, scene) in path_layer_scenes() {
            let legacy = experiment_renderer(Experiments::default())
                .render_scene_to_image(&scene, path_size)
                .unwrap();
            let image = both.render_scene_to_image(&scene, path_size).unwrap();
            assert_same_pixels(&legacy, &image, what);
        }
    }
}
