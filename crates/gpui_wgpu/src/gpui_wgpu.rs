// Completion callbacks own wgpu textures, and proving those `Send` walks deeper than
// the default limit: https://github.com/rust-lang/rust/issues/159228
#![recursion_limit = "256"]

mod cosmic_text_system;
mod wgpu_atlas;
mod wgpu_context;
mod wgpu_renderer;

pub(crate) use gpui::collections;

pub use cosmic_text_system::*;
pub use wgpu;
pub use wgpu_atlas::*;
pub use wgpu_context::*;
#[cfg(all(feature = "test-support", not(target_family = "wasm")))]
pub use wgpu_renderer::WgpuHeadlessRenderer;
pub use wgpu_renderer::{
    FontRasterizationSettings, GpuContext, SubpixelOrder, WgpuRenderer, WgpuSurfaceConfig,
};
