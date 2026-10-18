//! Shared typed rendering and shader contracts.

pub mod artifacts;
pub mod blur;
pub mod damage;
mod instances;
pub mod path_types;
pub mod retained;
pub mod shaders;
mod target_stack;

pub use instances::InstanceRange;
pub use target_stack::TargetStack;
