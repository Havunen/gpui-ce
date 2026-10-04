//! Shared typed rendering and shader contracts.

pub mod artifacts;
pub mod blur;
pub mod damage;
pub mod gpu_policy;
mod instances;
pub mod path_plan;
pub mod path_types;
pub mod shaders;

pub use instances::InstanceRange;
