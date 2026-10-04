//! Shared typed rendering and shader contracts.

pub mod artifacts;
pub mod blur;
pub mod damage;
pub mod gpu_policy;
mod instances;
#[cfg(any(test, feature = "test-support"))]
pub mod optimization_fixture;
pub mod path_cache;
pub mod path_plan;
pub mod path_types;
pub mod scratch_pool;
pub mod shaders;
pub mod sharing;

pub use instances::InstanceRange;
