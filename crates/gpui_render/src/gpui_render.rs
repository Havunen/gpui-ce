//! Shared typed rendering and shader contracts.

pub mod artifacts;
pub mod blur;
pub mod damage;
pub mod gpu_policy;
pub mod native_path_cache;
pub mod native_pool;
#[cfg(any(test, feature = "test-support"))]
pub mod optimization_fixture;
mod instances;
pub mod path_plan;
pub mod path_types;
pub mod shaders;

pub use instances::InstanceRange;
