#![cfg(any(target_os = "linux", target_os = "freebsd"))]
mod linux;

pub(crate) use gpui::collections;
pub(crate) use util;

pub use linux::current_platform;
