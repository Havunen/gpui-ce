#![cfg(any(target_os = "linux", target_os = "freebsd"))]
mod linux;

pub(crate) use gpui::collections;
pub(crate) use gpui_util as util;

pub use linux::current_platform;

/// Access the X11 file clipboard without opening a Wayland data source (WSLg).
#[cfg(feature = "x11")]
fn file_clipboard_bridge() -> Result<&'static linux::x11::clipboard::Clipboard, String> {
    static BRIDGE: std::sync::OnceLock<Result<linux::x11::clipboard::Clipboard, String>> =
        std::sync::OnceLock::new();
    BRIDGE
        .get_or_init(|| linux::x11::clipboard::Clipboard::new().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(Clone::clone)
}

/// Write native file formats through X11, including desktop cut metadata.
#[cfg(feature = "x11")]
pub fn write_files_to_x11_clipboard(files: &gpui::FileTransfer) -> Result<(), String> {
    use linux::x11::clipboard::{ClipboardKind, WaitConfig};
    file_clipboard_bridge()?
        .set_files(files, ClipboardKind::Clipboard, WaitConfig::None)
        .map_err(|e| e.to_string())
}

/// Read typed files from the X11 clipboard bridge. Text paths remain text.
#[cfg(feature = "x11")]
pub fn read_files_from_x11_clipboard() -> Option<gpui::FileTransfer> {
    file_clipboard_bridge()
        .ok()?
        .get_any(linux::x11::clipboard::ClipboardKind::Clipboard)
        .ok()?
        .file_transfer()
}

#[cfg(not(feature = "x11"))]
pub fn write_files_to_x11_clipboard(_: &gpui::FileTransfer) -> Result<(), String> {
    Err("X11 support is disabled".into())
}
#[cfg(not(feature = "x11"))]
pub fn read_files_from_x11_clipboard() -> Option<gpui::FileTransfer> {
    None
}
