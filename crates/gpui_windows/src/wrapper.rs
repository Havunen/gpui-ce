use std::ops::Deref;

use crate::bindings::Windows::Win32::{HCURSOR, HWND};

#[derive(Debug, Clone, Copy)]
pub(crate) struct SafeCursor {
    raw: HCURSOR,
}

unsafe impl Send for SafeCursor {}
unsafe impl Sync for SafeCursor {}

impl From<HCURSOR> for SafeCursor {
    fn from(value: HCURSOR) -> Self {
        SafeCursor { raw: value }
    }
}

impl Deref for SafeCursor {
    type Target = HCURSOR;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SafeHwnd {
    raw: HWND,
}

impl SafeHwnd {
    pub(crate) fn as_raw(&self) -> HWND {
        self.raw
    }
}

unsafe impl Send for SafeHwnd {}
unsafe impl Sync for SafeHwnd {}

impl From<HWND> for SafeHwnd {
    fn from(value: HWND) -> Self {
        SafeHwnd { raw: value }
    }
}

impl Deref for SafeHwnd {
    type Target = HWND;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

#[cfg(feature = "windows-wgpu")]
impl raw_window_handle::HasWindowHandle for SafeHwnd {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        let hwnd = std::num::NonZeroIsize::new(self.raw.0 as isize)
            .ok_or(raw_window_handle::HandleError::Unavailable)?;
        let raw = raw_window_handle::Win32WindowHandle::new(hwnd).into();
        // The renderer is detached in WM_DESTROY before this HWND goes away.
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(raw) })
    }
}

#[cfg(feature = "windows-wgpu")]
impl raw_window_handle::HasDisplayHandle for SafeHwnd {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Ok(raw_window_handle::DisplayHandle::windows())
    }
}
