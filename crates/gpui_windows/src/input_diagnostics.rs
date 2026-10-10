//! Opt-in input correlation for native performance captures.
use gpui::NativeInputDiagnostic;
use std::{cell::Cell, ffi::c_void, sync::OnceLock};

#[repr(C)]
pub(crate) struct ScreenPoint {
    pub x: i32,
    pub y: i32,
}

#[link(name = "user32")]
unsafe extern "system" {
    fn GetMessageExtraInfo() -> isize;
    fn ClientToScreen(hwnd: *mut c_void, point: *mut ScreenPoint) -> i32;
}
#[link(name = "kernel32")]
unsafe extern "system" {
    fn QueryPerformanceCounter(counter: *mut i64) -> i32;
    fn QueryPerformanceFrequency(frequency: *mut i64) -> i32;
}

pub(crate) fn tagged_input_sequence(extra: u64) -> Option<u64> {
    // SendInput accepts ULONG_PTR, but the mouse input path on Windows can
    // truncate dwExtraInfo to DWORD before WM_MOUSEMOVE. Keep the complete tag
    // in those preserved bits; reject high bits rather than silently aliasing
    // a different protocol or a sequence above this capture's bounded range.
    let sequence = extra & 0xffff;
    (extra >> 16 == 0x4743 && sequence != 0).then_some(sequence)
}

pub(crate) fn current_native_input() -> Option<NativeInputDiagnostic> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| {
        cfg!(target_pointer_width = "64")
            && std::env::var("GITCOMET_UI_INPUT_ROUTE").as_deref() == Ok("native")
    }) {
        return None;
    }
    let sequence = tagged_input_sequence(unsafe { GetMessageExtraInfo() } as u64)?;
    static FREQUENCY: OnceLock<u64> = OnceLock::new();
    let qpc_frequency = *FREQUENCY.get_or_init(|| {
        let mut frequency = 0;
        if unsafe { QueryPerformanceFrequency(&mut frequency) } != 0 {
            frequency.try_into().unwrap_or(0)
        } else {
            0
        }
    });
    let mut arrival_qpc = 0;
    if qpc_frequency == 0 || unsafe { QueryPerformanceCounter(&mut arrival_qpc) } == 0 {
        return None;
    }
    Some(NativeInputDiagnostic {
        sequence,
        arrival_qpc: arrival_qpc.try_into().ok()?,
        qpc_frequency,
    })
}

pub(crate) fn screen_point(hwnd: *mut c_void, mut point: ScreenPoint) -> Option<(i32, i32)> {
    (unsafe { ClientToScreen(hwnd, &mut point) } != 0).then_some((point.x, point.y))
}

pub(crate) struct NativeInputScope<'a> {
    cell: &'a Cell<Option<NativeInputDiagnostic>>,
    previous: Option<NativeInputDiagnostic>,
}
impl<'a> NativeInputScope<'a> {
    pub(crate) fn enter(
        cell: &'a Cell<Option<NativeInputDiagnostic>>,
        input: Option<NativeInputDiagnostic>,
    ) -> Self {
        Self {
            cell,
            previous: cell.replace(input),
        }
    }
}
impl Drop for NativeInputScope<'_> {
    fn drop(&mut self) {
        self.cell.set(self.previous);
    }
}

pub(crate) fn input_rate_keepalive(value: Option<&str>, display_fixed: bool) -> bool {
    match value {
        Some("off") => false,
        Some("on") => true,
        _ => !display_fixed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_input_tag_does_not_accept_ordinary_or_zero_extra_info() {
        assert_eq!(tagged_input_sequence(0), None);
        assert_eq!(tagged_input_sequence(123), None);
        assert_eq!(tagged_input_sequence(0x4743_0000), None);
        assert_eq!(tagged_input_sequence(0x4743_1234), Some(0x1234));
        assert_eq!(tagged_input_sequence(0x4743_ffff), Some(0xffff));
        assert_eq!(tagged_input_sequence(0x4743_0000_0000_1234), None);
        assert_eq!(tagged_input_sequence(0x0001_4743_1234), None);
        assert_eq!(tagged_input_sequence(0x4744_1234), None);
    }
    #[test]
    fn keepalive_override_preserves_auto_and_can_disable_unknown_refresh() {
        assert!(input_rate_keepalive(None, false));
        assert!(!input_rate_keepalive(Some("auto"), true));
        assert!(!input_rate_keepalive(Some("off"), false));
        assert!(input_rate_keepalive(Some("on"), true));
        assert!(input_rate_keepalive(Some("invalid"), false));
    }
    #[test]
    fn native_input_scope_restores_nested_dispatch_and_clears_afterwards() {
        let cell = Cell::new(None);
        let value = NativeInputDiagnostic {
            sequence: 1,
            arrival_qpc: 2,
            qpc_frequency: 3,
        };
        {
            let _outer = NativeInputScope::enter(&cell, Some(value));
            assert_eq!(cell.get(), Some(value));
            {
                let _inner = NativeInputScope::enter(&cell, None);
                assert_eq!(cell.get(), None);
            }
            assert_eq!(cell.get(), Some(value));
        }
        assert_eq!(cell.get(), None);
    }
}
