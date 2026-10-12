//! Frame context and the partial-redraw opt-out for the retained renderers.
use std::cell::Cell;

/// Window state passed to the platform renderer on this thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuFrameContext {
    /// Whether the submitting window has focus or the pointer.
    pub interactive: bool,
    /// Current painted scene revision.
    pub scene_revision: u64,
}
thread_local! {
    static CONTEXT: Cell<GpuFrameContext> = const {
        Cell::new(GpuFrameContext { interactive: false, scene_revision: 0 })
    };
}

/// Restores the previous context, including when drawing unwinds.
pub struct GpuFrameContextGuard(GpuFrameContext);
impl Drop for GpuFrameContextGuard {
    fn drop(&mut self) {
        CONTEXT.with(|context| context.set(self.0));
    }
}
/// Install a scoped renderer context, restoring the previous value when the guard drops.
pub fn set_gpu_frame_context(interactive: bool, scene_revision: u64) -> GpuFrameContextGuard {
    CONTEXT.with(|context| {
        GpuFrameContextGuard(context.replace(GpuFrameContext {
            interactive,
            scene_revision,
        }))
    })
}
/// Return the current thread's renderer context.
pub fn gpu_frame_context() -> GpuFrameContext {
    CONTEXT.with(Cell::get)
}
/// Partial redraw is on unless `GPUI_GPU_EXPERIMENTS` lists `no-partial-redraw`.
pub fn partial_redraw_enabled() -> bool {
    !std::env::var("GPUI_GPU_EXPERIMENTS").is_ok_and(|value| lists_no_partial_redraw(&value))
}

fn lists_no_partial_redraw(experiments: &str) -> bool {
    experiments
        .split(',')
        .any(|flag| flag.trim() == "no-partial-redraw")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scoped_context_restores_even_nested_calls() {
        let _outer = set_gpu_frame_context(true, 4);
        {
            let _inner = set_gpu_frame_context(false, 9);
            assert_eq!(gpu_frame_context().scene_revision, 9);
        }
        assert_eq!(gpu_frame_context().scene_revision, 4);
        assert!(gpu_frame_context().interactive);
    }

    #[test]
    fn only_a_listed_no_partial_redraw_opts_out() {
        assert!(lists_no_partial_redraw("no-partial-redraw"));
        assert!(lists_no_partial_redraw("other, no-partial-redraw "));
        assert!(!lists_no_partial_redraw(""));
        assert!(!lists_no_partial_redraw("none"));
        assert!(!lists_no_partial_redraw("partial-redraw"));
        assert!(!lists_no_partial_redraw("no-partial-redraw-later"));
    }
}
