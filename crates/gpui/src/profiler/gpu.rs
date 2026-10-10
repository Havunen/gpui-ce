//! Opt-in, asynchronous renderer telemetry. Missing timings never mean zero work.
use scheduler::Instant;
use serde::Serialize;
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

/// Rendering features implemented by the currently selected backend.
#[derive(Clone, Debug, Default, Serialize)]
pub struct RendererProfileCapabilities {
    /// Selected rendering backend.
    pub backend: String,
    /// Experiment flags implemented by this backend, regardless of whether enabled.
    pub implemented_experiments: Vec<String>,
    /// Experiment flags resolved and enabled in this renderer, including when queries are off.
    /// Scene eligibility and actual partial rendering remain recorded per frame.
    pub active_experiments: Vec<String>,
    /// Whether native whole-frame timestamp queries are supported.
    pub gpu_timestamps: bool,
    /// Whether native per-pass timestamp queries are supported.
    pub per_pass_timestamps: bool,
    /// Whether this run requested GPU timings.
    pub timing_enabled: bool,
    /// Why requested timing samples are unavailable.
    pub unavailable_reason: Option<String>,
    /// Effective Windows input-rate keepalive setting, absent on other platforms.
    pub windows_input_rate_keepalive: Option<bool>,
    /// Source of the effective Windows keepalive setting.
    pub windows_keepalive_policy: Option<String>,
    /// Actual swapchain present mode selected by the backend.
    pub present_mode: Option<String>,
    /// Actual configured swapchain alpha mode; absent when unavailable.
    pub surface_alpha_mode: Option<String>,
    /// Visually incomplete diagnostic controls implemented by the backend.
    pub implemented_diagnostic_ablations: Vec<String>,
    /// Diagnostic controls active in this renderer.
    pub active_diagnostic_ablations: Vec<String>,
}
impl RendererProfileCapabilities {
    /// Report an unsupported renderer without implying zero GPU work.
    pub fn unsupported(backend: impl Into<String>) -> Self {
        Self {
            backend: backend.into(),
            unavailable_reason: Some("Renderer does not expose GPU timing queries".into()),
            ..Default::default()
        }
    }
}

/// GPU duration of one render pass, measured by native queries.
#[derive(Clone, Debug, Serialize)]
pub struct GpuPassTiming {
    /// Render pass name.
    pub name: String,
    /// Native-query duration converted to nanoseconds.
    pub elapsed_ns: u64,
}

/// One submitted frame, including explicitly unavailable or discarded samples.
#[derive(Clone, Debug, Serialize)]
pub struct GpuEvent {
    /// Stable JSONL event name.
    pub event: &'static str,
    /// Monotonic submission time used to derive the consumer's relative timestamp.
    #[serde(skip)]
    pub submitted_at: Instant,
    /// Wall-clock submission time, for external-system correlation only.
    pub submitted_unix_ms: f64,
    /// GPUI window identity when a window context was installed.
    pub window: Option<String>,
    /// Process-unique renderer identity.
    pub renderer_id: u64,
    /// Window-scoped renderer submission identity.
    pub submission_id: u64,
    /// Painted scene revision submitted by this event.
    pub scene_revision: u64,
    /// Selected rendering backend.
    pub backend: String,
    /// Explicit sample outcome such as complete, incomplete, or unavailable.
    pub status: String,
    /// Explanation for failed or unavailable samples.
    pub reason: Option<String>,
    /// Whole-frame GPU query duration, absent when unavailable.
    pub elapsed_ns: Option<u64>,
    /// Completed individual pass durations.
    pub passes: Vec<GpuPassTiming>,
    /// Experiment flags active for this submission.
    pub experiments: Vec<String>,
    /// Visually incomplete diagnostic controls active for this submission.
    pub diagnostic_ablations: Vec<String>,
    /// Retained-scene result: full, partial, or reuse.
    pub damage_kind: String,
    /// Why retained drawing fell back to a complete scene, independently of timing status.
    pub damage_fallback_reason: Option<String>,
    /// Extra retained GPU allocations held by this renderer.
    pub retained_bytes: u64,
    /// Actual quad API draw calls, including damage-clear quads, for this submission.
    pub quad_draw_calls: u64,
    /// Additional ordered quad draw calls introduced by hardware scissor runs.
    pub quad_scissor_splits: u64,
}

/// Window identity and scene revision passed to the platform renderer on this thread.
#[derive(Clone, Debug, Default)]
pub struct GpuFrameContext {
    /// GPUI window identity.
    pub window: Option<String>,
    /// Whether the submitting window is active.
    pub active: bool,
    /// Window-scoped submission identity.
    pub submission_id: u64,
    /// Current painted scene revision.
    pub scene_revision: u64,
}
thread_local! { static CONTEXT: RefCell<GpuFrameContext> = RefCell::new(GpuFrameContext::default()); }

/// Restores the previous context, including when drawing unwinds.
pub struct GpuFrameContextGuard(GpuFrameContext);
impl Drop for GpuFrameContextGuard {
    fn drop(&mut self) {
        CONTEXT.with(|context| *context.borrow_mut() = self.0.clone());
    }
}
/// Install a scoped renderer context, restoring the previous value when the guard drops.
pub fn set_gpu_frame_context(
    window: Option<String>,
    active: bool,
    submission_id: u64,
    scene_revision: u64,
) -> GpuFrameContextGuard {
    CONTEXT.with(|context| {
        GpuFrameContextGuard(context.replace(GpuFrameContext {
            window,
            active,
            submission_id,
            scene_revision,
        }))
    })
}
/// Return the current thread's renderer context.
pub fn gpu_frame_context() -> GpuFrameContext {
    CONTEXT.with(|context| context.borrow().clone())
}
/// Whether the run needs window and submission context in its renderers.
pub fn frame_context_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| timings_enabled() || experiment_enabled("partial-redraw"))
}
/// Whether native GPU timings were explicitly requested by the environment.
pub fn timings_enabled() -> bool {
    std::env::var("GPUI_GPU_TIMINGS").is_ok_and(|value| matches!(value.trim(), "1" | "true" | "on"))
}
/// Whether an experiment flag was explicitly requested by the environment.
pub fn experiment_enabled(name: &str) -> bool {
    std::env::var("GPUI_GPU_EXPERIMENTS")
        .is_ok_and(|value| value.split(',').any(|flag| flag.trim() == name))
}

/// Cached, explicitly requested controls that omit rendering to isolate GPU cost.
/// These change the pixels and must never be treated as optimization candidates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiagnosticAblations(u8);

impl DiagnosticAblations {
    const NAMES: [&'static str; 7] = [
        "no-mono-glyphs",
        "no-subpixel-glyphs",
        "no-shadows",
        "no-paths",
        "no-quads",
        "no-images",
        "clear-only",
    ];

    /// Read controls once when constructing a renderer. Unknown flags fail closed.
    pub fn from_environment() -> Self {
        match std::env::var("GPUI_GPU_DIAGNOSTIC_ABLATIONS") {
            Ok(value) => Self::parse(&value).unwrap_or_else(|reason| {
                log::warn!("GPU diagnostic ablations disabled: {reason}");
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    /// Parse canonical comma-separated diagnostic flags without reading the environment.
    pub fn parse(value: &str) -> Result<Self, String> {
        let mut flags = 0;
        for token in value
            .split(',')
            .map(str::trim)
            .filter(|token| !token.is_empty())
        {
            let Some(index) = Self::NAMES.iter().position(|name| *name == token) else {
                return Err(format!("unsupported flag {token:?}"));
            };
            flags |= 1 << index;
        }
        Ok(Self(flags))
    }

    /// Canonical controls supported by both native Windows renderers.
    pub fn implemented_names() -> Vec<String> {
        Self::NAMES.iter().map(|name| (*name).into()).collect()
    }

    /// Canonical controls actually active, in stable order without duplicates.
    pub fn active_names(self) -> Vec<String> {
        Self::NAMES
            .iter()
            .enumerate()
            .filter(|(index, _)| self.0 & (1 << index) != 0)
            .map(|(_, name)| (*name).into())
            .collect()
    }

    /// Whether any visually incomplete control is active.
    pub fn enabled(self) -> bool {
        self.0 != 0
    }

    /// Whether drawing should retain only the normal target clear and presentation.
    pub fn clear_only(self) -> bool {
        self.0 & (1 << 6) != 0
    }

    /// Whether a compiled painter command is omitted by the active diagnostic controls.
    pub fn skips_command(self, command: &crate::RenderCommand) -> bool {
        use crate::{PrimitiveBatch, RenderCommand};
        if self.clear_only() {
            return true;
        }
        let bit = match command {
            RenderCommand::Batch(PrimitiveBatch::MonochromeSprites { .. }) => 0,
            RenderCommand::Batch(PrimitiveBatch::SubpixelSprites { .. }) => 1,
            RenderCommand::Batch(PrimitiveBatch::Shadows { .. }) => 2,
            RenderCommand::Batch(PrimitiveBatch::Paths { .. }) => 3,
            RenderCommand::Batch(PrimitiveBatch::Quads { .. }) => 4,
            RenderCommand::Batch(
                PrimitiveBatch::PolychromeSprites { .. } | PrimitiveBatch::Surfaces(_),
            ) => 5,
            _ => return false,
        };
        self.0 & (1 << bit) != 0
    }
}
/// Allocate a process-unique renderer identity.
pub fn next_renderer_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

impl GpuEvent {
    /// Create an incomplete event with the current window/submission context.
    pub fn new(backend: &str, renderer_id: u64) -> Self {
        let context = gpu_frame_context();
        Self {
            event: "gpu_frame",
            submitted_at: Instant::now(),
            submitted_unix_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0.0, |duration| duration.as_secs_f64() * 1000.0),
            window: context.window,
            renderer_id,
            submission_id: context.submission_id,
            scene_revision: context.scene_revision,
            backend: backend.into(),
            status: "incomplete".into(),
            reason: None,
            elapsed_ns: None,
            passes: Vec::new(),
            experiments: Vec::new(),
            diagnostic_ablations: Vec::new(),
            damage_kind: "full".into(),
            damage_fallback_reason: None,
            retained_bytes: 0,
            quad_draw_calls: 0,
            quad_scissor_splits: 0,
        }
    }
    /// Set an explicit unavailable or failed sample outcome.
    pub fn unavailable(mut self, status: &str, reason: impl Into<String>) -> Self {
        self.status = status.into();
        self.reason = Some(reason.into());
        self.elapsed_ns = None;
        self.passes.clear();
        self
    }
}

const EVENT_CAPACITY: usize = 4096;
static EVENTS: OnceLock<Mutex<VecDeque<GpuEvent>>> = OnceLock::new();
static DROPPED: AtomicU64 = AtomicU64::new(0);
/// Queue a completed or failed GPU sample without exceeding the bounded event queue.
pub fn record_gpu_event(event: GpuEvent) {
    let mut events = EVENTS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if events.len() == EVENT_CAPACITY {
        DROPPED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    events.push_back(event);
}
/// Drain completed/failed records and the cumulative number of queue drops.
pub fn take_gpu_events() -> (Vec<GpuEvent>, u64) {
    let events = EVENTS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .drain(..)
        .collect();
    (events, DROPPED.load(Ordering::Relaxed))
}

/// Optional retained rendering allocations share this limit on each device.
pub const RETAINED_DEVICE_LIMIT: u64 = 32 * 1024 * 1024;
struct DeviceBudget(AtomicU64);
/// A retained-allocation budget reservation, released when dropped.
pub struct GpuMemoryReservation {
    budget: Arc<DeviceBudget>,
    /// Number of bytes reserved against the device budget.
    pub bytes: u64,
}
impl Drop for GpuMemoryReservation {
    fn drop(&mut self) {
        self.budget.0.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
/// Reserve additional retained bytes on one device, failing when its shared budget is exceeded.
pub fn reserve_retained_bytes(
    backend: &str,
    device_key: usize,
    bytes: u64,
) -> Option<GpuMemoryReservation> {
    type Registry = HashMap<(String, usize), Weak<DeviceBudget>>;
    static BUDGETS: OnceLock<Mutex<Registry>> = OnceLock::new();
    let mut budgets = BUDGETS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    budgets.retain(|_, budget| budget.strong_count() != 0);
    let key = (backend.to_owned(), device_key);
    let budget = budgets
        .get(&key)
        .and_then(Weak::upgrade)
        .unwrap_or_else(|| {
            let budget = Arc::new(DeviceBudget(AtomicU64::new(0)));
            budgets.insert(key, Arc::downgrade(&budget));
            budget
        });
    budget
        .0
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
            used.checked_add(bytes)
                .filter(|total| *total <= RETAINED_DEVICE_LIMIT)
        })
        .ok()?;
    Some(GpuMemoryReservation { budget, bytes })
}

/// Read the Windows monotonic performance counter and its ticks-per-second frequency.
/// Returns `None` on other platforms or when the native query fails.
#[cfg(target_os = "windows")]
pub fn diagnostic_qpc() -> Option<(u64, u64)> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn QueryPerformanceCounter(value: *mut i64) -> i32;
        fn QueryPerformanceFrequency(value: *mut i64) -> i32;
    }
    let mut counter = 0i64;
    let mut frequency = 0i64;
    // SAFETY: Both functions receive a valid initialized LARGE_INTEGER-sized out pointer.
    unsafe {
        if QueryPerformanceCounter(&mut counter) == 0
            || QueryPerformanceFrequency(&mut frequency) == 0
        {
            return None;
        }
    }
    (counter >= 0 && frequency > 0).then_some((counter as u64, frequency as u64))
}

/// Read the Windows diagnostic clock, unavailable on this platform.
#[cfg(not(target_os = "windows"))]
pub fn diagnostic_qpc() -> Option<(u64, u64)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostic_flags_are_canonical_explicit_and_fail_closed() {
        assert!(!DiagnosticAblations::parse("").unwrap().enabled());
        let flags = DiagnosticAblations::parse(" no-paths,no-quads,no-paths ").unwrap();
        assert_eq!(flags.active_names(), ["no-paths", "no-quads"]);
        assert!(!flags.clear_only());
        assert!(DiagnosticAblations::parse("no-paths,no-unknown-work").is_err());
        assert!(
            DiagnosticAblations::parse("clear-only")
                .unwrap()
                .clear_only()
        );
    }

    #[test]
    fn diagnostic_controls_omit_only_requested_batches_and_clear_omits_filters() {
        use crate::{
            AtlasTextureId, AtlasTextureKind, FilterRenderTarget, PrimitiveBatch, RenderCommand,
        };
        let texture = |kind| AtlasTextureId { index: 0, kind };
        let commands = [
            (
                "no-mono-glyphs",
                PrimitiveBatch::MonochromeSprites {
                    texture_id: texture(AtlasTextureKind::Monochrome),
                    range: 0..1,
                },
            ),
            (
                "no-subpixel-glyphs",
                PrimitiveBatch::SubpixelSprites {
                    texture_id: texture(AtlasTextureKind::Subpixel),
                    range: 0..1,
                },
            ),
            (
                "no-shadows",
                PrimitiveBatch::Shadows {
                    range: 0..1,
                    smoothed: false,
                },
            ),
            (
                "no-paths",
                PrimitiveBatch::Paths {
                    range: 0..1,
                    rasterization_vertex_count: 3,
                    sprite_count: 1,
                },
            ),
            (
                "no-quads",
                PrimitiveBatch::Quads {
                    range: 0..1,
                    smoothed: false,
                },
            ),
            (
                "no-images",
                PrimitiveBatch::PolychromeSprites {
                    texture_id: texture(AtlasTextureKind::Polychrome),
                    range: 0..1,
                    smoothed: false,
                },
            ),
            ("no-images", PrimitiveBatch::Surfaces(0..1)),
        ];
        for name in DiagnosticAblations::NAMES
            .into_iter()
            .filter(|name| *name != "clear-only")
        {
            let flags = DiagnosticAblations::parse(name).unwrap();
            for (omitting, batch) in &commands {
                assert_eq!(
                    flags.skips_command(&RenderCommand::Batch(batch.clone())),
                    name == *omitting
                );
            }
            assert!(!flags.skips_command(&RenderCommand::Batch(PrimitiveBatch::Underlines(0..1))));
        }
        let filter = RenderCommand::BeginFilter {
            boundary_index: 0,
            target: FilterRenderTarget::Inline,
        };
        assert!(
            !DiagnosticAblations::parse("no-paths")
                .unwrap()
                .skips_command(&filter)
        );
        let clear = DiagnosticAblations::parse("clear-only").unwrap();
        assert!(clear.skips_command(&filter));
        for (_, batch) in commands {
            assert!(clear.skips_command(&RenderCommand::Batch(batch)));
            assert!(
                !DiagnosticAblations::default()
                    .skips_command(&RenderCommand::Batch(PrimitiveBatch::Underlines(0..1)))
            );
        }
    }

    #[test]
    fn retained_budget_is_device_wide_and_released() {
        let first = reserve_retained_bytes("test", 1, RETAINED_DEVICE_LIMIT - 1).unwrap();
        assert!(reserve_retained_bytes("test", 1, 2).is_none());
        assert!(reserve_retained_bytes("test", 2, 2).is_some());
        drop(first);
        assert!(reserve_retained_bytes("test", 1, RETAINED_DEVICE_LIMIT).is_some());
    }
    #[test]
    fn scoped_context_restores_even_nested_calls() {
        let _outer = set_gpu_frame_context(Some("a".into()), true, 3, 4);
        {
            let _inner = set_gpu_frame_context(Some("b".into()), false, 8, 9);
            assert_eq!(gpu_frame_context().submission_id, 8);
        }
        assert_eq!(gpu_frame_context().submission_id, 3);
    }
}
