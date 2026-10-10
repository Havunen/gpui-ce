//! D3D11 native queries, polled with DONOTFLUSH without waiting on the UI thread.
use crate::bindings::Windows::Win32::{ID3D11Asynchronous, ID3D11Device, ID3D11DeviceContext};
use gpui::profiler::gpu::{self, GpuEvent, GpuPassTiming, RendererProfileCapabilities};
use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    ffi::c_void,
    time::{Duration, Instant},
};
use windows_core::Interface;
const MAX_PENDING: usize = 8;
const MAX_PASSES: usize = 256;
#[repr(C)]
struct QueryDesc {
    query: i32,
    misc_flags: u32,
}
#[repr(C)]
#[derive(Default)]
struct Disjoint {
    frequency: u64,
    disjoint: i32,
}
// Generated bindings omit these methods. The stable COM ABI uses device slot 24
// for CreateQuery and immediate-context slots 27/28 for Begin/End. ID3D11Query
// derives from ID3D11Asynchronous, so its interface pointer is a valid upcast.
pub(super) fn query(device: &ID3D11Device, kind: i32) -> anyhow::Result<ID3D11Asynchronous> {
    type Create = unsafe extern "system" fn(
        *mut c_void,
        *const QueryDesc,
        *mut *mut c_void,
    ) -> windows_core::HRESULT;
    let mut output = std::ptr::null_mut();
    unsafe {
        let table = Interface::vtable(device) as *const _ as *const *const c_void;
        let create: Create = std::mem::transmute(*table.add(24));
        create(
            device.as_raw(),
            &QueryDesc {
                query: kind,
                misc_flags: 0,
            },
            &mut output,
        )
        .ok()?;
        anyhow::ensure!(!output.is_null(), "CreateQuery returned a null interface");
        Ok(ID3D11Asynchronous::from_raw(output))
    }
}
fn command(context: &ID3D11DeviceContext, slot: usize, query: &ID3D11Asynchronous) {
    unsafe {
        type Call = unsafe extern "system" fn(*mut c_void, *mut c_void);
        let table = Interface::vtable(context) as *const _ as *const *const c_void;
        let call: Call = std::mem::transmute(*table.add(slot));
        call(context.as_raw(), query.as_raw());
    }
}
pub(super) fn end_query(context: &ID3D11DeviceContext, query: &ID3D11Asynchronous) {
    command(context, 28, query);
}
pub(super) fn ready<T: Default>(
    context: &ID3D11DeviceContext,
    query: &ID3D11Asynchronous,
) -> anyhow::Result<Option<T>> {
    let mut data = T::default();
    let result = unsafe {
        context.GetData(
            query,
            Some(&mut data as *mut _ as *mut c_void),
            std::mem::size_of::<T>() as u32,
            1,
        )
    };
    result.ok()?;
    Ok((result.0 == 0).then_some(data))
}
pub(super) struct Pass {
    name: String,
    start: ID3D11Asynchronous,
    end: ID3D11Asynchronous,
}
struct Frame {
    event: GpuEvent,
    disjoint: ID3D11Asynchronous,
    start: ID3D11Asynchronous,
    end: ID3D11Asynchronous,
    passes: Vec<Pass>,
    overflow: bool,
    submitted: Instant,
}
pub(super) struct GpuTimings {
    pub diagnostic_ablations: gpu::DiagnosticAblations,
    enabled: bool,
    supported: Cell<bool>,
    reason: RefCell<Option<String>>,
    renderer_id: u64,
    current: RefCell<Option<Frame>>,
    pending: RefCell<VecDeque<Frame>>,
}
impl GpuTimings {
    pub fn new() -> Self {
        Self {
            diagnostic_ablations: gpu::DiagnosticAblations::from_environment(),
            enabled: gpu::timings_enabled(),
            supported: Cell::new(true),
            reason: RefCell::new(None),
            renderer_id: gpu::next_renderer_id(),
            current: RefCell::new(None),
            pending: RefCell::new(VecDeque::new()),
        }
    }
    pub fn capabilities(&self) -> RendererProfileCapabilities {
        RendererProfileCapabilities {
            backend: "d3d11".into(),
            implemented_experiments: vec!["partial-redraw".into()],
            gpu_timestamps: self.supported.get(),
            per_pass_timestamps: self.supported.get(),
            timing_enabled: self.enabled,
            unavailable_reason: self.reason.borrow().clone(),
            implemented_diagnostic_ablations: gpu::DiagnosticAblations::implemented_names(),
            active_diagnostic_ablations: self.diagnostic_ablations.active_names(),
            ..Default::default()
        }
    }
    fn new_event(&self) -> GpuEvent {
        let mut event = GpuEvent::new("d3d11", self.renderer_id);
        event.diagnostic_ablations = self.diagnostic_ablations.active_names();
        event
    }
    pub fn begin(&self, device: &ID3D11Device, context: &ID3D11DeviceContext) {
        if !self.enabled {
            return;
        }
        self.poll(context);
        let event = self.new_event();
        if !self.supported.get() {
            gpu::record_gpu_event(event.unavailable(
                "unsupported",
                self.reason.borrow().clone().unwrap_or_default(),
            ));
            return;
        }
        if self.pending.borrow().len() >= MAX_PENDING {
            gpu::record_gpu_event(
                event.unavailable("ring-full", "Eight D3D11 query captures are pending"),
            );
            return;
        }
        match (|| Ok::<_, anyhow::Error>((query(device, 3)?, query(device, 2)?, query(device, 2)?)))(
        ) {
            Ok((disjoint, start, end)) => {
                command(context, 27, &disjoint);
                end_query(context, &start);
                *self.current.borrow_mut() = Some(Frame {
                    event,
                    disjoint,
                    start,
                    end,
                    passes: Vec::new(),
                    overflow: false,
                    submitted: Instant::now(),
                });
            }
            Err(error) => {
                let reason = format!("Timestamp query creation failed: {error:#}");
                self.supported.set(false);
                *self.reason.borrow_mut() = Some(reason.clone());
                gpu::record_gpu_event(event.unavailable("query-failed", reason));
            }
        }
    }
    pub fn begin_pass(
        &self,
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        name: &str,
    ) -> Option<Pass> {
        let mut current = self.current.borrow_mut();
        let frame = current.as_mut()?;
        if frame.passes.len() >= MAX_PASSES {
            frame.overflow = true;
            return None;
        }
        match (query(device, 2), query(device, 2)) {
            (Ok(start), Ok(end)) => {
                end_query(context, &start);
                Some(Pass {
                    name: name.into(),
                    start,
                    end,
                })
            }
            _ => {
                frame.overflow = true;
                None
            }
        }
    }
    pub fn end_pass(&self, context: &ID3D11DeviceContext, pass: Option<Pass>) {
        if let Some(pass) = pass {
            end_query(context, &pass.end);
            if let Some(frame) = self.current.borrow_mut().as_mut() {
                frame.passes.push(pass);
            }
        }
    }
    pub fn describe_damage(&self, kind: &str, bytes: u64, experiments: Vec<String>) {
        if let Some(frame) = self.current.borrow_mut().as_mut() {
            frame.event.damage_kind = kind.into();
            frame.event.retained_bytes = bytes;
            frame.event.experiments = experiments;
        }
    }
    pub fn end(&self, context: &ID3D11DeviceContext, success: bool) {
        if let Some(frame) = self.current.borrow_mut().take() {
            end_query(context, &frame.end);
            end_query(context, &frame.disjoint);
            if success {
                self.pending.borrow_mut().push_back(frame);
            } else {
                gpu::record_gpu_event(
                    frame
                        .event
                        .unavailable("incomplete", "Frame failed before successful submission"),
                );
            }
        }
    }
    pub fn poll(&self, context: &ID3D11DeviceContext) {
        if !self.enabled {
            return;
        }
        let mut pending = self.pending.borrow_mut();
        while let Some(frame) = pending.front() {
            let result = (|| {
                let Some(disjoint) = ready::<Disjoint>(context, &frame.disjoint)? else {
                    return Ok::<_, anyhow::Error>(None);
                };
                if disjoint.disjoint != 0 || disjoint.frequency == 0 {
                    anyhow::bail!("D3D11 clock disjoint or zero frequency");
                }
                let Some(start) = ready::<u64>(context, &frame.start)? else {
                    return Ok(None);
                };
                let Some(end) = ready::<u64>(context, &frame.end)? else {
                    return Ok(None);
                };
                let ns = |delta: u64| {
                    ((delta as u128 * 1_000_000_000) / disjoint.frequency as u128)
                        .min(u64::MAX as u128) as u64
                };
                if end < start || frame.overflow {
                    anyhow::bail!("Incomplete, overflowing or nonmonotonic timestamp queries");
                }
                let mut passes = Vec::with_capacity(frame.passes.len());
                for pass in &frame.passes {
                    let Some(first) = ready::<u64>(context, &pass.start)? else {
                        return Ok(None);
                    };
                    let Some(last) = ready::<u64>(context, &pass.end)? else {
                        return Ok(None);
                    };
                    if last < first {
                        anyhow::bail!("Nonmonotonic pass timestamp");
                    }
                    passes.push(GpuPassTiming {
                        name: pass.name.clone(),
                        elapsed_ns: ns(last - first),
                    });
                }
                Ok(Some((ns(end - start), passes)))
            })();
            if matches!(result, Ok(None)) && frame.submitted.elapsed() < Duration::from_secs(2) {
                break;
            }
            let mut frame = pending.pop_front().unwrap();
            match result {
                Ok(Some((elapsed, passes))) => {
                    frame.event.status = "complete".into();
                    frame.event.elapsed_ns = Some(elapsed);
                    frame.event.passes = passes;
                }
                Ok(None) => {
                    frame.event = frame
                        .event
                        .unavailable("incomplete", "Queries unresolved after two seconds")
                }
                Err(error) => {
                    frame.event = frame
                        .event
                        .unavailable("query-failed", format!("{error:#}"))
                }
            }
            gpu::record_gpu_event(frame.event);
        }
    }
    pub fn discard(&self, reason: &str) {
        if let Some(frame) = self.current.borrow_mut().take() {
            gpu::record_gpu_event(frame.event.unavailable("incomplete", reason));
        }
        for frame in self.pending.borrow_mut().drain(..) {
            gpu::record_gpu_event(frame.event.unavailable("incomplete", reason));
        }
    }
    pub fn fail_latest_submission(&self, reason: &str) {
        if !self.enabled {
            return;
        }
        let context = gpu::gpu_frame_context();
        let mut pending = self.pending.borrow_mut();
        if pending.back().is_some_and(|frame| {
            frame.event.submission_id == context.submission_id
                && frame.event.window == context.window
        }) {
            let frame = pending.pop_back().unwrap();
            gpu::record_gpu_event(frame.event.unavailable("incomplete", reason));
        }
    }
    pub fn skipped(&self, reason: &str) {
        if self.enabled {
            gpu::record_gpu_event(self.new_event().unavailable("incomplete", reason));
        }
    }
}
impl Drop for GpuTimings {
    fn drop(&mut self) {
        self.discard("Renderer dropped or recovered before query readback");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_timestamp_queries_resolve_without_waiting_in_poll() -> anyhow::Result<()> {
        let devices = crate::directx_devices::DirectXDevices::new()?;
        let mut timings = GpuTimings::new();
        timings.enabled = true;
        let _scope =
            gpu::set_gpu_frame_context(Some("gpu-query-test".into()), true, u64::MAX - 1, 7);
        timings.begin(&devices.device, &devices.device_context);
        let pass = timings.begin_pass(
            &devices.device,
            &devices.device_context,
            "empty_query_validation",
        );
        timings.end_pass(&devices.device_context, pass);
        timings.end(&devices.device_context, true);
        unsafe {
            devices.device_context.Flush();
        }
        let start = Instant::now();
        while !timings.pending.borrow().is_empty() && start.elapsed() < Duration::from_secs(2) {
            timings.poll(&devices.device_context);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            timings.pending.borrow().is_empty(),
            "GPU queries failed to resolve"
        );
        let (events, _) = gpu::take_gpu_events();
        let event = events
            .into_iter()
            .find(|event| event.renderer_id == timings.renderer_id)
            .expect("query result must be emitted");
        assert_eq!(
            event.status, "complete",
            "query capture failed: {:?}",
            event.reason
        );
        assert_eq!(event.submission_id, u64::MAX - 1);
        assert!(event.elapsed_ns.is_some());
        assert_eq!(event.passes.len(), 1);
        Ok(())
    }
}
