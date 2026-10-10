//! Timestamp queries are resolved off the foreground thread, in a bounded ring.
use gpui::profiler::gpu::{self, GpuEvent, GpuPassTiming, RendererProfileCapabilities};
use std::{
    cell::RefCell,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

const MAX_PASSES: u32 = 256;
const MAX_PENDING: usize = 8;
pub(super) struct TimestampPass {
    pub query_set: wgpu::QuerySet,
    pub first: u32,
    pub last: u32,
}
impl TimestampPass {
    pub fn writes(&self) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(self.first),
            end_of_pass_write_index: Some(self.last),
        }
    }
}
struct Frame {
    event: GpuEvent,
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    labels: Vec<String>,
    overflow: bool,
}
struct Pending {
    frame: Frame,
    submission: wgpu::SubmissionIndex,
}
pub(super) struct GpuTimings {
    pub renderer_id: u64,
    pub diagnostic_ablations: gpu::DiagnosticAblations,
    pub clipped_quads: bool,
    quad_scissor_supported: bool,
    enabled: bool,
    supported: bool,
    backend: String,
    current: RefCell<Option<Frame>>,
    pending: Arc<AtomicUsize>,
    #[cfg(not(target_family = "wasm"))]
    sender: Option<std::sync::mpsc::SyncSender<Pending>>,
}

impl GpuTimings {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        backend: String,
        quad_scissor_supported: bool,
    ) -> Self {
        let enabled = gpu::timings_enabled();
        let supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
            && !cfg!(target_family = "wasm")
            && queue.get_timestamp_period().is_finite()
            && queue.get_timestamp_period() > 0.0;
        let pending = Arc::new(AtomicUsize::new(0));
        #[cfg(not(target_family = "wasm"))]
        let sender = if enabled && supported {
            let (sender, receiver) = std::sync::mpsc::sync_channel::<Pending>(MAX_PENDING);
            let pending_count = pending.clone();
            let period = queue.get_timestamp_period() as f64;
            std::thread::Builder::new()
                .name("gpui-gpu-timings".into())
                .spawn(move || {
                    while let Ok(mut capture) = receiver.recv() {
                        let (map_sender, map_receiver) = std::sync::mpsc::channel();
                        let bytes = (capture.frame.labels.len() * 16) as u64;
                        capture.frame.readback.slice(..bytes).map_async(
                            wgpu::MapMode::Read,
                            move |result| {
                                let _ = map_sender.send(result);
                            },
                        );
                        let result = device.poll(wgpu::PollType::Wait {
                            submission_index: Some(capture.submission),
                            timeout: Some(Duration::from_secs(2)),
                        });
                        let mapped = map_receiver.try_recv();
                        if result.is_ok() && matches!(mapped, Ok(Ok(()))) {
                            let slice = capture.frame.readback.slice(..bytes);
                            let data = slice.get_mapped_range();
                            let values: Vec<u64> = data
                                .chunks_exact(8)
                                .map(|word| u64::from_ne_bytes(word.try_into().unwrap()))
                                .collect();
                            if values.len() >= 2 && values.windows(2).all(|pair| pair[1] >= pair[0])
                            {
                                capture.frame.event.elapsed_ns = Some(
                                    ((values.last().unwrap() - values[0]) as f64 * period).round()
                                        as u64,
                                );
                                capture.frame.event.passes = capture
                                    .frame
                                    .labels
                                    .iter()
                                    .zip(values.chunks_exact(2))
                                    .map(|(name, pair)| GpuPassTiming {
                                        name: name.clone(),
                                        elapsed_ns: ((pair[1] - pair[0]) as f64 * period).round()
                                            as u64,
                                    })
                                    .collect();
                                capture.frame.event.status = "complete".into();
                            } else {
                                capture.frame.event = capture.frame.event.unavailable(
                                    "query-failed",
                                    "Nonmonotonic or incomplete timestamp values",
                                );
                            }
                            drop(data);
                            capture.frame.readback.unmap();
                        } else {
                            capture.frame.event = capture.frame.event.unavailable(
                                "query-failed",
                                format!(
                                    "Timestamp readback failed: poll={result:?}, map={mapped:?}"
                                ),
                            );
                        }
                        if capture.frame.overflow {
                            capture.frame.event = capture.frame.event.unavailable(
                                "query-failed",
                                "Per-frame timestamp query capacity exceeded",
                            );
                            capture.frame.event.elapsed_ns = None;
                            capture.frame.event.passes.clear();
                        }
                        gpu::record_gpu_event(capture.frame.event);
                        pending_count.fetch_sub(1, Ordering::AcqRel);
                    }
                })
                .ok()
                .map(|_| sender)
        } else {
            None
        };
        Self {
            renderer_id: gpu::next_renderer_id(),
            diagnostic_ablations: gpu::DiagnosticAblations::from_environment(),
            clipped_quads: quad_scissor_supported && gpu::experiment_enabled("clipped-quads"),
            quad_scissor_supported,
            enabled,
            supported,
            backend,
            current: RefCell::new(None),
            pending,
            #[cfg(not(target_family = "wasm"))]
            sender,
        }
    }

    pub fn capabilities(&self) -> RendererProfileCapabilities {
        let mut experiments = vec!["cropped-paths".into()];
        if self.quad_scissor_supported {
            experiments.push("clipped-quads".into());
        }
        if cfg!(target_os = "windows") {
            experiments.push("partial-redraw".into());
        }
        let available = self.supported && (!self.enabled || self.worker_available());
        RendererProfileCapabilities {
            backend: self.backend.clone(),
            implemented_experiments: experiments,
            active_experiments: if self.clipped_quads {
                vec!["clipped-quads".into()]
            } else {
                Vec::new()
            },
            gpu_timestamps: available,
            per_pass_timestamps: available,
            timing_enabled: self.enabled,
            unavailable_reason: (!available).then(|| {
                if self.supported {
                    "Timestamp readback worker could not start".into()
                } else {
                    "Adapter/device does not support native timestamp queries".into()
                }
            }),
            implemented_diagnostic_ablations: gpu::DiagnosticAblations::implemented_names(),
            active_diagnostic_ablations: self.diagnostic_ablations.active_names(),
            ..Default::default()
        }
    }
    fn worker_available(&self) -> bool {
        #[cfg(not(target_family = "wasm"))]
        {
            self.sender.is_some()
        }
        #[cfg(target_family = "wasm")]
        {
            false
        }
    }
    fn new_event(&self) -> GpuEvent {
        let mut event = GpuEvent::new(&self.backend, self.renderer_id);
        event.diagnostic_ablations = self.diagnostic_ablations.active_names();
        if self.clipped_quads {
            event.experiments.push("clipped-quads".into());
        }
        event
    }
    pub fn begin(&self, device: &wgpu::Device) {
        if !self.enabled {
            return;
        }
        let event = self.new_event();
        if !self.supported {
            gpu::record_gpu_event(event.unavailable("unsupported", "TIMESTAMP_QUERY unavailable"));
            return;
        }
        #[cfg(not(target_family = "wasm"))]
        if self.sender.is_none() {
            gpu::record_gpu_event(
                event.unavailable("query-failed", "Could not start timestamp readback worker"),
            );
            return;
        }
        if self
            .pending
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < MAX_PENDING).then_some(count + 1)
            })
            .is_err()
        {
            gpu::record_gpu_event(
                event.unavailable("ring-full", "Eight GPU query captures are still pending"),
            );
            return;
        }
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpui_timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_PASSES * 2,
        });
        let bytes = MAX_PASSES as u64 * 16;
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpui_timestamp_resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpui_timestamp_readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        *self.current.borrow_mut() = Some(Frame {
            event,
            queries,
            resolve,
            readback,
            labels: Vec::new(),
            overflow: false,
        });
    }
    pub fn pass(&self, name: &str) -> Option<TimestampPass> {
        let mut current = self.current.borrow_mut();
        let frame = current.as_mut()?;
        let index = frame.labels.len() as u32;
        if index == MAX_PASSES {
            frame.overflow = true;
            return None;
        }
        frame.labels.push(name.into());
        Some(TimestampPass {
            query_set: frame.queries.clone(),
            first: index * 2,
            last: index * 2 + 1,
        })
    }
    pub fn describe_damage(
        &self,
        kind: &str,
        retained_bytes: u64,
        fallback_reason: Option<&str>,
        diff_detail: Option<String>,
        mut experiments: Vec<String>,
    ) {
        if self.clipped_quads {
            experiments.push("clipped-quads".into());
        }
        if let Some(frame) = self.current.borrow_mut().as_mut() {
            frame.event.damage_kind = kind.into();
            frame.event.damage_fallback_reason = fallback_reason.map(str::to_owned);
            frame.event.damage_diff_detail = diff_detail;
            frame.event.retained_bytes = retained_bytes;
            frame.event.experiments = experiments;
        }
    }

    pub fn record_quad_draws(&self, draws: u64, splits: u64) {
        if !self.enabled {
            return;
        }
        if let Some(frame) = self.current.borrow_mut().as_mut() {
            frame.event.quad_draw_calls = frame.event.quad_draw_calls.saturating_add(draws);
            frame.event.quad_scissor_splits =
                frame.event.quad_scissor_splits.saturating_add(splits);
        }
    }
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(frame) = self
            .current
            .borrow()
            .as_ref()
            .filter(|frame| !frame.labels.is_empty())
        {
            let count = frame.labels.len() as u32 * 2;
            encoder.resolve_query_set(&frame.queries, 0..count, &frame.resolve, 0);
            encoder.copy_buffer_to_buffer(&frame.resolve, 0, &frame.readback, 0, count as u64 * 8);
        }
    }
    pub fn submitted(&self, submission: wgpu::SubmissionIndex) {
        let Some(frame) = self.current.borrow_mut().take() else {
            return;
        };
        if frame.labels.is_empty() {
            self.pending.fetch_sub(1, Ordering::AcqRel);
            gpu::record_gpu_event(
                frame
                    .event
                    .unavailable("incomplete", "No timestamped render passes"),
            );
            return;
        }
        #[cfg(not(target_family = "wasm"))]
        if let Some(sender) = &self.sender {
            if let Err(error) = sender.try_send(Pending { frame, submission }) {
                self.pending.fetch_sub(1, Ordering::AcqRel);
                let (capture, status, reason) = match error {
                    std::sync::mpsc::TrySendError::Full(capture) => {
                        (capture, "ring-full", "Readback worker queue is full")
                    }
                    std::sync::mpsc::TrySendError::Disconnected(capture) => {
                        (capture, "query-failed", "Readback worker disconnected")
                    }
                };
                gpu::record_gpu_event(capture.frame.event.unavailable(status, reason));
            }
        }
    }
    pub fn cancel(&self, reason: &str) {
        if let Some(frame) = self.current.borrow_mut().take() {
            self.pending.fetch_sub(1, Ordering::AcqRel);
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
        self.cancel("Renderer dropped before submission");
    }
}
