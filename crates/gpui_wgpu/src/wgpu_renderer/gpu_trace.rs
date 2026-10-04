//! Bounded asynchronous timestamp readback. Profiling never blocks the draw thread.
use gpui::gpu_profiler::{self, GpuFrameMetrics, GpuPassTiming};
use std::{
    cell::RefCell,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

const SLOTS: usize = 4;
const QUERIES: u32 = 1024;
struct Slot {
    buffer: wgpu::Buffer,
    busy: Arc<AtomicBool>,
}
struct Frame {
    slot: usize,
    names: Vec<&'static str>,
    metrics: GpuFrameMetrics,
}

pub(super) struct GpuTrace {
    queries: Option<wgpu::QuerySet>,
    resolve: Option<wgpu::Buffer>,
    slots: Vec<Slot>,
    active: RefCell<Option<Frame>>,
    period: f64,
    #[cfg(not(target_family = "wasm"))]
    poll: Option<std::sync::mpsc::SyncSender<wgpu::SubmissionIndex>>,
}

impl GpuTrace {
    pub(super) fn new(device: &Arc<wgpu::Device>, queue: &wgpu::Queue) -> Self {
        let supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let queries = supported.then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("gpui_gpu_timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERIES * SLOTS as u32,
            })
        });
        let resolve = supported.then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpui_gpu_timestamp_resolve"),
                size: QUERIES as u64 * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let slots = if supported {
            (0..SLOTS)
                .map(|_| Slot {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("gpui_gpu_timestamp_readback"),
                        size: QUERIES as u64 * 8,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }),
                    busy: Arc::new(AtomicBool::new(false)),
                })
                .collect()
        } else {
            Vec::new()
        };
        #[cfg(not(target_family = "wasm"))]
        let poll = supported.then(|| {
            let (tx, rx) = std::sync::mpsc::sync_channel(SLOTS);
            let device = device.clone();
            std::thread::Builder::new()
                .name("gpui-gpu-timing".into())
                .spawn(move || {
                    while let Ok(submission) = rx.recv() {
                        let _ = device.poll(wgpu::PollType::Wait {
                            submission_index: Some(submission),
                            timeout: Some(std::time::Duration::from_secs(5)),
                        });
                    }
                })
                .expect("GPU timing worker");
            tx
        });
        Self {
            queries,
            resolve,
            slots,
            active: RefCell::new(None),
            period: queue.get_timestamp_period() as f64,
            #[cfg(not(target_family = "wasm"))]
            poll,
        }
    }

    pub(super) fn begin(&self, mut metrics: GpuFrameMetrics) {
        self.cancel();
        let slot = self
            .slots
            .iter()
            .position(|s| !s.busy.swap(true, Ordering::AcqRel));
        let Some(slot) = slot else {
            if self.queries.is_some() {
                metrics.status = "query_ring_full";
                metrics.query_samples_dropped = 1;
            }
            gpu_profiler::record(metrics);
            return;
        };
        *self.active.borrow_mut() = Some(Frame {
            slot,
            names: Vec::with_capacity(32),
            metrics,
        });
    }

    pub(super) fn timestamps(
        &self,
        name: &'static str,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let mut active = self.active.borrow_mut();
        let frame = active.as_mut()?;
        if frame.names.len() >= QUERIES as usize / 2 {
            frame.metrics.query_samples_dropped += 1;
            return None;
        }
        let start = frame.slot as u32 * QUERIES + frame.names.len() as u32 * 2;
        frame.names.push(name);
        Some(wgpu::RenderPassTimestampWrites {
            query_set: self.queries.as_ref()?,
            beginning_of_pass_write_index: Some(start),
            end_of_pass_write_index: Some(start + 1),
        })
    }

    pub(super) fn cache_stats(&self, bytes: u64, hits: u64, misses: u64, retained_bytes: u64) {
        if let Some(frame) = self.active.borrow_mut().as_mut() {
            frame.metrics.memory.cached_bytes = Some(bytes);
            frame.metrics.memory.device_retention_bytes = Some(retained_bytes);
            frame.metrics.cache_hits = hits;
            frame.metrics.cache_misses = misses;
        }
    }

    pub(super) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let active = self.active.borrow();
        let Some(frame) = active.as_ref().filter(|f| !f.names.is_empty()) else {
            return;
        };
        let count = frame.names.len() as u32 * 2;
        let first = frame.slot as u32 * QUERIES;
        let resolve = self.resolve.as_ref().unwrap();
        encoder.resolve_query_set(
            self.queries.as_ref().unwrap(),
            first..first + count,
            resolve,
            0,
        );
        encoder.copy_buffer_to_buffer(
            resolve,
            0,
            &self.slots[frame.slot].buffer,
            0,
            count as u64 * 8,
        );
    }

    pub(super) fn submitted(&self, submission: wgpu::SubmissionIndex) {
        let Some(mut frame) = self.active.borrow_mut().take() else {
            return;
        };
        let busy = self.slots[frame.slot].busy.clone();
        if frame.names.is_empty() {
            frame.metrics.status = "no_timed_passes";
            gpu_profiler::record(frame.metrics);
            busy.store(false, Ordering::Release);
            return;
        }
        frame.metrics.submitted_at = std::time::Instant::now();
        let buffer = self.slots[frame.slot].buffer.clone();
        let readback = buffer.clone();
        let bytes = frame.names.len() as u64 * 16;
        let period = self.period;
        buffer
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    let mapped = readback.slice(..bytes).get_mapped_range();
                    let ticks: Vec<_> = mapped
                        .chunks_exact(8)
                        .map(|b| u64::from_ne_bytes(b.try_into().unwrap()))
                        .collect();
                    let mut first = u64::MAX;
                    let mut last = 0;
                    for (name, pair) in frame.names.iter().zip(ticks.chunks_exact(2)) {
                        if let Some(duration_ns) =
                            gpu_profiler::elapsed_ns(pair[0], pair[1], period)
                        {
                            first = first.min(pair[0]);
                            last = last.max(pair[1]);
                            frame.metrics.passes.push(GpuPassTiming {
                                name: (*name).into(),
                                duration_ns,
                            });
                        } else {
                            frame.metrics.query_samples_dropped += 1;
                        }
                    }
                    frame.metrics.gpu_duration_ns = if frame.metrics.query_samples_dropped == 0 {
                        gpu_profiler::elapsed_ns(first, last, period)
                    } else {
                        None
                    };
                    frame.metrics.pass_timing_complete = frame.metrics.query_samples_dropped == 0;
                    frame.metrics.status = if frame.metrics.gpu_duration_ns.is_some() {
                        "available"
                    } else {
                        "incomplete"
                    };
                    drop(mapped);
                    readback.unmap();
                } else {
                    frame.metrics.status = "readback_failed";
                }
                gpu_profiler::record(frame.metrics);
                busy.store(false, Ordering::Release);
            });
        #[cfg(not(target_family = "wasm"))]
        if let Some(poll) = &self.poll {
            let _ = poll.try_send(submission);
        }
        #[cfg(target_family = "wasm")]
        let _ = submission;
    }

    pub(super) fn cancel(&self) {
        if let Some(mut frame) = self.active.borrow_mut().take() {
            frame.metrics.status = "render_failed";
            gpu_profiler::record(frame.metrics);
            self.slots[frame.slot].busy.store(false, Ordering::Release);
        }
    }
}

impl Drop for GpuTrace {
    fn drop(&mut self) {
        self.cancel();
    }
}
