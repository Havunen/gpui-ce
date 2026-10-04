//! Metal stage-boundary counters with command-buffer timing as the fallback.
use block::ConcreteBlock;
use foreign_types::ForeignTypeRef;
use gpui::gpu_profiler::{self, GpuFrameMetrics, GpuPassTiming};
use metal::{CounterSampleBuffer, RenderPassDescriptorRef};
use objc2::{msg_send, rc::Retained, runtime::AnyObject};
use objc2_foundation::{NSData, NSRange};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

const SAMPLES: u64 = 1024;
struct Slot {
    buffer: CounterSampleBuffer,
    busy: Arc<AtomicBool>,
}
pub(crate) struct GpuTrace {
    device: metal::Device,
    slots: Vec<Slot>,
    renderer: u64,
    submission: Cell<u64>,
}
struct Recording {
    buffer: Option<CounterSampleBuffer>,
    busy: Option<Arc<AtomicBool>>,
    names: RefCell<Vec<&'static str>>,
    dropped: Cell<u64>,
}
thread_local! { static ACTIVE: RefCell<Option<Rc<Recording>>> = const { RefCell::new(None) }; }
pub(crate) struct Frame {
    recording: Rc<Recording>,
    previous: Option<Rc<Recording>>,
    metrics: Option<GpuFrameMetrics>,
    device: metal::Device,
    calibration: (u64, u64),
}
impl GpuTrace {
    pub(crate) fn new(device: &metal::Device) -> Self {
        let mut slots = Vec::new();
        if device.supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary) {
            if let Some(set) = device
                .counter_sets()
                .into_iter()
                .find(|s| s.name().eq_ignore_ascii_case("timestamp"))
            {
                for _ in 0..4 {
                    let descriptor = metal::CounterSampleBufferDescriptor::new();
                    descriptor.set_counter_set(&set);
                    descriptor.set_sample_count(SAMPLES);
                    descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
                    match device.new_counter_sample_buffer_with_descriptor(&descriptor) {
                        Ok(buffer) => slots.push(Slot {
                            buffer,
                            busy: Arc::new(AtomicBool::new(false)),
                        }),
                        Err(error) => {
                            log::warn!("Metal pass counters unavailable: {error}");
                            break;
                        }
                    }
                }
            }
        }
        Self {
            device: device.clone(),
            slots,
            renderer: gpu_profiler::next_renderer(),
            submission: Cell::new(0),
        }
    }
    pub(crate) fn begin(&self, scene: &gpui::Scene) -> Frame {
        self.submission.set(self.submission.get() + 1);
        let slot = self
            .slots
            .iter()
            .find(|slot| !slot.busy.swap(true, Ordering::AcqRel));
        let recording = Rc::new(Recording {
            buffer: slot.map(|s| s.buffer.clone()),
            busy: slot.map(|s| s.busy.clone()),
            names: RefCell::new(Vec::new()),
            dropped: Cell::new(u64::from(slot.is_none() && !self.slots.is_empty())),
        });
        let previous = ACTIVE.with_borrow_mut(|active| active.replace(recording.clone()));
        let (mut cpu, mut gpu) = (0, 0);
        if slot.is_some() {
            self.device.sample_timestamps(&mut cpu, &mut gpu);
        }
        Frame {
            recording,
            previous,
            metrics: Some(GpuFrameMetrics::new(
                scene,
                self.renderer,
                self.submission.get(),
                "metal",
            )),
            device: self.device.clone(),
            calibration: (cpu, gpu),
        }
    }
}
pub(crate) fn attach(name: &'static str, descriptor: &RenderPassDescriptorRef) {
    if !gpu_profiler::enabled() {
        return;
    }
    ACTIVE.with_borrow(|active| {
        let Some(recording) = active else {
            return;
        };
        let Some(buffer) = &recording.buffer else {
            return;
        };
        let mut names = recording.names.borrow_mut();
        let index = names.len() as u64 * 4;
        if index + 4 > SAMPLES {
            recording.dropped.set(recording.dropped.get() + 1);
            return;
        }
        let Some(attachment) = descriptor.sample_buffer_attachments().object_at(0) else {
            return;
        };
        attachment.set_sample_buffer(buffer);
        attachment.set_start_of_vertex_sample_index(index);
        attachment.set_end_of_vertex_sample_index(index + 1);
        attachment.set_start_of_fragment_sample_index(index + 2);
        attachment.set_end_of_fragment_sample_index(index + 3);
        names.push(name);
    });
}
impl Frame {
    pub(crate) fn finish(
        mut self,
        command: &metal::CommandBufferRef,
        path_size: (u32, u32),
        path_bytes: u64,
    ) {
        let mut metrics = self.metrics.take().unwrap();
        metrics.path_target = path_size;
        metrics.memory.path_bytes = path_bytes;
        metrics.query_samples_dropped = self.recording.dropped.get();
        let names = self.recording.names.borrow().clone();
        let samples = self.recording.buffer.clone();
        let busy = self.recording.busy.clone();
        let device = self.device.clone();
        let (cpu_start, gpu_start) = self.calibration;
        let metrics = std::sync::Mutex::new(Some(metrics));
        let callback = ConcreteBlock::new(move |command: &metal::CommandBufferRef| {
            let Some(mut metrics) = metrics.lock().unwrap().take() else {
                return;
            };
            let object = unsafe { &*command.as_ptr().cast::<AnyObject>() };
            let start: f64 = unsafe { msg_send![object, GPUStartTime] };
            let end: f64 = unsafe { msg_send![object, GPUEndTime] };
            if start > 0.0 && end >= start && end.is_finite() {
                metrics.gpu_duration_ns = Some(((end - start) * 1e9) as u64);
                metrics.status = "available";
            } else {
                metrics.status = "invalid_timestamps";
            }
            if let Some(samples) = &samples {
                let (mut cpu_end, mut gpu_end) = (0, 0);
                device.sample_timestamps(&mut cpu_end, &mut gpu_end);
                let period = timestamp_period(cpu_start, cpu_end, gpu_start, gpu_end);
                let object = unsafe { &*samples.as_ptr().cast::<AnyObject>() };
                // Avoid metal-rs 0.33's resolve_counter_range: its Vec has length
                // zero when computing the byte count, so it copies no data.
                let data: Option<Retained<NSData>> = unsafe {
                    msg_send![object, resolveCounterRange: NSRange::new(0, names.len() * 4)]
                };
                if let (Some(data), Some(period)) = (data, period) {
                    let length: usize = unsafe { msg_send![&*data, length] };
                    let bytes: *const u8 = unsafe { msg_send![&*data, bytes] };
                    if !bytes.is_null() && length >= names.len() * 32 {
                        let values = unsafe { std::slice::from_raw_parts(bytes, names.len() * 32) };
                        for (name, bytes) in names.iter().zip(values.chunks_exact(32)) {
                            let ticks: Vec<_> = bytes
                                .chunks_exact(8)
                                .map(|b| u64::from_ne_bytes(b.try_into().unwrap()))
                                .collect();
                            let start = ticks[0].min(ticks[2]);
                            let end = ticks[1].max(ticks[3]);
                            if let Some(duration_ns) = gpu_profiler::elapsed_ns(start, end, period)
                            {
                                metrics.passes.push(GpuPassTiming {
                                    name: (*name).into(),
                                    duration_ns,
                                });
                            }
                        }
                    }
                }
            }
            metrics.pass_timing_complete = samples.is_some()
                && metrics.query_samples_dropped == 0
                && metrics.passes.len() == names.len()
                && !names.is_empty();
            gpu_profiler::record(metrics);
            if let Some(busy) = &busy {
                busy.store(false, Ordering::Release);
            }
        });
        command.add_completed_handler(&callback.copy());
    }
}
impl Drop for Frame {
    fn drop(&mut self) {
        ACTIVE.with_borrow_mut(|active| *active = self.previous.take());
        if self.metrics.is_some() {
            if let Some(busy) = &self.recording.busy {
                busy.store(false, Ordering::Release);
            }
        }
    }
}
fn timestamp_period(cpu_start: u64, cpu_end: u64, gpu_start: u64, gpu_end: u64) -> Option<f64> {
    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }
    unsafe extern "C" {
        fn mach_timebase_info(info: *mut Timebase) -> i32;
    }
    let mut timebase = Timebase { numer: 0, denom: 0 };
    if unsafe { mach_timebase_info(&mut timebase) } != 0
        || timebase.denom == 0
        || cpu_end <= cpu_start
        || gpu_end <= gpu_start
    {
        return None;
    }
    let cpu_ns = (cpu_end - cpu_start) as f64 * timebase.numer as f64 / timebase.denom as f64;
    (cpu_ns >= 1_000_000.0).then_some(cpu_ns / (gpu_end - gpu_start) as f64)
}
