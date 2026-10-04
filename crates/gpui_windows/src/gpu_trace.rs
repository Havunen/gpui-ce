//! Nonblocking D3D11 timestamp collection, on the immediate context's UI thread.
use crate::bindings::Windows::Win32::*;
use gpui::gpu_profiler::{self, GpuFrameMetrics, GpuPassTiming};
use std::{cell::RefCell, rc::Rc};

const SLOTS: usize = 4;
const PAIRS: usize = 256;
struct Slot {
    disjoint: ID3D11Query,
    queries: Vec<ID3D11Query>,
    names: Vec<&'static str>,
    metrics: Option<GpuFrameMetrics>,
    pending: bool,
}
struct State {
    slots: Vec<Slot>,
    active: Option<usize>,
    submission: u64,
}
pub(crate) struct GpuTrace {
    context: ID3D11DeviceContext,
    state: RefCell<State>,
    renderer: u64,
}
pub(crate) struct Span {
    context: ID3D11DeviceContext,
    end: Option<ID3D11Query>,
}
impl Drop for Span {
    fn drop(&mut self) {
        if let Some(query) = &self.end {
            unsafe { self.context.End(query) };
        }
    }
}

impl GpuTrace {
    pub(crate) fn new(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
    ) -> anyhow::Result<Rc<Self>> {
        let query = |kind| -> anyhow::Result<ID3D11Query> {
            let mut result = None;
            unsafe {
                device
                    .CreateQuery(
                        &D3D11_QUERY_DESC {
                            Query: kind,
                            MiscFlags: 0,
                        },
                        Some(&mut result),
                    )
                    .ok()?;
            }
            result.ok_or_else(|| anyhow::anyhow!("missing timestamp query"))
        };
        let mut slots = Vec::new();
        for _ in 0..SLOTS {
            slots.push(Slot {
                disjoint: query(D3D11_QUERY_TIMESTAMP_DISJOINT)?,
                queries: (0..PAIRS * 2)
                    .map(|_| query(D3D11_QUERY_TIMESTAMP))
                    .collect::<anyhow::Result<_>>()?,
                names: Vec::new(),
                metrics: None,
                pending: false,
            });
        }
        let trace = Rc::new(Self {
            context: context.clone(),
            state: RefCell::new(State {
                slots,
                active: None,
                submission: 0,
            }),
            renderer: gpu_profiler::next_renderer(),
        });
        let weak = Rc::downgrade(&trace);
        gpu_profiler::register_poller(move || {
            if let Some(trace) = weak.upgrade() {
                trace.poll();
                true
            } else {
                false
            }
        });
        Ok(trace)
    }

    pub(crate) fn begin(&self, scene: &gpui::Scene) {
        self.poll();
        let mut state = self.state.borrow_mut();
        state.submission += 1;
        let mut metrics = GpuFrameMetrics::new(scene, self.renderer, state.submission, "d3d11");
        let Some(index) = state
            .slots
            .iter()
            .position(|s| !s.pending && s.metrics.is_none())
        else {
            metrics.status = "query_ring_full";
            metrics.query_samples_dropped = 1;
            gpu_profiler::record(metrics);
            return;
        };
        let slot = &mut state.slots[index];
        slot.names.clear();
        slot.names.push("frame");
        slot.metrics = Some(metrics);
        unsafe {
            self.context.Begin(&slot.disjoint);
            self.context.End(&slot.queries[0]);
        }
        state.active = Some(index);
    }
    pub(crate) fn span(&self, name: &'static str) -> Span {
        let mut state = self.state.borrow_mut();
        let end = if let Some(index) = state.active {
            let slot = &mut state.slots[index];
            if slot.names.len() < PAIRS {
                let i = slot.names.len() * 2;
                slot.names.push(name);
                unsafe {
                    self.context.End(&slot.queries[i]);
                }
                Some(slot.queries[i + 1].clone())
            } else {
                if let Some(m) = &mut slot.metrics {
                    m.query_samples_dropped += 1;
                }
                None
            }
        } else {
            None
        };
        Span {
            context: self.context.clone(),
            end,
        }
    }
    pub(crate) fn finish(&self, succeeded: bool, path_size: (u32, u32)) {
        let mut state = self.state.borrow_mut();
        if let Some(index) = state.active.take() {
            let slot = &mut state.slots[index];
            unsafe {
                self.context.End(&slot.queries[1]);
                self.context.End(&slot.disjoint);
            }
            slot.pending = true;
            if let Some(metrics) = &mut slot.metrics {
                metrics.path_target = path_size;
                metrics.memory.path_bytes = u64::from(path_size.0) * u64::from(path_size.1) * 4 * 5;
                if !succeeded {
                    metrics.status = "render_failed";
                }
            }
        }
    }
    fn data<T: Default>(&self, query: &ID3D11Query) -> Option<T> {
        let mut value = T::default();
        let result = unsafe {
            self.context.GetData(
                query,
                Some((&mut value as *mut T).cast()),
                std::mem::size_of::<T>() as u32,
                D3D11_ASYNC_GETDATA_DONOTFLUSH as u32,
            )
        };
        // S_FALSE means pending; HRESULT::is_ok() would incorrectly accept it.
        (result.0 == 0).then_some(value)
    }
    fn poll(&self) {
        let mut state = self.state.borrow_mut();
        for slot in &mut state.slots {
            if !slot.pending {
                continue;
            }
            let Some(disjoint) = self.data::<D3D11_QUERY_DATA_TIMESTAMP_DISJOINT>(&slot.disjoint)
            else {
                continue;
            };
            let values: Option<Vec<u64>> = slot.queries[..slot.names.len() * 2]
                .iter()
                .map(|query| self.data(query))
                .collect();
            let Some(values) = values else {
                continue;
            };
            let Some(mut metrics) = slot.metrics.take() else {
                continue;
            };
            if !disjoint.Disjoint.as_bool()
                && disjoint.Frequency > 0
                && metrics.status != "render_failed"
            {
                let period = 1_000_000_000.0 / disjoint.Frequency as f64;
                metrics.gpu_duration_ns = gpu_profiler::elapsed_ns(values[0], values[1], period);
                for (name, pair) in slot.names.iter().zip(values.chunks_exact(2)).skip(1) {
                    if let Some(duration_ns) = gpu_profiler::elapsed_ns(pair[0], pair[1], period) {
                        metrics.passes.push(GpuPassTiming {
                            name: (*name).into(),
                            duration_ns,
                        });
                    }
                }
                metrics.pass_timing_complete = metrics.query_samples_dropped == 0
                    && metrics.passes.len() + 1 == slot.names.len();
                metrics.status = if metrics.gpu_duration_ns.is_some() {
                    "available"
                } else {
                    "invalid_timestamps"
                };
            } else if metrics.status != "render_failed" {
                metrics.status = "disjoint";
            }
            gpu_profiler::record(metrics);
            slot.pending = false;
        }
    }
}
