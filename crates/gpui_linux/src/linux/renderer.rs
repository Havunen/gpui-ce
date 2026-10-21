use std::rc::Rc;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LinuxRendererPreference {
    #[default]
    Auto,
    Vulkan,
    Gl,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinuxRendererBackend {
    Vulkan,
    Gl,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinuxRendererEvent {
    Selected {
        backend: LinuxRendererBackend,
        overridden: bool,
    },
}

#[derive(Clone, Default)]
pub struct LinuxRendererOptions {
    pub preference: LinuxRendererPreference,
    pub on_event: Option<Rc<dyn Fn(LinuxRendererEvent)>>,
}

#[cfg(any(feature = "wayland", feature = "x11"))]
mod context {
    use super::*;
    use gpui::ForegroundExecutor;
    use gpui_wgpu::{
        CompositorGpuHint, GpuContext, WgpuBackend, WgpuDeviceRequirements, WgpuRenderer,
        WgpuSurfaceConfig, wgpu,
    };
    use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

    #[derive(Clone)]
    pub(crate) struct RendererContext(Rc<RendererState>);

    struct RendererState {
        gpu: GpuContext,
        backends: wgpu::Backends,
        overridden: bool,
        on_event: Option<Rc<dyn Fn(LinuxRendererEvent)>>,
        foreground_executor: ForegroundExecutor,
    }

    impl RendererContext {
        pub(crate) fn new(
            options: LinuxRendererOptions,
            foreground_executor: ForegroundExecutor,
        ) -> Self {
            Self::with_override(options, foreground_executor, wgpu::Backends::from_env())
        }

        fn with_override(
            options: LinuxRendererOptions,
            foreground_executor: ForegroundExecutor,
            backend_override: Option<wgpu::Backends>,
        ) -> Self {
            let backends = backend_override.unwrap_or_else(|| match options.preference {
                LinuxRendererPreference::Auto => wgpu::Backends::VULKAN | wgpu::Backends::GL,
                LinuxRendererPreference::Vulkan => wgpu::Backends::VULKAN,
                LinuxRendererPreference::Gl => wgpu::Backends::GL,
            });
            Self(Rc::new(RendererState {
                gpu: GpuContext::default(),
                backends,
                overridden: backend_override.is_some(),
                on_event: options.on_event,
                foreground_executor,
            }))
        }

        pub(crate) fn gpu(&self) -> &GpuContext {
            &self.0.gpu
        }

        pub(crate) fn create_renderer<W>(
            &self,
            window: &W,
            config: WgpuSurfaceConfig,
            compositor_gpu: Option<CompositorGpuHint>,
            extra_requirements: Option<WgpuDeviceRequirements>,
        ) -> anyhow::Result<WgpuRenderer>
        where
            W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
        {
            WgpuRenderer::new_with_backends(
                self.0.gpu.clone(),
                window,
                config,
                compositor_gpu,
                extra_requirements,
                self.0.backends,
            )
        }

        pub(crate) fn notify_selected(&self) {
            let backend = self
                .0
                .gpu
                .borrow()
                .as_ref()
                .map(|context| context.backend());
            if let Some(backend) = backend {
                self.emit_selected(backend);
            }
        }

        fn emit_selected(&self, backend: WgpuBackend) {
            let Some(callback) = self.0.on_event.clone() else {
                return;
            };
            let backend = match backend {
                WgpuBackend::Native(wgpu::Backend::Vulkan) => LinuxRendererBackend::Vulkan,
                WgpuBackend::Native(wgpu::Backend::Gl) => LinuxRendererBackend::Gl,
                backend => unreachable!("unsupported Linux renderer backend: {backend:?}"),
            };
            let event = LinuxRendererEvent::Selected {
                backend,
                overridden: self.0.overridden,
            };
            self.0
                .foreground_executor
                .spawn(async move { callback(event) })
                .detach();
        }
    }

    #[cfg(all(test, feature = "test-support"))]
    mod tests {
        use super::*;
        use gpui::TestDispatcher;
        use std::{cell::RefCell, sync::Arc};

        #[test]
        fn preferences_and_environment_overrides_resolve_before_gpu_initialization() {
            let executor = ForegroundExecutor::new(Arc::new(TestDispatcher::new(0)));
            for (preference, expected) in [
                (
                    LinuxRendererPreference::Auto,
                    wgpu::Backends::VULKAN | wgpu::Backends::GL,
                ),
                (LinuxRendererPreference::Vulkan, wgpu::Backends::VULKAN),
                (LinuxRendererPreference::Gl, wgpu::Backends::GL),
            ] {
                for backend_override in [
                    None,
                    Some(wgpu::Backends::GL),
                    Some(wgpu::Backends::VULKAN),
                    Some(wgpu::Backends::VULKAN | wgpu::Backends::GL),
                    Some(wgpu::Backends::empty()),
                    Some(wgpu::Backends::BROWSER_WEBGPU),
                ] {
                    let context = RendererContext::with_override(
                        LinuxRendererOptions {
                            preference,
                            on_event: None,
                        },
                        executor.clone(),
                        backend_override,
                    );
                    assert_eq!(context.0.backends, backend_override.unwrap_or(expected));
                    assert_eq!(context.0.overridden, backend_override.is_some());
                    assert!(context.gpu().borrow().is_none());
                }
            }
        }

        #[test]
        fn selection_callbacks_are_deferred_and_can_reenter_platform_state() {
            let dispatcher = TestDispatcher::new(0);
            let platform_state = Rc::new(RefCell::new(Vec::new()));
            let state_for_callback = platform_state.clone();
            let context = RendererContext::with_override(
                LinuxRendererOptions {
                    preference: LinuxRendererPreference::Auto,
                    on_event: Some(Rc::new(move |event| {
                        state_for_callback.borrow_mut().push(event);
                    })),
                },
                ForegroundExecutor::new(Arc::new(dispatcher.clone())),
                Some(wgpu::Backends::VULKAN | wgpu::Backends::GL),
            );
            let state_guard = platform_state.borrow_mut();
            let gpu_guard = context.gpu().borrow_mut();
            context.emit_selected(WgpuBackend::Native(wgpu::Backend::Vulkan));
            context.emit_selected(WgpuBackend::Native(wgpu::Backend::Gl));
            assert!(state_guard.is_empty());
            drop(gpu_guard);
            drop(state_guard);
            dispatcher.run_until_parked();
            assert_eq!(
                *platform_state.borrow(),
                [
                    LinuxRendererEvent::Selected {
                        backend: LinuxRendererBackend::Vulkan,
                        overridden: true
                    },
                    LinuxRendererEvent::Selected {
                        backend: LinuxRendererBackend::Gl,
                        overridden: true
                    },
                ]
            );
        }
    }
}

#[cfg(any(feature = "wayland", feature = "x11"))]
pub(crate) use context::RendererContext;
