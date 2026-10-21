#![cfg(all(target_os = "linux", any(feature = "x11", feature = "wayland")))]

use gpui::{
    App, Application, Bounds, Context, Platform, Window, WindowBounds, WindowOptions, div,
    prelude::*, px, rgb, size,
};
use gpui_ce_linux::{
    LinuxRendererBackend, LinuxRendererEvent, LinuxRendererOptions, LinuxRendererPreference,
    current_platform_with_renderer,
};
use gpui_wgpu::{WgpuBackend, WgpuContextHandle, wgpu};
use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
    time::Duration,
};

struct RendererProbe {
    rendered: Rc<Cell<bool>>,
    frame_completed: Rc<Cell<bool>>,
}

impl Render for RendererProbe {
    fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.rendered.set(true);
        let completed = self.frame_completed.clone();
        window.on_next_frame(move |window, _| {
            window.on_next_frame(move |window, _| {
                let context = window
                    .gpu_context_info()
                    .unwrap()
                    .downcast::<WgpuContextHandle>()
                    .unwrap();
                assert_eq!(context.backend(), WgpuBackend::Native(wgpu::Backend::Gl));
                completed.set(true);
            });
        });
        div().size_full().bg(rgb(0x2468ac))
    }
}

#[test]
#[ignore = "requires a native Linux display and OpenGL; run with WGPU_BACKEND unset"]
fn typed_gl_selection_renders_and_callbacks_can_reenter_platform() {
    assert!(
        std::env::var_os("WGPU_BACKEND").is_none(),
        "unset WGPU_BACKEND to test the typed preference"
    );
    assert!(
        matches!(gpui::guess_compositor(), "X11" | "Wayland"),
        "set DISPLAY or WAYLAND_DISPLAY for a compiled native backend"
    );
    let platform_holder: Rc<RefCell<Option<Weak<dyn Platform>>>> = Rc::default();
    let selected = Rc::new(Cell::new(false));
    let selected_by_callback = selected.clone();
    let holder_for_callback = platform_holder.clone();
    let platform = current_platform_with_renderer(
        false,
        LinuxRendererOptions {
            preference: LinuxRendererPreference::Gl,
            on_event: Some(Rc::new(move |event| {
                assert_eq!(
                    event,
                    LinuxRendererEvent::Selected {
                        backend: LinuxRendererBackend::Gl,
                        overridden: false
                    }
                );
                let platform = holder_for_callback
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .upgrade()
                    .unwrap();
                assert!(!platform.displays().is_empty());
                let _ = platform.active_window();
                selected_by_callback.set(true);
            })),
        },
    );
    *platform_holder.borrow_mut() = Some(Rc::downgrade(&platform));
    let rendered = Rc::new(Cell::new(false));
    let frame_completed = Rc::new(Cell::new(false));
    let probe_rendered = rendered.clone();
    let probe_completed = frame_completed.clone();
    let selection_for_exit = selected.clone();
    Application::with_platform(platform).run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(160.0), px(100.0)), cx);
        cx.open_window(
            WindowOptions::new().window_bounds(Some(WindowBounds::Windowed(bounds))),
            |_, cx| {
                cx.new(|_| RendererProbe {
                    rendered: probe_rendered,
                    frame_completed: probe_completed.clone(),
                })
            },
        )
        .unwrap();
        cx.spawn(async move |cx| {
            for _ in 0..100 {
                smol::Timer::after(Duration::from_millis(50)).await;
                if selection_for_exit.get() && probe_completed.get() {
                    break;
                }
            }
            cx.update(|cx| cx.quit());
        })
        .detach();
    });
    assert!(selected.get(), "renderer callback was not delivered");
    assert!(rendered.get(), "native window did not render");
    assert!(frame_completed.get(), "native frames did not complete");
}
