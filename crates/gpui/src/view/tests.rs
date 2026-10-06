use super::*;
use crate::{
    Animation, AnimationExt, InlineLayout, InlineLayoutRequest, LayoutDirection, LineLayout,
    ResolvedDirection, ShapedTextLayout, SharedString, TestAppContext, TextLayoutOptions, TextRun,
    WindowControlArea, WindowHandle, deferred, div, prelude::*, px, size,
};
use std::{cell::RefCell, rc::Rc, sync::Arc, time::Duration};

type Samples = Rc<RefCell<Vec<(usize, f32)>>>;

struct AnimatedView {
    animations: Vec<Animation>,
    samples: Samples,
}

impl Render for AnimatedView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let samples = self.samples.clone();
        div().size_full().with_animations(
            "fade",
            self.animations.clone(),
            move |element, phase, progress| {
                samples.borrow_mut().push((phase, progress));
                let opacity = match phase {
                    0 => progress,
                    1 => 1.0,
                    _ => 1.0 - progress,
                };
                element.opacity(opacity)
            },
        )
    }
}

struct CachedContainer {
    child: AnyView,
}

impl Render for CachedContainer {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        self.child
            .clone()
            .cached(StyleRefinement::default().size_full())
    }
}

struct AnimationRoot {
    animation: Entity<AnimatedView>,
    content: AnyView,
    mounted: bool,
    font_size: Pixels,
    padding: Pixels,
}

impl Render for AnimationRoot {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .text_size(self.font_size)
            .p(self.padding)
            .when(self.mounted, |element| {
                element.child(
                    self.content
                        .clone()
                        .cached(StyleRefinement::default().size_full()),
                )
            })
    }
}

fn open_animation(
    cx: &mut TestAppContext,
    animations: Vec<Animation>,
    nested: bool,
) -> (WindowHandle<AnimationRoot>, Samples) {
    let samples = Samples::default();
    let recorded = samples.clone();
    let window = cx.open_window(size(px(100.), px(100.)), move |_, cx| {
        let animation = cx.new(|_| AnimatedView {
            animations,
            samples: recorded,
        });
        let mut content = AnyView::from(animation.clone());
        if nested {
            content = cx.new(|_| CachedContainer { child: content }).into();
        }
        AnimationRoot {
            animation,
            content,
            mounted: true,
            font_size: px(14.),
            padding: px(0.),
        }
    });
    cx.run_until_parked();
    (window, samples)
}

fn assert_sample(samples: &Samples, phase: usize, progress: f32) {
    let actual = *samples.borrow().last().expect("animation rendered");
    assert_eq!(actual.0, phase, "animation restarted or changed phase");
    assert!(
        (actual.1 - progress).abs() < 0.01,
        "expected progress {progress}, got {actual:?}"
    );
}

fn advance_frame(window: WindowHandle<AnimationRoot>, cx: &mut TestAppContext, elapsed: Duration) {
    cx.executor().advance_clock(elapsed);
    window
        .update(cx, |_, window, cx| {
            assert!(window.simulate_next_frame(cx) > 0);
        })
        .unwrap();
    cx.run_until_parked();
}

fn reuse_frames(window: WindowHandle<AnimationRoot>, cx: &mut TestAppContext, samples: &Samples) {
    let rendered = samples.borrow().len();
    // Unrelated redraws must reuse the cached subtree, without advancing the
    // animation callback. Repeat to exercise the cache's rewritten ranges.
    for _ in 0..3 {
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap();
    }
    assert_eq!(samples.borrow().len(), rendered, "cache was not reused");
}

#[gpui::test]
fn cached_view_preserves_in_progress_animation_after_reuse(cx: &mut TestAppContext) {
    for nested in [false, true] {
        let (window, samples) =
            open_animation(cx, vec![Animation::new(Duration::from_secs(1))], nested);
        advance_frame(window, cx, Duration::from_millis(250));
        assert_sample(&samples, 0, 0.25);
        reuse_frames(window, cx, &samples);
        advance_frame(window, cx, Duration::from_millis(250));
        assert_sample(&samples, 0, 0.5);
    }
}

#[gpui::test]
fn cached_view_keeps_animation_state_when_bounds_or_text_style_change(cx: &mut TestAppContext) {
    for nested in [false, true] {
        for change_text_style in [false, true] {
            let (window, samples) =
                open_animation(cx, vec![Animation::new(Duration::from_secs(1))], nested);
            advance_frame(window, cx, Duration::from_millis(250));
            assert_sample(&samples, 0, 0.25);
            reuse_frames(window, cx, &samples);
            window
                .update(cx, |root, _, cx| {
                    if change_text_style {
                        root.font_size = px(18.);
                    } else {
                        root.padding = px(8.);
                    }
                    cx.notify();
                })
                .unwrap();
            cx.run_until_parked();
            assert_sample(&samples, 0, 0.25);
            reuse_frames(window, cx, &samples);
            advance_frame(window, cx, Duration::from_millis(250));
            assert_sample(&samples, 0, 0.5);
        }
    }
}

#[gpui::test]
fn cached_view_keeps_completed_mount_animation_after_reuse(cx: &mut TestAppContext) {
    for nested in [false, true] {
        let (window, samples) =
            open_animation(cx, vec![Animation::new(Duration::from_millis(120))], nested);
        advance_frame(window, cx, Duration::from_millis(150));
        assert_sample(&samples, 0, 1.0);
        reuse_frames(window, cx, &samples);
        window
            .update(cx, |root, _, cx| {
                root.animation.update(cx, |_, cx| cx.notify());
            })
            .unwrap();
        cx.run_until_parked();
        assert_sample(&samples, 0, 1.0);
        window
            .update(cx, |_, window, cx| {
                assert_eq!(window.simulate_next_frame(cx), 0);
            })
            .unwrap();
    }
}

#[gpui::test]
fn cached_view_keeps_toast_animation_phase_after_reuse(cx: &mut TestAppContext) {
    for nested in [false, true] {
        let (window, samples) = open_animation(
            cx,
            vec![
                Animation::new(Duration::from_millis(100)),
                Animation::new(Duration::from_secs(1)),
                Animation::new(Duration::from_millis(100)),
            ],
            nested,
        );
        advance_frame(window, cx, Duration::from_millis(150));
        advance_frame(window, cx, Duration::from_millis(300));
        assert_sample(&samples, 1, 0.3);
        reuse_frames(window, cx, &samples);
        advance_frame(window, cx, Duration::from_millis(200));
        assert_sample(&samples, 1, 0.5);
        advance_frame(window, cx, Duration::from_millis(700));
        advance_frame(window, cx, Duration::from_millis(50));
        assert_sample(&samples, 2, 0.5);
        reuse_frames(window, cx, &samples);
        advance_frame(window, cx, Duration::from_millis(25));
        assert_sample(&samples, 2, 0.75);
    }
}

#[gpui::test]
fn cached_view_restarts_mount_animation_only_after_unmount(cx: &mut TestAppContext) {
    for nested in [false, true] {
        let (window, samples) =
            open_animation(cx, vec![Animation::new(Duration::from_millis(120))], nested);
        advance_frame(window, cx, Duration::from_millis(150));
        assert_sample(&samples, 0, 1.0);
        reuse_frames(window, cx, &samples);
        window
            .update(cx, |root, _, cx| {
                root.mounted = false;
                cx.notify();
            })
            .unwrap();
        cx.run_until_parked();
        let rendered = samples.borrow().len();
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap();
        assert_eq!(samples.borrow().len(), rendered);
        window
            .update(cx, |root, _, cx| {
                root.mounted = true;
                cx.notify();
            })
            .unwrap();
        cx.run_until_parked();
        assert_sample(&samples, 0, 0.0);
        advance_frame(window, cx, Duration::from_millis(60));
        assert_sample(&samples, 0, 0.5);
    }
}

type RenderStates = Rc<RefCell<Vec<Entity<usize>>>>;
type LayoutSamples = Rc<RefCell<Vec<EarlyLayouts>>>;

struct EarlyLayouts {
    line: Arc<LineLayout>,
    text: Arc<ShapedTextLayout>,
    inline: Arc<InlineLayout>,
}

struct EarlyStateView {
    states: RenderStates,
    layouts: LayoutSamples,
}

impl Render for EarlyStateView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = window.use_state(cx, |_, _| 0usize);
        self.states.borrow_mut().push(state);
        let text: SharedString = "shaped during render".into();
        let runs = [TextRun {
            len: text.len(),
            ..Default::default()
        }];
        let line = window
            .text_system()
            .shape_line(text.clone(), px(14.), &runs)
            .layout;
        let shaped = window
            .text_system()
            .shape_text(text.clone(), px(14.), &runs, Some(px(40.)), None)
            .unwrap();
        let inline = window.text_system().layout_inline(InlineLayoutRequest {
            text: &text,
            runs: &runs,
            text_styles: &[],
            boxes: &[],
            font_size: px(14.),
            line_height: px(20.),
            text_metrics: Default::default(),
            options: TextLayoutOptions::default(),
            bidi_scopes: &[],
        });
        self.layouts.borrow_mut().push(EarlyLayouts {
            line,
            text: shaped.layout,
            inline,
        });
        div().size_full()
    }
}

fn render_after_cache_reuse(
    cx: &mut TestAppContext,
    nested: bool,
) -> (RenderStates, LayoutSamples) {
    let states = Rc::new(RefCell::new(Vec::new()));
    let layouts = Rc::new(RefCell::new(Vec::new()));
    let child = cx.new(|_| EarlyStateView {
        states: states.clone(),
        layouts: layouts.clone(),
    });
    let mut content = AnyView::from(child);
    if nested {
        content = cx.new(|_| CachedContainer { child: content }).into();
    }
    let window = cx.open_window(size(px(100.), px(100.)), move |_, _| CachedContainer {
        child: content,
    });
    cx.run_until_parked();
    let initial = states.borrow()[0].clone();
    initial.update(cx, |value, _| *value = 42);
    for _ in 0..3 {
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap();
    }
    assert_eq!(states.borrow().len(), 1, "cache was not reused");

    // State notifications must still invalidate their owning view after
    // reuse, and its render-time text must remain in the shaping cache.
    initial.update(cx, |_, cx| cx.notify());
    cx.run_until_parked();
    assert_eq!(states.borrow().len(), 2);
    (states, layouts)
}

#[gpui::test]
fn cached_view_preserves_render_state_and_notifications(cx: &mut TestAppContext) {
    for nested in [false, true] {
        let (states, _) = render_after_cache_reuse(cx, nested);
        let initial = states.borrow()[0].clone();
        let current = states.borrow().last().unwrap().clone();
        assert_eq!(current.entity_id(), initial.entity_id());
        assert_eq!(current.read_with(cx, |value, _| *value), 42);
    }
}

#[gpui::test]
fn cached_view_preserves_all_render_time_text_layouts(cx: &mut TestAppContext) {
    for nested in [false, true] {
        let (_, layouts) = render_after_cache_reuse(cx, nested);
        let layouts = layouts.borrow();
        assert!(
            Arc::ptr_eq(&layouts[0].line, &layouts[1].line),
            "line was reshaped"
        );
        assert!(
            Arc::ptr_eq(&layouts[0].text, &layouts[1].text),
            "text was reshaped"
        );
        assert!(
            Arc::ptr_eq(&layouts[0].inline, &layouts[1].inline),
            "inline text was reshaped"
        );
    }
}

struct DirectionView {
    text: SharedString,
    renders: Rc<RefCell<usize>>,
}

impl Render for DirectionView {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        *self.renders.borrow_mut() += 1;
        div().size_full().child(self.text.clone())
    }
}

struct DirectionRoot {
    child: Entity<DirectionView>,
    directions: Rc<RefCell<Vec<ResolvedDirection>>>,
}

impl Render for DirectionRoot {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let directions = self.directions.clone();
        div()
            .size_full()
            .direction(LayoutDirection::Auto)
            .child(
                self.child
                    .clone()
                    .cached(StyleRefinement::default().size_full()),
            )
            .on_children_prepainted(move |_, window, _| {
                directions.borrow_mut().push(window.resolved_direction());
            })
    }
}

#[gpui::test]
fn cached_view_preserves_auto_direction_and_updates_after_notify(cx: &mut TestAppContext) {
    let renders = Rc::new(RefCell::new(0));
    let directions = Rc::new(RefCell::new(Vec::new()));
    let child = cx.new(|_| DirectionView {
        text: "مرحبا".into(),
        renders: renders.clone(),
    });
    let window = cx.open_window(size(px(100.), px(100.)), |_, _| DirectionRoot {
        child: child.clone(),
        directions: directions.clone(),
    });
    cx.run_until_parked();
    for _ in 0..3 {
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap();
    }
    assert_eq!(*renders.borrow(), 1, "cache was not reused");
    assert!(
        directions
            .borrow()
            .iter()
            .all(|direction| *direction == ResolvedDirection::RightToLeft)
    );
    child.update(cx, |view, cx| {
        view.text = "English".into();
        cx.notify();
    });
    cx.run_until_parked();
    assert_eq!(*renders.borrow(), 2);
    assert_eq!(
        directions.borrow().last(),
        Some(&ResolvedDirection::LeftToRight)
    );
}

struct DeferredAnimationRoot {
    child: Entity<AnimatedView>,
}

impl Render for DeferredAnimationRoot {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(deferred(
            self.child
                .clone()
                .cached(StyleRefinement::default().size_full()),
        ))
    }
}

#[gpui::test]
fn cached_deferred_view_preserves_animation_after_reuse(cx: &mut TestAppContext) {
    let samples = Samples::default();
    let window = cx.open_window(size(px(100.), px(100.)), |_, cx| DeferredAnimationRoot {
        child: cx.new(|_| AnimatedView {
            animations: vec![Animation::new(Duration::from_secs(1))],
            samples: samples.clone(),
        }),
    });
    cx.run_until_parked();
    cx.executor().advance_clock(Duration::from_millis(250));
    window
        .update(cx, |_, window, cx| {
            assert!(window.simulate_next_frame(cx) > 0);
        })
        .unwrap();
    cx.run_until_parked();
    assert_sample(&samples, 0, 0.25);
    let rendered = samples.borrow().len();
    for _ in 0..3 {
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap();
    }
    assert_eq!(samples.borrow().len(), rendered, "cache was not reused");
    cx.executor().advance_clock(Duration::from_millis(250));
    window
        .update(cx, |_, window, cx| {
            assert!(window.simulate_next_frame(cx) > 0);
        })
        .unwrap();
    cx.run_until_parked();
    assert_sample(&samples, 0, 0.5);
}

struct WindowControlView {
    renders: Rc<RefCell<usize>>,
    actions: Rc<RefCell<usize>>,
}

impl Render for WindowControlView {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        *self.renders.borrow_mut() += 1;
        let actions = self.actions.clone();
        div()
            .id("close-button")
            .size_full()
            .role(accesskit::Role::Button)
            .window_control_area(WindowControlArea::Close)
            .on_a11y_action(accesskit::Action::Click, move |_, _, _| {
                *actions.borrow_mut() += 1;
            })
    }
}

struct WindowControlRoot {
    content: AnyView,
    mounted: bool,
    style: StyleRefinement,
}

impl Render for WindowControlRoot {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .child(
                div()
                    .id("uncached-control")
                    .absolute()
                    .size(px(10.))
                    .window_control_area(WindowControlArea::Min),
            )
            .when(self.mounted, |element| {
                element.child(self.content.clone().cached(self.style.clone()))
            })
    }
}

#[gpui::test]
fn cached_view_preserves_window_control_hitboxes(cx: &mut TestAppContext) {
    for nested in [false, true] {
        let renders = Rc::new(RefCell::new(0));
        let child = cx.new(|_| WindowControlView {
            renders: renders.clone(),
            actions: Default::default(),
        });
        let mut content = AnyView::from(child);
        if nested {
            content = cx.new(|_| CachedContainer { child: content }).into();
        }
        let window = cx.open_window(size(px(100.), px(100.)), move |_, _| WindowControlRoot {
            content,
            mounted: true,
            style: StyleRefinement::default().size_full(),
        });
        cx.run_until_parked();
        for frame in 0..4 {
            cx.update_window(window.into(), |_, window, cx| {
                if frame > 0 {
                    window.draw(cx).clear(cx);
                }
                let controls = &window.rendered_frame.window_control_hitboxes;
                assert_eq!(
                    controls.len(),
                    2,
                    "window control lost or duplicated on frame {frame}"
                );
                assert_eq!(controls[0].0, WindowControlArea::Min);
                assert_eq!(controls[1].0, WindowControlArea::Close);
                assert_eq!(controls[1].1.bounds.size, size(px(100.), px(100.)));
            })
            .unwrap();
        }
        assert_eq!(*renders.borrow(), 1, "cache was not reused");
        window
            .update(cx, |root, _, cx| {
                root.mounted = false;
                cx.notify();
            })
            .unwrap();
        cx.run_until_parked();
        cx.update_window(window.into(), |_, window, _| {
            let controls = &window.rendered_frame.window_control_hitboxes;
            assert_eq!(controls.len(), 1, "removed control survived unmount");
            assert_eq!(controls[0].0, WindowControlArea::Min);
        })
        .unwrap();
        window
            .update(cx, |root, _, cx| {
                root.mounted = true;
                cx.notify();
            })
            .unwrap();
        cx.run_until_parked();
        cx.update_window(window.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
            assert_eq!(window.rendered_frame.window_control_hitboxes.len(), 2);
        })
        .unwrap();
        assert_eq!(*renders.borrow(), 2);
    }
}

#[gpui::test]
fn cached_view_preserves_accessibility_nodes(cx: &mut TestAppContext) {
    for nested in [false, true] {
        let renders = Rc::new(RefCell::new(0));
        let actions = Rc::new(RefCell::new(0));
        let child = cx.new(|_| WindowControlView {
            renders: renders.clone(),
            actions: actions.clone(),
        });
        let mut content = AnyView::from(child);
        if nested {
            content = cx.new(|_| CachedContainer { child: content }).into();
        }
        let window = cx.open_window(size(px(100.), px(100.)), |_, _| WindowControlRoot {
            content,
            mounted: true,
            style: StyleRefinement::default().size_full(),
        });
        window
            .update(cx, |_, window, _| window.set_a11y_forced(true))
            .unwrap();
        cx.run_until_parked();
        for frame in 0..4 {
            cx.update_window(window.into(), |_, window, cx| {
                if frame > 0 {
                    window.draw(cx).clear(cx);
                }
                let buttons: Vec<_> = window
                    .a11y_tree()
                    .unwrap()
                    .nodes
                    .iter()
                    .filter(|(_, node)| node.role() == accesskit::Role::Button)
                    .map(|(id, _)| *id)
                    .collect();
                assert_eq!(buttons.len(), 1, "accessible button lost on frame {frame}");
                assert_eq!(
                    window.a11y_node_bounds(buttons[0]).unwrap().size,
                    size(px(100.), px(100.))
                );
                assert!(window.a11y.action_listeners.contains_key(&buttons[0]));
                #[cfg(not(target_family = "wasm"))]
                window.perform_a11y_action(buttons[0], accesskit::Action::Click, None, cx);
            })
            .unwrap();
        }
        #[cfg(not(target_family = "wasm"))]
        assert_eq!(
            *actions.borrow(),
            4,
            "accessible action was lost after redraw"
        );

        window
            .update(cx, |_, window, _| window.set_a11y_forced(false))
            .unwrap();
        cx.run_until_parked();
        let rendered = *renders.borrow();
        for _ in 0..3 {
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
        }
        assert_eq!(
            *renders.borrow(),
            rendered,
            "cache did not resume after accessibility was disabled"
        );
    }
}

#[gpui::test]
fn cached_view_preserves_fixed_size_while_accessibility_is_active(cx: &mut TestAppContext) {
    let child = cx.new(|_| WindowControlView {
        renders: Default::default(),
        actions: Default::default(),
    });
    let window = cx.open_window(size(px(100.), px(100.)), |_, _| WindowControlRoot {
        content: child.into(),
        mounted: true,
        style: StyleRefinement::default().w(px(40.)).h(px(20.)),
    });
    cx.run_until_parked();
    for active in [false, true, false] {
        window
            .update(cx, |_, window, _| window.set_a11y_forced(active))
            .unwrap();
        cx.run_until_parked();
        for _ in 0..3 {
            cx.update_window(window.into(), |_, window, cx| {
                window.draw(cx).clear(cx);
                let controls = &window.rendered_frame.window_control_hitboxes;
                assert_eq!(controls.len(), 2);
                assert_eq!(controls[1].1.bounds.size, size(px(40.), px(20.)));
            })
            .unwrap();
        }
    }
}
