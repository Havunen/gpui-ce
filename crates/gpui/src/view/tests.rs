use super::*;
use crate::{Animation, AnimationExt, TestAppContext, WindowHandle, div, prelude::*, px, size};
use std::{cell::RefCell, rc::Rc, time::Duration};

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
}

impl Render for AnimationRoot {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().size_full().when(self.mounted, |element| {
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
