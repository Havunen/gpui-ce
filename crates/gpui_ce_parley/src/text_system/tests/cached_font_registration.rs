use super::*;

struct CachedTextView {
    renders: Rc<Cell<usize>>,
}

impl Render for CachedTextView {
    fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        window.text_system().layout_line(
            "retained text",
            px(14.),
            &[text_run("retained text", IBM_PLEX.family)],
        );
        div().size_full().child("retained text")
    }
}

struct CachedTextRoot {
    content: gpui::AnyView,
    shape_sibling: bool,
}

impl Render for CachedTextRoot {
    fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        if self.shape_sibling {
            window.text_system().layout_line(
                "uncached sibling",
                px(14.),
                &[text_run("uncached sibling", IBM_PLEX.family)],
            );
        }
        div().size_full().child(
            self.content
                .clone()
                .cached(gpui::StyleRefinement::default().size_full()),
        )
    }
}

fn check_font_registration(shape_sibling: bool) {
    for nested in [false, true] {
        let mut cx = HeadlessAppContext::new(test_system());
        let renders = Rc::new(Cell::new(0));
        let window = cx
            .open_window(size(px(100.), px(100.)), |_, cx| {
                let mut content = gpui::AnyView::from(cx.new(|_| CachedTextView {
                    renders: renders.clone(),
                }));
                if nested {
                    content = cx
                        .new(|_| CachedTextRoot {
                            content,
                            shape_sibling: false,
                        })
                        .into();
                }
                cx.new(|_| CachedTextRoot {
                    content,
                    shape_sibling,
                })
            })
            .unwrap();
        cx.run_until_parked();
        let draw = |cx: &mut HeadlessAppContext| {
            cx.update(|cx| {
                cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                    .unwrap();
            });
        };
        for _ in 0..3 {
            draw(&mut cx);
        }
        assert_eq!(renders.get(), 1, "cache was not reused");

        // Use the real backend's registration path to invalidate its font
        // collection, without notifying the cached entity or refreshing the window.
        cx.update(|cx| {
            cx.text_system()
                .add_fonts(vec![Cow::Borrowed(IBM_PLEX.data)])
                .unwrap()
        });
        draw(&mut cx);
        assert_eq!(renders.get(), 2, "cached text did not observe new fonts");
        draw(&mut cx);
        assert_eq!(renders.get(), 2, "cache did not resume after font change");
    }
}

#[test]
fn font_registration_invalidates_cached_text_views() {
    check_font_registration(false);
}

#[test]
fn font_registration_before_cached_sibling_keeps_layout_ranges_valid() {
    check_font_registration(true);
}

struct LateFontRoot {
    content: gpui::Entity<CachedTextRoot>,
    register: Rc<Cell<bool>>,
}

impl Render for LateFontRoot {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let register = self.register.clone();
        div()
            .size_full()
            .child(self.content.clone())
            .on_children_prepainted(move |_, window, _| {
                if register.replace(false) {
                    window
                        .text_system()
                        .add_fonts(vec![Cow::Borrowed(IBM_PLEX.data)])
                        .unwrap();
                    window.text_system().layout_line(
                        "after registration",
                        px(14.),
                        &[text_run("after registration", IBM_PLEX.family)],
                    );
                }
            })
    }
}

#[test]
fn font_registration_during_prepaint_keeps_cached_paint_ranges_valid() {
    let mut cx = HeadlessAppContext::new(test_system());
    let renders = Rc::new(Cell::new(0));
    let register = Rc::new(Cell::new(false));
    let window = cx
        .open_window(size(px(100.), px(100.)), |_, cx| {
            let child = cx.new(|_| CachedTextView {
                renders: renders.clone(),
            });
            let content = cx.new(|_| CachedTextRoot {
                content: child.into(),
                shape_sibling: false,
            });
            cx.new(|_| LateFontRoot {
                content,
                register: register.clone(),
            })
        })
        .unwrap();
    cx.run_until_parked();
    register.set(true);
    for _ in 0..2 {
        cx.update(|cx| {
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
        });
    }
    assert_eq!(renders.get(), 2, "cached text did not observe new fonts");
}
