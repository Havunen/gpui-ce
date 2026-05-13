use crate::{
    ActiveTooltip, AnyView, App, Bounds, DispatchPhase, Element, ElementId, GlobalElementId,
    HighlightStyle, Hitbox, HitboxBehavior, InspectorElementId, IntoElement, LayoutId,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, Point, SharedString, Size, TextOverflow,
    TextRun, TextStyle, TooltipId, TruncateFrom, WhiteSpace, Window, WrappedLine,
    WrappedLineLayout, point, px, register_tooltip_mouse_handlers, set_tooltip_on_window,
};
use anyhow::Context as _;
use itertools::Itertools;
use smallvec::SmallVec;
use std::{
    borrow::Cow,
    cell::{Cell, RefCell},
    mem,
    ops::Range,
    rc::Rc,
    sync::Arc,
};
use util::ResultExt;

impl Element for &'static str {
    type RequestLayoutState = TextLayout;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut state = TextLayout::default();
        let layout_id = state.layout(SharedString::from(*self), None, window, cx);
        (layout_id, state)
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        text_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
        text_layout.prepaint(bounds, self)
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        text_layout: &mut TextLayout,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        text_layout.paint(self, window, cx)
    }
}

impl IntoElement for &'static str {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl IntoElement for String {
    type Element = SharedString;

    fn into_element(self) -> Self::Element {
        self.into()
    }
}

impl IntoElement for Cow<'static, str> {
    type Element = SharedString;

    fn into_element(self) -> Self::Element {
        self.into()
    }
}

impl Element for SharedString {
    type RequestLayoutState = TextLayout;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut state = TextLayout::default();
        let layout_id = state.layout(self.clone(), None, window, cx);
        (layout_id, state)
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        text_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
        text_layout.prepaint(bounds, self.as_ref())
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        text_layout: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        text_layout.paint(self.as_ref(), window, cx)
    }
}

impl IntoElement for SharedString {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// Renders text with runs of different styles.
///
/// Callers are responsible for setting the correct style for each run.
/// For text with a uniform style, you can usually avoid calling this constructor
/// and just pass text directly.
pub struct StyledText {
    text: SharedString,
    runs: Option<Vec<TextRun>>,
    delayed_highlights: Option<Vec<(Range<usize>, HighlightStyle)>>,
    layout: TextLayout,
}

impl StyledText {
    /// Construct a new styled text element from the given string.
    pub fn new(text: impl Into<SharedString>) -> Self {
        StyledText {
            text: text.into(),
            runs: None,
            delayed_highlights: None,
            layout: TextLayout::default(),
        }
    }

    /// Get the layout for this element. This can be used to map indices to pixels and vice versa.
    pub fn layout(&self) -> &TextLayout {
        &self.layout
    }

    /// Set the styling attributes for the given text, as well as
    /// as any ranges of text that have had their style customized.
    pub fn with_default_highlights(
        mut self,
        default_style: &TextStyle,
        highlights: impl IntoIterator<Item = (Range<usize>, HighlightStyle)>,
    ) -> Self {
        debug_assert!(
            self.delayed_highlights.is_none(),
            "Can't use `with_default_highlights` and `with_highlights`"
        );
        let runs = Self::compute_runs(&self.text, default_style, highlights);
        self.with_runs(runs)
    }

    /// Set the styling attributes for the given text, as well as
    /// as any ranges of text that have had their style customized.
    pub fn with_highlights(
        mut self,
        highlights: impl IntoIterator<Item = (Range<usize>, HighlightStyle)>,
    ) -> Self {
        debug_assert!(
            self.runs.is_none(),
            "Can't use `with_highlights` and `with_default_highlights`"
        );
        self.delayed_highlights = Some(
            highlights
                .into_iter()
                .inspect(|(run, _)| {
                    debug_assert!(self.text.is_char_boundary(run.start));
                    debug_assert!(self.text.is_char_boundary(run.end));
                })
                .collect::<Vec<_>>(),
        );
        self
    }

    fn compute_runs(
        text: &str,
        default_style: &TextStyle,
        highlights: impl IntoIterator<Item = (Range<usize>, HighlightStyle)>,
    ) -> Vec<TextRun> {
        let highlights = highlights.into_iter();
        let mut runs = Vec::with_capacity(highlights.size_hint().0 * 2 + 1);
        let default_run = default_style.to_run(0);
        let mut ix = 0;
        for (range, highlight) in highlights {
            if ix < range.start {
                debug_assert!(text.is_char_boundary(range.start));
                let mut run = default_run.clone();
                run.len = range.start - ix;
                runs.push(run);
            }
            debug_assert!(text.is_char_boundary(range.end));
            runs.push(
                default_style
                    .clone()
                    .highlight(highlight)
                    .to_run(range.len()),
            );
            ix = range.end;
        }
        if ix < text.len() {
            let mut run = default_run;
            run.len = text.len() - ix;
            runs.push(run);
        }
        runs
    }

    /// Set the text runs for this piece of text.
    pub fn with_runs(mut self, runs: Vec<TextRun>) -> Self {
        let mut text = &**self.text;
        for run in &runs {
            text = text.get(run.len..).unwrap_or_else(|| {
                #[cfg(debug_assertions)]
                panic!("invalid text run. Text: '{text}', run: {run:?}");
                #[cfg(not(debug_assertions))]
                panic!("invalid text run");
            });
        }
        assert!(text.is_empty(), "invalid text run");
        self.runs = Some(runs);
        self
    }
}

impl Element for StyledText {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let runs = self.runs.take().or_else(|| {
            self.delayed_highlights.take().map(|delayed_highlights| {
                Self::compute_runs(&self.text, &window.text_style(), delayed_highlights)
            })
        });

        let layout_id = self.layout.layout(self.text.clone(), runs, window, cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
        self.layout.prepaint(bounds, &self.text)
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.layout.paint(&self.text, window, cx)
    }
}

impl IntoElement for StyledText {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// The Layout for TextElement. This can be used to map indices to pixels and vice versa.
#[derive(Default, Clone)]
pub struct TextLayout(Rc<RefCell<Option<TextLayoutInner>>>);

struct TextLayoutInner {
    len: usize,
    lines: SmallVec<[WrappedLine; 1]>,
    line_metrics: SmallVec<[TextLayoutLineMetric; 1]>,
    line_height: Pixels,
    wrap_width: Option<Pixels>,
    size: Option<Size<Pixels>>,
    bounds: Option<Bounds<Pixels>>,
}

#[derive(Clone, Copy, Debug)]
struct TextLayoutLineMetric {
    start_ix: usize,
    end_ix: usize,
    origin_y: Pixels,
    height: Pixels,
}

fn text_layout_line_metrics(
    lines: &SmallVec<[WrappedLine; 1]>,
    line_height: Pixels,
) -> SmallVec<[TextLayoutLineMetric; 1]> {
    let mut metrics = SmallVec::new();
    let mut start_ix = 0;
    let mut origin_y = px(0.);

    for line in lines {
        let height = line.size(line_height).height;
        let end_ix = start_ix + line.len();
        metrics.push(TextLayoutLineMetric {
            start_ix,
            end_ix,
            origin_y,
            height,
        });
        start_ix = end_ix + 1;
        origin_y += height;
    }

    metrics
}

impl TextLayout {
    fn layout(
        &self,
        text: SharedString,
        runs: Option<Vec<TextRun>>,
        window: &mut Window,
        _: &mut App,
    ) -> LayoutId {
        let text_style = window.text_style();
        let font_size = text_style.font_size.to_pixels(window.rem_size());
        let line_height = text_style
            .line_height
            .to_pixels(font_size.into(), window.rem_size());

        let runs = if let Some(runs) = runs {
            runs
        } else {
            vec![text_style.to_run(text.len())]
        };
        window.request_measured_layout(Default::default(), {
            let element_state = self.clone();

            move |known_dimensions, available_space, window, cx| {
                let wrap_width = if text_style.white_space == WhiteSpace::Normal {
                    known_dimensions.width.or(match available_space.width {
                        crate::AvailableSpace::Definite(x) => Some(x),
                        _ => None,
                    })
                } else {
                    None
                };

                let (truncate_width, truncation_affix, truncate_from) =
                    if let Some(text_overflow) = text_style.text_overflow.clone() {
                        let width = known_dimensions.width.or(match available_space.width {
                            crate::AvailableSpace::Definite(x) => match text_style.line_clamp {
                                Some(max_lines) => Some(x * max_lines),
                                None => Some(x),
                            },
                            _ => None,
                        });

                        match text_overflow {
                            TextOverflow::Truncate(s) => (width, s, TruncateFrom::End),
                            TextOverflow::TruncateStart(s) => (width, s, TruncateFrom::Start),
                        }
                    } else {
                        (None, "".into(), TruncateFrom::End)
                    };

                // Only use cached layout if:
                // 1. We have a cached size
                // 2. wrap_width matches (or both are None)
                // 3. truncate_width is None (if truncate_width is Some, we need to re-layout
                //    because the previous layout may have been computed without truncation)
                if let Some(text_layout) = element_state.0.borrow().as_ref()
                    && let Some(size) = text_layout.size
                    && (wrap_width.is_none() || wrap_width == text_layout.wrap_width)
                    && truncate_width.is_none()
                {
                    return size;
                }

                let mut line_wrapper = cx.text_system().line_wrapper(text_style.font(), font_size);
                let (text, runs) = if let Some(truncate_width) = truncate_width {
                    line_wrapper.truncate_line(
                        text.clone(),
                        truncate_width,
                        &truncation_affix,
                        &runs,
                        truncate_from,
                    )
                } else {
                    (text.clone(), Cow::Borrowed(&*runs))
                };
                let len = text.len();

                let Some(lines) = window
                    .text_system()
                    .shape_text(
                        text,
                        font_size,
                        &runs,
                        wrap_width,            // Wrap if we know the width.
                        text_style.line_clamp, // Limit the number of lines if line_clamp is set.
                    )
                    .log_err()
                else {
                    element_state.0.borrow_mut().replace(TextLayoutInner {
                        lines: Default::default(),
                        line_metrics: Default::default(),
                        len: 0,
                        line_height,
                        wrap_width,
                        size: Some(Size::default()),
                        bounds: None,
                    });
                    return Size::default();
                };

                let mut size: Size<Pixels> = Size::default();
                let mut line_metrics = SmallVec::new();
                let mut line_start_ix = 0;
                let mut line_origin_y = px(0.);
                for line in &lines {
                    let line_size = line.size(line_height);
                    let line_end_ix = line_start_ix + line.len();
                    line_metrics.push(TextLayoutLineMetric {
                        start_ix: line_start_ix,
                        end_ix: line_end_ix,
                        origin_y: line_origin_y,
                        height: line_size.height,
                    });
                    size.height += line_size.height;
                    size.width = size.width.max(line_size.width).ceil();
                    line_start_ix = line_end_ix + 1;
                    line_origin_y += line_size.height;
                }

                element_state.0.borrow_mut().replace(TextLayoutInner {
                    lines,
                    line_metrics,
                    len,
                    line_height,
                    wrap_width,
                    size: Some(size),
                    bounds: None,
                });

                size
            }
        })
    }

    fn prepaint(&self, bounds: Bounds<Pixels>, text: &str) {
        let mut element_state = self.0.borrow_mut();
        let element_state = element_state
            .as_mut()
            .with_context(|| format!("measurement has not been performed on {text}"))
            .unwrap();
        element_state.bounds = Some(bounds);
    }

    fn paint(&self, text: &str, window: &mut Window, cx: &mut App) {
        let element_state = self.0.borrow();
        let element_state = element_state
            .as_ref()
            .with_context(|| format!("measurement has not been performed on {text}"))
            .unwrap();
        let bounds = element_state
            .bounds
            .with_context(|| format!("prepaint has not been performed on {text}"))
            .unwrap();

        let line_height = element_state.line_height;
        let mut line_origin = bounds.origin;
        let text_style = window.text_style();
        for line in &element_state.lines {
            line.paint_background(
                line_origin,
                line_height,
                text_style.text_align,
                Some(bounds),
                window,
                cx,
            )
            .log_err();
            line.paint(
                line_origin,
                line_height,
                text_style.text_align,
                Some(bounds),
                window,
                cx,
            )
            .log_err();
            line_origin.y += line.size(line_height).height;
        }
    }

    /// Get the byte index into the input of the pixel position.
    pub fn index_for_position(&self, position: Point<Pixels>) -> Result<usize, usize> {
        let element_state = self.0.borrow();
        let element_state = element_state
            .as_ref()
            .expect("measurement has not been performed");
        let bounds = element_state
            .bounds
            .expect("prepaint has not been performed");

        if position.y < bounds.top() {
            return Err(0);
        }

        let line_height = element_state.line_height;
        let y = position.y - bounds.origin.y;
        let line_ix = element_state
            .line_metrics
            .partition_point(|metric| y > metric.origin_y + metric.height);
        let Some(metric) = element_state.line_metrics.get(line_ix) else {
            return Err(element_state
                .line_metrics
                .last()
                .map_or(0, |metric| metric.end_ix));
        };

        let line_origin = point(bounds.origin.x, bounds.origin.y + metric.origin_y);
        let position_within_line = position - line_origin;
        match element_state.lines[line_ix].index_for_position(position_within_line, line_height) {
            Ok(index_within_line) => Ok(metric.start_ix + index_within_line),
            Err(index_within_line) => Err(metric.start_ix + index_within_line),
        }
    }

    /// Get the pixel position for the given byte index.
    pub fn position_for_index(&self, index: usize) -> Option<Point<Pixels>> {
        let element_state = self.0.borrow();
        let element_state = element_state
            .as_ref()
            .expect("measurement has not been performed");
        let bounds = element_state
            .bounds
            .expect("prepaint has not been performed");
        let line_ix = element_state
            .line_metrics
            .partition_point(|metric| index > metric.end_ix);
        let metric = element_state.line_metrics.get(line_ix)?;
        if index < metric.start_ix {
            return None;
        }

        let line_origin = point(bounds.origin.x, bounds.origin.y + metric.origin_y);
        let ix_within_line = index - metric.start_ix;
        Some(
            line_origin
                + element_state.lines[line_ix]
                    .position_for_index(ix_within_line, element_state.line_height)?,
        )
    }

    /// Retrieve the layout for the line containing the given byte index.
    pub fn line_layout_for_index(&self, index: usize) -> Option<Arc<WrappedLineLayout>> {
        let element_state = self.0.borrow();
        let element_state = element_state
            .as_ref()
            .expect("measurement has not been performed");
        let _ = element_state
            .bounds
            .expect("prepaint has not been performed");
        let line_ix = element_state
            .line_metrics
            .partition_point(|metric| index > metric.end_ix);
        let metric = element_state.line_metrics.get(line_ix)?;
        if index < metric.start_ix {
            return None;
        }

        Some(element_state.lines[line_ix].layout.clone())
    }

    /// The bounds of this layout.
    pub fn bounds(&self) -> Bounds<Pixels> {
        self.0.borrow().as_ref().unwrap().bounds.unwrap()
    }

    /// The line height for this layout.
    pub fn line_height(&self) -> Pixels {
        self.0.borrow().as_ref().unwrap().line_height
    }

    /// The UTF-8 length of the underlying text.
    pub fn len(&self) -> usize {
        self.0.borrow().as_ref().unwrap().len
    }

    /// The text for this layout.
    pub fn text(&self) -> String {
        self.0
            .borrow()
            .as_ref()
            .unwrap()
            .lines
            .iter()
            .map(|s| &s.text)
            .join("\n")
    }

    /// The text for this layout (with soft-wraps as newlines)
    pub fn wrapped_text(&self) -> String {
        let mut accumulator = String::new();

        for wrapped in self.0.borrow().as_ref().unwrap().lines.iter() {
            let mut seen = 0;
            for boundary in wrapped.layout.wrap_boundaries.iter() {
                let index = wrapped.layout.unwrapped_layout.runs[boundary.run_ix].glyphs
                    [boundary.glyph_ix]
                    .index;

                accumulator.push_str(&wrapped.text[seen..index]);
                accumulator.push('\n');
                seen = index;
            }
            accumulator.push_str(&wrapped.text[seen..]);
            accumulator.push('\n');
        }
        // Remove trailing newline
        accumulator.pop();
        accumulator
    }
}

/// A text element that can be interacted with.
pub struct InteractiveText {
    element_id: ElementId,
    text: StyledText,
    click_listener:
        Option<Box<dyn Fn(&[Range<usize>], InteractiveTextClickEvent, &mut Window, &mut App)>>,
    hover_listener: Option<Box<dyn Fn(Option<usize>, MouseMoveEvent, &mut Window, &mut App)>>,
    tooltip_builder: Option<Rc<dyn Fn(usize, &mut Window, &mut App) -> Option<AnyView>>>,
    tooltip_id: Option<TooltipId>,
    clickable_ranges: Vec<Range<usize>>,
}

struct InteractiveTextClickEvent {
    mouse_down_index: usize,
    mouse_up_index: usize,
}

#[doc(hidden)]
#[derive(Default)]
pub struct InteractiveTextState {
    mouse_down_index: Rc<Cell<Option<usize>>>,
    hovered_index: Rc<Cell<Option<usize>>>,
    active_tooltip: Rc<RefCell<Option<ActiveTooltip>>>,
}

/// InteractiveTest is a wrapper around StyledText that adds mouse interactions.
impl InteractiveText {
    /// Creates a new InteractiveText from the given text.
    pub fn new(id: impl Into<ElementId>, text: StyledText) -> Self {
        Self {
            element_id: id.into(),
            text,
            click_listener: None,
            hover_listener: None,
            tooltip_builder: None,
            tooltip_id: None,
            clickable_ranges: Vec::new(),
        }
    }

    /// on_click is called when the user clicks on one of the given ranges, passing the index of
    /// the clicked range.
    pub fn on_click(
        mut self,
        ranges: Vec<Range<usize>>,
        listener: impl Fn(usize, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.click_listener = Some(Box::new(move |ranges, event, window, cx| {
            for (range_ix, range) in ranges.iter().enumerate() {
                if range.contains(&event.mouse_down_index) && range.contains(&event.mouse_up_index)
                {
                    listener(range_ix, window, cx);
                }
            }
        }));
        self.clickable_ranges = ranges;
        self
    }

    /// on_hover is called when the mouse moves over a character within the text, passing the
    /// index of the hovered character, or None if the mouse leaves the text.
    pub fn on_hover(
        mut self,
        listener: impl Fn(Option<usize>, MouseMoveEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.hover_listener = Some(Box::new(listener));
        self
    }

    /// tooltip lets you specify a tooltip for a given character index in the string.
    pub fn tooltip(
        mut self,
        builder: impl Fn(usize, &mut Window, &mut App) -> Option<AnyView> + 'static,
    ) -> Self {
        self.tooltip_builder = Some(Rc::new(builder));
        self
    }
}

impl Element for InteractiveText {
    type RequestLayoutState = ();
    type PrepaintState = Hitbox;

    fn id(&self) -> Option<ElementId> {
        Some(self.element_id.clone())
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        self.text.request_layout(None, inspector_id, window, cx)
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        state: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Hitbox {
        window.with_optional_element_state::<InteractiveTextState, _>(
            global_id,
            |interactive_state, window| {
                let mut interactive_state = interactive_state
                    .map(|interactive_state| interactive_state.unwrap_or_default());

                if let Some(interactive_state) = interactive_state.as_mut() {
                    if self.tooltip_builder.is_some() {
                        self.tooltip_id =
                            set_tooltip_on_window(&interactive_state.active_tooltip, window);
                    } else {
                        // If there is no longer a tooltip builder, remove the active tooltip.
                        interactive_state.active_tooltip.take();
                    }
                }

                self.text
                    .prepaint(None, inspector_id, bounds, state, window, cx);
                let hitbox = window.insert_hitbox(bounds, HitboxBehavior::Normal);
                (hitbox, interactive_state)
            },
        )
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        hitbox: &mut Hitbox,
        window: &mut Window,
        cx: &mut App,
    ) {
        let current_view = window.current_view();
        let text_layout = self.text.layout().clone();
        window.with_element_state::<InteractiveTextState, _>(
            global_id.unwrap(),
            |interactive_state, window| {
                let mut interactive_state = interactive_state.unwrap_or_default();
                if let Some(click_listener) = self.click_listener.take() {
                    let mouse_position = window.mouse_position();
                    if let Ok(ix) = text_layout.index_for_position(mouse_position)
                        && self
                            .clickable_ranges
                            .iter()
                            .any(|range| range.contains(&ix))
                    {
                        window.set_cursor_style(crate::CursorStyle::PointingHand, hitbox)
                    }

                    let text_layout = text_layout.clone();
                    let mouse_down = interactive_state.mouse_down_index.clone();
                    if let Some(mouse_down_index) = mouse_down.get() {
                        let hitbox = hitbox.clone();
                        let clickable_ranges = mem::take(&mut self.clickable_ranges);
                        window.on_mouse_event(
                            move |event: &MouseUpEvent, phase, window: &mut Window, cx| {
                                if phase == DispatchPhase::Bubble && hitbox.is_hovered(window) {
                                    if let Ok(mouse_up_index) =
                                        text_layout.index_for_position(event.position)
                                    {
                                        click_listener(
                                            &clickable_ranges,
                                            InteractiveTextClickEvent {
                                                mouse_down_index,
                                                mouse_up_index,
                                            },
                                            window,
                                            cx,
                                        )
                                    }

                                    mouse_down.take();
                                    window.refresh();
                                }
                            },
                        );
                    } else {
                        let hitbox = hitbox.clone();
                        window.on_mouse_event(move |event: &MouseDownEvent, phase, window, _| {
                            if phase == DispatchPhase::Bubble
                                && hitbox.is_hovered(window)
                                && let Ok(mouse_down_index) =
                                    text_layout.index_for_position(event.position)
                            {
                                mouse_down.set(Some(mouse_down_index));
                                window.refresh();
                            }
                        });
                    }
                }

                window.on_mouse_event({
                    let mut hover_listener = self.hover_listener.take();
                    let hitbox = hitbox.clone();
                    let text_layout = text_layout.clone();
                    let hovered_index = interactive_state.hovered_index.clone();
                    move |event: &MouseMoveEvent, phase, window, cx| {
                        if phase == DispatchPhase::Bubble && hitbox.is_hovered(window) {
                            let current = hovered_index.get();
                            let updated = text_layout.index_for_position(event.position).ok();
                            if current != updated {
                                hovered_index.set(updated);
                                if let Some(hover_listener) = hover_listener.as_ref() {
                                    hover_listener(updated, event.clone(), window, cx);
                                }
                                cx.notify(current_view);
                            }
                        }
                    }
                });

                if let Some(tooltip_builder) = self.tooltip_builder.clone() {
                    let active_tooltip = interactive_state.active_tooltip.clone();
                    let build_tooltip = Rc::new({
                        let tooltip_is_hoverable = false;
                        let text_layout = text_layout.clone();
                        move |window: &mut Window, cx: &mut App| {
                            text_layout
                                .index_for_position(window.mouse_position())
                                .ok()
                                .and_then(|position| tooltip_builder(position, window, cx))
                                .map(|view| (view, tooltip_is_hoverable))
                        }
                    });

                    // Use bounds instead of testing hitbox since this is called during prepaint.
                    let check_is_hovered_during_prepaint = Rc::new({
                        let source_bounds = hitbox.bounds;
                        let text_layout = text_layout.clone();
                        let pending_mouse_down = interactive_state.mouse_down_index.clone();
                        move |window: &Window| {
                            text_layout
                                .index_for_position(window.mouse_position())
                                .is_ok()
                                && source_bounds.contains(&window.mouse_position())
                                && pending_mouse_down.get().is_none()
                        }
                    });

                    let check_is_hovered = Rc::new({
                        let hitbox = hitbox.clone();
                        let text_layout = text_layout.clone();
                        let pending_mouse_down = interactive_state.mouse_down_index.clone();
                        move |window: &Window| {
                            text_layout
                                .index_for_position(window.mouse_position())
                                .is_ok()
                                && hitbox.is_hovered(window)
                                && pending_mouse_down.get().is_none()
                        }
                    });

                    register_tooltip_mouse_handlers(
                        &active_tooltip,
                        self.tooltip_id,
                        build_tooltip,
                        check_is_hovered,
                        check_is_hovered_during_prepaint,
                        window,
                    );
                }

                self.text
                    .paint(None, inspector_id, bounds, &mut (), &mut (), window, cx);

                ((), interactive_state)
            },
        );
    }
}

impl IntoElement for InteractiveText {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FontId, FontWeight, GlyphId, HighlightStyle, LineLayout, ShapedGlyph, ShapedRun,
        SharedString, TextStyle, WrapBoundary, WrappedLine, WrappedLineLayout, point, px,
    };
    use smallvec::SmallVec;
    use std::{hint::black_box, sync::Arc};
    use util_macros::perf;

    #[test]
    fn test_into_element_for() {
        use crate::{ParentElement as _, SharedString, div};
        use std::borrow::Cow;

        let _ = div().child("static str");
        let _ = div().child("String".to_string());
        let _ = div().child(Cow::Borrowed("Cow"));
        let _ = div().child(SharedString::from("SharedString"));
    }

    #[perf(important)]
    fn perf_compute_runs_many_highlights() {
        const HIGHLIGHTS: usize = 4_096;

        let mut text = String::with_capacity(HIGHLIGHTS * 2);
        for _ in 0..HIGHLIGHTS {
            text.push_str("ab");
        }

        let highlights = (0..HIGHLIGHTS)
            .map(|ix| {
                let start = ix * 2;
                let style: HighlightStyle = if ix % 2 == 0 {
                    FontWeight::BOLD.into()
                } else {
                    HighlightStyle {
                        color: Some(crate::red()),
                        ..Default::default()
                    }
                };
                (start..start + 1, style)
            })
            .collect::<Vec<_>>();

        let runs = StyledText::compute_runs(&text, &TextStyle::default(), highlights);

        assert_eq!(runs.iter().map(|run| run.len).sum::<usize>(), text.len());
        black_box(runs.len());
    }

    #[perf(important)]
    fn perf_wrapped_text_many_soft_wraps() {
        const LINE_COUNT: usize = 256;
        const LINE_LEN: usize = 128;
        const WRAP_STEP: usize = 8;

        let layout = TextLayout::default();
        let mut lines = SmallVec::<[WrappedLine; 1]>::new();

        for line_ix in 0..LINE_COUNT {
            let text = SharedString::from("x".repeat(LINE_LEN));
            let glyphs = (0..LINE_LEN)
                .map(|ix| ShapedGlyph {
                    id: GlyphId((line_ix + ix) as u32),
                    position: point(px(ix as f32), px(0.0)),
                    index: ix,
                    is_emoji: false,
                })
                .collect::<Vec<_>>();
            let wrap_boundaries = (WRAP_STEP..LINE_LEN)
                .step_by(WRAP_STEP)
                .map(|glyph_ix| WrapBoundary {
                    run_ix: 0,
                    glyph_ix,
                })
                .collect::<SmallVec<[_; 1]>>();

            lines.push(WrappedLine {
                layout: Arc::new(WrappedLineLayout {
                    unwrapped_layout: Arc::new(LineLayout {
                        font_size: px(16.0),
                        width: px(LINE_LEN as f32),
                        ascent: px(12.0),
                        descent: px(4.0),
                        runs: vec![ShapedRun {
                            font_id: FontId(0),
                            glyphs,
                        }],
                        len: LINE_LEN,
                    }),
                    wrap_boundaries,
                    wrap_width: Some(px(WRAP_STEP as f32)),
                }),
                text,
                decoration_runs: Vec::new(),
            });
        }

        let line_metrics = text_layout_line_metrics(&lines, px(16.0));
        *layout.0.borrow_mut() = Some(TextLayoutInner {
            len: LINE_COUNT * LINE_LEN + LINE_COUNT - 1,
            lines,
            line_metrics,
            line_height: px(16.0),
            wrap_width: Some(px(WRAP_STEP as f32)),
            size: None,
            bounds: None,
        });

        let wrapped_text = layout.wrapped_text();

        assert!(wrapped_text.len() > LINE_COUNT * LINE_LEN);
        black_box(wrapped_text.len());
    }

    fn build_many_line_text_layout(line_count: usize, line_len: usize) -> TextLayout {
        let layout = TextLayout::default();
        let mut lines = SmallVec::<[WrappedLine; 1]>::new();
        let glyphs = (0..line_len)
            .map(|ix| ShapedGlyph {
                id: GlyphId(ix as u32),
                position: point(px(ix as f32), px(0.0)),
                index: ix,
                is_emoji: false,
            })
            .collect::<Vec<_>>();
        let line_layout = Arc::new(WrappedLineLayout {
            unwrapped_layout: Arc::new(LineLayout {
                font_size: px(16.0),
                width: px(line_len as f32),
                ascent: px(12.0),
                descent: px(4.0),
                runs: vec![ShapedRun {
                    font_id: FontId(0),
                    glyphs,
                }],
                len: line_len,
            }),
            wrap_boundaries: SmallVec::new(),
            wrap_width: None,
        });

        for ix in 0..line_count {
            lines.push(WrappedLine {
                layout: line_layout.clone(),
                text: SharedString::from(format!("{ix:04}-{}", "x".repeat(line_len - 5))),
                decoration_runs: Vec::new(),
            });
        }

        let line_height = px(16.0);
        let line_metrics = text_layout_line_metrics(&lines, line_height);
        *layout.0.borrow_mut() = Some(TextLayoutInner {
            len: line_count * line_len + line_count.saturating_sub(1),
            lines,
            line_metrics,
            line_height,
            wrap_width: None,
            size: None,
            bounds: Some(Bounds::new(
                point(px(0.0), px(0.0)),
                Size {
                    width: px(line_len as f32),
                    height: line_height * line_count,
                },
            )),
        });

        layout
    }

    fn build_text_layout_with_line_lengths(
        line_lengths: &[usize],
        line_height: Pixels,
    ) -> (TextLayout, Vec<Arc<WrappedLineLayout>>) {
        let layout = TextLayout::default();
        let mut lines = SmallVec::<[WrappedLine; 1]>::new();
        let mut line_layouts = Vec::new();

        for (line_ix, &line_len) in line_lengths.iter().enumerate() {
            let glyphs = (0..line_len)
                .map(|ix| ShapedGlyph {
                    id: GlyphId((line_ix * 100 + ix) as u32),
                    position: point(px(ix as f32), px(0.0)),
                    index: ix,
                    is_emoji: false,
                })
                .collect::<Vec<_>>();
            let line_layout = Arc::new(WrappedLineLayout {
                unwrapped_layout: Arc::new(LineLayout {
                    font_size: px(16.0),
                    width: px(line_len as f32),
                    ascent: px(12.0),
                    descent: px(4.0),
                    runs: vec![ShapedRun {
                        font_id: FontId(line_ix),
                        glyphs,
                    }],
                    len: line_len,
                }),
                wrap_boundaries: SmallVec::new(),
                wrap_width: None,
            });
            line_layouts.push(line_layout.clone());
            lines.push(WrappedLine {
                layout: line_layout,
                text: SharedString::from("x".repeat(line_len)),
                decoration_runs: Vec::new(),
            });
        }

        let line_metrics = text_layout_line_metrics(&lines, line_height);
        let len = line_lengths.iter().sum::<usize>() + line_lengths.len().saturating_sub(1);
        *layout.0.borrow_mut() = Some(TextLayoutInner {
            len,
            lines,
            line_metrics,
            line_height,
            wrap_width: None,
            size: None,
            bounds: Some(Bounds::new(
                point(px(0.0), px(0.0)),
                Size {
                    width: px(line_lengths.iter().copied().max().unwrap_or_default() as f32),
                    height: line_height * line_lengths.len(),
                },
            )),
        });

        (layout, line_layouts)
    }

    #[test]
    fn text_layout_lookup_methods_handle_line_boundaries() {
        let line_height = px(10.0);
        let (layout, line_layouts) = build_text_layout_with_line_lengths(&[4, 3, 2], line_height);

        assert_eq!(layout.len(), 11);
        assert_eq!(layout.index_for_position(point(px(1.0), px(-0.1))), Err(0));
        assert_eq!(layout.index_for_position(point(px(2.0), px(9.9))), Ok(2));
        assert_eq!(layout.index_for_position(point(px(2.0), px(10.0))), Err(0));
        assert_eq!(layout.index_for_position(point(px(2.0), px(10.1))), Ok(7));
        assert_eq!(layout.index_for_position(point(px(1.0), px(31.0))), Err(11));

        assert_eq!(layout.position_for_index(0), Some(point(px(0.0), px(0.0))));
        assert_eq!(layout.position_for_index(4), Some(point(px(4.0), px(0.0))));
        assert_eq!(layout.position_for_index(5), Some(point(px(0.0), px(10.0))));
        assert_eq!(layout.position_for_index(8), Some(point(px(3.0), px(10.0))));
        assert_eq!(layout.position_for_index(9), Some(point(px(0.0), px(20.0))));
        assert_eq!(
            layout.position_for_index(11),
            Some(point(px(2.0), px(20.0)))
        );
        assert_eq!(layout.position_for_index(12), None);

        assert!(Arc::ptr_eq(
            &layout.line_layout_for_index(4).unwrap(),
            &line_layouts[0]
        ));
        assert!(Arc::ptr_eq(
            &layout.line_layout_for_index(5).unwrap(),
            &line_layouts[1]
        ));
        assert!(Arc::ptr_eq(
            &layout.line_layout_for_index(9).unwrap(),
            &line_layouts[2]
        ));
        assert!(layout.line_layout_for_index(12).is_none());
    }

    #[test]
    fn text_layout_lookup_methods_handle_empty_middle_line() {
        let line_height = px(10.0);
        let (layout, line_layouts) = build_text_layout_with_line_lengths(&[2, 0, 2], line_height);

        assert_eq!(layout.len(), 6);
        assert_eq!(layout.index_for_position(point(px(0.0), px(10.1))), Err(3));
        assert_eq!(layout.position_for_index(2), Some(point(px(2.0), px(0.0))));
        assert_eq!(layout.position_for_index(3), Some(point(px(0.0), px(10.0))));
        assert_eq!(layout.position_for_index(4), Some(point(px(0.0), px(20.0))));
        assert_eq!(layout.position_for_index(6), Some(point(px(2.0), px(20.0))));
        assert_eq!(layout.position_for_index(7), None);

        assert!(Arc::ptr_eq(
            &layout.line_layout_for_index(3).unwrap(),
            &line_layouts[1]
        ));
        assert!(Arc::ptr_eq(
            &layout.line_layout_for_index(4).unwrap(),
            &line_layouts[2]
        ));
    }

    #[test]
    fn compute_runs_preserves_default_gaps_and_adjacent_highlights() {
        let text = "abcdef";
        let default_style = TextStyle::default();
        let bold: HighlightStyle = FontWeight::BOLD.into();
        let red = HighlightStyle {
            color: Some(crate::red()),
            ..Default::default()
        };

        let runs = StyledText::compute_runs(
            text,
            &default_style,
            vec![(1..3, bold), (3..4, red.clone())],
        );

        assert_eq!(runs.len(), 4);
        assert_eq!(runs[0], default_style.to_run(1));
        assert_eq!(runs[1], default_style.clone().highlight(bold).to_run(2));
        assert_eq!(runs[2], default_style.clone().highlight(red).to_run(1));
        assert_eq!(runs[3], default_style.to_run(2));
        assert_eq!(runs.iter().map(|run| run.len).sum::<usize>(), text.len());

        let empty_runs = StyledText::compute_runs("", &default_style, Vec::new());
        assert!(empty_runs.is_empty());

        let default_only = StyledText::compute_runs("abc", &default_style, Vec::new());
        assert_eq!(default_only, vec![default_style.to_run(3)]);

        let all_highlighted =
            StyledText::compute_runs("abc", &default_style, vec![(0..3, FontWeight::BOLD.into())]);
        assert_eq!(all_highlighted.len(), 1);
        assert_eq!(
            all_highlighted[0],
            default_style.highlight(FontWeight::BOLD).to_run(3)
        );
    }

    #[perf(important)]
    fn perf_text_layout_position_lookups_many_lines() {
        const LINE_COUNT: usize = 4_096;
        const LINE_LEN: usize = 64;

        let layout = build_many_line_text_layout(LINE_COUNT, LINE_LEN);
        let mut checksum = 0usize;
        let mut total_x = px(0.0);

        for line_ix in (0..LINE_COUNT).step_by(7) {
            let y = px(line_ix as f32 * 16.0 + 4.0);
            checksum ^= layout
                .index_for_position(point(px(10.25), y))
                .unwrap_or_else(|index| index);

            let index = line_ix * (LINE_LEN + 1) + LINE_LEN / 2;
            if let Some(position) = layout.position_for_index(index) {
                total_x += position.x;
            }
            if let Some(line) = layout.line_layout_for_index(index) {
                checksum ^= line.len();
            }
        }

        black_box(checksum);
        black_box(total_x);
    }
}
