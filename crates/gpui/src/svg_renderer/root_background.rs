//! The background Chromium paints under an SVG document's root element.
//!
//! usvg drops it, yet generated charts such as star-history.com draw black ink
//! on a white root background and vanish on a dark page without it. This
//! resolves the root's `background-color` through the author cascade the way
//! Chromium does for an SVG `<img>`: `<style>` sheets by specificity and source
//! order, then the inline `style` attribute, with `!important` reversing the
//! two. Invalid declarations are dropped, the `background` shorthand resets the
//! colour, and `currentColor` follows the root's `color`. A `background`
//! attribute is ignored, as in browsers: SVG has no such presentation attribute.
//!
//! Unsupported, so their declarations count as invalid or their rules never
//! match: `@media` and other at-rules (skipped whole, leaving the rules outside
//! them, which are usually the light scheme), selectors simplecss cannot parse
//! (`:not()`, `:is()`, `:nth-child()`, ...), `var()`, and colour functions
//! svgtypes cannot parse (`hwb()`, `lab()`, `oklch()`, ...).

use resvg::tiny_skia::Color;
use std::borrow::Cow;

/// The colour to fill the canvas with before drawing, if any.
///
/// Reads the document usvg is built from, so it costs no second parse. An SVG
/// with neither a root `style` attribute nor a `<style>` element, as icons
/// usually are, returns after one walk over the nodes without allocating.
pub(super) fn root_background(document: &roxmltree::Document) -> Option<Color> {
    let root = document.root_element();
    let sheets: Vec<Cow<str>> = document
        .descendants()
        .filter(is_css_style_element)
        .map(|style| strip_comments(text_content(style)))
        .collect();
    let inline = root
        .attribute("style")
        .map(|style| strip_comments(Cow::Borrowed(style)));
    if sheets.is_empty() && inline.is_none() {
        return None;
    }
    let cascade = Cascade::new(root, &sheets, inline.as_deref());

    let color = match cascade.resolve(&["background", "background-color"], parse_background)? {
        Paint::Transparent => return None,
        Paint::Color(color) => color,
        Paint::CurrentColor => cascade
            .resolve(&["color"], parse_color_property)
            .unwrap_or_else(svgtypes::Color::black),
    };
    (color.alpha > 0).then(|| Color::from_rgba8(color.red, color.green, color.blue, color.alpha))
}

/// The properties the background depends on; every other declaration is skipped.
const PROPERTIES: [&str; 3] = ["background", "background-color", "color"];

struct Declaration<'a> {
    name: &'a str,
    value: &'a str,
    important: bool,
}

/// The root's declarations in cascade order: the last valid one wins.
struct Cascade<'a> {
    declarations: Vec<Declaration<'a>>,
}

impl<'a> Cascade<'a> {
    fn new(
        root: roxmltree::Node<'a, '_>,
        sheets: &'a [Cow<'a, str>],
        inline: Option<&'a str>,
    ) -> Self {
        // Presentation attributes rank below every author rule. Of the
        // properties read here only `color` has one.
        let attributes = root.attribute("color").map(|value| Declaration {
            name: "color",
            value,
            important: false,
        });
        let mut rules: Vec<([u8; 3], &str)> = sheets
            .iter()
            .flat_map(|sheet| style_rules(sheet))
            // Parsing and matching selectors costs more than scanning declarations.
            .filter(|(_, block)| declarations(block).next().is_some())
            .filter_map(|(selectors, block)| {
                Some((matching_specificity(&XmlNode(root), selectors)?, block))
            })
            .collect();
        // Stable, so source order still decides between equal specificities.
        rules.sort_by_key(|(specificity, _)| *specificity);
        let mut declarations: Vec<_> = attributes
            .into_iter()
            .chain(rules.into_iter().flat_map(|(_, block)| declarations(block)))
            .chain(inline.into_iter().flat_map(declarations))
            .collect();
        // `!important` lifts a declaration above every normal one; the stable
        // sort keeps sheet-before-inline order within each group, so an
        // important inline declaration still beats an important rule.
        declarations.sort_by_key(|declaration| declaration.important);
        Self { declarations }
    }

    /// The value of the last declaration of `names` that `parse` accepts.
    fn resolve<T>(&self, names: &[&str], parse: impl Fn(&str, &str) -> Option<T>) -> Option<T> {
        self.declarations
            .iter()
            .rev()
            .filter(|d| names.iter().any(|name| d.name.eq_ignore_ascii_case(name)))
            .find_map(|d| parse(d.name, d.value))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Paint {
    Transparent,
    CurrentColor,
    Color(svgtypes::Color),
}

/// A `background-color` or `background` value, or `None` if it is invalid.
fn parse_background(name: &str, value: &str) -> Option<Paint> {
    // The root has no parent to inherit from, so each of these is the initial value.
    if is_css_wide_keyword(value) {
        return Some(Paint::Transparent);
    }
    if name.eq_ignore_ascii_case("background-color") {
        return parse_paint(value);
    }
    // The shorthand resets the colour to transparent unless it names one, and
    // only its final layer may.
    let mut layers = split_top_level(value, |c| c == ',').peekable();
    let mut paint = None;
    while let Some(layer) = layers.next() {
        let is_final = layers.peek().is_none();
        let mut tokens = split_top_level(layer, |c| c.is_whitespace() || c == '/')
            .filter(|token| !token.is_empty())
            .peekable();
        tokens.peek()?;
        for token in tokens {
            if is_background_component(token) {
                continue;
            }
            if !is_final || paint.is_some() {
                return None;
            }
            paint = Some(parse_paint(token)?);
        }
    }
    Some(paint.unwrap_or(Paint::Transparent))
}

/// A `color` value, or `None` if it is invalid.
fn parse_color_property(_name: &str, value: &str) -> Option<svgtypes::Color> {
    // Inheriting from no parent gives the initial value, which is black.
    if is_css_wide_keyword(value) || value.eq_ignore_ascii_case("currentcolor") {
        return Some(svgtypes::Color::black());
    }
    value.parse().ok()
}

fn parse_paint(value: &str) -> Option<Paint> {
    if value.eq_ignore_ascii_case("currentcolor") {
        return Some(Paint::CurrentColor);
    }
    value.parse().ok().map(Paint::Color)
}

fn is_css_wide_keyword(value: &str) -> bool {
    ["inherit", "initial", "unset", "revert", "revert-layer"]
        .iter()
        .any(|keyword| value.eq_ignore_ascii_case(keyword))
}

/// Whether `token` is a `background` shorthand component other than the colour.
fn is_background_component(token: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "none",
        "repeat",
        "repeat-x",
        "repeat-y",
        "no-repeat",
        "space",
        "round",
        "scroll",
        "fixed",
        "local",
        "left",
        "right",
        "top",
        "bottom",
        "center",
        "auto",
        "cover",
        "contain",
        "border-box",
        "padding-box",
        "content-box",
    ];
    const FUNCTIONS: &[&str] = &[
        "url",
        "image-set",
        "-webkit-image-set",
        "cross-fade",
        "-webkit-cross-fade",
        "calc",
        "min",
        "max",
        "clamp",
    ];
    if KEYWORDS
        .iter()
        .any(|keyword| token.eq_ignore_ascii_case(keyword))
    {
        return true;
    }
    if let Some((function, _)) = token.split_once('(')
        && token.ends_with(')')
    {
        let function = function.to_ascii_lowercase();
        return function.ends_with("gradient") || FUNCTIONS.contains(&function.as_str());
    }
    is_length_or_percentage(token)
}

/// `0`, `50%`, `1.5em`, `-10px`: a number with an optional unit.
fn is_length_or_percentage(token: &str) -> bool {
    let unit_start = token
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '+' | '-')))
        .unwrap_or(token.len());
    let (number, unit) = token.split_at(unit_start);
    number.parse::<f64>().is_ok()
        && (unit.is_empty() || unit == "%" || unit.chars().all(|c| c.is_ascii_alphabetic()))
}

/// `(selectors, declaration block)` of each style rule in `css`, in source
/// order. At-rules, `@media` included, are skipped whole.
fn style_rules(css: &str) -> impl Iterator<Item = (&str, &str)> {
    let mut rest = css.trim_start();
    std::iter::from_fn(move || {
        while !rest.is_empty() {
            let at_rule = rest.starts_with('@');
            let end = find_top_level(rest, |c| c == '{' || (at_rule && c == ';'))?;
            let (prelude, tail) = rest.split_at(end);
            let body = &tail[1..];
            if tail.starts_with(';') {
                rest = body.trim_start();
                continue;
            }
            // A block left open at the end of the sheet closes there.
            let close = find_top_level(body, |c| c == '}').unwrap_or(body.len());
            rest = body.get(close + 1..).unwrap_or_default().trim_start();
            if !at_rule {
                return Some((prelude, &body[..close]));
            }
        }
        None
    })
}

/// The specificity of the most specific selector in `selectors` that matches
/// `element`, or `None` if none does. One selector simplecss cannot parse
/// drops the whole list, as in CSS.
fn matching_specificity(element: &XmlNode, selectors: &str) -> Option<[u8; 3]> {
    let mut specificity = None;
    for selector in split_top_level(selectors, |c| c == ',') {
        let selector = with_root_as_first_child(selector.trim());
        let selector = simplecss::Selector::parse(&selector)?;
        if selector.matches(element) {
            specificity = specificity.max(Some(selector.specificity()));
        }
    }
    specificity
}

/// The declarations of [`PROPERTIES`] in a block or `style` attribute; ones
/// without a value are dropped.
fn declarations(block: &str) -> impl Iterator<Item = Declaration<'_>> {
    split_top_level(block, |c| c == ';').filter_map(|declaration| {
        let (name, value) = declaration.split_once(':')?;
        let name = name.trim();
        if !PROPERTIES
            .iter()
            .any(|property| name.eq_ignore_ascii_case(property))
        {
            return None;
        }
        let (value, important) = match value.rfind('!') {
            Some(bang) if value[bang + 1..].trim().eq_ignore_ascii_case("important") => {
                (&value[..bang], true)
            }
            _ => (value, false),
        };
        let value = value.trim();
        (!value.is_empty()).then_some(Declaration {
            name,
            value,
            important,
        })
    })
}

/// `css` with each comment replaced by the space it separates tokens with.
fn strip_comments(css: Cow<'_, str>) -> Cow<'_, str> {
    if !css.contains("/*") {
        return css;
    }
    let mut result = String::with_capacity(css.len());
    let mut rest = css.as_ref();
    while let Some(start) = rest.find("/*") {
        result.push_str(&rest[..start]);
        result.push(' ');
        let comment = &rest[start + 2..];
        rest = comment.find("*/").map_or("", |end| &comment[end + 2..]);
    }
    result.push_str(rest);
    Cow::Owned(result)
}

/// `value` split at each `is_separator` outside strings and brackets, keeping
/// `rgb(0, 0, 0)` and `url("a b")` whole.
fn split_top_level(value: &str, is_separator: impl Fn(char) -> bool) -> impl Iterator<Item = &str> {
    let mut rest = Some(value);
    std::iter::from_fn(move || {
        let current = rest?;
        match find_top_level(current, &is_separator) {
            Some(index) => {
                let separator = current[index..].chars().next().map_or(0, char::len_utf8);
                rest = Some(&current[index + separator..]);
                Some(&current[..index])
            }
            None => {
                rest = None;
                Some(current)
            }
        }
    })
}

/// The byte index of the first `target` character outside strings and brackets.
fn find_top_level(text: &str, target: impl Fn(char) -> bool) -> Option<usize> {
    let mut depth = 0usize;
    let mut quote = None;
    let mut chars = text.char_indices();
    while let Some((index, c)) = chars.next() {
        if let Some(open) = quote {
            if c == '\\' {
                chars.next();
            } else if c == open {
                quote = None;
            }
            continue;
        }
        if depth == 0 && target(c) {
            return Some(index);
        }
        match c {
            '\\' => {
                chars.next();
            }
            '"' | '\'' => quote = Some(c),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    None
}

/// A `<style>` element whose contents are CSS.
fn is_css_style_element(node: &roxmltree::Node) -> bool {
    node.has_tag_name("style")
        && node
            .attribute("type")
            .is_none_or(|kind| kind.is_empty() || kind.eq_ignore_ascii_case("text/css"))
}

/// The element's text, CDATA sections included.
fn text_content<'a>(node: roxmltree::Node<'a, '_>) -> Cow<'a, str> {
    let mut texts = node.children().filter_map(|child| child.text());
    let first = texts.next().unwrap_or_default();
    match texts.next() {
        None => Cow::Borrowed(first),
        Some(second) => Cow::Owned([first, second].into_iter().chain(texts).collect()),
    }
}

/// simplecss has no `:root`, and drops any rule that uses it. The only element
/// matched here is the document root, which is also the first child, and
/// neither pseudo-class can match its absent ancestors, so `:first-child`
/// stands in exactly, with the same specificity.
fn with_root_as_first_child(css: &str) -> Cow<'_, str> {
    if !css.contains(':') {
        return Cow::Borrowed(css);
    }
    let mut result = String::with_capacity(css.len());
    let mut rest = css;
    while let Some(index) = rest.find(':') {
        let (before, after) = rest.split_at(index);
        result.push_str(before);
        let name = after.get(1..5);
        let ends = !after[1..]
            .get(4..)
            .and_then(|tail| tail.chars().next())
            .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'));
        if name.is_some_and(|name| name.eq_ignore_ascii_case("root")) && ends {
            result.push_str(":first-child");
            rest = &after[5..];
        } else {
            result.push(':');
            rest = &after[1..];
        }
    }
    result.push_str(rest);
    Cow::Owned(result)
}

struct XmlNode<'a, 'input>(roxmltree::Node<'a, 'input>);

impl simplecss::Element for XmlNode<'_, '_> {
    fn parent_element(&self) -> Option<Self> {
        self.0.parent_element().map(XmlNode)
    }

    fn prev_sibling_element(&self) -> Option<Self> {
        self.0.prev_sibling_element().map(XmlNode)
    }

    fn has_local_name(&self, local_name: &str) -> bool {
        self.0.tag_name().name() == local_name
    }

    fn attribute_matches(&self, local_name: &str, operator: simplecss::AttributeOperator) -> bool {
        self.0
            .attribute(local_name)
            .is_some_and(|value| operator.matches(value))
    }

    fn pseudo_class_matches(&self, class: simplecss::PseudoClass) -> bool {
        // A static image is never hovered, focused or visited.
        matches!(class, simplecss::PseudoClass::FirstChild)
            && self.0.prev_sibling_element().is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHITE: Option<[u8; 4]> = Some([255, 255, 255, 255]);
    const BLACK: Option<[u8; 4]> = Some([0, 0, 0, 255]);

    fn background(root_attributes: &str, sheet: &str) -> Option<[u8; 4]> {
        let style = if sheet.is_empty() {
            String::new()
        } else {
            format!("<style>{sheet}</style>")
        };
        let svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8" {root_attributes}>{style}<rect width="4" height="4"/></svg>"#
        );
        background_of(&svg).map(|color| {
            let color = color.to_color_u8();
            [color.red(), color.green(), color.blue(), color.alpha()]
        })
    }

    fn background_of(svg: &str) -> Option<Color> {
        root_background(&roxmltree::Document::parse(svg).expect("parse svg"))
    }

    // Each expectation is what headless Chromium paints for the same SVG in an `<img>`.

    #[test]
    fn reads_inline_style() {
        assert_eq!(
            background(r#"style="stroke-width:3;background:#fff""#, ""),
            WHITE
        );
        assert_eq!(
            background(r#"style="background-color: rgb(0, 0, 255)""#, ""),
            Some([0, 0, 255, 255])
        );
        assert_eq!(background(r#"style="BACKGROUND:#FFF""#, ""), WHITE);
        assert_eq!(
            background(r#"style="background:rgba(255,255,255,0.5)""#, ""),
            Some([255, 255, 255, 128])
        );
    }

    #[test]
    fn ignores_background_attributes() {
        assert_eq!(
            background(
                r##"background="#fff" background-color="#fff" viewport-fill="#fff""##,
                ""
            ),
            None
        );
    }

    #[test]
    fn reads_style_sheets_matching_the_root() {
        for (attributes, sheet) in [
            ("", "svg{background:#fff}"),
            ("", ":root{background:#fff}"),
            ("", ":ROOT { background: #fff }"),
            ("", "*{background:#fff}"),
            (r#"id="a""#, "#a{background:#fff}"),
            (r#"class="c""#, ".c{background:#fff}"),
            ("", "svg:first-child{background:#fff}"),
            ("", "g, svg{background:#fff}"),
            ("", "<![CDATA[svg{background:#fff}]]>"),
        ] {
            assert_eq!(background(attributes, sheet), WHITE, "{attributes} {sheet}");
        }
        for sheet in [
            "g{background:#fff}",
            ":root g{background:#fff}",
            ":rooted{background:#fff}",
        ] {
            assert_eq!(background("", sheet), None, "{sheet}");
        }
    }

    #[test]
    fn parses_sheets_like_css() {
        for sheet in [
            "/* svg{background:#000} */ svg{background:#fff}",
            "svg{background:#fff/* ; background:#000 */}",
            "@import url(a.css); svg{background:#fff}",
            "@media print{svg{background:#000}} svg{background:#fff}",
            "svg{background:#fff} @media print{svg{background:#000}}",
            "svg{background:#fff} g, svg:hover{background:#000}",
            "svg{background:#fff",
        ] {
            assert_eq!(background("", sheet), WHITE, "{sheet}");
        }
        // One selector simplecss cannot parse drops the whole rule.
        assert_eq!(background("", "svg, svg:bogus{background:#fff}"), None);
    }

    #[test]
    fn ignores_non_css_style_elements() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg"><style type="text/foo">svg{background:#fff}</style></svg>"#;
        assert_eq!(background_of(svg), None);
    }

    #[test]
    fn cascades_like_css() {
        for (attributes, sheet) in [
            (r#"style="background:#fff""#, "svg{background:#000}"),
            (
                r#"style="background:#000""#,
                "svg{background:#fff!important}",
            ),
            (
                r#"style="background:#fff!important""#,
                "svg{background:#000!important}",
            ),
            (r#"id="a""#, "#a{background:#fff} svg{background:#000}"),
            ("", "svg{background:#000} svg{background:#fff}"),
            (r#"style="background:#000;background-color:#fff""#, ""),
            (r#"style="background:#fff !important;background:#000""#, ""),
        ] {
            assert_eq!(background(attributes, sheet), WHITE, "{attributes} {sheet}");
        }
    }

    #[test]
    fn drops_invalid_declarations() {
        for (attributes, sheet) in [
            ("", "svg{background:#fff;background:bogus}"),
            ("", "svg{background:#fff;background-color:bogus}"),
            (r#"style="background:#fff;background:bogus""#, ""),
            (r#"style="background:#fff;background:#000 #000""#, ""),
            (r#"style="background:#fff;background:#000, url(a.png)""#, ""),
        ] {
            assert_eq!(background(attributes, sheet), WHITE, "{attributes} {sheet}");
        }
    }

    #[test]
    fn shorthand_without_a_colour_is_transparent() {
        for style in [
            "background-color:#fff;background:url(x.png)",
            "background:#fff;background:none",
            "background:#fff;background-color:inherit",
            "background:transparent",
        ] {
            assert_eq!(
                background(&format!(r#"style="{style}""#), ""),
                None,
                "{style}"
            );
        }
    }

    #[test]
    fn takes_the_colour_from_the_final_layer() {
        for style in [
            "background: url(x.png) no-repeat #fff",
            "background:url(a.png), #fff",
            "background:center/cover no-repeat #fff",
            "background:linear-gradient(transparent, transparent) 0 0 / 50% auto #fff",
            "background:linear-gradient(rgba(0, 0, 0, 0), transparent) #fff",
            "background:url('a;b c.png') #fff",
            "background:#fff/* comment */",
        ] {
            assert_eq!(
                background(&format!(r#"style="{style}""#), ""),
                WHITE,
                "{style}"
            );
        }
    }

    #[test]
    fn resolves_current_color() {
        assert_eq!(
            background(r#"style="color:#fff;background:currentColor""#, ""),
            WHITE
        );
        assert_eq!(
            background(r##"color="#fff" style="background:currentColor""##, ""),
            WHITE
        );
        assert_eq!(
            background(
                r##"color="#000" style="background:currentColor""##,
                "svg{color:#fff}"
            ),
            WHITE
        );
        assert_eq!(background(r#"style="background:currentColor""#, ""), BLACK);
    }
}
