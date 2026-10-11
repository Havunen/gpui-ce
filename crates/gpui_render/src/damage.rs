//! Conservative retained-scene comparison for the opt-in partial-redraw experiment.
//! Exact typed comparisons avoid formatting, hashing padding, or trusting a hash collision.
//! Local colour/quad changes and bounded disjoint painter swaps can reuse the target.
use crate::blur::GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS;
use gpui::{
    Bounds, ContentMask, MonochromeSprite, Path, PathVertex, PolychromeSprite, PrimitiveBatch,
    Quad, RenderCommand, ScaledPixels, Scene, SceneHsla, Shadow, SubpixelSprite,
    TransformationMatrix, Underline, point, size,
};
use std::mem::size_of;

const MAX_SNAPSHOT_BYTES: usize = 2 * 1024 * 1024;
const MAX_REGIONS: usize = 8;
const MAX_QUAD_EDITS: usize = 8;
const MAX_GLYPH_EDITS: usize = 128;
const MAX_COMMUTATIONS: usize = 8;
// Prototype fallback budgets. The original strict comparer remains the fast path.
const DAMAGE_TILE_SIZE: u32 = 64;
const MAX_DAMAGE_TILES: usize = 4096;
const MAX_TILE_PRIMITIVES: usize = 8192;
const MAX_TILE_REFERENCES: usize = 65536;
const MAX_TILE_COMPARISONS: usize = 131072;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// A clipped damage rectangle in physical target pixels.
pub struct DamageRect {
    /// Left target coordinate.
    pub x: u32,
    /// Top target coordinate.
    pub y: u32,
    /// Rectangle width.
    pub width: u32,
    /// Rectangle height.
    pub height: u32,
}

impl DamageRect {
    /// Convert integer target coordinates to GPUI scene bounds.
    pub fn bounds(self) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(self.x as f32), ScaledPixels(self.y as f32)),
            size: size(
                ScaledPixels(self.width as f32),
                ScaledPixels(self.height as f32),
            ),
        }
    }

    /// Whether the rectangle intersects scene bounds.
    pub fn intersects(self, bounds: Bounds<ScaledPixels>) -> bool {
        !self.bounds().intersect(&bounds).is_empty()
    }

    /// The smallest rectangle covering both.
    pub fn union(self, other: Self) -> Self {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        Self {
            x,
            y,
            width: self
                .x
                .saturating_add(self.width)
                .max(other.x.saturating_add(other.width))
                - x,
            height: self
                .y
                .saturating_add(self.height)
                .max(other.y.saturating_add(other.height))
                - y,
        }
    }
}

#[derive(Clone, Debug)]
/// Bounded exact copies of the last successfully rendered scene's primitives.
pub struct SceneSnapshot {
    commands: Vec<RenderCommand>,
    quads: Vec<Quad>,
    paths: Vec<Path<ScaledPixels>>,
    monochrome_sprites: Vec<MonochromeSprite>,
    subpixel_sprites: Vec<SubpixelSprite>,
    shadows: Vec<Shadow>,
    underlines: Vec<Underline>,
    polychrome_sprites: Vec<PolychromeSprite>,
    atlas_generation: u64,
    width: u32,
    height: u32,
}

/// A stable reason why a scene cannot safely use retained damage comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneSnapshotUnsupportedReason {
    /// The physical render target has a zero dimension.
    InvalidTargetSize,
    /// A surface can change its pixels independently of the scene primitives.
    Surface,
    /// A backdrop filter reads pixels beyond local damage bounds.
    BackdropFilter,
    /// An element filter introduces an unsupported render-target boundary.
    FilterBoundary,
    /// Geometry is nonfinite or has an invalid negative extent.
    NonfiniteGeometry,
    /// Exact primitive copies would exceed the bounded snapshot budget.
    SnapshotBudget,
}

impl SceneSnapshotUnsupportedReason {
    /// Stable name of the reason, used as a full-frame fallback reason.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidTargetSize => "invalid-target-size",
            Self::Surface => "surface",
            Self::BackdropFilter => "backdrop-filter",
            Self::FilterBoundary => "filter-boundary",
            Self::NonfiniteGeometry => "nonfinite-geometry",
            Self::SnapshotBudget => "snapshot-budget",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Conservative amount of rendering needed to replace a retained scene.
pub enum Damage {
    /// The whole scene must be rendered normally.
    Full,
    /// The retained scene already matches the new scene.
    Reuse,
    /// Clear and redraw each region with all intersecting painter batches.
    Partial(Vec<DamageRect>),
}

impl Damage {
    /// Stable name of the selected rendering path.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Reuse => "reuse",
            Self::Partial(_) => "partial",
        }
    }
}

fn finite_bounds(bounds: Bounds<ScaledPixels>) -> bool {
    bounds.left().0.is_finite()
        && bounds.top().0.is_finite()
        && bounds.right().0.is_finite()
        && bounds.bottom().0.is_finite()
        && bounds.size.width.0 >= 0.0
        && bounds.size.height.0 >= 0.0
}

fn rectangle(bounds: Bounds<ScaledPixels>, width: u32, height: u32) -> DamageRect {
    if bounds.is_empty() {
        return DamageRect {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        };
    }
    // Include the antialias fringe; then clip to the actual target extent.
    let x = (bounds.left().0.floor() - 1.0).max(0.0).min(width as f32) as u32;
    let y = (bounds.top().0.floor() - 1.0).max(0.0).min(height as f32) as u32;
    let right = (bounds.right().0.ceil() + 1.0).max(0.0).min(width as f32) as u32;
    let bottom = (bounds.bottom().0.ceil() + 1.0).max(0.0).min(height as f32) as u32;
    DamageRect {
        x,
        y,
        width: right.saturating_sub(x),
        height: bottom.saturating_sub(y),
    }
}

fn snapshot_bytes(scene: &Scene) -> Option<usize> {
    let mut bytes = size_of::<SceneSnapshot>();
    let mut add = |count: usize, item_size: usize| -> Option<()> {
        bytes = bytes.checked_add(count.checked_mul(item_size)?)?;
        (bytes <= MAX_SNAPSHOT_BYTES).then_some(())
    };
    add(scene.render_commands().len(), size_of::<RenderCommand>())?;
    add(scene.quads.len(), size_of::<Quad>())?;
    add(scene.paths.len(), size_of::<Path<ScaledPixels>>())?;
    for path in &scene.paths {
        add(path.vertices.len(), size_of::<PathVertex<ScaledPixels>>())?;
    }
    add(
        scene.monochrome_sprites.len(),
        size_of::<MonochromeSprite>(),
    )?;
    add(scene.subpixel_sprites.len(), size_of::<SubpixelSprite>())?;
    add(scene.shadows.len(), size_of::<Shadow>())?;
    add(scene.underlines.len(), size_of::<Underline>())?;
    add(
        scene.polychrome_sprites.len(),
        size_of::<PolychromeSprite>(),
    )?;
    Some(bytes)
}

#[derive(Clone, Copy)]
enum PaintedPrimitive<'a> {
    Quad(&'a Quad, bool),
    Shadow(&'a Shadow, bool),
    // Keep path rasterization grouping exact: splitting a group can change compositing.
    Path(&'a Path<ScaledPixels>, usize, usize, usize, usize),
    Underline(&'a Underline),
    Monochrome(&'a MonochromeSprite, gpui::AtlasTextureId),
    Subpixel(&'a SubpixelSprite, gpui::AtlasTextureId),
    Polychrome(&'a PolychromeSprite, bool, gpui::AtlasTextureId),
}

impl PaintedPrimitive<'_> {
    fn is_quad(self) -> bool {
        matches!(self, Self::Quad(..))
    }

    fn is_glyph(self) -> bool {
        matches!(self, Self::Monochrome(..) | Self::Subpixel(..))
    }

    fn quad_mismatch_reason(self, other: Self) -> &'static str {
        let (Self::Quad(next, n_smoothing), Self::Quad(old, o_smoothing)) = (self, other) else {
            return "quad-geometry-or-painter-order-change";
        };
        if n_smoothing != o_smoothing {
            "quad-pipeline-change"
        } else if next.bounds.size != old.bounds.size {
            "quad-extent-change"
        } else if next.content_mask != old.content_mask {
            "quad-content-mask-change"
        } else if next.corner_radii != old.corner_radii
            || next.corner_smoothing != old.corner_smoothing
            || next.border_widths != old.border_widths
            || next.border_style != old.border_style
            || next.border_dashed_length != old.border_dashed_length
            || next.border_dashed_gap != old.border_dashed_gap
            || next.padding != old.padding
        {
            "quad-shape-change"
        } else {
            "quad-painter-convergence-change"
        }
    }

    fn geometry_matches(self, other: Self) -> bool {
        match (self, other) {
            (Self::Quad(next, next_smoothing), Self::Quad(old, old_smoothing)) => {
                let mut next = *next;
                let mut old = *old;
                next.order = 0;
                old.order = 0;
                next.background = Default::default();
                old.background = Default::default();
                next.border_color = Default::default();
                old.border_color = Default::default();
                next_smoothing == old_smoothing && next == old
            }
            (Self::Shadow(next, next_smoothing), Self::Shadow(old, old_smoothing)) => {
                let mut next = *next;
                let mut old = *old;
                next.order = 0;
                old.order = 0;
                next_smoothing == old_smoothing && next == old
            }
            (
                Self::Path(next, n_start, n_end, n_vertices, n_sprites),
                Self::Path(old, o_start, o_end, o_vertices, o_sprites),
            ) => {
                n_start == o_start
                    && n_end == o_end
                    && n_vertices == o_vertices
                    && n_sprites == o_sprites
                    && next.id == old.id
                    && next.bounds == old.bounds
                    && next.content_mask == old.content_mask
                    && next.vertices == old.vertices
            }
            (Self::Underline(next), Self::Underline(old)) => {
                let mut next = *next;
                let mut old = *old;
                next.order = 0;
                old.order = 0;
                next == old
            }
            (Self::Monochrome(next, n_texture), Self::Monochrome(old, o_texture)) => {
                let mut next = *next;
                let mut old = *old;
                next.order = 0;
                old.order = 0;
                next.color = SceneHsla::default();
                old.color = SceneHsla::default();
                n_texture == o_texture && next == old
            }
            (Self::Subpixel(next, n_texture), Self::Subpixel(old, o_texture)) => {
                let mut next = *next;
                let mut old = *old;
                next.order = 0;
                old.order = 0;
                next.color = SceneHsla::default();
                old.color = SceneHsla::default();
                n_texture == o_texture && next == old
            }
            (
                Self::Polychrome(next, n_smoothing, n_texture),
                Self::Polychrome(old, o_smoothing, o_texture),
            ) => {
                let mut next = *next;
                let mut old = *old;
                next.order = 0;
                old.order = 0;
                n_smoothing == o_smoothing && n_texture == o_texture && next == old
            }
            _ => false,
        }
    }

    fn colour_changed(self, other: Self) -> bool {
        match (self, other) {
            (Self::Quad(next, _), Self::Quad(old, _)) => {
                next.background != old.background || next.border_color != old.border_color
            }
            (Self::Path(next, ..), Self::Path(old, ..)) => next.color != old.color,
            (Self::Monochrome(next, _), Self::Monochrome(old, _)) => next.color != old.color,
            (Self::Subpixel(next, _), Self::Subpixel(old, _)) => next.color != old.color,
            _ => false,
        }
    }

    fn bounds(self) -> Bounds<ScaledPixels> {
        match self {
            Self::Quad(p, _) => p.bounds.intersect(&p.content_mask.bounds),
            Self::Path(p, ..) => p.clipped_bounds(),
            Self::Monochrome(p, _) => {
                transformed(p.bounds, &p.transformation).intersect(&p.content_mask.bounds)
            }
            Self::Subpixel(p, _) => {
                transformed(p.bounds, &p.transformation).intersect(&p.content_mask.bounds)
            }
            // These primitives never qualify for colour damage or structural edits.
            _ => Bounds::default(),
        }
    }

    fn commutation_bounds(self) -> Option<Bounds<ScaledPixels>> {
        match self {
            Self::Quad(..) | Self::Path(..) | Self::Monochrome(..) | Self::Subpixel(..) => {
                Some(self.bounds())
            }
            // Blur, underline coverage and transformed colour sprites need a separate
            // coverage proof. Missing bounds must never prove that two paints are disjoint.
            _ => None,
        }
    }

    fn exact_matches(self, other: Self) -> bool {
        self.geometry_matches(other) && !self.colour_changed(other)
    }

    fn comparison_cost(self, other: Self) -> usize {
        match (self, other) {
            (Self::Path(next, ..), Self::Path(old, ..)) => {
                1 + next.vertices.len() + old.vertices.len()
            }
            _ => 1,
        }
    }

    fn tile_coverage(self, width: u32, height: u32) -> Result<DamageRect, &'static str> {
        let (bounds, mask) = match self {
            Self::Quad(p, _) => (p.bounds, p.content_mask.bounds),
            Self::Path(p, ..) => (p.bounds, p.content_mask.bounds),
            Self::Monochrome(p, _) => (
                transformed(p.bounds, &p.transformation),
                p.content_mask.bounds,
            ),
            Self::Subpixel(p, _) => (
                transformed(p.bounds, &p.transformation),
                p.content_mask.bounds,
            ),
            Self::Shadow(p, _) => (shadow_geometry(p), p.content_mask.bounds),
            Self::Underline(p) => (
                p.bounds.dilate(ScaledPixels(p.thickness.0 * 4.0)),
                p.content_mask.bounds,
            ),
            Self::Polychrome(p, ..) => (p.bounds, p.content_mask.bounds),
        };
        if !finite_bounds(bounds) || !finite_bounds(mask) {
            return Err("tile-nonfinite-coverage");
        }
        Ok(rectangle(bounds.intersect(&mask), width, height))
    }

    // A key narrows exact comparisons; it never establishes descriptor equality.
    // Canonicalize signed zero because typed float equality treats both as equal.
    fn descriptor_key(self) -> (u8, [u32; 4], u32, u32) {
        let (kind, bounds, texture, tile) = match self {
            Self::Quad(p, _) => (0, p.bounds, 0, 0),
            Self::Shadow(p, _) => (1, p.bounds, 0, 0),
            Self::Path(p, ..) => (2, p.bounds, 0, 0),
            Self::Underline(p) => (3, p.bounds, 0, 0),
            Self::Monochrome(p, t) => (4, p.bounds, t.index, p.tile.tile_id.0),
            Self::Subpixel(p, t) => (5, p.bounds, t.index, p.tile.tile_id.0),
            Self::Polychrome(p, _, t) => (6, p.bounds, t.index, p.tile.tile_id.0),
        };
        let bits = |value: f32| if value == 0.0 { 0 } else { value.to_bits() };
        (
            kind,
            [
                bits(bounds.origin.x.0),
                bits(bounds.origin.y.0),
                bits(bounds.size.width.0),
                bits(bounds.size.height.0),
            ],
            texture,
            tile,
        )
    }
}

struct TileReferences {
    offsets: Vec<usize>,
    ids: Vec<u32>,
}

fn tile_spans(region: DamageRect, columns: u32) -> impl Iterator<Item = usize> {
    let left = region.x / DAMAGE_TILE_SIZE;
    let top = region.y / DAMAGE_TILE_SIZE;
    let right = region
        .x
        .saturating_add(region.width)
        .div_ceil(DAMAGE_TILE_SIZE);
    let bottom = region
        .y
        .saturating_add(region.height)
        .div_ceil(DAMAGE_TILE_SIZE);
    (top..bottom).flat_map(move |y| (left..right).map(move |x| (y * columns + x) as usize))
}

fn tile_coverages(
    sequence: &[PaintedPrimitive<'_>],
    width: u32,
    height: u32,
    references_left: &mut usize,
) -> Result<Vec<DamageRect>, &'static str> {
    let mut coverage = Vec::with_capacity(sequence.len());
    for primitive in sequence {
        let region = primitive.tile_coverage(width, height)?;
        if region.width != 0 && region.height != 0 {
            let columns =
                (region.x + region.width).div_ceil(DAMAGE_TILE_SIZE) - region.x / DAMAGE_TILE_SIZE;
            let rows =
                (region.y + region.height).div_ceil(DAMAGE_TILE_SIZE) - region.y / DAMAGE_TILE_SIZE;
            *references_left = references_left
                .checked_sub(columns as usize * rows as usize)
                .ok_or("tile-reference-budget")?;
        }
        coverage.push(region);
    }
    Ok(coverage)
}

fn tile_references(
    coverage: &[DamageRect],
    ids: &[u32],
    columns: u32,
    tiles: usize,
) -> TileReferences {
    let mut counts = vec![0usize; tiles];
    for &region in coverage {
        if region.width != 0 && region.height != 0 {
            for tile in tile_spans(region, columns) {
                counts[tile] += 1;
            }
        }
    }
    let mut offsets = Vec::with_capacity(tiles + 1);
    offsets.push(0);
    for count in counts {
        offsets.push(offsets.last().copied().unwrap() + count);
    }
    let mut cursor = offsets[..tiles].to_vec();
    let mut references = vec![0; offsets[tiles]];
    // Fill in original painter order. No global rank or primitive-kind sort is used.
    for (&region, id) in coverage.iter().zip(ids) {
        if region.width != 0 && region.height != 0 {
            for tile in tile_spans(region, columns) {
                references[cursor[tile]] = *id;
                cursor[tile] += 1;
            }
        }
    }
    TileReferences {
        offsets,
        ids: references,
    }
}

fn tile_regions(
    dirty: &[bool],
    columns: u32,
    width: u32,
    height: u32,
) -> Result<Damage, &'static str> {
    let mut regions: Vec<DamageRect> = Vec::new();
    let rows = height.div_ceil(DAMAGE_TILE_SIZE);
    for y in 0..rows {
        let mut x = 0;
        while x < columns {
            if !dirty[(y * columns + x) as usize] {
                x += 1;
                continue;
            }
            let start = x;
            while x < columns && dirty[(y * columns + x) as usize] {
                x += 1;
            }
            let region = DamageRect {
                x: start * DAMAGE_TILE_SIZE,
                y: y * DAMAGE_TILE_SIZE,
                width: (x * DAMAGE_TILE_SIZE).min(width) - start * DAMAGE_TILE_SIZE,
                height: DAMAGE_TILE_SIZE.min(height - y * DAMAGE_TILE_SIZE),
            };
            if let Some(previous) = regions.iter_mut().find(|previous| {
                previous.x == region.x
                    && previous.width == region.width
                    && previous.y + previous.height == region.y
            }) {
                previous.height += region.height;
            } else {
                regions.push(region);
                if regions.len() > MAX_REGIONS {
                    return Err("tile-region-budget");
                }
            }
        }
    }
    if regions.is_empty() {
        return Ok(Damage::Reuse);
    }
    let area: u128 = regions
        .iter()
        .map(|r| r.width as u128 * r.height as u128)
        .sum();
    if area * 100 > width as u128 * height as u128 * 40 {
        Err("tile-damage-area-budget")
    } else {
        Ok(Damage::Partial(regions))
    }
}

// Glyph badges can introduce a whole run, rather than a single primitive. Only
// finite glyphs survive capture, and an exact following
// anchor preserves the unchanged painter sequence. Keep old/new runs separate:
// a union spanning distant rows could merge both full-width row fills into one
// unnecessarily large damage rectangle.
fn glyph_run_to_anchor(
    sequence: &[PaintedPrimitive<'_>],
    start: usize,
    anchor: PaintedPrimitive<'_>,
) -> Option<usize> {
    for count in 1..=MAX_GLYPH_EDITS {
        if !sequence.get(start + count - 1)?.is_glyph() {
            return None;
        }
        if sequence.get(start + count)?.geometry_matches(anchor) {
            return Some(count);
        }
    }
    None
}

fn glyph_prefix_len(sequence: &[PaintedPrimitive<'_>], start: usize) -> usize {
    sequence[start..]
        .iter()
        .take(MAX_GLYPH_EDITS)
        .take_while(|primitive| primitive.is_glyph())
        .count()
}

fn paired_glyph_runs(
    next: &[PaintedPrimitive<'_>],
    old: &[PaintedPrimitive<'_>],
    n: usize,
    o: usize,
) -> Option<(usize, usize)> {
    let mut best = None;
    let mut best_edits = MAX_GLYPH_EDITS + 1;
    let mut ambiguous = false;
    let next_len = glyph_prefix_len(next, n);
    let old_len = glyph_prefix_len(old, o);
    for next_count in 1..=next_len {
        if next_count >= best_edits {
            break;
        }
        for old_count in 1..=old_len
            .min(MAX_GLYPH_EDITS - next_count)
            .min(best_edits - next_count)
        {
            let edits = next_count + old_count;
            if edits > best_edits {
                continue;
            }
            let anchored = match (next.get(n + next_count), old.get(o + old_count)) {
                (Some(next), Some(old)) => next.geometry_matches(*old),
                (None, None) => true,
                _ => false,
            };
            if anchored {
                if edits < best_edits {
                    best = Some((next_count, old_count));
                    best_edits = edits;
                    ambiguous = false;
                } else {
                    ambiguous = true;
                }
            }
        }
    }
    if ambiguous { None } else { best }
}

// A parent hover fill and its canvas badge can be adjacent in the finished
// painter sequence. Neither edit has an immediate unchanged anchor on its own.
// Skip only one quad followed by a bounded glyph prefix, and require the exact
// next primitive to re-establish convergence. All later paints are still
// checked by the main comparison; paths, shadows and extra quads cannot be
// hidden in this prefix. Replacing a quad is conservative because both its old
// and new clipped raster coverage are damaged, and current batches are replayed
// in full painter order. Capture already excludes nonfinite bounds and filters.
fn quad_glyph_run_to_anchor(
    sequence: &[PaintedPrimitive<'_>],
    start: usize,
    anchor: PaintedPrimitive<'_>,
) -> Option<usize> {
    if !sequence.get(start)?.is_quad() {
        return None;
    }
    glyph_run_to_anchor(sequence, start + 1, anchor)
}

fn paired_quad_glyph_runs(
    next: &[PaintedPrimitive<'_>],
    old: &[PaintedPrimitive<'_>],
    n: usize,
    o: usize,
) -> Option<(usize, usize)> {
    if !next.get(n)?.is_quad() || !old.get(o)?.is_quad() {
        return None;
    }
    let next_len = glyph_prefix_len(next, n + 1);
    let old_len = glyph_prefix_len(old, o + 1);
    let mut best = None;
    let mut best_edits = MAX_GLYPH_EDITS + 1;
    let mut ambiguous = false;
    for next_count in 0..=next_len {
        if next_count > best_edits {
            break;
        }
        for old_count in 0..=old_len
            .min(MAX_GLYPH_EDITS - next_count)
            .min(best_edits - next_count)
        {
            let edits = next_count + old_count;
            if edits == 0 || edits > best_edits {
                continue;
            }
            // Unlike a pure glyph tail, a translated quad still needs a real
            // unchanged anchor. End-of-scene is not proof of painter convergence.
            let anchored = next
                .get(n + 1 + next_count)
                .zip(old.get(o + 1 + old_count))
                .is_some_and(|(next, old)| next.geometry_matches(*old));
            if anchored {
                if edits < best_edits {
                    best = Some((next_count, old_count));
                    best_edits = edits;
                    ambiguous = false;
                } else {
                    ambiguous = true;
                }
            }
        }
    }
    if ambiguous { None } else { best }
}

fn glyph_run_bounds(run: &[PaintedPrimitive<'_>]) -> Bounds<ScaledPixels> {
    run.iter()
        .map(|primitive| primitive.bounds())
        .filter(|bounds| !bounds.is_empty())
        .reduce(|bounds, next| bounds.union(&next))
        .unwrap_or_default()
}

fn disjoint_swap(
    next: &[PaintedPrimitive<'_>],
    old: &[PaintedPrimitive<'_>],
    n: usize,
    o: usize,
    width: u32,
    height: u32,
) -> bool {
    let (Some(&next_a), Some(&next_b), Some(&old_a), Some(&old_b)) =
        (next.get(n), next.get(n + 1), old.get(o), old.get(o + 1))
    else {
        return false;
    };
    if !next_a.geometry_matches(old_b) || !next_b.geometry_matches(old_a) {
        return false;
    }
    let (Some(a), Some(b)) = (next_a.commutation_bounds(), next_b.commutation_bounds()) else {
        return false;
    };
    let a = rectangle(a, width, height);
    let b = rectangle(b, width, height);
    // The exact geometry/pipeline matches above give the old pair the same coverage.
    // Expand both by the AA fringe before permitting an adjacent painter-order swap.
    !a.intersects(b.bounds())
}

impl SceneSnapshot {
    fn painter_sequence(&self) -> Option<Vec<PaintedPrimitive<'_>>> {
        let mut sequence = Vec::with_capacity(
            self.quads.len()
                + self.paths.len()
                + self.monochrome_sprites.len()
                + self.subpixel_sprites.len()
                + self.shadows.len()
                + self.underlines.len()
                + self.polychrome_sprites.len(),
        );
        for command in &self.commands {
            let RenderCommand::Batch(batch) = command else {
                return None;
            };
            match batch {
                PrimitiveBatch::Quads { range, smoothed } => sequence.extend(
                    self.quads
                        .get(range.clone())?
                        .iter()
                        .map(|p| PaintedPrimitive::Quad(p, *smoothed)),
                ),
                PrimitiveBatch::Shadows { range, smoothed } => sequence.extend(
                    self.shadows
                        .get(range.clone())?
                        .iter()
                        .map(|p| PaintedPrimitive::Shadow(p, *smoothed)),
                ),
                PrimitiveBatch::Paths {
                    range,
                    rasterization_vertex_count,
                    sprite_count,
                } => sequence.extend(self.paths.get(range.clone())?.iter().map(|p| {
                    PaintedPrimitive::Path(
                        p,
                        range.start,
                        range.end,
                        *rasterization_vertex_count,
                        *sprite_count,
                    )
                })),
                PrimitiveBatch::Underlines(range) => sequence.extend(
                    self.underlines
                        .get(range.clone())?
                        .iter()
                        .map(PaintedPrimitive::Underline),
                ),
                PrimitiveBatch::MonochromeSprites { range, texture_id } => sequence.extend(
                    self.monochrome_sprites
                        .get(range.clone())?
                        .iter()
                        .map(|p| PaintedPrimitive::Monochrome(p, *texture_id)),
                ),
                PrimitiveBatch::SubpixelSprites { range, texture_id } => sequence.extend(
                    self.subpixel_sprites
                        .get(range.clone())?
                        .iter()
                        .map(|p| PaintedPrimitive::Subpixel(p, *texture_id)),
                ),
                PrimitiveBatch::PolychromeSprites {
                    range,
                    texture_id,
                    smoothed,
                } => sequence.extend(
                    self.polychrome_sprites
                        .get(range.clone())?
                        .iter()
                        .map(|p| PaintedPrimitive::Polychrome(p, *smoothed, *texture_id)),
                ),
                _ => return None,
            }
        }
        Some(sequence)
    }
    /// Capture a finished scene when its geometry is supported and fits the snapshot budget.
    pub fn capture(scene: &Scene, atlas_generation: u64, width: u32, height: u32) -> Option<Self> {
        Self::capture_checked(scene, atlas_generation, width, height).ok()
    }

    /// Capture a supported finished scene, retaining a precise reason for full redraw fallback.
    pub fn capture_checked(
        scene: &Scene,
        atlas_generation: u64,
        width: u32,
        height: u32,
    ) -> Result<Self, SceneSnapshotUnsupportedReason> {
        use SceneSnapshotUnsupportedReason as Unsupported;
        // Surfaces can update without primitive changes. Filters sample outside local bounds.
        if width == 0 || height == 0 {
            return Err(Unsupported::InvalidTargetSize);
        }
        if !scene.surfaces.is_empty() {
            return Err(Unsupported::Surface);
        }
        if !scene.backdrop_filters.is_empty() {
            return Err(Unsupported::BackdropFilter);
        }
        if !scene.filter_boundaries.is_empty() {
            return Err(Unsupported::FilterBoundary);
        }
        snapshot_bytes(scene).ok_or(Unsupported::SnapshotBudget)?;
        let visible = |bounds, mask| finite_bounds(bounds) && finite_bounds(mask);
        for (bounds, mask, transformation) in scene
            .monochrome_sprites
            .iter()
            .map(|p| (p.bounds, p.content_mask.bounds, p.transformation))
            .chain(
                scene
                    .subpixel_sprites
                    .iter()
                    .map(|p| (p.bounds, p.content_mask.bounds, p.transformation)),
            )
        {
            // f32::min and max skip NaN, so check what transformed() can't.
            let matrix = transformation.rotation_scale.iter().flatten();
            if !visible(bounds, mask)
                || !matrix
                    .chain(&transformation.translation)
                    .all(|value| value.is_finite())
            {
                return Err(Unsupported::NonfiniteGeometry);
            }
        }
        if scene
            .quads
            .iter()
            .any(|p| !visible(p.bounds, p.content_mask.bounds))
            || scene.paths.iter().any(|p| {
                !visible(p.bounds, p.content_mask.bounds)
                    || p.vertices.iter().any(|vertex| {
                        !vertex.xy_position.x.0.is_finite()
                            || !vertex.xy_position.y.0.is_finite()
                            || !vertex.st_position.x.is_finite()
                            || !vertex.st_position.y.is_finite()
                            || !finite_bounds(vertex.content_mask.bounds)
                    })
            })
            || scene.shadows.iter().any(|p| {
                !visible(p.bounds, p.content_mask.bounds)
                    || (p.inset.is_enabled() && !finite_bounds(p.element_bounds))
                    || !p.blur_radius.0.is_finite()
                    || p.blur_radius.0 < 0.0
            })
            || scene.underlines.iter().any(|p| {
                !visible(p.bounds, p.content_mask.bounds)
                    || !p.thickness.0.is_finite()
                    || p.thickness.0 < 0.0
            })
            || scene
                .polychrome_sprites
                .iter()
                .any(|p| !visible(p.bounds, p.content_mask.bounds))
        {
            return Err(Unsupported::NonfiniteGeometry);
        }
        Ok(Self {
            commands: scene.render_commands().to_vec(),
            quads: scene.quads.clone(),
            paths: scene.paths.clone(),
            monochrome_sprites: scene.monochrome_sprites.clone(),
            subpixel_sprites: scene.subpixel_sprites.clone(),
            shadows: scene.shadows.clone(),
            underlines: scene.underlines.clone(),
            polychrome_sprites: scene.polychrome_sprites.clone(),
            atlas_generation,
            width,
            height,
        })
    }

    /// Verify painter order and classify exact reuse or bounded local colour/quad changes.
    pub fn compare(&self, previous: &Self) -> Damage {
        self.compare_checked(previous).unwrap_or(Damage::Full)
    }

    /// Compare supported scenes, reporting a static reason when full rendering is required.
    pub fn compare_checked(&self, previous: &Self) -> Result<Damage, &'static str> {
        self.compare_impl(previous)
    }

    /// Prototype fallback that proves equality of exact local painter sequences.
    ///
    /// Global numeric painter ranks may differ, but unchanged descriptors remain
    /// exact and path rasterization groups retain their geometry and membership.
    /// Every tile with a different painted sequence is cleared and replayed.
    /// Explicit budgets bound temporary memory, matching effort and damage area.
    pub fn compare_tiled_checked(&self, previous: &Self) -> Result<Damage, &'static str> {
        if self.atlas_generation != previous.atlas_generation {
            return Err("atlas-generation-change");
        }
        if self.width != previous.width || self.height != previous.height {
            return Err("target-extent-change");
        }
        let columns = self.width.div_ceil(DAMAGE_TILE_SIZE);
        let rows = self.height.div_ceil(DAMAGE_TILE_SIZE);
        let tiles = (columns as usize)
            .checked_mul(rows as usize)
            .filter(|count| *count > 0 && *count <= MAX_DAMAGE_TILES)
            .ok_or("tile-count-budget")?;
        let count = |snapshot: &Self| {
            snapshot.quads.len()
                + snapshot.paths.len()
                + snapshot.shadows.len()
                + snapshot.underlines.len()
                + snapshot.monochrome_sprites.len()
                + snapshot.subpixel_sprites.len()
                + snapshot.polychrome_sprites.len()
        };
        if count(self) > MAX_TILE_PRIMITIVES || count(previous) > MAX_TILE_PRIMITIVES {
            return Err("tile-primitive-budget");
        }
        let (Some(next), Some(old)) = (self.painter_sequence(), previous.painter_sequence()) else {
            return Err("unsupported-painter-sequence");
        };
        if next.len() > MAX_TILE_PRIMITIVES || old.len() > MAX_TILE_PRIMITIVES {
            return Err("tile-primitive-budget");
        }
        let mut comparisons_left = MAX_TILE_COMPARISONS;
        let mut compare = |next: PaintedPrimitive<'_>,
                           old: PaintedPrimitive<'_>,
                           exact: bool|
         -> Result<bool, &'static str> {
            comparisons_left = comparisons_left
                .checked_sub(next.comparison_cost(old))
                .ok_or("tile-comparison-budget")?;
            Ok(if exact {
                next.exact_matches(old)
            } else {
                next.geometry_matches(old)
            })
        };
        let mut indexed_old: Vec<_> = old
            .iter()
            .enumerate()
            .map(|(index, p)| (p.descriptor_key(), index))
            .collect();
        indexed_old.sort_unstable();
        let mut matched_old = vec![false; old.len()];
        let old_ids: Vec<u32> = (0..old.len() as u32).collect();
        let mut next_ids = Vec::with_capacity(next.len());
        let mut quad_edits = 0;
        let mut glyph_edits = 0;
        let mut unmatched = |primitive: PaintedPrimitive<'_>| -> Result<(), &'static str> {
            // An edited primitive gets a fresh ID, so every tile covering its old
            // or new extent repaints. A path's tiles also replay the rest of its
            // rasterization group.
            match primitive {
                PaintedPrimitive::Monochrome(..) | PaintedPrimitive::Subpixel(..) => {
                    glyph_edits += 1;
                    if glyph_edits > MAX_GLYPH_EDITS {
                        return Err("tile-glyph-edit-budget");
                    }
                }
                _ => {
                    quad_edits += 1;
                    if quad_edits > MAX_QUAD_EDITS {
                        return Err("tile-quad-edit-budget");
                    }
                }
            }
            Ok(())
        };
        for (next_index, current) in next.iter().copied().enumerate() {
            let key = current.descriptor_key();
            let start = indexed_old.partition_point(|(old_key, _)| old_key < &key);
            let end = indexed_old.partition_point(|(old_key, _)| old_key <= &key);
            let mut matched = None;
            let mut recolored = None;
            for &(_, old_index) in &indexed_old[start..end] {
                // Count every candidate, including consumed duplicates. Repeated
                // descriptors cannot cause an unbounded quadratic bucket scan.
                if compare(current, old[old_index], true)? && !matched_old[old_index] {
                    matched = Some(old_index);
                    break;
                }
            }
            if matched.is_none() {
                for &(_, old_index) in &indexed_old[start..end] {
                    if compare(current, old[old_index], false)? && !matched_old[old_index] {
                        recolored = Some(old_index);
                        break;
                    }
                }
            }
            if let Some(old_index) = matched {
                matched_old[old_index] = true;
                next_ids.push(old_index as u32);
            } else if let Some(old_index) = recolored {
                // Existing strict comparison does not count local color changes as
                // structural edits. Consume the same geometry but use a fresh paint
                // token so every affected tile is still cleared and replayed.
                matched_old[old_index] = true;
                next_ids.push((old.len() + next_index) as u32);
            } else {
                unmatched(current)?;
                next_ids.push((old.len() + next_index) as u32);
            }
        }
        for (index, primitive) in old.iter().copied().enumerate() {
            if !matched_old[index] {
                unmatched(primitive)?;
            }
        }
        let mut references_left = MAX_TILE_REFERENCES;
        // Prove the combined reference budget before either CSR index allocation.
        let old_coverage = tile_coverages(&old, self.width, self.height, &mut references_left)?;
        let next_coverage = tile_coverages(&next, self.width, self.height, &mut references_left)?;
        let old_tiles = tile_references(&old_coverage, &old_ids, columns, tiles);
        let next_tiles = tile_references(&next_coverage, &next_ids, columns, tiles);
        let mut dirty = vec![false; tiles];
        for tile in 0..tiles {
            let old_ids = &old_tiles.ids[old_tiles.offsets[tile]..old_tiles.offsets[tile + 1]];
            let next_ids = &next_tiles.ids[next_tiles.offsets[tile]..next_tiles.offsets[tile + 1]];
            if old_ids.len() != next_ids.len() {
                dirty[tile] = true;
                continue;
            }
            for (next, old) in next_ids.iter().zip(old_ids) {
                comparisons_left = comparisons_left
                    .checked_sub(1)
                    .ok_or("tile-comparison-budget")?;
                if next != old {
                    dirty[tile] = true;
                    break;
                }
            }
        }
        tile_regions(&dirty, columns, self.width, self.height)
    }

    fn compare_impl(&self, previous: &Self) -> Result<Damage, &'static str> {
        if self.atlas_generation != previous.atlas_generation {
            return Err("atlas-generation-change");
        }
        if self.width != previous.width || self.height != previous.height {
            return Err("target-extent-change");
        }
        if self.paths.len() != previous.paths.len() {
            return Err("path-count-change");
        }
        if self.shadows.len() != previous.shadows.len() {
            return Err("shadow-count-change");
        }
        if self.underlines.len() != previous.underlines.len() {
            return Err("underline-count-change");
        }
        if self.polychrome_sprites.len() != previous.polychrome_sprites.len() {
            return Err("polychrome-count-change");
        }
        let mut regions: Vec<DamageRect> = Vec::new();
        let mut damage = |bounds| -> bool {
            let mut region = rectangle(bounds, self.width, self.height);
            if region.width == 0 || region.height == 0 {
                return true;
            }
            let mut index = 0;
            while index < regions.len() {
                if region.intersects(regions[index].bounds()) {
                    region = region.union(regions.remove(index));
                    index = 0;
                } else {
                    index += 1;
                }
            }
            regions.push(region);
            regions.len() <= MAX_REGIONS
        };
        let (Some(next), Some(old)) = (self.painter_sequence(), previous.painter_sequence()) else {
            return Err("unsupported-painter-sequence");
        };
        let mut n = 0;
        let mut o = 0;
        let mut edits = 0;
        let mut glyph_edits = 0;
        let mut commutations = 0;
        while n < next.len() || o < old.len() {
            match (next.get(n).copied(), old.get(o).copied()) {
                (Some(current), Some(prior)) if current.geometry_matches(prior) => {
                    if current.colour_changed(prior) && !damage(current.bounds()) {
                        return Err("damage-region-budget");
                    }
                    n += 1;
                    o += 1;
                }
                (Some(current), Some(prior)) => {
                    if disjoint_swap(&next, &old, n, o, self.width, self.height) {
                        let next_b = next[n + 1];
                        let old_b = old[o + 1];
                        if (current.colour_changed(old_b) && !damage(current.bounds()))
                            || (next_b.colour_changed(prior) && !damage(next_b.bounds()))
                        {
                            return Err("damage-region-budget");
                        }
                        n += 2;
                        o += 2;
                        commutations += 1;
                        if commutations > MAX_COMMUTATIONS {
                            return Err("commutation-budget");
                        }
                        continue;
                    }
                    let inserted_quad_glyphs = current
                        .is_quad()
                        .then(|| quad_glyph_run_to_anchor(&next, n, prior))
                        .flatten();
                    let removed_quad_glyphs = prior
                        .is_quad()
                        .then(|| quad_glyph_run_to_anchor(&old, o, current))
                        .flatten();
                    if inserted_quad_glyphs.is_some() && removed_quad_glyphs.is_some() {
                        return Err("ambiguous-quad-glyph-reorder");
                    }
                    let paired_quad_glyphs = paired_quad_glyph_runs(&next, &old, n, o);
                    if paired_quad_glyphs.is_some()
                        && (inserted_quad_glyphs.is_some() || removed_quad_glyphs.is_some())
                    {
                        return Err("ambiguous-quad-glyph-edit");
                    }
                    if inserted_quad_glyphs.is_some()
                        || removed_quad_glyphs.is_some()
                        || paired_quad_glyphs.is_some()
                    {
                        let (next_quad, next_glyphs, old_quad, old_glyphs) =
                            if let Some(count) = inserted_quad_glyphs {
                                (true, count, false, 0)
                            } else if let Some(count) = removed_quad_glyphs {
                                (false, 0, true, count)
                            } else {
                                let (next_count, old_count) = paired_quad_glyphs.unwrap();
                                (true, next_count, true, old_count)
                            };
                        edits += usize::from(next_quad) + usize::from(old_quad);
                        glyph_edits += next_glyphs + old_glyphs;
                        if edits > MAX_QUAD_EDITS {
                            return Err("quad-edit-budget");
                        }
                        if glyph_edits > MAX_GLYPH_EDITS {
                            return Err("glyph-edit-budget");
                        }
                        // Damage each old/new fill and badge separately. In particular,
                        // no bounding union spans the space between distant hovered rows.
                        if (next_quad && !damage(current.bounds()))
                            || (old_quad && !damage(prior.bounds()))
                            || (next_glyphs > 0
                                && !damage(glyph_run_bounds(&next[n + 1..n + 1 + next_glyphs])))
                            || (old_glyphs > 0
                                && !damage(glyph_run_bounds(&old[o + 1..o + 1 + old_glyphs])))
                        {
                            return Err("damage-region-budget");
                        }
                        n += usize::from(next_quad) + next_glyphs;
                        o += usize::from(old_quad) + old_glyphs;
                        continue;
                    }
                    let inserted_glyphs = current
                        .is_glyph()
                        .then(|| glyph_run_to_anchor(&next, n, prior))
                        .flatten();
                    let removed_glyphs = prior
                        .is_glyph()
                        .then(|| glyph_run_to_anchor(&old, o, current))
                        .flatten();
                    if inserted_glyphs.is_some() && removed_glyphs.is_some() {
                        // An overlapping reorder cannot masquerade as two edits.
                        return Err("ambiguous-glyph-reorder");
                    }
                    if let Some(count) = inserted_glyphs {
                        glyph_edits += count;
                        if glyph_edits > MAX_GLYPH_EDITS {
                            return Err("glyph-edit-budget");
                        }
                        if !damage(glyph_run_bounds(&next[n..n + count])) {
                            return Err("damage-region-budget");
                        }
                        n += count;
                        continue;
                    }
                    if let Some(count) = removed_glyphs {
                        glyph_edits += count;
                        if glyph_edits > MAX_GLYPH_EDITS {
                            return Err("glyph-edit-budget");
                        }
                        if !damage(glyph_run_bounds(&old[o..o + count])) {
                            return Err("damage-region-budget");
                        }
                        o += count;
                        continue;
                    }
                    if current.is_glyph() && prior.is_glyph() {
                        if let Some((next_count, old_count)) = paired_glyph_runs(&next, &old, n, o)
                        {
                            glyph_edits += next_count + old_count;
                            if glyph_edits > MAX_GLYPH_EDITS {
                                return Err("glyph-edit-budget");
                            }
                            if !damage(glyph_run_bounds(&next[n..n + next_count]))
                                || !damage(glyph_run_bounds(&old[o..o + old_count]))
                            {
                                return Err("damage-region-budget");
                            }
                            n += next_count;
                            o += old_count;
                            continue;
                        }
                    }
                    // Permit one sparse inserted/removed quad only when the immediately
                    // following exact geometry re-establishes the unchanged painter order.
                    let inserted = current.is_quad()
                        && next.get(n + 1).is_some_and(|p| p.geometry_matches(prior));
                    let removed = prior.is_quad()
                        && old.get(o + 1).is_some_and(|p| current.geometry_matches(*p));
                    if inserted && removed {
                        return Err("ambiguous-quad-reorder");
                    }
                    if !inserted && !removed {
                        // A changed/moved quad affects only its old and new clipped
                        // raster coverage. Keep a strict unchanged following anchor,
                        // possibly two disjoint paints whose adjacent swap is verified
                        // on the next iteration. This also supports inherited row masks
                        // and rounded/bordered shapes without assuming they are equal.
                        let paired = current.is_quad()
                            && prior.is_quad()
                            && (next
                                .get(n + 1)
                                .zip(old.get(o + 1))
                                .is_some_and(|(next, old)| next.geometry_matches(*old))
                                || disjoint_swap(
                                    &next,
                                    &old,
                                    n + 1,
                                    o + 1,
                                    self.width,
                                    self.height,
                                ));
                        if !paired {
                            return Err(match (current, prior) {
                                (PaintedPrimitive::Path(..), PaintedPrimitive::Path(..)) => {
                                    "path-geometry-or-group-change"
                                }
                                (PaintedPrimitive::Shadow(..), PaintedPrimitive::Shadow(..)) => {
                                    "shadow-geometry-or-order-change"
                                }
                                (
                                    PaintedPrimitive::Underline(..),
                                    PaintedPrimitive::Underline(..),
                                ) => "underline-geometry-or-order-change",
                                (
                                    PaintedPrimitive::Polychrome(..),
                                    PaintedPrimitive::Polychrome(..),
                                ) => "polychrome-geometry-or-order-change",
                                _ if current.is_glyph() || prior.is_glyph() => {
                                    "glyph-run-anchor-change"
                                }
                                _ => current.quad_mismatch_reason(prior),
                            });
                        }
                        if !damage(current.bounds()) || !damage(prior.bounds()) {
                            return Err("damage-region-budget");
                        }
                        n += 1;
                        o += 1;
                        edits += 2;
                    } else if inserted {
                        if !damage(current.bounds()) {
                            return Err("damage-region-budget");
                        }
                        n += 1;
                        edits += 1;
                    } else {
                        if !damage(prior.bounds()) {
                            return Err("damage-region-budget");
                        }
                        o += 1;
                        edits += 1;
                    }
                }
                (Some(current), None) if current.is_quad() => {
                    if !damage(current.bounds()) {
                        return Err("damage-region-budget");
                    }
                    n += 1;
                    edits += 1;
                }
                (None, Some(prior)) if prior.is_quad() => {
                    if !damage(prior.bounds()) {
                        return Err("damage-region-budget");
                    }
                    o += 1;
                    edits += 1;
                }
                (Some(current), None) if current.is_glyph() => {
                    let count = glyph_prefix_len(&next, n);
                    glyph_edits += count;
                    if glyph_edits > MAX_GLYPH_EDITS {
                        return Err("glyph-edit-budget");
                    }
                    if !damage(glyph_run_bounds(&next[n..n + count])) {
                        return Err("damage-region-budget");
                    }
                    n += count;
                }
                (None, Some(prior)) if prior.is_glyph() => {
                    let count = glyph_prefix_len(&old, o);
                    glyph_edits += count;
                    if glyph_edits > MAX_GLYPH_EDITS {
                        return Err("glyph-edit-budget");
                    }
                    if !damage(glyph_run_bounds(&old[o..o + count])) {
                        return Err("damage-region-budget");
                    }
                    o += count;
                }
                _ => return Err("unsupported-primitive-tail"),
            }
            if edits > MAX_QUAD_EDITS {
                return Err("quad-edit-budget");
            }
        }
        if regions.is_empty() {
            return Ok(Damage::Reuse);
        }
        let damaged: u128 = regions
            .iter()
            .map(|region| region.width as u128 * region.height as u128)
            .sum();
        if damaged * 100 > self.width as u128 * self.height as u128 * 40 {
            Err("damage-area-budget")
        } else {
            Ok(Damage::Partial(regions))
        }
    }
}

/// What a shadow rasterizes, as the shader's `shadow_geometry`: the element for
/// an inset shadow, otherwise its bounds grown by the blur's cutoff.
fn shadow_geometry(shadow: &Shadow) -> Bounds<ScaledPixels> {
    if shadow.inset.is_enabled() {
        shadow.element_bounds
    } else {
        shadow.bounds.dilate(ScaledPixels(
            GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * shadow.blur_radius.0,
        ))
    }
}

/// The region a sprite covers after its transformation, which the shader
/// applies as `rotation_scale * position + translation`.
fn transformed(
    bounds: Bounds<ScaledPixels>,
    matrix: &TransformationMatrix,
) -> Bounds<ScaledPixels> {
    if *matrix == TransformationMatrix::unit() {
        return bounds;
    }
    let [[a, b], [c, d]] = matrix.rotation_scale;
    let [tx, ty] = matrix.translation;
    let (x0, y0) = (bounds.origin.x.0, bounds.origin.y.0);
    let (x1, y1) = (x0 + bounds.size.width.0, y0 + bounds.size.height.0);
    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
    let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
        let (x, y) = (a * x + b * y + tx, c * x + d * y + ty);
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    Bounds::from_corners(
        point(ScaledPixels(min_x), ScaledPixels(min_y)),
        point(ScaledPixels(max_x), ScaledPixels(max_y)),
    )
}

/// Whether a current painter batch can affect a damage rectangle.
pub fn batch_intersects(scene: &Scene, batch: &PrimitiveBatch, region: DamageRect) -> bool {
    let visible = |bounds: Bounds<ScaledPixels>, mask: Bounds<ScaledPixels>| {
        region.intersects(bounds.intersect(&mask).dilate(ScaledPixels(1.0)))
    };
    match batch {
        PrimitiveBatch::Quads { range, .. } => scene.quads[range.clone()]
            .iter()
            .any(|p| visible(p.bounds, p.content_mask.bounds)),
        PrimitiveBatch::Shadows { range, .. } => scene.shadows[range.clone()]
            .iter()
            .any(|p| visible(shadow_geometry(p), p.content_mask.bounds)),
        PrimitiveBatch::Paths { range, .. } => scene.paths[range.clone()]
            .iter()
            .any(|p| region.intersects(p.clipped_bounds().dilate(ScaledPixels(1.0)))),
        PrimitiveBatch::Underlines(range) => scene.underlines[range.clone()].iter().any(|p| {
            visible(
                p.bounds.dilate(ScaledPixels(p.thickness.0 * 4.0)),
                p.content_mask.bounds,
            )
        }),
        PrimitiveBatch::MonochromeSprites { range, .. } => {
            scene.monochrome_sprites[range.clone()].iter().any(|p| {
                visible(
                    transformed(p.bounds, &p.transformation),
                    p.content_mask.bounds,
                )
            })
        }
        PrimitiveBatch::SubpixelSprites { range, .. } => {
            scene.subpixel_sprites[range.clone()].iter().any(|p| {
                visible(
                    transformed(p.bounds, &p.transformation),
                    p.content_mask.bounds,
                )
            })
        }
        PrimitiveBatch::PolychromeSprites { range, .. } => scene.polychrome_sprites[range.clone()]
            .iter()
            .any(|p| visible(p.bounds, p.content_mask.bounds)),
        _ => true,
    }
}

/// Make the fill that resets `region` to the target's raw RGBA clear colour
/// before the intersecting painter batches replay.
pub fn clear_quad(region: DamageRect, clear_color: [f32; 4]) -> Quad {
    let [r, g, b, a] = clear_color;
    Quad {
        bounds: region.bounds(),
        content_mask: ContentMask {
            bounds: region.bounds(),
            ..Default::default()
        },
        background: gpui::rgb_to_hsla(gpui::Rgba::new(r, g, b, a)).into(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::Hsla;
    fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(width), ScaledPixels(height)),
        }
    }
    fn scene(color: Hsla) -> Scene {
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            bounds: bounds(0.0, 10.0, 200.0, 20.0),
            content_mask: ContentMask {
                bounds: bounds(0.0, 0.0, 200.0, 200.0),
                ..Default::default()
            },
            background: color.into(),
            ..Default::default()
        });
        scene.finish();
        scene
    }
    fn snapshot(scene: &Scene) -> SceneSnapshot {
        SceneSnapshot::capture(scene, 1, 200, 200).unwrap()
    }

    fn glyph(index: u32, x: f32, y: f32, subpixel: bool) -> gpui::Primitive {
        let tile = gpui::AtlasTile {
            texture_id: gpui::AtlasTextureId {
                index: 0,
                kind: if subpixel {
                    gpui::AtlasTextureKind::Subpixel
                } else {
                    gpui::AtlasTextureKind::Monochrome
                },
            },
            tile_id: gpui::TileId(index),
            padding: 0,
            bounds: Bounds::new(
                point(gpui::DevicePixels(0), gpui::DevicePixels(0)),
                size(gpui::DevicePixels(8), gpui::DevicePixels(8)),
            ),
        };
        let mask = ContentMask {
            bounds: bounds(0.0, 0.0, 200.0, 200.0),
            ..Default::default()
        };
        if subpixel {
            SubpixelSprite {
                order: 0,
                padding: 0,
                bounds: bounds(x, y, 8.0, 10.0),
                content_mask: mask,
                color: gpui::white().into(),
                tile,
                transformation: Default::default(),
            }
            .into()
        } else {
            MonochromeSprite {
                order: 0,
                padding: 0,
                bounds: bounds(x, y, 8.0, 10.0),
                content_mask: mask,
                color: gpui::white().into(),
                tile,
                transformation: Default::default(),
            }
            .into()
        }
    }

    fn badge_scene(hover: Option<(usize, usize)>, subpixel: bool) -> Scene {
        let mut scene = Scene::default();
        for row in 0..2 {
            let y = if row == 0 { 20.0 } else { 165.0 };
            scene.push_layer(bounds(0.0, y, 200.0, 14.0));
            if hover.is_some_and(|(hovered, _)| hovered == row) {
                scene.insert_primitive(Quad {
                    bounds: bounds(0.0, y, 200.0, 14.0),
                    content_mask: ContentMask {
                        bounds: bounds(0.0, 0.0, 200.0, 200.0),
                        ..Default::default()
                    },
                    background: gpui::rgba(0x55779944).into(),
                    ..Default::default()
                });
            }
            // A stable graph-node quad and static text anchor remain present in
            // every actual finished row; hover fills and badges are omitted at rest.
            scene.insert_primitive(Quad {
                bounds: bounds(8.0, y + 3.0, 3.0, 3.0),
                content_mask: ContentMask {
                    bounds: bounds(0.0, 0.0, 200.0, 200.0),
                    ..Default::default()
                },
                background: gpui::white().into(),
                ..Default::default()
            });
            if let Some((hovered, count)) = hover
                && hovered == row
            {
                for index in 0..count {
                    scene.insert_primitive(glyph(
                        index as u32 + 1,
                        40.0 + index as f32 * 7.0,
                        y + 1.0,
                        subpixel,
                    ));
                }
            }
            scene.insert_primitive(glyph(999, 180.0, y + 1.0, subpixel));
            scene.pop_layer();
        }
        scene.finish();
        scene
    }

    fn joint_badge_scene(
        hover: Option<(usize, usize)>,
        subpixel: bool,
        interleaved_graph: bool,
        anchor: bool,
    ) -> Scene {
        let mut scene = Scene::default();
        let mask = ContentMask {
            bounds: bounds(0.0, 0.0, 200.0, 200.0),
            ..Default::default()
        };
        scene.insert_primitive(Quad {
            bounds: bounds(0.0, 0.0, 200.0, 200.0),
            content_mask: mask,
            background: gpui::rgba(0x22334477).into(),
            ..Default::default()
        });
        scene.raise_order_floor();
        if let Some((row, _)) = hover {
            let y = if row == 0 { 20.0 } else { 165.0 };
            scene.insert_primitive(Quad {
                bounds: bounds(0.0, y, 200.0, 14.0),
                // Actual element masks may change with the hovered row, not
                // merely its bounds origin. Both old/new masks need coverage.
                content_mask: ContentMask {
                    bounds: bounds(2.0, y + 0.25, 194.0, 13.25),
                    fade_out: gpui::Edges::all(ScaledPixels(0.75)),
                },
                background: gpui::rgba(0x55779944).into(),
                corner_radii: gpui::Corners::all(ScaledPixels(2.5)),
                ..Default::default()
            });
        }
        scene.raise_order_floor();
        if interleaved_graph {
            let mut path = Path::new(point(gpui::px(8.0), gpui::px(20.0)));
            path.line_to(point(gpui::px(18.0), gpui::px(20.0)));
            path.line_to(point(gpui::px(8.0), gpui::px(30.0)));
            path.line_to(point(gpui::px(8.0), gpui::px(20.0)));
            let mut path = path.scale(1.0);
            path.content_mask = mask;
            path.color = gpui::white().into();
            scene.insert_primitive(path);
            scene.raise_order_floor();
        }
        if let Some((row, count)) = hover {
            let y = if row == 0 { 20.0 } else { 165.0 };
            for index in 0..count {
                scene.insert_primitive(glyph(
                    index as u32 + 1,
                    40.0 + index as f32 * 7.0,
                    y + 1.0,
                    subpixel,
                ));
            }
        }
        if anchor {
            scene.raise_order_floor();
            scene.insert_primitive(Quad {
                bounds: bounds(10.0, 6.0, 100.0, 5.0),
                content_mask: mask,
                background: gpui::rgba(0xee884466).into(),
                ..Default::default()
            });
        }
        scene.finish();
        scene
    }

    fn tiled_row_scene(hover: Option<usize>, subpixel: bool) -> Scene {
        let mut scene = Scene::default();
        let mask = ContentMask {
            bounds: bounds(0.0, 0.0, 512.0, 1024.0),
            ..Default::default()
        };
        scene.insert_primitive(Quad {
            bounds: mask.bounds,
            content_mask: mask,
            background: gpui::rgba(0x22334488).into(),
            ..Default::default()
        });
        for row in 0..20 {
            let y = 16.25 + 40.0 * row as f32;
            // Match ordinary history rows: no explicit per-row painter layer.
            if hover == Some(row) {
                scene.insert_primitive(Quad {
                    bounds: bounds(0.0, y, 512.0, 18.5),
                    content_mask: ContentMask {
                        bounds: bounds(0.5, y + 0.25, 511.0, 18.0),
                        fade_out: gpui::Edges::all(ScaledPixels(0.75)),
                    },
                    background: gpui::rgba(0x55779944).into(),
                    corner_radii: gpui::Corners::all(ScaledPixels(2.25)),
                    ..Default::default()
                });
            }
            // Reproduce the live multi-Quad row blocks (66px/4px/24px), rather
            // than a single small node that the strict anchor path can accept.
            for (x, width) in [(134.0, 66.0), (200.0, 4.0), (410.0, 24.0)] {
                scene.insert_primitive(Quad {
                    bounds: bounds(x, y, width, 18.5),
                    content_mask: mask,
                    background: if hover == Some(row) {
                        gpui::rgba(0x55779999)
                    } else {
                        gpui::rgba(0x22334499)
                    }
                    .into(),
                    ..Default::default()
                });
            }
            for index in 0..20 {
                let mut primitive = glyph(index + 1, 12.0 + index as f32 * 19.0, y + 2.0, subpixel);
                match &mut primitive {
                    gpui::Primitive::MonochromeSprite(p) => p.content_mask = mask,
                    gpui::Primitive::SubpixelSprite(p) => p.content_mask = mask,
                    _ => unreachable!(),
                }
                scene.insert_primitive(primitive);
            }
        }
        scene.finish();
        scene
    }

    fn tiled_snapshot(scene: &Scene) -> SceneSnapshot {
        SceneSnapshot::capture(scene, 1, 512, 1024).unwrap()
    }

    fn full_mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: bounds(0.0, 0.0, 512.0, 1024.0),
            ..Default::default()
        }
    }

    fn blurred_shadow(x: f32) -> Shadow {
        Shadow {
            order: 0,
            blur_radius: ScaledPixels(4.0),
            bounds: bounds(x, 100.0, 20.0, 20.0),
            corner_radii: Default::default(),
            content_mask: full_mask(),
            color: gpui::black().into(),
            element_bounds: bounds(x, 100.0, 20.0, 20.0),
            element_corner_radii: Default::default(),
            inset: Default::default(),
            corner_smoothing: 0.0,
        }
    }

    #[test]
    fn blurred_shadows_damage_the_shader_blur_extent() {
        let mut scene = Scene::default();
        scene.insert_primitive(blurred_shadow(100.0));
        scene.finish();
        let batch = scene
            .render_commands()
            .iter()
            .find_map(|command| match command {
                RenderCommand::Batch(batch @ PrimitiveBatch::Shadows { .. }) => Some(batch.clone()),
                _ => None,
            })
            .unwrap();
        let column = |x| DamageRect {
            x,
            y: 100,
            width: 1,
            height: 1,
        };
        // The shader draws 3 standard deviations, 12px here, plus the 1px fringe.
        assert!(batch_intersects(&scene, &batch, column(87)));
        assert!(!batch_intersects(&scene, &batch, column(85)));
    }

    #[test]
    fn transformed_sprites_cover_their_rotated_bounds() {
        let matrix = TransformationMatrix::unit()
            .translate(point(ScaledPixels(110.0), ScaledPixels(105.0)))
            .rotate(gpui::radians(std::f32::consts::FRAC_PI_2))
            .translate(point(ScaledPixels(-110.0), ScaledPixels(-105.0)));
        let covered = transformed(bounds(100.0, 100.0, 20.0, 10.0), &matrix);
        let close = |a: ScaledPixels, b: f32| (a.0 - b).abs() < 1e-3;
        assert!(close(covered.origin.x, 105.0) && close(covered.origin.y, 95.0));
        assert!(close(covered.size.width, 10.0) && close(covered.size.height, 20.0));
    }

    #[test]
    fn moved_shadows_and_paths_repaint_only_their_tiles() {
        let path = |x: f32| {
            let mut path = Path::new(point(gpui::px(x), gpui::px(600.0)));
            path.line_to(point(gpui::px(x + 20.0), gpui::px(600.0)));
            path.line_to(point(gpui::px(x), gpui::px(620.0)));
            let mut path = path.scale(1.0);
            path.content_mask = full_mask();
            path.color = gpui::white().into();
            path
        };
        let scene = |x: f32| {
            let mut scene = Scene::default();
            scene.insert_primitive(Quad {
                bounds: bounds(0.0, 0.0, 512.0, 1024.0),
                content_mask: full_mask(),
                background: gpui::blue().into(),
                ..Default::default()
            });
            scene.insert_primitive(blurred_shadow(x));
            scene.insert_primitive(path(x));
            scene.finish();
            tiled_snapshot(&scene)
        };
        let first = scene(100.0);
        let Ok(Damage::Partial(regions)) = scene(300.0).compare_tiled_checked(&first) else {
            panic!("moving a blurred shadow and a path repaints only their tiles");
        };
        let covers = |x: u32, y: u32| {
            regions
                .iter()
                .any(|r| r.x <= x && x < r.x + r.width && r.y <= y && y < r.y + r.height)
        };
        // Old and new blur extents, and old and new paths.
        assert!(covers(90, 110) && covers(130, 110) && covers(290, 110) && covers(330, 110));
        assert!(covers(105, 605) && covers(305, 605));
        assert!(!covers(500, 1000));
    }

    #[test]
    fn tiled_normal_bounds_tree_rows_preserve_local_painter_order_and_sparse_damage() {
        let mut strict_rejection_observed = false;
        for subpixel in [false, true] {
            let first = tiled_snapshot(&tiled_row_scene(Some(0), subpixel));
            let last = tiled_snapshot(&tiled_row_scene(Some(19), subpixel));
            strict_rejection_observed |= last.compare_checked(&first).is_err();
            let tiled = last.compare_tiled_checked(&first);
            let Ok(Damage::Partial(regions)) = tiled else {
                panic!(
                    "old=0 current=19 subpixel={subpixel}: local exact painter sequence must qualify, got {tiled:?}"
                );
            };
            assert_eq!(regions.len(), 2);
            assert!(
                regions
                    .iter()
                    .any(|r| r.intersects(bounds(0.0, 16.25, 512.0, 18.5)))
            );
            assert!(
                regions
                    .iter()
                    .any(|r| r.intersects(bounds(0.0, 776.25, 512.0, 18.5)))
            );
            assert!(regions.iter().all(|r| r.height <= 128));
            assert_eq!(last.compare_tiled_checked(&last), Ok(Damage::Reuse));
            let empty_hover = tiled_snapshot(&tiled_row_scene(None, subpixel));
            assert!(
                matches!(
                    empty_hover.compare_tiled_checked(&last),
                    Ok(Damage::Partial(_))
                ),
                "old=19 current=None subpixel={subpixel}"
            );
        }
        assert!(
            strict_rejection_observed,
            "old=0 current=19: at least one complete multi-Quad no-layer scene must expose the strict convergence gate"
        );
    }

    #[test]
    fn tiled_overlapping_order_and_duplicate_descriptors_require_actual_repainting() {
        let mut source = Scene::default();
        let a = Quad {
            bounds: bounds(10.0, 10.0, 16.0, 16.0),
            content_mask: ContentMask {
                bounds: bounds(0.0, 0.0, 512.0, 1024.0),
                ..Default::default()
            },
            background: gpui::rgba(0xff000080).into(),
            ..Default::default()
        };
        let b = Quad {
            background: gpui::rgba(0x0000ff80).into(),
            ..a
        };
        source.insert_primitive(a);
        source.insert_primitive(a);
        source.insert_primitive(b);
        source.finish();
        let first = tiled_snapshot(&source);
        let mut reordered = first.clone();
        reordered.quads.swap(1, 2);
        let Damage::Partial(regions) = reordered.compare_tiled_checked(&first).unwrap() else {
            panic!("interleaving identical overlapping copies changes composited pixels");
        };
        assert_eq!(
            regions,
            vec![DamageRect {
                x: 0,
                y: 0,
                width: 64,
                height: 64
            }]
        );

        let mut disjoint = first;
        disjoint.quads[2].bounds.origin.x.0 = 150.0;
        let mut swapped = disjoint.clone();
        swapped.quads.swap(1, 2);
        assert_eq!(swapped.compare_tiled_checked(&disjoint), Ok(Damage::Reuse));
        swapped.quads[0].bounds.origin.x.0 = -0.0;
        let mut signed_zero = swapped.clone();
        signed_zero.quads[0].bounds.origin.x.0 = 0.0;
        assert_eq!(
            signed_zero.compare_tiled_checked(&swapped),
            Ok(Damage::Reuse)
        );
    }

    #[test]
    fn tiled_path_paint_group_and_vertex_changes_repaint_the_path() {
        let mut scene = Scene::default();
        let mut path = Path::new(point(gpui::px(8.25), gpui::px(8.25)));
        path.line_to(point(gpui::px(28.25), gpui::px(8.25)));
        path.line_to(point(gpui::px(8.25), gpui::px(28.25)));
        path.line_to(point(gpui::px(8.25), gpui::px(8.25)));
        let mut path = path.scale(1.0);
        path.content_mask = ContentMask {
            bounds: bounds(0.0, 0.0, 512.0, 1024.0),
            ..Default::default()
        };
        path.color = gpui::white().into();
        scene.insert_primitive(path);
        scene.finish();
        let first = tiled_snapshot(&scene);
        let mut changed = first.clone();
        changed.paths[0].color = gpui::black().into();
        assert!(
            matches!(
                changed.compare_tiled_checked(&first),
                Ok(Damage::Partial(_))
            ),
            "same pathID with changed paint must receive a different local token"
        );
        let RenderCommand::Batch(PrimitiveBatch::Paths { sprite_count, .. }) =
            &mut changed.commands[0]
        else {
            panic!("path fixture")
        };
        *sprite_count += 1;
        let Ok(Damage::Partial(regions)) = changed.compare_tiled_checked(&first) else {
            panic!("a changed rasterization group repaints the path's tiles");
        };
        assert!(
            regions
                .iter()
                .any(|region| region.intersects(first.paths[0].bounds))
        );
        let mut changed = first.clone();
        changed.paths[0].vertices[0].xy_position.x.0 += 0.25;
        assert!(matches!(
            changed.compare_tiled_checked(&first),
            Ok(Damage::Partial(_))
        ));
    }

    #[test]
    fn tiled_fractional_masks_cover_both_sides_of_tile_edges_and_clip_target_edges() {
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            bounds: bounds(62.25, 62.25, 20.5, 20.5),
            content_mask: ContentMask {
                bounds: bounds(63.75, 63.75, 12.5, 12.5),
                fade_out: gpui::Edges::all(ScaledPixels(1.25)),
            },
            background: gpui::rgba(0xff112280).into(),
            corner_radii: gpui::Corners::all(ScaledPixels(2.25)),
            ..Default::default()
        });
        scene.finish();
        let first = tiled_snapshot(&scene);
        let mut changed = first.clone();
        changed.quads[0].content_mask.fade_out.top.0 += 0.25;
        assert_eq!(
            changed.compare_tiled_checked(&first),
            Ok(Damage::Partial(vec![DamageRect {
                x: 0,
                y: 0,
                width: 128,
                height: 128
            }]))
        );
        let mut edge = first;
        edge.width = 501;
        edge.height = 999;
        edge.quads[0].bounds = bounds(495.0, 992.0, 20.0, 20.0);
        edge.quads[0].content_mask.bounds = bounds(0.0, 0.0, 501.0, 999.0);
        let mut changed = edge.clone();
        changed.quads[0].background = gpui::black().into();
        assert_eq!(
            changed.compare_tiled_checked(&edge),
            Ok(Damage::Partial(vec![DamageRect {
                x: 448,
                y: 960,
                width: 53,
                height: 39
            }]))
        );
    }

    #[test]
    fn tiled_matching_reference_and_edit_budgets_fail_closed() {
        let first = tiled_snapshot(&tiled_row_scene(None, false));
        let mut changed = first.clone();
        changed.atlas_generation += 1;
        assert_eq!(
            changed.compare_tiled_checked(&first),
            Err("atlas-generation-change")
        );
        changed = first.clone();
        changed.width += 1;
        assert_eq!(
            changed.compare_tiled_checked(&first),
            Err("target-extent-change")
        );
        changed = first.clone();
        changed.width = 5000;
        changed.height = 5000;
        assert_eq!(
            changed.compare_tiled_checked(&changed),
            Err("tile-count-budget")
        );
        changed = first.clone();
        changed
            .quads
            .resize(MAX_TILE_PRIMITIVES + 1, changed.quads[0]);
        assert_eq!(
            changed.compare_tiled_checked(&changed),
            Err("tile-primitive-budget")
        );

        let mut duplicates = tiled_snapshot(&scene(gpui::white()));
        duplicates.quads[0].bounds = bounds(10.0, 10.0, 4.0, 4.0);
        duplicates.quads.resize(1024, duplicates.quads[0]);
        duplicates.commands = vec![RenderCommand::Batch(PrimitiveBatch::Quads {
            range: 0..1024,
            smoothed: false,
        })];
        assert_eq!(
            duplicates.compare_tiled_checked(&duplicates),
            Err("tile-comparison-budget")
        );
        duplicates.width = 4096;
        duplicates.height = 4096;
        duplicates.quads.resize(17, duplicates.quads[0]);
        for quad in &mut duplicates.quads {
            quad.bounds = bounds(0.0, 0.0, 4096.0, 4096.0);
            quad.content_mask.bounds = quad.bounds;
        }
        duplicates.commands = vec![RenderCommand::Batch(PrimitiveBatch::Quads {
            range: 0..17,
            smoothed: false,
        })];
        assert_eq!(
            duplicates.compare_tiled_checked(&duplicates),
            Err("tile-reference-budget")
        );

        let mut changed = first.clone();
        for quad in &mut changed.quads[1..6] {
            quad.background = gpui::black().into();
        }
        assert!(
            matches!(
                changed.compare_tiled_checked(&first),
                Ok(Damage::Partial(_))
            ),
            "local cutout recolors do not consume structural edit budget"
        );
        for quad in &mut changed.quads[1..6] {
            quad.bounds.size.width.0 += 0.25;
        }
        assert_eq!(
            changed.compare_tiled_checked(&first),
            Err("tile-quad-edit-budget")
        );
        let mut glyphs = Scene::default();
        for index in 0..MAX_GLYPH_EDITS + 1 {
            glyphs.insert_primitive(glyph(index as u32, 10.0, 10.0, false));
        }
        glyphs.finish();
        let mut empty = Scene::default();
        empty.finish();
        assert_eq!(
            tiled_snapshot(&glyphs).compare_tiled_checked(&tiled_snapshot(&empty)),
            Err("tile-glyph-edit-budget")
        );
    }

    #[test]
    fn tiled_region_and_actual_replay_area_budgets_fail_closed() {
        let columns = 16;
        let mut dirty = vec![false; columns * columns];
        for y in (0..columns).step_by(2).take(8) {
            dirty[y * columns] = true;
        }
        dirty[2] = true;
        assert_eq!(
            tile_regions(&dirty, columns as u32, 1024, 1024),
            Err("tile-region-budget")
        );
        let mut dirty = vec![false; 8 * 16];
        for value in &mut dirty[..8 * 8] {
            *value = true;
        }
        assert_eq!(
            tile_regions(&dirty, 8, 512, 1024),
            Err("tile-damage-area-budget")
        );
        // Horizontal tiles form one replay run, then vertically merge before the
        // region cap. The actual rectangle area is the final budget authority.
        let mut dirty = vec![false; 16 * 16];
        for y in 0..4 {
            for x in 0..16 {
                dirty[y * 16 + x] = true;
            }
        }
        assert_eq!(
            tile_regions(&dirty, 16, 1024, 1024),
            Ok(Damage::Partial(vec![DamageRect {
                x: 0,
                y: 0,
                width: 1024,
                height: 256
            }]))
        );
    }

    #[test]
    fn combined_hover_quad_and_badge_converge_without_an_immediate_fill_anchor() {
        for subpixel in [false, true] {
            for interleaved_graph in [false, true] {
                let mut previous =
                    snapshot(&joint_badge_scene(None, subpixel, interleaved_graph, true));
                for hover in [Some((0, 7)), Some((1, 9)), Some((0, 5)), None] {
                    let next =
                        snapshot(&joint_badge_scene(hover, subpixel, interleaved_graph, true));
                    if hover.is_some() && !interleaved_graph {
                        let sequence = next.painter_sequence().unwrap();
                        assert!(
                            sequence
                                .windows(2)
                                .any(|pair| pair[0].is_quad() && pair[1].is_glyph()),
                            "fixture must place the whole badge immediately after its fill"
                        );
                    }
                    let Damage::Partial(regions) = next
                        .compare_checked(&previous)
                        .expect("bounded combined edit")
                    else {
                        panic!("combined hover edits must actually be partial");
                    };
                    assert!(regions.len() <= 2);
                    assert!(
                        regions.iter().all(|region| region.height <= 17),
                        "far old/new fills and badges need separate damage: {regions:?}"
                    );
                    assert_eq!(next.compare_checked(&next), Ok(Damage::Reuse));
                    previous = next;
                }
            }
        }
        let first = snapshot(&joint_badge_scene(Some((0, 7)), false, false, false));
        let next = snapshot(&joint_badge_scene(Some((1, 9)), false, false, false));
        assert_eq!(
            next.compare(&first),
            Damage::Full,
            "translated quad and badge without real painter convergence stay conservative"
        );
    }

    #[test]
    fn normal_bounds_tree_row_hover_keeps_unproven_glyph_rank_permutations_conservative() {
        let row_scene = |hovered: usize| {
            let mut scene = Scene::default();
            let mask = ContentMask {
                bounds: bounds(0.0, 0.0, 200.0, 200.0),
                ..Default::default()
            };
            scene.insert_primitive(Quad {
                bounds: bounds(0.0, 0.0, 200.0, 200.0),
                content_mask: mask,
                background: gpui::rgba(0x223344ff).into(),
                ..Default::default()
            });
            for row in 0..2 {
                let y = 20.0 + row as f32 * 140.0;
                if hovered == row {
                    scene.insert_primitive(Quad {
                        bounds: bounds(0.0, y, 200.0, 14.0),
                        content_mask: mask,
                        background: gpui::rgba(0x55779944).into(),
                        ..Default::default()
                    });
                }
                for index in 0..3 {
                    scene.insert_primitive(glyph(
                        (row * 3 + index) as u32 + 1,
                        40.0 + index as f32 * 10.0,
                        y + 1.0,
                        false,
                    ));
                }
            }
            // Deliberately use ordinary bounds-tree insertion, not per-row
            // layers or artificial order floors. A hovered row's unchanged
            // text moves to a later rank relative to the other row.
            scene.finish();
            snapshot(&scene)
        };
        let old = row_scene(0);
        let next = row_scene(1);
        assert_eq!(
            next.compare_checked(&old),
            Err("quad-painter-convergence-change")
        );
    }

    #[test]
    fn hover_badge_glyph_runs_insert_move_erase_with_separate_far_row_damage() {
        for subpixel in [false, true] {
            let mut previous = snapshot(&badge_scene(None, subpixel));
            for hover in [Some((0, 7)), Some((1, 9)), Some((0, 5)), None] {
                let next = snapshot(&badge_scene(hover, subpixel));
                let damage = next
                    .compare_checked(&previous)
                    .expect("supported badge edit");
                let Damage::Partial(regions) = damage else {
                    panic!("badge transition must actually use partial damage: {damage:?}");
                };
                assert!(regions.len() <= 2);
                assert!(
                    regions.iter().all(|region| region.height <= 17),
                    "far rows must retain separate damage bands: {regions:?}"
                );
                assert_eq!(next.compare_checked(&next), Ok(Damage::Reuse));
                previous = next;
            }
        }
    }

    #[test]
    fn glyph_edits_preserve_order_atlas_budget_and_local_coverage_fallbacks() {
        let mut empty = Scene::default();
        empty.finish();
        let empty = snapshot(&empty);
        let mut excessive = Scene::default();
        for index in 0..=MAX_GLYPH_EDITS {
            excessive.insert_primitive(glyph(index as u32, 20.0, 20.0, false));
        }
        excessive.finish();
        assert_eq!(
            snapshot(&excessive).compare_checked(&empty),
            Err("glyph-edit-budget")
        );
        let first = |reverse: bool| {
            let mut scene = Scene::default();
            for index in if reverse { [2, 1] } else { [1, 2] } {
                scene.insert_primitive(glyph(index, 20.0, 20.0, false));
            }
            scene.finish();
            snapshot(&scene)
        };
        assert_eq!(
            first(true).compare_checked(&first(false)),
            Err("ambiguous-glyph-reorder")
        );
        let badge = badge_scene(Some((0, 7)), false);
        assert_eq!(
            SceneSnapshot::capture(&badge, 2, 200, 200)
                .unwrap()
                .compare_checked(&snapshot(&badge)),
            Err("atlas-generation-change")
        );
        let mut large = Scene::default();
        let gpui::Primitive::MonochromeSprite(mut sprite) = glyph(1, 0.0, 0.0, false) else {
            unreachable!()
        };
        sprite.bounds = bounds(0.0, 0.0, 200.0, 100.0);
        large.insert_primitive(sprite);
        large.finish();
        assert_eq!(
            snapshot(&large).compare_checked(&empty),
            Err("damage-area-budget")
        );
    }

    #[test]
    fn colour_only_damage_covers_the_antialias_fringe() {
        let first = snapshot(&scene(gpui::white()));
        assert_eq!(first.compare(&first), Damage::Reuse);
        let changed = snapshot(&scene(gpui::black()));
        assert_eq!(
            changed.compare(&first),
            Damage::Partial(vec![DamageRect {
                x: 0,
                y: 9,
                width: 200,
                height: 22
            }])
        );
    }

    #[test]
    fn geometry_masks_and_atlas_changes_invalidate_the_retained_scene() {
        let original = scene(gpui::white());
        let first = snapshot(&original);
        for field in 0..2 {
            let mut changed = scene(gpui::black());
            match field {
                0 => changed.quads[0].bounds.origin.x.0 += 1.0,
                _ => changed.quads[0].content_mask.bounds.size.width.0 -= 1.0,
            }
            assert_eq!(snapshot(&changed).compare(&first), Damage::Full);
        }
        assert_eq!(
            SceneSnapshot::capture(&original, 2, 200, 200)
                .unwrap()
                .compare(&first),
            Damage::Full
        );
        assert_eq!(
            SceneSnapshot::capture(&original, 1, 201, 200)
                .unwrap()
                .compare(&first),
            Damage::Full
        );
    }

    #[test]
    fn transparent_row_hover_quad_moves_preserve_relative_painter_order() {
        let rows = |hovered: usize| {
            let mut scene = Scene::default();
            for row in 0..2 {
                let y = 10.0 + row as f32 * 40.0;
                scene.push_layer(bounds(0.0, y, 200.0, 20.0));
                if row == hovered {
                    scene.insert_primitive(Quad {
                        bounds: bounds(0.0, y, 200.0, 20.0),
                        content_mask: ContentMask {
                            bounds: bounds(0.0, 0.0, 200.0, 200.0),
                            ..Default::default()
                        },
                        background: gpui::rgba(0x55779944).into(),
                        ..Default::default()
                    });
                }
                scene.insert_primitive(Quad {
                    bounds: bounds(5.0, y + 4.0, 5.0, 5.0),
                    content_mask: ContentMask {
                        bounds: bounds(0.0, 0.0, 200.0, 200.0),
                        ..Default::default()
                    },
                    background: gpui::white().into(),
                    ..Default::default()
                });
                scene.pop_layer();
            }
            scene.finish();
            scene
        };
        let first = rows(0);
        let next = rows(1);
        let mut next = snapshot(&next);
        // Absolute values are incidental once the flattened painter order is verified.
        for quad in &mut next.quads {
            quad.order += 42;
        }
        let Damage::Partial(regions) = next.compare(&snapshot(&first)) else {
            panic!("omitted transparent fills must qualify without adding any UI primitive");
        };
        assert_eq!(regions.len(), 2);
        assert!(
            regions
                .iter()
                .any(|region| region.y == 9 && region.height == 22)
        );
        assert!(
            regions
                .iter()
                .any(|region| region.y == 49 && region.height == 22)
        );
    }

    #[test]
    fn ambiguous_sparse_quad_alignment_falls_back_instead_of_assuming_reordering() {
        let mut original = scene(gpui::white());
        let mut second = original.quads[0];
        original.quads[0].bounds = bounds(10.0, 10.0, 5.0, 5.0);
        second.bounds = bounds(14.0, 10.0, 5.0, 5.0);
        original.quads.push(second);
        original.finish();
        let mut first = snapshot(&original);
        first.commands = vec![RenderCommand::Batch(PrimitiveBatch::Quads {
            range: 0..2,
            smoothed: false,
        })];
        let mut next = first.clone();
        next.quads.swap(0, 1);
        assert_eq!(next.compare(&first), Damage::Full);
    }

    #[test]
    fn adjacent_disjoint_swap_preserves_colour_damage_and_checks_antialias_overlap() {
        let mut original = scene(gpui::white());
        original.quads[0].bounds = bounds(10.0, 10.0, 5.0, 5.0);
        let mut second = original.quads[0];
        second.bounds = bounds(30.0, 10.0, 5.0, 5.0);
        original.quads.push(second);
        original.finish();
        let first = snapshot(&original);
        let mut next = first.clone();
        next.quads.swap(0, 1);
        assert_eq!(next.compare(&first), Damage::Reuse);
        next.quads[1].background = gpui::black().into();
        assert!(matches!(next.compare(&first), Damage::Partial(_)));

        // Geometric bounds are disjoint, but their one-pixel AA fringes overlap.
        let mut first = first;
        first.quads[1].bounds.origin.x.0 = 16.0;
        let mut next = first.clone();
        next.quads.swap(0, 1);
        assert_eq!(next.compare(&first), Damage::Full);
    }

    #[test]
    fn unsupported_coverage_does_not_prove_disjoint_reordering() {
        let mut original = Scene::default();
        let shadow = Shadow {
            order: 0,
            blur_radius: ScaledPixels(1.0),
            bounds: bounds(10.0, 90.0, 5.0, 5.0),
            corner_radii: Default::default(),
            content_mask: ContentMask {
                bounds: bounds(0.0, 0.0, 200.0, 200.0),
                ..Default::default()
            },
            color: gpui::black().into(),
            element_bounds: bounds(10.0, 90.0, 5.0, 5.0),
            element_corner_radii: Default::default(),
            inset: Default::default(),
            corner_smoothing: 0.0,
        };
        original.insert_primitive(shadow);
        original.insert_primitive(Shadow {
            bounds: bounds(90.0, 90.0, 5.0, 5.0),
            element_bounds: bounds(90.0, 90.0, 5.0, 5.0),
            ..shadow
        });
        original.finish();
        let mut first = snapshot(&original);
        first.commands = vec![RenderCommand::Batch(PrimitiveBatch::Shadows {
            range: 0..2,
            smoothed: false,
        })];
        let mut next = first.clone();
        next.shadows.swap(0, 1);
        assert_eq!(next.compare(&first), Damage::Full);
    }

    #[test]
    fn inset_shadow_replay_covers_the_element_outside_its_offset_hole() {
        let mut scene = Scene::default();
        scene.insert_primitive(Shadow {
            order: 0,
            blur_radius: ScaledPixels(0.5),
            bounds: bounds(140.0, 140.0, 20.0, 20.0),
            corner_radii: Default::default(),
            content_mask: ContentMask {
                bounds: bounds(0.0, 0.0, 200.0, 200.0),
                ..Default::default()
            },
            color: gpui::black().into(),
            element_bounds: bounds(10.0, 10.0, 180.0, 180.0),
            element_corner_radii: Default::default(),
            inset: true.into(),
            corner_smoothing: 0.0,
        });
        scene.finish();
        let batch = PrimitiveBatch::Shadows {
            range: 0..1,
            smoothed: false,
        };
        let outside_hole = DamageRect {
            x: 20,
            y: 20,
            width: 8,
            height: 8,
        };
        assert!(batch_intersects(&scene, &batch, outside_hole));
        scene.shadows[0].inset = false.into();
        assert!(!batch_intersects(&scene, &batch, outside_hole));
        scene.shadows[0].inset = true.into();
        scene.shadows[0].element_bounds.origin.x.0 = f32::INFINITY;
        assert!(matches!(
            SceneSnapshot::capture_checked(&scene, 1, 200, 200),
            Err(SceneSnapshotUnsupportedReason::NonfiniteGeometry)
        ));
    }

    #[test]
    fn headless_row_fixture_allows_disjoint_path_overlay_reordering() {
        let rows = |hovered: usize| {
            let mut scene = Scene::default();
            let mask = ContentMask {
                bounds: bounds(0.0, 0.0, 200.0, 100.0),
                ..Default::default()
            };
            scene.insert_primitive(Quad {
                bounds: bounds(4.0, 10.0 + hovered as f32 * 25.0, 192.0, 14.0),
                content_mask: mask,
                background: gpui::hsla(0.6, 0.4, 0.4, 0.7).into(),
                ..Default::default()
            });
            let mut path = gpui::Path::new(gpui::point(gpui::px(20.0), gpui::px(8.0)));
            path.line_to(gpui::point(gpui::px(38.0), gpui::px(8.0)));
            path.line_to(gpui::point(gpui::px(20.0), gpui::px(26.0)));
            path.line_to(gpui::point(gpui::px(20.0), gpui::px(8.0)));
            let mut path = path.scale(1.0);
            path.content_mask = ContentMask {
                bounds: bounds(20.0, 10.0, 18.0, 14.0),
                ..Default::default()
            };
            path.color = if hovered == 0 {
                gpui::hsla(0.6, 0.4, 0.4, 0.7)
            } else {
                gpui::black()
            }
            .into();
            scene.insert_primitive(path);
            scene.insert_primitive(Quad {
                bounds: bounds(80.0, 5.0, 15.0, 70.0),
                content_mask: ContentMask {
                    bounds: bounds(82.0, 0.0, 10.0, 100.0),
                    ..Default::default()
                },
                background: gpui::hsla(0.02, 0.8, 0.6, 0.4).into(),
                corner_radii: gpui::Corners::all(ScaledPixels(4.0)),
                ..Default::default()
            });
            scene.finish();
            SceneSnapshot::capture(&scene, 1, 200, 100).unwrap()
        };
        let mut previous = rows(0);
        assert_eq!(previous.compare(&previous), Damage::Reuse);
        for row in [1, 2, 0] {
            let next = rows(row);
            assert!(
                matches!(next.compare(&previous), Damage::Partial(_)),
                "row {row}: {:?}",
                next.compare(&previous)
            );
            previous = next;
        }
    }

    #[test]
    fn paired_quad_replacement_requires_an_unchanged_following_anchor() {
        let mut original = scene(gpui::white());
        let mut anchor = original.quads[0];
        anchor.bounds = bounds(5.0, 90.0, 5.0, 5.0);
        original.quads.push(anchor);
        original.finish();
        let mut first = snapshot(&original);
        first.commands = vec![RenderCommand::Batch(PrimitiveBatch::Quads {
            range: 0..2,
            smoothed: false,
        })];
        let mut next = first.clone();
        next.quads[0].bounds.origin.y.0 += 40.0;
        assert!(matches!(next.compare(&first), Damage::Partial(_)));
        next.quads[0].bounds.size.width.0 -= 1.0;
        next.quads[0].content_mask.bounds = bounds(2.0, 49.0, 190.0, 23.0);
        next.quads[0].content_mask.fade_out = gpui::Edges::all(ScaledPixels(0.5));
        next.quads[0].corner_radii = gpui::Corners::all(ScaledPixels(3.5));
        let Damage::Partial(regions) = next.compare_checked(&first).unwrap() else {
            panic!("anchored finite quad changes must damage old and new coverage");
        };
        assert!(
            regions
                .iter()
                .any(|region| region.intersects(first.quads[0].bounds))
        );
        assert!(regions.iter().any(|region| {
            region.intersects(
                next.quads[0]
                    .bounds
                    .intersect(&next.quads[0].content_mask.bounds),
            )
        }));

        first.commands = vec![RenderCommand::Batch(PrimitiveBatch::Quads {
            range: 0..1,
            smoothed: false,
        })];
        next.commands = first.commands.clone();
        assert_eq!(
            next.compare_checked(&first),
            Err("quad-extent-change"),
            "without convergence a changed extent still falls back with its precise mismatch"
        );
    }

    #[test]
    fn graph_path_recolour_is_partial_but_vertex_changes_fallback() {
        let path_scene = |color: Hsla| {
            let mut scene = Scene::default();
            let mut path = gpui::Path::new(gpui::point(gpui::px(5.0), gpui::px(10.0)));
            path.line_to(gpui::point(gpui::px(15.0), gpui::px(10.0)));
            path.line_to(gpui::point(gpui::px(15.0), gpui::px(20.0)));
            path.line_to(gpui::point(gpui::px(5.0), gpui::px(10.0)));
            let mut path = path.scale(1.0);
            path.content_mask = ContentMask {
                bounds: bounds(0.0, 0.0, 200.0, 200.0),
                ..Default::default()
            };
            path.color = color.into();
            scene.insert_primitive(path);
            scene.finish();
            scene
        };
        let first = snapshot(&path_scene(gpui::white()));
        let mut changed = path_scene(gpui::black());
        assert!(matches!(
            snapshot(&changed).compare(&first),
            Damage::Partial(_)
        ));
        changed.paths[0].vertices[0].xy_position.x.0 += 1.0;
        assert_eq!(snapshot(&changed).compare(&first), Damage::Full);
    }

    #[test]
    fn invisible_colour_changes_do_not_create_damage() {
        let mut first = scene(gpui::white());
        let mut next = scene(gpui::black());
        first.quads[0].content_mask.bounds = bounds(500.0, 500.0, 10.0, 10.0);
        next.quads[0].content_mask.bounds = first.quads[0].content_mask.bounds;
        assert_eq!(snapshot(&next).compare(&snapshot(&first)), Damage::Reuse);
    }

    #[test]
    fn large_damage_and_snapshot_memory_are_bounded() {
        let mut first = scene(gpui::white());
        let mut next = scene(gpui::black());
        first.quads[0].bounds = bounds(0.0, 0.0, 200.0, 200.0);
        next.quads[0].bounds = first.quads[0].bounds;
        assert_eq!(snapshot(&next).compare(&snapshot(&first)), Damage::Full);
        let quad = first.quads[0];
        first.quads = vec![quad; MAX_SNAPSHOT_BYTES / size_of::<Quad>() + 1];
        first.finish();
        assert!(SceneSnapshot::capture(&first, 1, 200, 200).is_none());
        assert_eq!(
            SceneSnapshot::capture_checked(&first, 1, 200, 200).unwrap_err(),
            SceneSnapshotUnsupportedReason::SnapshotBudget
        );
    }

    #[test]
    fn checked_snapshot_capture_reports_stable_conservative_reasons() {
        use SceneSnapshotUnsupportedReason as Unsupported;
        let check = |scene: &Scene, expected: Unsupported| {
            assert_eq!(
                SceneSnapshot::capture_checked(scene, 1, 200, 200).unwrap_err(),
                expected
            );
            assert!(SceneSnapshot::capture(scene, 1, 200, 200).is_none());
        };
        let mut empty = Scene::default();
        empty.finish();
        assert_eq!(
            SceneSnapshot::capture_checked(&empty, 1, 0, 200).unwrap_err(),
            Unsupported::InvalidTargetSize
        );
        let mut surface = Scene::default();
        surface.surfaces.push(gpui::PaintSurface {
            order: 0,
            bounds: bounds(0.0, 0.0, 10.0, 10.0),
            content_mask: ContentMask::default(),
            source: gpui::SurfaceSource::Unsupported(size(
                gpui::DevicePixels(10),
                gpui::DevicePixels(10),
            )),
        });
        check(&surface, Unsupported::Surface);
        let mut backdrop = Scene::default();
        backdrop.backdrop_filters.push(Default::default());
        check(&backdrop, Unsupported::BackdropFilter);
        let mut filtered = Scene::default();
        filtered.filter_boundaries.push(gpui::FilterBoundary {
            order: 0,
            bounds: bounds(0.0, 0.0, 10.0, 10.0),
            content_mask: ContentMask::default(),
            corner_radii: Default::default(),
            corner_smoothing: 0.0,
            filters: Default::default(),
            opacity: 1.0,
            is_start: true,
        });
        check(&filtered, Unsupported::FilterBoundary);
        let mut invalid = scene(gpui::white());
        invalid.quads[0].bounds.origin.x.0 = f32::INFINITY;
        check(&invalid, Unsupported::NonfiniteGeometry);
        let mut glyphs = Scene::default();
        glyphs.insert_primitive(MonochromeSprite {
            order: 0,
            padding: 0,
            bounds: bounds(0.0, 0.0, 8.0, 8.0),
            content_mask: ContentMask {
                bounds: bounds(0.0, 0.0, 200.0, 200.0),
                ..Default::default()
            },
            color: gpui::white().into(),
            tile: gpui::AtlasTile {
                texture_id: gpui::AtlasTextureId {
                    index: 0,
                    kind: gpui::AtlasTextureKind::Monochrome,
                },
                tile_id: gpui::TileId(0),
                padding: 0,
                bounds: Bounds::new(
                    point(gpui::DevicePixels(0), gpui::DevicePixels(0)),
                    size(gpui::DevicePixels(8), gpui::DevicePixels(8)),
                ),
            },
            transformation: gpui::TransformationMatrix::unit()
                .translate(point(ScaledPixels(1.0), ScaledPixels(0.0))),
        });
        glyphs.finish();
        // A transformed glyph is captured, covering its transformed bounds.
        assert!(SceneSnapshot::capture(&glyphs, 1, 200, 200).is_some());
        glyphs.monochrome_sprites[0].bounds.size.width.0 = f32::NAN;
        check(&glyphs, Unsupported::NonfiniteGeometry);
        for (reason, name) in [
            (Unsupported::InvalidTargetSize, "invalid-target-size"),
            (Unsupported::Surface, "surface"),
            (Unsupported::BackdropFilter, "backdrop-filter"),
            (Unsupported::FilterBoundary, "filter-boundary"),
            (Unsupported::NonfiniteGeometry, "nonfinite-geometry"),
            (Unsupported::SnapshotBudget, "snapshot-budget"),
        ] {
            assert_eq!(reason.as_str(), name);
        }
    }

    #[test]
    fn invalid_geometry_and_filters_fail_closed_and_clear_uses_the_target_colour() {
        let mut scene = scene(gpui::white());
        scene.quads[0].bounds.origin.x.0 = f32::NAN;
        assert!(SceneSnapshot::capture(&scene, 1, 200, 200).is_none());
        scene.quads.clear();
        scene.backdrop_filters.push(Default::default());
        assert!(SceneSnapshot::capture(&scene, 1, 200, 200).is_none());
        let region = DamageRect {
            x: 1,
            y: 2,
            width: 3,
            height: 4,
        };
        assert_eq!(
            clear_quad(region, [0.0, 0.0, 0.0, 1.0]).background,
            gpui::black().into()
        );
    }
}
