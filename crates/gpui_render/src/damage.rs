//! Conservative retained-scene comparison for the opt-in partial-redraw experiment.
//! Exact typed comparisons avoid formatting, hashing padding, or trusting a hash collision.
//! Local colour/quad changes and bounded disjoint painter swaps can reuse the target.
use gpui::{
    Bounds, ContentMask, Hsla, MonochromeSprite, Path, PathVertex, PolychromeSprite,
    PrimitiveBatch, Quad, RenderCommand, ScaledPixels, Scene, SceneHsla, Shadow, SubpixelSprite,
    Underline, point, size,
};
use std::mem::size_of;

const MAX_SNAPSHOT_BYTES: usize = 2 * 1024 * 1024;
const MAX_REGIONS: usize = 8;
const MAX_QUAD_EDITS: usize = 8;
const MAX_COMMUTATIONS: usize = 8;

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

    fn union(self, other: Self) -> Self {
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
    /// Stable diagnostic name for the selected rendering path.
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

    fn translated_quad_matches(self, other: Self) -> bool {
        let (Self::Quad(next, n_smoothing), Self::Quad(old, o_smoothing)) = (self, other) else {
            return false;
        };
        let mut next = *next;
        let mut old = *old;
        next.order = 0;
        old.order = 0;
        next.bounds.origin = Default::default();
        old.bounds.origin = Default::default();
        next.background = Default::default();
        old.background = Default::default();
        next.border_color = Default::default();
        old.border_color = Default::default();
        n_smoothing == o_smoothing && next == old
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
            Self::Monochrome(p, _) => p.bounds.intersect(&p.content_mask.bounds),
            Self::Subpixel(p, _) => p.bounds.intersect(&p.content_mask.bounds),
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
        // Surfaces can update without primitive changes. Filters sample outside local bounds.
        if width == 0
            || height == 0
            || !scene.surfaces.is_empty()
            || !scene.backdrop_filters.is_empty()
            || !scene.filter_boundaries.is_empty()
        {
            return None;
        }
        snapshot_bytes(scene)?;
        let visible = |bounds, mask| finite_bounds(bounds) && finite_bounds(mask);
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
            || scene.monochrome_sprites.iter().any(|p| {
                p.transformation != Default::default() || !visible(p.bounds, p.content_mask.bounds)
            })
            || scene.subpixel_sprites.iter().any(|p| {
                p.transformation != Default::default() || !visible(p.bounds, p.content_mask.bounds)
            })
            || scene.shadows.iter().any(|p| {
                !visible(p.bounds, p.content_mask.bounds)
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
            return None;
        }
        Some(Self {
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
        if self.atlas_generation != previous.atlas_generation
            || self.width != previous.width
            || self.height != previous.height
            || self.paths.len() != previous.paths.len()
            || self.monochrome_sprites.len() != previous.monochrome_sprites.len()
            || self.subpixel_sprites.len() != previous.subpixel_sprites.len()
            || self.shadows.len() != previous.shadows.len()
            || self.underlines.len() != previous.underlines.len()
            || self.polychrome_sprites.len() != previous.polychrome_sprites.len()
        {
            return Damage::Full;
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
            return Damage::Full;
        };
        let mut n = 0;
        let mut o = 0;
        let mut edits = 0;
        let mut commutations = 0;
        while n < next.len() || o < old.len() {
            match (next.get(n).copied(), old.get(o).copied()) {
                (Some(current), Some(prior)) if current.geometry_matches(prior) => {
                    if current.colour_changed(prior) && !damage(current.bounds()) {
                        return Damage::Full;
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
                            return Damage::Full;
                        }
                        n += 2;
                        o += 2;
                        commutations += 1;
                        if commutations > MAX_COMMUTATIONS {
                            return Damage::Full;
                        }
                        continue;
                    }
                    // Permit one sparse inserted/removed quad only when the immediately
                    // following exact geometry re-establishes the unchanged painter order.
                    let inserted = current.is_quad()
                        && next.get(n + 1).is_some_and(|p| p.geometry_matches(prior));
                    let removed = prior.is_quad()
                        && old.get(o + 1).is_some_and(|p| current.geometry_matches(*p));
                    if inserted && removed {
                        return Damage::Full;
                    }
                    if !inserted && !removed {
                        // One omitted hover fill may move to another row without any
                        // intervening primitive. Require identical quad shape/pipeline and
                        // an exact following anchor, possibly two disjoint paints whose
                        // adjacent swap is verified on the next loop iteration.
                        let paired = current.translated_quad_matches(prior)
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
                        if !paired || !damage(current.bounds()) || !damage(prior.bounds()) {
                            return Damage::Full;
                        }
                        n += 1;
                        o += 1;
                        edits += 2;
                    } else if inserted {
                        if !damage(current.bounds()) {
                            return Damage::Full;
                        }
                        n += 1;
                        edits += 1;
                    } else {
                        if !damage(prior.bounds()) {
                            return Damage::Full;
                        }
                        o += 1;
                        edits += 1;
                    }
                }
                (Some(current), None) if current.is_quad() => {
                    if !damage(current.bounds()) {
                        return Damage::Full;
                    }
                    n += 1;
                    edits += 1;
                }
                (None, Some(prior)) if prior.is_quad() => {
                    if !damage(prior.bounds()) {
                        return Damage::Full;
                    }
                    o += 1;
                    edits += 1;
                }
                _ => return Damage::Full,
            }
            if edits > MAX_QUAD_EDITS {
                return Damage::Full;
            }
        }
        if regions.is_empty() {
            return Damage::Reuse;
        }
        let damaged: u128 = regions
            .iter()
            .map(|region| region.width as u128 * region.height as u128)
            .sum();
        if damaged * 100 > self.width as u128 * self.height as u128 * 40 {
            Damage::Full
        } else {
            Damage::Partial(regions)
        }
    }
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
        PrimitiveBatch::Shadows { range, .. } => scene.shadows[range.clone()].iter().any(|p| {
            visible(
                p.bounds.dilate(ScaledPixels(p.blur_radius.0 * 4.0)),
                p.content_mask.bounds,
            )
        }),
        PrimitiveBatch::Paths { range, .. } => scene.paths[range.clone()]
            .iter()
            .any(|p| region.intersects(p.clipped_bounds().dilate(ScaledPixels(1.0)))),
        PrimitiveBatch::Underlines(range) => scene.underlines[range.clone()].iter().any(|p| {
            visible(
                p.bounds.dilate(ScaledPixels(p.thickness.0 * 4.0)),
                p.content_mask.bounds,
            )
        }),
        PrimitiveBatch::MonochromeSprites { range, .. } => scene.monochrome_sprites[range.clone()]
            .iter()
            .any(|p| visible(p.bounds, p.content_mask.bounds)),
        PrimitiveBatch::SubpixelSprites { range, .. } => scene.subpixel_sprites[range.clone()]
            .iter()
            .any(|p| visible(p.bounds, p.content_mask.bounds)),
        PrimitiveBatch::PolychromeSprites { range, .. } => scene.polychrome_sprites[range.clone()]
            .iter()
            .any(|p| visible(p.bounds, p.content_mask.bounds)),
        _ => true,
    }
}

/// Make the opaque clear fill used before replaying the intersecting current painter batches.
pub fn clear_quad(region: DamageRect, clear_color: Hsla) -> Quad {
    Quad {
        bounds: region.bounds(),
        content_mask: ContentMask {
            bounds: region.bounds(),
            ..Default::default()
        },
        background: clear_color.into(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn paired_hover_fill_translation_requires_an_unchanged_following_anchor() {
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
        assert_eq!(next.compare(&first), Damage::Full);
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
            clear_quad(region, gpui::black()).background,
            gpui::black().into()
        );
    }
}
