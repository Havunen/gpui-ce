//! Damage for partial redraw: which part of the retained frame the next frame
//! changes.
//!
//! Each frame hashes, per 64 px tile, everything painted into that tile in
//! painter order. The bounding box of the tiles whose hash changed repaints,
//! or the whole frame when it covers over half. A primitive's global draw
//! order is left out of its hash: GPUI
//! raises it for everything painted over a new primitive (a hovered row's
//! content over its highlight), which moves that content in the sorted scene
//! without changing any other tile.
use crate::blur::GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS;
use gpui::{
    Bounds, ContentMask, MonochromeSprite, PolychromeSprite, PrimitiveBatch, Quad, RenderCommand,
    ScaledPixels, Scene, Shadow, SubpixelSprite, TransformationMatrix, Underline, point, size,
};

const DAMAGE_TILE_SIZE: u32 = 64;

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
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// How much of a retained frame the next frame repaints.
pub enum Damage {
    /// The whole scene must be rendered normally.
    Full,
    /// The retained scene already matches the new scene.
    Reuse,
    /// Clear and redraw this region with all intersecting painter batches.
    Partial(DamageRect),
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

/// What a finished frame painted into each tile of its target.
#[derive(Clone, Debug)]
pub struct SceneSnapshot {
    tiles: Vec<u64>,
    atlas_generation: u64,
    width: u32,
    height: u32,
}

impl SceneSnapshot {
    /// Hash a finished scene into the tiles of a `width` by `height` target.
    /// `None` for what partial redraw can't follow: an empty target, surfaces
    /// (which change without the scene), filters (which sample outside their
    /// bounds) and non-finite geometry.
    pub fn capture(scene: &Scene, atlas_generation: u64, width: u32, height: u32) -> Option<Self> {
        if width == 0
            || height == 0
            || !scene.surfaces.is_empty()
            || !scene.backdrop_filters.is_empty()
            || !scene.filter_boundaries.is_empty()
        {
            return None;
        }
        let columns = width.div_ceil(DAMAGE_TILE_SIZE);
        let rows = height.div_ceil(DAMAGE_TILE_SIZE);
        let mut tiles = vec![SEED; columns as usize * rows as usize];
        // Mixes `hash` into every tile the clipped, antialiased geometry touches.
        let mut paint =
            |geometry: Bounds<ScaledPixels>, mask: &ContentMask<ScaledPixels>, hash: u64| {
                if !finite_bounds(geometry) || !finite_bounds(mask.bounds) {
                    return None;
                }
                let region = rectangle(geometry.intersect(&mask.bounds), width, height);
                if region.width != 0 && region.height != 0 {
                    for tile in tile_spans(region, columns) {
                        tiles[tile] = mix(tiles[tile], hash);
                    }
                }
                Some(())
            };
        for command in scene.render_commands() {
            let RenderCommand::Batch(batch) = command else {
                return None;
            };
            match batch {
                PrimitiveBatch::Quads { range, smoothed } => {
                    for quad in &scene.quads[range.clone()] {
                        let key = Quad { order: 0, ..*quad };
                        let hash = hash_of(1 + u64::from(*smoothed), &[key]);
                        paint(quad.bounds, &quad.content_mask, hash)?;
                    }
                }
                PrimitiveBatch::Shadows { range, smoothed } => {
                    for shadow in &scene.shadows[range.clone()] {
                        // NaN or negative blur has no shader extent to cover.
                        if shadow.blur_radius.0.is_nan() || shadow.blur_radius.0 < 0.0 {
                            return None;
                        }
                        let key = Shadow {
                            order: 0,
                            ..*shadow
                        };
                        let hash = hash_of(3 + u64::from(*smoothed), &[key]);
                        paint(shadow_geometry(shadow), &shadow.content_mask, hash)?;
                    }
                }
                PrimitiveBatch::Paths {
                    range,
                    rasterization_vertex_count,
                    sprite_count,
                } => {
                    // A batch's paths rasterize together, so each hashes its group too.
                    let group = [range.len(), *rasterization_vertex_count, *sprite_count]
                        .into_iter()
                        .fold(SEED ^ 5, |hash, value| mix(hash, value as u64));
                    for path in &scene.paths[range.clone()] {
                        let hash = [
                            hash_of(6, &path.vertices),
                            hash_of(7, &[path.bounds]),
                            hash_of(8, &[path.content_mask]),
                            hash_of(9, &[path.color]),
                        ]
                        .into_iter()
                        .fold(group, mix);
                        paint(path.bounds, &path.content_mask, hash)?;
                    }
                }
                PrimitiveBatch::Underlines(range) => {
                    for underline in &scene.underlines[range.clone()] {
                        if underline.thickness.0.is_nan() || underline.thickness.0 < 0.0 {
                            return None;
                        }
                        let key = Underline {
                            order: 0,
                            ..*underline
                        };
                        // Wavy underlines swing past their bounds.
                        let geometry = underline
                            .bounds
                            .dilate(ScaledPixels(underline.thickness.0 * 4.0));
                        paint(geometry, &underline.content_mask, hash_of(10, &[key]))?;
                    }
                }
                PrimitiveBatch::MonochromeSprites { range, .. } => {
                    for sprite in &scene.monochrome_sprites[range.clone()] {
                        let key = MonochromeSprite {
                            order: 0,
                            ..*sprite
                        };
                        let geometry = sprite_geometry(sprite.bounds, &sprite.transformation)?;
                        paint(geometry, &sprite.content_mask, hash_of(11, &[key]))?;
                    }
                }
                PrimitiveBatch::SubpixelSprites { range, .. } => {
                    for sprite in &scene.subpixel_sprites[range.clone()] {
                        let key = SubpixelSprite {
                            order: 0,
                            ..*sprite
                        };
                        let geometry = sprite_geometry(sprite.bounds, &sprite.transformation)?;
                        paint(geometry, &sprite.content_mask, hash_of(12, &[key]))?;
                    }
                }
                PrimitiveBatch::PolychromeSprites {
                    range, smoothed, ..
                } => {
                    for sprite in &scene.polychrome_sprites[range.clone()] {
                        let key = PolychromeSprite {
                            order: 0,
                            ..*sprite
                        };
                        let hash = hash_of(13 + u64::from(*smoothed), &[key]);
                        paint(sprite.bounds, &sprite.content_mask, hash)?;
                    }
                }
                _ => return None,
            }
        }
        Some(Self {
            tiles,
            atlas_generation,
            width,
            height,
        })
    }

    /// The tiles to repaint over `previous` so the target shows this frame.
    pub fn compare(&self, previous: &Self) -> Damage {
        if (self.atlas_generation, self.width, self.height)
            != (previous.atlas_generation, previous.width, previous.height)
        {
            return Damage::Full;
        }
        let dirty: Vec<bool> = self
            .tiles
            .iter()
            .zip(&previous.tiles)
            .map(|(next, old)| next != old)
            .collect();
        let columns = self.width.div_ceil(DAMAGE_TILE_SIZE);
        dirty_bounds(&dirty, columns, self.width, self.height)
    }
}

const SEED: u64 = 0x9e37_79b9_7f4a_7c15;

fn mix(hash: u64, value: u64) -> u64 {
    (hash.rotate_left(5) ^ value).wrapping_mul(0x5170_9ab3_b8d4_d9e3)
}

/// Hashes the bytes of scene values in four independent lanes.
///
/// Only for the `#[repr(C)]` scene types, whose fields are four-byte scalars
/// with explicit padding fields, so every byte is initialized.
fn hash_of<T>(kind: u64, values: &[T]) -> u64 {
    // SAFETY: see above; the slice covers exactly the values' memory.
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    let word = |chunk: &[u8]| {
        let mut word = [0; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        u64::from_le_bytes(word)
    };
    let mut lanes = [
        SEED ^ kind,
        SEED,
        SEED.rotate_left(21),
        SEED.rotate_left(42),
    ];
    let mut blocks = bytes.chunks_exact(32);
    for block in &mut blocks {
        for (lane, chunk) in lanes.iter_mut().zip(block.chunks_exact(8)) {
            *lane = mix(*lane, word(chunk));
        }
    }
    let mut hash =
        lanes[0] ^ lanes[1].rotate_left(16) ^ lanes[2].rotate_left(32) ^ lanes[3].rotate_left(48);
    for chunk in blocks.remainder().chunks(8) {
        hash = mix(hash, word(chunk));
    }
    hash ^= bytes.len() as u64;
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^ (hash >> 33)
}

/// The bounds a transformed sprite's shader covers. `None` for a non-finite
/// transform: f32::min and max skip NaN, so `transformed` can't tell.
fn sprite_geometry(
    bounds: Bounds<ScaledPixels>,
    matrix: &TransformationMatrix,
) -> Option<Bounds<ScaledPixels>> {
    let finite = matrix
        .rotation_scale
        .iter()
        .flatten()
        .chain(&matrix.translation)
        .all(|value| value.is_finite());
    (finite && finite_bounds(bounds)).then(|| transformed(bounds, matrix))
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

/// The tile-aligned box around every dirty tile, clipped to the target, or a
/// full redraw when it covers over half of it.
fn dirty_bounds(dirty: &[bool], columns: u32, width: u32, height: u32) -> Damage {
    let mut tiles = (0..dirty.len() as u32)
        .filter(|&tile| dirty[tile as usize])
        .map(|tile| (tile % columns, tile / columns));
    let Some((x, y)) = tiles.next() else {
        return Damage::Reuse;
    };
    let (mut left, mut right, mut bottom) = (x, x, y);
    for (x, y) in tiles {
        left = left.min(x);
        right = right.max(x);
        bottom = y;
    }
    let (x, y) = (left * DAMAGE_TILE_SIZE, y * DAMAGE_TILE_SIZE);
    let region = DamageRect {
        x,
        y,
        width: ((right + 1) * DAMAGE_TILE_SIZE).min(width) - x,
        height: ((bottom + 1) * DAMAGE_TILE_SIZE).min(height) - y,
    };
    if 2 * u64::from(region.width) * u64::from(region.height) > u64::from(width) * u64::from(height)
    {
        Damage::Full
    } else {
        Damage::Partial(region)
    }
}

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
    use gpui::{Edges, Hsla, Path, Primitive};

    fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(width), ScaledPixels(height)),
        }
    }

    fn full_mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: bounds(0.0, 0.0, 512.0, 1024.0),
            ..Default::default()
        }
    }

    fn quad(bounds: Bounds<ScaledPixels>, color: impl Into<gpui::Background>) -> Quad {
        Quad {
            bounds,
            content_mask: full_mask(),
            background: color.into(),
            ..Default::default()
        }
    }

    fn finished(primitives: impl IntoIterator<Item = Primitive>) -> Scene {
        let mut scene = Scene::default();
        for primitive in primitives {
            scene.insert_primitive(primitive);
        }
        scene.finish();
        scene
    }

    /// Hashes `primitives` for a 512 by 1024 target.
    fn capture(primitives: impl IntoIterator<Item = Primitive>) -> SceneSnapshot {
        SceneSnapshot::capture(&finished(primitives), 1, 512, 1024).unwrap()
    }

    fn region(damage: Damage) -> DamageRect {
        match damage {
            Damage::Partial(region) => region,
            other => panic!("expected partial damage, got {other:?}"),
        }
    }

    fn rect(x: u32, y: u32, width: u32, height: u32) -> DamageRect {
        DamageRect {
            x,
            y,
            width,
            height,
        }
    }

    fn glyph(index: u32, x: f32, y: f32, subpixel: bool) -> Primitive {
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
        if subpixel {
            SubpixelSprite {
                order: 0,
                padding: 0,
                bounds: bounds(x, y, 8.0, 10.0),
                content_mask: full_mask(),
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
                content_mask: full_mask(),
                color: gpui::white().into(),
                tile,
                transformation: Default::default(),
            }
            .into()
        }
    }

    #[test]
    fn changes_repaint_only_the_tiles_they_touch() {
        let background = quad(bounds(0.0, 0.0, 512.0, 1024.0), gpui::black());
        let with = |extra: Quad| capture([background.into(), extra.into()]);
        let base = capture([background.into()]);
        let small = quad(bounds(70.0, 10.0, 20.0, 20.0), gpui::red());
        assert_eq!(with(small).compare(&with(small)), Damage::Reuse);
        let recoloured = Quad {
            background: gpui::blue().into(),
            ..small
        };
        assert_eq!(
            region(with(recoloured).compare(&with(small))),
            rect(64, 0, 64, 64)
        );
        // A move repaints where the quad was and where it is.
        let moved = Quad {
            bounds: bounds(200.0, 280.0, 20.0, 20.0),
            ..small
        };
        assert_eq!(
            region(with(moved).compare(&with(small))),
            rect(64, 0, 192, 320)
        );
        // The antialias fringe reaches across a tile edge.
        let edge = Quad {
            bounds: bounds(100.0, 64.5, 10.0, 10.0),
            ..small
        };
        assert_eq!(region(with(edge).compare(&base)), rect(64, 0, 64, 128));
        // Content masks clip what a primitive can touch.
        let clipped = Quad {
            bounds: bounds(0.0, 0.0, 300.0, 300.0),
            content_mask: ContentMask {
                bounds: bounds(10.0, 10.0, 20.0, 20.0),
                ..Default::default()
            },
            ..small
        };
        assert_eq!(region(with(clipped).compare(&base)), rect(0, 0, 64, 64));
    }

    #[test]
    fn global_draw_order_is_ignored_but_order_within_a_tile_counts() {
        let background = quad(bounds(0.0, 0.0, 512.0, 1024.0), gpui::black());
        let a = quad(bounds(10.0, 10.0, 20.0, 20.0), gpui::red());
        let b = quad(bounds(20.0, 20.0, 20.0, 20.0), gpui::blue());
        let far = quad(bounds(400.0, 400.0, 20.0, 20.0), gpui::green());
        let frame = |quads: [Quad; 4]| capture(quads.map(Primitive::from));
        let first = frame([background, far, a, b]);
        // Painting an unrelated quad later only moves it in the sorted scene.
        assert_eq!(
            frame([background, a, b, far]).compare(&first),
            Damage::Reuse
        );
        // Swapping overlapping quads changes which one is on top.
        assert_eq!(
            region(frame([background, b, a, far]).compare(&first)),
            rect(0, 0, 64, 64)
        );
        // A quad slid under one end of a long bar raises the bar's draw order,
        // but only the tile it was slid into changes.
        let bar = quad(bounds(10.0, 200.0, 300.0, 10.0), gpui::white());
        let under = quad(bounds(20.0, 195.0, 20.0, 20.0), gpui::red());
        let alone = capture([background.into(), bar.into()]);
        assert_eq!(
            region(capture([background.into(), under.into(), bar.into()]).compare(&alone)),
            rect(0, 192, 64, 64)
        );
    }

    /// History-like rows. A hovered row's highlight sits under its quads and
    /// glyphs, which raises their draw order, and adds a badge.
    fn rows(hover: Option<usize>, subpixel: bool) -> SceneSnapshot {
        let mut primitives: Vec<Primitive> =
            vec![quad(bounds(0.0, 0.0, 512.0, 1024.0), gpui::rgba(0x22334488)).into()];
        for row in 0..20 {
            let y = 16.25 + 40.0 * row as f32;
            let hovered = hover == Some(row);
            if hovered {
                primitives.push(
                    Quad {
                        content_mask: ContentMask {
                            bounds: bounds(0.5, y + 0.25, 511.0, 18.0),
                            fade_out: Edges::all(ScaledPixels(0.75)),
                        },
                        corner_radii: gpui::Corners::all(ScaledPixels(2.25)),
                        ..quad(bounds(0.0, y, 512.0, 18.5), gpui::rgba(0x55779944))
                    }
                    .into(),
                );
                primitives.extend((0..4).map(|index| {
                    glyph(
                        30 + index * 7,
                        440.0 + index as f32 * 9.0,
                        y + 2.0,
                        subpixel,
                    )
                }));
            }
            for (x, width) in [(134.0, 66.0), (200.0, 4.0), (410.0, 24.0)] {
                let color = if hovered { 0x55779999 } else { 0x22334499 };
                primitives.push(quad(bounds(x, y, width, 18.5), gpui::rgba(color)).into());
            }
            primitives.extend(
                (0..20)
                    .map(|index| glyph(index + 1, 12.0 + index as f32 * 19.0, y + 2.0, subpixel)),
            );
        }
        capture(primitives)
    }

    #[test]
    fn hovering_the_next_row_repaints_only_both_rows() {
        for subpixel in [false, true] {
            let first = rows(Some(3), subpixel);
            // The hover moved both rows' content in the sorted scene, but only
            // their own tiles changed.
            assert_eq!(
                region(rows(Some(4), subpixel).compare(&first)),
                rect(0, 128, 512, 128)
            );
            assert_eq!(
                region(rows(None, subpixel).compare(&first)),
                rect(0, 128, 512, 64)
            );
            // One box spans both rows, so a jump across most of the list repaints in full.
            assert_eq!(rows(Some(19), subpixel).compare(&first), Damage::Full);
        }
    }

    #[test]
    fn fractional_masks_cover_both_sides_of_tile_edges_and_clip_to_the_target() {
        let fading = |top: f32| {
            Primitive::from(Quad {
                bounds: bounds(62.25, 62.25, 20.5, 20.5),
                content_mask: ContentMask {
                    bounds: bounds(63.75, 63.75, 12.5, 12.5),
                    fade_out: Edges {
                        top: ScaledPixels(top),
                        ..Edges::all(ScaledPixels(1.25))
                    },
                },
                background: gpui::rgba(0xff112280).into(),
                corner_radii: gpui::Corners::all(ScaledPixels(2.25)),
                ..Default::default()
            })
        };
        assert_eq!(
            region(capture([fading(1.5)]).compare(&capture([fading(1.25)]))),
            rect(0, 0, 128, 128)
        );
        let edge = |color: Hsla| {
            let quad = Quad {
                content_mask: ContentMask {
                    bounds: bounds(0.0, 0.0, 501.0, 999.0),
                    ..Default::default()
                },
                ..quad(bounds(495.0, 992.0, 20.0, 20.0), color)
            };
            SceneSnapshot::capture(&finished([quad.into()]), 1, 501, 999).unwrap()
        };
        assert_eq!(
            region(edge(gpui::black()).compare(&edge(gpui::white()))),
            rect(448, 960, 53, 39)
        );
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

    fn path(x: f32, y: f32, color: Hsla) -> Path<ScaledPixels> {
        let mut path = Path::new(point(gpui::px(x), gpui::px(y)));
        path.line_to(point(gpui::px(x + 20.0), gpui::px(y)));
        path.line_to(point(gpui::px(x), gpui::px(y + 20.0)));
        let mut path = path.scale(1.0);
        path.content_mask = full_mask();
        path.color = color.into();
        path
    }

    #[test]
    fn blurred_shadows_damage_the_shader_blur_extent() {
        let scene = finished([blurred_shadow(100.0).into()]);
        let batch = scene
            .render_commands()
            .iter()
            .find_map(|command| match command {
                RenderCommand::Batch(batch @ PrimitiveBatch::Shadows { .. }) => Some(batch.clone()),
                _ => None,
            })
            .unwrap();
        let column = |x| rect(x, 100, 1, 1);
        // The shader draws 3 standard deviations, 12px here, plus the 1px fringe.
        assert!(batch_intersects(&scene, &batch, column(87)));
        assert!(!batch_intersects(&scene, &batch, column(85)));
        // The blur reaches the tile left of the shadow's own bounds, and the
        // tile row below.
        let moved = capture([blurred_shadow(80.0).into()]);
        assert_eq!(
            region(moved.compare(&capture([blurred_shadow(72.0).into()]))),
            rect(0, 64, 128, 128)
        );
    }

    #[test]
    fn moved_shadows_and_paths_repaint_only_their_tiles() {
        let frame = |x: f32| {
            capture([
                quad(bounds(0.0, 0.0, 512.0, 1024.0), gpui::blue()).into(),
                blurred_shadow(x).into(),
                path(x, 600.0, gpui::white()).into(),
            ])
        };
        let r = region(frame(300.0).compare(&frame(100.0)));
        let covers =
            |x: u32, y: u32| r.x <= x && x < r.x + r.width && r.y <= y && y < r.y + r.height;
        // Old and new blur extents, and old and new paths.
        assert!(covers(90, 110) && covers(130, 110) && covers(290, 110) && covers(330, 110));
        assert!(covers(105, 605) && covers(305, 605));
        assert!(!covers(500, 1000));
    }

    #[test]
    fn rotated_sprites_damage_their_transformed_bounds() {
        let sprite = |angle: f32| {
            let mut sprite = glyph(1, 66.0, 66.0, false);
            if let Primitive::MonochromeSprite(sprite) = &mut sprite {
                sprite.bounds = bounds(66.0, 66.0, 32.0, 32.0);
                let center =
                    |sign: f32| point(ScaledPixels(82.0 * sign), ScaledPixels(82.0 * sign));
                sprite.transformation = TransformationMatrix::unit()
                    .translate(center(1.0))
                    .rotate(gpui::radians(angle))
                    .translate(center(-1.0));
            }
            capture([sprite])
        };
        // At 45 degrees the corners reach past x and y 64 into three more tiles.
        assert_eq!(
            region(sprite(std::f32::consts::FRAC_PI_4).compare(&sprite(0.0))),
            rect(0, 0, 128, 128)
        );
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
    fn paths_hash_their_paint_vertices_and_rasterization_group() {
        let alone = capture([path(8.25, 8.25, gpui::white()).into()]);
        assert_eq!(
            region(capture([path(8.25, 8.25, gpui::black()).into()]).compare(&alone)),
            rect(0, 0, 64, 64)
        );
        assert_eq!(
            region(capture([path(8.5, 8.25, gpui::white()).into()]).compare(&alone)),
            rect(0, 0, 64, 64)
        );
        // A path joining the batch changes how the first one rasterizes too.
        let joined = finished([
            path(8.25, 8.25, gpui::white()).into(),
            path(400.0, 8.25, gpui::white()).into(),
        ]);
        assert!(matches!(
            joined.render_commands(),
            [RenderCommand::Batch(PrimitiveBatch::Paths { range, .. })] if range.len() == 2
        ));
        let joined = SceneSnapshot::capture(&joined, 1, 512, 1024).unwrap();
        assert_eq!(region(joined.compare(&alone)), rect(0, 0, 448, 64));
    }

    #[test]
    fn targets_atlases_and_unsupported_scenes_repaint_everything() {
        let plain = finished([quad(bounds(0.0, 10.0, 200.0, 20.0), gpui::white()).into()]);
        let first = SceneSnapshot::capture(&plain, 1, 512, 1024).unwrap();
        // Sprites refer to atlas tiles, which a new generation may have reused.
        let reallocated = SceneSnapshot::capture(&plain, 2, 512, 1024).unwrap();
        assert_eq!(reallocated.compare(&first), Damage::Full);
        let resized = SceneSnapshot::capture(&plain, 1, 511, 1024).unwrap();
        assert_eq!(resized.compare(&first), Damage::Full);
        assert!(SceneSnapshot::capture(&plain, 1, 0, 1024).is_none());
        let rejected = |change: &dyn Fn(&mut Scene)| {
            let mut scene = finished([
                quad(bounds(0.0, 10.0, 200.0, 20.0), gpui::white()).into(),
                blurred_shadow(100.0).into(),
                glyph(1, 40.0, 40.0, false),
            ]);
            assert!(SceneSnapshot::capture(&scene, 1, 512, 1024).is_some());
            change(&mut scene);
            assert!(SceneSnapshot::capture(&scene, 1, 512, 1024).is_none());
        };
        rejected(&|scene| {
            scene.surfaces.push(gpui::PaintSurface {
                order: 0,
                bounds: bounds(0.0, 0.0, 10.0, 10.0),
                content_mask: ContentMask::default(),
                source: gpui::SurfaceSource::Unsupported(size(
                    gpui::DevicePixels(10),
                    gpui::DevicePixels(10),
                )),
            })
        });
        rejected(&|scene| scene.backdrop_filters.push(Default::default()));
        rejected(&|scene| {
            scene.filter_boundaries.push(gpui::FilterBoundary {
                order: 0,
                bounds: bounds(0.0, 0.0, 10.0, 10.0),
                content_mask: ContentMask::default(),
                corner_radii: Default::default(),
                corner_smoothing: 0.0,
                filters: Default::default(),
                opacity: 1.0,
                is_start: true,
            })
        });
        rejected(&|scene| scene.quads[0].bounds.origin.x.0 = f32::INFINITY);
        rejected(&|scene| scene.quads[0].content_mask.bounds.size.width.0 = f32::NAN);
        rejected(&|scene| scene.shadows[0].blur_radius.0 = -1.0);
        rejected(&|scene| {
            scene.shadows[0].inset = true.into();
            scene.shadows[0].element_bounds.origin.x.0 = f32::INFINITY;
        });
        rejected(&|scene| scene.monochrome_sprites[0].transformation.translation[0] = f32::NAN);
        rejected(&|scene| scene.monochrome_sprites[0].bounds.size.width.0 = f32::NAN);
    }

    #[test]
    fn dirty_tiles_repaint_as_one_box_clipped_to_the_target_up_to_half_of_it() {
        let tile = |x: usize, y: usize| y * 16 + x;
        let mut dirty = vec![false; 16 * 16];
        assert_eq!(dirty_bounds(&dirty, 16, 1000, 1000), Damage::Reuse);
        dirty[tile(2, 1)] = true;
        dirty[tile(6, 5)] = true;
        assert_eq!(
            dirty_bounds(&dirty, 16, 1000, 1000),
            Damage::Partial(rect(128, 64, 320, 320))
        );
        let mut corner = vec![false; 16 * 16];
        corner[tile(15, 15)] = true;
        assert_eq!(
            dirty_bounds(&corner, 16, 1000, 1000),
            Damage::Partial(rect(960, 960, 40, 40))
        );
        let mut half = vec![false; 16 * 16];
        half[tile(0, 0)] = true;
        half[tile(15, 8)] = true;
        assert_eq!(dirty_bounds(&half, 16, 1000, 1000), Damage::Full);
    }

    #[test]
    fn inset_shadow_replay_covers_the_element_outside_its_offset_hole() {
        let mut scene = finished([Shadow {
            blur_radius: ScaledPixels(0.5),
            bounds: bounds(140.0, 140.0, 20.0, 20.0),
            element_bounds: bounds(10.0, 10.0, 180.0, 180.0),
            inset: true.into(),
            ..blurred_shadow(0.0)
        }
        .into()]);
        let batch = PrimitiveBatch::Shadows {
            range: 0..1,
            smoothed: false,
        };
        let outside_hole = rect(20, 20, 8, 8);
        assert!(batch_intersects(&scene, &batch, outside_hole));
        scene.shadows[0].inset = false.into();
        assert!(!batch_intersects(&scene, &batch, outside_hole));
    }

    #[test]
    fn clear_quads_use_the_target_colour() {
        assert_eq!(
            clear_quad(rect(1, 2, 3, 4), [0.0, 0.0, 0.0, 1.0]).background,
            gpui::black().into()
        );
    }
}
