//! Screen regions that changed between consecutive scenes (partial redraw).
//!
//! Each frame hashes, per square tile of the viewport, everything drawn into
//! that tile in draw order. Tiles whose hash differs from the previous frame
//! changed; their bounding rectangle is the damage a partial redraw repaints.
//! A primitive's global draw order is left out of its hash, so unrelated
//! additions elsewhere do not dirty a tile, while the order within a tile
//! still counts.

use gpui::{
    Bounds, ContentMask, DevicePixels, PrimitiveBatch, RenderCommand, ScaledPixels, Scene,
    ShaderBool, Size, TransformationMatrix, point,
};
use gpui_render::blur::GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS;

/// Tile edge in device pixels.
const TILE: i32 = 64;

/// What changed since the previous frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Damage {
    /// Unknown or unsupported: repaint everything.
    Full,
    /// Identical to the previous frame.
    None,
    /// Repaint this region, in device pixels, tile aligned and inside the viewport.
    Rect {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
}

#[derive(Default)]
pub(crate) struct DamageTracker {
    /// The previous frame's viewport; `None` when there is nothing to compare.
    size: Option<Size<DevicePixels>>,
    atlas_generation: u64,
    hashes: Vec<u64>,
    scratch: Vec<u64>,
}

impl DamageTracker {
    /// Forget the previous frame: the next update reports `Full`.
    pub(crate) fn invalidate(&mut self) {
        self.size = None;
    }

    /// Compares `scene` with the previous update's scene. Sprites refer to
    /// atlas tiles, so a changed atlas generation (tiles freed and possibly
    /// reused) repaints everything.
    pub(crate) fn update(
        &mut self,
        scene: &Scene,
        viewport: Size<DevicePixels>,
        atlas_generation: u64,
    ) -> Damage {
        let (width, height) = (viewport.width.0.max(0), viewport.height.0.max(0));
        let grid = Grid {
            columns: ((width + TILE - 1) / TILE).max(1) as usize,
            rows: ((height + TILE - 1) / TILE).max(1) as usize,
            width,
            height,
        };
        self.scratch.clear();
        self.scratch.resize(grid.columns * grid.rows, SEED);
        if !hash_scene(scene, &mut self.scratch, &grid) {
            self.invalidate();
            return Damage::Full;
        }
        let comparable = self.size == Some(viewport) && self.atlas_generation == atlas_generation;
        std::mem::swap(&mut self.hashes, &mut self.scratch);
        self.size = Some(viewport);
        self.atlas_generation = atlas_generation;
        if !comparable {
            return Damage::Full;
        }
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (usize::MAX, usize::MAX, 0, 0);
        for (index, (new, old)) in self.hashes.iter().zip(&self.scratch).enumerate() {
            if new != old {
                let (x, y) = (index % grid.columns, index / grid.columns);
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
        if min_x == usize::MAX {
            return Damage::None;
        }
        let x = (min_x as i32 * TILE).min(width);
        let y = (min_y as i32 * TILE).min(height);
        let right = ((max_x as i32 + 1) * TILE).min(width);
        let bottom = ((max_y as i32 + 1) * TILE).min(height);
        Damage::Rect {
            x: x as u32,
            y: y as u32,
            width: (right - x).max(0) as u32,
            height: (bottom - y).max(0) as u32,
        }
    }
}

struct Grid {
    columns: usize,
    rows: usize,
    width: i32,
    height: i32,
}

const SEED: u64 = 0x9e37_79b9_7f4a_7c15;

fn mix(hash: u64, value: u64) -> u64 {
    (hash.rotate_left(5) ^ value).wrapping_mul(0x5170_9ab3_b8d4_d9e3)
}

/// The bytes of a scene primitive.
///
/// Only for the `#[repr(C)]` scene types, whose fields are four-byte scalars
/// with explicit padding fields, so every byte is initialized.
fn primitive_bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: see above; the slice covers exactly the values' memory.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// Hashes `bytes` in four independent lanes, then avalanches the result.
fn hash_bytes(kind: u64, bytes: &[u8]) -> u64 {
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

/// Adds `hash` to every tile the primitive can touch: its geometry clipped to
/// its content mask and the viewport, rounded out to whole pixels. Pixels are
/// shaded only when their centre (for paths, a sample) lies inside the
/// geometry, so nothing outside that range changes.
fn mark(
    tiles: &mut [u64],
    grid: &Grid,
    geometry: &Bounds<ScaledPixels>,
    mask: &ContentMask<ScaledPixels>,
    hash: u64,
) {
    let visible = geometry.intersect(&mask.bounds);
    if !(visible.size.width.0 > 0.0 && visible.size.height.0 > 0.0) {
        return;
    }
    let left = visible.origin.x.0.floor().max(0.0) as i32;
    let top = visible.origin.y.0.floor().max(0.0) as i32;
    let right = ((visible.origin.x.0 + visible.size.width.0).ceil() as i32).min(grid.width);
    let bottom = ((visible.origin.y.0 + visible.size.height.0).ceil() as i32).min(grid.height);
    if right <= left || bottom <= top {
        return;
    }
    let columns = (left / TILE) as usize..=(((right - 1) / TILE) as usize).min(grid.columns - 1);
    for row in (top / TILE) as usize..=(((bottom - 1) / TILE) as usize).min(grid.rows - 1) {
        let line = &mut tiles[row * grid.columns..(row + 1) * grid.columns];
        for tile in &mut line[columns.clone()] {
            *tile = mix(*tile, hash);
        }
    }
}

/// Returns false for scenes partial redraw does not support: filters sample
/// pixels outside their own bounds and surfaces change without the scene.
fn hash_scene(scene: &Scene, tiles: &mut [u64], grid: &Grid) -> bool {
    for command in scene.render_commands() {
        let RenderCommand::Batch(batch) = command else {
            return false;
        };
        match batch {
            PrimitiveBatch::Quads { range, smoothed } => {
                for quad in &scene.quads[range.clone()] {
                    let mut key = *quad;
                    key.order = 0;
                    let hash = hash_bytes(1 + u64::from(*smoothed), primitive_bytes(&[key]));
                    mark(tiles, grid, &quad.bounds, &quad.content_mask, hash);
                }
            }
            PrimitiveBatch::Shadows { range, smoothed } => {
                for shadow in &scene.shadows[range.clone()] {
                    let mut key = *shadow;
                    key.order = 0;
                    let hash = hash_bytes(3 + u64::from(*smoothed), primitive_bytes(&[key]));
                    // Blurred shadows extend past their bounds; inset ones stay
                    // inside the element.
                    let geometry = if shadow.inset == ShaderBool::Enabled {
                        shadow.element_bounds
                    } else {
                        shadow.bounds.dilate(ScaledPixels(
                            GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * shadow.blur_radius.0.max(0.0),
                        ))
                    };
                    mark(tiles, grid, &geometry, &shadow.content_mask, hash);
                }
            }
            PrimitiveBatch::Underlines(range) => {
                for underline in &scene.underlines[range.clone()] {
                    let mut key = *underline;
                    key.order = 0;
                    let hash = hash_bytes(5, primitive_bytes(&[key]));
                    mark(
                        tiles,
                        grid,
                        &underline.bounds,
                        &underline.content_mask,
                        hash,
                    );
                }
            }
            PrimitiveBatch::MonochromeSprites { range, .. } => {
                for sprite in &scene.monochrome_sprites[range.clone()] {
                    let mut key = *sprite;
                    key.order = 0;
                    let hash = hash_bytes(6, primitive_bytes(&[key]));
                    let geometry = transformed(sprite.bounds, &sprite.transformation);
                    mark(tiles, grid, &geometry, &sprite.content_mask, hash);
                }
            }
            PrimitiveBatch::PolychromeSprites {
                range, smoothed, ..
            } => {
                for sprite in &scene.polychrome_sprites[range.clone()] {
                    let mut key = *sprite;
                    key.order = 0;
                    let hash = hash_bytes(7 + u64::from(*smoothed), primitive_bytes(&[key]));
                    mark(tiles, grid, &sprite.bounds, &sprite.content_mask, hash);
                }
            }
            PrimitiveBatch::Paths { range, .. } => {
                for path in &scene.paths[range.clone()] {
                    let mut hash = hash_bytes(9, primitive_bytes(&path.vertices));
                    hash = mix(hash, hash_bytes(10, primitive_bytes(&[path.bounds])));
                    hash = mix(hash, hash_bytes(11, primitive_bytes(&[path.content_mask])));
                    hash = mix(hash, hash_bytes(12, primitive_bytes(&[path.color])));
                    mark(tiles, grid, &path.bounds, &path.content_mask, hash);
                }
            }
            PrimitiveBatch::Surfaces(_)
            | PrimitiveBatch::BackdropFilters(_)
            | PrimitiveBatch::SubpixelSprites { .. }
            | PrimitiveBatch::FilterBoundary(_) => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{Corners, Quad, Shadow, hsla, size, solid_background};

    const VIEWPORT: Size<DevicePixels> = Size {
        width: DevicePixels(640),
        height: DevicePixels(480),
    };

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(width), ScaledPixels(height)),
        }
    }

    fn quad(bounds: Bounds<ScaledPixels>, hue: f32) -> Quad {
        Quad {
            bounds,
            content_mask: ContentMask {
                bounds: rect(0.0, 0.0, 640.0, 480.0),
                ..Default::default()
            },
            background: solid_background(hsla(hue, 0.5, 0.5, 1.0)),
            ..Default::default()
        }
    }

    fn scene(quads: &[Quad]) -> Scene {
        let mut scene = Scene::default();
        for quad in quads {
            scene.insert_primitive(*quad);
        }
        scene.finish();
        scene
    }

    fn rect_damage(x: u32, y: u32, width: u32, height: u32) -> Damage {
        Damage::Rect {
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn identical_scenes_have_no_damage() {
        let quads = [
            quad(rect(0.0, 0.0, 640.0, 480.0), 0.1),
            quad(rect(10.0, 10.0, 20.0, 20.0), 0.2),
        ];
        let mut tracker = DamageTracker::default();
        assert_eq!(tracker.update(&scene(&quads), VIEWPORT, 0), Damage::Full);
        assert_eq!(tracker.update(&scene(&quads), VIEWPORT, 0), Damage::None);
    }

    #[test]
    fn changes_damage_the_tiles_they_touch() {
        let background = quad(rect(0.0, 0.0, 640.0, 480.0), 0.1);
        let small = |bounds, hue| scene(&[background, quad(bounds, hue)]);
        let mut tracker = DamageTracker::default();
        tracker.update(&small(rect(70.0, 10.0, 20.0, 20.0), 0.2), VIEWPORT, 0);
        // A colour change stays within the quad's tile.
        assert_eq!(
            tracker.update(&small(rect(70.0, 10.0, 20.0, 20.0), 0.3), VIEWPORT, 0),
            rect_damage(64, 0, 64, 64)
        );
        // A move damages where it was and where it is now.
        assert_eq!(
            tracker.update(&small(rect(200.0, 300.0, 20.0, 20.0), 0.3), VIEWPORT, 0),
            rect_damage(64, 0, 192, 320)
        );
        // Partially covered pixels count, also across a tile edge.
        tracker.update(&scene(&[background]), VIEWPORT, 0);
        assert_eq!(
            tracker.update(&small(rect(200.0, 300.0, 20.0, 19.5), 0.3), VIEWPORT, 0),
            rect_damage(192, 256, 64, 64)
        );
        tracker.update(&scene(&[background]), VIEWPORT, 0);
        assert_eq!(
            tracker.update(&small(rect(200.0, 300.0, 20.0, 20.5), 0.3), VIEWPORT, 0),
            rect_damage(192, 256, 64, 128)
        );
        tracker.update(&scene(&[background]), VIEWPORT, 0);
        assert_eq!(
            tracker.update(&small(rect(100.0, 63.5, 10.0, 10.0), 0.3), VIEWPORT, 0),
            rect_damage(64, 0, 64, 128)
        );
        // Content masks clip what a primitive can touch.
        let mut clipped = quad(rect(0.0, 0.0, 300.0, 300.0), 0.4);
        clipped.content_mask.bounds = rect(10.0, 10.0, 20.0, 20.0);
        tracker.update(&scene(&[background]), VIEWPORT, 0);
        assert_eq!(
            tracker.update(&scene(&[background, clipped]), VIEWPORT, 0),
            rect_damage(0, 0, 64, 64)
        );
    }

    #[test]
    fn draw_order_counts_only_within_a_tile() {
        let background = quad(rect(0.0, 0.0, 640.0, 480.0), 0.1);
        let a = quad(rect(10.0, 10.0, 20.0, 20.0), 0.2);
        let b = quad(rect(20.0, 20.0, 20.0, 20.0), 0.3);
        let far = quad(rect(400.0, 400.0, 20.0, 20.0), 0.4);
        let mut tracker = DamageTracker::default();
        tracker.update(&scene(&[background, far, a, b]), VIEWPORT, 0);
        // Moving an unrelated primitive earlier in the frame shifts every
        // later draw order but damages nothing.
        assert_eq!(
            tracker.update(&scene(&[background, a, b, far]), VIEWPORT, 0),
            Damage::None
        );
        // Swapping overlapping primitives changes what is on top.
        assert_eq!(
            tracker.update(&scene(&[background, b, a, far]), VIEWPORT, 0),
            rect_damage(0, 0, 64, 64)
        );
    }

    #[test]
    fn viewport_and_atlas_changes_repaint_everything() {
        let quads = [quad(rect(0.0, 0.0, 640.0, 480.0), 0.1)];
        let mut tracker = DamageTracker::default();
        tracker.update(&scene(&quads), VIEWPORT, 0);
        assert_eq!(tracker.update(&scene(&quads), VIEWPORT, 1), Damage::Full);
        assert_eq!(tracker.update(&scene(&quads), VIEWPORT, 1), Damage::None);
        let wider = size(DevicePixels(700), DevicePixels(480));
        assert_eq!(tracker.update(&scene(&quads), wider, 1), Damage::Full);
        tracker.invalidate();
        assert_eq!(tracker.update(&scene(&quads), wider, 1), Damage::Full);
    }

    #[test]
    fn blurred_shadows_damage_their_blur() {
        let background = quad(rect(0.0, 0.0, 640.0, 480.0), 0.1);
        let shadow = |x: f32| Shadow {
            order: 0,
            blur_radius: ScaledPixels(10.0),
            bounds: rect(x, 200.0, 40.0, 40.0),
            corner_radii: Corners::default(),
            content_mask: ContentMask {
                bounds: rect(0.0, 0.0, 640.0, 480.0),
                ..Default::default()
            },
            color: solid_background(hsla(0.0, 0.0, 0.0, 0.5)),
            element_bounds: rect(x, 200.0, 40.0, 40.0),
            element_corner_radii: Corners::default(),
            inset: ShaderBool::Disabled,
            corner_smoothing: 0.0,
        };
        let frame = |x: f32| {
            let mut scene = Scene::default();
            scene.insert_primitive(background);
            scene.insert_primitive(shadow(x));
            scene.finish();
            scene
        };
        let mut tracker = DamageTracker::default();
        tracker.update(&frame(300.0), VIEWPORT, 0);
        // The blur reaches 30 px past the bounds: 270..370 then 271..371.
        assert_eq!(
            tracker.update(&frame(301.0), VIEWPORT, 0),
            rect_damage(256, 128, 128, 192)
        );
    }

    #[test]
    fn transformed_sprites_cover_their_rotated_bounds() {
        let bounds = rect(100.0, 100.0, 20.0, 10.0);
        let matrix = TransformationMatrix::unit()
            .translate(point(ScaledPixels(110.0), ScaledPixels(105.0)))
            .rotate(gpui::radians(std::f32::consts::FRAC_PI_2))
            .translate(point(ScaledPixels(-110.0), ScaledPixels(-105.0)));
        let covered = transformed(bounds, &matrix);
        let close = |a: ScaledPixels, b: f32| (a.0 - b).abs() < 1e-3;
        assert!(
            close(covered.origin.x, 105.0) && close(covered.origin.y, 95.0),
            "{covered:?}"
        );
        assert!(
            close(covered.size.width, 10.0) && close(covered.size.height, 20.0),
            "{covered:?}"
        );
    }

    #[test]
    fn unsupported_scenes_repaint_everything() {
        let mut tracker = DamageTracker::default();
        let mut scene = Scene::default();
        scene.insert_primitive(quad(rect(0.0, 0.0, 640.0, 480.0), 0.1));
        scene.finish();
        tracker.update(&scene, VIEWPORT, 0);
        let mut with_filter = Scene::default();
        with_filter.insert_primitive(quad(rect(0.0, 0.0, 640.0, 480.0), 0.1));
        with_filter.insert_primitive(gpui::BackdropFilter {
            bounds: rect(10.0, 10.0, 100.0, 100.0),
            content_mask: ContentMask {
                bounds: rect(0.0, 0.0, 640.0, 480.0),
                ..Default::default()
            },
            ..Default::default()
        });
        with_filter.finish();
        assert_eq!(tracker.update(&with_filter, VIEWPORT, 0), Damage::Full);
        // And the next supported frame has nothing to compare against.
        assert_eq!(tracker.update(&scene, VIEWPORT, 0), Damage::Full);
    }
}
