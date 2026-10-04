//! Geometry and packing used by path caches and rasterization prepasses.
use crate::{path_types, shaders::interface};
use gpui::{Bounds, Path, ScaledPixels};
use std::hash::Hasher;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
impl PixelRect {
    pub fn bytes(self) -> u64 {
        u64::from(self.width) * u64::from(self.height) * 4
    }
    pub fn origin(self) -> gpui::Point<ScaledPixels> {
        gpui::point(ScaledPixels(self.x as f32), ScaledPixels(self.y as f32))
    }
}
pub fn visible_rect(
    bounds: impl Iterator<Item = Bounds<ScaledPixels>>,
    viewport: (u32, u32),
) -> Option<PixelRect> {
    let mut left = viewport.0 as f32;
    let mut top = viewport.1 as f32;
    let mut right = 0_f32;
    let mut bottom = 0_f32;
    for b in bounds {
        if b.right().0 <= 0.0
            || b.bottom().0 <= 0.0
            || b.left().0 >= viewport.0 as f32
            || b.top().0 >= viewport.1 as f32
        {
            continue;
        }
        left = left.min(b.left().0);
        top = top.min(b.top().0);
        right = right.max(b.right().0);
        bottom = bottom.max(b.bottom().0);
    }
    if right <= left || bottom <= top {
        return None;
    }
    let x = (left.floor() as u32).saturating_sub(2);
    let y = (top.floor() as u32).saturating_sub(2);
    let right = (right.ceil() as u32).saturating_add(2).min(viewport.0);
    let bottom = (bottom.ceil() as u32).saturating_add(2).min(viewport.1);
    Some(PixelRect {
        x,
        y,
        width: right.saturating_sub(x),
        height: bottom.saturating_sub(y),
    })
}
pub fn fingerprint(paths: &[Path<ScaledPixels>]) -> u64 {
    let mut hash = std::hash::DefaultHasher::new();
    for vertex in path_types::rasterization_vertices(paths) {
        hash.write(interface::bytes_of(&vertex));
    }
    hash.finish()
}

/// Pack disjoint tiles, preserving each batch's painter order when compositing.
/// No allocation larger than the existing path scratch target is permitted.
pub fn pack(rects: &[PixelRect], extent: (u32, u32)) -> Option<Vec<PixelRect>> {
    if rects.len() < 2 {
        return None;
    }
    let mut order: Vec<_> = (0..rects.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(rects[i].height));
    let mut result = vec![PixelRect::default(); rects.len()];
    let (mut x, mut y, mut height) = (0_u32, 0_u32, 0_u32);
    for i in order {
        let r = rects[i];
        if r.width == 0 || r.height == 0 || r.width > extent.0 {
            return None;
        }
        if x.saturating_add(r.width) > extent.0 {
            x = 0;
            y = y.checked_add(height)?;
            height = 0;
        }
        if y.checked_add(r.height)? > extent.1 {
            return None;
        }
        result[i] = PixelRect { x, y, ..r };
        x += r.width;
        height = height.max(r.height);
    }
    Some(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packing_never_overlaps_tiles_and_never_grows_the_target() {
        let rects = [
            PixelRect {
                x: 30,
                y: 50,
                width: 40,
                height: 10,
            },
            PixelRect {
                x: 30,
                y: 50,
                width: 40,
                height: 20,
            },
            PixelRect {
                x: 60,
                y: 80,
                width: 20,
                height: 15,
            },
        ];
        let packed = pack(&rects, (80, 40)).unwrap();
        for (i, a) in packed.iter().enumerate() {
            assert!(a.x + a.width <= 80 && a.y + a.height <= 40);
            assert_eq!((a.width, a.height), (rects[i].width, rects[i].height));
            for b in packed.iter().skip(i + 1) {
                assert!(
                    a.x + a.width <= b.x
                        || b.x + b.width <= a.x
                        || a.y + a.height <= b.y
                        || b.y + b.height <= a.y
                );
            }
        }
        assert!(pack(&rects, (39, 40)).is_none());
        assert!(pack(&rects, (80, 19)).is_none());
        assert!(pack(&rects[..1], (80, 40)).is_none());
    }
}
