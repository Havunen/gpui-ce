//! Bounded hardware scissor runs keep original quad vertices and shader interpolation.
use crate::damage::DamageRect;
use gpui::Quad;
use std::ops::Range;

const MAX_RUNS: usize = 32;
const MIN_QUAD_AREA: f32 = 4096.0;

/// A consecutive painter-order run sharing one conservative hardware scissor.
pub struct QuadScissorRun {
    /// Instance range relative to the original batch; no re-upload is required.
    pub range: Range<usize>,
    /// Scissor intersected with the caller's existing damage/target scissor.
    pub scissor: DamageRect,
}

/// Iterator bounded to 32 runs per batch; remaining instances use the original scissor.
pub struct QuadScissorRuns<'a> {
    quads: &'a [Quad],
    active: DamageRect,
    cursor: usize,
    runs: usize,
}

/// Clip only large quads losing at least half their area, without changing shader inputs.
pub fn quad_scissor_runs(quads: &[Quad], active: DamageRect) -> QuadScissorRuns<'_> {
    QuadScissorRuns {
        quads,
        active,
        cursor: 0,
        runs: 0,
    }
}

impl Iterator for QuadScissorRuns<'_> {
    type Item = QuadScissorRun;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor == self.quads.len() {
            return None;
        }
        let start = self.cursor;
        if self.runs == MAX_RUNS - 1 {
            self.cursor = self.quads.len();
            self.runs += 1;
            return Some(QuadScissorRun {
                range: start..self.cursor,
                scissor: self.active,
            });
        }
        let scissor = quad_scissor(&self.quads[start], self.active);
        self.cursor += 1;
        while self.cursor < self.quads.len()
            && quad_scissor(&self.quads[self.cursor], self.active) == scissor
        {
            self.cursor += 1;
        }
        self.runs += 1;
        Some(QuadScissorRun {
            range: start..self.cursor,
            scissor,
        })
    }
}

fn quad_scissor(quad: &Quad, active: DamageRect) -> DamageRect {
    let bounds = quad.bounds;
    let mask = quad.content_mask.bounds;
    let x = bounds.origin.x.0;
    let y = bounds.origin.y.0;
    let width = bounds.size.width.0;
    let height = bounds.size.height.0;
    let mx = mask.origin.x.0;
    let my = mask.origin.y.0;
    let mw = mask.size.width.0;
    let mh = mask.size.height.0;
    if ![x, y, width, height, mx, my, mw, mh]
        .iter()
        .all(|v| v.is_finite())
        || width <= 0.0
        || height <= 0.0
        || mw < 0.0
        || mh < 0.0
        || width * height < MIN_QUAD_AREA
    {
        return active;
    }
    // A pixel fringe preserves clip-edge interpolation, sample coverage and
    // helper invocations. Original geometry, paints, fade and SDF math remain intact.
    let left = x.max(mx - 1.0);
    let top = y.max(my - 1.0);
    let right = (x + width).min(mx + mw + 1.0);
    let bottom = (y + height).min(my + mh + 1.0);
    if (right - left).max(0.0) * (bottom - top).max(0.0) > width * height * 0.5 {
        return active;
    }
    let active_right = active.x.saturating_add(active.width);
    let active_bottom = active.y.saturating_add(active.height);
    let left = (left.floor() as u32).clamp(active.x, active_right);
    let top = (top.floor() as u32).clamp(active.y, active_bottom);
    let right = (right.ceil() as u32).clamp(left, active_right);
    let bottom = (bottom.ceil() as u32).clamp(top, active_bottom);
    DamageRect {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{Bounds, ContentMask, ScaledPixels, point, size};
    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(w), ScaledPixels(h)),
        )
    }
    #[test]
    fn fractional_scissor_keeps_fringe_damage_clip_and_small_quad_fallback() {
        let active = DamageRect {
            x: 0,
            y: 0,
            width: 200,
            height: 100,
        };
        let mut quad = Quad {
            bounds: bounds(7.25, 8.5, 180.25, 83.25),
            content_mask: ContentMask {
                bounds: bounds(184.5, 8.5, 3.0, 83.25),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(
            quad_scissor(&quad, active),
            DamageRect {
                x: 183,
                y: 8,
                width: 5,
                height: 84
            }
        );
        let damage = DamageRect {
            x: 184,
            y: 10,
            width: 2,
            height: 5,
        };
        assert_eq!(quad_scissor(&quad, damage), damage);
        quad.bounds = bounds(0.0, 0.0, 8.0, 8.0);
        assert_eq!(quad_scissor(&quad, active), active);
        quad.bounds = bounds(0.0, 0.0, f32::NAN, 100.0);
        assert_eq!(quad_scissor(&quad, active), active);
    }
    #[test]
    fn runs_preserve_all_order_and_bound_splits_without_allocating_uploads() {
        let active = DamageRect {
            x: 0,
            y: 0,
            width: 200,
            height: 100,
        };
        let quads: Vec<_> = (0..80)
            .map(|x| Quad {
                bounds: bounds(0.0, 0.0, 200.0, 100.0),
                content_mask: ContentMask {
                    bounds: bounds(x as f32 * 2.0, 0.0, 1.5, 100.0),
                    ..Default::default()
                },
                ..Default::default()
            })
            .collect();
        let runs: Vec<_> = quad_scissor_runs(&quads, active).collect();
        assert_eq!(runs.len(), MAX_RUNS);
        assert_eq!(runs.last().unwrap().scissor, active);
        let indices: Vec<_> = runs.into_iter().flat_map(|r| r.range).collect();
        assert_eq!(indices, (0..80).collect::<Vec<_>>());
    }
}
