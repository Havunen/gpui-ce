//! Conservative damage for colour-only changes in an otherwise identical scene.
//! Layout, ordering, filters, surfaces and shadows always require a full redraw.
use crate::{
    path_plan::{PixelRect, visible_rect},
    path_types::{self, PathRasterizationVertex},
    shaders::interface::{BufferData, bytes_of},
};
use gpui::{Bounds, MonochromeSprite, Quad, RenderCommand, ScaledPixels, Scene};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Damage {
    Full,
    Unchanged,
    Rect(PixelRect),
}
pub struct Snapshot {
    commands: Vec<RenderCommand>,
    quads: Vec<Quad>,
    sprites: Vec<MonochromeSprite>,
    paths: Vec<PathRasterizationVertex>,
    fixed: Vec<u8>,
    atlas_generation: u64,
}
fn append<T: BufferData>(bytes: &mut Vec<u8>, values: &[T]) {
    bytes.extend_from_slice(&(values.len() as u64).to_ne_bytes());
    for value in values {
        bytes.extend_from_slice(bytes_of(value));
    }
}
impl Snapshot {
    pub fn capture(scene: &Scene, atlas_generation: u64) -> Option<Self> {
        if !scene.shadows.is_empty()
            || !scene.surfaces.is_empty()
            || scene.requires_offscreen_rendering()
        {
            return None;
        }
        let mut fixed = Vec::new();
        append(&mut fixed, &scene.subpixel_sprites);
        append(&mut fixed, &scene.polychrome_sprites);
        append(&mut fixed, &scene.underlines);
        Some(Self {
            commands: scene.render_commands().to_vec(),
            quads: scene.quads.clone(),
            sprites: scene.monochrome_sprites.clone(),
            paths: path_types::rasterization_vertices(&scene.paths).collect(),
            fixed,
            atlas_generation,
        })
    }
    pub fn compare(&self, previous: &Self, viewport: (u32, u32)) -> Damage {
        if self.commands != previous.commands
            || self.fixed != previous.fixed
            || self.paths != previous.paths
            || self.atlas_generation != previous.atlas_generation
            || self.quads.len() != previous.quads.len()
            || self.sprites.len() != previous.sprites.len()
        {
            return Damage::Full;
        }
        let mut changed = Vec::<Bounds<ScaledPixels>>::new();
        for (new, old) in self.quads.iter().zip(&previous.quads) {
            if bytes_of(new) == bytes_of(old) {
                continue;
            }
            let mut geometry = *new;
            geometry.background = old.background;
            geometry.border_color = old.border_color;
            if bytes_of(&geometry) != bytes_of(old) {
                return Damage::Full;
            }
            changed.push(old.bounds.intersect(&old.content_mask.bounds));
        }
        for (new, old) in self.sprites.iter().zip(&previous.sprites) {
            if bytes_of(new) == bytes_of(old) {
                continue;
            }
            let mut geometry = *new;
            geometry.color = old.color;
            if bytes_of(&geometry) != bytes_of(old)
                || new.transformation != gpui::TransformationMatrix::default()
            {
                return Damage::Full;
            }
            changed.push(old.bounds.intersect(&old.content_mask.bounds));
        }
        match visible_rect(changed.into_iter(), viewport) {
            None => Damage::Unchanged,
            Some(rect)
                if u64::from(rect.width) * u64::from(rect.height) * 10
                    <= u64::from(viewport.0) * u64::from(viewport.1) * 4 =>
            {
                Damage::Rect(rect)
            }
            Some(_) => Damage::Full,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{ContentMask, point, size};
    fn scene(x: f32, colour: u32) -> Scene {
        let mut scene = Scene::default();
        scene.quads.push(Quad {
            bounds: Bounds::new(
                point(ScaledPixels(x), ScaledPixels(10.)),
                size(ScaledPixels(40.), ScaledPixels(20.)),
            ),
            content_mask: ContentMask {
                bounds: Bounds::new(
                    point(ScaledPixels(0.), ScaledPixels(0.)),
                    size(ScaledPixels(200.), ScaledPixels(100.)),
                ),
                ..Default::default()
            },
            background: gpui::rgba(colour).into(),
            ..Default::default()
        });
        scene.finish();
        scene
    }
    #[test]
    fn colour_damage_is_bounded_but_geometry_and_atlas_changes_redraw_everything() {
        let previous = Snapshot::capture(&scene(10., 0x000000ff), 1).unwrap();
        assert_eq!(previous.compare(&previous, (200, 100)), Damage::Unchanged);
        let hover = Snapshot::capture(&scene(10., 0xff000088), 1).unwrap();
        assert_eq!(
            hover.compare(&previous, (200, 100)),
            Damage::Rect(PixelRect {
                x: 8,
                y: 8,
                width: 44,
                height: 24
            })
        );
        assert_eq!(hover.compare(&previous, (40, 30)), Damage::Full);
        assert_eq!(
            Snapshot::capture(&scene(11., 0xff000088), 1)
                .unwrap()
                .compare(&previous, (200, 100)),
            Damage::Full
        );
        assert_eq!(
            Snapshot::capture(&scene(10., 0x000000ff), 2)
                .unwrap()
                .compare(&previous, (200, 100)),
            Damage::Full
        );
    }
}
