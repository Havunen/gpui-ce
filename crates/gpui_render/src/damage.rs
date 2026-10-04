//! Conservative damage for colour-only changes in an otherwise identical scene.
//! Layout, ordering, filters, surfaces and changed shadows require a full redraw.
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
        if !scene.surfaces.is_empty() || scene.requires_offscreen_rendering() {
            return None;
        }
        let bytes = scene
            .quads
            .len()
            .saturating_mul(std::mem::size_of::<Quad>())
            .saturating_add(
                scene
                    .monochrome_sprites
                    .len()
                    .saturating_mul(std::mem::size_of::<MonochromeSprite>()),
            )
            .saturating_add(
                path_types::rasterization_vertex_count(&scene.paths)
                    .saturating_mul(std::mem::size_of::<PathRasterizationVertex>()),
            )
            .saturating_add(
                scene
                    .shadows
                    .len()
                    .saturating_mul(std::mem::size_of::<gpui::Shadow>()),
            )
            .saturating_add(
                scene
                    .polychrome_sprites
                    .len()
                    .saturating_mul(std::mem::size_of::<gpui::PolychromeSprite>()),
            )
            .saturating_add(
                scene
                    .subpixel_sprites
                    .len()
                    .saturating_mul(std::mem::size_of::<gpui::SubpixelSprite>()),
            )
            .saturating_add(
                scene
                    .underlines
                    .len()
                    .saturating_mul(std::mem::size_of::<gpui::Underline>()),
            );
        if bytes > 2 * 1024 * 1024 {
            return None;
        }
        let mut fixed = Vec::new();
        append(&mut fixed, &scene.shadows);
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
        let mut scene = unfinished_scene(x, colour);
        scene.finish();
        scene
    }
    fn unfinished_scene(x: f32, colour: u32) -> Scene {
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

    fn glyph_scene(colour: gpui::Hsla, transformation: gpui::TransformationMatrix) -> Scene {
        let bounds = Bounds::new(
            point(ScaledPixels(20.), ScaledPixels(30.)),
            size(ScaledPixels(8.), ScaledPixels(12.)),
        );
        let mut scene = Scene::default();
        scene.monochrome_sprites.push(MonochromeSprite {
            order: 0,
            padding: 0,
            bounds,
            content_mask: ContentMask {
                bounds: Bounds::new(
                    point(ScaledPixels(0.), ScaledPixels(0.)),
                    size(ScaledPixels(200.), ScaledPixels(100.)),
                ),
                ..Default::default()
            },
            color: colour.into(),
            tile: gpui::AtlasTile {
                texture_id: gpui::AtlasTextureId {
                    index: 0,
                    kind: gpui::AtlasTextureKind::Monochrome,
                },
                tile_id: gpui::TileId(1),
                padding: 0,
                bounds: Default::default(),
            },
            transformation,
        });
        scene.finish();
        scene
    }

    #[test]
    fn glyph_recolours_are_bounded_unless_transformed() {
        let (white, red) = (gpui::white(), gpui::hsla(0., 1., 0.5, 1.));
        let unit = gpui::TransformationMatrix::default();
        let previous = Snapshot::capture(&glyph_scene(white, unit), 1).unwrap();
        assert_eq!(
            Snapshot::capture(&glyph_scene(red, unit), 1)
                .unwrap()
                .compare(&previous, (200, 100)),
            Damage::Rect(PixelRect {
                x: 18,
                y: 28,
                width: 12,
                height: 16
            })
        );
        // A transformed glyph may paint outside its bounds.
        let rotated = gpui::TransformationMatrix::unit().rotate(gpui::radians(0.5));
        let previous = Snapshot::capture(&glyph_scene(white, rotated), 1).unwrap();
        assert_eq!(
            Snapshot::capture(&glyph_scene(red, rotated), 1)
                .unwrap()
                .compare(&previous, (200, 100)),
            Damage::Full
        );
    }

    #[test]
    fn changed_shadows_redraw_everything_and_offscreen_scenes_are_not_tracked() {
        let previous = Snapshot::capture(&scene(10., 0x000000ff), 1).unwrap();
        let mut shadowed = unfinished_scene(10., 0x000000ff);
        let bounds = Bounds::new(
            point(ScaledPixels(15.), ScaledPixels(40.)),
            size(ScaledPixels(100.), ScaledPixels(30.)),
        );
        shadowed.shadows.push(gpui::Shadow {
            order: 0,
            bounds,
            content_mask: ContentMask {
                bounds,
                ..Default::default()
            },
            blur_radius: ScaledPixels(6.),
            color: gpui::rgba(0x00880088).into(),
            corner_radii: Default::default(),
            element_bounds: bounds,
            element_corner_radii: Default::default(),
            inset: gpui::ShaderBool::Disabled,
            corner_smoothing: 0.,
        });
        shadowed.finish();
        assert_eq!(
            Snapshot::capture(&shadowed, 1)
                .unwrap()
                .compare(&previous, (200, 100)),
            Damage::Full
        );

        let mut filtered = unfinished_scene(10., 0x000000ff);
        let bounds = Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(ScaledPixels(50.), ScaledPixels(50.)),
        );
        filtered.insert_primitive(gpui::BackdropFilter {
            bounds,
            content_mask: ContentMask {
                bounds,
                ..Default::default()
            },
            filters: vec![gpui::ScaledFilter::Blur(ScaledPixels(2.))].into(),
            opacity: 1.,
            ..Default::default()
        });
        filtered.finish();
        assert!(Snapshot::capture(&filtered, 1).is_none());
    }
}
