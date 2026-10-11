//! Opt-in per-frame scene composition statistics (`GPUI_SCENE_STATS=1`).
//!
//! Prints one line per drawn frame to stderr: primitive counts, the area each
//! primitive kind covers relative to the viewport (overdraw), and the render
//! passes the frame needs. `GPUI_SCENE_STATS=2` adds the largest quads.
//! Diagnostics only; nothing runs unless the variable is set.

use gpui::{
    Bounds, ContentMask, DevicePixels, PrimitiveBatch, RenderCommand, ScaledPixels, Scene, Size,
};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        std::env::var("GPUI_SCENE_STATS")
            .ok()
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(0)
    })
}

fn covered(bounds: &Bounds<ScaledPixels>, mask: &ContentMask<ScaledPixels>) -> f64 {
    let visible = bounds.intersect(&mask.bounds);
    (visible.size.width.0.max(0.0) as f64) * (visible.size.height.0.max(0.0) as f64)
}

pub(crate) fn log(scene: &Scene, viewport: Size<DevicePixels>) {
    static FRAME: AtomicU64 = AtomicU64::new(0);
    let frame = FRAME.fetch_add(1, Ordering::Relaxed);
    let viewport_area = (viewport.width.0.max(1) as f64) * (viewport.height.0.max(1) as f64);
    let ratio = |area: f64| area / viewport_area;

    let quad_area: f64 = scene
        .quads
        .iter()
        .map(|q| covered(&q.bounds, &q.content_mask))
        .sum();
    // What the rasterizer covers: content masks clip in the fragment shader,
    // so a quad's whole bounds (within the viewport) is shaded.
    let viewport_bounds = Bounds {
        origin: gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
        size: gpui::size(
            ScaledPixels(viewport.width.0 as f32),
            ScaledPixels(viewport.height.0 as f32),
        ),
    };
    let raster = |bounds: &Bounds<ScaledPixels>| {
        covered(
            bounds,
            &ContentMask {
                bounds: viewport_bounds,
                ..Default::default()
            },
        )
    };
    let quad_raster: f64 = scene.quads.iter().map(|q| raster(&q.bounds)).sum();
    let mono_raster: f64 = scene
        .monochrome_sprites
        .iter()
        .map(|q| raster(&q.bounds))
        .sum();
    let shadow_raster: f64 = scene.shadows.iter().map(|q| raster(&q.bounds)).sum();
    let opaque_quad_area: f64 = scene
        .quads
        .iter()
        .filter(|q| crate::metal_renderer::quad_is_opaque(q))
        .map(|q| covered(&q.bounds, &q.content_mask))
        .sum();
    let mono_area: f64 = scene
        .monochrome_sprites
        .iter()
        .map(|s| covered(&s.bounds, &s.content_mask))
        .sum();
    let poly_area: f64 = scene
        .polychrome_sprites
        .iter()
        .map(|s| covered(&s.bounds, &s.content_mask))
        .sum();
    let shadow_area: f64 = scene
        .shadows
        .iter()
        .map(|s| covered(&s.bounds, &s.content_mask))
        .sum();
    let path_area: f64 = scene
        .paths
        .iter()
        .map(|p| covered(&p.bounds, &p.content_mask))
        .sum();

    let mut commands = 0usize;
    let mut path_batches = Vec::new();
    for command in scene.render_commands() {
        commands += 1;
        if let RenderCommand::Batch(PrimitiveBatch::Paths { range, .. }) = command {
            let paths = &scene.paths[range.clone()];
            let union = paths
                .iter()
                .map(|p| p.bounds.intersect(&p.content_mask.bounds))
                .reduce(|a, b| a.union(&b));
            if let Some(union) = union {
                path_batches.push(format!(
                    "{}@{:.0},{:.0} {:.0}x{:.0}",
                    paths.len(),
                    union.origin.x.0,
                    union.origin.y.0,
                    union.size.width.0,
                    union.size.height.0
                ));
            }
        }
    }

    eprintln!(
        "gpui-scene frame={frame} viewport={}x{} commands={commands} quads={} quad_cover={:.2} opaque_quad_cover={:.2} quad_raster={:.2} mono_raster={:.2} shadow_raster={:.2} mono={} mono_cover={:.2} poly={} poly_cover={:.2} shadows={} shadow_cover={:.2} underlines={} paths={} path_cover={:.3} path_batches=[{}] offscreen={}",
        viewport.width.0,
        viewport.height.0,
        scene.quads.len(),
        ratio(quad_area),
        ratio(opaque_quad_area),
        ratio(quad_raster),
        ratio(mono_raster),
        ratio(shadow_raster),
        scene.monochrome_sprites.len(),
        ratio(mono_area),
        scene.polychrome_sprites.len(),
        ratio(poly_area),
        scene.shadows.len(),
        ratio(shadow_area),
        scene.underlines.len(),
        scene.paths.len(),
        ratio(path_area),
        path_batches.join("; "),
        scene.requires_offscreen_rendering(),
    );

    if level() >= 2 {
        for shadow in &scene.shadows {
            eprintln!(
                "gpui-scene   shadow cover={:.3} at {:.0},{:.0} {:.0}x{:.0} blur={:.1} order={}",
                ratio(covered(&shadow.bounds, &shadow.content_mask)),
                shadow.bounds.origin.x.0,
                shadow.bounds.origin.y.0,
                shadow.bounds.size.width.0,
                shadow.bounds.size.height.0,
                shadow.blur_radius.0,
                shadow.order,
            );
        }
        let mut largest: Vec<_> = scene
            .quads
            .iter()
            .map(|q| (covered(&q.bounds, &q.content_mask), q))
            .filter(|(area, quad)| {
                *area >= viewport_area * if level() >= 3 { 0.002 } else { 0.05 }
                    && (level() < 3 || !crate::metal_renderer::quad_is_opaque(quad))
            })
            .collect();
        largest.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (area, quad) in largest.iter().take(16) {
            eprintln!(
                "gpui-scene   quad cover={:.2} at {:.0},{:.0} {:.0}x{:.0} order={} bg={:?} border={:?} radii={:?}",
                ratio(*area),
                quad.bounds.origin.x.0,
                quad.bounds.origin.y.0,
                quad.bounds.size.width.0,
                quad.bounds.size.height.0,
                quad.order,
                quad.background,
                quad.border_widths,
                quad.corner_radii,
            );
        }
    }
}
