//! Hold a rendered path scene alive for a process GPU-memory sampler.
//! Build with --features test-support; compare this same executable with
//! GPUI_GPU_EXPERIMENTS set to an empty string and to cropped-paths. This measures
//! allocation, not application latency or GPU execution time.
use gpui::{Bounds, ContentMask, DevicePixels, PathBuilder, Scene, point, px, size};
use gpui_ce_wgpu::WgpuHeadlessRenderer;
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    let mut builder = PathBuilder::fill();
    builder.move_to(point(px(16.), px(16.)));
    builder.line_to(point(px(500.), px(16.)));
    builder.line_to(point(px(16.), px(1390.)));
    builder.close();
    let mut path = builder.build()?.scale(1.);
    path.content_mask = ContentMask {
        bounds: Bounds::new(
            point(0., 0.).map(gpui::ScaledPixels),
            size(2560., 1440.).map(gpui::ScaledPixels),
        ),
        ..Default::default()
    };
    path.color = gpui::solid_background(gpui::rgb(0x35ad75));
    let mut scene = Scene::default();
    scene.insert_primitive(path);
    scene.finish();
    let mut renderer = WgpuHeadlessRenderer::new()?;
    for _ in 0..5 {
        renderer.render_scene_and_wait(&scene, size(DevicePixels(2560), DevicePixels(1440)))?;
    }
    println!("ready {}", std::process::id());
    io::stdout().flush()?;
    io::stdin().read_line(&mut String::new())?;
    Ok(())
}
