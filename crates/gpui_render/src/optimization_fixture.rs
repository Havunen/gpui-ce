//! Shared native pixel fixture: fractional/negative paths, gradients, hover, geometry,
//! popup removal, transparency, and size changes. Kept out of production builds.
use gpui::*;

pub fn viewport(frame: usize) -> Size<DevicePixels> {
    if frame == 10 {
        size(DevicePixels(360), DevicePixels(240))
    } else {
        size(DevicePixels(512), DevicePixels(320))
    }
}

pub fn scene(frame: usize) -> Scene {
    let mask = ContentMask {
        bounds: Bounds::new(
            point(ScaledPixels(-20.), ScaledPixels(-20.)),
            size(ScaledPixels(600.), ScaledPixels(400.)),
        ),
        ..Default::default()
    };
    let mut scene = Scene::default();
    for row in 0..4 {
        let x = if row == 0 { -5.25 } else { 16.25 } + if frame == 7 { 4. } else { 0. };
        let y = 12.25 + row as f32 * 50.;
        let mut builder = PathBuilder::fill();
        builder.move_to(point(px(x), px(y)));
        builder.line_to(point(px(x + 24.5), px(y + 9.25)));
        builder.line_to(point(px(x + 8.25), px(y + 24.5)));
        builder.close();
        let mut path = builder.build().unwrap().scale(1.0);
        path.order = row * 2 + 1;
        path.content_mask = mask;
        path.color = if frame >= 5 && row == 1 {
            rgba(0x00ffff88).into()
        } else {
            linear_gradient(
                90.,
                linear_color_stop(rgba(0xff000088), 0.),
                linear_color_stop(rgba(0x00ff0088), 1.),
            )
        };
        scene.paths.push(path);
        scene.quads.push(Quad {
            order: row * 2 + 2,
            bounds: Bounds::new(
                point(ScaledPixels(10.), ScaledPixels(30. + row as f32 * 50.)),
                size(ScaledPixels(32.), ScaledPixels(10.)),
            ),
            content_mask: mask,
            background: if frame.is_multiple_of(2) {
                rgba(0xffffff80).into()
            } else {
                rgba(0x0000ff80).into()
            },
            ..Default::default()
        });
    }
    if frame == 8 {
        scene.quads.push(Quad {
            order: 100,
            bounds: Bounds::new(
                point(ScaledPixels(10.), ScaledPixels(10.)),
                size(ScaledPixels(100.), ScaledPixels(100.)),
            ),
            content_mask: mask,
            background: rgba(0x00880088).into(),
            ..Default::default()
        });
    }
    // An offscreen batch must not clear a packed scratch target.
    let mut invisible = scene.paths[0].clone();
    invisible.content_mask.bounds = Bounds::new(
        point(ScaledPixels(700.), ScaledPixels(0.)),
        size(ScaledPixels(10.), ScaledPixels(10.)),
    );
    invisible.order = 4;
    scene.paths.push(invisible);
    scene.finish();
    scene
}

pub fn modes() -> impl Iterator<Item = crate::gpu_policy::GpuOptions> {
    [
        "cropped-paths,cached-layers",
        "cropped-paths,batched-paths",
        "cropped-paths,cached-layers,batched-paths",
        "cropped-paths,partial-redraw",
        "cropped-paths,cached-layers,batched-paths,partial-redraw,pooled-targets",
    ]
    .into_iter()
    .map(crate::gpu_policy::GpuOptions::parse)
}
