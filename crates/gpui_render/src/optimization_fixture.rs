//! Shared native pixel fixture: fractional/negative paths, gradients, hover, geometry,
//! popup removal, transparency, and size changes. Kept out of production builds.
//!
//! Backends render each scene with and without experiments and compare the pixels.
//! Comparing two renderers alone cannot catch a defect they share (paths that never
//! paint pass every comparison), so the scenes also come with absolute expectations.
use gpui::*;

pub fn viewport(frame: usize) -> Size<DevicePixels> {
    if frame == 10 {
        size(DevicePixels(360), DevicePixels(240))
    } else {
        size(DevicePixels(512), DevicePixels(320))
    }
}

/// The top-left corner of the triangle in `row`, as `scene(frame)` draws it.
fn triangle_origin(frame: usize, row: usize) -> (f32, f32) {
    let x = if row == 0 { -5.25 } else { 16.25 } + if frame == 7 { 4. } else { 0. };
    (x, 12.25 + row as f32 * 50.)
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
        let (x, y) = triangle_origin(frame, row);
        let mut builder = PathBuilder::fill();
        builder.move_to(point(px(x), px(y)));
        builder.line_to(point(px(x + 24.5), px(y + 9.25)));
        builder.line_to(point(px(x + 8.25), px(y + 24.5)));
        builder.close();
        let mut path = builder.build().unwrap().scale(1.0);
        path.order = row as u32 * 2 + 1;
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
            order: row as u32 * 2 + 2,
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

/// Asserts that every triangle `scene(frame)` draws painted its interior. The
/// background clears to black (or transparent), and every path color is bright.
pub fn assert_paths_painted(rgba: &[u8], width: u32, frame: usize) {
    for row in 0..4 {
        let (x, y) = triangle_origin(frame, row);
        // The centroid, clear of the row's quad.
        let (x, y) = ((x + 32.75 / 3.) as u32, (y + 33.75 / 3.) as u32);
        let offset = (y * width + x) as usize * 4;
        let pixel = &rgba[offset..offset + 4];
        assert!(
            pixel[..3].iter().any(|channel| *channel >= 32),
            "frame {frame}: the row {row} path did not paint ({x}, {y}): {pixel:?}"
        );
    }
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

/// The largest difference between corresponding channels of two equally sized images.
pub fn max_difference(actual: &[u8], expected: &[u8]) -> u8 {
    assert_eq!(actual.len(), expected.len(), "the images differ in size");
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| actual.abs_diff(*expected))
        .max()
        .unwrap_or(0)
}

/// Triangle positions and viewports for [`cropped_path_scene`]: the cropped target
/// grows, is reused, and is recreated for a smaller viewport.
pub const CROPPED_PATH_CASES: [(f32, f32, Size<DevicePixels>); 4] = {
    const LARGE: Size<DevicePixels> = Size {
        width: DevicePixels(512),
        height: DevicePixels(320),
    };
    const SMALL: Size<DevicePixels> = Size {
        width: DevicePixels(360),
        height: DevicePixels(240),
    };
    [
        (16., 16., LARGE),
        (180., 120., LARGE),
        (16., 16., LARGE),
        (40., 35., SMALL),
    ]
};

/// A scene whose paths need only part of the viewport, for cropped path targets.
pub fn cropped_path_scene(x: f32, y: f32) -> Scene {
    let mask = ContentMask {
        bounds: Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(ScaledPixels(1024.), ScaledPixels(640.)),
        ),
        ..Default::default()
    };
    let triangle = |x: f32, y: f32, edge: f32| {
        let mut builder = PathBuilder::fill();
        builder.move_to(point(px(x), px(y)));
        builder.line_to(point(px(x + edge), px(y)));
        builder.line_to(point(px(x), px(y + edge)));
        builder.close();
        let mut path = builder.build().unwrap().scale(1.0);
        path.content_mask = mask;
        path.color = linear_gradient(
            90.,
            linear_color_stop(rgb(0xff0000), 0.),
            linear_color_stop(rgb(0x00ff00), 1.),
        );
        path
    };
    let black_quad = |bounds| Quad {
        bounds,
        content_mask: mask,
        background: solid_background(black()),
        ..Default::default()
    };
    let mut scene = Scene::default();
    scene.insert_primitive(black_quad(mask.bounds));
    scene.insert_primitive(triangle(x, y, 47.5));
    // A later path beyond the viewport makes the combined path sprite span empty
    // space past the cropped target, which must stay black rather than smear its edge.
    scene.insert_primitive(black_quad(Bounds::new(
        point(ScaledPixels(0.), ScaledPixels(0.)),
        size(ScaledPixels(1.), ScaledPixels(1.)),
    )));
    scene.insert_primitive(triangle(800., 10., 40.));
    // Negative geometry and a clipped curve exercise coordinates and clipping
    // independently of the viewport's dimensions.
    let mut curve = PathBuilder::stroke(px(1.6));
    curve.move_to(point(px(-10.), px(30.)));
    curve.cubic_bezier_to(
        point(px(40.), px(70.)),
        point(px(24.), px(22.)),
        point(px(5.), px(60.)),
    );
    let mut curve = curve.build().unwrap().scale(1.0);
    curve.content_mask = ContentMask {
        bounds: Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(ScaledPixels(30.), ScaledPixels(60.)),
        ),
        ..Default::default()
    };
    curve.color = solid_background(white());
    scene.insert_primitive(curve);
    scene.finish();
    scene
}

/// Asserts that the triangle `cropped_path_scene(x, y)` draws painted, and that the
/// space its path sprites span beyond the triangle stayed black.
pub fn assert_cropped_paths_painted(rgba: &[u8], width: u32, x: f32, y: f32) {
    let pixel = |x: u32, y: u32| {
        let offset = (y * width + x) as usize * 4;
        &rgba[offset..offset + 4]
    };
    let inside = pixel(x as u32 + 8, y as u32 + 8);
    assert!(inside[0] > 100, "the triangle did not paint: {inside:?}");
    let beyond = pixel(300, 200);
    assert_eq!(
        &beyond[..3],
        &[0, 0, 0],
        "the cropped target's edge smeared"
    );
}
