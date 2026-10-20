use super::WgpuRenderer;
use crate::WgpuContext;
use gpui::{
    BackdropFilter, Bounds, ContentMask, Corners, DevicePixels, FilterBoundary, Quad, ScaledFilter,
    ScaledPixels, Scene, point, size, solid_background,
};

fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds::new(
        point(ScaledPixels(x), ScaledPixels(y)),
        size(ScaledPixels(width), ScaledPixels(height)),
    )
}

fn ordinary_scene() -> Scene {
    let mut scene = Scene::default();
    let bounds = bounds(0.0, 0.0, 96.0, 96.0);
    scene.insert_primitive(Quad {
        bounds,
        content_mask: ContentMask {
            bounds,
            ..Default::default()
        },
        background: solid_background(gpui::white()),
        ..Default::default()
    });
    scene.finish();
    scene
}

/// Backdrop blur, isolated blur, and an opacity group with no blur.
pub(super) fn filter_scene(kind: usize) -> Scene {
    let mut scene = Scene::default();
    let bounds = bounds(20.0, 20.0, 56.0, 56.0);
    let content_mask = ContentMask {
        bounds: self::bounds(0.0, 0.0, 96.0, 96.0),
        ..Default::default()
    };
    let corner_radii = Corners::all(ScaledPixels(12.0));
    if kind == 0 {
        scene.insert_primitive(Quad {
            bounds,
            content_mask,
            background: gpui::checkerboard(gpui::white(), 2.0),
            ..Default::default()
        });
        scene.raise_order_floor();
        scene.insert_primitive(BackdropFilter {
            order: 0,
            bounds,
            content_mask,
            corner_radii,
            corner_smoothing: 0.0,
            filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(5.0))],
            opacity: 1.0,
        });
    } else {
        let boundary = FilterBoundary {
            order: 0,
            bounds,
            content_mask,
            corner_radii,
            corner_smoothing: 0.0,
            filters: if kind == 1 {
                smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(5.0))]
            } else {
                smallvec::smallvec![]
            },
            opacity: 0.5,
            is_start: true,
        };
        scene.insert_primitive(boundary.clone());
        scene.raise_order_floor();
        scene.insert_primitive(Quad {
            bounds,
            content_mask,
            background: solid_background(gpui::white()),
            ..Default::default()
        });
        scene.raise_order_floor();
        scene.insert_primitive(FilterBoundary {
            is_start: false,
            ..boundary
        });
    }
    scene.finish();
    scene
}

fn handles(renderer: &WgpuRenderer) -> Option<[wgpu::RenderPipeline; 3]> {
    renderer
        .resources()
        .pipelines
        .filters
        .as_ref()
        .map(|filters| {
            [&filters.downsample, &filters.blur, &filters.composite]
                .map(|pipeline| std::ops::Deref::deref(pipeline).clone())
        })
}

#[test]
fn lazy_filter_pipelines_skip_ordinary_frames_and_match_first_use_pixels() -> anyhow::Result<()> {
    let _gpu_test_guard = crate::test_gpu::guard();
    let context = WgpuContext::new_headless(None)?;
    let target = size(DevicePixels(96), DevicePixels(96));
    let mut lazy = WgpuRenderer::new_headless(&context, target)?;
    let mut prepared = WgpuRenderer::new_headless(&context, target)?;
    lazy.retained.disable_for_tests();
    prepared.retained.disable_for_tests();
    assert!(handles(&lazy).is_none());
    for _ in 0..2 {
        assert_eq!(
            lazy.render_to_image(&ordinary_scene())?.get_pixel(48, 48).0,
            [255; 4]
        );
        assert!(handles(&lazy).is_none());
    }
    prepared
        .resources_mut()
        .pipelines
        .prepare_filters(&context.device)?;
    let mut previous = None;
    for kind in 0..3 {
        lazy.rebuild_pipelines();
        assert!(handles(&lazy).is_none(), "rebuild resets lazy filters");
        let scene = filter_scene(kind);
        let image = lazy.render_to_image(&scene)?;
        assert_eq!(image, prepared.render_to_image(&scene)?, "kind {kind}");
        assert!(image.pixels().any(|pixel| pixel.0[..3] != [0; 3]));
        if kind == 2 {
            let pixel = image.get_pixel(48, 48).0;
            assert!(
                pixel[..3]
                    .iter()
                    .all(|&channel| (126..=129).contains(&channel)),
                "zero-blur opacity group must composite half-opacity white: {pixel:?}"
            );
            assert_eq!(pixel[3], 255);
        }
        let created = handles(&lazy).expect("filter group prepared on first use");
        if let Some(previous) = previous {
            assert_ne!(created, previous, "rebuild replaces the old group");
        }
        assert_eq!(lazy.render_to_image(&scene)?, image);
        assert_eq!(handles(&lazy), Some(created.clone()), "reuse the group");
        previous = Some(created);
    }
    let fresh_context = WgpuContext::new_headless(None)?;
    let mut recreated = WgpuRenderer::new_headless(&fresh_context, target)?;
    recreated.retained.disable_for_tests();
    assert!(handles(&recreated).is_none());
    assert_eq!(
        recreated.render_to_image(&filter_scene(2))?,
        prepared.render_to_image(&filter_scene(2))?
    );
    assert_ne!(
        handles(&recreated),
        previous,
        "new device needs its own group"
    );
    assert!(context.uncaptured_error_slot().lock().unwrap().is_none());
    assert!(
        fresh_context
            .uncaptured_error_slot()
            .lock()
            .unwrap()
            .is_none()
    );
    Ok(())
}

#[cfg(target_os = "windows")]
#[test]
fn lazy_filter_pipelines_preserve_retained_copy_blit_and_reuse() -> anyhow::Result<()> {
    let _gpu_test_guard = crate::test_gpu::guard();
    let context = WgpuContext::new_headless(None)?;
    assert_eq!(context.adapter.get_info().backend, wgpu::Backend::Dx12);
    let mut renderer =
        WgpuRenderer::new_headless(&context, size(DevicePixels(96), DevicePixels(96)))?;
    renderer.retained.enable_for_tests();
    let scene = ordinary_scene();
    let _frame = gpui::profiler::gpu::set_gpu_frame_context(true, 1);
    let image = renderer.render_to_image(&scene)?;
    assert_eq!(image.get_pixel(48, 48).0, [255; 4]);
    assert_eq!(renderer.last_retained_presentation, "copy");
    assert!(
        handles(&renderer).is_none(),
        "texture copies need no shaders"
    );

    renderer
        .target
        .update_surface_copy_support(wgpu::TextureUsages::RENDER_ATTACHMENT, &[]);
    assert_eq!(renderer.render_to_image(&scene)?, image);
    assert_eq!(renderer.last_retained_presentation, "blit");
    let first_blit = handles(&renderer).expect("retained fallback needs filter shaders");
    assert_eq!(renderer.render_to_image(&scene)?, image);
    assert_eq!(renderer.retained.last_damage(), "reuse");
    assert_eq!(handles(&renderer), Some(first_blit.clone()));

    // Reuse has no scene requirements. The actual retained presentation must
    // still recreate its shader group after a pipeline collection is rebuilt.
    renderer.rebuild_pipelines();
    assert!(handles(&renderer).is_none());
    assert_eq!(renderer.render_to_image(&scene)?, image);
    assert_eq!(renderer.retained.last_damage(), "reuse");
    assert_eq!(renderer.last_retained_presentation, "blit");
    assert_ne!(handles(&renderer).unwrap(), first_blit);
    assert!(context.uncaptured_error_slot().lock().unwrap().is_none());
    Ok(())
}
