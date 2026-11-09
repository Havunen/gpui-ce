// Backport of zed-industries/zed#65472 (Apache-2.0).
// Original implementation: Steve Shreeve and Anthony Eid.
// Commits: fa7daa2 and 0d1c7a2 (merged as 0373366f43e3).
// Adapted for gpui-ce's metal 0.33 retained object wrappers.

use foreign_types::ForeignTypeRef;
use objc2::{
    AnyThread, DefinedClass, MainThreadMarker, MainThreadOnly, Message, define_class, msg_send,
    rc::Retained,
    runtime::{AnyClass, ProtocolObject},
};
use objc2_foundation::{NSObject, NSObjectProtocol, NSRunLoop, NSRunLoopCommonModes};
use objc2_quartz_core::{
    CAMetalDisplayLink, CAMetalDisplayLinkDelegate, CAMetalDisplayLinkUpdate, CAMetalLayer,
};

fn retain_drawable(
    drawable: &ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>,
) -> metal::MetalDrawable {
    // SAFETY: CAMetalDisplayLinkUpdate::drawable returns a CAMetalDrawable.
    // Both wrappers name that same Objective-C protocol. Borrowing first and
    // then cloning takes our own retain; it does not consume objc2's retain.
    unsafe {
        metal::MetalDrawableRef::from_ptr(
            drawable as *const ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>
                as *mut metal::CAMetalDrawable,
        )
    }
    .to_owned()
}

fn layer_object(layer: &metal::MetalLayerRef) -> &CAMetalLayer {
    // SAFETY: Both wrappers name CAMetalLayer. The returned borrow has the
    // lifetime of metal's retained layer and transfers no ownership.
    unsafe { &*(layer.as_ptr() as *const CAMetalLayer) }
}

struct MetalDisplayLinkDelegateIvars {
    callback: Box<dyn Fn(metal::MetalDrawable)>,
}

define_class!(
    // SAFETY: `NSObject` has no subclassing requirements.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = MetalDisplayLinkDelegateIvars]
    struct MetalDisplayLinkDelegate;

    unsafe impl NSObjectProtocol for MetalDisplayLinkDelegate {}

    unsafe impl CAMetalDisplayLinkDelegate for MetalDisplayLinkDelegate {
        #[unsafe(method(metalDisplayLink:needsUpdate:))]
        fn needs_update(&self, _link: &CAMetalDisplayLink, update: &CAMetalDisplayLinkUpdate) {
            // The callback may destroy its frame source, releasing our owner's retain.
            let delegate = self.retain();
            let drawable = retain_drawable(&update.drawable());
            (delegate.ivars().callback)(drawable);
        }
    }
);

impl MetalDisplayLinkDelegate {
    fn new(
        main_thread: MainThreadMarker,
        callback: impl Fn(metal::MetalDrawable) + 'static,
    ) -> Retained<Self> {
        let this = Self::alloc(main_thread).set_ivars(MetalDisplayLinkDelegateIvars {
            callback: Box::new(callback),
        });
        // SAFETY: `NSObject`'s `init` is its designated initializer.
        unsafe { msg_send![super(this), init] }
    }
}

/// A main-run-loop source of Metal drawables on macOS 14+ and iOS/iPadOS 17+.
///
/// Callbacks run on the run loop the link is registered with:
/// <https://developer.apple.com/documentation/quartzcore/cametaldisplaylink>.
/// This wrapper uses the main run loop because GPUI renders on the foreground thread.
pub struct MetalFrameSource {
    link: Retained<CAMetalDisplayLink>,
    // The link holds its delegate weakly.
    _delegate: Retained<MetalDisplayLinkDelegate>,
    _main_thread: MainThreadMarker,
}

impl MetalFrameSource {
    /// Whether this system has `CAMetalDisplayLink` (macOS 14+, iOS/iPadOS 17+).
    pub fn is_available() -> bool {
        AnyClass::get(c"CAMetalDisplayLink").is_some()
    }

    /// Starts delivering updates for `layer`, or returns `None` where
    /// `CAMetalDisplayLink` is unavailable.
    pub fn new(
        layer: &metal::MetalLayerRef,
        main_thread: MainThreadMarker,
        callback: impl Fn(metal::MetalDrawable) + 'static,
    ) -> Option<Self> {
        if !Self::is_available() {
            return None;
        }
        let delegate = MetalDisplayLinkDelegate::new(main_thread, callback);
        let layer = layer_object(layer);
        let link = CAMetalDisplayLink::initWithMetalLayer(CAMetalDisplayLink::alloc(), layer);
        link.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
        // SAFETY: the retained marker confines the source to the main thread,
        // and the main run loop outlives the source.
        unsafe { link.addToRunLoop_forMode(&NSRunLoop::mainRunLoop(), NSRunLoopCommonModes) };
        Some(Self {
            link,
            _delegate: delegate,
            _main_thread: main_thread,
        })
    }

    /// A paused link delivers no updates and acquires no drawables.
    pub fn set_paused(&self, paused: bool) {
        self.link.setPaused(paused);
    }
}

impl Drop for MetalFrameSource {
    fn drop(&mut self) {
        self.link.invalidate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use foreign_types::ForeignType;
    use gpui::{DevicePixels, size};

    #[test]
    fn bridged_drawable_outlives_objc2_owner_and_autorelease_pool() {
        let drawable = objc2::rc::autoreleasepool(|_| {
            let mut renderer = crate::metal_renderer::MetalRenderer::new(Default::default(), false);
            renderer.update_drawable_size(size(DevicePixels(64), DevicePixels(48)));
            let layer = layer_object(renderer.layer().expect("window renderer has a layer"));
            let original = layer
                .nextDrawable()
                .expect("window layer supplies a drawable");
            let retained = retain_drawable(&original);
            assert_eq!(
                retained.as_ptr().cast_const().cast(),
                Retained::as_ptr(&original)
            );
            drop(original);
            retained
        });
        assert_eq!(drawable.texture().width(), 64);
        assert_eq!(drawable.texture().height(), 48);
    }

    #[test]
    fn detached_link_restores_normal_layer_drawable_acquisition() {
        if !MetalFrameSource::is_available() {
            return;
        }
        let mut renderer = crate::metal_renderer::MetalRenderer::new(Default::default(), false);
        renderer.update_drawable_size(size(DevicePixels(64), DevicePixels(64)));
        let layer = layer_object(renderer.layer().expect("window renderer has a layer"));
        // Keep this link off the run loop. Only its layer attachment and
        // invalidation contract are under test here.
        let link = CAMetalDisplayLink::initWithMetalLayer(CAMetalDisplayLink::alloc(), layer);
        renderer.set_display_link_attached(true);
        let mut scene = gpui::Scene::default();
        scene.finish();
        renderer.draw(&scene);
        link.invalidate();
        renderer.set_display_link_attached(false);
        let drawable = renderer
            .layer()
            .unwrap()
            .next_drawable()
            .expect("invalidation restores native drawable acquisition")
            .to_owned();
        assert_eq!(drawable.texture().width(), 64);
        assert_eq!(drawable.texture().height(), 64);
        drop(drawable);
        renderer.draw(&scene);
    }
}
