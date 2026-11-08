//! Cross-platform headless app context for tests that need real text shaping.
//!
//! This replaces the macOS-only `HeadlessMetalAppContext` with a platform-neutral
//! implementation backed by `TestPlatform`. Tests supply a real `PlatformTextSystem`
//! to get accurate glyph measurements while keeping everything else deterministic.
//!
//! Optionally, a renderer factory can be provided to enable real GPU rendering
//! and screenshot capture via [`HeadlessAppContext::capture_screenshot`].

use crate::{
    AnyView, AnyWindowHandle, App, AppCell, AppContext, AssetRegistry, AssetSource,
    BackgroundExecutor, Bounds, Context, Entity, EntityId, EventEmitter, ForegroundExecutor,
    Global, Hsla, Pixels, PlatformHeadlessRenderer, PlatformTextSystem, Render, Reservation,
    ScaledPixels, Size, Task, TestDispatcher, TestPlatform, TextSystem, Window, WindowBounds,
    WindowHandle, WindowOptions,
    app::{GpuiBorrow, GpuiMode},
};
use anyhow::Result;
use image::RgbaImage;
use std::{future::Future, rc::Rc, sync::Arc, time::Duration};

/// A headless app for tests that need real text layout, without a window on screen.
///
/// Like [`TestAppContext`](crate::TestAppContext), this runs on a deterministic
/// scheduler and a test platform. Unlike it, you supply the text system, so text
/// is measured the way the platform measures it. Windows are hidden and unfocused.
/// Use [`HeadlessAppContext::debug_bounds`] and the `*_bounds` methods to check
/// where things were painted, and [`HeadlessAppContext::capture_screenshot`] to
/// get pixels when a renderer is configured.
///
/// The text system comes from a platform or text crate. For example, the
/// `ParleyTextSystem` in the `gpui_ce_parley` crate measures text with real fonts.
/// This example uses [`TestTextSystem`](crate::TestTextSystem), which is for tests
/// that don't check text metrics:
///
/// ```
/// use std::sync::Arc;
///
/// use gpui::{AppContext, HeadlessAppContext, TestTextSystem};
///
/// let mut cx = HeadlessAppContext::new(Arc::new(TestTextSystem));
/// let count = cx.new(|_| 1usize);
/// assert_eq!(cx.read_entity(&count, |count, _| *count), 1);
/// ```
pub struct HeadlessAppContext {
    /// The underlying app cell.
    pub app: Rc<AppCell>,
    /// The background executor for running async tasks.
    pub background_executor: BackgroundExecutor,
    /// The foreground executor for running tasks on the main thread.
    pub foreground_executor: ForegroundExecutor,
    dispatcher: TestDispatcher,
    text_system: Arc<TextSystem>,
}

impl HeadlessAppContext {
    /// Creates a headless context that measures text with `platform_text_system`.
    ///
    /// It has no asset source and no renderer, so [`HeadlessAppContext::capture_screenshot`]
    /// returns an error. Use [`HeadlessAppContext::with_platform`] for screenshots.
    pub fn new(platform_text_system: Arc<dyn PlatformTextSystem>) -> Self {
        Self::with_platform(platform_text_system, Arc::new(()), || None)
    }

    /// Creates a headless context with `platform_text_system` and `asset_source`, and no renderer.
    pub fn with_asset_source(
        platform_text_system: Arc<dyn PlatformTextSystem>,
        asset_source: Arc<dyn AssetSource>,
    ) -> Self {
        Self::with_platform(platform_text_system, asset_source, || None)
    }

    /// Creates a headless context with a text system, an asset source and a renderer factory.
    ///
    /// `renderer_factory` creates the renderer that windows draw with. Return `None`
    /// from it to draw without a renderer, which disables screenshots. The scheduler seed
    /// is read from `SEED`, and is `0` when it's unset or isn't a number.
    pub fn with_platform(
        platform_text_system: Arc<dyn PlatformTextSystem>,
        asset_source: Arc<dyn AssetSource>,
        renderer_factory: impl Fn() -> Option<Box<dyn PlatformHeadlessRenderer>> + 'static,
    ) -> Self {
        let seed = std::env::var("SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let dispatcher = TestDispatcher::new(seed);
        let arc_dispatcher = Arc::new(dispatcher.clone());
        let background_executor = BackgroundExecutor::new(arc_dispatcher.clone());
        let foreground_executor = ForegroundExecutor::new(arc_dispatcher);

        let renderer_factory: Box<dyn Fn() -> Option<Box<dyn PlatformHeadlessRenderer>>> =
            Box::new(renderer_factory);
        let platform = TestPlatform::with_platform(
            background_executor.clone(),
            foreground_executor.clone(),
            platform_text_system.clone(),
            Some(renderer_factory),
        );

        let text_system = Arc::new(TextSystem::new(platform_text_system));
        let http_client = crate::http_client::FakeHttpClient::with_404_response();
        let app = App::new_app(
            platform,
            AssetRegistry::from(asset_source).into(),
            http_client,
        );
        app.borrow_mut().mode = GpuiMode::test();

        Self {
            app,
            background_executor,
            foreground_executor,
            dispatcher,
            text_system,
        }
    }

    /// Opens a hidden, unfocused window of `size` at the origin, with a root view from `build_root`.
    ///
    /// # Errors
    ///
    /// Returns an error if the window can't be opened.
    pub fn open_window<V: Render + 'static>(
        &mut self,
        size: Size<Pixels>,
        build_root: impl FnOnce(&mut Window, &mut App) -> Entity<V>,
    ) -> Result<WindowHandle<V>> {
        use crate::{point, px};

        let bounds = Bounds {
            origin: point(px(0.0), px(0.0)),
            size,
        };

        let mut cx = self.app.borrow_mut();
        cx.open_window(
            WindowOptions::new()
                .window_bounds(Some(WindowBounds::Windowed(bounds)))
                .focus(false)
                .show(false),
            build_root,
        )
    }

    /// Runs pending tasks until none can make progress. This doesn't move the simulated clock.
    pub fn run_until_parked(&self) {
        self.dispatcher.run_until_parked();
    }

    /// Moves the simulated clock forward by `duration`, and runs the tasks that become ready.
    pub fn advance_clock(&self, duration: Duration) {
        self.dispatcher.advance_clock(duration);
    }

    /// Lets blocking waits in the scheduler sleep in real time, for tests that wait on real I/O.
    ///
    /// Pair it with [`HeadlessAppContext::forbid_parking`]. See
    /// [`BackgroundExecutor::allow_parking`](crate::BackgroundExecutor::allow_parking).
    pub fn allow_parking(&self) {
        self.dispatcher.allow_parking();
    }

    /// Turns off the parking mode from [`HeadlessAppContext::allow_parking`], returning to deterministic execution.
    pub fn forbid_parking(&self) {
        self.dispatcher.forbid_parking();
    }

    /// Runs `f` with mutable access to the [`App`]. This doesn't run pending tasks.
    pub fn update<R>(&mut self, f: impl FnOnce(&mut App) -> R) -> R {
        let mut app = self.app.borrow_mut();
        f(&mut app)
    }

    /// Runs `f` with the window's root view, the [`Window`] and the [`App`].
    ///
    /// This doesn't paint the window. Methods that read painted output, such as
    /// [`HeadlessAppContext::debug_bounds`], see the last frame that was painted.
    ///
    /// # Errors
    ///
    /// Returns an error if `window` has been closed.
    pub fn update_window<R>(
        &mut self,
        window: AnyWindowHandle,
        f: impl FnOnce(AnyView, &mut Window, &mut App) -> R,
    ) -> Result<R> {
        let mut app = self.app.borrow_mut();
        app.update_window(window, f)
    }

    /// Returns the bounds of the element with `debug_selector(selector)` in the last painted frame, or `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if `window` has been closed.
    pub fn debug_bounds(
        &mut self,
        window: AnyWindowHandle,
        selector: &str,
    ) -> Result<Option<Bounds<Pixels>>> {
        self.update_window(window, |_, window, _| {
            window.rendered_frame.debug_bounds.get(selector).copied()
        })
    }

    /// Returns the device-pixel bounds of each solid quad in `color` in the last painted frame.
    ///
    /// The color must match exactly. Quads with other colors are skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if `window` has been closed.
    pub fn solid_quad_bounds(
        &mut self,
        window: AnyWindowHandle,
        color: Hsla,
    ) -> Result<Vec<Bounds<ScaledPixels>>> {
        let color = color.into();
        self.update_window(window, |_, window, _| {
            window
                .rendered_frame
                .scene
                .quads
                .iter()
                .filter(|quad| quad.background.solid == color)
                .map(|quad| quad.bounds)
                .collect()
        })
    }

    /// Returns the device-pixel bounds of each monochrome glyph painted in `color` in the last painted frame.
    ///
    /// The color must match exactly.
    ///
    /// # Errors
    ///
    /// Returns an error if `window` has been closed.
    pub fn glyph_bounds(
        &mut self,
        window: AnyWindowHandle,
        color: Hsla,
    ) -> Result<Vec<Bounds<ScaledPixels>>> {
        let color = color.into();
        self.update_window(window, |_, window, _| {
            window
                .rendered_frame
                .scene
                .monochrome_sprites
                .iter()
                .filter(|sprite| sprite.color == color)
                .map(|sprite| sprite.bounds)
                .collect()
        })
    }

    /// Returns the device-pixel bounds of each underline painted in `color` in the last painted frame.
    ///
    /// The color must match exactly.
    ///
    /// # Errors
    ///
    /// Returns an error if `window` has been closed.
    pub fn underline_bounds(
        &mut self,
        window: AnyWindowHandle,
        color: Hsla,
    ) -> Result<Vec<Bounds<ScaledPixels>>> {
        let color = color.into();
        self.update_window(window, |_, window, _| {
            window
                .rendered_frame
                .scene
                .underlines
                .iter()
                .filter(|underline| underline.color == color)
                .map(|underline| underline.bounds)
                .collect()
        })
    }

    /// Renders `window`'s last painted frame to an image.
    ///
    /// # Errors
    ///
    /// Returns an error if the context was created without a renderer, which is the
    /// case for [`HeadlessAppContext::new`] and [`HeadlessAppContext::with_asset_source`].
    /// Also returns an error if the window has been closed.
    pub fn capture_screenshot(&mut self, window: AnyWindowHandle) -> Result<RgbaImage> {
        let mut app = self.app.borrow_mut();
        app.update_window(window, |_, window, cx| {
            if let Some(arena_clear_needed) = window.redraw_if_rendered_frame_atlas_is_stale(cx) {
                let image = window.render_to_image();
                arena_clear_needed.clear(cx);
                image
            } else {
                window.render_to_image()
            }
        })?
    }

    /// Returns the text system.
    pub fn text_system(&self) -> &Arc<TextSystem> {
        &self.text_system
    }

    /// Returns the background executor.
    pub fn background_executor(&self) -> &BackgroundExecutor {
        &self.background_executor
    }

    /// Returns the foreground executor.
    pub fn foreground_executor(&self) -> &ForegroundExecutor {
        &self.foreground_executor
    }
}

impl Drop for HeadlessAppContext {
    fn drop(&mut self) {
        // Shut down the app so windows are closed and entity handles are
        // released before the LeakDetector runs.
        self.app.borrow_mut().shutdown();
    }
}

impl AppContext for HeadlessAppContext {
    fn new<T: 'static>(&mut self, build_entity: impl FnOnce(&mut Context<T>) -> T) -> Entity<T> {
        let mut app = self.app.borrow_mut();
        app.new(build_entity)
    }

    fn reserve_entity<T: 'static>(&mut self) -> Reservation<T> {
        let mut app = self.app.borrow_mut();
        app.reserve_entity()
    }

    fn insert_entity<T: 'static>(
        &mut self,
        reservation: Reservation<T>,
        build_entity: impl FnOnce(&mut Context<T>) -> T,
    ) -> Entity<T> {
        let mut app = self.app.borrow_mut();
        app.insert_entity(reservation, build_entity)
    }

    fn update_entity<T: 'static, R>(
        &mut self,
        handle: &Entity<T>,
        update: impl FnOnce(&mut T, &mut Context<T>) -> R,
    ) -> R {
        let mut app = self.app.borrow_mut();
        app.update_entity(handle, update)
    }

    fn as_mut<'a, T>(&'a mut self, _: &Entity<T>) -> GpuiBorrow<'a, T>
    where
        T: 'static,
    {
        panic!("Cannot use as_mut with HeadlessAppContext. Call update() instead.")
    }

    fn read_entity<T, R>(&self, handle: &Entity<T>, read: impl FnOnce(&T, &App) -> R) -> R
    where
        T: 'static,
    {
        let app = self.app.borrow();
        app.read_entity(handle, read)
    }

    fn notify(&mut self, entity_id: EntityId) {
        let mut app = self.app.borrow_mut();
        app.notify(entity_id);
    }

    fn emit<EntityType, EventType>(&mut self, entity: &Entity<EntityType>, event: EventType)
    where
        EntityType: EventEmitter<EventType>,
        EventType: 'static,
    {
        let mut app = self.app.borrow_mut();
        app.emit(entity, event)
    }

    fn update_window<T, F>(&mut self, window: AnyWindowHandle, f: F) -> Result<T>
    where
        F: FnOnce(AnyView, &mut Window, &mut App) -> T,
    {
        let mut lock = self.app.borrow_mut();
        lock.update_window(window, f)
    }

    fn with_window<R>(
        &mut self,
        entity_id: EntityId,
        f: impl FnOnce(&mut Window, &mut App) -> R,
    ) -> Option<R> {
        let mut lock = self.app.borrow_mut();
        lock.with_window(entity_id, f)
    }

    fn read_window<T, R>(
        &self,
        window: &WindowHandle<T>,
        read: impl FnOnce(Entity<T>, &App) -> R,
    ) -> Result<R>
    where
        T: 'static,
    {
        let app = self.app.borrow();
        app.read_window(window, read)
    }

    fn background_spawn<R>(&self, future: impl Future<Output = R> + Send + 'static) -> Task<R>
    where
        R: Send + 'static,
    {
        self.background_executor.spawn(future)
    }

    fn read_global<G, R>(&self, callback: impl FnOnce(&G, &App) -> R) -> R
    where
        G: Global,
    {
        let app = self.app.borrow();
        app.read_global(callback)
    }

    fn insert_global_entity<T: 'static>(&mut self, entity: Entity<T>) {
        let mut app = self.app.borrow_mut();
        app.insert_global_entity(entity)
    }

    fn remove_global_entity<T: 'static>(&mut self, entity: &Entity<T>) {
        let mut app = self.app.borrow_mut();
        app.remove_global_entity(entity)
    }

    fn global_entities<T: 'static>(&self) -> impl Iterator<Item = Entity<T>> {
        let app = self.app.borrow();
        let iter = app.global_entities();
        // Cloning the iterator here so that lock can be released when function is going out of scope
        iter.collect::<Vec<_>>().into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AnyView, AtlasKey, AtlasTextureId, AtlasTile, Bounds, DevicePixels, GlyphAtlasCache,
        GlyphAtlasEntry, IntoElement, ParentElement as _, PlatformAtlas, PrimitiveBatch,
        RenderGlyphParams, RenderImage, Scene, StyleRefinement, Styled as _, TestTextSystem,
        TileId, ValidatedRasterizedGlyph, div, img, px, size,
    };
    use anyhow::Result;
    use image::{Frame as ImageFrame, ImageBuffer, Rgba, RgbaImage};
    use parking_lot::Mutex;
    use smallvec::SmallVec;
    use std::{
        borrow::Cow,
        collections::{HashMap, HashSet},
    };

    #[derive(Default)]
    struct ResettableAtlas {
        state: Mutex<ResettableAtlasState>,
    }

    #[derive(Default)]
    struct ResettableAtlasState {
        tiles_by_key: HashMap<AtlasKey, AtlasTile>,
        glyph_cache: GlyphAtlasCache,
        live_textures: HashSet<AtlasTextureId>,
        next_texture_index: u32,
        next_tile_id: u32,
        generation: u64,
    }

    impl ResettableAtlasState {
        fn insert_tile(&mut self, key: AtlasKey, size: Size<DevicePixels>) -> AtlasTile {
            let texture_id = AtlasTextureId {
                index: self.next_texture_index,
                kind: key.texture_kind(),
            };
            self.next_texture_index += 1;

            let tile = AtlasTile {
                texture_id,
                tile_id: TileId(self.next_tile_id),
                padding: 0,
                bounds: Bounds {
                    origin: Default::default(),
                    size,
                },
            };
            self.next_tile_id += 1;
            self.live_textures.insert(texture_id);
            self.tiles_by_key.insert(key, tile);
            tile
        }
    }

    impl ResettableAtlas {
        fn reset_device_resources_for_test(&self) {
            let mut state = self.state.lock();
            state.tiles_by_key.clear();
            state.glyph_cache.clear();
            state.live_textures.clear();
            state.next_texture_index = 0;
            state.next_tile_id = 0;
            state.generation = state.generation.wrapping_add(1);
        }

        fn has_texture(&self, id: AtlasTextureId) -> bool {
            self.state.lock().live_textures.contains(&id)
        }
    }

    impl PlatformAtlas for ResettableAtlas {
        fn get_or_insert_with<'a>(
            &self,
            key: &AtlasKey,
            build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
        ) -> Result<Option<AtlasTile>> {
            if let Some(tile) = self.state.lock().tiles_by_key.get(key).cloned() {
                return Ok(Some(tile));
            }

            let Some((size, _bytes)) = build()? else {
                return Ok(None);
            };

            let mut state = self.state.lock();
            if let Some(tile) = state.tiles_by_key.get(key).cloned() {
                return Ok(Some(tile));
            }

            Ok(Some(state.insert_tile(key.clone(), size)))
        }

        fn get_or_insert_glyph_with(
            &self,
            params: &RenderGlyphParams,
            build: &mut dyn FnMut() -> Result<ValidatedRasterizedGlyph>,
        ) -> Result<GlyphAtlasEntry> {
            if let Some(entry) = self.state.lock().glyph_cache.get(params) {
                return Ok(entry);
            }

            let glyph = build()?;

            let mut state = self.state.lock();
            let tile = if glyph.size == Size::default() {
                None
            } else {
                let key = (params.clone(), glyph.format).into();
                Some(state.insert_tile(key, glyph.size))
            };

            Ok(state.glyph_cache.insert(params, &glyph, tile))
        }

        fn remove(&self, key: &AtlasKey) {
            let mut state = self.state.lock();
            state.glyph_cache.remove(key);
            if let Some(tile) = state.tiles_by_key.remove(key) {
                state.live_textures.remove(&tile.texture_id);
                state.generation = state.generation.wrapping_add(1);
            }
        }

        fn generation(&self) -> u64 {
            self.state.lock().generation
        }
    }

    struct CheckingHeadlessRenderer {
        atlas: Arc<ResettableAtlas>,
    }

    impl PlatformHeadlessRenderer for CheckingHeadlessRenderer {
        fn render_scene_to_image(
            &mut self,
            scene: &Scene,
            size: Size<DevicePixels>,
        ) -> Result<RgbaImage> {
            for batch in scene.batches() {
                let texture_id = match batch {
                    PrimitiveBatch::MonochromeSprites { texture_id, .. }
                    | PrimitiveBatch::SubpixelSprites { texture_id, .. }
                    | PrimitiveBatch::PolychromeSprites { texture_id, .. } => texture_id,
                    _ => continue,
                };

                assert!(
                    self.atlas.has_texture(texture_id),
                    "stale atlas texture id after renderer reset: {texture_id:?}",
                );
            }

            Ok(RgbaImage::from_pixel(
                size.width.0.max(1) as u32,
                size.height.0.max(1) as u32,
                Rgba([0, 0, 0, 0]),
            ))
        }

        fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
            self.render_scene_to_image(scene, size).map(|_| ())
        }

        fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
            self.atlas.clone()
        }
    }

    struct ImageRoot {
        image: Arc<RenderImage>,
    }

    struct CachedImageRoot {
        child: Entity<ImageRoot>,
    }

    impl Render for CachedImageRoot {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            AnyView::from(self.child.clone()).cached(StyleRefinement::default().size(px(1.0)))
        }
    }

    impl Render for ImageRoot {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div().child(img(self.image.clone()).size(px(1.0)))
        }
    }

    fn test_image() -> Arc<RenderImage> {
        let frame = ImageFrame::new(ImageBuffer::from_pixel(1, 1, Rgba([0, 0, 0, 255])));
        Arc::new(RenderImage::new(SmallVec::from_iter([frame])))
    }

    #[test]
    fn non_forced_present_after_atlas_reset_can_redraw_stale_scene() -> Result<()> {
        let atlas = Arc::new(ResettableAtlas::default());
        let renderer_atlas = atlas.clone();
        let mut cx =
            HeadlessAppContext::with_platform(Arc::new(TestTextSystem), Arc::new(()), move || {
                Some(Box::new(CheckingHeadlessRenderer {
                    atlas: renderer_atlas.clone(),
                }))
            });

        let image = test_image();
        let window = cx.open_window(size(px(10.0), px(10.0)), move |_window, cx| {
            cx.new(|_| ImageRoot { image })
        })?;

        cx.capture_screenshot(window.into())?;
        atlas.reset_device_resources_for_test();

        // This models a non-forced Windows paint/present after DirectX device
        // recovery cleared the atlas, but before the forced full redraw happens.
        cx.capture_screenshot(window.into())?;
        Ok(())
    }

    #[test]
    fn non_forced_redraw_after_atlas_reset_must_not_reuse_cached_sprite_scene() -> Result<()> {
        let atlas = Arc::new(ResettableAtlas::default());
        let renderer_atlas = atlas.clone();
        let mut cx =
            HeadlessAppContext::with_platform(Arc::new(TestTextSystem), Arc::new(()), move || {
                Some(Box::new(CheckingHeadlessRenderer {
                    atlas: renderer_atlas.clone(),
                }))
            });

        let image = test_image();
        let window = cx.open_window(size(px(10.0), px(10.0)), move |_window, cx| {
            let child = cx.new(|_| ImageRoot { image });
            cx.new(|_| CachedImageRoot { child })
        })?;

        cx.update_window(window.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
            window.render_to_image()
        })??;

        atlas.reset_device_resources_for_test();

        cx.update_window(window.into(), |_, window, cx| {
            window.invalidator.set_dirty(true);
            let arena_clear_needed = window.draw(cx);
            let image = window.render_to_image();
            arena_clear_needed.clear(cx);
            image
        })??;

        Ok(())
    }
}
