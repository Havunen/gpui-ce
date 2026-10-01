//! Xdnd source. Selection ownership lives until XdndFinished, including INCR
//! reads; the receiving application alone decides whether a drop succeeded.
use gpui::{FileTransfer, FileTransferCompletion, FileTransferOperation};
use std::time::{Duration, Instant};
use x11rb::{
    CURRENT_TIME, NONE,
    connection::Connection,
    protocol::{Event, xproto::*},
    rust_connection::RustConnection,
    wrapper::ConnectionExt as _,
};

x11rb::atom_manager! {
    Atoms: AtomCookies { XdndAware, XdndProxy, XdndEnter, XdndLeave, XdndPosition, XdndStatus, XdndDrop, XdndFinished, XdndSelection, XdndActionCopy, XdndActionMove, TARGETS, INCR, URI: b"text/uri-list", }
}

/// The newest protocol version this source speaks.
const VERSION: u32 = 5;
/// After XdndDrop, and again after each selection request, the target gets
/// this long to send XdndFinished before the drop counts as lost.
const FINISHED_TIMEOUT: Duration = Duration::from_secs(30);
/// No drag holds the pointer grab longer than this.
const SESSION_TIMEOUT: Duration = Duration::from_secs(600);
/// A release without an accepting XdndStatus abandons the drop after this.
const STATUS_GRACE: Duration = Duration::from_millis(250);
const POLL_INTERVAL: Duration = Duration::from_millis(8);
const INCR_CHUNK: usize = 64 * 1024;

struct Increment {
    requestor: u32,
    property: u32,
    offset: usize,
}

pub(super) fn run(files: FileTransfer) -> FileTransferCompletion {
    run_on(files, None, FINISHED_TIMEOUT)
}

fn run_on(
    files: FileTransfer,
    display: Option<&str>,
    finished_timeout: Duration,
) -> FileTransferCompletion {
    let result =
        Session::start(display).and_then(|mut session| session.run(&files, finished_timeout));
    if let Err(error) = &result {
        log::warn!("Native X11 drag stopped: {error}");
    }
    FileTransferCompletion {
        files,
        operation: result.ok().flatten(),
        source_removed: false,
    }
}

fn send(connection: &RustConnection, target: u32, kind: u32, data: [u32; 5]) -> anyhow::Result<()> {
    connection
        .send_event(
            false,
            target,
            EventMask::NO_EVENT,
            ClientMessageEvent::new(32, target, kind, data),
        )?
        .check()?;
    connection.flush()?;
    Ok(())
}

/// The operation a target performed, from the data of its XdndFinished
/// message. Before version 5 the message carries no result, so a drop the
/// target accepted counts as performed with the action it accepted.
fn finished_operation(
    version: u32,
    data: [u32; 5],
    accepted_action: u32,
    requested: FileTransferOperation,
    copy: u32,
    move_: u32,
) -> Option<FileTransferOperation> {
    let (accepted, action) = if version >= 5 {
        (data[1] & 1 != 0, data[2])
    } else {
        (true, accepted_action)
    };
    if !accepted {
        None
    } else if action == copy {
        Some(FileTransferOperation::Copy)
    } else if action == move_ && requested == FileTransferOperation::Move {
        Some(FileTransferOperation::Move)
    } else {
        None
    }
}

/// One drag: a selection-owning source window with the pointer grabbed.
struct Session {
    connection: RustConnection,
    atoms: Atoms,
    root: u32,
    source: u32,
    /// The aware window under the pointer, and the protocol version shared with it.
    target: u32,
    target_version: u32,
    /// The action the target accepted in its latest XdndStatus.
    accepted: Option<u32>,
    dropped: bool,
}

impl Session {
    fn start(display: Option<&str>) -> anyhow::Result<Self> {
        let (connection, screen) = x11rb::connect(display)?;
        let root = connection.setup().roots[screen].root;
        let atoms = Atoms::new(&connection)?.reply()?;
        let source = connection.generate_id()?;
        connection
            .create_window(
                0,
                source,
                root,
                -100,
                -100,
                1,
                1,
                0,
                WindowClass::INPUT_ONLY,
                0,
                &CreateWindowAux::new()
                    .override_redirect(1)
                    .event_mask(EventMask::PROPERTY_CHANGE),
            )?
            .check()?;
        connection.map_window(source)?.check()?;
        connection
            .set_selection_owner(source, atoms.XdndSelection, CURRENT_TIME)?
            .check()?;
        let grab = connection
            .grab_pointer(
                false,
                source,
                EventMask::POINTER_MOTION | EventMask::BUTTON_RELEASE | EventMask::BUTTON_PRESS,
                GrabMode::ASYNC,
                GrabMode::ASYNC,
                NONE,
                NONE,
                CURRENT_TIME,
            )?
            .reply()?;
        if grab.status != GrabStatus::SUCCESS {
            anyhow::bail!("Pointer grab was rejected");
        }
        let _ = connection
            .grab_keyboard(
                false,
                source,
                CURRENT_TIME,
                GrabMode::ASYNC,
                GrabMode::ASYNC,
            )?
            .reply()?;
        connection.flush()?;
        Ok(Self {
            connection,
            atoms,
            root,
            source,
            target: NONE,
            target_version: VERSION,
            accepted: None,
            dropped: false,
        })
    }

    fn run(
        &mut self,
        files: &FileTransfer,
        finished_timeout: Duration,
    ) -> anyhow::Result<Option<FileTransferOperation>> {
        let result = self.drive(files, finished_timeout);
        // A target that was entered but not dropped onto must learn the drag
        // is over, however the session ended, or it keeps its drop state.
        if !self.dropped && self.target != NONE {
            let _ = self.send(self.target, self.atoms.XdndLeave);
        }
        result
    }

    fn send(&self, target: u32, kind: u32) -> anyhow::Result<()> {
        send(&self.connection, target, kind, [self.source, 0, 0, 0, 0])
    }

    fn drive(
        &mut self,
        files: &FileTransfer,
        finished_timeout: Duration,
    ) -> anyhow::Result<Option<FileTransferOperation>> {
        let bytes = files.uri_list();
        anyhow::ensure!(!bytes.is_empty(), "No local file URLs");
        let mut requested_action = NONE;
        let mut time = CURRENT_TIME;
        let mut released = None;
        let mut increments: Vec<Increment> = Vec::new();
        let mut last_position = None;
        let session_deadline = Instant::now() + SESSION_TIMEOUT;
        let mut finished_deadline = None;
        loop {
            let now = Instant::now();
            if now > session_deadline || finished_deadline.is_some_and(|deadline| now > deadline) {
                return Ok(None);
            }
            if !self.dropped {
                let pointer = self.connection.query_pointer(self.root)?.reply()?;
                if !pointer.mask.contains(KeyButMask::BUTTON1) {
                    if self.target == NONE {
                        return Ok(None);
                    }
                    let released_at = *released.get_or_insert(now);
                    if self.accepted.is_some() {
                        send(
                            &self.connection,
                            self.target,
                            self.atoms.XdndDrop,
                            [self.source, 0, time, 0, 0],
                        )?;
                        self.connection.ungrab_pointer(CURRENT_TIME)?;
                        self.connection.ungrab_keyboard(CURRENT_TIME)?;
                        self.connection.flush()?;
                        self.dropped = true;
                        finished_deadline = Some(now + finished_timeout);
                    } else if released_at.elapsed() > STATUS_GRACE {
                        return Ok(None);
                    }
                } else {
                    let (next_target, version) = self.target_under_pointer()?;
                    if next_target != self.target {
                        if self.target != NONE {
                            self.send(self.target, self.atoms.XdndLeave)?;
                        }
                        self.target = next_target;
                        self.target_version = version.min(VERSION);
                        self.accepted = None;
                        last_position = None;
                        if self.target != NONE {
                            send(
                                &self.connection,
                                self.target,
                                self.atoms.XdndEnter,
                                [self.source, self.target_version << 24, self.atoms.URI, 0, 0],
                            )?;
                        }
                    }
                    requested_action = if files.operation == FileTransferOperation::Move
                        && !pointer.mask.contains(KeyButMask::CONTROL)
                    {
                        self.atoms.XdndActionMove
                    } else {
                        self.atoms.XdndActionCopy
                    };
                    let coordinates =
                        ((pointer.root_x as u16 as u32) << 16) | pointer.root_y as u16 as u32;
                    if self.target != NONE && last_position != Some((coordinates, requested_action))
                    {
                        send(
                            &self.connection,
                            self.target,
                            self.atoms.XdndPosition,
                            [self.source, 0, coordinates, time, requested_action],
                        )?;
                        last_position = Some((coordinates, requested_action));
                    }
                }
            }
            while let Some(event) = self.connection.poll_for_event()? {
                match event {
                    Event::MotionNotify(event) => time = event.time,
                    Event::ButtonRelease(event) => time = event.time,
                    Event::KeyPress(event) => {
                        let keys = self
                            .connection
                            .get_keyboard_mapping(event.detail, 1)?
                            .reply()?;
                        if keys.keysyms.contains(&0xff1b) {
                            return Ok(None);
                        }
                    }
                    Event::ClientMessage(event)
                        if event.type_ == self.atoms.XdndStatus && !self.dropped =>
                    {
                        let data = event.data.as_data32();
                        if data[0] == self.target {
                            let usable = data[4] == self.atoms.XdndActionCopy
                                || data[4] == self.atoms.XdndActionMove
                                    && requested_action == self.atoms.XdndActionMove;
                            self.accepted = (data[1] & 1 != 0 && usable).then_some(data[4]);
                        }
                    }
                    Event::ClientMessage(event)
                        if event.type_ == self.atoms.XdndFinished && self.dropped =>
                    {
                        let data = event.data.as_data32();
                        if data[0] == self.target {
                            return Ok(finished_operation(
                                self.target_version,
                                data,
                                self.accepted.unwrap_or(NONE),
                                files.operation,
                                self.atoms.XdndActionCopy,
                                self.atoms.XdndActionMove,
                            ));
                        }
                    }
                    Event::SelectionRequest(request)
                        if request.selection == self.atoms.XdndSelection =>
                    {
                        self.answer_selection_request(&request, &bytes, &mut increments)?;
                        if self.dropped {
                            finished_deadline = Some(Instant::now() + finished_timeout);
                        }
                    }
                    Event::PropertyNotify(event) if event.state == Property::DELETE => {
                        if let Some(index) = increments
                            .iter()
                            .position(|i| i.requestor == event.window && i.property == event.atom)
                        {
                            let increment = &mut increments[index];
                            let end = (increment.offset + INCR_CHUNK).min(bytes.len());
                            self.connection
                                .change_property8(
                                    PropMode::REPLACE,
                                    increment.requestor,
                                    increment.property,
                                    self.atoms.URI,
                                    &bytes[increment.offset..end],
                                )?
                                .check()?;
                            if increment.offset == bytes.len() {
                                increments.remove(index);
                            } else {
                                increment.offset = end;
                            }
                            self.connection.flush()?;
                            if self.dropped {
                                finished_deadline = Some(Instant::now() + finished_timeout);
                            }
                        }
                    }
                    Event::SelectionClear(_) => return Ok(None),
                    _ => {}
                }
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    fn answer_selection_request(
        &self,
        request: &SelectionRequestEvent,
        bytes: &[u8],
        increments: &mut Vec<Increment>,
    ) -> anyhow::Result<()> {
        let property = if request.property == NONE {
            request.target
        } else {
            request.property
        };
        let response = if request.target == self.atoms.TARGETS {
            self.connection
                .change_property32(
                    PropMode::REPLACE,
                    request.requestor,
                    property,
                    AtomEnum::ATOM,
                    &[self.atoms.TARGETS, self.atoms.URI],
                )?
                .check()?;
            property
        } else if request.target == self.atoms.URI {
            if bytes.len() <= INCR_CHUNK {
                self.connection
                    .change_property8(
                        PropMode::REPLACE,
                        request.requestor,
                        property,
                        self.atoms.URI,
                        bytes,
                    )?
                    .check()?;
            } else {
                self.connection
                    .change_window_attributes(
                        request.requestor,
                        &ChangeWindowAttributesAux::new().event_mask(EventMask::PROPERTY_CHANGE),
                    )?
                    .check()?;
                self.connection
                    .change_property32(
                        PropMode::REPLACE,
                        request.requestor,
                        property,
                        self.atoms.INCR,
                        &[bytes.len() as u32],
                    )?
                    .check()?;
                increments.push(Increment {
                    requestor: request.requestor,
                    property,
                    offset: 0,
                });
            }
            property
        } else {
            NONE
        };
        self.connection
            .send_event(
                false,
                request.requestor,
                EventMask::NO_EVENT,
                SelectionNotifyEvent {
                    response_type: SELECTION_NOTIFY_EVENT,
                    sequence: 0,
                    time: request.time,
                    requestor: request.requestor,
                    selection: request.selection,
                    target: request.target,
                    property: response,
                },
            )?
            .check()?;
        self.connection.flush()?;
        Ok(())
    }

    /// The deepest Xdnd-aware window under the pointer, honouring XdndProxy,
    /// and the protocol version it announces.
    fn target_under_pointer(&self) -> anyhow::Result<(u32, u32)> {
        let mut current = self.root;
        let mut target = NONE;
        let mut version = 0;
        // Walk through window-manager frames to the deepest aware client window.
        for _ in 0..64 {
            if current != self.source
                && let Some(announced) = self.window_property(current, self.atoms.XdndAware)?
            {
                target = current;
                version = announced;
            }
            // A window destroyed during the walk simply has no children.
            let Some(pointer) = self.connection.query_pointer(current)?.reply().ok() else {
                break;
            };
            if pointer.child == NONE || pointer.child == self.source {
                break;
            }
            current = pointer.child;
        }
        if target != NONE
            && let Some(proxy) = self.window_property(target, self.atoms.XdndProxy)?
            && self.window_property(proxy, self.atoms.XdndProxy)? == Some(proxy)
        {
            target = proxy;
            if let Some(announced) = self.window_property(proxy, self.atoms.XdndAware)? {
                version = announced;
            }
        }
        Ok((target, version))
    }

    /// The first 32-bit value of `property` on `window`, or `None` when the
    /// property is absent or the window no longer exists. Only losing the
    /// connection is an error.
    fn window_property(&self, window: u32, property: u32) -> anyhow::Result<Option<u32>> {
        Ok(self
            .connection
            .get_property(false, window, property, AtomEnum::ANY, 0, 1)?
            .reply()
            .ok()
            .and_then(|reply| reply.value32()?.next()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linux::x11::test_display::Xvfb;
    use std::{path::PathBuf, thread::JoinHandle};
    use x11rb::protocol::xtest::ConnectionExt as _;

    const COPY: u32 = 1;
    const MOVE: u32 = 2;

    #[test]
    fn finished_messages_are_read_by_the_target_version() {
        let finished = |version, data, accepted| {
            finished_operation(
                version,
                data,
                accepted,
                FileTransferOperation::Move,
                COPY,
                MOVE,
            )
        };
        // Version 5 targets report the result in the message.
        assert_eq!(
            finished(5, [0, 1, COPY, 0, 0], MOVE),
            Some(FileTransferOperation::Copy)
        );
        assert_eq!(
            finished(5, [0, 1, MOVE, 0, 0], MOVE),
            Some(FileTransferOperation::Move)
        );
        assert_eq!(finished(5, [0, 0, 0, 0, 0], MOVE), None);
        // Older targets leave those fields zero: the drop they accepted was performed.
        assert_eq!(
            finished(4, [0, 0, 0, 0, 0], MOVE),
            Some(FileTransferOperation::Move)
        );
        assert_eq!(
            finished(4, [0, 0, 0, 0, 0], COPY),
            Some(FileTransferOperation::Copy)
        );
        // A move the application did not ask for is not reported as one.
        assert_eq!(
            finished_operation(
                5,
                [0, 1, MOVE, 0, 0],
                MOVE,
                FileTransferOperation::Copy,
                COPY,
                MOVE
            ),
            None
        );
    }

    /// An Xdnd-aware window on a private server, driven by the test as the drop
    /// target, with the pointer over it and button 1 held through XTest.
    struct Target {
        server: Xvfb,
        connection: RustConnection,
        atoms: Atoms,
        root: u32,
        window: u32,
    }

    impl Target {
        fn start(version: u32) -> Option<Self> {
            let server = Xvfb::start()?;
            let (connection, screen) = x11rb::connect(Some(&server.display)).unwrap();
            let root = connection.setup().roots[screen].root;
            let atoms = Atoms::new(&connection).unwrap().reply().unwrap();
            let window = connection.generate_id().unwrap();
            connection
                .create_window(
                    0,
                    window,
                    root,
                    0,
                    0,
                    64,
                    64,
                    0,
                    WindowClass::INPUT_OUTPUT,
                    0,
                    &CreateWindowAux::new().override_redirect(1),
                )
                .unwrap()
                .check()
                .unwrap();
            connection
                .change_property32(
                    PropMode::REPLACE,
                    window,
                    atoms.XdndAware,
                    AtomEnum::ATOM,
                    &[version],
                )
                .unwrap()
                .check()
                .unwrap();
            connection.map_window(window).unwrap().check().unwrap();
            connection
                .warp_pointer(NONE, root, 0, 0, 0, 0, 16, 16)
                .unwrap()
                .check()
                .unwrap();
            let target = Self {
                server,
                connection,
                atoms,
                root,
                window,
            };
            target.button(BUTTON_PRESS_EVENT);
            Some(target)
        }

        fn button(&self, event: u8) {
            self.connection
                .xtest_fake_input(event, 1, CURRENT_TIME, self.root, 0, 0, 0)
                .unwrap()
                .check()
                .unwrap();
            self.connection.flush().unwrap();
        }

        fn set_proxy(&self, proxy: u32) {
            self.connection
                .change_property32(
                    PropMode::REPLACE,
                    self.window,
                    self.atoms.XdndProxy,
                    AtomEnum::WINDOW,
                    &[proxy],
                )
                .unwrap()
                .check()
                .unwrap();
            self.connection.flush().unwrap();
        }

        fn drag(&self, finished_timeout: Duration) -> JoinHandle<FileTransferCompletion> {
            let display = self.server.display.clone();
            let files = FileTransfer {
                paths: gpui::ExternalPaths([PathBuf::from("/tmp/dragged")].into_iter().collect()),
                operation: FileTransferOperation::Copy,
                ownership: 11,
            };
            std::thread::spawn(move || run_on(files, Some(&display), finished_timeout))
        }

        /// The next Xdnd message of the given kind, panicking if another kind
        /// arrives first or nothing arrives in time.
        fn expect(&self, kind: u32) -> [u32; 5] {
            let deadline = Instant::now() + Duration::from_secs(10);
            while Instant::now() < deadline {
                if let Some(Event::ClientMessage(message)) =
                    self.connection.poll_for_event().unwrap()
                    && message.window == self.window
                {
                    assert_eq!(message.type_, kind, "unexpected Xdnd message");
                    return message.data.as_data32();
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            panic!("no Xdnd message arrived");
        }

        fn reply(&self, source: u32, kind: u32, data: [u32; 5]) {
            send(&self.connection, source, kind, data).unwrap();
        }
    }

    macro_rules! target_or_skip {
        ($version:expr) => {
            match Target::start($version) {
                Some(target) => target,
                None => {
                    eprintln!("Xvfb is not available; skipping");
                    return;
                }
            }
        };
    }

    #[test]
    fn drops_onto_older_targets_are_reported_as_performed() {
        let target = target_or_skip!(4);
        let session = target.drag(FINISHED_TIMEOUT);
        let [source, flags, ..] = target.expect(target.atoms.XdndEnter);
        assert_eq!(flags >> 24, 4, "the source must speak the target's version");
        target.expect(target.atoms.XdndPosition);
        target.reply(
            source,
            target.atoms.XdndStatus,
            [target.window, 1, 0, 0, target.atoms.XdndActionCopy],
        );
        target.button(BUTTON_RELEASE_EVENT);
        target.expect(target.atoms.XdndDrop);
        // Before version 5, XdndFinished says nothing about the outcome.
        target.reply(
            source,
            target.atoms.XdndFinished,
            [target.window, 0, 0, 0, 0],
        );
        let completion = session.join().unwrap();
        assert_eq!(completion.operation, Some(FileTransferOperation::Copy));
    }

    #[test]
    fn a_target_refusing_the_drop_is_told_the_drag_ended() {
        let target = target_or_skip!(5);
        let session = target.drag(FINISHED_TIMEOUT);
        let [source, ..] = target.expect(target.atoms.XdndEnter);
        target.expect(target.atoms.XdndPosition);
        target.reply(source, target.atoms.XdndStatus, [target.window, 0, 0, 0, 0]);
        target.button(BUTTON_RELEASE_EVENT);
        target.expect(target.atoms.XdndLeave);
        assert_eq!(session.join().unwrap().operation, None);
    }

    #[test]
    fn a_target_that_never_finishes_does_not_hold_the_drag() {
        let target = target_or_skip!(5);
        let session = target.drag(Duration::from_millis(500));
        let [source, ..] = target.expect(target.atoms.XdndEnter);
        target.expect(target.atoms.XdndPosition);
        target.reply(
            source,
            target.atoms.XdndStatus,
            [target.window, 1, 0, 0, target.atoms.XdndActionCopy],
        );
        target.button(BUTTON_RELEASE_EVENT);
        target.expect(target.atoms.XdndDrop);
        let dropped = Instant::now();
        assert_eq!(session.join().unwrap().operation, None);
        assert!(dropped.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn a_proxy_that_no_longer_exists_does_not_stop_the_drag() {
        let target = target_or_skip!(5);
        target.set_proxy(target.connection.generate_id().unwrap());
        let session = target.drag(FINISHED_TIMEOUT);
        target.expect(target.atoms.XdndEnter);
        target.button(BUTTON_RELEASE_EVENT);
        assert_eq!(session.join().unwrap().operation, None);
    }
}
