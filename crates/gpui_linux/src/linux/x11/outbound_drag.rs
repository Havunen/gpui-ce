//! Xdnd version 5 source. Selection ownership lives until XdndFinished, including
//! INCR reads; the receiving application alone decides whether a drop succeeded.
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

struct Increment {
    requestor: u32,
    property: u32,
    offset: usize,
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

fn target_under_pointer(
    connection: &RustConnection,
    atoms: &Atoms,
    root: u32,
    source: u32,
) -> anyhow::Result<u32> {
    let mut current = root;
    let mut target = NONE;
    // Walk through window-manager frames to the deepest aware client window.
    for _ in 0..64 {
        let aware = connection
            .get_property(false, current, atoms.XdndAware, AtomEnum::ATOM, 0, 1)?
            .reply()?;
        if current != source && aware.value32().is_some_and(|mut v| v.next().is_some()) {
            target = current;
        }
        let child = connection.query_pointer(current)?.reply()?.child;
        if child == NONE || child == source {
            break;
        }
        current = child;
    }
    if target != NONE {
        let proxy = connection
            .get_property(false, target, atoms.XdndProxy, AtomEnum::WINDOW, 0, 1)?
            .reply()?
            .value32()
            .and_then(|mut p| p.next());
        if let Some(proxy) = proxy {
            let valid = connection
                .get_property(false, proxy, atoms.XdndProxy, AtomEnum::WINDOW, 0, 1)?
                .reply()?
                .value32()
                .and_then(|mut p| p.next())
                == Some(proxy);
            if valid {
                target = proxy;
            }
        }
    }
    Ok(target)
}

pub(super) fn run(files: FileTransfer) -> FileTransferCompletion {
    let result = run_session(&files);
    if let Err(error) = &result {
        log::warn!("Native X11 drag stopped: {error}");
    }
    FileTransferCompletion {
        files,
        operation: result.ok().flatten(),
        source_removed: false,
    }
}

fn run_session(files: &FileTransfer) -> anyhow::Result<Option<FileTransferOperation>> {
    let (connection, screen) = x11rb::connect(None)?;
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
    let bytes = files.uri_list();
    anyhow::ensure!(!bytes.is_empty(), "No local file URLs");
    let mut target = NONE;
    let mut accepted = false;
    let mut requested_action = NONE;
    let mut time = CURRENT_TIME;
    let mut dropped = false;
    let mut released = None;
    let mut increments: Vec<Increment> = Vec::new();
    let mut last_position = None;
    let deadline = Instant::now() + Duration::from_secs(600);
    loop {
        if Instant::now() > deadline {
            return Ok(None);
        }
        if !dropped {
            let pointer = connection.query_pointer(root)?.reply()?;
            if !pointer.mask.contains(KeyButMask::BUTTON1) {
                if target == NONE {
                    return Ok(None);
                }
                let released_at = *released.get_or_insert_with(Instant::now);
                if !accepted && released_at.elapsed() > Duration::from_millis(250) {
                    return Ok(None);
                }
                if accepted {
                    send(&connection, target, atoms.XdndDrop, [source, 0, time, 0, 0])?;
                    connection.ungrab_pointer(CURRENT_TIME)?;
                    connection.ungrab_keyboard(CURRENT_TIME)?;
                    connection.flush()?;
                    dropped = true;
                }
            } else {
                let next_target = target_under_pointer(&connection, &atoms, root, source)?;
                if next_target != target {
                    if target != NONE {
                        send(&connection, target, atoms.XdndLeave, [source, 0, 0, 0, 0])?;
                    }
                    target = next_target;
                    accepted = false;
                    last_position = None;
                    if target != NONE {
                        send(
                            &connection,
                            target,
                            atoms.XdndEnter,
                            [source, 5 << 24, atoms.URI, 0, 0],
                        )?;
                    }
                }
                requested_action = if files.operation == FileTransferOperation::Move
                    && !pointer.mask.contains(KeyButMask::CONTROL)
                {
                    atoms.XdndActionMove
                } else {
                    atoms.XdndActionCopy
                };
                let coordinates =
                    ((pointer.root_x as u16 as u32) << 16) | pointer.root_y as u16 as u32;
                if target != NONE && last_position != Some((coordinates, requested_action)) {
                    send(
                        &connection,
                        target,
                        atoms.XdndPosition,
                        [source, 0, coordinates, time, requested_action],
                    )?;
                    last_position = Some((coordinates, requested_action));
                }
            }
        }
        while let Some(event) = connection.poll_for_event()? {
            match event {
                Event::MotionNotify(event) => time = event.time,
                Event::ButtonRelease(event) => time = event.time,
                Event::KeyPress(event) => {
                    let keys = connection.get_keyboard_mapping(event.detail, 1)?.reply()?;
                    if keys.keysyms.contains(&0xff1b) {
                        if target != NONE {
                            let _ =
                                send(&connection, target, atoms.XdndLeave, [source, 0, 0, 0, 0]);
                        }
                        return Ok(None);
                    }
                }
                Event::ClientMessage(event) if event.type_ == atoms.XdndStatus && !dropped => {
                    let data = event.data.as_data32();
                    if data[0] == target {
                        accepted = data[1] & 1 != 0
                            && (data[4] == atoms.XdndActionCopy
                                || data[4] == atoms.XdndActionMove
                                    && requested_action == atoms.XdndActionMove);
                    }
                }
                Event::ClientMessage(event) if event.type_ == atoms.XdndFinished && dropped => {
                    let data = event.data.as_data32();
                    if data[0] == target {
                        return Ok(if data[1] & 1 == 0 {
                            None
                        } else if data[2] == atoms.XdndActionCopy {
                            Some(FileTransferOperation::Copy)
                        } else if data[2] == atoms.XdndActionMove
                            && files.operation == FileTransferOperation::Move
                        {
                            Some(FileTransferOperation::Move)
                        } else {
                            None
                        });
                    }
                }
                Event::SelectionRequest(request) if request.selection == atoms.XdndSelection => {
                    let property = if request.property == NONE {
                        request.target
                    } else {
                        request.property
                    };
                    let response = if request.target == atoms.TARGETS {
                        connection
                            .change_property32(
                                PropMode::REPLACE,
                                request.requestor,
                                property,
                                AtomEnum::ATOM,
                                &[atoms.TARGETS, atoms.URI],
                            )?
                            .check()?;
                        property
                    } else if request.target == atoms.URI {
                        if bytes.len() <= 64 * 1024 {
                            connection
                                .change_property8(
                                    PropMode::REPLACE,
                                    request.requestor,
                                    property,
                                    atoms.URI,
                                    &bytes,
                                )?
                                .check()?;
                        } else {
                            connection
                                .change_window_attributes(
                                    request.requestor,
                                    &ChangeWindowAttributesAux::new()
                                        .event_mask(EventMask::PROPERTY_CHANGE),
                                )?
                                .check()?;
                            connection
                                .change_property32(
                                    PropMode::REPLACE,
                                    request.requestor,
                                    property,
                                    atoms.INCR,
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
                    connection
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
                    connection.flush()?;
                }
                Event::PropertyNotify(event) if event.state == Property::DELETE => {
                    if let Some(index) = increments
                        .iter()
                        .position(|i| i.requestor == event.window && i.property == event.atom)
                    {
                        let increment = &mut increments[index];
                        let end = (increment.offset + 64 * 1024).min(bytes.len());
                        connection
                            .change_property8(
                                PropMode::REPLACE,
                                increment.requestor,
                                increment.property,
                                atoms.URI,
                                &bytes[increment.offset..end],
                            )?
                            .check()?;
                        if increment.offset == bytes.len() {
                            increments.remove(index);
                        } else {
                            increment.offset = end;
                        }
                        connection.flush()?;
                    }
                }
                Event::SelectionClear(_) => return Ok(None),
                _ => {}
            }
        }
        std::thread::sleep(Duration::from_millis(8));
    }
}
