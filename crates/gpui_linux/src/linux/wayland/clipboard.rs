use std::{
    fs::File,
    io::{ErrorKind, Write},
    os::fd::{AsRawFd, BorrowedFd, OwnedFd},
};

use calloop::{LoopHandle, PostAction};
use filedescriptor::Pipe;
use strum::IntoEnumIterator;
use wayland_client::{Connection, protocol::wl_data_offer::WlDataOffer};
use wayland_protocols::wp::primary_selection::zv1::client::zwp_primary_selection_offer_v1::ZwpPrimarySelectionOfferV1;

use crate::linux::{
    WaylandClientStatePtr,
    platform::{PIPE_READ_TIMEOUT, read_fd_with_timeout},
};
use gpui::{ClipboardEntry, ClipboardItem, Image, ImageFormat, hash};

/// Text mime types that we'll offer to other programs.
pub(crate) const TEXT_MIME_TYPES: [&str; 3] =
    ["text/plain;charset=utf-8", "UTF8_STRING", "text/plain"];
pub(crate) const FILE_LIST_MIME_TYPE: &str = "text/uri-list";

/// Text mime types that we'll accept from other programs.
pub(crate) const ALLOWED_TEXT_MIME_TYPES: [&str; 2] = ["text/plain;charset=utf-8", "UTF8_STRING"];

pub(crate) struct Clipboard {
    connection: Connection,
    loop_handle: LoopHandle<'static, WaylandClientStatePtr>,
    self_mime: String,

    // Internal clipboard
    contents: Option<ClipboardItem>,
    primary_contents: Option<ClipboardItem>,

    // External clipboard
    cached_read: Option<ClipboardItem>,
    current_offer: Option<DataOffer<WlDataOffer>>,
    cached_primary_read: Option<ClipboardItem>,
    current_primary_offer: Option<DataOffer<ZwpPrimarySelectionOfferV1>>,
}

pub(crate) trait ReceiveData {
    fn receive_data(&self, mime_type: String, fd: BorrowedFd<'_>);
}

impl ReceiveData for WlDataOffer {
    fn receive_data(&self, mime_type: String, fd: BorrowedFd<'_>) {
        self.receive(mime_type, fd);
    }
}

impl ReceiveData for ZwpPrimarySelectionOfferV1 {
    fn receive_data(&self, mime_type: String, fd: BorrowedFd<'_>) {
        self.receive(mime_type, fd);
    }
}

#[derive(Clone, Debug)]
/// Wrapper for `WlDataOffer` and `ZwpPrimarySelectionOfferV1`, used to help track mime types.
pub(crate) struct DataOffer<T: ReceiveData> {
    pub inner: T,
    mime_types: Vec<String>,
}

impl<T: ReceiveData> DataOffer<T> {
    pub fn new(offer: T) -> Self {
        Self {
            inner: offer,
            mime_types: Vec::new(),
        }
    }

    pub fn add_mime_type(&mut self, mime_type: String) {
        self.mime_types.push(mime_type)
    }

    pub(crate) fn has_mime_type(&self, mime_type: &str) -> bool {
        self.mime_types.iter().any(|t| t == mime_type)
    }

    fn read_bytes(&self, connection: &Connection, mime_type: &str) -> Option<Vec<u8>> {
        let pipe = Pipe::new().unwrap();
        self.inner.receive_data(mime_type.to_string(), unsafe {
            BorrowedFd::borrow_raw(pipe.write.as_raw_fd())
        });
        let fd = pipe.read;
        drop(pipe.write);

        connection.flush().unwrap();

        match read_fd_with_timeout(fd, PIPE_READ_TIMEOUT) {
            Ok(bytes) => Some(bytes),
            Err(err) => {
                log::error!("error reading clipboard pipe: {err:?}");
                None
            }
        }
    }

    fn read_text(&self, read: &mut impl FnMut(&str) -> Option<Vec<u8>>) -> Option<ClipboardItem> {
        let mime_type = self.mime_types.iter().find(|&mime_type| {
            ALLOWED_TEXT_MIME_TYPES
                .iter()
                .any(|&allowed| allowed == mime_type)
        })?;
        let bytes = read(mime_type)?;
        let text_content = match String::from_utf8(bytes) {
            Ok(content) => content,
            Err(e) => {
                log::error!("Failed to convert clipboard content to UTF-8: {}", e);
                return None;
            }
        };

        // Normalize the text to unix line endings, otherwise
        // copying from eg: firefox inserts a lot of blank
        // lines, and that is super annoying.
        let result = text_content.replace("\r\n", "\n");
        Some(ClipboardItem::new_string(result))
    }

    fn read_image(&self, read: &mut impl FnMut(&str) -> Option<Vec<u8>>) -> Option<ClipboardItem> {
        for format in ImageFormat::iter() {
            let mime_type = format.mime_type();
            if !self.has_mime_type(mime_type) {
                continue;
            }

            if let Some(bytes) = read(mime_type) {
                let id = hash(&bytes);
                return Some(ClipboardItem {
                    entries: vec![ClipboardEntry::Image(Image { format, bytes, id })],
                });
            }
        }
        None
    }

    fn read_clipboard(
        &self,
        read: &mut impl FnMut(&str) -> Option<Vec<u8>>,
    ) -> Option<ClipboardItem> {
        let files = [
            gpui::FILE_TRANSFER_MIME,
            gpui::COPIED_FILES_MIME,
            gpui::URI_LIST_MIME,
        ]
        .into_iter()
        .find_map(|mime| {
            if !self.has_mime_type(mime) {
                return None;
            }
            let mut files = gpui::FileTransfer::decode(&read(mime)?, mime)?;
            if mime == gpui::URI_LIST_MIME
                && self.has_mime_type(gpui::KDE_CUT_MIME)
                && read(gpui::KDE_CUT_MIME).is_some_and(|b| b == b"1")
            {
                files.operation = gpui::FileTransferOperation::Move;
            }
            Some(files)
        });
        let text = self.read_text(read);
        if let Some(files) = files {
            let mut item = text.unwrap_or_else(|| ClipboardItem {
                entries: Vec::new(),
            });
            item.entries.push(ClipboardEntry::Files(files));
            Some(item)
        } else {
            text.or_else(|| self.read_image(read))
        }
    }
}

impl Clipboard {
    pub fn new(
        connection: Connection,
        loop_handle: LoopHandle<'static, WaylandClientStatePtr>,
    ) -> Self {
        Self {
            connection,
            loop_handle,
            self_mime: format!("pid/{}", std::process::id()),

            contents: None,
            primary_contents: None,

            cached_read: None,
            current_offer: None,
            cached_primary_read: None,
            current_primary_offer: None,
        }
    }

    pub fn set(&mut self, item: ClipboardItem) {
        self.contents = Some(item);
    }

    pub fn set_primary(&mut self, item: ClipboardItem) {
        self.primary_contents = Some(item);
    }

    pub fn set_offer(&mut self, data_offer: Option<DataOffer<WlDataOffer>>) {
        self.cached_read = None;
        self.current_offer = data_offer;
    }

    pub fn set_primary_offer(&mut self, data_offer: Option<DataOffer<ZwpPrimarySelectionOfferV1>>) {
        self.cached_primary_read = None;
        self.current_primary_offer = data_offer;
    }

    pub fn self_mime(&self) -> String {
        self.self_mime.clone()
    }

    pub fn send(&self, mime_type: String, fd: OwnedFd) {
        if let Some(bytes) = self
            .contents
            .as_ref()
            .and_then(|c| c.file_transfer())
            .and_then(|f| f.encode(&mime_type))
        {
            self.send_bytes(fd, bytes);
            return;
        }
        if let Some(text) = self.contents.as_ref().and_then(|contents| contents.text()) {
            self.send_bytes(fd, text.as_bytes().to_owned());
        }
    }

    pub fn send_primary(&self, _mime_type: String, fd: OwnedFd) {
        if let Some(text) = self
            .primary_contents
            .as_ref()
            .and_then(|contents| contents.text())
        {
            self.send_bytes(fd, text.as_bytes().to_owned());
        }
    }

    pub fn read(&mut self) -> Option<ClipboardItem> {
        let offer = self.current_offer.as_ref()?;
        if let Some(cached) = self.cached_read.clone() {
            return Some(cached);
        }

        if offer.has_mime_type(&self.self_mime) {
            return self.contents.clone();
        }

        let item = offer.read_clipboard(&mut |mime| offer.read_bytes(&self.connection, mime))?;

        self.cached_read = Some(item.clone());
        Some(item)
    }

    pub fn read_primary(&mut self) -> Option<ClipboardItem> {
        let offer = self.current_primary_offer.as_ref()?;
        if let Some(cached) = self.cached_primary_read.clone() {
            return Some(cached);
        }

        if offer.has_mime_type(&self.self_mime) {
            return self.primary_contents.clone();
        }

        let mut read = |mime: &str| offer.read_bytes(&self.connection, mime);
        let item = offer
            .read_text(&mut read)
            .or_else(|| offer.read_image(&mut read))?;

        self.cached_primary_read = Some(item.clone());
        Some(item)
    }

    pub fn send_bytes(&self, fd: OwnedFd, bytes: Vec<u8>) {
        let mut written = 0;
        self.loop_handle
            .insert_source(
                calloop::generic::Generic::new(
                    File::from(fd),
                    calloop::Interest::WRITE,
                    calloop::Mode::Level,
                ),
                move |_, file, _| {
                    let file = unsafe { file.get_mut() };
                    loop {
                        match file.write(&bytes[written..]) {
                            Ok(n) if written + n == bytes.len() => {
                                written += n;
                                break Ok(PostAction::Remove);
                            }
                            Ok(n) => written += n,
                            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                                break Ok(PostAction::Continue);
                            }
                            Err(_) => break Ok(PostAction::Remove),
                        }
                    }
                },
            )
            .unwrap();
    }

    /// File formats the *clipboard* data source should advertise, on top of the
    /// text types it always offers. Describes `contents` only; the primary
    /// selection carries text and must not consult this.
    pub fn file_mime_types(&self) -> &'static [&'static str] {
        if self
            .contents
            .as_ref()
            .and_then(|c| c.file_transfer())
            .is_some()
        {
            &[
                gpui::FILE_TRANSFER_MIME,
                gpui::COPIED_FILES_MIME,
                gpui::URI_LIST_MIME,
                gpui::KDE_CUT_MIME,
            ]
        } else {
            &[]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Offer;
    impl ReceiveData for Offer {
        fn receive_data(&self, _: String, _: BorrowedFd<'_>) {
            unreachable!("tests supply the offered bytes directly")
        }
    }

    fn read(formats: &[(&str, &[u8])]) -> ClipboardItem {
        let mut offer = DataOffer::new(Offer);
        for (mime, _) in formats {
            offer.add_mime_type((*mime).into());
        }
        offer
            .read_clipboard(&mut |mime| {
                formats
                    .iter()
                    .find(|(offered, _)| *offered == mime)
                    .map(|(_, bytes)| bytes.to_vec())
            })
            .unwrap()
    }

    #[test]
    fn file_offers_keep_their_explicit_text() {
        for mime in ALLOWED_TEXT_MIME_TYPES {
            let item = read(&[
                (gpui::URI_LIST_MIME, b"file:///tmp/a\r\nfile:///tmp/b\r\n"),
                (gpui::KDE_CUT_MIME, b"1"),
                (mime, b"First file\r\nSecond file"),
            ]);
            let files = item.file_transfer().unwrap();
            assert_eq!(files.paths.paths().len(), 2);
            assert_eq!(files.operation, gpui::FileTransferOperation::Move);
            assert_eq!(item.text().as_deref(), Some("First file\nSecond file"));
        }
    }

    #[test]
    fn missing_or_invalid_text_does_not_discard_files() {
        for text in [None, Some(&b"\xff"[..])] {
            let mut formats = vec![(gpui::URI_LIST_MIME, &b"file:///tmp/a\r\nfile:///tmp/b"[..])];
            if let Some(text) = text {
                formats.push((ALLOWED_TEXT_MIME_TYPES[0], text));
            }
            let item = read(&formats);
            assert!(item.file_transfer().is_some());
            assert_eq!(item.text().as_deref(), Some("/tmp/a\n/tmp/b"));
        }
    }

    #[test]
    fn non_file_uri_lists_still_paste_as_text() {
        let item = read(&[
            (gpui::URI_LIST_MIME, b"https://example.com"),
            (ALLOWED_TEXT_MIME_TYPES[0], b"Link label"),
        ]);
        assert!(item.file_transfer().is_none());
        assert_eq!(item.text().as_deref(), Some("Link label"));
    }
}
