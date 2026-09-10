use crate::{ClipboardEntry, ClipboardItem, ExternalPaths};
use std::path::PathBuf;

/// Negotiated operation for a native file transfer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FileTransferOperation {
    /// Duplicate source items at the destination.
    #[default]
    Copy,
    /// Transfer ownership of source items after successful completion.
    Move,
}

/// Native files are distinct from strings that happen to contain paths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileTransfer {
    /// Absolute, native source pathnames.
    pub paths: ExternalPaths,
    /// Requested operation before negotiation.
    pub operation: FileTransferOperation,
    /// Application-generated token, preserved only by a compatible owner.
    pub ownership: u64,
}

/// Native file clipboard format identifier.
pub const FILE_TRANSFER_MIME: &str = "application/x-gpui-file-transfer";
/// Native file clipboard format identifier.
pub const COPIED_FILES_MIME: &str = "x-special/gnome-copied-files";
/// Native file clipboard format identifier.
pub const URI_LIST_MIME: &str = "text/uri-list";
/// Native file clipboard format identifier.
pub const KDE_CUT_MIME: &str = "application/x-kde-cutselection";

impl FileTransfer {
    /// Encode local file URLs with native path escaping.
    pub fn uri_list(&self) -> Vec<u8> {
        self.paths
            .paths()
            .iter()
            .filter_map(|path| url::Url::from_file_path(path).ok())
            .map(|url| format!("{url}\r\n"))
            .collect::<String>()
            .into_bytes()
    }

    /// Encode one of the advertised file clipboard formats.
    pub fn encode(&self, mime: &str) -> Option<Vec<u8>> {
        let operation = if self.operation == FileTransferOperation::Move {
            "cut"
        } else {
            "copy"
        };
        let mut out = match mime {
            URI_LIST_MIME => vec![],
            COPIED_FILES_MIME => format!("{operation}\n").into_bytes(),
            FILE_TRANSFER_MIME => format!("{operation}\n{}\n", self.ownership).into_bytes(),
            KDE_CUT_MIME => {
                return Some(
                    if self.operation == FileTransferOperation::Move {
                        b"1"
                    } else {
                        b"0"
                    }
                    .to_vec(),
                );
            }
            _ => return None,
        };
        if mime == COPIED_FILES_MIME {
            // Nautilus splits on LF and rejects empty lines, including a
            // trailing newline. This is distinct from RFC text/uri-list.
            out.extend(
                self.paths
                    .paths()
                    .iter()
                    .filter_map(|path| url::Url::from_file_path(path).ok())
                    .map(|url| url.to_string())
                    .collect::<Vec<_>>()
                    .join("\n")
                    .as_bytes(),
            );
        } else {
            out.extend(self.uri_list());
        }
        Some(out)
    }

    /// Decode only local file URLs in a typed native format.
    pub fn decode(bytes: &[u8], mime: &str) -> Option<Self> {
        let text = std::str::from_utf8(bytes).ok()?;
        let mut lines = text.lines();
        let operation = match mime {
            URI_LIST_MIME => FileTransferOperation::Copy,
            FILE_TRANSFER_MIME | COPIED_FILES_MIME => match lines.next()? {
                "copy" => FileTransferOperation::Copy,
                "cut" => FileTransferOperation::Move,
                _ => return None,
            },
            _ => return None,
        };
        let ownership = if mime == FILE_TRANSFER_MIME {
            lines.next()?.parse().ok()?
        } else {
            0
        };
        let paths = lines
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                let url = url::Url::parse(line).ok()?;
                if url.scheme() != "file"
                    || url
                        .host_str()
                        .is_some_and(|h| !h.is_empty() && h != "localhost")
                    || url.query().is_some()
                    || url.fragment().is_some()
                {
                    return None;
                }
                url.to_file_path()
                    .ok()
                    .filter(|p: &PathBuf| p.is_absolute())
            })
            .collect::<Option<Vec<_>>>()?;
        (!paths.is_empty()).then(|| Self {
            paths: ExternalPaths(paths.into_iter().collect()),
            operation,
            ownership,
        })
    }
}

impl ClipboardItem {
    /// Extract typed files, keeping ordinary text distinct.
    pub fn file_transfer(&self) -> Option<FileTransfer> {
        self.entries.iter().find_map(|entry| match entry {
            ClipboardEntry::Files(files) => Some(files.clone()),
            ClipboardEntry::ExternalPaths(paths) => Some(FileTransfer {
                paths: paths.clone(),
                operation: FileTransferOperation::Copy,
                ownership: 0,
            }),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nautilus_payload_has_lf_separators_and_no_empty_lines() {
        let root = if cfg!(windows) { "C:/tmp" } else { "/tmp" };
        let urls = if cfg!(windows) {
            ["file:///C:/tmp/a", "file:///C:/tmp/b"]
        } else {
            ["file:///tmp/a", "file:///tmp/b"]
        };
        let files = FileTransfer {
            paths: ExternalPaths(
                [PathBuf::from(root).join("a"), PathBuf::from(root).join("b")].into(),
            ),
            operation: FileTransferOperation::Copy,
            ownership: 7,
        };
        assert_eq!(
            files.encode(COPIED_FILES_MIME).unwrap(),
            format!("copy\n{}\n{}", urls[0], urls[1]).as_bytes()
        );
        assert_eq!(
            files.uri_list(),
            format!("{}\r\n{}\r\n", urls[0], urls[1]).as_bytes()
        );
    }

    #[test]
    fn paste_receipt_reports_completion_once_and_drop_reports_cancellation() {
        let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let observed = events.clone();
        let receipt = FilePaste::new(move |operation| observed.borrow_mut().push(operation));
        let retained = receipt.clone();
        drop(receipt);
        assert!(events.borrow().is_empty());
        retained.complete(Some(FileTransferOperation::Move));
        let observed = events.clone();
        drop(FilePaste::new(move |operation| {
            observed.borrow_mut().push(operation)
        }));
        assert_eq!(*events.borrow(), [Some(FileTransferOperation::Move), None]);
    }
    #[test]
    fn native_names_roundtrip_without_treating_text_as_files() {
        let files = FileTransfer {
            paths: ExternalPaths(
                [PathBuf::from(if cfg!(windows) {
                    "C:/tmp/a b#%.txt"
                } else {
                    "/tmp/a b\n#%.txt"
                })]
                .into_iter()
                .collect(),
            ),
            operation: FileTransferOperation::Move,
            ownership: 19,
        };
        assert_eq!(
            FileTransfer::decode(
                &files.encode(FILE_TRANSFER_MIME).unwrap(),
                FILE_TRANSFER_MIME
            ),
            Some(files)
        );
        assert!(FileTransfer::decode(b"/tmp/file", URI_LIST_MIME).is_none());
        assert!(FileTransfer::decode(b"file://host/remote", URI_LIST_MIME).is_none());
        assert!(
            ClipboardItem::new_string("/tmp/file".into())
                .file_transfer()
                .is_none()
        );
    }
}

/// A completed native transfer. `None` means cancellation or rejection.
#[derive(Clone, Debug)]
pub struct FileTransferCompletion {
    /// The source items supplied when the transfer began.
    pub files: FileTransfer,
    /// The operation actually negotiated with the receiving application.
    pub operation: Option<FileTransferOperation>,
    /// True when the receiver owns source removal (e.g. a Shell optimized move).
    pub source_removed: bool,
}

static COMPLETIONS: std::sync::Mutex<Vec<FileTransferCompletion>> =
    std::sync::Mutex::new(Vec::new());
static ACTIVE_TRANSFERS: std::sync::Mutex<std::collections::BTreeSet<u64>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());
impl FileTransfer {
    /// Native adapters mark asynchronous extraction before releasing their drag loop.
    pub fn set_active(&self, active: bool) {
        let mut transfers = ACTIVE_TRANSFERS.lock().unwrap_or_else(|e| e.into_inner());
        if active {
            transfers.insert(self.ownership);
        } else {
            transfers.remove(&self.ownership);
        }
    }
}
impl FileTransferCompletion {
    /// Native adapters report completion after their protocol's final event.
    pub fn report(self) {
        self.files.set_active(false);
        COMPLETIONS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(self);
    }
}
impl crate::App {
    /// Whether a native receiver is still extracting this clipboard payload.
    pub fn file_transfer_is_active(&self, ownership: u64) -> bool {
        ACTIVE_TRANSFERS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&ownership)
    }

    /// Includes completed transfers whose source cleanup has not been consumed.
    pub fn has_pending_native_file_transfers(&self) -> bool {
        !ACTIVE_TRANSFERS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
            || !COMPLETIONS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
    }
    /// Drain native transfer results. Callers retain ownership tokens to match
    /// results to their source snapshots and perform any required source cleanup.
    pub fn take_file_transfer_completions(&mut self) -> Vec<FileTransferCompletion> {
        std::mem::take(&mut *COMPLETIONS.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

/// A native payload may need metadata prepared on a background worker.
pub enum ExternalDragPayloadResolution {
    /// Available without background work.
    Ready(Option<crate::ExternalDragPayload>),
    /// The pointer gesture remains cancellable while this task runs.
    Pending(crate::Task<Option<crate::ExternalDragPayload>>),
}

/// A lease on the clipboard data object that supplied a paste. Keeping this
/// object alive lets platforms report completion to the original owner even
/// when another application replaces the clipboard during the transfer.
#[derive(Clone)]
pub struct FilePaste(std::rc::Rc<FilePasteState>);

struct FilePasteState {
    completion: std::cell::RefCell<Option<Box<dyn FnOnce(Option<FileTransferOperation>)>>>,
}

impl std::fmt::Debug for FilePaste {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilePaste")
            .field("pending", &self.0.completion.borrow().is_some())
            .finish()
    }
}

impl FilePaste {
    /// Constructed by native clipboard adapters.
    pub fn new(completion: impl FnOnce(Option<FileTransferOperation>) + 'static) -> Self {
        Self(std::rc::Rc::new(FilePasteState {
            completion: std::cell::RefCell::new(Some(Box::new(completion))),
        }))
    }

    /// Report a fully completed paste. `None` preserves a cancelled or partial
    /// transfer. Move means the receiver already removed the source items.
    pub fn complete(self, operation: Option<FileTransferOperation>) {
        let completion = self.0.completion.borrow_mut().take();
        if let Some(completion) = completion {
            completion(operation);
        }
    }
}

impl Drop for FilePasteState {
    fn drop(&mut self) {
        if let Some(completion) = self.completion.get_mut().take() {
            completion(None);
        }
    }
}

/// The native drop remains pending until the receiving application finishes
/// extracting the files. Dropping this lease cancels the transfer.
#[derive(Clone, Debug)]
pub struct FileDropTransfer {
    /// The operation agreed with the source application.
    pub operation: FileTransferOperation,
    /// Whether this particular source guarantees removal after successful Move.
    /// Local file URL transfers normally delegate the move to the receiver;
    /// a protocol completion alone does not imply source-side deletion.
    pub source_owns_move: bool,
    /// A completion tied to the original offer or data object.
    pub completion: FilePaste,
}
