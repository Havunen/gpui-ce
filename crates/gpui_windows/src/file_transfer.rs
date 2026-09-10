//! Shell data object wrapper retaining transfer-result formats for the caller.
use gpui::{FileTransfer, FileTransferCompletion, FileTransferOperation};
use std::{cell::RefCell, os::windows::ffi::OsStrExt, rc::Rc, sync::LazyLock};
use windows::{
    Win32::{
        Foundation::{E_INVALIDARG, E_OUTOFMEMORY, GlobalFree, HWND},
        System::{Com::*, DataExchange::RegisterClipboardFormatW, Memory::*, Ole::*},
        UI::Shell::*,
    },
    core::{BOOL, HRESULT, PCWSTR, Ref, Result, implement},
};

static PREFERRED: LazyLock<u16> = LazyLock::new(|| unsafe {
    RegisterClipboardFormatW(windows::core::w!("Preferred DropEffect")) as u16
});
static PERFORMED: LazyLock<u16> = LazyLock::new(|| unsafe {
    RegisterClipboardFormatW(windows::core::w!("Performed DropEffect")) as u16
});
static LOGICAL: LazyLock<u16> = LazyLock::new(|| unsafe {
    RegisterClipboardFormatW(windows::core::w!("Logical Performed DropEffect")) as u16
});
static PASTED: LazyLock<u16> = LazyLock::new(|| unsafe {
    RegisterClipboardFormatW(windows::core::w!("Paste Succeeded")) as u16
});
static PRIVATE: LazyLock<u16> = LazyLock::new(|| unsafe {
    RegisterClipboardFormatW(windows::core::w!("application/x-gpui-file-transfer")) as u16
});
static TARGET_CLSID: LazyLock<u16> =
    LazyLock::new(|| unsafe { RegisterClipboardFormatW(windows::core::w!("TargetCLSID")) as u16 });

#[derive(Default)]
struct TransferResult {
    performed: Option<u32>,
    logical: Option<u32>,
    reported: bool,
    async_mode: bool,
    in_operation: bool,
    pasted: Option<u32>,
    drag_effect: Option<u32>,
    recycle_bin: bool,
}

#[implement(IDataObject, IDataObjectAsyncCapability)]
struct FileDataObject {
    inner: IDataObject,
    files: FileTransfer,
    result: Rc<RefCell<TransferResult>>,
    clipboard: bool,
}

#[allow(non_snake_case)]
impl IDataObject_Impl for FileDataObject_Impl {
    fn GetData(&self, f: *const FORMATETC) -> Result<STGMEDIUM> {
        unsafe { self.inner.GetData(f) }
    }
    fn GetDataHere(&self, f: *const FORMATETC, m: *mut STGMEDIUM) -> Result<()> {
        unsafe { self.inner.GetDataHere(f, m) }
    }
    fn QueryGetData(&self, f: *const FORMATETC) -> HRESULT {
        unsafe { self.inner.QueryGetData(f) }
    }
    fn GetCanonicalFormatEtc(&self, f: *const FORMATETC, out: *mut FORMATETC) -> HRESULT {
        unsafe { self.inner.GetCanonicalFormatEtc(f, out) }
    }
    fn EnumFormatEtc(&self, direction: u32) -> Result<IEnumFORMATETC> {
        unsafe { self.inner.EnumFormatEtc(direction) }
    }
    fn DAdvise(&self, f: *const FORMATETC, flags: u32, sink: Ref<IAdviseSink>) -> Result<u32> {
        unsafe { self.inner.DAdvise(f, flags, sink.as_ref()) }
    }
    fn DUnadvise(&self, connection: u32) -> Result<()> {
        unsafe { self.inner.DUnadvise(connection) }
    }
    fn EnumDAdvise(&self) -> Result<IEnumSTATDATA> {
        unsafe { self.inner.EnumDAdvise() }
    }
    fn SetData(&self, f: *const FORMATETC, medium: *const STGMEDIUM, release: BOOL) -> Result<()> {
        if f.is_null() || medium.is_null() {
            return Err(E_INVALIDARG.into());
        }
        // The caller owns valid FORMATETC/STGMEDIUM arguments for this COM call.
        // Read before forwarding: the underlying object may consume the medium.
        let format = unsafe { (*f).cfFormat };
        let recycle_bin = unsafe {
            if format == *TARGET_CLSID
                && (*medium).tymed == TYMED_HGLOBAL.0 as u32
                && GlobalSize((*medium).u.hGlobal) >= std::mem::size_of::<windows::core::GUID>()
            {
                let ptr = GlobalLock((*medium).u.hGlobal);
                if ptr.is_null() {
                    false
                } else {
                    let clsid = std::ptr::read_unaligned(ptr.cast::<windows::core::GUID>());
                    let _ = GlobalUnlock((*medium).u.hGlobal);
                    clsid == CLSID_RecycleBin
                }
            } else {
                false
            }
        };
        let value = unsafe {
            if (*medium).tymed == TYMED_HGLOBAL.0 as u32 && GlobalSize((*medium).u.hGlobal) >= 4 {
                let ptr = GlobalLock((*medium).u.hGlobal);
                if ptr.is_null() {
                    None
                } else {
                    let value = std::ptr::read_unaligned(ptr.cast::<u32>());
                    let _ = GlobalUnlock((*medium).u.hGlobal);
                    Some(value)
                }
            } else {
                None
            }
        };
        unsafe {
            self.inner.SetData(f, medium, release.as_bool())?;
        }
        if let Some(value) = value {
            let mut state = self.result.borrow_mut();
            state.recycle_bin |= recycle_bin;
            if format == *PERFORMED {
                state.performed = Some(value);
            }
            if format == *LOGICAL {
                state.logical = Some(value);
            }
            if format == *PASTED {
                state.pasted = Some(value);
            }
            if self.clipboard && format == *PASTED && !state.reported && !state.in_operation {
                state.reported = true;
                let operation = if value != 0 && state.recycle_bin || value == DROPEFFECT_MOVE.0 {
                    Some(FileTransferOperation::Move)
                } else if value == DROPEFFECT_COPY.0 {
                    Some(FileTransferOperation::Copy)
                } else {
                    None
                };
                // Both PasteSucceeded=MOVE and Performed=MOVE are required
                // before a clipboard source must delete its originals.
                FileTransferCompletion {
                    files: self.files.clone(),
                    operation,
                    source_removed: !state.recycle_bin
                        && state.performed != Some(DROPEFFECT_MOVE.0),
                }
                .report();
            }
        }
        Ok(())
    }
}

fn transfer_operation(effect: u32) -> Option<FileTransferOperation> {
    if effect == DROPEFFECT_MOVE.0 {
        Some(FileTransferOperation::Move)
    } else if effect == DROPEFFECT_COPY.0 {
        Some(FileTransferOperation::Copy)
    } else {
        None
    }
}

#[allow(non_snake_case)]
impl IDataObjectAsyncCapability_Impl for FileDataObject_Impl {
    fn SetAsyncMode(&self, enabled: BOOL) -> Result<()> {
        self.result.borrow_mut().async_mode = enabled.as_bool();
        Ok(())
    }
    fn GetAsyncMode(&self) -> Result<BOOL> {
        Ok(self.result.borrow().async_mode.into())
    }
    fn StartOperation(&self, _: Ref<IBindCtx>) -> Result<()> {
        self.result.borrow_mut().in_operation = true;
        self.files.set_active(true);
        Ok(())
    }
    fn InOperation(&self) -> Result<BOOL> {
        Ok(self.result.borrow().in_operation.into())
    }
    fn EndOperation(&self, result: HRESULT, _: Ref<IBindCtx>, effects: u32) -> Result<()> {
        let mut state = self.result.borrow_mut();
        state.in_operation = false;
        self.files.set_active(false);
        if !state.reported {
            state.reported = true;
            let logical = if self.clipboard {
                state.pasted
            } else {
                state.logical.or(state.drag_effect).or(Some(effects))
            };
            let operation = result
                .is_ok()
                .then(|| {
                    if state.recycle_bin && logical.is_some_and(|effect| effect != 0) {
                        Some(FileTransferOperation::Move)
                    } else {
                        logical.and_then(transfer_operation)
                    }
                })
                .flatten();
            FileTransferCompletion {
                files: self.files.clone(),
                operation,
                source_removed: !state.recycle_bin && effects != DROPEFFECT_MOVE.0,
            }
            .report();
        }
        Ok(())
    }
}

impl Drop for FileDataObject {
    fn drop(&mut self) {
        if !self.result.borrow().reported {
            FileTransferCompletion {
                files: self.files.clone(),
                operation: None,
                source_removed: false,
            }
            .report();
        }
    }
}

fn set_bytes(object: &IDataObject, format: u16, bytes: &[u8]) -> Result<()> {
    unsafe {
        let memory = GlobalAlloc(GMEM_MOVEABLE, bytes.len())?;
        let ptr = GlobalLock(memory);
        if ptr.is_null() {
            let _ = GlobalFree(Some(memory));
            return Err(windows::core::Error::from(E_OUTOFMEMORY));
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len());
        let _ = GlobalUnlock(memory);
        let format = FORMATETC {
            cfFormat: format,
            ptd: std::ptr::null_mut(),
            dwAspect: DVASPECT_CONTENT.0,
            lindex: -1,
            tymed: TYMED_HGLOBAL.0 as u32,
        };
        let medium = STGMEDIUM {
            tymed: TYMED_HGLOBAL.0 as u32,
            u: STGMEDIUM_0 { hGlobal: memory },
            pUnkForRelease: std::mem::ManuallyDrop::new(None),
        };
        if let Err(error) = object.SetData(&format, &medium, true) {
            let _ = GlobalFree(Some(memory));
            return Err(error);
        }
    }
    Ok(())
}

fn data_object(
    files: FileTransfer,
    clipboard: bool,
) -> Result<(IDataObject, Rc<RefCell<TransferResult>>)> {
    struct IdLists(Vec<*mut Common::ITEMIDLIST>);
    impl Drop for IdLists {
        fn drop(&mut self) {
            for list in &self.0 {
                unsafe { CoTaskMemFree(Some((*list).cast())) };
            }
        }
    }
    let mut lists = IdLists(Vec::new());
    for path in files.paths.paths() {
        let mut native: Vec<u16> = path.as_os_str().encode_wide().collect();
        if native.contains(&0) {
            return Err(E_INVALIDARG.into());
        }
        native.push(0);
        let mut pidl = std::ptr::null_mut();
        unsafe {
            SHParseDisplayName(PCWSTR(native.as_ptr()), None, &mut pidl, 0, None)?;
        }
        lists.0.push(pidl);
    }
    let items = unsafe {
        SHCreateShellItemArrayFromIDLists(
            &lists.0.iter().map(|p| p.cast_const()).collect::<Vec<_>>(),
        )?
    };
    let inner: IDataObject = unsafe { items.BindToHandler(None, &BHID_DataObject)? };
    let preferred = if files.operation == FileTransferOperation::Move {
        DROPEFFECT_MOVE
    } else {
        DROPEFFECT_COPY
    };
    set_bytes(&inner, *PREFERRED, &preferred.0.to_le_bytes())?;
    if let Some(bytes) = files.encode(gpui::FILE_TRANSFER_MIME) {
        set_bytes(&inner, *PRIVATE, &bytes)?;
    }
    let result = Rc::new(RefCell::new(TransferResult {
        async_mode: true,
        ..Default::default()
    }));
    let object = FileDataObject {
        inner,
        files,
        result: result.clone(),
        clipboard,
    }
    .into();
    Ok((object, result))
}

pub(crate) fn write_files(files: FileTransfer) -> Result<()> {
    let (object, _) = data_object(files, true)?;
    unsafe { OleSetClipboard(&object) }
}

pub(crate) fn drag_files(window: HWND, files: FileTransfer) -> Option<FileTransferCompletion> {
    let mut completion = FileTransferCompletion {
        files: files.clone(),
        operation: None,
        source_removed: false,
    };
    if let Ok((object, result)) = data_object(files.clone(), false) {
        let allowed = if files.operation == FileTransferOperation::Move {
            DROPEFFECT_COPY | DROPEFFECT_MOVE
        } else {
            DROPEFFECT_COPY
        };
        if let Ok(effect) = unsafe { SHDoDragDrop(Some(window), &object, None, allowed) } {
            let mut state = result.borrow_mut();
            state.drag_effect = Some(effect.0);
            if state.in_operation || state.reported {
                return None;
            }
            let logical = state.logical.unwrap_or(effect.0);
            completion.operation =
                if logical != 0 && state.recycle_bin || logical == DROPEFFECT_MOVE.0 {
                    Some(FileTransferOperation::Move)
                } else if logical == DROPEFFECT_COPY.0 {
                    Some(FileTransferOperation::Copy)
                } else {
                    None
                };
            completion.source_removed = !state.recycle_bin
                && (effect != DROPEFFECT_MOVE
                    || state.performed.is_some_and(|p| p != DROPEFFECT_MOVE.0));
        }
    }
    Some(completion)
}

/// Shell completion is delivered to the captured IDataObject, never to whichever
/// application happens to own the clipboard after the filesystem worker ends.
pub(crate) fn capture_paste(files: &FileTransfer) -> Option<gpui::FilePaste> {
    use windows::Win32::System::{DataExchange::GetClipboardSequenceNumber, Ole::OleGetClipboard};
    let sequence = unsafe { GetClipboardSequenceNumber() };
    let object = unsafe { OleGetClipboard() }.ok()?;
    if crate::clipboard::read_from_clipboard()?
        .file_transfer()
        .as_ref()
        != Some(files)
        || unsafe { GetClipboardSequenceNumber() } != sequence
    {
        return None;
    }
    Some(gpui::FilePaste::new(move |operation| {
        let Some(operation) = operation else {
            return;
        };
        let logical = if operation == FileTransferOperation::Move {
            DROPEFFECT_MOVE.0
        } else {
            DROPEFFECT_COPY.0
        };
        // Our filesystem service has already moved the originals. Reporting
        // an optimized move prevents the source from deleting them again.
        let performed = if operation == FileTransferOperation::Move {
            0u32
        } else {
            DROPEFFECT_COPY.0
        };
        let result = set_bytes(&object, *PERFORMED, &performed.to_le_bytes())
            .and_then(|_| set_bytes(&object, *LOGICAL, &logical.to_le_bytes()))
            .and_then(|_| set_bytes(&object, *PASTED, &logical.to_le_bytes()));
        if let Err(error) = result {
            log::error!("Could not report file paste completion: {error}");
        }
    }))
}

/// Retain the Shell source until asynchronous file extraction finishes. Our
/// receiver performs an optimized move and must never ask it to delete again.
pub(crate) fn capture_drop(
    object: &IDataObject,
    operation: FileTransferOperation,
) -> gpui::FileDropTransfer {
    use windows::core::Interface;
    let object = object.clone();
    let asynchronous =
        object
            .cast::<IDataObjectAsyncCapability>()
            .ok()
            .filter(|capability| unsafe {
                capability
                    .GetAsyncMode()
                    .is_ok_and(|enabled| enabled.as_bool())
                    && capability.StartOperation(None).is_ok()
            });
    gpui::FileDropTransfer {
        operation,
        source_owns_move: false,
        completion: gpui::FilePaste::new(move |completed| {
            let logical = completed.map_or(0, |operation| {
                if operation == FileTransferOperation::Move {
                    DROPEFFECT_MOVE.0
                } else {
                    DROPEFFECT_COPY.0
                }
            });
            let performed = if completed == Some(FileTransferOperation::Copy) {
                DROPEFFECT_COPY.0
            } else {
                0
            };
            let result = set_bytes(&object, *PERFORMED, &performed.to_le_bytes())
                .and_then(|_| set_bytes(&object, *LOGICAL, &logical.to_le_bytes()));
            if let Err(error) = result {
                log::error!("Could not report file drop completion: {error}");
            }
            if let Some(capability) = asynchronous {
                let result = if completed.is_some() {
                    windows::Win32::Foundation::S_OK
                } else {
                    windows::Win32::Foundation::E_ABORT
                };
                if let Err(error) = unsafe { capability.EndOperation(result, None, performed) } {
                    log::error!("Could not finish asynchronous file drop: {error}");
                }
            }
        }),
    }
}
