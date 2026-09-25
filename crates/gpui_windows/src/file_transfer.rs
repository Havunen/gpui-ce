//! Shell data object wrapper retaining transfer-result formats for the caller.
use crate::bindings::Windows::Win32::*;
use gpui::{FileTransfer, FileTransferCompletion, FileTransferOperation};
use std::{cell::RefCell, os::windows::ffi::OsStrExt, rc::Rc, sync::LazyLock};
use windows_core::{BOOL, Error, GUID, HRESULT, Interface, PCWSTR, Ref, Result, implement};

const EFFECT_COPY: u32 = DROPEFFECT_COPY as u32;
const EFFECT_MOVE: u32 = DROPEFFECT_MOVE as u32;

static PREFERRED: LazyLock<u16> = LazyLock::new(|| unsafe {
    RegisterClipboardFormatW(windows_core::w!("Preferred DropEffect")) as u16
});
static PERFORMED: LazyLock<u16> = LazyLock::new(|| unsafe {
    RegisterClipboardFormatW(windows_core::w!("Performed DropEffect")) as u16
});
static LOGICAL: LazyLock<u16> = LazyLock::new(|| unsafe {
    RegisterClipboardFormatW(windows_core::w!("Logical Performed DropEffect")) as u16
});
static PASTED: LazyLock<u16> = LazyLock::new(|| unsafe {
    RegisterClipboardFormatW(windows_core::w!("Paste Succeeded")) as u16
});
static PRIVATE: LazyLock<u16> = LazyLock::new(|| unsafe {
    RegisterClipboardFormatW(windows_core::w!("application/x-gpui-file-transfer")) as u16
});
static TARGET_CLSID: LazyLock<u16> =
    LazyLock::new(|| unsafe { RegisterClipboardFormatW(windows_core::w!("TargetCLSID")) as u16 });

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

/// Forwards an inner result unchanged. `HRESULT::ok` would collapse success
/// codes such as `S_FALSE` or `DATA_S_SAMEFORMATETC` into `S_OK`.
fn forward(result: HRESULT) -> Result<()> {
    if result == S_OK {
        Ok(())
    } else {
        Err(Error::from_hresult(result))
    }
}

#[allow(non_snake_case)]
impl IDataObject_Impl for FileDataObject_Impl {
    fn GetData(&self, f: *const FORMATETC) -> Result<STGMEDIUM> {
        unsafe { self.inner.GetData(f) }
    }
    fn GetDataHere(&self, f: *const FORMATETC, m: *mut STGMEDIUM) -> Result<()> {
        forward(unsafe { self.inner.GetDataHere(f, m) })
    }
    fn QueryGetData(&self, f: *const FORMATETC) -> Result<()> {
        forward(unsafe { self.inner.QueryGetData(f) })
    }
    fn GetCanonicalFormatEtc(&self, f: *const FORMATETC, out: *mut FORMATETC) -> Result<()> {
        forward(unsafe { self.inner.GetCanonicalFormatEtc(f, out) })
    }
    fn EnumFormatEtc(&self, direction: u32) -> Result<IEnumFORMATETC> {
        unsafe { self.inner.EnumFormatEtc(direction) }
    }
    fn DAdvise(&self, f: *const FORMATETC, flags: u32, sink: Ref<IAdviseSink>) -> Result<u32> {
        unsafe { self.inner.DAdvise(f, flags, sink.as_ref()) }
    }
    fn DUnadvise(&self, connection: u32) -> Result<()> {
        forward(unsafe { self.inner.DUnadvise(connection) })
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
        let format = unsafe { (*f).cfFormat.0 };
        let recycle_bin = unsafe {
            if format == *TARGET_CLSID
                && (*medium).tymed == TYMED_HGLOBAL as u32
                && GlobalSize((*medium).Anonymous.hGlobal) >= std::mem::size_of::<GUID>()
            {
                let ptr = GlobalLock((*medium).Anonymous.hGlobal);
                if ptr.is_null() {
                    false
                } else {
                    let clsid = std::ptr::read_unaligned(ptr.cast::<GUID>());
                    let _ = GlobalUnlock((*medium).Anonymous.hGlobal);
                    clsid == CLSID_RecycleBin
                }
            } else {
                false
            }
        };
        let value = unsafe {
            if (*medium).tymed == TYMED_HGLOBAL as u32
                && GlobalSize((*medium).Anonymous.hGlobal) >= 4
            {
                let ptr = GlobalLock((*medium).Anonymous.hGlobal);
                if ptr.is_null() {
                    None
                } else {
                    let value = std::ptr::read_unaligned(ptr.cast::<u32>());
                    let _ = GlobalUnlock((*medium).Anonymous.hGlobal);
                    Some(value)
                }
            } else {
                None
            }
        };
        unsafe {
            forward(self.inner.SetData(f, medium, release.as_bool()))?;
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
                let operation = if value != 0 && state.recycle_bin || value == EFFECT_MOVE {
                    Some(FileTransferOperation::Move)
                } else if value == EFFECT_COPY {
                    Some(FileTransferOperation::Copy)
                } else {
                    None
                };
                // Both PasteSucceeded=MOVE and Performed=MOVE are required
                // before a clipboard source must delete its originals.
                FileTransferCompletion {
                    files: self.files.clone(),
                    operation,
                    source_removed: !state.recycle_bin && state.performed != Some(EFFECT_MOVE),
                }
                .report();
            }
        }
        Ok(())
    }
}

fn transfer_operation(effect: u32) -> Option<FileTransferOperation> {
    if effect == EFFECT_MOVE {
        Some(FileTransferOperation::Move)
    } else if effect == EFFECT_COPY {
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
                source_removed: !state.recycle_bin && effects != EFFECT_MOVE,
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
        let memory = GlobalAlloc(GMEM_MOVEABLE as u32, bytes.len());
        if memory.is_invalid() {
            return Err(Error::from_thread());
        }
        let ptr = GlobalLock(memory);
        if ptr.is_null() {
            let _ = GlobalFree(memory);
            return Err(E_OUTOFMEMORY.into());
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len());
        let _ = GlobalUnlock(memory);
        let format = FORMATETC {
            cfFormat: CLIPFORMAT(format),
            ptd: std::ptr::null_mut(),
            dwAspect: DVASPECT_CONTENT as u32,
            lindex: -1,
            tymed: TYMED_HGLOBAL as u32,
        };
        let medium = STGMEDIUM {
            tymed: TYMED_HGLOBAL as u32,
            Anonymous: uSTGMEDIUM_0 { hGlobal: memory },
            pUnkForRelease: std::mem::ManuallyDrop::new(None),
        };
        if let Err(error) = object.SetData(&format, &medium, true).ok() {
            let _ = GlobalFree(memory);
            return Err(error);
        }
    }
    Ok(())
}

fn data_object(
    files: FileTransfer,
    clipboard: bool,
) -> Result<(IDataObject, Rc<RefCell<TransferResult>>)> {
    struct IdLists(Vec<LPITEMIDLIST>);
    impl Drop for IdLists {
        fn drop(&mut self) {
            for list in &self.0 {
                unsafe { CoTaskMemFree(list.cast()) };
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
            SHParseDisplayName(
                PCWSTR(native.as_ptr()),
                None::<&IBindCtx>,
                &mut pidl,
                SFGAOF(0),
                None,
            )
            .ok()?;
        }
        lists.0.push(pidl);
    }
    let items = unsafe {
        SHCreateShellItemArrayFromIDLists(
            &lists.0.iter().map(|p| p.cast_const()).collect::<Vec<_>>(),
        )?
    };
    let inner: IDataObject = unsafe { items.BindToHandler(None::<&IBindCtx>, &BHID_DataObject)? };
    let preferred = if files.operation == FileTransferOperation::Move {
        EFFECT_MOVE
    } else {
        EFFECT_COPY
    };
    set_bytes(&inner, *PREFERRED, &preferred.to_le_bytes())?;
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
    unsafe { OleSetClipboard(&object).ok() }
}

pub(crate) fn drag_files(window: HWND, files: FileTransfer) -> Option<FileTransferCompletion> {
    let mut completion = FileTransferCompletion {
        files: files.clone(),
        operation: None,
        source_removed: false,
    };
    if let Ok((object, result)) = data_object(files.clone(), false) {
        let allowed = if files.operation == FileTransferOperation::Move {
            EFFECT_COPY | EFFECT_MOVE
        } else {
            EFFECT_COPY
        };
        if let Ok(effect) =
            unsafe { SHDoDragDrop(Some(window), &object, None::<&IDropSource>, allowed) }
        {
            let mut state = result.borrow_mut();
            state.drag_effect = Some(effect);
            if state.in_operation || state.reported {
                return None;
            }
            let logical = state.logical.unwrap_or(effect);
            completion.operation = if logical != 0 && state.recycle_bin || logical == EFFECT_MOVE {
                Some(FileTransferOperation::Move)
            } else if logical == EFFECT_COPY {
                Some(FileTransferOperation::Copy)
            } else {
                None
            };
            completion.source_removed = !state.recycle_bin
                && (effect != EFFECT_MOVE || state.performed.is_some_and(|p| p != EFFECT_MOVE));
        }
    }
    Some(completion)
}

/// Shell completion is delivered to the captured IDataObject, never to whichever
/// application happens to own the clipboard after the filesystem worker ends.
pub(crate) fn capture_paste(files: &FileTransfer) -> Option<gpui::FilePaste> {
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
            EFFECT_MOVE
        } else {
            EFFECT_COPY
        };
        // Our filesystem service has already moved the originals. Reporting
        // an optimized move prevents the source from deleting them again.
        let performed = if operation == FileTransferOperation::Move {
            0u32
        } else {
            EFFECT_COPY
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
    let object = object.clone();
    let asynchronous =
        object
            .cast::<IDataObjectAsyncCapability>()
            .ok()
            .filter(|capability| unsafe {
                capability
                    .GetAsyncMode()
                    .is_ok_and(|enabled| enabled.as_bool())
                    && capability.StartOperation(None::<&IBindCtx>).is_ok()
            });
    gpui::FileDropTransfer {
        operation,
        source_owns_move: false,
        completion: gpui::FilePaste::new(move |completed| {
            let logical = completed.map_or(0, |operation| {
                if operation == FileTransferOperation::Move {
                    EFFECT_MOVE
                } else {
                    EFFECT_COPY
                }
            });
            let performed = if completed == Some(FileTransferOperation::Copy) {
                EFFECT_COPY
            } else {
                0
            };
            let result = set_bytes(&object, *PERFORMED, &performed.to_le_bytes())
                .and_then(|_| set_bytes(&object, *LOGICAL, &logical.to_le_bytes()));
            if let Err(error) = result {
                log::error!("Could not report file drop completion: {error}");
            }
            if let Some(capability) = asynchronous {
                let result = if completed.is_some() { S_OK } else { E_ABORT };
                if let Err(error) =
                    unsafe { capability.EndOperation(result, None::<&IBindCtx>, performed) }.ok()
                {
                    log::error!("Could not finish asynchronous file drop: {error}");
                }
            }
        }),
    }
}
