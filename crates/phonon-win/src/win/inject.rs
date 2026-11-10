//! Put finished text into whatever window has focus.
//!
//! The clipboard and Control-V, as on macOS; the previous clipboard text is put
//! back afterwards. Typing with `SendInput` Unicode key events is only the
//! fallback for when the clipboard cannot be used: Windows 11's Notepad turns a
//! burst of those events into "Can ??????" (everything after the first word
//! became question marks, 2026-11-05 on 26100), while the same text pasted, or
//! typed one character per call with a pause, arrives intact.

use std::ffi::c_void;

use anyhow::{bail, Context, Result};
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, GetClipboardData, OpenClipboard, SetClipboardData,
};
use windows_sys::Win32::System::Memory::{GlobalAlloc, GlobalLock, GlobalUnlock, GMEM_MOVEABLE};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, VK_CONTROL,
    VK_V,
};

const CF_UNICODETEXT: u32 = 13;

/// Insert `text` into the focused window. Returns the path that was used.
pub fn insert(text: &str) -> Result<&'static str> {
    if text.is_empty() {
        return Ok("skipped");
    }
    match paste_text(text) {
        Ok(()) => Ok("pasted"),
        Err(error) => {
            eprintln!("phonon: pasting failed, typing instead: {error:#}");
            let units: Vec<u16> = text.encode_utf16().collect();
            type_text(&units)?;
            Ok("typed")
        }
    }
}

/// Send every code unit as a Unicode key press and release, one character per
/// call with a short pause, which is what Windows 11 Notepad needs to keep them.
fn type_text(units: &[u16]) -> Result<()> {
    for unit in units {
        type_unit(*unit)?;
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    Ok(())
}

fn type_unit(unit: u16) -> Result<()> {
    let mut events: Vec<INPUT> = Vec::with_capacity(2);
    for up in [false, true] {
        let mut input: INPUT = unsafe { std::mem::zeroed() };
        input.r#type = INPUT_KEYBOARD;
        input.Anonymous.ki = KEYBDINPUT {
            wVk: 0,
            wScan: unit,
            dwFlags: KEYEVENTF_UNICODE | if up { KEYEVENTF_KEYUP } else { 0 },
            time: 0,
            dwExtraInfo: 0,
        };
        events.push(input);
    }
    let sent = unsafe {
        SendInput(
            events.len() as u32,
            events.as_ptr(),
            std::mem::size_of::<INPUT>() as i32,
        )
    };
    if sent as usize != events.len() {
        bail!("SendInput accepted {sent} of {} events", events.len());
    }
    Ok(())
}

/// Put `text` on the clipboard, press Control-V, then put the old text back.
fn paste_text(text: &str) -> Result<()> {
    let previous = read_clipboard_text();
    set_clipboard_text(text).context("write the clipboard")?;
    press_paste().context("send Control-V")?;
    // The target window reads the clipboard on its own schedule. Restoring it
    // instantly would race the paste.
    std::thread::sleep(std::time::Duration::from_millis(250));
    if let Some(previous) = previous {
        let _ = set_clipboard_text(&previous);
    }
    Ok(())
}

/// Leave `text` on the clipboard, for the user to paste (the tray's recent list).
pub fn copy(text: &str) -> Result<()> {
    set_clipboard_text(text).context("write the clipboard")
}

/// Open the clipboard, retrying briefly: another process may hold it.
fn open_clipboard() -> Result<()> {
    for _ in 0..20 {
        if unsafe { OpenClipboard(std::ptr::null_mut()) } != 0 {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    bail!("another program is holding the clipboard")
}

fn read_clipboard_text() -> Option<String> {
    if open_clipboard().is_err() {
        return None;
    }
    let text = unsafe {
        let handle: HANDLE = GetClipboardData(CF_UNICODETEXT);
        if handle.is_null() {
            None
        } else {
            let pointer = GlobalLock(handle).cast::<u16>();
            if pointer.is_null() {
                None
            } else {
                let mut length = 0usize;
                while *pointer.add(length) != 0 {
                    length += 1;
                }
                let slice = std::slice::from_raw_parts(pointer, length);
                let value = String::from_utf16_lossy(slice);
                let _ = GlobalUnlock(handle);
                Some(value)
            }
        }
    };
    unsafe { CloseClipboard() };
    text
}

fn set_clipboard_text(text: &str) -> Result<()> {
    let mut units: Vec<u16> = text.encode_utf16().collect();
    units.push(0);
    let bytes = units.len() * std::mem::size_of::<u16>();

    open_clipboard()?;
    let result = unsafe {
        if EmptyClipboard() == 0 {
            Err(anyhow::anyhow!("EmptyClipboard failed"))
        } else {
            let handle = GlobalAlloc(GMEM_MOVEABLE, bytes);
            if handle.is_null() {
                Err(anyhow::anyhow!("GlobalAlloc failed"))
            } else {
                let pointer = GlobalLock(handle);
                if pointer.is_null() {
                    Err(anyhow::anyhow!("GlobalLock failed"))
                } else {
                    std::ptr::copy_nonoverlapping(units.as_ptr() as *const c_void, pointer, bytes);
                    let _ = GlobalUnlock(handle);
                    // The clipboard owns the block once SetClipboardData succeeds.
                    if SetClipboardData(CF_UNICODETEXT, handle).is_null() {
                        Err(anyhow::anyhow!("SetClipboardData failed"))
                    } else {
                        Ok(())
                    }
                }
            }
        }
    };
    unsafe { CloseClipboard() };
    result
}

fn press_paste() -> Result<()> {
    let key = |code: u16, up: bool| {
        let mut input: INPUT = unsafe { std::mem::zeroed() };
        input.r#type = INPUT_KEYBOARD;
        input.Anonymous.ki = KEYBDINPUT {
            wVk: code,
            wScan: 0,
            dwFlags: if up { KEYEVENTF_KEYUP } else { 0 },
            time: 0,
            dwExtraInfo: 0,
        };
        input
    };
    let events = [
        key(VK_CONTROL, false),
        key(VK_V, false),
        key(VK_V, true),
        key(VK_CONTROL, true),
    ];
    let sent = unsafe {
        SendInput(
            events.len() as u32,
            events.as_ptr(),
            std::mem::size_of::<INPUT>() as i32,
        )
    };
    if sent as usize != events.len() {
        bail!("SendInput accepted {sent} of {} paste events", events.len());
    }
    Ok(())
}
