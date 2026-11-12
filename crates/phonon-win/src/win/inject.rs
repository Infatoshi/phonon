//! Put finished text into whatever window has focus.
//!
//! The clipboard and Control-V, as on macOS; the previous clipboard text is put
//! back afterwards, unless settings.json `clipboard_after_paste` is "dictation",
//! which leaves the dictation there. Nothing is typed key by key: Windows 11's
//! Notepad turns typed key events into "Can ??????", repeats and gaps
//! (2026-11-05 and 2026-11-12), while the same text pasted arrives intact. When
//! the clipboard cannot be used, the caller keeps the text for the user.

use std::ffi::c_void;

use anyhow::{bail, Context, Result};
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, GetClipboardData, OpenClipboard, SetClipboardData,
};
use windows_sys::Win32::System::Memory::{GlobalAlloc, GlobalLock, GlobalUnlock, GMEM_MOVEABLE};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, VK_CONTROL, VK_V,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    GetClassNameW, GetForegroundWindow, GetWindowThreadProcessId,
};

use crate::settings::ClipboardAfterPaste;

const CF_UNICODETEXT: u32 = 13;

/// Whether a paste now would land nowhere (see `pipeline::nowhere_to_paste`):
/// Control-V into the desktop, the taskbar or no window at all is dropped
/// without a word, and the clipboard restore then took the text away too.
pub fn nowhere_to_paste() -> bool {
    let window = unsafe { GetForegroundWindow() };
    if window.is_null() {
        return crate::pipeline::nowhere_to_paste("", false, true);
    }
    let mut class = [0u16; 256];
    let length = unsafe { GetClassNameW(window, class.as_mut_ptr(), class.len() as i32) };
    let class = String::from_utf16_lossy(&class[..length.max(0) as usize]);
    let mut pid = 0u32;
    unsafe { GetWindowThreadProcessId(window, &mut pid) };
    crate::pipeline::nowhere_to_paste(&class, pid == std::process::id(), false)
}

/// Insert `text` into the focused window. `after` says what the clipboard holds
/// afterwards (settings.json `clipboard_after_paste`). Returns the path that was used.
pub fn insert(text: &str, after: ClipboardAfterPaste) -> Result<&'static str> {
    if text.is_empty() {
        return Ok("skipped");
    }
    // The clipboard work runs on its own thread with a time limit: GetClipboardData
    // and EmptyClipboard wait on the clipboard's current owner without one, and an
    // owner that stops answering held the dictation until it recovered (25 s in the
    // Windows 11 QA VM with a hung OLE owner, 2026-11-12). Past the limit the
    // caller keeps the text for the user (tray: Copy last dictation) and says so.
    // Phonon does not type it instead: typed key events reached Windows 11 Notepad
    // with characters lost or changed in 2 of 6 idle trials (Unicode events) and
    // 2 of 2 under CPU load (real key presses), where a paste is exact. The stuck thread
    // may still put the dictation on the clipboard later; nothing presses
    // Control-V after this returns, so it cannot paste late.
    let owned = text.to_string();
    match bounded(CLIPBOARD_WAIT, move || load_clipboard(&owned, after)) {
        Some(Ok(previous)) => {
            press_paste().context("send Control-V")?;
            if let Some(previous) = after.to_restore(previous) {
                // The target window reads the clipboard on its own schedule.
                // Restoring it instantly would race the paste.
                std::thread::sleep(std::time::Duration::from_millis(250));
                let _ = set_clipboard_text(&previous);
            }
            Ok("pasted")
        }
        Some(Err(error)) => Err(error),
        None => Err(anyhow::anyhow!(
            "the clipboard did not answer within {} ms",
            CLIPBOARD_WAIT.as_millis()
        )),
    }
}

/// Read the text to restore later (unless the user keeps the dictation on the
/// clipboard), then put `text` on the clipboard. Returns the text to restore.
fn load_clipboard(text: &str, after: ClipboardAfterPaste) -> Result<Option<String>> {
    let previous = match after {
        ClipboardAfterPaste::Restore => read_clipboard_text(),
        ClipboardAfterPaste::Dictation => None,
    };
    set_clipboard_text(text).context("write the clipboard")?;
    Ok(previous)
}

/// How long reading the previous clipboard and writing the dictation may take
/// together. Both normally finish in a few milliseconds; a clipboard manager or
/// remote-desktop client that holds it open briefly gets a few seconds.
const CLIPBOARD_WAIT: std::time::Duration = std::time::Duration::from_millis(3000);

/// Run `work` on its own thread and give up waiting after `wait`. A thread
/// that is still stuck is left behind.
fn bounded<T: Send + 'static>(
    wait: std::time::Duration,
    work: impl FnOnce() -> T + Send + 'static,
) -> Option<T> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = sender.send(work());
    });
    receiver.recv_timeout(wait).ok()
}

/// Leave `text` on the clipboard, for the user to paste (the tray's recent list).
/// Bounded like the paste: a hung clipboard owner must not hold up the caller.
pub fn copy(text: &str) -> Result<()> {
    let owned = text.to_string();
    match bounded(COPY_WAIT, move || set_clipboard_text(&owned)) {
        Some(result) => result.context("write the clipboard"),
        None => bail!(
            "the clipboard did not answer within {} ms",
            COPY_WAIT.as_millis()
        ),
    }
}

const COPY_WAIT: std::time::Duration = std::time::Duration::from_millis(1000);

/// Open the clipboard, retrying for up to 2 s: another process may hold it
/// (the paste as a whole is bounded by `CLIPBOARD_WAIT`).
fn open_clipboard() -> Result<()> {
    for _ in 0..80 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn a_read_that_never_answers_is_abandoned_in_time() {
        let started = Instant::now();
        let result = bounded(Duration::from_millis(100), || {
            std::thread::sleep(Duration::from_secs(30));
            7
        });
        assert_eq!(result, None);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn a_read_that_answers_is_kept() {
        assert_eq!(
            bounded(CLIPBOARD_WAIT, || Some("before".to_string())),
            Some(Some("before".to_string()))
        );
    }
}
