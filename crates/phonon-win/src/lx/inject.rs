//! Put finished text into the focused window on Linux: clipboard plus Control-V.
//!
//! Wayland lets no ordinary client send keys to another, so the keystroke comes
//! from a uinput virtual keyboard, which every compositor and X server treats as
//! real hardware. The clipboard goes through arboard (Wayland data-control where
//! the compositor offers it, otherwise X11, which GNOME and KDE bridge to their
//! Wayland clipboard). The previous clipboard text is put back afterwards, unless
//! settings.json `clipboard_after_paste` is "dictation", which leaves the dictation there.
//! Terminals paste with Control-Shift-V, so in a terminal the text stays on the
//! clipboard instead of appearing.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};

use super::hook::KEY_LEFTCTRL;
use crate::settings::ClipboardAfterPaste;

/// Phonon's virtual keyboard, which the key reader skips.
pub const VIRTUAL_KEYBOARD_NAME: &str = "Phonon virtual keyboard";
const KEY_V: u16 = 47;
const EV_SYN: u16 = 0;
const EV_KEY: u16 = 1;
const SYN_REPORT: u16 = 0;

// ioctl numbers from linux/uinput.h
const UI_SET_EVBIT: libc::c_ulong = 0x4004_5564;
const UI_SET_KEYBIT: libc::c_ulong = 0x4004_5565;
const UI_DEV_SETUP: libc::c_ulong = 0x405c_5503;
const UI_DEV_CREATE: libc::c_ulong = 0x5501;

#[repr(C)]
struct UinputSetup {
    bustype: u16,
    vendor: u16,
    product: u16,
    version: u16,
    name: [u8; 80],
    ff_effects_max: u32,
}

static KEYBOARD: OnceLock<Mutex<File>> = OnceLock::new();
static CLIPBOARD: OnceLock<Mutex<arboard::Clipboard>> = OnceLock::new();

/// Create the virtual keyboard and open the clipboard. Done once at startup:
/// a compositor needs a moment to adopt a new input device.
pub fn prepare() -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .open("/dev/uinput")
        .with_context(|| {
            format!(
                "open /dev/uinput (run `{} setup` once)",
                super::command_name()
            )
        })?;
    let fd = file.as_raw_fd();
    let mut setup = UinputSetup {
        bustype: 0x06, // BUS_VIRTUAL
        vendor: 0x7068,
        product: 0x6f6e,
        version: 1,
        name: [0; 80],
        ff_effects_max: 0,
    };
    setup.name[..VIRTUAL_KEYBOARD_NAME.len()].copy_from_slice(VIRTUAL_KEYBOARD_NAME.as_bytes());
    unsafe {
        if libc::ioctl(fd, UI_SET_EVBIT, EV_KEY as libc::c_int) < 0
            || libc::ioctl(fd, UI_SET_KEYBIT, KEY_LEFTCTRL as libc::c_int) < 0
            || libc::ioctl(fd, UI_SET_KEYBIT, KEY_V as libc::c_int) < 0
            || libc::ioctl(fd, UI_DEV_SETUP, &setup as *const UinputSetup) < 0
            || libc::ioctl(fd, UI_DEV_CREATE) < 0
        {
            bail!(
                "create the virtual keyboard: {}",
                std::io::Error::last_os_error()
            );
        }
    }
    let _ = KEYBOARD.set(Mutex::new(file));
    let clipboard = arboard::Clipboard::new().map_err(|e| anyhow!("open the clipboard: {e}"))?;
    let _ = CLIPBOARD.set(Mutex::new(clipboard));
    Ok(())
}

fn emit(file: &mut File, kind: u16, code: u16, value: i32) -> std::io::Result<()> {
    let mut event = [0u8; 24]; // the kernel stamps the time
    event[16..18].copy_from_slice(&kind.to_ne_bytes());
    event[18..20].copy_from_slice(&code.to_ne_bytes());
    event[20..24].copy_from_slice(&value.to_ne_bytes());
    file.write_all(&event)
}

fn control_v() -> Result<()> {
    let keyboard = KEYBOARD
        .get()
        .ok_or_else(|| anyhow!("no virtual keyboard"))?;
    let mut file = keyboard
        .lock()
        .map_err(|_| anyhow!("keyboard lock poisoned"))?;
    for (code, value) in [(KEY_LEFTCTRL, 1), (KEY_V, 1), (KEY_V, 0), (KEY_LEFTCTRL, 0)] {
        emit(&mut file, EV_KEY, code, value)?;
        emit(&mut file, EV_SYN, SYN_REPORT, 0)?;
        std::thread::sleep(Duration::from_millis(8));
    }
    Ok(())
}

/// Leave `text` on the clipboard, for the user to paste (the tray's recent list).
pub fn copy(text: &str) -> Result<()> {
    let clipboard = CLIPBOARD.get().ok_or_else(|| anyhow!("no clipboard"))?;
    clipboard
        .lock()
        .map_err(|_| anyhow!("clipboard lock poisoned"))?
        .set_text(text.to_string())
        .map_err(|e| anyhow!("set the clipboard: {e}"))
}

/// Insert `text` into the focused window. `after` says what the clipboard holds
/// afterwards (settings.json `clipboard_after_paste`). Returns the path used.
pub fn insert(text: &str, after: ClipboardAfterPaste) -> Result<&'static str> {
    if text.is_empty() {
        return Ok("skipped");
    }
    let clipboard = CLIPBOARD.get().ok_or_else(|| anyhow!("no clipboard"))?;
    let previous = {
        let mut c = clipboard
            .lock()
            .map_err(|_| anyhow!("clipboard lock poisoned"))?;
        let previous = match after {
            ClipboardAfterPaste::Restore => c.get_text().ok(),
            ClipboardAfterPaste::Dictation => None,
        };
        c.set_text(text.to_string())
            .map_err(|e| anyhow!("set the clipboard: {e}"))?;
        previous
    };
    // the clipboard owner change has to reach the compositor before the paste
    std::thread::sleep(Duration::from_millis(40));
    control_v()?;
    // apps fetch the pasted data asynchronously; restore only after they had time
    if let Some(previous) = after.to_restore(previous) {
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(600));
            if let Some(clipboard) = CLIPBOARD.get() {
                if let Ok(mut c) = clipboard.lock() {
                    let _ = c.set_text(previous);
                }
            }
        });
    }
    Ok("pasted")
}
