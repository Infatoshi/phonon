//! Put finished text into the focused window on Linux: clipboard plus Shift-Insert.
//!
//! Wayland lets no ordinary client send keys to another, so the keystroke comes
//! from a uinput virtual keyboard, which every compositor and X server treats as
//! real hardware. The clipboard goes through arboard (Wayland data-control where
//! the compositor offers it, otherwise X11, which GNOME and KDE bridge to their
//! Wayland clipboard). The previous clipboard text is put back afterwards, unless
//! settings.json `clipboard_after_paste` is "dictation", which leaves the dictation there.
//!
//! The paste key is Shift-Insert, not Control-V, with the text on both the
//! clipboard and the primary selection. GTK, Qt, Firefox, Chromium and Electron
//! apps and LibreOffice paste the clipboard on Shift-Insert; terminals (GNOME
//! Terminal, Ptyxis, Konsole, xterm) paste the primary selection on it. Control-V
//! in a terminal is the tty's "literal next" key: nothing was pasted and the
//! user's next key (Enter) was typed as ^M (Ubuntu QA VM, GNOME Terminal,
//! 2026-11-12).

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use arboard::{GetExtLinux, LinuxClipboardKind, SetExtLinux};

use super::hook::KEY_LEFTCTRL;
use crate::settings::ClipboardAfterPaste;

/// Phonon's virtual keyboard, which the key reader skips.
pub const VIRTUAL_KEYBOARD_NAME: &str = "Phonon virtual keyboard";
const KEY_V: u16 = 47;
const KEY_LEFTSHIFT: u16 = 42;
const KEY_INSERT: u16 = 110;
/// The paste chord, as (key, value) events: Shift down, Insert down, Insert up, Shift up.
const PASTE_CHORD: [(u16, i32); 4] = [
    (KEY_LEFTSHIFT, 1),
    (KEY_INSERT, 1),
    (KEY_INSERT, 0),
    (KEY_LEFTSHIFT, 0),
];
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
            || libc::ioctl(fd, UI_SET_KEYBIT, KEY_LEFTSHIFT as libc::c_int) < 0
            || libc::ioctl(fd, UI_SET_KEYBIT, KEY_INSERT as libc::c_int) < 0
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

fn press_paste() -> Result<()> {
    let keyboard = KEYBOARD
        .get()
        .ok_or_else(|| anyhow!("no virtual keyboard"))?;
    let mut file = keyboard
        .lock()
        .map_err(|_| anyhow!("keyboard lock poisoned"))?;
    for (code, value) in PASTE_CHORD {
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
    let (previous, previous_primary) = {
        let mut c = clipboard
            .lock()
            .map_err(|_| anyhow!("clipboard lock poisoned"))?;
        let (previous, previous_primary) = match after {
            ClipboardAfterPaste::Restore => (
                c.get_text().ok(),
                c.get().clipboard(LinuxClipboardKind::Primary).text().ok(),
            ),
            ClipboardAfterPaste::Dictation => (None, None),
        };
        c.set_text(text.to_string())
            .map_err(|e| anyhow!("set the clipboard: {e}"))?;
        // Terminals paste the primary selection on Shift-Insert. Without it the
        // clipboard alone still serves every other app, so a failure is not fatal.
        if let Err(e) = c
            .set()
            .clipboard(LinuxClipboardKind::Primary)
            .text(text.to_string())
        {
            eprintln!("phonon: could not set the primary selection: {e}");
        }
        (previous, previous_primary)
    };
    // the clipboard owner change has to reach the compositor before the paste
    std::thread::sleep(Duration::from_millis(40));
    press_paste()?;
    // apps fetch the pasted data asynchronously; restore only after they had time
    let previous = after.to_restore(previous);
    let previous_primary = after.to_restore(previous_primary);
    if previous.is_some() || previous_primary.is_some() {
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(600));
            if let Some(clipboard) = CLIPBOARD.get() {
                if let Ok(mut c) = clipboard.lock() {
                    if let Some(previous) = previous {
                        let _ = c.set_text(previous);
                    }
                    if let Some(previous) = previous_primary {
                        let _ = c
                            .set()
                            .clipboard(LinuxClipboardKind::Primary)
                            .text(previous);
                    }
                }
            }
        });
    }
    Ok("pasted")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paste_is_shift_insert_and_releases_every_key() {
        let downs: Vec<u16> = PASTE_CHORD
            .iter()
            .filter(|(_, v)| *v == 1)
            .map(|(k, _)| *k)
            .collect();
        assert_eq!(downs, vec![KEY_LEFTSHIFT, KEY_INSERT]);
        // never Control: Control-V is the terminal's literal-next key
        assert!(PASTE_CHORD.iter().all(|(k, _)| *k != KEY_LEFTCTRL));
        for (key, _) in PASTE_CHORD {
            let net: i32 = PASTE_CHORD
                .iter()
                .filter(|(k, _)| *k == key)
                .map(|(_, v)| if *v == 1 { 1 } else { -1 })
                .sum();
            assert_eq!(net, 0, "key {key} left down");
        }
        // Shift is held around Insert
        assert_eq!(PASTE_CHORD.first(), Some(&(KEY_LEFTSHIFT, 1)));
        assert_eq!(PASTE_CHORD.last(), Some(&(KEY_LEFTSHIFT, 0)));
    }
}
