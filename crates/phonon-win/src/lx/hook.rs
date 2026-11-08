//! The global dictation key on Linux, read from evdev.
//!
//! X11 and every Wayland compositor sit on top of the kernel's evdev devices, so
//! reading `/dev/input/event*` sees the key no matter which window has focus or
//! which session type runs. Wayland gives ordinary clients no global key hook.
//! Reading needs the udev rule in `linux/70-phonon.rules` (the .deb installs it;
//! `setup` prints the one-time command otherwise). The key is not swallowed: the focused window still sees a bare
//! Right Control, which does nothing on its own.

use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use phonon_hotkey::{Action, HoldKeyTapLatch};

use super::inject::VIRTUAL_KEYBOARD_NAME;

/// Linux input event codes (linux/input-event-codes.h).
pub const KEY_LEFTCTRL: u16 = 29;
pub const KEY_RIGHTCTRL: u16 = 97;
pub const KEY_RIGHTALT: u16 = 100;
pub const KEY_RIGHTSHIFT: u16 = 54;
pub const KEY_CAPSLOCK: u16 = 58;
pub const KEY_F13: u16 = 183;
const EV_KEY: u16 = 1;

/// Resolve a key name to an evdev code. The names match the Windows build.
pub fn key_code(name: &str) -> Result<u16> {
    Ok(match name.trim().to_ascii_lowercase().as_str() {
        "rightctrl" | "right_ctrl" | "rctrl" => KEY_RIGHTCTRL,
        "leftctrl" | "left_ctrl" | "lctrl" => KEY_LEFTCTRL,
        "rightalt" | "right_alt" | "ralt" => KEY_RIGHTALT,
        "rightshift" | "right_shift" | "rshift" => KEY_RIGHTSHIFT,
        "capslock" | "caps" => KEY_CAPSLOCK,
        "f13" => KEY_F13,
        other => bail!(
            "unknown hotkey {other:?}; use rightctrl, leftctrl, rightalt, rightshift, capslock, or f13"
        ),
    })
}

pub fn key_label(code: u16) -> &'static str {
    match code {
        KEY_RIGHTCTRL => "Right Ctrl",
        KEY_LEFTCTRL => "Left Ctrl",
        KEY_RIGHTALT => "Right Alt",
        KEY_RIGHTSHIFT => "Right Shift",
        KEY_CAPSLOCK => "Caps Lock",
        KEY_F13 => "F13",
        _ => "the dictation key",
    }
}

/// The key from `PHONON_HOTKEY` (or the Windows name `PHONON_WIN_HOTKEY`), default Right Ctrl.
pub fn configured_key() -> (&'static str, u16) {
    let name = std::env::var("PHONON_HOTKEY")
        .or_else(|_| std::env::var("PHONON_WIN_HOTKEY"))
        .unwrap_or_else(|_| "rightctrl".into());
    let code = key_code(&name).unwrap_or_else(|error| {
        eprintln!("phonon: {error:#}; falling back to rightctrl");
        KEY_RIGHTCTRL
    });
    (key_label(code), code)
}

struct State {
    latch: HoldKeyTapLatch,
    held: bool,
    started: Instant,
    sender: Sender<Action>,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

/// Keyboards Phonon can read now, and those it cannot (permission).
pub fn devices() -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut readable = Vec::new();
    let mut denied = Vec::new();
    let Ok(entries) = std::fs::read_dir("/dev/input") else {
        return (readable, denied);
    };
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("event"))
        })
        .collect();
    paths.sort();
    for path in paths {
        if device_name(&path).as_deref() == Some(VIRTUAL_KEYBOARD_NAME) || !is_keyboard(&path) {
            continue;
        }
        match File::open(&path) {
            Ok(_) => readable.push(path),
            Err(_) => denied.push(path),
        }
    }
    (readable, denied)
}

fn sys_dir(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    Some(PathBuf::from(format!("/sys/class/input/{name}/device")))
}

fn device_name(path: &Path) -> Option<String> {
    std::fs::read_to_string(sys_dir(path)?.join("name"))
        .ok()
        .map(|s| s.trim().to_string())
}

/// A device whose key capability bitmap includes the letter row is a keyboard.
/// sysfs prints the bitmap as space-separated hex words, most significant first.
fn is_keyboard(path: &Path) -> bool {
    let Some(dir) = sys_dir(path) else {
        return false;
    };
    let Ok(text) = std::fs::read_to_string(dir.join("capabilities/key")) else {
        return false;
    };
    has_key(&text, 30) && has_key(&text, KEY_LEFTCTRL) // KEY_A and a control key
}

fn has_key(bitmap: &str, code: u16) -> bool {
    let words: Vec<&str> = bitmap.split_whitespace().collect();
    let bits = usize::BITS as usize;
    let index = code as usize / bits;
    if index >= words.len() {
        return false;
    }
    let word = words[words.len() - 1 - index];
    u64::from_str_radix(word, 16).is_ok_and(|w| w >> (code as usize % bits) & 1 == 1)
}

/// Start one reader thread per keyboard, and a watcher that picks up keyboards
/// plugged in later. Fails only when no keyboard is readable at all.
pub fn install(watched: u16, sender: Sender<Action>) -> Result<()> {
    if STATE
        .set(Mutex::new(State {
            latch: HoldKeyTapLatch::default(),
            held: false,
            started: Instant::now(),
            sender,
        }))
        .is_err()
    {
        bail!("the key reader is already running");
    }
    let (readable, denied) = devices();
    if readable.is_empty() {
        bail!(
            "Phonon cannot read the keyboard ({} keyboard device(s), none readable). Run `{} setup` once.",
            denied.len(),
            super::command_name()
        );
    }
    let open: Arc<Mutex<HashSet<PathBuf>>> = Arc::default();
    for path in readable {
        spawn_reader(path, watched, open.clone());
    }
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(3));
        for path in devices().0 {
            spawn_reader(path, watched, open.clone());
        }
    });
    Ok(())
}

fn spawn_reader(path: PathBuf, watched: u16, open: Arc<Mutex<HashSet<PathBuf>>>) {
    if !open.lock().unwrap().insert(path.clone()) {
        return;
    }
    std::thread::spawn(move || {
        if let Ok(mut file) = File::open(&path) {
            // struct input_event on 64-bit: timeval (16), type u16, code u16, value i32
            let mut buf = [0u8; 24];
            while file.read_exact(&mut buf).is_ok() {
                let kind = u16::from_ne_bytes([buf[16], buf[17]]);
                let code = u16::from_ne_bytes([buf[18], buf[19]]);
                let value = i32::from_ne_bytes([buf[20], buf[21], buf[22], buf[23]]);
                if kind == EV_KEY && code == watched {
                    key(value);
                }
            }
        }
        // unplugged or lost access: let the watcher reopen it
        open.lock().unwrap().remove(&path);
    });
}

/// value 1 = press, 0 = release, 2 = auto-repeat (dropped, as on Windows).
fn key(value: i32) {
    let Some(state) = STATE.get() else { return };
    let Ok(mut s) = state.lock() else { return };
    let now = s.started.elapsed().as_nanos() as u64;
    let action = match value {
        1 if !s.held || s.latch.latched() => {
            s.held = true;
            s.latch.key_down(now)
        }
        0 => {
            s.held = false;
            s.latch.key_up(now)
        }
        _ => Action::None,
    };
    if action != Action::None {
        let _ = s.sender.send(action);
    }
}

pub fn reset_latch() {
    if let Some(state) = STATE.get() {
        if let Ok(mut s) = state.lock() {
            s.latch.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_match_the_windows_build() {
        assert_eq!(key_code("rightctrl").unwrap(), KEY_RIGHTCTRL);
        assert_eq!(key_code(" Right_Ctrl ").unwrap(), KEY_RIGHTCTRL);
        assert_eq!(key_label(key_code("capslock").unwrap()), "Caps Lock");
        assert!(key_code("rihgtctrl").is_err());
    }

    #[test]
    fn capability_bitmap_is_read_from_the_last_word() {
        // a real laptop keyboard (word 0 holds codes 0-63, written last)
        let kb = "1000000000007 ff9f207ac14057ff febeffdfffefffff fffffffffffffffe";
        assert!(has_key(kb, 30));
        assert!(has_key(kb, KEY_RIGHTCTRL));
        // a power button: only KEY_POWER (116)
        assert!(!has_key("10000000000000 0", 30));
    }
}
