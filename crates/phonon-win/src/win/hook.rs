//! The global dictation key, through a low-level keyboard hook.
//!
//! `WH_KEYBOARD_LL` sees every key before the focused window does, which is what
//! a push-to-talk key needs. The hook itself only reads a virtual key code and
//! pushes a decision onto a channel: a hook that blocks for longer than the
//! system timeout is removed by Windows, so no work happens here.
//!
//! The default key is Right Control. Phonon swallows it, so Right Control stops
//! acting as a modifier while Phonon runs. Left Control is untouched.

use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use anyhow::{bail, Result};
use phonon_hotkey::{Action, HoldKeyTapLatch};
use windows_sys::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, SetWindowsHookExW, UnhookWindowsHookEx, HHOOK, KBDLLHOOKSTRUCT, LLKHF_INJECTED,
    WH_KEYBOARD_LL, WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP,
};

/// Virtual key codes Phonon can be bound to.
const VK_RCONTROL: u32 = 0xA3;
const VK_LCONTROL: u32 = 0xA2;
const VK_RMENU: u32 = 0xA5;
const VK_RSHIFT: u32 = 0xA1;
const VK_CAPITAL: u32 = 0x14;
const VK_F13: u32 = 0x7C;

/// Resolve a key name to a virtual key code. Unknown names are rejected so a
/// typo in the setting cannot silently leave the user with no dictation key.
pub fn key_code(name: &str) -> Result<u32> {
    let code = match name.trim().to_ascii_lowercase().as_str() {
        "rightctrl" | "right_ctrl" | "rctrl" => VK_RCONTROL,
        "leftctrl" | "left_ctrl" | "lctrl" => VK_LCONTROL,
        "rightalt" | "right_alt" | "ralt" => VK_RMENU,
        "rightshift" | "right_shift" | "rshift" => VK_RSHIFT,
        "capslock" | "caps" => VK_CAPITAL,
        "f13" => VK_F13,
        other => bail!(
            "unknown hotkey {other:?}; use rightctrl, leftctrl, rightalt, rightshift, capslock, or f13"
        ),
    };
    Ok(code)
}

/// The key name Phonon uses, from `PHONON_WIN_HOTKEY` or the default.
pub fn configured_key() -> (String, u32) {
    let name = std::env::var("PHONON_WIN_HOTKEY").unwrap_or_else(|_| "rightctrl".into());
    match key_code(&name) {
        Ok(code) => (name, code),
        Err(error) => {
            eprintln!("phonon: {error:#}; falling back to rightctrl");
            ("rightctrl".into(), VK_RCONTROL)
        }
    }
}

struct HookState {
    latch: HoldKeyTapLatch,
    sender: Sender<Action>,
    watched: u32,
    started: Instant,
}

static STATE: OnceLock<Mutex<HookState>> = OnceLock::new();
static HOOK: Mutex<isize> = Mutex::new(0);

/// Install the hook. It must be installed from the thread that runs the message
/// loop, and that loop must keep running.
pub fn install(watched: u32, sender: Sender<Action>) -> Result<()> {
    if STATE
        .set(Mutex::new(HookState {
            latch: HoldKeyTapLatch::default(),
            sender,
            watched,
            started: Instant::now(),
        }))
        .is_err()
    {
        bail!("the keyboard hook is already installed");
    }
    let handle = unsafe { SetWindowsHookExW(WH_KEYBOARD_LL, Some(proc), std::ptr::null_mut(), 0) };
    if handle.is_null() {
        bail!("SetWindowsHookExW failed; Phonon cannot see the dictation key");
    }
    *HOOK.lock().unwrap() = handle as isize;
    Ok(())
}

/// Remove the hook. Windows also removes it when the process exits.
pub fn uninstall() {
    let mut handle = HOOK.lock().unwrap();
    if *handle != 0 {
        unsafe { UnhookWindowsHookEx(*handle as HHOOK) };
        *handle = 0;
    }
}

/// Forget a latch, after something other than the key stopped the recording.
pub fn reset_latch() {
    if let Some(state) = STATE.get() {
        if let Ok(mut state) = state.lock() {
            state.latch.reset();
        }
    }
}

unsafe extern "system" fn proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if code < 0 {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    }
    let event = &*(lparam as *const KBDLLHOOKSTRUCT);
    // Phonon's own Control-V for pasting must not feed the latch.
    let injected = event.flags & LLKHF_INJECTED != 0;
    let Some(state) = STATE.get() else {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    };
    let Ok(mut state) = state.lock() else {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    };
    if injected || event.vkCode != state.watched {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    }

    let now = state.started.elapsed().as_nanos() as u64;
    let message = wparam as u32;
    let action = match message {
        WM_KEYDOWN | WM_SYSKEYDOWN => {
            // Windows repeats key-down while a key is held. Only the first press
            // of a run may reach the state machine.
            if state.latch.latched() || !HELD.swap(true, std::sync::atomic::Ordering::SeqCst) {
                Some(state.latch.key_down(now))
            } else {
                None
            }
        }
        WM_KEYUP | WM_SYSKEYUP => {
            HELD.store(false, std::sync::atomic::Ordering::SeqCst);
            Some(state.latch.key_up(now))
        }
        _ => None,
    };
    if let Some(action) = action {
        if action != Action::None {
            let _ = state.sender.send(action);
        }
    }
    // Swallow the key so the focused window never sees the dictation key.
    1
}

/// Whether the dictation key is down right now, so auto-repeat can be dropped.
static HELD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_names_resolve() {
        assert_eq!(key_code("rightctrl").unwrap(), VK_RCONTROL);
        assert_eq!(key_code("  Right_Ctrl ").unwrap(), VK_RCONTROL);
        assert_eq!(key_code("F13").unwrap(), VK_F13);
        assert_eq!(key_code("capslock").unwrap(), VK_CAPITAL);
    }

    #[test]
    fn a_typo_is_reported_not_ignored() {
        let error = key_code("rihgtctrl").unwrap_err().to_string();
        assert!(error.contains("unknown hotkey"));
        assert!(error.contains("rightctrl"));
    }
}
