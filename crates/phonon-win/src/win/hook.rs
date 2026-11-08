//! The global dictation key, through a low-level keyboard hook.
//!
//! `WH_KEYBOARD_LL` sees every key before the focused window does, which is what
//! a push-to-talk key needs. The hook itself only reads a virtual key code and
//! pushes a decision onto a channel: a hook that blocks for longer than the
//! system timeout is removed by Windows, so no work happens here.
//!
//! The default key is Right Control. Phonon swallows the key it uses, so Right
//! Control stops acting as a modifier while Phonon runs. Left Control is untouched.
//!
//! The key comes from, in order: `PHONON_WIN_HOTKEY`, the choice saved from the tray
//! menu (`hotkey.txt` in the data folder), the default. Laptops with a Copilot key
//! have no Right Control: when the hook sees the Copilot key (it sends F23) and the
//! user never chose a key, Phonon moves to Right Alt (or Ctrl+Space where Right Alt
//! is AltGr) and says so.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use anyhow::{bail, Result};
use phonon_hotkey::{Action, HoldKeyTapLatch};
use windows_sys::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, GetKeyboardLayout, VkKeyScanExW, VK_CONTROL,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, GetForegroundWindow, GetWindowThreadProcessId, SetWindowsHookExW,
    UnhookWindowsHookEx, HHOOK, KBDLLHOOKSTRUCT, LLKHF_INJECTED, WH_KEYBOARD_LL, WM_KEYDOWN,
    WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP,
};

/// Virtual key codes Phonon can be bound to.
const VK_RCONTROL: u32 = 0xA3;
const VK_LCONTROL: u32 = 0xA2;
const VK_RMENU: u32 = 0xA5;
const VK_RSHIFT: u32 = 0xA1;
const VK_CAPITAL: u32 = 0x14;
const VK_F13: u32 = 0x7C;
const VK_SPACE: u32 = 0x20;
/// The Copilot key arrives as Left Win + Left Shift + F23.
const VK_F23: u32 = 0x86;
/// Not a virtual key: Space while either Control is held.
const KEY_CTRL_SPACE: u32 = 0x1_0000 | VK_SPACE;

/// The keys the tray menu offers, as (setting name, code).
pub const CHOICES: [(&str, u32); 6] = [
    ("rightctrl", VK_RCONTROL),
    ("rightalt", VK_RMENU),
    ("ctrlspace", KEY_CTRL_SPACE),
    ("rightshift", VK_RSHIFT),
    ("capslock", VK_CAPITAL),
    ("f13", VK_F13),
];

/// Resolve a key name to a code. Unknown names are rejected so a typo in the
/// setting cannot silently leave the user with no dictation key.
pub fn key_code(name: &str) -> Result<u32> {
    let code = match name.trim().to_ascii_lowercase().as_str() {
        "rightctrl" | "right_ctrl" | "rctrl" => VK_RCONTROL,
        "leftctrl" | "left_ctrl" | "lctrl" => VK_LCONTROL,
        "rightalt" | "right_alt" | "ralt" => VK_RMENU,
        "rightshift" | "right_shift" | "rshift" => VK_RSHIFT,
        "capslock" | "caps" => VK_CAPITAL,
        "f13" => VK_F13,
        "ctrlspace" | "ctrl_space" | "ctrl+space" => KEY_CTRL_SPACE,
        other => bail!(
            "unknown hotkey {other:?}; use rightctrl, leftctrl, rightalt, rightshift, capslock, f13, or ctrlspace"
        ),
    };
    Ok(code)
}

/// The key as the keyboard prints it, for the tooltip, the menu and the balloons.
/// The setting's spelling ("rightctrl") is not something a user reads.
pub fn key_label(code: u32) -> &'static str {
    match code {
        VK_RCONTROL => "Right Ctrl",
        VK_LCONTROL => "Left Ctrl",
        VK_RMENU => "Right Alt",
        VK_RSHIFT => "Right Shift",
        VK_CAPITAL => "Caps Lock",
        VK_F13 => "F13",
        KEY_CTRL_SPACE => "Ctrl+Space",
        _ => "the dictation key",
    }
}

fn saved_path() -> std::path::PathBuf {
    crate::paths::data_root().join("hotkey.txt")
}

/// Whether the key was picked by the user (environment or tray menu), so the
/// Copilot-key fallback must leave it alone.
fn user_chose() -> bool {
    std::env::var_os("PHONON_WIN_HOTKEY").is_some() || saved_path().is_file()
}

/// The key Phonon starts with, as its display label and code.
pub fn configured_key() -> (&'static str, u32) {
    let name = std::env::var("PHONON_WIN_HOTKEY")
        .ok()
        .or_else(|| std::fs::read_to_string(saved_path()).ok())
        .unwrap_or_else(|| "rightctrl".into());
    let code = match key_code(&name) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("phonon: {error:#}; falling back to rightctrl");
            VK_RCONTROL
        }
    };
    (key_label(code), code)
}

/// The key in use right now (it can change from the tray menu).
pub fn current_key() -> (&'static str, u32) {
    let code = WATCHED.load(Ordering::Relaxed);
    (key_label(code), code)
}

/// Use `code` from now on and remember it across restarts.
pub fn choose(code: u32) -> Result<()> {
    let name = CHOICES
        .iter()
        .find(|(_, c)| *c == code)
        .map(|(name, _)| *name)
        .ok_or_else(|| anyhow::anyhow!("not a menu key: {code:#x}"))?;
    let path = saved_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&path, name)?;
    switch_to(code);
    Ok(())
}

fn switch_to(code: u32) {
    reset_latch();
    HELD.store(false, Ordering::SeqCst);
    COMBO.store(false, Ordering::SeqCst);
    WATCHED.store(code, Ordering::SeqCst);
}

/// Whether the active keyboard layout types characters with AltGr (Right Alt
/// acting as Ctrl+Alt), in which case Right Alt must not become the dictation key.
fn layout_uses_altgr() -> bool {
    let thread = unsafe { GetWindowThreadProcessId(GetForegroundWindow(), std::ptr::null_mut()) };
    let layout = unsafe { GetKeyboardLayout(thread) };
    "@€{}[]\\|~".encode_utf16().any(|ch| {
        let scan = unsafe { VkKeyScanExW(ch, layout) } as u16;
        // High byte: 2 = Ctrl, 4 = Alt; both together is AltGr.
        scan != 0xFFFF && (scan >> 8) & 0x06 == 0x06
    })
}

/// Called on the main thread after the hook saw the Copilot key. Returns the
/// new key's label when Phonon moved off Right Ctrl, for the balloon.
pub fn on_copilot_key() -> Option<&'static str> {
    if user_chose() || RIGHT_CTRL_SEEN.load(Ordering::Relaxed) {
        return None;
    }
    if WATCHED.load(Ordering::Relaxed) != VK_RCONTROL {
        return None;
    }
    let fallback = if layout_uses_altgr() {
        KEY_CTRL_SPACE
    } else {
        VK_RMENU
    };
    // Saved like a menu pick, so the next start uses it without another Copilot press.
    if let Err(error) = choose(fallback) {
        eprintln!("phonon: could not save the dictation key: {error:#}");
        switch_to(fallback);
    }
    Some(key_label(fallback))
}

struct HookState {
    latch: HoldKeyTapLatch,
    sender: Sender<Action>,
    started: Instant,
}

static STATE: OnceLock<Mutex<HookState>> = OnceLock::new();
static HOOK: Mutex<isize> = Mutex::new(0);
static WATCHED: AtomicU32 = AtomicU32::new(VK_RCONTROL);
/// A Right Ctrl press was seen, so this keyboard has the default key.
static RIGHT_CTRL_SEEN: AtomicBool = AtomicBool::new(false);
static COPILOT_SEEN: AtomicBool = AtomicBool::new(false);
/// Ctrl+Space: the Space press that started dictation is still down.
static COMBO: AtomicBool = AtomicBool::new(false);
/// Whether the dictation key is down right now, so auto-repeat can be dropped.
static HELD: AtomicBool = AtomicBool::new(false);

/// Install the hook. It must be installed from the thread that runs the message
/// loop, and that loop must keep running.
pub fn install(watched: u32, sender: Sender<Action>) -> Result<()> {
    WATCHED.store(watched, Ordering::SeqCst);
    if STATE
        .set(Mutex::new(HookState {
            latch: HoldKeyTapLatch::default(),
            sender,
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

fn accept_injected() -> bool {
    static ACCEPT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ACCEPT.get_or_init(|| std::env::var_os("PHONON_WIN_ACCEPT_INJECTED").is_some_and(|v| v == "1"))
}

/// What a key event means for the watched key: a press, a release, or nothing.
#[derive(Debug, PartialEq, Eq)]
enum Edge {
    Down,
    Up,
    Ignore,
}

/// Map one hook event to an edge of the watched key. `ctrl_down` is the Control
/// state at the time of the event; `combo` is whether a Ctrl+Space press is open.
fn edge(watched: u32, vk: u32, down: bool, ctrl_down: bool, combo: bool) -> Edge {
    if watched == KEY_CTRL_SPACE {
        if vk != VK_SPACE {
            return Edge::Ignore;
        }
        return match (down, ctrl_down || combo, combo) {
            (true, true, _) => Edge::Down,
            (false, _, true) => Edge::Up,
            _ => Edge::Ignore,
        };
    }
    if vk != watched {
        return Edge::Ignore;
    }
    if down {
        Edge::Down
    } else {
        Edge::Up
    }
}

unsafe extern "system" fn proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if code < 0 {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    }
    let event = &*(lparam as *const KBDLLHOOKSTRUCT);
    let message = wparam as u32;
    let down = matches!(message, WM_KEYDOWN | WM_SYSKEYDOWN);
    let up = matches!(message, WM_KEYUP | WM_SYSKEYUP);
    // Phonon's own Control-V for pasting must not feed the latch. Automated
    // tests on a machine without a keyboard set PHONON_WIN_ACCEPT_INJECTED=1
    // to drive the key with SendInput; never set it for real use.
    let injected = event.flags & LLKHF_INJECTED != 0 && !accept_injected();
    if !injected && down {
        if event.vkCode == VK_RCONTROL {
            RIGHT_CTRL_SEEN.store(true, Ordering::Relaxed);
        } else if event.vkCode == VK_F23 && !COPILOT_SEEN.swap(true, Ordering::Relaxed) {
            super::tray::post_copilot_key();
        }
    }
    let Some(state) = STATE.get() else {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    };
    let Ok(mut state) = state.lock() else {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    };
    if injected || !(down || up) {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    }
    let watched = WATCHED.load(Ordering::Relaxed);
    // The system's own key state, not one tracked here: a Control release that
    // happens while an administrator window has focus never reaches this hook,
    // and a stuck "held" would turn every Space into a dictation.
    let ctrl_down = GetAsyncKeyState(VK_CONTROL as i32) as u16 & 0x8000 != 0;
    let combo = COMBO.load(Ordering::SeqCst);
    let now = state.started.elapsed().as_nanos() as u64;
    let action = match edge(watched, event.vkCode, down, ctrl_down, combo) {
        Edge::Ignore => return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam),
        Edge::Down => {
            if watched == KEY_CTRL_SPACE {
                COMBO.store(true, Ordering::SeqCst);
            }
            // Windows repeats key-down while a key is held. Only the first press
            // of a run may reach the state machine.
            if state.latch.latched() || !HELD.swap(true, Ordering::SeqCst) {
                Some(state.latch.key_down(now))
            } else {
                None
            }
        }
        Edge::Up => {
            COMBO.store(false, Ordering::SeqCst);
            HELD.store(false, Ordering::SeqCst);
            Some(state.latch.key_up(now))
        }
    };
    if let Some(action) = action {
        if action != Action::None {
            let _ = state.sender.send(action);
        }
    }
    // Swallow the key so the focused window never sees the dictation key.
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_names_resolve() {
        assert_eq!(key_code("rightctrl").unwrap(), VK_RCONTROL);
        assert_eq!(key_code("  Right_Ctrl ").unwrap(), VK_RCONTROL);
        assert_eq!(key_code("F13").unwrap(), VK_F13);
        assert_eq!(key_code("capslock").unwrap(), VK_CAPITAL);
        assert_eq!(key_code("Ctrl+Space").unwrap(), KEY_CTRL_SPACE);
    }

    #[test]
    fn labels_read_like_the_keyboard() {
        assert_eq!(key_label(key_code("rightctrl").unwrap()), "Right Ctrl");
        assert_eq!(key_label(key_code("capslock").unwrap()), "Caps Lock");
        assert_eq!(key_label(KEY_CTRL_SPACE), "Ctrl+Space");
    }

    #[test]
    fn a_typo_is_reported_not_ignored() {
        let error = key_code("rihgtctrl").unwrap_err().to_string();
        assert!(error.contains("unknown hotkey"));
        assert!(error.contains("rightctrl"));
    }

    #[test]
    fn every_menu_choice_has_a_name_and_a_label() {
        for (name, code) in CHOICES {
            assert_eq!(key_code(name).unwrap(), code);
            assert_ne!(key_label(code), "the dictation key");
        }
    }

    #[test]
    fn single_keys_map_to_edges() {
        assert_eq!(
            edge(VK_RCONTROL, VK_RCONTROL, true, true, false),
            Edge::Down
        );
        assert_eq!(
            edge(VK_RCONTROL, VK_RCONTROL, false, false, false),
            Edge::Up
        );
        assert_eq!(
            edge(VK_RCONTROL, VK_LCONTROL, true, true, false),
            Edge::Ignore
        );
    }

    #[test]
    fn ctrl_space_needs_control_to_start_but_not_to_end() {
        // Space alone types a space.
        assert_eq!(
            edge(KEY_CTRL_SPACE, VK_SPACE, true, false, false),
            Edge::Ignore
        );
        assert_eq!(
            edge(KEY_CTRL_SPACE, VK_SPACE, true, true, false),
            Edge::Down
        );
        // Auto-repeat while held, even after Control is let go.
        assert_eq!(
            edge(KEY_CTRL_SPACE, VK_SPACE, true, false, true),
            Edge::Down
        );
        // Releasing Control first still ends on the Space release.
        assert_eq!(edge(KEY_CTRL_SPACE, VK_SPACE, false, false, true), Edge::Up);
        // A Space release that did not start dictation passes through.
        assert_eq!(
            edge(KEY_CTRL_SPACE, VK_SPACE, false, false, false),
            Edge::Ignore
        );
        // Control itself is never swallowed.
        assert_eq!(
            edge(KEY_CTRL_SPACE, VK_LCONTROL, true, true, false),
            Edge::Ignore
        );
    }
}
