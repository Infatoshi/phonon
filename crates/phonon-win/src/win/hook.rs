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
//! menu or the settings window (`shortcut_mode` in settings.json, else the older
//! `hotkey.txt`), the default. A change in settings.json applies on the worker's
//! next 500 ms tick (`follow_settings`). Laptops with a Copilot key
//! have no Right Control: when the hook sees the Copilot key (it sends F23) and the
//! user never chose a key, Phonon moves to Right Alt (or Ctrl+Space where Right Alt
//! is AltGr) and says so.
//!
//! Windows drops a low-level hook without a word (a missed timeout, another app's
//! hook installed in front, a session switch), so the hook is kept alive: it is
//! re-installed whenever another app comes to the foreground and again 1.5 s
//! later, and a 1 s watchdog (`watchdog_tick`) re-installs it when the session
//! sees input the hook never did, and finishes a hold whose key-up was lost.
//! The decisions live in `watchdog.rs`, tested on every platform.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU32, AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use anyhow::{bail, Result};
use phonon_hotkey::{Action, HoldKeyTapLatch};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::System::SystemInformation::GetTickCount;
use windows_sys::Win32::System::Threading::GetCurrentProcessId;
use windows_sys::Win32::UI::Accessibility::{SetWinEventHook, UnhookWinEvent, HWINEVENTHOOK};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, GetKeyboardLayout, GetLastInputInfo, SendInput, VkKeyScanExW, INPUT,
    INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, LASTINPUTINFO, VK_CONTROL,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, GetForegroundWindow, GetWindowThreadProcessId, KillTimer, SetTimer,
    SetWindowsHookExW, SystemParametersInfoW, UnhookWindowsHookEx, EVENT_SYSTEM_FOREGROUND, HHOOK,
    KBDLLHOOKSTRUCT, LLKHF_INJECTED, SPI_GETKEYBOARDDELAY, WH_KEYBOARD_LL, WINEVENT_OUTOFCONTEXT,
    WINEVENT_SKIPOWNPROCESS, WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP,
};

use super::watchdog::{
    is_altgr_synthetic_lctrl, needs_menu_bar_mask, repeat_delay_from_spi, tick_distance_u32,
    HookWatchdog, PHONON_INJECTED_EXTRA_INFO, VK_MENU_MASK,
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

/// The saved key: `shortcut_mode` in settings.json, or `hotkey.txt` from before it.
fn saved_name() -> Option<String> {
    crate::settings::Settings::load().shortcut()
}

/// Whether the key was picked by the user (environment, tray menu or settings
/// window), so the Copilot-key fallback must leave it alone.
fn user_chose() -> bool {
    std::env::var_os("PHONON_WIN_HOTKEY").is_some() || saved_name().is_some()
}

/// The key Phonon starts with, as its display label and code.
pub fn configured_key() -> (&'static str, u32) {
    let name = std::env::var("PHONON_WIN_HOTKEY")
        .ok()
        .or_else(saved_name)
        .unwrap_or_else(|| crate::settings::DEFAULT_SHORTCUT.into());
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
    crate::settings::set(crate::settings::SHORTCUT_MODE, name.into())?;
    switch_to(code);
    Ok(())
}

/// Called on the worker's tick when settings.json changed: follow a key picked
/// in the settings window. Returns the new key's label when it moved.
pub fn follow_settings() -> Option<&'static str> {
    let (label, code) = configured_key();
    if code == WATCHED.load(Ordering::Relaxed) {
        return None;
    }
    switch_to(code);
    eprintln!("phonon: dictation key is now {label}");
    Some(label)
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
/// `GetTickCount` at the hook's last callback, for the watchdog.
static LAST_HOOK_TICK: AtomicU32 = AtomicU32::new(0);
/// The last real key event the hook saw was the dictation key going down, so its
/// auto-repeat should follow while it is held. Any other key ends a key's
/// repeat, and a key the hook swallows never shows in `GetAsyncKeyState`; so this,
/// not the system's key state, is what tells a held key from a lost key-up.
static LAST_EVENT_WATCHED: AtomicBool = AtomicBool::new(false);
static WIN_EVENT_HOOK: AtomicIsize = AtomicIsize::new(0);
static TIMER: AtomicUsize = AtomicUsize::new(0);
static REARM_FAILURE_LOGGED: AtomicBool = AtomicBool::new(false);
static CLOCK: OnceLock<Instant> = OnceLock::new();

/// A delayed re-arm after a foreground change: Tauri 2 and raw-input apps show
/// their window first and install their own hook right after.
const FOREGROUND_REARM_DELAY_MS: u64 = 1_500;
const WATCHDOG_PERIOD_MS: u32 = 1_000;
/// The streak length at which a silent hook is worth a log line: a keyboard idle
/// while only the mouse moves also looks silent, and one re-arm costs nothing.
const LOGGED_STREAK: u32 = 2;

/// The watchdog's state. It lives on the hook thread: the timer and the
/// foreground event are delivered there by the message loop.
struct Watch {
    dog: HookWatchdog,
    /// `LAST_HOOK_TICK` as of the previous tick.
    seen_tick: u32,
    rearm_at_ms: Option<u64>,
    /// A "silent" line was written for the current streak.
    logged: bool,
}

thread_local! {
    static WATCH: RefCell<Option<Watch>> = const { RefCell::new(None) };
}

/// Milliseconds since the hook was installed.
fn now_ms() -> u64 {
    CLOCK.get_or_init(Instant::now).elapsed().as_millis() as u64
}

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
    *HOOK.lock().unwrap_or_else(|e| e.into_inner()) = handle as isize;

    // Keeping the hook alive is best effort: without these Phonon works as before.
    let now = now_ms();
    WATCH.with(|watch| {
        *watch.borrow_mut() = Some(Watch {
            dog: HookWatchdog::new(now),
            seen_tick: LAST_HOOK_TICK.load(Ordering::Relaxed),
            rearm_at_ms: None,
            logged: false,
        })
    });
    let timer = unsafe {
        SetTimer(
            std::ptr::null_mut(),
            0,
            WATCHDOG_PERIOD_MS,
            Some(watchdog_timer),
        )
    };
    if timer == 0 {
        eprintln!("phonon: could not start the keyboard hook's watchdog");
    }
    TIMER.store(timer, Ordering::SeqCst);
    let events = unsafe {
        SetWinEventHook(
            EVENT_SYSTEM_FOREGROUND,
            EVENT_SYSTEM_FOREGROUND,
            std::ptr::null_mut(),
            Some(foreground_proc),
            0,
            0,
            WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS,
        )
    };
    if events.is_null() {
        eprintln!(
            "phonon: could not watch the foreground window; the hook re-arms on the watchdog only"
        );
    }
    WIN_EVENT_HOOK.store(events as isize, Ordering::SeqCst);
    Ok(())
}

/// Remove the hook. Windows also removes it when the process exits.
pub fn uninstall() {
    let timer = TIMER.swap(0, Ordering::SeqCst);
    if timer != 0 {
        unsafe { KillTimer(std::ptr::null_mut(), timer) };
    }
    let events = WIN_EVENT_HOOK.swap(0, Ordering::SeqCst);
    if events != 0 {
        unsafe { UnhookWinEvent(events as HWINEVENTHOOK) };
    }
    let mut handle = HOOK.lock().unwrap_or_else(|e| e.into_inner());
    if *handle != 0 {
        unsafe { UnhookWindowsHookEx(*handle as HHOOK) };
        *handle = 0;
    }
}

/// Put a fresh hook at the front of the chain. The new one is installed before
/// the old one goes, so there is no moment without a hook; if Windows refuses the
/// new one, the old one stays. Runs on the hook thread only.
fn rearm_hook() -> bool {
    let fresh = unsafe { SetWindowsHookExW(WH_KEYBOARD_LL, Some(proc), std::ptr::null_mut(), 0) };
    if fresh.is_null() {
        if !REARM_FAILURE_LOGGED.swap(true, Ordering::Relaxed) {
            eprintln!("phonon: could not re-install the keyboard hook; keeping the current one");
        }
        return false;
    }
    let old = std::mem::replace(
        &mut *HOOK.lock().unwrap_or_else(|e| e.into_inner()),
        fresh as isize,
    );
    if old != 0 {
        unsafe { UnhookWindowsHookEx(old as HHOOK) };
    }
    true
}

fn foreground_pid() -> u32 {
    let window = unsafe { GetForegroundWindow() };
    if window.is_null() {
        return 0;
    }
    let mut pid = 0;
    unsafe { GetWindowThreadProcessId(window, &mut pid) };
    pid
}

/// Another app came to the front: it may have installed its own hook ahead of
/// Phonon's. Re-arm now and once more shortly after.
unsafe extern "system" fn foreground_proc(
    _hook: HWINEVENTHOOK,
    event: u32,
    window: HWND,
    _object: i32,
    _child: i32,
    _thread: u32,
    _time: u32,
) {
    if event != EVENT_SYSTEM_FOREGROUND || window.is_null() {
        return;
    }
    let mut pid = 0;
    GetWindowThreadProcessId(window, &mut pid);
    if pid == GetCurrentProcessId() {
        return;
    }
    rearm_hook();
    let due = now_ms() + FOREGROUND_REARM_DELAY_MS;
    WATCH.with(|watch| {
        if let Ok(mut watch) = watch.try_borrow_mut() {
            if let Some(watch) = watch.as_mut() {
                watch.rearm_at_ms = Some(due);
            }
        }
    });
}

unsafe extern "system" fn watchdog_timer(_window: HWND, _message: u32, _id: usize, _time: u32) {
    watchdog_tick();
}

/// What one watchdog tick found, for the log lines written after the state is let go.
#[derive(Default)]
struct Tick {
    recovered_after: Option<u32>,
    went_silent: bool,
    heal: bool,
}

/// Every second on the hook thread: run a due foreground re-arm, re-arm a hook
/// that is silent while the session has input, and finish a hold whose key-up
/// never arrived.
pub fn watchdog_tick() {
    let now_tick = unsafe { GetTickCount() };
    let now = now_ms();
    let tick = WATCH.with(|watch| {
        let mut tick = Tick::default();
        let Ok(mut watch) = watch.try_borrow_mut() else {
            return tick;
        };
        let Some(watch) = watch.as_mut() else {
            return tick;
        };

        let hook_tick = LAST_HOOK_TICK.load(Ordering::Relaxed);
        if hook_tick != watch.seen_tick {
            watch.seen_tick = hook_tick;
            let at = now.saturating_sub(u64::from(tick_distance_u32(now_tick, hook_tick)));
            if watch.logged {
                tick.recovered_after = Some(watch.dog.silent_streak());
            }
            watch.logged = false;
            watch.dog.record_hook_event(at);
        }

        if watch.rearm_at_ms.is_some_and(|due| now >= due) {
            watch.rearm_at_ms = None;
            rearm_hook();
        }

        let mut input = LASTINPUTINFO {
            cbSize: std::mem::size_of::<LASTINPUTINFO>() as u32,
            dwTime: 0,
        };
        if unsafe { GetLastInputInfo(&mut input) } != 0 {
            let idle = u64::from(tick_distance_u32(now_tick, input.dwTime));
            let last_input = now.saturating_sub(idle);
            let own_front = foreground_pid() == unsafe { GetCurrentProcessId() };
            if watch.dog.should_rearm(now, last_input, own_front) {
                rearm_hook();
                watch.dog.record_rearm(now);
                if watch.dog.silent_streak() == LOGGED_STREAK && !watch.logged {
                    watch.logged = true;
                    tick.went_silent = true;
                }
            }
        }

        if HELD.load(Ordering::SeqCst) {
            let watched = WATCHED.load(Ordering::Relaxed);
            let vk = if watched == KEY_CTRL_SPACE {
                VK_SPACE
            } else {
                watched
            };
            let async_down = unsafe { GetAsyncKeyState(vk as i32) } as u16 & 0x8000 != 0;
            let maybe_down = async_down || !LAST_EVENT_WATCHED.load(Ordering::Relaxed);
            tick.heal = HookWatchdog::should_heal_stuck_hold(
                now,
                true,
                watch.dog.last_hook_ms(),
                maybe_down,
                keyboard_repeat_delay_ms(),
            );
        }
        tick
    });

    if let Some(rearms) = tick.recovered_after {
        eprintln!("phonon: keyboard hook answering again after {rearms} re-arms");
    }
    if tick.went_silent {
        eprintln!("phonon: keyboard hook silent while the session has input; re-arming");
    }
    if tick.heal && heal_stuck_hold() {
        eprintln!("phonon: the dictation key's release never arrived; finishing the dictation");
    }
}

/// The user's keyboard repeat delay, in milliseconds.
fn keyboard_repeat_delay_ms() -> u64 {
    let mut level: u32 = 1;
    let ok = unsafe {
        SystemParametersInfoW(SPI_GETKEYBOARDDELAY, 0, (&mut level as *mut u32).cast(), 0)
    };
    repeat_delay_from_spi(if ok != 0 { level } else { 1 })
}

/// Feed the latch the key-up it missed, as the hook would have.
fn heal_stuck_hold() -> bool {
    let Some(state) = STATE.get() else {
        return false;
    };
    let Ok(mut state) = state.lock() else {
        return false;
    };
    if !HELD.swap(false, Ordering::SeqCst) {
        return false;
    }
    COMBO.store(false, Ordering::SeqCst);
    LAST_EVENT_WATCHED.store(false, Ordering::Relaxed);
    let now = state.started.elapsed().as_nanos() as u64;
    let action = state.latch.key_up(now);
    if action != Action::None {
        let _ = state.sender.send(action);
    }
    true
}

/// Tap an unassigned key, tagged as Phonon's own, so a swallowed Alt or Win
/// release does not open the focused window's menu bar or the Start menu and
/// take the paste. Never called with the hook state locked.
fn tap_menu_mask() {
    let key = |up: bool| {
        let mut input: INPUT = unsafe { std::mem::zeroed() };
        input.r#type = INPUT_KEYBOARD;
        input.Anonymous.ki = KEYBDINPUT {
            wVk: VK_MENU_MASK,
            wScan: 0,
            dwFlags: if up { KEYEVENTF_KEYUP } else { 0 },
            time: 0,
            dwExtraInfo: PHONON_INJECTED_EXTRA_INFO,
        };
        input
    };
    let events = [key(false), key(true)];
    unsafe {
        SendInput(
            events.len() as u32,
            events.as_ptr(),
            std::mem::size_of::<INPUT>() as i32,
        )
    };
}

/// Forget a latch, after something other than the key stopped the recording.
pub fn reset_latch() {
    if let Some(state) = STATE.get() {
        if let Ok(mut state) = state.lock() {
            state.latch.reset();
        }
    }
}

/// Whether a tap has latched the recording on, so it may stop by itself. A held key is not.
pub fn latched() -> bool {
    STATE
        .get()
        .and_then(|state| state.lock().ok().map(|state| state.latch.latched()))
        .unwrap_or(false)
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
    // Proof of life for the watchdog. Nothing here may log or allocate.
    LAST_HOOK_TICK.store(GetTickCount(), Ordering::Relaxed);
    if code < 0 {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    }
    let event = &*(lparam as *const KBDLLHOOKSTRUCT);
    // Phonon's own menu-bar mask, and the Left Control that AltGr layouts put in
    // front of Right Alt: neither is a key the user pressed.
    if event.dwExtraInfo == PHONON_INJECTED_EXTRA_INFO
        || is_altgr_synthetic_lctrl(event.vkCode, event.scanCode, event.flags)
    {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    }
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
    if injected || !(down || up) {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    }
    let Some(state) = STATE.get() else {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    };
    let watched = WATCHED.load(Ordering::Relaxed);
    // The system's own key state, not one tracked here: a Control release that
    // happens while an administrator window has focus never reaches this hook,
    // and a stuck "held" would turn every Space into a dictation.
    let ctrl_down = GetAsyncKeyState(VK_CONTROL as i32) as u16 & 0x8000 != 0;
    let combo = COMBO.load(Ordering::SeqCst);
    let edge = edge(watched, event.vkCode, down, ctrl_down, combo);
    LAST_EVENT_WATCHED.store(edge == Edge::Down, Ordering::Relaxed);
    if edge == Edge::Ignore {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    }
    {
        let Ok(mut state) = state.lock() else {
            return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
        };
        let now = state.started.elapsed().as_nanos() as u64;
        let action = if edge == Edge::Down {
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
        } else {
            COMBO.store(false, Ordering::SeqCst);
            HELD.store(false, Ordering::SeqCst);
            Some(state.latch.key_up(now))
        };
        if let Some(action) = action {
            if action != Action::None {
                let _ = state.sender.send(action);
            }
        }
    }
    if edge == Edge::Up && needs_menu_bar_mask(watched) {
        tap_menu_mask();
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

    /// The settings window offers `settings::SHORTCUTS`; each must be a key the
    /// hook installs, with the label the tray shows.
    #[test]
    fn the_settings_window_offers_the_tray_keys() {
        assert_eq!(crate::settings::SHORTCUTS.len(), CHOICES.len());
        for ((name, label), (menu_name, code)) in crate::settings::SHORTCUTS.iter().zip(CHOICES) {
            assert_eq!(*name, menu_name);
            assert_eq!(key_label(key_code(name).unwrap()), *label);
            assert_eq!(key_code(name).unwrap(), code);
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
