//! The notification-area icon, its menu, and its balloon messages.
//!
//! Phonon has no window on Windows. The tray icon is the whole user interface:
//! its tooltip carries first-run download progress and the current state, and
//! its balloons announce the things a user must not miss, such as a failed
//! download or the moment dictation becomes usable.
//!
//! The icon is owned by a message-only window. That window also runs the message
//! loop the keyboard hook needs, so both live on the main thread.

use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock};

use anyhow::{bail, Result};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Shell::{
    Shell_NotifyIconW, NIF_ICON, NIF_INFO, NIF_MESSAGE, NIF_TIP, NIIF_ERROR, NIIF_INFO, NIM_ADD,
    NIM_DELETE, NIM_MODIFY, NOTIFYICONDATAW,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    AppendMenuW, CreatePopupMenu, CreateWindowExW, DefWindowProcW, DestroyMenu, DispatchMessageW,
    GetCursorPos, GetMessageW, LoadIconW, PostQuitMessage, RegisterClassW, SetForegroundWindow,
    TrackPopupMenu, TranslateMessage, HMENU, IDI_APPLICATION, MF_DISABLED, MF_GRAYED, MF_SEPARATOR,
    MF_STRING, MSG, TPM_BOTTOMALIGN, TPM_RIGHTBUTTON, WM_APP, WM_COMMAND, WM_DESTROY, WM_NULL,
    WM_RBUTTONUP, WNDCLASSW, WS_OVERLAPPED,
};

/// Sent to the message-only window when the icon is clicked.
const WM_TRAY: u32 = WM_APP + 1;
/// Posted by the worker thread when the tooltip text has changed.
const WM_REFRESH: u32 = WM_APP + 2;

const ID_TOGGLE: usize = 1;
const ID_OPEN_FOLDER: usize = 2;
const ID_QUIT: usize = 3;

/// What the user picked from the menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Start or stop dictation without touching the keyboard.
    Toggle,
    /// Open the Phonon folder in Explorer.
    OpenFolder,
    /// Leave.
    Quit,
}

static TRAY: OnceLock<Mutex<TrayHandle>> = OnceLock::new();
static COMMANDS: OnceLock<Mutex<Sender<Command>>> = OnceLock::new();
static STATUS: Mutex<String> = Mutex::new(String::new());

/// The window handle, held as an integer so it can live in a static.
struct TrayHandle {
    hwnd: isize,
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Copy `text` into a fixed-size wide buffer, truncating and terminating it.
fn fill(buffer: &mut [u16], text: &str) {
    let units: Vec<u16> = text.encode_utf16().take(buffer.len() - 1).collect();
    buffer[..units.len()].copy_from_slice(&units);
    buffer[units.len()] = 0;
    for slot in buffer.iter_mut().skip(units.len() + 1) {
        *slot = 0;
    }
}

fn icon_data(hwnd: HWND) -> NOTIFYICONDATAW {
    let mut data: NOTIFYICONDATAW = unsafe { std::mem::zeroed() };
    data.cbSize = std::mem::size_of::<NOTIFYICONDATAW>() as u32;
    data.hWnd = hwnd;
    data.uID = 1;
    data
}

/// Create the message-only window and add the icon.
pub fn create(commands: Sender<Command>) -> Result<()> {
    if COMMANDS.set(Mutex::new(commands)).is_err() {
        bail!("the tray icon already exists");
    }
    let class_name = wide("PhononTrayWindow");
    let instance = unsafe { GetModuleHandleW(std::ptr::null()) };
    let mut class: WNDCLASSW = unsafe { std::mem::zeroed() };
    class.lpfnWndProc = Some(window_proc);
    class.hInstance = instance;
    class.lpszClassName = class_name.as_ptr();
    if unsafe { RegisterClassW(&class) } == 0 {
        bail!("RegisterClassW failed");
    }

    let title = wide("Phonon");
    let hwnd = unsafe {
        CreateWindowExW(
            0,
            class_name.as_ptr(),
            title.as_ptr(),
            WS_OVERLAPPED,
            0,
            0,
            0,
            0,
            // HWND_MESSAGE: a window with no display of its own.
            -3isize as HWND,
            std::ptr::null_mut(),
            instance,
            std::ptr::null(),
        )
    };
    if hwnd.is_null() {
        bail!("CreateWindowExW failed");
    }
    let _ = TRAY.set(Mutex::new(TrayHandle {
        hwnd: hwnd as isize,
    }));

    let mut data = icon_data(hwnd);
    data.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    data.uCallbackMessage = WM_TRAY;
    data.hIcon = unsafe { LoadIconW(std::ptr::null_mut(), IDI_APPLICATION) };
    fill(&mut data.szTip, "Phonon: starting");
    if unsafe { Shell_NotifyIconW(NIM_ADD, &data) } == 0 {
        bail!("Shell_NotifyIconW could not add the icon");
    }
    Ok(())
}

fn hwnd() -> Option<HWND> {
    TRAY.get()
        .and_then(|tray| tray.lock().ok())
        .map(|tray| tray.hwnd as HWND)
}

/// Replace the tooltip. Safe to call from any thread.
pub fn set_status(text: &str) {
    if let Ok(mut status) = STATUS.lock() {
        *status = text.to_string();
    }
    let Some(hwnd) = hwnd() else { return };
    let mut data = icon_data(hwnd);
    data.uFlags = NIF_TIP;
    // The tooltip holds 127 units. Longer text is cut, not dropped.
    fill(&mut data.szTip, &format!("Phonon: {text}"));
    unsafe { Shell_NotifyIconW(NIM_MODIFY, &data) };
}

/// The current tooltip text, without the prefix.
pub fn status() -> String {
    STATUS.lock().map(|text| text.clone()).unwrap_or_default()
}

/// Show a balloon. `is_error` picks the icon Windows draws beside it.
pub fn notify(title: &str, body: &str, is_error: bool) {
    let Some(hwnd) = hwnd() else {
        eprintln!("phonon: {title}: {body}");
        return;
    };
    let mut data = icon_data(hwnd);
    data.uFlags = NIF_INFO;
    fill(&mut data.szInfoTitle, title);
    fill(&mut data.szInfo, body);
    data.dwInfoFlags = if is_error { NIIF_ERROR } else { NIIF_INFO };
    unsafe { Shell_NotifyIconW(NIM_MODIFY, &data) };
}

/// Remove the icon. Without this Windows leaves a dead icon behind until the
/// user hovers over it.
pub fn destroy() {
    let Some(hwnd) = hwnd() else { return };
    let data = icon_data(hwnd);
    unsafe { Shell_NotifyIconW(NIM_DELETE, &data) };
}

/// Run the message loop until the user quits. The keyboard hook needs this loop.
pub fn run_message_loop() {
    let mut message: MSG = unsafe { std::mem::zeroed() };
    loop {
        let result = unsafe { GetMessageW(&mut message, std::ptr::null_mut(), 0, 0) };
        if result <= 0 {
            break;
        }
        unsafe {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

/// Ask the loop to stop, from any thread.
pub fn post_quit() {
    if let Some(hwnd) = hwnd() {
        unsafe {
            windows_sys::Win32::UI::WindowsAndMessaging::PostMessageW(hwnd, WM_DESTROY, 0, 0)
        };
    }
}

fn send(command: Command) {
    if let Some(sender) = COMMANDS.get().and_then(|sender| sender.lock().ok()) {
        let _ = sender.send(command);
    }
}

fn show_menu(hwnd: HWND) {
    let menu: HMENU = unsafe { CreatePopupMenu() };
    if menu.is_null() {
        return;
    }
    let heading = wide(&format!("Phonon: {}", status()));
    let toggle = wide("Start or stop dictation");
    let folder = wide("Open the Phonon folder");
    let quit = wide("Quit Phonon");
    unsafe {
        AppendMenuW(
            menu,
            MF_STRING | MF_DISABLED | MF_GRAYED,
            0,
            heading.as_ptr(),
        );
        AppendMenuW(menu, MF_SEPARATOR, 0, std::ptr::null());
        AppendMenuW(menu, MF_STRING, ID_TOGGLE, toggle.as_ptr());
        AppendMenuW(menu, MF_STRING, ID_OPEN_FOLDER, folder.as_ptr());
        AppendMenuW(menu, MF_SEPARATOR, 0, std::ptr::null());
        AppendMenuW(menu, MF_STRING, ID_QUIT, quit.as_ptr());

        let mut point: POINT = std::mem::zeroed();
        GetCursorPos(&mut point);
        // Without this the menu does not close when the user clicks elsewhere.
        SetForegroundWindow(hwnd);
        TrackPopupMenu(
            menu,
            TPM_RIGHTBUTTON | TPM_BOTTOMALIGN,
            point.x,
            point.y,
            0,
            hwnd,
            std::ptr::null(),
        );
        windows_sys::Win32::UI::WindowsAndMessaging::PostMessageW(hwnd, WM_NULL, 0, 0);
        DestroyMenu(menu);
    }
}

unsafe extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match message {
        WM_TRAY => {
            if lparam as u32 == WM_RBUTTONUP {
                show_menu(hwnd);
            }
            0
        }
        WM_COMMAND => {
            match wparam & 0xFFFF {
                ID_TOGGLE => send(Command::Toggle),
                ID_OPEN_FOLDER => send(Command::OpenFolder),
                ID_QUIT => send(Command::Quit),
                _ => {}
            }
            0
        }
        WM_REFRESH => 0,
        WM_DESTROY => {
            PostQuitMessage(0);
            0
        }
        _ => DefWindowProcW(hwnd, message, wparam, lparam),
    }
}
