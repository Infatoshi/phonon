//! The notification-area icon, its menu, and its balloon messages.
//!
//! Phonon has no window on Windows. The tray icon is the whole user interface:
//! its tooltip carries first-run download progress and the current state, and
//! its balloons announce the things a user must not miss, such as a failed
//! download or the moment dictation becomes usable.
//!
//! The icon is owned by a message-only window. That window also runs the message
//! loop the keyboard hook needs, so both live on the main thread.

use std::collections::HashSet;
use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock};

use anyhow::{bail, Result};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows_sys::Win32::Security::{
    GetTokenInformation, TokenElevation, TOKEN_ELEVATION, TOKEN_QUERY,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, OpenProcess, OpenProcessToken, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows_sys::Win32::UI::Accessibility::{SetWinEventHook, HWINEVENTHOOK};
use windows_sys::Win32::UI::Shell::{
    Shell_NotifyIconW, NIF_ICON, NIF_INFO, NIF_MESSAGE, NIF_TIP, NIIF_ERROR, NIIF_INFO, NIM_ADD,
    NIM_DELETE, NIM_MODIFY, NOTIFYICONDATAW,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    AppendMenuW, CreatePopupMenu, CreateWindowExW, DefWindowProcW, DestroyMenu, DispatchMessageW,
    GetCursorPos, GetForegroundWindow, GetMessageW, GetSystemMetrics, GetWindowThreadProcessId,
    LoadIconW, LoadImageW, PostQuitMessage, RegisterClassW, SetForegroundWindow, TrackPopupMenu,
    TranslateMessage, EVENT_SYSTEM_FOREGROUND, HMENU, IDI_APPLICATION, IMAGE_ICON, MF_CHECKED,
    MF_DISABLED, MF_GRAYED, MF_POPUP, MF_SEPARATOR, MF_STRING, MSG, SM_CXSMICON, SM_CYSMICON,
    TPM_BOTTOMALIGN, TPM_RIGHTBUTTON, WINEVENT_OUTOFCONTEXT, WM_APP, WM_COMMAND, WM_DESTROY,
    WM_NULL, WM_RBUTTONUP, WNDCLASSW, WS_OVERLAPPED,
};

/// Sent to the message-only window when the icon is clicked.
const WM_TRAY: u32 = WM_APP + 1;
/// Posted by the worker thread when the tooltip text has changed.
const WM_REFRESH: u32 = WM_APP + 2;
/// Posted by the keyboard hook the first time it sees the Copilot key.
const WM_COPILOT: u32 = WM_APP + 3;

const ID_TOGGLE: usize = 1;
const ID_OPEN_FOLDER: usize = 2;
const ID_QUIT: usize = 3;
/// "Dictation key" submenu items: ID_KEY + index into `hook::CHOICES`.
const ID_KEY: usize = 100;

/// What the user picked from the menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Start or stop dictation without touching the keyboard.
    Toggle,
    /// Open the Phonon folder in Explorer.
    OpenFolder,
    /// Leave.
    Quit,
    /// The dictation key changed (menu pick or Copilot-key fallback).
    KeyChanged,
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
    // The Phonon icon that build.rs embeds as resource 1 (winresource's
    // application icon ID). A build without the resource gets the stock icon.
    // MAKEINTRESOURCEW(1), at the notification area's small-icon size: LoadIconW
    // only loads the large size and returned nothing for a 256-pixel-only icon.
    data.hIcon = unsafe {
        LoadImageW(
            instance,
            std::ptr::without_provenance(1),
            IMAGE_ICON,
            GetSystemMetrics(SM_CXSMICON),
            GetSystemMetrics(SM_CYSMICON),
            0,
        )
    } as _;
    if data.hIcon.is_null() {
        data.hIcon = unsafe { LoadIconW(std::ptr::null_mut(), IDI_APPLICATION) };
    }
    fill(&mut data.szTip, "Phonon: starting");
    if unsafe { Shell_NotifyIconW(NIM_ADD, &data) } == 0 {
        bail!("Shell_NotifyIconW could not add the icon");
    }
    watch_foreground();
    Ok(())
}

/// Whether a process runs with an elevated (administrator) token. `None` when
/// Windows will not say, for example for protected system processes.
fn elevated(process: HANDLE) -> Option<bool> {
    let mut token: HANDLE = std::ptr::null_mut();
    if unsafe { OpenProcessToken(process, TOKEN_QUERY, &mut token) } == 0 {
        return None;
    }
    let mut elevation = TOKEN_ELEVATION { TokenIsElevated: 0 };
    let mut size = 0u32;
    let ok = unsafe {
        GetTokenInformation(
            token,
            TokenElevation,
            (&mut elevation as *mut TOKEN_ELEVATION).cast(),
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut size,
        )
    };
    unsafe { CloseHandle(token) };
    (ok != 0).then_some(elevation.TokenIsElevated != 0)
}

fn self_elevated() -> bool {
    static SELF: OnceLock<bool> = OnceLock::new();
    *SELF.get_or_init(|| elevated(unsafe { GetCurrentProcess() }).unwrap_or(false))
}

/// Windows keeps a process without administrator rights from sending keys to,
/// or even hearing keys typed into, an administrator window (UIPI). The hook
/// never sees the dictation key there, so without a word the key would just do
/// nothing. Say so the first time each such window comes to the front.
fn watch_foreground() {
    if self_elevated() {
        return;
    }
    // Out of context: the callback runs on this thread, from the message loop.
    let hook = unsafe {
        SetWinEventHook(
            EVENT_SYSTEM_FOREGROUND,
            EVENT_SYSTEM_FOREGROUND,
            std::ptr::null_mut(),
            Some(on_foreground),
            0,
            0,
            WINEVENT_OUTOFCONTEXT,
        )
    };
    if hook.is_null() {
        eprintln!("phonon: SetWinEventHook failed; administrator windows will not be flagged");
    }
}

static WARNED: Mutex<Option<HashSet<u32>>> = Mutex::new(None);

unsafe extern "system" fn on_foreground(
    _hook: HWINEVENTHOOK,
    _event: u32,
    window: HWND,
    _object: i32,
    _child: i32,
    _thread: u32,
    _time: u32,
) {
    let window = if window.is_null() {
        GetForegroundWindow()
    } else {
        window
    };
    let mut pid = 0u32;
    GetWindowThreadProcessId(window, &mut pid);
    if pid == 0 || pid == std::process::id() {
        return;
    }
    let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
    if process.is_null() {
        return;
    }
    let is_admin = elevated(process);
    CloseHandle(process);
    if is_admin != Some(true) {
        return;
    }
    let first = WARNED
        .lock()
        .map(|mut seen| seen.get_or_insert_with(HashSet::new).insert(pid))
        .unwrap_or(false);
    if first {
        eprintln!("phonon: foreground pid {pid} is elevated");
        notify(
            "Phonon can't type into admin windows unless it runs as admin",
            "This window runs as administrator, so Windows keeps Phonon from hearing the dictation key or typing here. Start Phonon as administrator to dictate into it.",
            false,
        );
    }
}

/// Called from the keyboard hook: handle the Copilot key on this thread, out of
/// the hook's time budget.
pub fn post_copilot_key() {
    if let Some(hwnd) = hwnd() {
        unsafe {
            windows_sys::Win32::UI::WindowsAndMessaging::PostMessageW(hwnd, WM_COPILOT, 0, 0)
        };
    }
}

fn pick_key(index: usize) {
    let Some(&(_, code)) = super::hook::CHOICES.get(index) else {
        return;
    };
    match super::hook::choose(code) {
        Ok(()) => {
            let label = super::hook::key_label(code);
            notify(
                &format!("Dictation key: {label}"),
                &format!("Hold {label} to dictate, or tap it to start and tap again to stop."),
                false,
            );
            send(Command::KeyChanged);
        }
        Err(error) => notify(
            "Phonon could not change the key",
            &format!("{error:#}"),
            true,
        ),
    }
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
    let keys_title = wide(&format!("Dictation key: {}", super::hook::current_key().0));
    let keys: HMENU = unsafe { CreatePopupMenu() };
    let (_, current) = super::hook::current_key();
    let labels: Vec<Vec<u16>> = super::hook::CHOICES
        .iter()
        .map(|(_, code)| wide(super::hook::key_label(*code)))
        .collect();
    unsafe {
        for (index, ((_, code), label)) in super::hook::CHOICES.iter().zip(&labels).enumerate() {
            let checked = if *code == current { MF_CHECKED } else { 0 };
            AppendMenuW(keys, MF_STRING | checked, ID_KEY + index, label.as_ptr());
        }
        AppendMenuW(
            menu,
            MF_STRING | MF_DISABLED | MF_GRAYED,
            0,
            heading.as_ptr(),
        );
        AppendMenuW(menu, MF_SEPARATOR, 0, std::ptr::null());
        AppendMenuW(menu, MF_STRING, ID_TOGGLE, toggle.as_ptr());
        AppendMenuW(
            menu,
            MF_STRING | MF_POPUP,
            keys as usize,
            keys_title.as_ptr(),
        );
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
                id if (ID_KEY..ID_KEY + super::hook::CHOICES.len()).contains(&id) => {
                    pick_key(id - ID_KEY)
                }
                _ => {}
            }
            0
        }
        WM_REFRESH => 0,
        WM_COPILOT => {
            if let Some(label) = super::hook::on_copilot_key() {
                notify(
                    &format!("Dictation key: {label}"),
                    &format!("This keyboard has a Copilot key instead of Right Ctrl, so Phonon uses {label}. Hold it to dictate; change it from the tray menu."),
                    false,
                );
                send(Command::KeyChanged);
            }
            0
        }
        WM_DESTROY => {
            PostQuitMessage(0);
            0
        }
        _ => DefWindowProcW(hwnd, message, wparam, lparam),
    }
}
