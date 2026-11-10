//! Tray icon (StatusNotifierItem over D-Bus) and desktop notifications.
//!
//! KDE, Cinnamon, XFCE and Ubuntu's GNOME (AppIndicator extension, on by
//! default) show it. Stock GNOME without the extension shows nothing; Phonon
//! still works, and notifications still appear.

use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock};

use ksni::blocking::{Handle, TrayMethods};

#[derive(Debug, Clone, Copy)]
pub enum Command {
    Toggle,
    OpenFolder,
    Quit,
    /// Turn local history on or off.
    ToggleHistory,
    /// Delete the history log and the kept recordings.
    ClearHistory,
    /// Copy this entry of the recent list (see `recent_text`) to the clipboard.
    CopyRecent(usize),
}

struct PhononTray {
    status: String,
    recording: bool,
    commands: Sender<Command>,
    /// Local history as the menu shows it; the worker owns the files (`set_history`).
    history: bool,
    recent: Vec<String>,
}

impl PhononTray {
    /// "Keep history", "Recent dictations" and "Clear history".
    fn history_items(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::{CheckmarkItem, StandardItem, SubMenu};
        let mut recent: Vec<ksni::MenuItem<Self>> = self
            .recent
            .iter()
            .enumerate()
            .map(|(index, text)| {
                StandardItem {
                    // an underscore marks the access key; a doubled one is a literal underscore
                    label: crate::history::label(text).replace('_', "__"),
                    activate: Box::new(move |t: &mut Self| {
                        let _ = t.commands.send(Command::CopyRecent(index));
                    }),
                    ..Default::default()
                }
                .into()
            })
            .collect();
        if recent.is_empty() {
            recent.push(
                StandardItem {
                    label: if self.history {
                        "Nothing yet"
                    } else {
                        "Turn on Keep history to list them"
                    }
                    .into(),
                    enabled: false,
                    ..Default::default()
                }
                .into(),
            );
        }
        vec![
            ksni::MenuItem::Separator,
            CheckmarkItem {
                label: "Keep history".into(),
                checked: self.history,
                activate: Box::new(|t: &mut Self| {
                    let _ = t.commands.send(Command::ToggleHistory);
                }),
                ..Default::default()
            }
            .into(),
            SubMenu {
                label: "Recent dictations".into(),
                submenu: recent,
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: "Clear history".into(),
                activate: Box::new(|t: &mut Self| {
                    let _ = t.commands.send(Command::ClearHistory);
                }),
                ..Default::default()
            }
            .into(),
            ksni::MenuItem::Separator,
        ]
    }
}

impl ksni::Tray for PhononTray {
    fn id(&self) -> String {
        "phonon".into()
    }
    fn title(&self) -> String {
        "Phonon".into()
    }
    fn icon_name(&self) -> String {
        if self.recording {
            "audio-input-microphone"
        } else {
            idle_icon()
        }
        .into()
    }
    fn tool_tip(&self) -> ksni::ToolTip {
        ksni::ToolTip {
            title: "Phonon".into(),
            description: self.status.clone(),
            ..Default::default()
        }
    }
    fn activate(&mut self, _x: i32, _y: i32) {
        let _ = self.commands.send(Command::Toggle);
    }
    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::StandardItem;
        let item = |label: &str, command: Command| {
            StandardItem {
                label: label.into(),
                activate: Box::new(move |t: &mut Self| {
                    let _ = t.commands.send(command);
                }),
                ..Default::default()
            }
            .into()
        };
        let mut items: Vec<ksni::MenuItem<Self>> = vec![
            StandardItem {
                // the status lines are written lowercase for "Phonon: ..." elsewhere; a menu line starts capitalized
                label: capitalized(&self.status),
                enabled: false,
                ..Default::default()
            }
            .into(),
            ksni::MenuItem::Separator,
            item(
                if self.recording {
                    "Stop dictation"
                } else {
                    "Start dictation"
                },
                Command::Toggle,
            ),
            item("Open Phonon folder", Command::OpenFolder),
        ];
        items.extend(self.history_items());
        items.push(item("Quit Phonon", Command::Quit));
        items
    }
}

fn capitalized(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// Phonon's own icon when the .deb installed it into the icon theme, otherwise a stock microphone.
fn idle_icon() -> &'static str {
    static INSTALLED: OnceLock<bool> = OnceLock::new();
    let installed = *INSTALLED.get_or_init(|| {
        [
            "/usr/share/icons/hicolor/48x48/apps/phonon.png",
            "/usr/local/share/icons/hicolor/48x48/apps/phonon.png",
        ]
        .iter()
        .any(|p| std::path::Path::new(p).exists())
    });
    if installed {
        "phonon"
    } else {
        "audio-input-microphone-symbolic"
    }
}

static HANDLE: OnceLock<Mutex<Option<Handle<PhononTray>>>> = OnceLock::new();
static STATUS: Mutex<String> = Mutex::new(String::new());

/// Show the tray icon. A desktop without a tray host is not an error.
pub fn create(commands: Sender<Command>) {
    let tray = PhononTray {
        status: "starting".into(),
        recording: false,
        commands,
        history: false,
        recent: Vec::new(),
    };
    // At login Phonon can start before the shell's tray host (Ubuntu's AppIndicator extension) is up;
    // without this the icon never appeared (Ubuntu 26.04 VM, 2026-11-06). ksni registers once the host appears.
    let handle = match tray.assume_sni_available(true).spawn() {
        Ok(handle) => Some(handle),
        Err(error) => {
            eprintln!("phonon: no tray icon ({error}); notifications only");
            None
        }
    };
    let _ = HANDLE.set(Mutex::new(handle));
}

pub fn set_status(text: &str) {
    *STATUS.lock().unwrap() = text.to_string();
    let recording = text.starts_with("recording");
    if let Some(Ok(guard)) = HANDLE.get().map(|h| h.lock()) {
        if let Some(handle) = guard.as_ref() {
            let text = text.to_string();
            handle.update(move |t| {
                t.status = text;
                t.recording = recording;
            });
        }
    }
}

static RECENT: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Show history as on or off, with these recent dictations (newest first).
pub fn set_history(on: bool, recent: Vec<String>) {
    *RECENT.lock().unwrap() = recent.clone();
    if let Some(Ok(guard)) = HANDLE.get().map(|h| h.lock()) {
        if let Some(handle) = guard.as_ref() {
            handle.update(move |t| {
                t.history = on;
                t.recent = recent;
            });
        }
    }
}

/// The full text behind "Recent dictations" item `index`, as the menu listed it.
pub fn recent_text(index: usize) -> Option<String> {
    RECENT.lock().unwrap().get(index).cloned()
}

pub fn status() -> String {
    STATUS.lock().unwrap().clone()
}

pub fn notify(title: &str, body: &str, is_error: bool) {
    eprintln!("phonon: {title}: {body}");
    let urgency = if is_error { "critical" } else { "normal" };
    let icon = match idle_icon() {
        "phonon" => "--icon=phonon",
        _ => "--icon=audio-input-microphone",
    };
    let _ = std::process::Command::new("notify-send")
        .args(["--app-name=Phonon", icon, "-u", urgency, title, body])
        .spawn()
        .and_then(|mut c| c.wait());
}
