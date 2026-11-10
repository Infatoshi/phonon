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
}

struct PhononTray {
    status: String,
    recording: bool,
    /// The writing mode the menu shows checked; the file in the data folder is
    /// what dictations read.
    mode: &'static str,
    commands: Sender<Command>,
}

impl PhononTray {
    /// Save a pick from the "Writing mode" submenu. A small file write, so it
    /// happens right here in the menu callback and the next dictation sees it.
    fn pick_mode(&mut self, index: usize) {
        let Some(mode) = crate::mode::names().nth(index) else {
            return;
        };
        match crate::mode::choose(mode) {
            Ok(()) => self.mode = mode,
            Err(error) => eprintln!("phonon: could not change the writing mode: {error:#}"),
        }
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
        use ksni::menu::{RadioGroup, RadioItem, StandardItem, SubMenu};
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
        vec![
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
            SubMenu {
                label: format!("Writing mode: {}", crate::mode::label(self.mode)),
                submenu: vec![RadioGroup {
                    selected: crate::mode::names()
                        .position(|name| name == self.mode)
                        .unwrap_or(0),
                    select: Box::new(|t: &mut Self, index| t.pick_mode(index)),
                    options: crate::mode::names()
                        .map(|name| RadioItem {
                            label: crate::mode::menu_label(name),
                            ..Default::default()
                        })
                        .collect(),
                }
                .into()],
                ..Default::default()
            }
            .into(),
            item("Open Phonon folder", Command::OpenFolder),
            item("Quit Phonon", Command::Quit),
        ]
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
        mode: crate::mode::current(),
        commands,
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
