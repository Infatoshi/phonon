//! Tray icon (StatusNotifierItem over D-Bus) and desktop notifications.
//!
//! KDE, Cinnamon, XFCE and Ubuntu's GNOME (AppIndicator extension, on by
//! default) show it. Stock GNOME without the extension shows nothing; Phonon
//! still works, and notifications still appear.

use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock};

use ksni::blocking::{Handle, TrayMethods};

#[derive(Debug, Clone)]
pub enum Command {
    Toggle,
    OpenFolder,
    Quit,
    /// Turn local history on or off.
    ToggleHistory,
    /// Delete the history log and the kept recordings.
    ClearHistory,
    /// Copy this text to the clipboard: the full text of the "Recent dictations"
    /// item clicked, or the last dictation.
    CopyRecent(String),
}

struct PhononTray {
    status: String,
    recording: bool,
    commands: Sender<Command>,
    /// Local history as the menu shows it; the worker owns the files (`set_history`).
    history: bool,
    recent: Vec<String>,
    /// The last dictation's final text, for "Copy last dictation": in memory
    /// only, whatever Keep history says, so a paste that went nowhere is never
    /// lost and nothing is written to disk. Gone when Phonon quits.
    last: Option<String>,
    /// The dictation indicator's current frame (src/indicator.rs), shown in place of the icon;
    /// `None` at rest.
    frame: Option<ksni::Icon>,
}

impl PhononTray {
    /// "Copy last dictation", greyed until there is one.
    fn copy_last_item(&self) -> ksni::MenuItem<Self> {
        let last = self.last.clone();
        ksni::menu::StandardItem {
            label: "Copy last dictation".into(),
            enabled: last.is_some(),
            activate: Box::new(move |t: &mut Self| {
                if let Some(text) = &last {
                    let _ = t.commands.send(Command::CopyRecent(text.clone()));
                }
            }),
            ..Default::default()
        }
        .into()
    }

    /// "Keep history", "Recent dictations" and "Clear history".
    fn history_items(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::{CheckmarkItem, StandardItem, SubMenu};
        // Each item carries the text it shows, so a refresh of `recent` after the
        // menu was sent to the panel cannot change what a click copies.
        let shown = crate::history::Shown::of(&self.recent);
        let mut recent: Vec<ksni::MenuItem<Self>> = shown
            .labels()
            .into_iter()
            .enumerate()
            .filter_map(|(index, label)| Some((label, shown.text(index)?)))
            .map(|(label, text)| {
                StandardItem {
                    // an underscore marks the access key; a doubled one is a literal underscore
                    label: label.replace('_', "__"),
                    activate: Box::new(move |t: &mut Self| {
                        let _ = t.commands.send(Command::CopyRecent(text.clone()));
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
    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        self.frame.iter().cloned().collect()
    }
    fn icon_name(&self) -> String {
        // a host shows the named icon in preference to a pixmap, so name none while a frame shows
        if self.frame.is_some() {
            ""
        } else if self.recording {
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
                    let _ = t.commands.send(command.clone());
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
            self.copy_last_item(),
            StandardItem {
                label: "Settings\u{2026}".into(),
                activate: Box::new(|_: &mut Self| crate::window::open()),
                ..Default::default()
            }
            .into(),
            item("Open Phonon folder", Command::OpenFolder),
        ];
        items.extend(self.history_items());
        if let Some(found) = crate::update::available() {
            items.push(
                StandardItem {
                    label: format!("Download Phonon {}\u{2026}", found.title),
                    activate: Box::new(move |_: &mut Self| crate::update::open_page(&found.page)),
                    ..Default::default()
                }
                .into(),
            );
        }
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
        last: None,
        frame: None,
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
    // the dictation indicator (src/indicator.rs): sleeps until the first key press
    crate::indicator::start(crate::indicator::Sink {
        size: || 32,
        show: show_frame,
        reduced_motion,
    });
}

/// Put an indicator frame (straight-alpha RGBA, `size` square) in the tray, or with `None` the
/// normal icon back. The tray protocol wants ARGB32 in network byte order.
fn show_frame(rgba: Option<&[u8]>, size: u32) {
    let frame = rgba.map(|rgba| ksni::Icon {
        width: size as i32,
        height: size as i32,
        data: rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[3], p[0], p[1], p[2]])
            .collect(),
    });
    if let Some(Ok(guard)) = HANDLE.get().map(|h| h.lock()) {
        if let Some(handle) = guard.as_ref() {
            handle.update(move |t| t.frame = frame);
        }
    }
}

/// The desktop's animations setting where one can be read (GNOME's `enable-animations`, which GNOME
/// Shell, Cinnamon and MATE follow); a desktop without it counts as animations on. The indicator
/// calls this off its drawing thread, at start and at most once a minute after a dictation; a
/// `gsettings` that does not answer within a second is killed and counts as animations on.
fn reduced_motion() -> bool {
    use std::io::Read;
    use std::time::{Duration, Instant};
    let Ok(mut child) = std::process::Command::new("gsettings")
        .args(["get", "org.gnome.desktop.interface", "enable-animations"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        return false;
    };
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
        }
    }
    let mut out = String::new();
    if let Some(mut stdout) = child.stdout.take() {
        let _ = stdout.read_to_string(&mut out);
    }
    out.trim() == "false"
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

/// Rebuild the menu (an update check found a release: `update::available`).
pub fn refresh() {
    if let Some(Ok(guard)) = HANDLE.get().map(|h| h.lock()) {
        if let Some(handle) = guard.as_ref() {
            handle.update(|_| {});
        }
    }
}

/// Show history as on or off, with these recent dictations (newest first).
pub fn set_history(on: bool, recent: Vec<String>) {
    if let Some(Ok(guard)) = HANDLE.get().map(|h| h.lock()) {
        if let Some(handle) = guard.as_ref() {
            handle.update(move |t| {
                t.history = on;
                t.recent = recent;
            });
        }
    }
}

/// Remember (or, with `None`, forget) the last dictation for "Copy last dictation".
pub fn set_last(text: Option<String>) {
    if let Some(Ok(guard)) = HANDLE.get().map(|h| h.lock()) {
        if let Some(handle) = guard.as_ref() {
            handle.update(move |t| t.last = text);
        }
    }
}

pub fn status() -> String {
    STATUS.lock().unwrap().clone()
}

/// A notification, also logged to stderr in full. stderr is the systemd user
/// journal for an autostarted Phonon, so a body that could hold dictated text
/// goes through `notify_private` instead.
pub fn notify(title: &str, body: &str, is_error: bool) {
    eprintln!("phonon: {title}: {body}");
    show(title, body, is_error);
}

/// A notification whose body stays out of the log: only the title is logged.
pub fn notify_private(title: &str, body: &str, is_error: bool) {
    eprintln!("phonon: {title}");
    show(title, body, is_error);
}

fn show(title: &str, body: &str, is_error: bool) {
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
