//! The settings window on Windows and Linux: `phonon-win settings`, a separate
//! process the tray starts, so the resident dictation process never loads a UI
//! toolkit and a window crash cannot take dictation down.
//!
//! egui through eframe. Rows follow the Mac's SettingsView
//! (bar/Sources/MainWindow.swift), limited to what this engine acts on. Every
//! change is written to settings.json at once (settings.rs); the running app
//! reads it at the next dictation, and picks up a new key on its next tick.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use eframe::egui;

use crate::history::{self, Entry};
use crate::settings::{self, ClipboardAfterPaste, Settings};
use crate::{autostart, dictionary, manifest, mode, paths, single};

pub const TITLE: &str = "Phonon settings";

/// Start the settings window as its own process. Called from the tray.
pub fn open() {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(error) => {
            eprintln!("phonon: cannot find this executable to open settings: {error}");
            return;
        }
    };
    let mut command = std::process::Command::new(exe);
    command.arg("settings");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: the executable is a console program; no console for the window
        command.creation_flags(0x0800_0000);
    }
    match command.spawn() {
        // reap it when it closes
        Ok(mut child) => {
            std::thread::spawn(move || {
                let _ = child.wait();
            });
        }
        Err(error) => eprintln!("phonon: could not open settings: {error}"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    General,
    History,
    Dictionary,
}

impl Page {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "general" | "settings" => Some(Self::General),
            "history" => Some(Self::History),
            "dictionary" => Some(Self::Dictionary),
            _ => None,
        }
    }
}

/// Run the window until it is closed. A second copy raises the first (Windows)
/// or exits.
pub fn run(page: Option<Page>) -> Result<()> {
    #[cfg(windows)]
    unsafe {
        // started from a terminal the console stays; started by the tray there is none
        let _ = windows_sys::Win32::System::Console::FreeConsole();
    }
    let root = paths::data_root();
    let Some(_guard) = single::claim_settings_window(&root)? else {
        eprintln!("phonon: the settings window is already open");
        raise_existing();
        return Ok(());
    };
    #[cfg(target_os = "linux")]
    prefer_x11_on_gnome();
    let viewport = egui::ViewportBuilder::default()
        .with_title(TITLE)
        // Wayland and X11 match the window to phonon.desktop (its icon and name)
        .with_app_id("phonon")
        .with_inner_size([720.0, 620.0])
        .with_min_inner_size([560.0, 420.0]);
    // Phonon's icon in the title bar, the taskbar and alt-tab, not eframe's default "e".
    // eframe already links the PNG decoder for this.
    let viewport =
        match eframe::icon_data::from_png_bytes(include_bytes!("../linux/icons/phonon-128.png")) {
            Ok(icon) => viewport.with_icon(icon),
            Err(error) => {
                eprintln!("phonon: the window icon did not decode: {error}");
                viewport
            }
        };
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    let app = App::new(root, page.unwrap_or(Page::General));
    eframe::run_native(
        TITLE,
        options,
        Box::new(move |cc| {
            text_sizes(&cc.egui_ctx);
            cc.egui_ctx.set_theme(egui::ThemePreference::System);
            #[cfg(target_os = "linux")]
            linux_theme::follow(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
    .map_err(|error| anyhow!("open the settings window: {error}"))
}

/// egui's sizes are for dense tool panels; a settings window reads better a
/// step larger, nearer the desktop's own dialogs.
fn text_sizes(ctx: &egui::Context) {
    use egui::{FontFamily, FontId, TextStyle};
    let sizes = [
        (TextStyle::Heading, 20.0, FontFamily::Proportional),
        (TextStyle::Body, 14.5, FontFamily::Proportional),
        (TextStyle::Button, 14.5, FontFamily::Proportional),
        (TextStyle::Small, 12.0, FontFamily::Proportional),
        (TextStyle::Monospace, 13.5, FontFamily::Monospace),
    ];
    ctx.all_styles_mut(|style| {
        for (text_style, size, family) in sizes.clone() {
            style
                .text_styles
                .insert(text_style, FontId::new(size, family));
        }
        style.spacing.interact_size.y = 24.0;
        style.spacing.item_spacing.y = 6.0;
    });
}

/// GNOME draws no title bar for a Wayland window that does not draw its own,
/// and winit's stand-in has no title. Under XWayland, GNOME draws its own
/// title bar, in the desktop's light or dark style. Other Wayland desktops (KDE)
/// decorate Wayland windows themselves, so they stay on Wayland.
#[cfg(target_os = "linux")]
fn prefer_x11_on_gnome() {
    let gnome = std::env::var("XDG_CURRENT_DESKTOP")
        .map(|desktop| desktop.to_ascii_uppercase().contains("GNOME"))
        .unwrap_or(false);
    if gnome && std::env::var_os("DISPLAY").is_some() {
        // one thread so far: nothing else reads the environment concurrently
        std::env::remove_var("WAYLAND_DISPLAY");
    }
}

#[cfg(windows)]
fn raise_existing() {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        FindWindowW, SetForegroundWindow, ShowWindow, SW_RESTORE,
    };
    let title: Vec<u16> = TITLE.encode_utf16().chain([0]).collect();
    // SAFETY: a NUL-terminated title; a null class matches any window class.
    unsafe {
        let hwnd = FindWindowW(std::ptr::null(), title.as_ptr());
        if !hwnd.is_null() {
            ShowWindow(hwnd, SW_RESTORE);
            SetForegroundWindow(hwnd);
        }
    }
}

#[cfg(not(windows))]
fn raise_existing() {}

/// The desktop's light or dark choice on Linux, through the XDG desktop portal
/// (GNOME, KDE and others), with GNOME's own key as a fallback. winit does not
/// report it on every desktop, so this asks directly and keeps asking, so a
/// switch in the desktop's settings reaches an open window.
#[cfg(target_os = "linux")]
mod linux_theme {
    use eframe::egui;
    use std::process::Command;

    fn output(program: &str, args: &[&str]) -> Option<String> {
        let out = Command::new(program).args(args).output().ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    /// `Some(true)` for dark, `Some(false)` for light, `None` when nothing says.
    pub fn dark() -> Option<bool> {
        // org.freedesktop.appearance color-scheme: 0 no preference, 1 dark, 2 light
        if let Some(text) = output(
            "busctl",
            &[
                "--user",
                "call",
                "org.freedesktop.portal.Desktop",
                "/org/freedesktop/portal/desktop",
                "org.freedesktop.portal.Settings",
                "ReadOne",
                "ss",
                "org.freedesktop.appearance",
                "color-scheme",
            ],
        ) {
            match text.split_whitespace().last() {
                Some("1") => return Some(true),
                Some("2") => return Some(false),
                _ => {}
            }
        }
        let scheme = output(
            "gsettings",
            &["get", "org.gnome.desktop.interface", "color-scheme"],
        )?;
        Some(scheme.contains("dark"))
    }

    pub fn follow(ctx: &egui::Context) {
        let ctx = ctx.clone();
        let apply = move |dark: Option<bool>| match dark {
            Some(true) => ctx.set_theme(egui::ThemePreference::Dark),
            Some(false) => ctx.set_theme(egui::ThemePreference::Light),
            None => ctx.set_theme(egui::ThemePreference::System),
        };
        let first = dark();
        apply(first);
        std::thread::spawn(move || {
            let mut last = first;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let now = dark();
                if now != last {
                    apply(now);
                    last = now;
                }
            }
        });
    }
}

/// How the window shows a time: "just now", "5 min ago", "3 days ago". Local
/// clock formatting would need the time zone database; relative times do not.
pub fn ago(then_ms: u64, now_ms: u64) -> String {
    let seconds = now_ms.saturating_sub(then_ms) / 1000;
    match seconds {
        0..=59 => "just now".into(),
        60..=3599 => format!("{} min ago", seconds / 60),
        3600..=86_399 => {
            let hours = seconds / 3600;
            format!("{hours} hour{} ago", if hours == 1 { "" } else { "s" })
        }
        _ => {
            let days = seconds / 86_400;
            format!("{days} day{} ago", if days == 1 { "" } else { "s" })
        }
    }
}

/// The label for a retention choice, as the Mac words it.
pub fn retention_label(days: u32) -> String {
    match days {
        0 => "Until I delete them".into(),
        days => format!("{days} days"),
    }
}

pub fn clipboard_label(rule: ClipboardAfterPaste) -> &'static str {
    match rule {
        ClipboardAfterPaste::Restore => "What you copied before",
        ClipboardAfterPaste::Dictation => "Your dictation",
    }
}

/// How many dictations the History page lists.
const HISTORY_SHOWN: usize = 200;

/// The newest recording, when it was written in the last few seconds: a
/// dictation in progress (Linux writes it as it records), which Clear must leave.
fn recording_in_progress(root: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(root.join(paths::RECORDINGS)).ok()?;
    entries
        .flatten()
        .filter_map(|entry| {
            let modified = entry.metadata().ok()?.modified().ok()?;
            let age = modified.elapsed().unwrap_or_default();
            (age < Duration::from_secs(10)).then(|| (modified, entry.path()))
        })
        .max_by_key(|(modified, _)| *modified)
        .map(|(_, path)| path)
}

struct App {
    root: PathBuf,
    page: Page,
    shortcut: String,
    style: &'static str,
    clipboard: ClipboardAfterPaste,
    history_on: bool,
    retention: u32,
    login: bool,
    engine: Option<String>,
    /// Newest first.
    history: Vec<Entry>,
    history_watch: settings::Watch,
    settings_watch: settings::Watch,
    last_poll: Instant,
    confirm_clear: bool,
    /// The last message for the user, and whether it is an error.
    message: Option<(String, bool)>,
    /// The dictionary as on disk, its entries sorted by phrase.
    dictionary: dictionary::DictionaryFile,
    dictionary_watch: settings::Watch,
    /// The Dictionary page's input line: a word, or "from => to".
    new_entry: String,
}

/// What the Dictionary page's input line asks for: a word to spell as written,
/// or a replacement written "heard => written" (also "->").
#[derive(Debug, PartialEq, Eq)]
pub enum NewEntry {
    Word(String),
    Replacement(String, String),
}

pub fn parse_entry(text: &str) -> Option<NewEntry> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    for arrow in ["=>", "->", "\u{2192}"] {
        if let Some((from, to)) = text.split_once(arrow) {
            let (from, to) = (from.trim(), to.trim());
            if from.is_empty() || to.is_empty() {
                return None;
            }
            return Some(NewEntry::Replacement(from.into(), to.into()));
        }
    }
    Some(NewEntry::Word(text.into()))
}

impl App {
    /// A window on data root `root` before anything is read.
    fn blank(root: PathBuf, page: Page) -> Self {
        Self {
            root,
            page,
            shortcut: settings::DEFAULT_SHORTCUT.into(),
            style: crate::polish::DEFAULT_MODE,
            clipboard: ClipboardAfterPaste::Restore,
            history_on: false,
            retention: 0,
            login: false,
            engine: None,
            history: Vec::new(),
            history_watch: settings::Watch::of(history::LOG_FILE),
            settings_watch: settings::Watch::of(settings::FILE),
            last_poll: Instant::now(),
            confirm_clear: false,
            message: None,
            dictionary: dictionary::DictionaryFile::default(),
            dictionary_watch: settings::Watch::of(dictionary::FILE),
            new_entry: String::new(),
        }
    }

    fn new(root: PathBuf, page: Page) -> Self {
        let mut app = Self::blank(root, page);
        app.login = autostart::enabled();
        app.reload_settings();
        app.reload_history();
        app.reload_dictionary();
        app
    }

    fn dictionary_path(&self) -> PathBuf {
        self.root.join(dictionary::FILE)
    }

    fn reload_dictionary(&mut self) {
        let mut file = dictionary::load_at(&self.dictionary_path());
        file.entries
            .sort_by_key(|entry| (entry.phrase.to_lowercase(), entry.replacement.clone()));
        self.dictionary = file;
    }

    /// Change the dictionary as it is on disk now (another window or a hand
    /// edit may have changed it), save, and show the result.
    fn edit_dictionary(
        &mut self,
        change: impl FnOnce(&mut dictionary::DictionaryFile) -> bool,
    ) -> bool {
        let path = self.dictionary_path();
        let mut file = dictionary::load_at(&path);
        if !change(&mut file) {
            return false;
        }
        match dictionary::save_at(&mut file, &path) {
            Ok(()) => {
                self.reload_dictionary();
                true
            }
            Err(error) => {
                self.message = Some((format!("Could not save the dictionary: {error:#}"), true));
                false
            }
        }
    }

    fn add_entry(&mut self) {
        let Some(entry) = parse_entry(&self.new_entry) else {
            self.message = Some((
                "Type a word, or a replacement as: heard => written".into(),
                true,
            ));
            return;
        };
        let added = match &entry {
            NewEntry::Word(word) => {
                let word = word.clone();
                self.edit_dictionary(|file| dictionary::add_word(file, &word))
            }
            NewEntry::Replacement(from, to) => {
                let (from, to) = (from.clone(), to.clone());
                self.edit_dictionary(|file| dictionary::add_replacement(file, &from, &to))
            }
        };
        if added {
            self.new_entry.clear();
            self.message = Some(("Added. Applies from the next dictation.".into(), false));
        } else if self.message.as_ref().is_none_or(|(_, error)| !error) {
            self.message = Some(("That is already in the dictionary.".into(), false));
        }
    }

    fn dictionary_page(&mut self, ui: &mut egui::Ui) {
        ui.heading("Dictionary");
        ui.label(
            egui::RichText::new(
                "Words Phonon should spell your way, and replacements: say one thing, get another.",
            )
            .weak(),
        );
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let field = ui.add(
                egui::TextEdit::singleline(&mut self.new_entry)
                    .hint_text("Kubernetes   or   kates => k8s")
                    .desired_width((ui.available_width() - 70.0).max(160.0)),
            );
            let enter = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if ui.button("Add").clicked() || enter {
                self.add_entry();
                field.request_focus();
            }
        });
        ui.add_space(6.0);
        if self.dictionary.entries.is_empty() {
            ui.label(egui::RichText::new("No words yet.").weak());
            return;
        }
        let mut removed = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (index, entry) in self.dictionary.entries.iter().enumerate() {
                    ui.horizontal(|ui| {
                        match &entry.replacement {
                            Some(written) => {
                                ui.label(&entry.phrase);
                                // egui's bundled font has no arrow glyph
                                ui.label(egui::RichText::new("=>").weak());
                                ui.label(egui::RichText::new(written).strong());
                            }
                            None => {
                                ui.label(egui::RichText::new(&entry.phrase).strong());
                            }
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Remove").clicked() {
                                removed = Some(index);
                            }
                        });
                    });
                    ui.separator();
                }
            });
        if let Some(index) = removed {
            if let Some(entry) = self.dictionary.entries.get(index).cloned() {
                if self.edit_dictionary(|file| {
                    dictionary::remove(file, &entry.phrase, entry.replacement.as_deref())
                }) {
                    self.message = Some((format!("Removed {}.", entry.phrase), false));
                }
            }
        }
    }

    fn reload_settings(&mut self) {
        let s = Settings::load_in(&self.root);
        self.shortcut = s
            .shortcut()
            .filter(|name| settings::SHORTCUTS.iter().any(|(n, _)| n == name))
            .unwrap_or_else(|| settings::DEFAULT_SHORTCUT.into());
        self.style = s.writing_style();
        self.clipboard = s.clipboard_after_paste();
        self.history_on = s.local_history();
        self.retention = s.history_retention_days();
        self.engine = settings::engine_status_in(&self.root);
    }

    fn reload_history(&mut self) {
        let mut entries = history::read_in(&self.root);
        entries.reverse();
        entries.retain(|entry| !entry.final_text.trim().is_empty());
        entries.truncate(HISTORY_SHOWN);
        self.history = entries;
    }

    /// Pick up changes made elsewhere (the tray menu, a new dictation) every second.
    fn poll(&mut self) {
        if self.last_poll.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.last_poll = Instant::now();
        if self.settings_watch.changed_in(&self.root) {
            self.reload_settings();
        }
        if self.history_watch.changed_in(&self.root) {
            self.reload_history();
        }
        if self.dictionary_watch.changed_in(&self.root) {
            self.reload_dictionary();
        }
    }

    fn save(&mut self, key: &str, value: serde_json::Value) {
        match settings::set_in(&self.root, key, value) {
            Ok(()) => {
                self.message = Some(("Saved. Applies from the next dictation.".into(), false))
            }
            Err(error) => self.message = Some((format!("Could not save: {error:#}"), true)),
        }
    }

    fn general(&mut self, ui: &mut egui::Ui) {
        heading(ui, "Dictation");
        row(
            ui,
            "Shortcut",
            "Hold it to dictate, or tap it to start and tap again to stop.",
            |ui, app: &mut Self| {
                let current = settings::SHORTCUTS
                    .iter()
                    .find(|(name, _)| *name == app.shortcut)
                    .map(|(_, label)| *label)
                    .unwrap_or("Right Ctrl");
                let mut picked = app.shortcut.clone();
                egui::ComboBox::from_id_salt("shortcut")
                    .selected_text(current)
                    .width(200.0)
                    .show_ui(ui, |ui| {
                        for (name, label) in settings::SHORTCUTS {
                            ui.selectable_value(&mut picked, name.to_string(), *label);
                        }
                    });
                if picked != app.shortcut {
                    app.shortcut = picked.clone();
                    app.save(settings::SHORTCUT_MODE, picked.into());
                }
            },
            self,
        );
        row(
            ui,
            "Writing style",
            "How the correction model rewrites what you said.",
            |ui, app: &mut Self| {
                let mut picked = app.style;
                egui::ComboBox::from_id_salt("style")
                    .selected_text(mode::label(app.style))
                    .width(200.0)
                    .show_ui(ui, |ui| {
                        for name in mode::names() {
                            ui.selectable_value(&mut picked, name, mode::label(name));
                        }
                    });
                if picked != app.style {
                    app.style = picked;
                    app.save(settings::WRITING_STYLE, picked.into());
                }
            },
            self,
        );
        row(
            ui,
            "After a paste, the clipboard holds",
            "Phonon pastes through the clipboard.",
            |ui, app: &mut Self| {
                let mut picked = app.clipboard;
                egui::ComboBox::from_id_salt("clipboard")
                    .selected_text(clipboard_label(app.clipboard))
                    .width(200.0)
                    .show_ui(ui, |ui| {
                        for rule in [ClipboardAfterPaste::Restore, ClipboardAfterPaste::Dictation] {
                            ui.selectable_value(&mut picked, rule, clipboard_label(rule));
                        }
                    });
                if picked != app.clipboard {
                    app.clipboard = picked;
                    app.save(settings::CLIPBOARD_AFTER_PASTE, picked.name().into());
                }
            },
            self,
        );

        heading(ui, "History");
        row(
            ui,
            "Keep history",
            "Keep each recording and its text on this computer after it is typed.",
            |ui, app: &mut Self| {
                let mut on = app.history_on;
                if ui.checkbox(&mut on, "").changed() {
                    app.history_on = on;
                    app.save(settings::LOCAL_HISTORY, on.into());
                }
            },
            self,
        );
        row(
            ui,
            "Keep recordings for",
            "Older dictations and their recordings are deleted.",
            |ui, app: &mut Self| {
                let mut picked = app.retention;
                egui::ComboBox::from_id_salt("retention")
                    .selected_text(retention_label(app.retention))
                    .width(200.0)
                    .show_ui(ui, |ui| {
                        for days in settings::RETENTION_CHOICES {
                            ui.selectable_value(&mut picked, days, retention_label(days));
                        }
                    });
                if picked != app.retention {
                    app.retention = picked;
                    app.save(settings::HISTORY_RETENTION_DAYS, picked.into());
                    match history::prune_older_in(&app.root, picked, history::now_ms()) {
                        Ok(0) => {}
                        Ok(removed) => {
                            app.message =
                                Some((format!("Deleted {removed} older dictations."), false));
                            app.reload_history();
                        }
                        Err(error) => {
                            app.message = Some((format!("Could not delete: {error:#}"), true))
                        }
                    }
                }
            },
            self,
        );

        heading(ui, "System");
        row(
            ui,
            "Launch at login",
            "Start Phonon when you sign in.",
            |ui, app: &mut Self| {
                let mut on = app.login;
                if ui.checkbox(&mut on, "").changed() {
                    match autostart::set(on) {
                        Ok(()) => {
                            app.login = autostart::enabled();
                            app.message = Some((
                                if app.login {
                                    "Phonon starts when you sign in."
                                } else {
                                    "Phonon no longer starts when you sign in."
                                }
                                .into(),
                                false,
                            ));
                        }
                        Err(error) => {
                            app.message = Some((format!("Could not change it: {error:#}"), true))
                        }
                    }
                }
            },
            self,
        );
        ui.add_space(6.0);
        ui.label(egui::RichText::new("Models").strong());
        ui.label(
            egui::RichText::new(format!(
                "{}\nSpeech model: {}\nCorrection model: {}",
                self.engine
                    .as_deref()
                    .unwrap_or("Phonon has not finished starting yet."),
                manifest::ASR_MODEL_ID,
                manifest::POLISH_MODEL_ID,
            ))
            .weak(),
        );
        ui.label(egui::RichText::new(format!("Data folder: {}", self.root.display())).weak());
    }

    fn history_page(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Recent dictations");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if self.confirm_clear {
                    if ui.button("Cancel").clicked() {
                        self.confirm_clear = false;
                    }
                    if ui
                        .button(
                            egui::RichText::new("Delete everything")
                                .color(ui.visuals().error_fg_color),
                        )
                        .clicked()
                    {
                        self.confirm_clear = false;
                        let keep = recording_in_progress(&self.root);
                        match history::clear_in(&self.root, keep.as_deref()) {
                            Ok(removed) => {
                                self.message = Some((
                                    format!(
                                        "Removed the history and {removed} recording{}.",
                                        if removed == 1 { "" } else { "s" }
                                    ),
                                    false,
                                ));
                                self.reload_history();
                            }
                            Err(error) => {
                                self.message = Some((format!("Could not clear: {error:#}"), true))
                            }
                        }
                    }
                } else if ui
                    .add_enabled(!self.history.is_empty(), egui::Button::new("Clear all"))
                    .clicked()
                {
                    self.confirm_clear = true;
                }
            });
        });
        if self.confirm_clear {
            ui.label("Delete every dictation and recording Phonon kept? This cannot be undone.");
        }
        ui.add_space(4.0);
        if !self.history_on {
            ui.label(
                egui::RichText::new(
                    "Keep history is off, so Phonon keeps nothing after it types. Turn it on in Settings.",
                )
                .weak(),
            );
        }
        if self.history.is_empty() {
            ui.label(egui::RichText::new("Nothing yet.").weak());
            return;
        }
        let now = history::now_ms();
        let mut copied = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (index, entry) in self.history.iter().enumerate() {
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "{}  \u{b7}  {:.0} s",
                                    ago(entry.time_unix_ms, now),
                                    entry.duration_s.max(1.0)
                                ))
                                .weak()
                                .small(),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.small_button("Copy").clicked() {
                                        copied = Some(index);
                                    }
                                },
                            );
                        });
                        ui.add(egui::Label::new(entry.final_text.trim()).wrap());
                    });
                    ui.add_space(2.0);
                }
            });
        if let Some(index) = copied {
            if let Some(entry) = self.history.get(index) {
                let text = entry.final_text.trim().to_string();
                self.message = Some(match copy(ui.ctx(), text) {
                    Ok(()) => ("Copied to the clipboard.".into(), false),
                    Err(error) => (format!("Could not copy: {error:#}"), true),
                });
            }
        }
    }
}

/// Put `text` on the system clipboard. Windows: egui's clipboard (the system
/// keeps the text after the window closes). Linux: arboard, as the tray app
/// does; it serves the text while the window is open, and GNOME and KDE keep
/// it after that.
#[cfg(windows)]
fn copy(ctx: &egui::Context, text: String) -> Result<()> {
    ctx.copy_text(text);
    Ok(())
}

#[cfg(target_os = "linux")]
fn copy(_ctx: &egui::Context, text: String) -> Result<()> {
    use std::sync::{Mutex, OnceLock};
    static CLIPBOARD: OnceLock<Mutex<Option<arboard::Clipboard>>> = OnceLock::new();
    let slot = CLIPBOARD.get_or_init(|| Mutex::new(arboard::Clipboard::new().ok()));
    let mut guard = slot
        .lock()
        .map_err(|_| anyhow!("clipboard lock poisoned"))?;
    let clipboard = guard
        .as_mut()
        .ok_or_else(|| anyhow!("no clipboard on this desktop"))?;
    clipboard
        .set_text(text)
        .map_err(|error| anyhow!("set the clipboard: {error}"))
}

fn heading(ui: &mut egui::Ui, text: &str) {
    ui.add_space(10.0);
    ui.label(egui::RichText::new(text).strong().size(15.0));
    ui.separator();
}

/// One setting: title and help on the left, the control on the right.
fn row(
    ui: &mut egui::Ui,
    title: &str,
    help: &str,
    control: impl FnOnce(&mut egui::Ui, &mut App),
    app: &mut App,
) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width((ui.available_width() - 230.0).max(200.0));
            ui.label(title);
            ui.label(egui::RichText::new(help).weak().small());
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            control(ui, app)
        });
    });
    ui.add_space(6.0);
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll();
        // poll for outside changes even when the mouse is still
        ui.ctx().request_repaint_after(Duration::from_secs(1));
        egui::Panel::left("pages")
            .resizable(false)
            .exact_size(150.0)
            .show(ui, |ui| {
                ui.add_space(12.0);
                ui.heading("Phonon");
                ui.add_space(8.0);
                for (page, label) in [
                    (Page::General, "Settings"),
                    (Page::History, "History"),
                    (Page::Dictionary, "Dictionary"),
                ] {
                    if ui.selectable_label(self.page == page, label).clicked() {
                        self.page = page;
                        self.confirm_clear = false;
                    }
                }
            });
        egui::Panel::bottom("message")
            .show_separator_line(false)
            .show(ui, |ui| {
                let (text, error) = self
                    .message
                    .clone()
                    .unwrap_or_else(|| (String::new(), false));
                let text = egui::RichText::new(text).small();
                ui.label(if error {
                    text.color(ui.visuals().error_fg_color)
                } else {
                    text.weak()
                });
            });
        egui::CentralPanel::default().show(ui, |ui| match self.page {
            Page::General => {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.general(ui));
            }
            Page::History => self.history_page(ui),
            Page::Dictionary => self.dictionary_page(ui),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_read_as_ago() {
        let now = 10 * 86_400_000;
        assert_eq!(ago(now - 5_000, now), "just now");
        assert_eq!(ago(now - 120_000, now), "2 min ago");
        assert_eq!(ago(now - 3_600_000, now), "1 hour ago");
        assert_eq!(ago(now - 3 * 86_400_000, now), "3 days ago");
        assert_eq!(
            ago(now + 5_000, now),
            "just now",
            "a clock step back is not negative"
        );
    }

    #[test]
    fn labels_match_the_mac() {
        assert_eq!(retention_label(0), "Until I delete them");
        assert_eq!(retention_label(30), "30 days");
        assert_eq!(
            clipboard_label(ClipboardAfterPaste::Restore),
            "What you copied before"
        );
        assert_eq!(
            clipboard_label(ClipboardAfterPaste::Dictation),
            "Your dictation"
        );
        assert_eq!(Page::parse("History"), Some(Page::History));
        assert_eq!(Page::parse("nope"), None);
    }

    /// Each row the window writes is what the engine then reads.
    #[test]
    fn each_row_reaches_the_engine() {
        let root = std::env::temp_dir().join(format!("phonon-window-rows-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let mut app = App::blank(root.clone(), Page::General);
        app.save(settings::SHORTCUT_MODE, "f13".into());
        app.save(settings::WRITING_STYLE, "prompt".into());
        app.save(
            settings::CLIPBOARD_AFTER_PASTE,
            ClipboardAfterPaste::Dictation.name().into(),
        );
        app.save(settings::LOCAL_HISTORY, true.into());
        app.save(settings::HISTORY_RETENTION_DAYS, 7.into());
        assert_eq!(app.message.as_ref().map(|m| m.1), Some(false));
        let s = Settings::load_in(&root);
        // the key the hook installs
        assert_eq!(s.shortcut().as_deref(), Some("f13"));
        // the mode in the prompt
        let prompt = crate::polish::prompt("hello", s.writing_style());
        assert!(prompt.contains(crate::polish::mode_line("prompt")));
        // history on, and the retention the next append applies
        assert!(history::enabled_in(&root));
        assert_eq!(s.history_retention_days(), 7);
        // the clipboard rule: nothing is restored, the dictation stays
        assert_eq!(
            s.clipboard_after_paste().to_restore(Some("before".into())),
            None
        );
        app.reload_settings();
        assert_eq!(app.shortcut, "f13");
        assert_eq!(app.style, "prompt");
        assert!(app.history_on);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_window_icon_decodes() {
        let icon =
            eframe::icon_data::from_png_bytes(include_bytes!("../linux/icons/phonon-128.png"))
                .unwrap();
        assert_eq!((icon.width, icon.height), (128, 128));
    }

    #[test]
    fn the_input_line_reads_words_and_replacements() {
        assert_eq!(
            parse_entry("  Kubernetes "),
            Some(NewEntry::Word("Kubernetes".into()))
        );
        assert_eq!(
            parse_entry("kates => k8s"),
            Some(NewEntry::Replacement("kates".into(), "k8s".into()))
        );
        assert_eq!(
            parse_entry("pee are -> PR"),
            Some(NewEntry::Replacement("pee are".into(), "PR".into()))
        );
        assert_eq!(parse_entry(""), None);
        assert_eq!(parse_entry("=> k8s"), None);
        assert_eq!(parse_entry("kates =>"), None);
    }

    /// The Dictionary page writes the Mac-format dictionary.json the engine reads.
    #[test]
    fn the_dictionary_page_adds_and_removes_in_the_engines_file() {
        let root = std::env::temp_dir().join(format!("phonon-window-dict-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let mut app = App::blank(root.clone(), Page::Dictionary);
        app.new_entry = "Kubernetes".into();
        app.add_entry();
        assert!(app.new_entry.is_empty(), "the line clears after an add");
        app.new_entry = "kates => k8s".into();
        app.add_entry();
        app.new_entry = "kubernetes".into();
        app.add_entry();
        assert_eq!(
            app.message.as_ref().map(|m| m.0.as_str()),
            Some("That is already in the dictionary.")
        );
        let engine = dictionary::load_at(&root.join(dictionary::FILE));
        assert_eq!(engine.entries.len(), 2);
        assert!(engine
            .entries
            .iter()
            .any(|e| e.phrase == "kates" && e.replacement.as_deref() == Some("k8s")));
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join(dictionary::FILE)).unwrap())
                .unwrap();
        assert!(raw.get("schema_version").is_some());
        assert!(raw["entries"][0].get("spoken_forms").is_some());
        assert!(app.edit_dictionary(|file| dictionary::remove(file, "Kubernetes", None)));
        assert_eq!(app.dictionary.entries.len(), 1);
        assert_eq!(
            dictionary::load_at(&root.join(dictionary::FILE))
                .entries
                .len(),
            1
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
