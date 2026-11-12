//! `settings.json` in the data root: the settings window's file, read by the
//! engine at each dictation.
//!
//! The keys are the Mac's (bar/Sources/AppData.swift `NativeSettings.CodingKeys`)
//! wherever the Mac has the setting, so the two files mean the same thing key for
//! key: `shortcut_mode`, `writing_style`, `local_history`,
//! `history_retention_days`, `clipboard_after_paste`, plus `launch_at_login`.
//!
//! Rules:
//! - Unknown keys are kept: a write reads the file, changes one key and writes
//!   the whole object back.
//! - A write goes to a temporary file that is then renamed over the old one, so
//!   a crash leaves the old file or the new one, never half of one.
//! - A file that is not a JSON object is renamed aside (`settings.json.corrupt`)
//!   and the defaults apply.
//! - Before this file existed the tray kept one setting per file: `hotkey.txt`
//!   (Windows), `writing_mode.txt` and `history.txt`. A key missing here falls back
//!   to its old file, so nobody's behaviour changes; the next write of that key
//!   puts it here, and from then on this file wins.

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::{Map, Value};

use crate::paths;
use crate::polish::DEFAULT_MODE;

pub const FILE: &str = "settings.json";
/// Where an unreadable settings file is moved.
pub const CORRUPT_FILE: &str = "settings.json.corrupt";

pub const SHORTCUT_MODE: &str = "shortcut_mode";
pub const WRITING_STYLE: &str = "writing_style";
pub const LOCAL_HISTORY: &str = "local_history";
pub const HISTORY_RETENTION_DAYS: &str = "history_retention_days";
pub const CLIPBOARD_AFTER_PASTE: &str = "clipboard_after_paste";
pub const LAUNCH_AT_LOGIN: &str = "launch_at_login";

/// The legacy one-setting files.
pub const LEGACY_HOTKEY: &str = "hotkey.txt";
pub const LEGACY_MODE: &str = "writing_mode.txt";
pub const LEGACY_HISTORY: &str = "history.txt";

/// The default dictation key's setting name.
pub const DEFAULT_SHORTCUT: &str = "rightctrl";

/// The keys the settings window and the tray offer, as (setting name, label).
/// Linux reads evdev, which has no Ctrl+Space chord.
#[cfg(windows)]
pub const SHORTCUTS: &[(&str, &str)] = &[
    ("rightctrl", "Right Ctrl"),
    ("rightalt", "Right Alt"),
    ("ctrlspace", "Ctrl+Space"),
    ("rightshift", "Right Shift"),
    ("capslock", "Caps Lock"),
    ("f13", "F13"),
];
#[cfg(not(windows))]
pub const SHORTCUTS: &[(&str, &str)] = &[
    ("rightctrl", "Right Ctrl"),
    ("rightalt", "Right Alt"),
    ("rightshift", "Right Shift"),
    ("capslock", "Caps Lock"),
    ("f13", "F13"),
];

/// The retention choices the Mac offers (0 keeps history until it is cleared).
pub const RETENTION_CHOICES: [u32; 4] = [0, 7, 30, 90];

/// settings.json `clipboard_after_paste`, the Mac's values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipboardAfterPaste {
    /// The dictation stays on the clipboard.
    Dictation,
    /// What was copied before comes back once the paste has landed. The
    /// Windows and Linux default, as before this setting existed.
    Restore,
}

impl ClipboardAfterPaste {
    pub fn name(self) -> &'static str {
        match self {
            Self::Dictation => "dictation",
            Self::Restore => "restore",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "dictation" => Some(Self::Dictation),
            "restore" => Some(Self::Restore),
            _ => None,
        }
    }

    /// The clipboard text to put back after a paste: the previous text under
    /// `Restore`, nothing under `Dictation` (the dictation stays).
    pub fn to_restore(self, previous: Option<String>) -> Option<String> {
        match self {
            Self::Restore => previous,
            Self::Dictation => None,
        }
    }
}

/// One read of the settings, with the legacy files as fallback.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Settings {
    map: Map<String, Value>,
    /// Values read from the legacy files, used only for keys the map lacks.
    legacy_shortcut: Option<String>,
    legacy_mode: Option<&'static str>,
    legacy_history: Option<bool>,
}

fn read_map(root: &Path) -> Map<String, Value> {
    let path = root.join(FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            if error.kind() != std::io::ErrorKind::NotFound {
                eprintln!("phonon: could not read {}: {error}", path.display());
            }
            return Map::new();
        }
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(map)) => map,
        _ => {
            let aside = root.join(CORRUPT_FILE);
            eprintln!(
                "phonon: {} is not a settings object; moved it to {} and using the defaults",
                path.display(),
                aside.display()
            );
            let _ = std::fs::rename(&path, &aside);
            Map::new()
        }
    }
}

impl Settings {
    /// The settings in data root `root`.
    pub fn load_in(root: &Path) -> Self {
        let map = read_map(root);
        let legacy_shortcut = std::fs::read_to_string(root.join(LEGACY_HOTKEY))
            .ok()
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty());
        let mode_path = root.join(LEGACY_MODE);
        let legacy_mode = mode_path
            .is_file()
            .then(|| crate::mode::load_from(&mode_path));
        let legacy_history = std::fs::read_to_string(root.join(LEGACY_HISTORY))
            .ok()
            .map(|text| crate::history::parse_switch(&text));
        Self {
            map,
            legacy_shortcut,
            legacy_mode,
            legacy_history,
        }
    }

    /// The settings in the real data root.
    pub fn load() -> Self {
        Self::load_in(&paths::data_root())
    }

    fn string(&self, key: &str) -> Option<&str> {
        self.map.get(key).and_then(Value::as_str)
    }

    /// The dictation key's setting name, as saved (the hook validates it).
    /// `None` means the user never picked one.
    pub fn shortcut(&self) -> Option<String> {
        self.string(SHORTCUT_MODE)
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())
            .or_else(|| self.legacy_shortcut.clone())
    }

    /// The refiner mode for the next dictation. The Mac's "app" (follow the
    /// focused app) cannot be honoured here yet and means minimal.
    pub fn writing_style(&self) -> &'static str {
        match self.string(WRITING_STYLE) {
            Some(text) => crate::mode::parse(text).unwrap_or(DEFAULT_MODE),
            None => self.legacy_mode.unwrap_or(DEFAULT_MODE),
        }
    }

    /// Local history; off unless turned on, as on the Mac.
    pub fn local_history(&self) -> bool {
        match self.map.get(LOCAL_HISTORY) {
            Some(Value::Bool(on)) => *on,
            Some(_) => false,
            None => self.legacy_history.unwrap_or(false),
        }
    }

    /// Days of history to keep; 0 keeps it until it is cleared (the Mac's default).
    pub fn history_retention_days(&self) -> u32 {
        self.map
            .get(HISTORY_RETENTION_DAYS)
            .and_then(Value::as_u64)
            .map(|days| days.min(u32::MAX as u64) as u32)
            .unwrap_or(0)
    }

    /// What the clipboard holds after a paste; `Restore` unless set.
    pub fn clipboard_after_paste(&self) -> ClipboardAfterPaste {
        self.string(CLIPBOARD_AFTER_PASTE)
            .and_then(ClipboardAfterPaste::parse)
            .unwrap_or(ClipboardAfterPaste::Restore)
    }

    /// The saved launch-at-login choice. The operating system's own state
    /// (`autostart::enabled`) is the truth; this records what the user picked.
    #[cfg(test)]
    pub fn launch_at_login(&self) -> Option<bool> {
        self.map.get(LAUNCH_AT_LOGIN).and_then(Value::as_bool)
    }

    #[cfg(test)]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.map.get(key)
    }
}

/// Change one key in the settings file at `root`, keeping every other key.
pub fn set_in(root: &Path, key: &str, value: Value) -> Result<()> {
    let mut map = read_map(root);
    map.insert(key.to_string(), value);
    write_map(root, &map)
}

/// Change one key in the real settings file.
pub fn set(key: &str, value: Value) -> Result<()> {
    set_in(&paths::data_root(), key, value)
}

fn write_map(root: &Path, map: &Map<String, Value>) -> Result<()> {
    std::fs::create_dir_all(root).with_context(|| format!("create {}", root.display()))?;
    let path = root.join(FILE);
    let mut text = serde_json::to_string_pretty(map)?;
    text.push('\n');
    // A name per process, so the tray and the settings window never share one.
    let staging = root.join(format!("{FILE}.{}.tmp", std::process::id()));
    std::fs::write(&staging, text).with_context(|| format!("write {}", staging.display()))?;
    std::fs::rename(&staging, &path).with_context(|| format!("replace {}", path.display()))?;
    Ok(())
}

/// A file's modification time, for a cheap "did it change" check.
fn stamp(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
}

/// Reports a change of one file in the data root since the last call. The
/// engine's tick calls this every 500 ms; it reads only the file's metadata.
pub struct Watch {
    file: &'static str,
    last: Option<Option<std::time::SystemTime>>,
}

impl Watch {
    /// Watch `file` (a name in the data root), e.g. `FILE` or `history::LOG_FILE`.
    pub fn of(file: &'static str) -> Self {
        Self { file, last: None }
    }

    /// True on the first call and whenever the file's modification time moved
    /// (or it appeared or went).
    pub fn changed_in(&mut self, root: &Path) -> bool {
        let now = stamp(&root.join(self.file));
        let changed = self.last != Some(now);
        self.last = Some(now);
        changed
    }
}

/// What the running engine reported at its last start (device and models), for
/// the settings window's status line. One line of text in the data root.
pub const ENGINE_FILE: &str = "engine-status.txt";

pub fn record_engine_in(root: &Path, text: &str) {
    let path = root.join(ENGINE_FILE);
    if let Err(error) = std::fs::write(&path, format!("{text}\n")) {
        eprintln!("phonon: could not write {}: {error}", path.display());
    }
}

/// Record where the engine that just passed its startup check runs.
pub fn record_engine(device: crate::polish::Device) {
    let place = match device {
        crate::polish::Device::Gpu => "the GPU",
        crate::polish::Device::Cpu => "the CPU",
    };
    record_engine_in(
        &paths::data_root(),
        &format!("Speech recognition on the CPU, correction on {place}"),
    )
}

pub fn engine_status_in(root: &Path) -> Option<String> {
    std::fs::read_to_string(root.join(ENGINE_FILE))
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("phonon-win-settings-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn no_file_means_the_defaults_of_today() {
        let root = scratch("defaults");
        let s = Settings::load_in(&root);
        assert_eq!(s.shortcut(), None);
        assert_eq!(s.writing_style(), "minimal");
        assert!(!s.local_history());
        assert_eq!(s.history_retention_days(), 0);
        assert_eq!(s.clipboard_after_paste(), ClipboardAfterPaste::Restore);
        assert_eq!(s.launch_at_login(), None);
        assert!(!root.join(FILE).exists(), "reading writes nothing");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn every_key_round_trips_with_the_macs_names() {
        let root = scratch("round");
        set_in(&root, SHORTCUT_MODE, "capslock".into()).unwrap();
        set_in(&root, WRITING_STYLE, "formal".into()).unwrap();
        set_in(&root, LOCAL_HISTORY, true.into()).unwrap();
        set_in(&root, HISTORY_RETENTION_DAYS, 30.into()).unwrap();
        set_in(&root, CLIPBOARD_AFTER_PASTE, "dictation".into()).unwrap();
        set_in(&root, LAUNCH_AT_LOGIN, false.into()).unwrap();
        let s = Settings::load_in(&root);
        assert_eq!(s.shortcut().as_deref(), Some("capslock"));
        assert_eq!(s.writing_style(), "formal");
        assert!(s.local_history());
        assert_eq!(s.history_retention_days(), 30);
        assert_eq!(s.clipboard_after_paste(), ClipboardAfterPaste::Dictation);
        assert_eq!(s.launch_at_login(), Some(false));
        let raw: Value =
            serde_json::from_str(&std::fs::read_to_string(root.join(FILE)).unwrap()).unwrap();
        assert_eq!(raw["shortcut_mode"], "capslock");
        assert_eq!(raw["writing_style"], "formal");
        assert_eq!(raw["local_history"], true);
        assert_eq!(raw["history_retention_days"], 30);
        assert_eq!(raw["clipboard_after_paste"], "dictation");
        assert_eq!(raw["launch_at_login"], false);
        // no staging file left behind
        let names: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec![FILE.to_string()]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn unknown_keys_survive_a_write() {
        let root = scratch("unknown");
        std::fs::write(
            root.join(FILE),
            r#"{"schema_version": 7, "screen_context": true, "meeting_rules": [{"a": 1}], "writing_style": "casual"}"#,
        )
        .unwrap();
        set_in(&root, WRITING_STYLE, "prompt".into()).unwrap();
        let raw: Value =
            serde_json::from_str(&std::fs::read_to_string(root.join(FILE)).unwrap()).unwrap();
        assert_eq!(raw["schema_version"], 7);
        assert_eq!(raw["screen_context"], true);
        assert_eq!(raw["meeting_rules"][0]["a"], 1);
        assert_eq!(raw["writing_style"], "prompt");
        let s = Settings::load_in(&root);
        assert_eq!(s.get("schema_version"), Some(&Value::from(7)));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_corrupt_file_is_moved_aside_and_the_defaults_apply() {
        for bad in ["{\"writing_style\": \"formal\"", "[1, 2]", "", "\"text\""] {
            let root = scratch("corrupt");
            std::fs::write(root.join(FILE), bad).unwrap();
            let s = Settings::load_in(&root);
            assert_eq!(s.writing_style(), "minimal", "{bad:?}");
            assert!(!root.join(FILE).exists(), "{bad:?}");
            assert_eq!(
                std::fs::read_to_string(root.join(CORRUPT_FILE)).unwrap(),
                bad
            );
            // the next write starts a fresh file
            set_in(&root, WRITING_STYLE, "casual".into()).unwrap();
            assert_eq!(Settings::load_in(&root).writing_style(), "casual");
            let _ = std::fs::remove_dir_all(&root);
        }
    }

    #[test]
    fn bad_values_fall_back_per_key() {
        let root = scratch("values");
        std::fs::write(
            root.join(FILE),
            r#"{"writing_style": "app", "local_history": "yes", "history_retention_days": -3, "clipboard_after_paste": "sometimes", "shortcut_mode": "  "}"#,
        )
        .unwrap();
        let s = Settings::load_in(&root);
        // the Mac's "follow the app" is not available here: minimal
        assert_eq!(s.writing_style(), "minimal");
        assert!(!s.local_history());
        assert_eq!(s.history_retention_days(), 0);
        assert_eq!(s.clipboard_after_paste(), ClipboardAfterPaste::Restore);
        assert_eq!(s.shortcut(), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn migration_from_each_legacy_file() {
        let root = scratch("legacy");
        std::fs::write(root.join(LEGACY_HOTKEY), "f13\n").unwrap();
        let s = Settings::load_in(&root);
        assert_eq!(s.shortcut().as_deref(), Some("f13"));
        assert_eq!(s.writing_style(), "minimal");
        assert!(!s.local_history());

        std::fs::write(root.join(LEGACY_MODE), "Casual\n").unwrap();
        assert_eq!(Settings::load_in(&root).writing_style(), "casual");

        std::fs::write(root.join(LEGACY_HISTORY), "on\n").unwrap();
        assert!(Settings::load_in(&root).local_history());
        assert!(
            !root.join(FILE).exists(),
            "migration reads, it does not write"
        );

        // a key in settings.json wins over its legacy file; other keys still fall back
        set_in(&root, LOCAL_HISTORY, false.into()).unwrap();
        let s = Settings::load_in(&root);
        assert!(!s.local_history());
        assert_eq!(s.writing_style(), "casual");
        assert_eq!(s.shortcut().as_deref(), Some("f13"));
        set_in(&root, SHORTCUT_MODE, "rightalt".into()).unwrap();
        set_in(&root, WRITING_STYLE, "formal".into()).unwrap();
        let s = Settings::load_in(&root);
        assert_eq!(s.shortcut().as_deref(), Some("rightalt"));
        assert_eq!(s.writing_style(), "formal");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_clipboard_rule_decides_what_comes_back() {
        let previous = Some("copied before".to_string());
        assert_eq!(
            ClipboardAfterPaste::Restore.to_restore(previous.clone()),
            previous
        );
        assert_eq!(ClipboardAfterPaste::Dictation.to_restore(previous), None);
        assert_eq!(ClipboardAfterPaste::Restore.to_restore(None), None);
        for rule in [ClipboardAfterPaste::Dictation, ClipboardAfterPaste::Restore] {
            assert_eq!(ClipboardAfterPaste::parse(rule.name()), Some(rule));
        }
    }

    #[test]
    fn the_engine_status_round_trips() {
        let root = scratch("engine");
        assert_eq!(engine_status_in(&root), None);
        record_engine_in(&root, "Correction on the CPU");
        assert_eq!(
            engine_status_in(&root).as_deref(),
            Some("Correction on the CPU")
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_watch_sees_a_write() {
        let root = scratch("watch");
        let mut watch = Watch::of(FILE);
        assert!(watch.changed_in(&root), "the first look counts as a change");
        assert!(!watch.changed_in(&root));
        set_in(&root, WRITING_STYLE, "formal".into()).unwrap();
        assert!(watch.changed_in(&root));
        assert!(!watch.changed_in(&root));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_offered_shortcuts_resolve_on_this_platform() {
        assert_eq!(SHORTCUTS[0].0, DEFAULT_SHORTCUT);
        for (name, _) in SHORTCUTS {
            assert!(!name.is_empty());
        }
    }
}
