//! Local dictation history, the Windows and Linux counterpart of the Mac's
//! `local_history` setting.
//!
//! Off by default, as on the Mac since settings schema 2. The switch is one
//! word in `history.txt` in the data root, next to `hotkey.txt`: this crate has
//! no settings file, and a single-purpose file needs no schema, no migration and
//! no read-modify-write shared with other settings. When it is off Phonon keeps
//! no transcript and no recording (see `keep_wav`).
//!
//! When it is on, each dictation appends one JSON line to `history.jsonl` in the
//! data root, and its recording stays in `recordings/`. The log keeps the newest
//! `MAX_ENTRIES` dictations; older lines are dropped together with their
//! recordings. The Mac keeps history until the owner deletes it, with an
//! optional 7, 30 or 90 day window; a tray has no room for that picker, so this
//! is a fixed count instead.
//!
//! Nothing here writes dictated text anywhere but `history.jsonl`, and nothing
//! here logs it.

use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::paths;

/// The on/off switch, in the data root.
pub const SWITCH_FILE: &str = "history.txt";
/// One JSON object per line, oldest first, in the data root.
pub const LOG_FILE: &str = "history.jsonl";
/// Dictations kept; at about ten seconds each that is some 320 MB of 16 kHz recordings.
pub const MAX_ENTRIES: usize = 1000;
/// How many the tray's "Recent dictations" submenu lists (the Mac's Home page shows five).
pub const RECENT: usize = 5;
/// Characters of a dictation shown in a menu label, before the ellipsis.
pub const LABEL_CHARS: usize = 50;
/// A recording this long is written before transcription, so a crash cannot lose it.
pub const SAVE_FIRST_SECONDS: f64 = 60.0;

/// One dictation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// When the recording stopped, in milliseconds since the Unix epoch.
    pub time_unix_ms: u64,
    /// Length of the recording.
    pub duration_s: f64,
    /// What the recogniser heard.
    pub raw: String,
    /// What Phonon inserted (empty when nothing was said or transcription failed).
    #[serde(rename = "final")]
    pub final_text: String,
    /// The writing mode, once the app has them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// The recording's file name in `recordings/`, when it was kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wav: Option<String>,
}

impl Entry {
    /// An entry for a dictation whose recording is (or was meant to be) `wav`.
    /// The file name is stored only when the file exists.
    pub fn new(
        time_unix_ms: u64,
        duration_s: f64,
        raw: &str,
        final_text: &str,
        wav: &Path,
    ) -> Self {
        Self {
            time_unix_ms,
            duration_s: (duration_s * 100.0).round() / 100.0,
            raw: raw.to_string(),
            final_text: final_text.to_string(),
            mode: None,
            wav: wav
                .is_file()
                .then(|| wav.file_name().map(|n| n.to_string_lossy().into_owned()))
                .flatten(),
        }
    }
}

fn recordings_in(root: &Path) -> PathBuf {
    root.join(paths::RECORDINGS)
}

/// What the switch file says. Anything but "on" (or a file that is missing) is off.
pub fn parse_switch(text: &str) -> bool {
    matches!(
        text.trim().to_ascii_lowercase().as_str(),
        "on" | "true" | "1" | "yes"
    )
}

pub fn enabled_in(root: &Path) -> bool {
    fs::read_to_string(root.join(SWITCH_FILE))
        .map(|text| parse_switch(&text))
        .unwrap_or(false)
}

pub fn set_enabled_in(root: &Path, on: bool) -> Result<()> {
    fs::create_dir_all(root)?;
    let path = root.join(SWITCH_FILE);
    fs::write(&path, if on { "on\n" } else { "off\n" })
        .with_context(|| format!("write {}", path.display()))
}

/// Whether history is on, in the real data root.
pub fn enabled() -> bool {
    enabled_in(&paths::data_root())
}

/// Every entry, oldest first. A line that does not parse (a torn write) is skipped.
pub fn read_in(root: &Path) -> Vec<Entry> {
    let Ok(text) = fs::read_to_string(root.join(LOG_FILE)) else {
        return Vec::new();
    };
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

/// Append one entry, then drop the oldest beyond `cap` with their recordings.
pub fn append_in(root: &Path, entry: &Entry, cap: usize) -> Result<()> {
    fs::create_dir_all(root)?;
    let path = root.join(LOG_FILE);
    let mut line = serde_json::to_string(entry)?;
    line.push('\n');
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut file| file.write_all(line.as_bytes()))
        .with_context(|| format!("append to {}", path.display()))?;

    let entries = read_in(root);
    if entries.len() <= cap {
        return Ok(());
    }
    let (dropped, kept) = entries.split_at(entries.len() - cap);
    let mut text = String::new();
    for entry in kept {
        text.push_str(&serde_json::to_string(entry)?);
        text.push('\n');
    }
    // Replace in one rename, so a crash leaves the old log or the new one.
    let staging = root.join(format!("{LOG_FILE}.tmp"));
    fs::write(&staging, text).with_context(|| format!("write {}", staging.display()))?;
    fs::rename(&staging, &path).with_context(|| format!("replace {}", path.display()))?;
    let recordings = recordings_in(root);
    for name in dropped.iter().filter_map(|entry| entry.wav.as_deref()) {
        // Only a bare file name, never a path out of the folder.
        if Path::new(name).file_name() == Some(name.as_ref()) {
            remove_if_present(&recordings.join(name));
        }
    }
    Ok(())
}

/// Append to the real history.
pub fn append(entry: &Entry) -> Result<()> {
    append_in(&paths::data_root(), entry, MAX_ENTRIES)
}

/// The newest `RECENT` non-empty finals, newest first.
pub fn recent(entries: &[Entry]) -> Vec<String> {
    entries
        .iter()
        .rev()
        .map(|entry| entry.final_text.trim())
        .filter(|text| !text.is_empty())
        .take(RECENT)
        .map(str::to_string)
        .collect()
}

/// The recent list from the real history, or nothing when history is off.
pub fn recent_now() -> Vec<String> {
    let root = paths::data_root();
    if enabled_in(&root) {
        recent(&read_in(&root))
    } else {
        Vec::new()
    }
}

/// A one-line menu label: whitespace runs become one space, and text over
/// `LABEL_CHARS` characters is cut on a character boundary with an ellipsis.
pub fn label(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= LABEL_CHARS {
        return flat;
    }
    let cut: String = flat.chars().take(LABEL_CHARS).collect();
    format!("{}\u{2026}", cut.trim_end())
}

/// Remove the log and every recording in `recordings/` except `keep` (a
/// recording in progress). Returns how many recordings were removed.
pub fn clear_in(root: &Path, keep: Option<&Path>) -> Result<usize> {
    remove_if_present(&root.join(LOG_FILE));
    let mut removed = 0;
    let entries = match fs::read_dir(recordings_in(root)) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error).context("list the recordings"),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_wav = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"));
        if !is_wav || keep.is_some_and(|keep| keep == path) {
            continue;
        }
        if fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

/// Clear the real history.
pub fn clear(keep: Option<&Path>) -> Result<usize> {
    clear_in(&paths::data_root(), keep)
}

/// Whether a dictation's recording stays on disk. With history on, always.
/// With it off, only a recording over `SAVE_FIRST_SECONDS` whose transcription
/// failed, so a long dictation can still be recovered; every other recording
/// is deleted once its text is in.
pub fn keep_wav(history_on: bool, transcribed: bool, seconds: f64) -> bool {
    history_on || (!transcribed && seconds > SAVE_FIRST_SECONDS)
}

/// Delete a file that may not exist.
pub fn remove_if_present(path: &Path) {
    if let Err(error) = fs::remove_file(path) {
        if error.kind() != ErrorKind::NotFound {
            eprintln!("phonon: could not delete {}: {error}", path.display());
        }
    }
}

/// Milliseconds since the Unix epoch.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("phonon-win-history-{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entry(time: u64, text: &str, wav: Option<&str>) -> Entry {
        Entry {
            time_unix_ms: time,
            duration_s: 1.5,
            raw: text.to_lowercase(),
            final_text: text.to_string(),
            mode: None,
            wav: wav.map(str::to_string),
        }
    }

    #[test]
    fn the_switch_is_off_unless_it_says_on() {
        assert!(parse_switch("on"));
        assert!(parse_switch(" ON\n"));
        assert!(parse_switch("true"));
        assert!(!parse_switch("off"));
        assert!(!parse_switch(""));
        assert!(!parse_switch("maybe"));
    }

    #[test]
    fn the_switch_defaults_off_and_round_trips() {
        let root = scratch("switch");
        assert!(
            !enabled_in(&root),
            "no file means off, as on a new Mac install"
        );
        set_enabled_in(&root, true).unwrap();
        assert!(enabled_in(&root));
        set_enabled_in(&root, false).unwrap();
        assert!(!enabled_in(&root));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn appended_entries_read_back_in_order_with_the_field_names() {
        let root = scratch("append");
        append_in(&root, &entry(1, "One.", Some("1.wav")), 10).unwrap();
        append_in(&root, &entry(2, "Two.", None), 10).unwrap();
        let read = read_in(&root);
        assert_eq!(
            read,
            vec![entry(1, "One.", Some("1.wav")), entry(2, "Two.", None)]
        );
        let text = fs::read_to_string(root.join(LOG_FILE)).unwrap();
        let first: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        for key in ["time_unix_ms", "duration_s", "raw", "final", "wav"] {
            assert!(first.get(key).is_some(), "missing {key}");
        }
        assert!(first.get("mode").is_none(), "no mode until the app has one");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_torn_line_is_skipped() {
        let root = scratch("torn");
        append_in(&root, &entry(1, "One.", None), 10).unwrap();
        fs::write(
            root.join(LOG_FILE),
            format!(
                "{}\n{{\"time_unix_ms\": 2, \"raw\n",
                serde_json::to_string(&entry(1, "One.", None)).unwrap()
            ),
        )
        .unwrap();
        assert_eq!(read_in(&root).len(), 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_cap_drops_the_oldest_with_their_recordings() {
        let root = scratch("cap");
        let recordings = root.join(paths::RECORDINGS);
        fs::create_dir_all(&recordings).unwrap();
        for time in 1..=5u64 {
            let name = format!("{time}.wav");
            fs::write(recordings.join(&name), b"RIFF").unwrap();
            append_in(&root, &entry(time, "Text.", Some(&name)), 3).unwrap();
        }
        let times: Vec<u64> = read_in(&root).iter().map(|e| e.time_unix_ms).collect();
        assert_eq!(times, vec![3, 4, 5]);
        assert!(!recordings.join("1.wav").exists());
        assert!(!recordings.join("2.wav").exists());
        assert!(recordings.join("3.wav").exists());
        assert!(recordings.join("5.wav").exists());
        assert!(!root.join(format!("{LOG_FILE}.tmp")).exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_dropped_entry_cannot_delete_outside_the_recordings_folder() {
        let root = scratch("escape");
        let outside = root.join("keep.wav");
        fs::write(&outside, b"RIFF").unwrap();
        append_in(&root, &entry(1, "A.", Some("../keep.wav")), 1).unwrap();
        append_in(&root, &entry(2, "B.", None), 1).unwrap();
        assert!(outside.exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn recent_is_the_newest_five_non_empty_finals() {
        let mut entries = Vec::new();
        for time in 1..=8u64 {
            entries.push(entry(time, &format!("Text {time}."), None));
        }
        entries.push(entry(9, "", None));
        entries.push(entry(10, "   ", None));
        entries.push(entry(11, " Last. ", None));
        assert_eq!(
            recent(&entries),
            vec!["Last.", "Text 8.", "Text 7.", "Text 6.", "Text 5."]
        );
        assert!(recent(&[entry(1, "", None)]).is_empty());
        assert_eq!(recent(&entries[..2]), vec!["Text 2.", "Text 1."]);
    }

    #[test]
    fn labels_cut_on_character_boundaries() {
        assert_eq!(label("Short."), "Short.");
        assert_eq!(label("two\nlines\t here"), "two lines here");
        let exact = "a".repeat(LABEL_CHARS);
        assert_eq!(label(&exact), exact);
        // Multi-byte characters: a byte slice at 50 would panic or split one.
        let long = "\u{e9}t\u{e9} \u{1f600}".repeat(20);
        let cut = label(&long);
        assert!(cut.ends_with('\u{2026}'));
        assert!(cut.chars().count() <= LABEL_CHARS + 1);
        assert!(long.starts_with(cut.trim_end_matches('\u{2026}')));
        let words = "word ".repeat(12);
        // The cut falls just after a space; no trailing space before the ellipsis.
        assert!(!label(&words).contains(" \u{2026}"));
    }

    #[test]
    fn clear_removes_the_log_and_recordings_but_not_the_one_in_progress() {
        let root = scratch("clear");
        let recordings = root.join(paths::RECORDINGS);
        fs::create_dir_all(&recordings).unwrap();
        append_in(&root, &entry(1, "One.", Some("1.wav")), 10).unwrap();
        fs::write(recordings.join("1.wav"), b"RIFF").unwrap();
        fs::write(recordings.join("old.WAV"), b"RIFF").unwrap();
        fs::write(recordings.join("active.wav"), b"RIFF").unwrap();
        fs::write(recordings.join("notes.txt"), b"x").unwrap();
        let active = recordings.join("active.wav");
        assert_eq!(clear_in(&root, Some(&active)).unwrap(), 2);
        assert!(!root.join(LOG_FILE).exists());
        assert!(read_in(&root).is_empty());
        assert!(active.exists());
        assert!(recordings.join("notes.txt").exists());
        // Clearing again, and clearing a root with no recordings folder, is fine.
        assert_eq!(clear_in(&root, None).unwrap(), 1);
        assert_eq!(clear_in(&scratch("clear-empty"), None).unwrap(), 0);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn recordings_are_kept_only_with_history_or_after_a_long_failure() {
        assert!(keep_wav(true, true, 3.0));
        assert!(keep_wav(true, false, 3.0));
        assert!(!keep_wav(false, true, 3.0));
        assert!(!keep_wav(false, true, 400.0));
        assert!(!keep_wav(false, false, 3.0));
        assert!(keep_wav(false, false, 400.0));
    }

    #[test]
    fn an_entry_names_its_recording_only_when_the_file_exists() {
        let root = scratch("entry");
        let wav = root.join("9.wav");
        assert_eq!(Entry::new(9, 1.234, "a", "A.", &wav).wav, None);
        fs::write(&wav, b"RIFF").unwrap();
        let made = Entry::new(9, 1.234, "a", "A.", &wav);
        assert_eq!(made.wav.as_deref(), Some("9.wav"));
        assert_eq!(made.duration_s, 1.23);
        let _ = fs::remove_dir_all(&root);
    }
}
