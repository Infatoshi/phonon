//! Labeled pairs for training and evaluation: what the recognizer heard,
//! what Phonon wrote, and what the user kept.
//!
//! One row per dictation in the corpus (which only holds dictations made
//! with local history on), and optionally per dictation in a copy of Wispr
//! Flow's `flow.sqlite`. The edited text is the user's own edit when one
//! was observed (`user_edit` in the corpus entry, `editedText` in Wispr);
//! otherwise it is the refined text and the row says it was not edited.
//!
//! Every row carries a `split`. Rows from before the cutoff day are `train`
//! and rows from the cutoff day on are `eval`, so held-out days never mix
//! with training days; ids listed as holdouts are `eval` whatever their date.

use crate::{app_context, data, edits};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub const DAY_MS: i64 = 86_400_000;
/// Days held out for evaluation by default, counted back from today.
pub const DEFAULT_EVAL_DAYS: i64 = 14;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Eval,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairRow {
    pub id: String,
    /// `phonon` or `wispr`.
    pub source: String,
    /// RFC 3339, UTC.
    pub timestamp: String,
    pub timestamp_unix_ms: i64,
    pub app_category: Option<String>,
    /// The dictation's WAV, when it is on disk.
    pub audio_path: Option<String>,
    /// The recognizer's text.
    pub raw_text: String,
    /// What Phonon (or Wispr) wrote.
    pub refined_text: String,
    /// What the user kept: their edit when one was observed, else the
    /// refined text.
    pub edited_text: String,
    /// The edited text differs from the refined text.
    pub edited: bool,
    /// An edit observation exists for this dictation.
    pub edit_observed: bool,
    /// `edits::EditOutcome` in snake case, when observed.
    pub edit_outcome: Option<String>,
    /// Why the observation ended, when known.
    pub end_reason: Option<String>,
    /// The observation can serve as a label: it ended for a reason Wispr
    /// also learns from, and it is not a deletion, a rewrite into a new
    /// message, or cut off mid-word. Unobserved rows are learnable as
    /// unedited.
    pub learnable: bool,
    /// A transcript the user set by hand in History, when there is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intended_text: Option<String>,
    pub split: Split,
}

#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    /// Only rows at or after this time.
    pub since_ms: Option<i64>,
    /// Rows at or after this time are `eval`; earlier ones are `train`.
    pub eval_from_ms: i64,
    /// Only rows the user edited, with a learnable observation.
    pub edited_only: bool,
    /// Ids that are `eval` whatever their date (existing holdout sets).
    pub holdout_ids: HashSet<String>,
}

impl ExportOptions {
    pub fn split(&self, id: &str, timestamp_ms: i64) -> Split {
        if timestamp_ms >= self.eval_from_ms || self.holdout_ids.contains(id) {
            Split::Eval
        } else {
            Split::Train
        }
    }

    fn keeps(&self, row: &PairRow) -> bool {
        self.since_ms
            .is_none_or(|since| row.timestamp_unix_ms >= since)
            && (!self.edited_only || (row.edited && row.learnable))
    }
}

/// What an export found besides its rows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExportCounts {
    /// Corpus entries whose metadata.json could not be read.
    pub unreadable: usize,
    /// Dictations with no speech or no text.
    pub empty: usize,
    /// Dictations into a password prompt or password manager, never exported.
    pub private: usize,
}

/// Password prompts and password managers (bundle id prefixes): what was
/// dictated into them is never exported. Mirrors the bar's always-private
/// list.
pub const PRIVATE_APPS: &[&str] = &[
    "com.apple.SecurityAgent",
    "com.apple.LocalAuthentication.UIAgent",
    "com.apple.keychainaccess",
    "com.apple.Passwords",
    "com.1password.",
    "com.agilebits.onepassword",
    "com.bitwarden.desktop",
    "com.lastpass.",
    "com.dashlane.",
    "org.keepassxc.",
    "me.proton.pass",
    "in.sinew.Enpass",
    "com.keepersecurity.",
    "com.callpod.keepermac",
];

/// Password managers' web vaults; their subdomains count too.
pub const PRIVATE_HOSTS: &[&str] = &[
    "1password.com",
    "1password.eu",
    "bitwarden.com",
    "bitwarden.eu",
    "lastpass.com",
    "dashlane.com",
    "pass.proton.me",
    "keepersecurity.com",
    "keepersecurity.eu",
];

/// Whether a dictation went into a password prompt, manager or vault.
pub fn is_private_target(bundle_id: Option<&str>, host: Option<&str>) -> bool {
    let app = bundle_id.is_some_and(|bundle| {
        PRIVATE_APPS
            .iter()
            .any(|prefix| bundle.to_lowercase().starts_with(&prefix.to_lowercase()))
    });
    let site = host.is_some_and(|host| {
        PRIVATE_HOSTS
            .iter()
            .any(|vault| host == *vault || host.ends_with(&format!(".{vault}")))
    });
    app || site
}

fn outcome_name(outcome: edits::EditOutcome) -> String {
    serde_json::to_value(outcome)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_default()
}

/// Whether an observation can serve as a label (see `PairRow::learnable`).
pub fn learnable(outcome: Option<edits::EditOutcome>, end_reason: Option<&str>) -> bool {
    let reason_ok = end_reason.is_none_or(edits::end_reason_is_learnable);
    let outcome_ok = !matches!(
        outcome,
        Some(
            edits::EditOutcome::Removed
                | edits::EditOutcome::Rewrite
                | edits::EditOutcome::CaptureError
        )
    );
    reason_ok && outcome_ok
}

/// Rows for every dictation in a Phonon corpus directory.
pub fn phonon_rows(corpus: &Path, options: &ExportOptions) -> Result<(Vec<PairRow>, ExportCounts)> {
    let mut rows = Vec::new();
    let mut counts = ExportCounts::default();
    if !corpus.is_dir() {
        return Ok((rows, counts));
    }
    let mut directories: Vec<PathBuf> = fs::read_dir(corpus)
        .with_context(|| format!("read {}", corpus.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.join("metadata.json").is_file())
        .collect();
    directories.sort();
    for directory in directories {
        let Ok(metadata) = data::load_recording(&directory.join("metadata.json")) else {
            counts.unreadable += 1;
            continue;
        };
        let refined = if metadata.final_transcript.trim().is_empty() {
            metadata.raw_transcript.trim().to_string()
        } else {
            metadata.final_transcript.trim().to_string()
        };
        if metadata.speech_detected == Some(false) || refined.is_empty() {
            counts.empty += 1;
            continue;
        }
        if metadata
            .app_context
            .as_ref()
            .is_some_and(|app| is_private_target(Some(&app.bundle_id), app.url_host.as_deref()))
        {
            counts.private += 1;
            continue;
        }
        let timestamp_ms = i64::try_from(metadata.created_at_unix_ms).unwrap_or(i64::MAX);
        let audio = directory.join(&metadata.audio_file);
        let (edited_text, edit_outcome, end_reason, row_learnable) = match &metadata.user_edit {
            Some(edit) => (
                edit.edited_text.trim().to_string(),
                Some(outcome_name(edit.outcome)),
                Some(edit.end_reason.clone()),
                learnable(Some(edit.outcome), Some(&edit.end_reason)),
            ),
            None => (refined.clone(), None, None, true),
        };
        let row = PairRow {
            id: metadata.id.clone(),
            source: "phonon".into(),
            timestamp: rfc3339_utc(timestamp_ms),
            timestamp_unix_ms: timestamp_ms,
            app_category: metadata
                .app_context
                .as_ref()
                .map(|app| app.category.as_str().to_string()),
            audio_path: audio.is_file().then(|| audio.display().to_string()),
            raw_text: metadata.raw_transcript.trim().to_string(),
            edited: edited_text != refined,
            refined_text: refined,
            edited_text,
            edit_observed: metadata.user_edit.is_some(),
            edit_outcome,
            end_reason,
            learnable: row_learnable,
            intended_text: metadata
                .intended_transcript
                .as_ref()
                .map(|text| text.trim().to_string())
                .filter(|text| !text.is_empty()),
            split: options.split(&metadata.id, timestamp_ms),
        };
        if options.keeps(&row) {
            rows.push(row);
        }
    }
    rows.sort_by_key(|row| row.timestamp_unix_ms);
    Ok((rows, counts))
}

/// Wispr Flow's `stripFormattingMarkupTags`: list tags out, then trim.
pub fn strip_markup(text: &str) -> String {
    let mut output = text.to_string();
    for tag in ["<ol>", "</ol>", "<li>", "</li>", "<ul>", "</ul>"] {
        output = output.replace(tag, "");
    }
    output.trim().to_string()
}

#[derive(Debug, Deserialize)]
struct WisprHistoryRow {
    id: String,
    #[serde(default)]
    asr: Option<String>,
    #[serde(default)]
    formatted: Option<String>,
    #[serde(default)]
    edited: Option<String>,
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    app: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    end_reason: Option<String>,
}

/// A working copy of a SQLite database (with its WAL and shared-memory
/// files) in a fresh temporary directory, removed when dropped. The user's
/// database is only ever read by the copy.
struct DatabaseCopy {
    directory: PathBuf,
    database: PathBuf,
}

impl DatabaseCopy {
    fn new(source: &Path) -> Result<Self> {
        if !source.is_file() {
            bail!("Wispr Flow database not found: {}", source.display());
        }
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let directory =
            std::env::temp_dir().join(format!("phonon-wispr-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&directory)?;
        let copy = Self {
            database: directory.join("flow.sqlite"),
            directory,
        };
        fs::copy(source, &copy.database).with_context(|| format!("copy {}", source.display()))?;
        for suffix in ["-wal", "-shm"] {
            let side = PathBuf::from(format!("{}{suffix}", source.display()));
            if side.is_file() {
                fs::copy(&side, format!("{}{suffix}", copy.database.display()))?;
            }
        }
        Ok(copy)
    }
}

impl Drop for DatabaseCopy {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

fn sqlite_json(database: &Path, query: &str) -> Result<Vec<u8>> {
    let output = Command::new("sqlite3")
        .arg("-json")
        .arg(database)
        .arg(query)
        .output()
        .context("run sqlite3 to read the Wispr Flow history")?;
    if !output.status.success() {
        bail!(
            "sqlite3 failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
}

fn host_of(url: &str) -> Option<String> {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let host = rest.split(['/', '?', '#', ':']).next().unwrap_or("");
    app_context::normalize_host(host)
}

/// Rows for a copy of Wispr Flow's `flow.sqlite` History table, marked
/// `source = wispr`. With `audio_dir`, each kept row's audio (16 kHz mono
/// WAV in the database) is written there as `<id>.wav`.
pub fn wispr_rows(
    database: &Path,
    options: &ExportOptions,
    audio_dir: Option<&Path>,
) -> Result<(Vec<PairRow>, ExportCounts)> {
    let copy = DatabaseCopy::new(database)?;
    let query = "SELECT transcriptEntityId AS id, asrText AS asr, formattedText AS formatted, \
                 editedText AS edited, timestamp, app, url, \
                 contentObservationEndReason AS end_reason \
                 FROM History WHERE isArchived = 0 ORDER BY timestamp";
    let output = sqlite_json(&copy.database, query)?;
    let history: Vec<WisprHistoryRow> = if output.iter().all(u8::is_ascii_whitespace) {
        Vec::new()
    } else {
        serde_json::from_slice(&output).context("parse the Wispr Flow history")?
    };
    let dictionary = data::DictionaryFile::default();
    let mut rows = Vec::new();
    let mut counts = ExportCounts::default();
    for entry in history {
        let raw = strip_markup(entry.asr.as_deref().unwrap_or(""));
        let refined = strip_markup(entry.formatted.as_deref().unwrap_or(""));
        let refined = if refined.is_empty() {
            raw.clone()
        } else {
            refined
        };
        let Some(timestamp_ms) = entry.timestamp.as_deref().and_then(parse_timestamp) else {
            counts.unreadable += 1;
            continue;
        };
        if refined.is_empty() {
            counts.empty += 1;
            continue;
        }
        let host = entry.url.as_deref().and_then(host_of);
        if is_private_target(entry.app.as_deref(), host.as_deref()) {
            counts.private += 1;
            continue;
        }
        let observed = entry
            .edited
            .as_deref()
            .map(strip_markup)
            .filter(|text| !text.is_empty());
        let end_reason = entry.end_reason.clone().filter(|reason| !reason.is_empty());
        let (edited_text, outcome) = match &observed {
            Some(text) => (
                text.clone(),
                Some(edits::analyze(&refined, text, None, &dictionary).outcome),
            ),
            None => (refined.clone(), None),
        };
        let category = entry
            .app
            .as_deref()
            .filter(|app| !app.is_empty())
            .map(|app| {
                app_context::categorize(app, host.as_deref())
                    .as_str()
                    .to_string()
            });
        let row = PairRow {
            split: options.split(&entry.id, timestamp_ms),
            id: entry.id,
            source: "wispr".into(),
            timestamp: rfc3339_utc(timestamp_ms),
            timestamp_unix_ms: timestamp_ms,
            app_category: category,
            audio_path: None,
            raw_text: raw,
            edited: edited_text != refined,
            refined_text: refined,
            edited_text,
            edit_observed: observed.is_some(),
            edit_outcome: outcome.map(outcome_name),
            learnable: learnable(outcome, end_reason.as_deref()),
            end_reason,
            intended_text: None,
        };
        if options.keeps(&row) {
            rows.push(row);
        }
    }
    if let Some(audio_dir) = audio_dir {
        write_wispr_audio(&copy.database, audio_dir, &mut rows)?;
    }
    Ok((rows, counts))
}

/// Write the kept rows' audio blobs as WAV files and point the rows at them.
fn write_wispr_audio(database: &Path, audio_dir: &Path, rows: &mut [PairRow]) -> Result<()> {
    fs::create_dir_all(audio_dir)?;
    let directory = fs::canonicalize(audio_dir)?;
    // Ids are UUIDs; anything else is left without audio rather than quoted.
    let ids: Vec<&str> = rows
        .iter()
        .map(|row| row.id.as_str())
        .filter(|id| !id.is_empty() && id.chars().all(|c| c.is_ascii_hexdigit() || c == '-'))
        .collect();
    for batch in ids.chunks(500) {
        let list = batch
            .iter()
            .map(|id| format!("'{id}'"))
            .collect::<Vec<_>>()
            .join(",");
        let prefix = directory.display().to_string().replace('\'', "''");
        let query = format!(
            "SELECT transcriptEntityId AS id, writefile('{prefix}/' || transcriptEntityId || '.wav', audio) AS bytes \
             FROM History WHERE audio IS NOT NULL AND transcriptEntityId IN ({list})"
        );
        sqlite_json(database, &query)?;
    }
    for row in rows.iter_mut() {
        let path = directory.join(format!("{}.wav", row.id));
        if path.is_file() {
            row.audio_path = Some(path.display().to_string());
        }
    }
    Ok(())
}

// --- dates (UTC, no time zone database) -----------------------------------

/// Days since 1970-01-01 for a proleptic Gregorian date.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let month_index = (month + 9) % 12;
    let day_of_year = (153 * month_index + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let days = days + 719_468;
    let era = days.div_euclid(146_097);
    let day_of_era = days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// `2026-10-29T18:00:00.123Z`.
pub fn rfc3339_utc(unix_ms: i64) -> String {
    let days = unix_ms.div_euclid(DAY_MS);
    let in_day = unix_ms.rem_euclid(DAY_MS);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        in_day / 3_600_000,
        in_day / 60_000 % 60,
        in_day / 1_000 % 60,
        in_day % 1_000
    )
}

/// `YYYY-MM-DD` as UTC midnight, in unix milliseconds.
pub fn parse_date(text: &str) -> Result<i64> {
    let parts: Vec<&str> = text.trim().split('-').collect();
    let number = |part: &str| part.parse::<i64>().ok();
    match parts.as_slice() {
        [year, month, day] if year.len() == 4 => {
            let (Some(year), Some(month), Some(day)) = (number(year), number(month), number(day))
            else {
                bail!("not a date (YYYY-MM-DD): {text}");
            };
            if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
                bail!("not a date (YYYY-MM-DD): {text}");
            }
            Ok(days_from_civil(year, month, day) * DAY_MS)
        }
        _ => bail!("not a date (YYYY-MM-DD): {text}"),
    }
}

/// Wispr's `2026-10-12 16:52:42.760 +00:00` (also with `T`, without
/// fraction or offset, or with `Z`), in unix milliseconds.
pub fn parse_timestamp(text: &str) -> Option<i64> {
    let text = text.trim();
    let (date, rest) = text.split_at_checked(10)?;
    let days_ms = parse_date(date).ok()?;
    let rest = rest.trim_start_matches(['T', ' ']);
    let (clock, zone) = match rest.find(['+', '-', 'Z', ' ']) {
        Some(at) => (&rest[..at], rest[at..].trim()),
        None => (rest, ""),
    };
    let mut fields = clock.split(':');
    let hours: i64 = fields.next()?.parse().ok()?;
    let minutes: i64 = fields.next()?.parse().ok()?;
    let seconds: f64 = fields.next().unwrap_or("0").parse().ok()?;
    let mut ms =
        days_ms + hours * 3_600_000 + minutes * 60_000 + (seconds * 1_000.0).round() as i64;
    if let Some(sign) = zone.chars().next().filter(|c| matches!(c, '+' | '-')) {
        let mut offset = zone[1..].split(':');
        let offset_hours: i64 = offset.next()?.parse().ok()?;
        let offset_minutes: i64 = offset.next().unwrap_or("0").parse().ok()?;
        let offset_ms = (offset_hours * 60 + offset_minutes) * 60_000;
        ms -= if sign == '+' { offset_ms } else { -offset_ms };
    }
    Some(ms)
}

/// UTC midnight `days` days before `now_ms`: the default first eval day.
pub fn default_eval_from(now_ms: i64, days: i64) -> i64 {
    (now_ms.div_euclid(DAY_MS) - days) * DAY_MS
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

/// Holdout ids from a file: one id per line, or JSONL rows with an `id` or
/// `transcriptEntityId` field.
pub fn read_holdout_ids(path: &Path) -> Result<HashSet<String>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut ids = HashSet::new();
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        if line.starts_with('{') {
            let value: serde_json::Value =
                serde_json::from_str(line).with_context(|| format!("parse {}", path.display()))?;
            for key in ["id", "transcriptEntityId"] {
                if let Some(id) = value.get(key).and_then(serde_json::Value::as_str) {
                    ids.insert(id.to_string());
                }
            }
        } else {
            ids.insert(line.to_string());
        }
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edits::{EditOutcome, UserEditRecord};

    fn temp_dir(name: &str) -> PathBuf {
        // Tests run in parallel and the clock is coarse: a counter keeps
        // their directories apart.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("phonon-pairs-{name}-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        directory
    }

    fn user_edit(pasted: &str, edited: &str, outcome: EditOutcome, end: &str) -> UserEditRecord {
        UserEditRecord {
            schema_version: 1,
            pasted_text: pasted.into(),
            edited_text: edited.into(),
            appended_words: 0,
            skipped: Vec::new(),
            edited_text_within_threshold: None,
            outcome,
            word_edits: String::new(),
            punctuation_edits: String::new(),
            edit_distance: 0.0,
            end_reason: end.into(),
            end_last_keystroke: None,
            end_last_keystroke_age_ms: None,
            observed_ms: 1_000,
            suggestions: Vec::new(),
        }
    }

    /// A corpus entry as the engine writes it, plus a WAV when `audio`.
    fn entry(
        corpus: &Path,
        id: &str,
        unix_ms: u128,
        raw: &str,
        refined: &str,
        edit: Option<UserEditRecord>,
        audio: bool,
    ) {
        let directory = corpus.join(id);
        fs::create_dir_all(&directory).unwrap();
        let metadata = serde_json::json!({
            "schema_version": 2,
            "id": id,
            "created_at_unix_ms": unix_ms,
            "source": "bar",
            "audio_file": "audio.wav",
            "raw_transcript": raw,
            "final_transcript": refined,
            "app_context": {"bundle_id": "com.tinyspeck.slackmacgap", "category": "chat"},
            "user_edit": edit,
        });
        fs::write(
            directory.join("metadata.json"),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();
        if audio {
            fs::write(directory.join("audio.wav"), b"RIFF0000WAVE").unwrap();
        }
    }

    const OCT_01: i64 = 1_790_812_800_000; // 2026-10-01T00:00:00Z
    const OCT_20: i64 = OCT_01 + 19 * DAY_MS;

    fn fixture_corpus() -> PathBuf {
        let corpus = temp_dir("corpus");
        entry(
            &corpus,
            "a-edited",
            (OCT_01 + 3_600_000) as u128,
            "we benchmark coo blast",
            "We benchmark coo blast.",
            Some(user_edit(
                "We benchmark coo blast.",
                "We benchmark cuBLAS.",
                EditOutcome::Correction,
                "observation_window_elapsed",
            )),
            true,
        );
        entry(
            &corpus,
            "b-untouched",
            (OCT_20 + 60_000) as u128,
            "ship it today",
            "Ship it today.",
            None,
            false,
        );
        entry(
            &corpus,
            "c-removed",
            (OCT_01 + 2 * DAY_MS) as u128,
            "never mind",
            "Never mind.",
            Some(user_edit(
                "Never mind.",
                "",
                EditOutcome::Removed,
                "textbox_emptied",
            )),
            false,
        );
        entry(
            &corpus,
            "d-anchor",
            (OCT_20 + 2 * 60_000) as u128,
            "use coo blast",
            "Use coo blast.",
            Some(user_edit(
                "Use coo blast.",
                "Use cuBLAS.",
                EditOutcome::Correction,
                "anchor_mismatch",
            )),
            false,
        );
        // No speech, and a damaged entry.
        let silent = corpus.join("e-silent");
        fs::create_dir_all(&silent).unwrap();
        fs::write(
            silent.join("metadata.json"),
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 2, "id": "e-silent", "created_at_unix_ms": OCT_20 as u128,
                "source": "bar", "audio_file": "audio.wav", "speech_detected": false,
            }))
            .unwrap(),
        )
        .unwrap();
        let broken = corpus.join("f-broken");
        fs::create_dir_all(&broken).unwrap();
        fs::write(broken.join("metadata.json"), b"{not json").unwrap();
        // Dictated into a password prompt: never exported.
        let prompt = corpus.join("g-prompt");
        fs::create_dir_all(&prompt).unwrap();
        fs::write(
            prompt.join("metadata.json"),
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 2, "id": "g-prompt", "created_at_unix_ms": OCT_20 as u128,
                "source": "bar", "audio_file": "audio.wav", "raw_transcript": "open sesame",
                "final_transcript": "Open sesame.",
                "app_context": {"bundle_id": "com.apple.SecurityAgent", "category": "other"},
            }))
            .unwrap(),
        )
        .unwrap();
        corpus
    }

    fn options(eval_from_ms: i64) -> ExportOptions {
        ExportOptions {
            eval_from_ms,
            ..Default::default()
        }
    }

    #[test]
    fn corpus_rows_carry_the_edit_or_the_refined_text_marked_unedited() {
        let corpus = fixture_corpus();
        let (rows, counts) = phonon_rows(&corpus, &options(OCT_20)).unwrap();
        assert_eq!(
            counts,
            ExportCounts {
                unreadable: 1,
                empty: 1,
                private: 1,
            },
            "a damaged entry, a silent one and a password prompt"
        );
        let ids: Vec<&str> = rows.iter().map(|row| row.id.as_str()).collect();
        assert_eq!(ids, ["a-edited", "c-removed", "b-untouched", "d-anchor"]);

        let edited = &rows[0];
        assert_eq!(edited.source, "phonon");
        assert_eq!(edited.timestamp, "2026-10-01T01:00:00.000Z");
        assert_eq!(edited.app_category.as_deref(), Some("chat"));
        assert!(edited
            .audio_path
            .as_deref()
            .unwrap()
            .ends_with("a-edited/audio.wav"));
        assert_eq!(edited.raw_text, "we benchmark coo blast");
        assert_eq!(edited.refined_text, "We benchmark coo blast.");
        assert_eq!(edited.edited_text, "We benchmark cuBLAS.");
        assert!(edited.edited && edited.edit_observed && edited.learnable);
        assert_eq!(edited.edit_outcome.as_deref(), Some("correction"));
        assert_eq!(
            edited.end_reason.as_deref(),
            Some("observation_window_elapsed")
        );
        assert_eq!(edited.split, Split::Train);

        let untouched = &rows[2];
        assert_eq!(untouched.edited_text, untouched.refined_text);
        assert!(!untouched.edited && !untouched.edit_observed && untouched.learnable);
        assert_eq!(untouched.edit_outcome, None);
        assert_eq!(untouched.audio_path, None, "no WAV on disk");
        assert_eq!(untouched.split, Split::Eval, "on the cutoff day");

        // A deleted dictation and an anchor mismatch are recorded but are
        // not labels.
        assert!(!rows[1].learnable && rows[1].edited);
        assert!(!rows[3].learnable && rows[3].edited);
        fs::remove_dir_all(corpus).unwrap();
    }

    #[test]
    fn filters_and_holdouts() {
        let corpus = fixture_corpus();
        let edited_only = ExportOptions {
            edited_only: true,
            ..options(OCT_20)
        };
        let (rows, _) = phonon_rows(&corpus, &edited_only).unwrap();
        assert_eq!(
            rows.iter().map(|row| row.id.as_str()).collect::<Vec<_>>(),
            ["a-edited"],
            "only learnable edits"
        );
        let since = ExportOptions {
            since_ms: Some(OCT_20),
            ..options(OCT_20)
        };
        let (rows, _) = phonon_rows(&corpus, &since).unwrap();
        assert!(rows.iter().all(|row| row.timestamp_unix_ms >= OCT_20));
        assert_eq!(rows.len(), 2);
        // A listed holdout id is eval even from a training day.
        let holdout = ExportOptions {
            holdout_ids: HashSet::from(["a-edited".to_string()]),
            ..options(OCT_20)
        };
        let (rows, _) = phonon_rows(&corpus, &holdout).unwrap();
        assert_eq!(rows[0].split, Split::Eval);
        // No row of a held-out day is ever train.
        let (rows, _) = phonon_rows(&corpus, &options(OCT_01 + 2 * DAY_MS)).unwrap();
        for row in rows {
            assert_eq!(
                row.split == Split::Eval,
                row.timestamp_unix_ms >= OCT_01 + 2 * DAY_MS,
                "{}",
                row.id
            );
        }
        fs::remove_dir_all(corpus).unwrap();
    }

    #[test]
    fn dates_round_trip_and_parse_wispr_timestamps() {
        assert_eq!(parse_date("2026-10-01").unwrap(), OCT_01);
        assert_eq!(parse_date("1970-01-01").unwrap(), 0);
        assert_eq!(rfc3339_utc(OCT_01 + 61_001), "2026-10-01T00:01:01.001Z");
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(rfc3339_utc(951_782_400_000), "2000-02-29T00:00:00.000Z");
        assert!(parse_date("2026-13-01").is_err());
        assert!(parse_date("yesterday").is_err());
        assert_eq!(
            parse_timestamp("2026-10-01 01:00:00.250 +00:00"),
            Some(OCT_01 + 3_600_250)
        );
        assert_eq!(
            parse_timestamp("2026-10-01T01:00:00-06:00"),
            Some(OCT_01 + 7 * 3_600_000)
        );
        assert_eq!(
            parse_timestamp("2026-10-01 01:00:00Z"),
            Some(OCT_01 + 3_600_000)
        );
        assert_eq!(parse_timestamp("garbage"), None);
        assert_eq!(default_eval_from(OCT_20 + 5_000, 14), OCT_20 - 14 * DAY_MS);
        assert_eq!(
            host_of("https://app.slack.com/client/T1"),
            Some("app.slack.com".into())
        );
        assert_eq!(host_of("docs.google.com"), Some("docs.google.com".into()));
    }

    fn sqlite_available() -> bool {
        Command::new("sqlite3")
            .arg("-version")
            .output()
            .is_ok_and(|output| output.status.success())
    }

    /// A small database with Wispr's History columns the export reads.
    fn fixture_wispr_db() -> (PathBuf, PathBuf) {
        let directory = temp_dir("wispr");
        let database = directory.join("flow.sqlite");
        let wav = "52494646240000005741564566";
        let sql = format!(
            "CREATE TABLE History (transcriptEntityId VARCHAR(36) PRIMARY KEY, asrText TEXT, \
             formattedText TEXT, editedText TEXT, timestamp DATETIME, audio BLOB, app VARCHAR(255), \
             url VARCHAR(255), isArchived TINYINT(1) NOT NULL DEFAULT 0, \
             contentObservationEndReason VARCHAR(255));
             INSERT INTO History VALUES ('0a1b2c3d-0000-4000-8000-000000000001', 'ask clod code', \
             'Ask clod code.', 'Ask Claude Code.', '2026-10-01 01:00:00.000 +00:00', X'{wav}', \
             'com.google.Chrome', 'https://claude.ai/new', 0, 'textbox_emptied');
             INSERT INTO History VALUES ('0a1b2c3d-0000-4000-8000-000000000002', 'ship it', \
             '<ol><li>Ship it.</li></ol>', NULL, '2026-10-21 09:00:00.000 +00:00', NULL, \
             'com.apple.mail', NULL, 0, NULL);
             INSERT INTO History VALUES ('0a1b2c3d-0000-4000-8000-000000000003', 'hidden', \
             'Hidden.', NULL, '2026-10-02 09:00:00.000 +00:00', NULL, 'com.apple.Notes', NULL, 1, NULL);
             INSERT INTO History VALUES ('0a1b2c3d-0000-4000-8000-000000000004', 'open sesame', \
             'Open sesame.', NULL, '2026-10-03 09:00:00.000 +00:00', NULL, 'com.apple.SecurityAgent', \
             NULL, 0, NULL);
             INSERT INTO History VALUES ('0a1b2c3d-0000-4000-8000-000000000005', 'vault note', \
             'Vault note.', NULL, '2026-10-03 10:00:00.000 +00:00', NULL, 'com.google.Chrome', \
             'https://my.1password.com/vaults', 0, NULL);"
        );
        let status = Command::new("sqlite3")
            .arg(&database)
            .arg(sql)
            .status()
            .unwrap();
        assert!(status.success());
        (directory, database)
    }

    #[test]
    fn wispr_history_rows_share_the_schema() {
        if !sqlite_available() {
            return;
        }
        let (directory, database) = fixture_wispr_db();
        let before = fs::read(&database).unwrap();
        let audio_dir = directory.join("audio");
        let (rows, counts) = wispr_rows(&database, &options(OCT_20), Some(&audio_dir)).unwrap();
        assert_eq!(
            counts,
            ExportCounts {
                private: 2,
                ..Default::default()
            },
            "a password prompt and a vault page are never exported"
        );
        assert_eq!(rows.len(), 2, "archived rows are left out");
        let edited = &rows[0];
        assert_eq!(edited.source, "wispr");
        assert_eq!(edited.id, "0a1b2c3d-0000-4000-8000-000000000001");
        assert_eq!(edited.raw_text, "ask clod code");
        assert_eq!(edited.refined_text, "Ask clod code.");
        assert_eq!(edited.edited_text, "Ask Claude Code.");
        assert!(edited.edited && edited.edit_observed && edited.learnable);
        assert_eq!(edited.edit_outcome.as_deref(), Some("correction"));
        assert_eq!(edited.end_reason.as_deref(), Some("textbox_emptied"));
        assert_eq!(edited.app_category.as_deref(), Some("ai_prompt"));
        assert_eq!(edited.split, Split::Train);
        let wav = edited.audio_path.as_deref().expect("audio written");
        assert_eq!(&fs::read(wav).unwrap()[..4], b"RIFF");

        let plain = &rows[1];
        assert_eq!(
            plain.refined_text, "Ship it.",
            "Wispr's list markup is stripped"
        );
        assert_eq!(plain.edited_text, "Ship it.");
        assert!(!plain.edited && !plain.edit_observed);
        assert_eq!(plain.app_category.as_deref(), Some("email"));
        assert_eq!(plain.audio_path, None);
        assert_eq!(plain.split, Split::Eval);

        // Without --with-audio nothing is written; the database is untouched.
        let (rows, _) = wispr_rows(&database, &options(OCT_20), None).unwrap();
        assert!(rows.iter().all(|row| row.audio_path.is_none()));
        assert_eq!(fs::read(&database).unwrap(), before);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn holdout_ids_read_from_lines_or_jsonl() {
        let directory = temp_dir("holdout");
        let lines = directory.join("ids.txt");
        fs::write(&lines, "a\n\nb\n").unwrap();
        assert_eq!(
            read_holdout_ids(&lines).unwrap(),
            HashSet::from(["a".to_string(), "b".to_string()])
        );
        let jsonl = directory.join("rows.jsonl");
        fs::write(
            &jsonl,
            "{\"transcriptEntityId\":\"c\",\"text\":\"x\"}\n{\"id\":\"d\"}\n",
        )
        .unwrap();
        assert_eq!(
            read_holdout_ids(&jsonl).unwrap(),
            HashSet::from(["c".to_string(), "d".to_string()])
        );
        fs::remove_dir_all(directory).unwrap();
    }
}
