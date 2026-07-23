use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const DATA_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DictionaryEntry {
    pub phrase: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replacement: Option<String>,
    #[serde(default)]
    pub spoken_forms: Vec<String>,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub starred: bool,
    #[serde(default)]
    pub usage_count: u64,
}

impl DictionaryEntry {
    pub fn canonical(&self) -> &str {
        self.replacement.as_deref().unwrap_or(&self.phrase)
    }

    pub fn forms(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.phrase.as_str()).chain(self.spoken_forms.iter().map(String::as_str))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DictionaryFile {
    pub schema_version: u32,
    pub updated_at_unix_ms: u128,
    pub entries: Vec<DictionaryEntry>,
}

impl Default for DictionaryFile {
    fn default() -> Self {
        Self {
            schema_version: DATA_SCHEMA_VERSION,
            updated_at_unix_ms: now_unix_ms(),
            entries: Vec::new(),
        }
    }
}

impl DictionaryFile {
    pub fn load() -> Result<Self> {
        let path = dictionary_path()?;
        if !path.is_file() {
            return Ok(Self::default());
        }
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
    }

    pub fn save(&mut self) -> Result<()> {
        self.schema_version = DATA_SCHEMA_VERSION;
        self.updated_at_unix_ms = now_unix_ms();
        self.entries.sort_by(|a, b| {
            a.phrase
                .to_lowercase()
                .cmp(&b.phrase.to_lowercase())
                .then_with(|| a.replacement.cmp(&b.replacement))
        });
        write_json_atomic(&dictionary_path()?, self)
    }

    pub fn merge(&mut self, incoming: impl IntoIterator<Item = DictionaryEntry>) -> usize {
        let mut entries: BTreeMap<(String, Option<String>), DictionaryEntry> = self
            .entries
            .drain(..)
            .map(|entry| (dictionary_key(&entry), entry))
            .collect();
        let before = entries.len();
        for mut entry in incoming {
            entry.phrase = entry.phrase.trim().to_string();
            entry.replacement = entry
                .replacement
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty() && value != &entry.phrase);
            entry.spoken_forms = normalized_unique(entry.spoken_forms);
            if entry.phrase.is_empty() {
                continue;
            }
            let key = dictionary_key(&entry);
            entries
                .entry(key)
                .and_modify(|existing| {
                    existing.starred |= entry.starred;
                    existing.usage_count = existing.usage_count.max(entry.usage_count);
                    existing.spoken_forms.extend(entry.spoken_forms.clone());
                    existing.spoken_forms = normalized_unique(existing.spoken_forms.clone());
                    if existing.source.is_empty() {
                        existing.source.clone_from(&entry.source);
                    }
                })
                .or_insert(entry);
        }
        self.entries = entries.into_values().collect();
        self.entries.len().saturating_sub(before)
    }

    pub fn relevant_entries(&self, transcript: &str) -> Vec<&DictionaryEntry> {
        let normalized = normalize(transcript);
        let transcript_words: Vec<&str> = normalized.split_whitespace().collect();
        let mut matches: Vec<(&DictionaryEntry, u64)> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let mut best = 0_u64;
                for form in entry.forms() {
                    let form = normalize(form);
                    if form.is_empty() {
                        continue;
                    }
                    if contains_phrase(&normalized, &form) {
                        best = best.max(1_000_000);
                        continue;
                    }
                    let form_words: Vec<&str> = form.split_whitespace().collect();
                    if form_words.len() == 1 {
                        for word in &transcript_words {
                            if plausibly_same_word(word, form_words[0]) {
                                best = best.max(500_000);
                            }
                        }
                        let form_key = phonetic_key(&form);
                        if form_key.len() >= 3 {
                            for window_len in 1..=3.min(transcript_words.len()) {
                                for window in transcript_words.windows(window_len) {
                                    if phonetic_key(&window.join(" ")) == form_key {
                                        best = best.max(750_000);
                                    }
                                }
                            }
                        }
                    }
                }
                (best > 0).then_some((entry, best + entry.usage_count))
            })
            .collect();
        matches.sort_by(|(a, a_score), (b, b_score)| {
            b_score
                .cmp(a_score)
                .then_with(|| b.starred.cmp(&a.starred))
                .then_with(|| a.phrase.cmp(&b.phrase))
        });
        matches.into_iter().map(|(entry, _)| entry).collect()
    }

    pub fn prepare_polish_input(&self, transcript: &str) -> String {
        self.prepare_polish_input_with_context(transcript, "")
    }

    pub fn prepare_polish_input_with_context(&self, transcript: &str, screen_text: &str) -> String {
        if self.entries.is_empty() {
            return transcript.to_string();
        }
        let likely_entries = self.relevant_entries(transcript);
        let normalized_screen = normalize(screen_text);
        let exact_corrected = self.apply_exact_replacements(transcript).text;
        let phonetic_matches = self.phonetic_matches(&exact_corrected);
        let mut pre_corrected = exact_corrected;
        for (spoken, canonical) in &phonetic_matches {
            pre_corrected = replace_ascii_phrase(&pre_corrected, spoken, canonical).0;
        }
        let terms = likely_entries
            .iter()
            .filter(|entry| entry.replacement.is_none())
            .map(|entry| entry.canonical())
            .collect::<Vec<_>>();
        let likely_terms = likely_entries
            .iter()
            .map(|entry| entry.canonical())
            .collect::<Vec<_>>();
        let replacements = likely_entries
            .iter()
            .filter_map(|entry| {
                entry
                    .replacement
                    .as_ref()
                    .map(|replacement| format!("{} => {replacement}", entry.phrase))
            })
            .collect::<Vec<_>>();
        let screen_terms = self
            .screen_confirmed_entries(transcript, screen_text)
            .iter()
            .filter(|entry| contains_phrase(&normalized_screen, &normalize(entry.canonical())))
            .map(|entry| entry.canonical())
            .collect::<Vec<_>>();
        let phonetic_matches = phonetic_matches
            .iter()
            .map(|(spoken, canonical)| format!("{spoken} => {canonical}"))
            .collect::<Vec<_>>();
        format!(
            "<phonon_dictionary>\ncanonical_terms: {}\nlikely_terms: {}\nscreen_confirmed_terms: {}\nphonetic_matches: {}\nexact_replacements: {}\n</phonon_dictionary>\n<transcript>\n{}\n</transcript>",
            terms.join(", "),
            likely_terms.join(", "),
            screen_terms.join(", "),
            phonetic_matches.join("; "),
            replacements.join("; "),
            pre_corrected.trim()
        )
    }

    fn phonetic_matches(&self, transcript: &str) -> Vec<(String, String)> {
        let mut by_key: BTreeMap<String, Option<String>> = BTreeMap::new();
        for entry in &self.entries {
            for form in entry.forms() {
                if normalize(form).split_whitespace().count() != 1 {
                    continue;
                }
                let key = phonetic_key(form);
                if key.len() < 3 {
                    continue;
                }
                let canonical = entry.canonical().to_string();
                by_key
                    .entry(key)
                    .and_modify(|existing| {
                        if existing.as_deref() != Some(canonical.as_str()) {
                            *existing = None;
                        }
                    })
                    .or_insert(Some(canonical));
            }
        }

        let normalized = normalize(transcript);
        let words = normalized.split_whitespace().collect::<Vec<_>>();
        let mut occupied = vec![false; words.len()];
        let mut matches = Vec::new();
        for window_len in (1..=3.min(words.len())).rev() {
            for start in 0..=words.len() - window_len {
                if occupied[start..start + window_len]
                    .iter()
                    .any(|value| *value)
                {
                    continue;
                }
                let spoken = words[start..start + window_len].join(" ");
                let Some(Some(canonical)) = by_key.get(&phonetic_key(&spoken)) else {
                    continue;
                };
                if normalize(canonical) == spoken {
                    continue;
                }
                occupied[start..start + window_len].fill(true);
                matches.push((spoken, canonical.clone()));
            }
        }
        matches
    }

    pub fn screen_confirmed_terms(&self, transcript: &str, screen_text: &str) -> Vec<String> {
        self.screen_confirmed_entries(transcript, screen_text)
            .into_iter()
            .map(|entry| entry.canonical().to_string())
            .collect()
    }

    fn screen_confirmed_entries(
        &self,
        transcript: &str,
        screen_text: &str,
    ) -> Vec<&DictionaryEntry> {
        let normalized_screen = normalize(screen_text);
        self.entries
            .iter()
            .filter(|entry| {
                contains_phrase(&normalized_screen, &normalize(entry.canonical()))
                    && entry
                        .forms()
                        .any(|form| contextually_similar_phrase(transcript, form))
            })
            .collect()
    }

    pub fn apply_exact_replacements(&self, text: &str) -> CorrectionResult {
        let mut output = text.to_string();
        let mut applied = Vec::new();
        for entry in &self.entries {
            let Some(replacement) = entry.replacement.as_deref() else {
                continue;
            };
            for form in entry.forms() {
                let (next, count) = replace_ascii_phrase(&output, form, replacement);
                if count > 0 {
                    output = next;
                    applied.push(AppliedCorrection {
                        from: form.to_string(),
                        to: replacement.to_string(),
                        count,
                    });
                }
            }
        }
        CorrectionResult {
            text: output,
            applied,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AppliedCorrection {
    pub from: String,
    pub to: String,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorrectionResult {
    pub text: String,
    pub applied: Vec<AppliedCorrection>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LlmMetadata {
    pub latency_ms: f64,
    pub ttft_ms: f64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecordingMetadata {
    pub schema_version: u32,
    pub id: String,
    pub created_at_unix_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recorded_at: Option<String>,
    pub source: String,
    pub audio_file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub microphone: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_detected: Option<bool>,
    #[serde(default)]
    pub raw_transcript: String,
    #[serde(default)]
    pub final_transcript: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intended_transcript: Option<String>,
    #[serde(default)]
    pub dictionary_corrections: Vec<AppliedCorrection>,
    #[serde(default)]
    pub screen_context_terms: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm: Option<LlmMetadata>,
}

impl RecordingMetadata {
    fn new(id: String, source: &str, audio_file: String) -> Self {
        Self {
            schema_version: DATA_SCHEMA_VERSION,
            id,
            created_at_unix_ms: now_unix_ms(),
            recorded_at: None,
            source: source.to_string(),
            audio_file,
            microphone: None,
            audio_duration_ms: None,
            speech_detected: None,
            raw_transcript: String::new(),
            final_transcript: String::new(),
            intended_transcript: None,
            dictionary_corrections: Vec::new(),
            screen_context_terms: Vec::new(),
            llm: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct UsageStats {
    pub recordings: u64,
    pub words: u64,
    pub speaking_ms: u64,
    pub dictionary_fixes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SettingsFile {
    pub schema_version: u32,
    #[serde(default = "default_true")]
    pub streaming: bool,
    #[serde(default = "default_true")]
    pub local_history: bool,
    #[serde(default = "default_true")]
    pub screen_context: bool,
    #[serde(default = "default_microphone_priority")]
    pub microphone_priority: Vec<String>,
    #[serde(default = "default_true")]
    pub instant_mic: bool,
    #[serde(default = "default_shortcut_mode")]
    pub shortcut_mode: String,
}

impl Default for SettingsFile {
    fn default() -> Self {
        Self {
            schema_version: DATA_SCHEMA_VERSION,
            streaming: true,
            local_history: true,
            screen_context: true,
            microphone_priority: default_microphone_priority(),
            instant_mic: true,
            shortcut_mode: default_shortcut_mode(),
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_microphone_priority() -> Vec<String> {
    Vec::new()
}

fn default_shortcut_mode() -> String {
    "both".into()
}

impl SettingsFile {
    pub fn load_or_create() -> Result<Self> {
        let path = settings_path()?;
        if path.is_file() {
            let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            let mut settings: Self = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", path.display()))?;
            settings.schema_version = DATA_SCHEMA_VERSION;
            write_json_atomic(&path, &settings)?;
            return Ok(settings);
        }
        let settings = Self::default();
        write_json_atomic(&path, &settings)?;
        Ok(settings)
    }
}

pub fn app_support_dir() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home)
        .join("Library")
        .join("Application Support")
        .join("Phonon"))
}

pub fn dictionary_path() -> Result<PathBuf> {
    Ok(app_support_dir()?.join("dictionary.json"))
}

pub fn settings_path() -> Result<PathBuf> {
    Ok(app_support_dir()?.join("settings.json"))
}

pub fn corpus_dir() -> Result<PathBuf> {
    Ok(app_support_dir()?.join("Corpus"))
}

pub fn metadata_path_for_audio(audio_path: &Path) -> Result<PathBuf> {
    let parent = audio_path
        .parent()
        .with_context(|| format!("audio path has no parent: {}", audio_path.display()))?;
    Ok(parent.join("metadata.json"))
}

pub fn register_recording(audio_path: &Path, source: &str) -> Result<RecordingMetadata> {
    let parent = audio_path
        .parent()
        .with_context(|| format!("audio path has no parent: {}", audio_path.display()))?;
    fs::create_dir_all(parent)?;
    let id = parent
        .file_name()
        .and_then(|value| value.to_str())
        .context("recording directory has no UTF-8 id")?
        .to_string();
    let path = parent.join("metadata.json");
    if path.is_file() {
        return load_recording(&path);
    }
    let audio_file = audio_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("audio.wav")
        .to_string();
    let mut metadata = RecordingMetadata::new(id, source, audio_file);
    metadata.audio_duration_ms = wav_duration_ms(audio_path).ok();
    save_recording(&metadata)?;
    Ok(metadata)
}

pub fn import_recording(
    audio_path: &Path,
    id: &str,
    source: &str,
    final_transcript: &str,
    recorded_at: Option<String>,
) -> Result<(RecordingMetadata, bool)> {
    if id.is_empty() || id.contains('/') || id.contains('\\') || id == "." || id == ".." {
        bail!("invalid recording id: {id:?}");
    }
    if !audio_path.is_file() {
        bail!("audio file not found: {}", audio_path.display());
    }
    let directory = corpus_dir()?.join(id);
    let metadata_path = directory.join("metadata.json");
    if metadata_path.is_file() {
        return Ok((load_recording(&metadata_path)?, false));
    }
    fs::create_dir_all(&directory)?;
    let target = directory.join("audio.wav");
    fs::copy(audio_path, &target)
        .with_context(|| format!("copy {} to {}", audio_path.display(), target.display()))?;
    let mut metadata = RecordingMetadata::new(id.to_string(), source, "audio.wav".into());
    metadata.recorded_at = recorded_at;
    metadata.audio_duration_ms = wav_duration_ms(&target).ok();
    metadata.final_transcript = final_transcript.trim().to_string();
    save_recording(&metadata)?;
    Ok((metadata, true))
}

pub fn load_recording(path: &Path) -> Result<RecordingMetadata> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn save_recording(metadata: &RecordingMetadata) -> Result<()> {
    let path = corpus_dir()?.join(&metadata.id).join("metadata.json");
    write_json_atomic(&path, metadata)
}

pub fn set_speech_detected(id: &str, detected: bool) -> Result<()> {
    let mut metadata = load_recording_by_id(id)?;
    metadata.speech_detected = Some(detected);
    save_recording(&metadata)
}

pub fn list_recordings() -> Result<Vec<RecordingMetadata>> {
    let directory = corpus_dir()?;
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut recordings = Vec::new();
    for entry in fs::read_dir(&directory)? {
        let entry = entry?;
        let path = entry.path().join("metadata.json");
        if path.is_file() {
            recordings.push(load_recording(&path)?);
        }
    }
    recordings.sort_by_key(|recording| Reverse(recording.created_at_unix_ms));
    Ok(recordings)
}

pub fn load_recording_by_id(id: &str) -> Result<RecordingMetadata> {
    let path = corpus_dir()?.join(id).join("metadata.json");
    if !path.is_file() {
        bail!("recording not found: {id}");
    }
    load_recording(&path)
}

pub fn usage_stats() -> Result<UsageStats> {
    let mut stats = UsageStats::default();
    for recording in list_recordings()? {
        stats.recordings += 1;
        let text = if recording.final_transcript.is_empty() {
            &recording.raw_transcript
        } else {
            &recording.final_transcript
        };
        stats.words += text.split_whitespace().count() as u64;
        stats.speaking_ms += recording.audio_duration_ms.unwrap_or(0);
        stats.dictionary_fixes += recording
            .dictionary_corrections
            .iter()
            .map(|correction| correction.count)
            .sum::<u64>();
    }
    Ok(stats)
}

pub fn set_intended_transcript(id: &str, intended: &str) -> Result<RecordingMetadata> {
    let mut metadata = load_recording_by_id(id)?;
    metadata.intended_transcript = Some(intended.trim().to_string());
    save_recording(&metadata)?;
    Ok(metadata)
}

pub fn extract_transcript_payload(output: &str) -> &str {
    let Some(start) = output.find("<transcript>") else {
        let trimmed = output.trim();
        if trimmed.contains("<phonon_dictionary")
            || trimmed.contains("</phonon_dictionary>")
            || trimmed.contains("canonical_terms:")
        {
            return "";
        }
        return trimmed;
    };
    let content_start = start + "<transcript>".len();
    let remaining = &output[content_start..];
    let end = remaining.find("</transcript>").unwrap_or(remaining.len());
    remaining[..end].trim()
}

pub fn safe_polish_output(
    dictionary: &DictionaryFile,
    source: &str,
    polished: &str,
) -> CorrectionResult {
    let polished = extract_transcript_payload(polished);
    let source_corrected = dictionary.apply_exact_replacements(source);
    let polished_corrected = dictionary.apply_exact_replacements(polished);
    let source_words = normalized_words(&source_corrected.text);
    let polished_words = normalized_words(&polished_corrected.text);
    let collapsed =
        polished.trim().is_empty() || (source_words.len() > 1 && polished_words.len() <= 1);
    let implausible_expansion = polished_words.len() > source_words.len().saturating_mul(2) + 8
        || polished.len() > source.len().saturating_mul(4) + 64;
    let meaning_only_deletion = source_words.len().saturating_sub(polished_words.len()) <= 2
        && deletes_meaningful_words_only(&source_words, &polished_words);
    if collapsed || implausible_expansion || meaning_only_deletion {
        source_corrected
    } else {
        polished_corrected
    }
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|character| character.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect()
        })
        .filter(|word: &String| !word.is_empty())
        .collect()
}

fn deletes_meaningful_words_only(source: &[String], candidate: &[String]) -> bool {
    if candidate.len() >= source.len() {
        return false;
    }
    let mut candidate_index = 0;
    let mut removed = Vec::new();
    for (source_index, word) in source.iter().enumerate() {
        if candidate.get(candidate_index) == Some(word) {
            candidate_index += 1;
        } else {
            removed.push(source_index);
        }
    }
    if candidate_index != candidate.len() {
        return false;
    }
    removed.into_iter().any(|index| {
        let word = source[index].as_str();
        let filler = matches!(word, "uh" | "um" | "erm" | "hmm");
        let duplicate = index > 0 && source[index - 1] == source[index]
            || index + 1 < source.len() && source[index + 1] == source[index];
        !filler && !duplicate
    })
}

pub fn delete_recording(id: &str) -> Result<()> {
    if id.is_empty() || id.contains('/') || id.contains('\\') || id == "." || id == ".." {
        bail!("invalid recording id: {id:?}");
    }
    let directory = corpus_dir()?.join(id);
    let metadata = directory.join("metadata.json");
    if !metadata.is_file() {
        bail!("recording not found: {id}");
    }
    fs::remove_dir_all(&directory).with_context(|| format!("delete {}", directory.display()))
}

fn dictionary_key(entry: &DictionaryEntry) -> (String, Option<String>) {
    (
        entry.phrase.trim().to_lowercase(),
        entry.replacement.as_ref().map(|value| value.to_lowercase()),
    )
}

fn normalized_unique(values: Vec<String>) -> Vec<String> {
    let mut by_key = BTreeMap::new();
    for value in values {
        let value = value.trim().to_string();
        if !value.is_empty() {
            by_key.entry(value.to_lowercase()).or_insert(value);
        }
    }
    by_key.into_values().collect()
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn contains_phrase(text: &str, phrase: &str) -> bool {
    format!(" {text} ").contains(&format!(" {phrase} "))
}

fn plausibly_same_word(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    let longest = left.chars().count().max(right.chars().count());
    if longest < 5 || left.chars().next() != right.chars().next() {
        return false;
    }
    let allowed = if longest >= 6 { 2 } else { 1 };
    levenshtein(left, right) <= allowed
}

fn phonetic_key(value: &str) -> String {
    let mut output = String::new();
    for character in normalize(value)
        .chars()
        .filter(|character| character.is_alphanumeric())
    {
        let character = match character {
            'c' | 'q' => 'k',
            other => other,
        };
        if matches!(character, 'a' | 'e' | 'i' | 'o' | 'u' | 'y') {
            continue;
        }
        if !output.ends_with(character) {
            output.push(character);
        }
    }
    output
}

/// OCR provides an independent, exact observation of the canonical term, so
/// this match can tolerate a wider ASR miss than global dictionary retrieval.
/// It remains bounded to nearby transcript word windows and never activates a
/// term that is absent from the screen.
fn contextually_similar_phrase(transcript: &str, form: &str) -> bool {
    let transcript = normalize(transcript);
    let form = normalize(form);
    if transcript.is_empty() || form.is_empty() {
        return false;
    }
    if contains_phrase(&transcript, &form) {
        return true;
    }

    let transcript_words = transcript.split_whitespace().collect::<Vec<_>>();
    let form_words = form.split_whitespace().collect::<Vec<_>>();
    let compact_form = form_words.concat();
    let shortest_window = form_words.len().saturating_sub(1).max(1);
    let longest_window = (form_words.len() + 1).min(transcript_words.len());
    for window_len in shortest_window..=longest_window {
        for window in transcript_words.windows(window_len) {
            if broadly_similar_word(&window.concat(), &compact_form) {
                return true;
            }
        }
    }
    false
}

fn broadly_similar_word(left: &str, right: &str) -> bool {
    let longest = left.chars().count().max(right.chars().count());
    if longest < 5 {
        return false;
    }
    let distance = levenshtein(left, right);
    distance <= 3 && distance * 2 <= longest
}

fn levenshtein(left: &str, right: &str) -> usize {
    let right: Vec<char> = right.chars().collect();
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    for (row, left_character) in left.chars().enumerate() {
        let mut current = vec![row + 1];
        for (column, right_character) in right.iter().enumerate() {
            let cost = usize::from(left_character != *right_character);
            current.push(
                (current[column] + 1)
                    .min(previous[column + 1] + 1)
                    .min(previous[column] + cost),
            );
        }
        previous = current;
    }
    previous[right.len()]
}

fn replace_ascii_phrase(text: &str, from: &str, to: &str) -> (String, u64) {
    let needle = from.trim();
    if needle.is_empty() || needle.eq_ignore_ascii_case(to) {
        return (text.to_string(), 0);
    }
    let lower_text = text.to_ascii_lowercase();
    let lower_needle = needle.to_ascii_lowercase();
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    let mut count = 0;
    while let Some(relative) = lower_text[cursor..].find(&lower_needle) {
        let start = cursor + relative;
        let end = start + lower_needle.len();
        let left_boundary = start == 0 || !text.as_bytes()[start - 1].is_ascii_alphanumeric();
        let right_boundary = end == text.len() || !text.as_bytes()[end].is_ascii_alphanumeric();
        if left_boundary && right_boundary {
            output.push_str(&text[cursor..start]);
            output.push_str(to);
            cursor = end;
            count += 1;
        } else {
            output.push_str(&text[cursor..end]);
            cursor = end;
        }
    }
    output.push_str(&text[cursor..]);
    (output, count)
}

fn wav_duration_ms(path: &Path) -> Result<u64> {
    let bytes = fs::read(path).with_context(|| format!("read wav {}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("unsupported WAV header");
    }
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]) as u64;
    let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]) as u64;
    let bits = u16::from_le_bytes([bytes[34], bytes[35]]) as u64;
    let data_bytes = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]) as u64;
    let bytes_per_second = sample_rate * channels * bits / 8;
    if bytes_per_second == 0 {
        bail!("invalid WAV rate");
    }
    Ok(data_bytes * 1_000 / bytes_per_second)
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(&temporary, bytes).with_context(|| format!("write {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| format!("replace {}", path.display()))?;
    Ok(())
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{
        extract_transcript_payload, safe_polish_output, DictionaryEntry, DictionaryFile,
        SettingsFile,
    };

    fn entry(phrase: &str, replacement: Option<&str>) -> DictionaryEntry {
        DictionaryEntry {
            phrase: phrase.to_string(),
            replacement: replacement.map(str::to_string),
            spoken_forms: Vec::new(),
            source: "test".into(),
            starred: false,
            usage_count: 0,
        }
    }

    #[test]
    fn exact_replacements_are_case_insensitive_and_word_bounded() {
        let dictionary = DictionaryFile {
            entries: vec![entry("black well", Some("Blackwell"))],
            ..Default::default()
        };
        let result = dictionary.apply_exact_replacements("BLACK WELL and black wellness");
        assert_eq!(result.text, "Blackwell and black wellness");
        assert_eq!(result.applied[0].count, 1);
    }

    #[test]
    fn relevant_entries_retrieve_exact_and_fuzzy_terms() {
        let dictionary = DictionaryFile {
            entries: vec![
                entry("black well", Some("Blackwell")),
                entry("Pufferlib", None),
                entry("unrelated", None),
            ],
            ..Default::default()
        };
        let entries = dictionary.relevant_entries("Use black well with puffer lib");
        assert!(entries.iter().any(|entry| entry.phrase == "black well"));
        assert!(!entries.iter().any(|entry| entry.phrase == "unrelated"));
    }

    #[test]
    fn global_fuzzy_retrieval_rejects_short_acronyms_and_unrelated_words() {
        let dictionary = DictionaryFile {
            entries: vec![
                entry("FOV", None),
                entry("Vercel", None),
                entry("Claude", None),
            ],
            ..Default::default()
        };
        let entries = dictionary.relevant_entries("Ask cloud for this kernel");
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.canonical())
                .collect::<Vec<_>>(),
            vec!["Claude"]
        );
    }

    #[test]
    fn merge_deduplicates_case_insensitively() {
        let mut dictionary = DictionaryFile::default();
        assert_eq!(dictionary.merge([entry("vLLM", None)]), 1);
        assert_eq!(dictionary.merge([entry("VLLM", None)]), 0);
        assert_eq!(dictionary.entries.len(), 1);
    }

    #[test]
    fn older_settings_gain_native_app_defaults() {
        let settings: SettingsFile = serde_json::from_str(
            r#"{"schema_version":1,"streaming":true,"local_history":true,"screen_context":true,"microphone_priority":["USB Microphone"]}"#,
        )
        .unwrap();
        assert!(settings.instant_mic);
        assert_eq!(settings.shortcut_mode, "both");
    }

    #[test]
    fn extracts_transcript_when_model_echoes_dictionary_envelope() {
        let output =
            "<phonon_dictionary>terms</phonon_dictionary>\n<transcript>\nUse vLLM.\n</transcript>";
        assert_eq!(extract_transcript_payload(output), "Use vLLM.");
        assert_eq!(extract_transcript_payload("Use CUDA."), "Use CUDA.");
        assert_eq!(
            extract_transcript_payload("<phonon_dictionary>\ncanonical_terms: CUDA"),
            ""
        );
    }

    #[test]
    fn safe_polish_rejects_dictionary_envelopes_and_implausible_expansion() {
        let dictionary = DictionaryFile::default();
        assert_eq!(
            safe_polish_output(
                &dictionary,
                "Can you hear me?",
                "<phonon_dictionary>\ncanonical_terms: CUDA, cuDNN"
            )
            .text,
            "Can you hear me?"
        );
        assert_eq!(
            safe_polish_output(
                &dictionary,
                "Okay.",
                "This response expanded into far too many unrelated words and should never be inserted into the focused application."
            )
            .text,
            "Okay."
        );
    }

    #[test]
    fn safe_polish_preserves_meaningful_small_word_deletions() {
        let dictionary = DictionaryFile {
            entries: vec![entry("h one hundred", Some("H100"))],
            ..Default::default()
        };
        let result = safe_polish_output(
            &dictionary,
            "Move the job to an h one hundred.",
            "Move the job to H100.",
        );
        assert_eq!(result.text, "Move the job to an H100.");
    }

    #[test]
    fn safe_polish_allows_filler_and_duplicate_removal() {
        let dictionary = DictionaryFile::default();
        assert_eq!(
            safe_polish_output(&dictionary, "Uh run the test.", "Run the test.").text,
            "Run the test."
        );
        assert_eq!(
            safe_polish_output(&dictionary, "Run the the test.", "Run the test.").text,
            "Run the test."
        );
    }

    #[test]
    fn screen_context_only_boosts_terms_relevant_to_the_transcript() {
        let dictionary = DictionaryFile {
            entries: vec![
                entry("Claude", None),
                entry("CUDA", None),
                entry("unrelated", None),
            ],
            ..Default::default()
        };
        let terms = dictionary.screen_confirmed_terms(
            "Ask cloud to repair the CUDA code.",
            "Claude Settings CUDA Toolkit unrelated",
        );
        assert!(terms.contains(&"Claude".to_string()));
        assert!(terms.contains(&"CUDA".to_string()));
        assert!(!terms.contains(&"unrelated".to_string()));
    }

    #[test]
    fn phonetic_retrieval_and_screen_context_recover_a_wider_asr_miss() {
        let dictionary = DictionaryFile {
            entries: vec![entry("cuBLAS", None), entry("unrelated", None)],
            ..Default::default()
        };
        assert_eq!(
            dictionary
                .relevant_entries("Use Kubloss here")
                .into_iter()
                .map(|entry| entry.canonical())
                .collect::<Vec<_>>(),
            vec!["cuBLAS"]
        );

        let terms = dictionary.screen_confirmed_terms(
            "Use Kubloss here",
            "CUDA Toolkit cuBLAS documentation unrelated",
        );
        assert_eq!(terms, vec!["cuBLAS"]);

        let input = dictionary.prepare_polish_input_with_context(
            "Use Kubloss here",
            "CUDA Toolkit cuBLAS documentation",
        );
        assert!(input.contains("canonical_terms: cuBLAS"));
        assert!(input.contains("screen_confirmed_terms: cuBLAS"));
    }

    #[test]
    fn polish_input_considers_the_complete_dictionary_but_sends_only_relevant_terms() {
        let dictionary = DictionaryFile {
            entries: vec![
                entry("cuBLAS", None),
                entry("cuDNN", None),
                entry("unrelated", None),
            ],
            ..Default::default()
        };
        let transcript = "Do you know the difference between Koo Bloss and Koo DNN?";
        let relevant = dictionary
            .relevant_entries(transcript)
            .into_iter()
            .map(|entry| entry.canonical())
            .collect::<Vec<_>>();
        assert!(relevant.contains(&"cuBLAS"));
        assert!(relevant.contains(&"cuDNN"));

        let input = dictionary.prepare_polish_input(transcript);
        assert!(input.contains("canonical_terms: cuBLAS, cuDNN"));
        assert!(!input.contains("unrelated"));
        assert!(input.contains("likely_terms: cuBLAS, cuDNN"));
        assert!(input.contains("phonetic_matches: koo bloss => cuBLAS; koo dnn => cuDNN"));
        assert!(input.contains(
            "<transcript>\nDo you know the difference between cuBLAS and cuDNN?\n</transcript>"
        ));
    }
}
