use anyhow::{bail, Context, Result};
pub use phonon_llm::PolishConfig;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
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

/// One term for the ASR sidecar to favor, with the ways it may be said
/// (`{"cmd":"terms","entries":[{"term":..,"spoken_forms":[..]}]}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsrTermEntry {
    pub term: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spoken_forms: Vec<String>,
}

/// Most unfamiliar screen terms one dictation carries (ASR biasing and the envelope).
pub const MAX_SCREEN_TERMS: usize = 24;

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
        Self::load_from(&dictionary_path()?)
    }

    /// `load` from a file of the caller's choosing (Windows and Linux keep
    /// `dictionary.json` in their own data root); no file is an empty dictionary.
    pub fn load_from(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
    }

    pub fn save(&mut self) -> Result<()> {
        self.save_to(&dictionary_path()?)
    }

    /// `save` to a file of the caller's choosing: sorted, stamped, written atomically.
    pub fn save_to(&mut self, path: &Path) -> Result<()> {
        self.schema_version = DATA_SCHEMA_VERSION;
        self.updated_at_unix_ms = now_unix_ms();
        self.entries.sort_by(|a, b| {
            a.phrase
                .to_lowercase()
                .cmp(&b.phrase.to_lowercase())
                .then_with(|| a.replacement.cmp(&b.replacement))
        });
        write_json_atomic(path, self)
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
        // The phonetic keys of every 1- to 3-word window, once per transcript rather than once
        // per dictionary entry (that loop made the final polish's envelope cost about 0.18 ms
        // per transcript word with a 105 KB dictionary).
        let window_keys: std::collections::HashSet<String> = (1..=3.min(transcript_words.len()))
            .flat_map(|window_len| {
                transcript_words
                    .windows(window_len)
                    .map(|window| phonetic_key(&window.join(" ")))
            })
            .collect();
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
                        if form_key.len() >= 3 && window_keys.contains(&form_key) {
                            best = best.max(750_000);
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

    /// Entries ordered by starred first, then highest usage count, keeping file order on ties.
    fn asr_ordered_entries(&self) -> Vec<&DictionaryEntry> {
        let mut entries: Vec<&DictionaryEntry> = self.entries.iter().collect();
        entries.sort_by(|a, b| {
            b.starred
                .cmp(&a.starred)
                .then_with(|| b.usage_count.cmp(&a.usage_count))
        });
        entries
    }

    /// Ordinary single lowercase English words (`table`, `feed`) are skipped for ASR boosting;
    /// capitalized proper nouns (`Tetra`, `Claude`, `Claude Code`, `Opus`, `Codex`) and entries
    /// with explicit spoken forms or replacements are kept, matching `sidecar/asr_terms.py`.
    fn include_in_asr(entry: &DictionaryEntry) -> bool {
        let term = entry.canonical().trim();
        if term.is_empty() {
            return false;
        }
        if !entry.spoken_forms.is_empty() || entry.replacement.is_some() {
            return true;
        }
        let plain_lowercase_word = term
            .chars()
            .all(|character| character.is_alphabetic() && character.is_lowercase());
        !(plain_lowercase_word && is_ordinary_english(term))
    }

    /// The written forms speech recognition boosts (the ASR sidecar's `terms`): each
    /// entry's canonical spelling once, minus plain lowercase ordinary English words.
    pub fn asr_terms(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        self.asr_ordered_entries()
            .into_iter()
            .filter(|entry| Self::include_in_asr(entry))
            .map(|entry| entry.canonical().trim())
            .filter(|term| seen.insert(term.to_string()))
            .map(str::to_string)
            .collect()
    }

    /// `asr_terms` with what each term may sound like (the ASR sidecar's `entries`): the
    /// entry's spoken forms plus, for a replacement, the phrase it replaces. Same terms in
    /// the same order as `asr_terms`; entries sharing a term pool their forms.
    pub fn asr_term_entries(&self) -> Vec<AsrTermEntry> {
        let mut out: Vec<AsrTermEntry> = Vec::new();
        let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for entry in self.asr_ordered_entries() {
            if !Self::include_in_asr(entry) {
                continue;
            }
            let term = entry.canonical().trim();
            let slot = *index.entry(term.to_string()).or_insert_with(|| {
                out.push(AsrTermEntry {
                    term: term.to_string(),
                    spoken_forms: Vec::new(),
                });
                out.len() - 1
            });
            let target = &mut out[slot];
            let replaced = entry
                .replacement
                .as_ref()
                .map(|_| entry.phrase.as_str())
                .into_iter();
            for form in entry
                .spoken_forms
                .iter()
                .map(String::as_str)
                .chain(replaced)
            {
                let form = form.trim();
                if form.is_empty()
                    || form.eq_ignore_ascii_case(&target.term)
                    || target
                        .spoken_forms
                        .iter()
                        .any(|seen| seen.eq_ignore_ascii_case(form))
                {
                    continue;
                }
                target.spoken_forms.push(form.to_string());
            }
        }
        out
    }

    pub fn prepare_polish_input(&self, transcript: &str) -> String {
        self.prepare_polish_input_with_context(transcript, "")
    }

    pub fn prepare_polish_input_with_context(&self, transcript: &str, screen_text: &str) -> String {
        self.prepare_polish_input_with_screen_terms(transcript, screen_text, &[])
    }

    /// The dictionary envelope, with `extra_screen_terms` (unfamiliar terms read off the
    /// screen at key down, `extract_screen_terms`) added to `screen_confirmed_terms` even
    /// when the dictionary does not hold them. Without extras this is exactly
    /// `prepare_polish_input_with_context` as it always was.
    pub fn prepare_polish_input_with_screen_terms(
        &self,
        transcript: &str,
        screen_text: &str,
        extra_screen_terms: &[String],
    ) -> String {
        if self.entries.is_empty() && extra_screen_terms.is_empty() {
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
        let screen_terms = merge_screen_terms(&screen_terms, extra_screen_terms);
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

    /// Phonetic pre-correction is deliberately blind to meaning: it matches on a
    /// vowel-free consonant skeleton, so `please` and `pallas`, or `profile` and
    /// `prefill`, are indistinguishable. Guessing is only acceptable on tokens
    /// the speech model is unlikely to have got right in the first place, so an
    /// ordinary English word is never replaced. A term the user actually taught
    /// is an exact replacement and does not come through here.
    fn phonetic_matches(&self, transcript: &str) -> Vec<(String, String)> {
        // Key -> the canonical spelling plus the word counts it was indexed
        // under. Both matter. Indexing only single-word forms left every
        // multi-word term unreachable, and the counts bound how far a match may
        // stretch: with vowels dropped, `koo bloss is` keys the same as `koo
        // bloss`, so an unbounded window absorbs the next word and deletes it.
        let mut by_key: BTreeMap<String, Option<(String, BTreeSet<usize>)>> = BTreeMap::new();
        for entry in &self.entries {
            for form in entry.forms() {
                let form_words = normalize(form).split_whitespace().count();
                if form_words == 0 || form_words > MAX_PHONETIC_WINDOW {
                    // Longer than any window the scan below builds, so a key
                    // for it could never be matched.
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
                        match existing {
                            // `black well` and `blackwell` are one term written
                            // two ways, not an ambiguity. Keep whichever spells
                            // the capitals, since restoring them is the point.
                            Some((known, counts)) if same_term(known, &canonical) => {
                                counts.insert(form_words);
                                if capital_count(&canonical) > capital_count(known) {
                                    *known = canonical.clone();
                                }
                            }
                            // Two genuinely different terms sound alike; neither
                            // can be chosen safely, so drop the key entirely.
                            _ => *existing = None,
                        }
                    })
                    .or_insert_with(|| Some((canonical, BTreeSet::from([form_words]))));
            }
        }

        let normalized = normalize(transcript);
        let words = normalized.split_whitespace().collect::<Vec<_>>();
        // The same words as written, for the casing `normalize` drops.
        let written = written_words(transcript);
        let mut occupied = vec![false; words.len()];
        let mut matches = Vec::new();
        for window_len in (1..=MAX_PHONETIC_WINDOW.min(words.len())).rev() {
            for start in 0..=words.len() - window_len {
                if occupied[start..start + window_len]
                    .iter()
                    .any(|value| *value)
                {
                    continue;
                }
                let spoken = words[start..start + window_len].join(" ");
                let Some(Some((canonical, form_words))) = by_key.get(&phonetic_key(&spoken)) else {
                    continue;
                };
                // A recognizer splits or joins a term by one word, writing
                // `black well` for `Blackwell`. It does not fuse a term with
                // two of its neighbours, so a window that far from the taught
                // form is a key collision rather than a mishearing.
                if !form_words
                    .iter()
                    .any(|count| window_len.abs_diff(*count) <= 1)
                {
                    continue;
                }
                // A word the key cannot represent at all -- `a`, `I` -- is free
                // to absorb, and absorbing it deletes it from the transcript.
                // Every word in the window has to contribute a sound.
                if words[start..start + window_len]
                    .iter()
                    .any(|word| phonetic_key(word).is_empty())
                {
                    continue;
                }
                if normalize(canonical) == spoken {
                    continue;
                }
                if is_ordinary_english(&spoken) {
                    continue;
                }
                // Dropping leading vowels collapses unrelated words (`offloaded` -> `FluidAudio`,
                // `overdid` -> `Verda`, `iphone` -> `Phonon`). Require leading vowel/consonant
                // parity between what was heard and the canonical term.
                if starts_with_vowel(&spoken) != starts_with_vowel(canonical) {
                    continue;
                }
                // Short all-caps acronyms (`MCP`, `SDK`) share 3-consonant skeletons with much
                // longer lowercase words (`memcopy` -> `MCP`). Only match them on spaced letters
                // or similarly short tokens, never on a single long word.
                if window_len == 1
                    && is_short_acronym(canonical)
                    && spoken.chars().count() >= canonical.chars().count() + 2
                {
                    continue;
                }
                // A word the recognizer capitalised inside (`iPhone`, `macOS`)
                // is a name it spelled on purpose, not a mishearing; the word
                // list is too old to know it. `iPhone` keys like `Phonon`.
                if written
                    .get(start..start + window_len)
                    .is_some_and(|window| window.iter().any(|word| capitalised_inside(word)))
                {
                    continue;
                }
                // A window longer than the taught form is either the recognizer
                // splitting the term into syllables, or the term reaching out and
                // swallowing the word beside it. Syllables are not words, so any
                // real English in an over-long window means the latter: `is not`
                // becoming `Sonnet`, or `my macbook` losing the `my`.
                let absorbing = form_words.iter().all(|count| window_len > *count);
                if absorbing
                    && words[start..start + window_len]
                        .iter()
                        .any(|word| is_real_word(word))
                {
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
    /// What the user changed after insertion; see `edits::UserEditRecord`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_edit: Option<crate::edits::UserEditRecord>,
    /// The app (and page host) the dictation went to, with its category.
    /// Kept only with local history on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_context: Option<crate::app_context::AppContext>,
    /// The text around the cursor at key-down, when the user turned
    /// text-box context on. Kept only with local history on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub textbox_context: Option<crate::app_context::TextboxContext>,
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
            user_edit: None,
            app_context: None,
            textbox_context: None,
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
    /// Bundle IDs of competing dictation apps the owner told the native app not
    /// to ask about again. Owned by the Swift bar; kept here so a rewrite of
    /// the file by the CLI does not drop it.
    #[serde(default)]
    pub competitor_quit_muted: Vec<String>,
    /// Put `profile/user.md` and `profile/vocab.md` in the correction prompt.
    #[serde(default = "default_profile_prefix")]
    pub profile_prefix: bool,
    /// Watch the field after insertion and propose terms the user corrected.
    #[serde(default = "default_true")]
    pub edit_learning: bool,
    /// Refine while the speaker talks (SPEC "Streaming refine"). Absent: on for
    /// Macs with 16 GB of memory or more, off below (it keeps the GPU busy about
    /// 28% of dictation time on an M4 Max and is unmeasured on 8 GB machines).
    /// `PHONON_STREAM_REFINE=0` or `=1` overrides both for one engine run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_refine: Option<bool>,
    /// Live text: show the corrected text so far above the bar while the key is held
    /// (each background refine replaces it). On by default since 1.0; needs streaming refine.
    /// The engine reads it at every `stream_start`, so a change applies to the next dictation.
    #[serde(default = "default_live_text")]
    pub live_text: bool,
    /// How live text streams in: "fade" (new words fade in, the uncorrected tail is
    /// dimmed) or "blur" (the tail is blurred and sharpens once corrected). Display only:
    /// the bar reads it; the engine keeps it on a rewrite.
    #[serde(default = "default_live_text_style")]
    pub live_text_style: String,
    /// Number pass (`phonon_llm::numpass`): after the refiner, spoken phone numbers, money
    /// amounts and word dates are written as digits. Off by default until the do-no-harm
    /// audit passes its frozen bar on the final rule; then the default flips to on.
    /// Read again when the file changes, so a change applies to the next dictation.
    #[serde(default)]
    pub number_pass: bool,
    /// Style preferences the user approved, by app category
    /// (`{"chat": ["lowercase_start"]}`).
    #[serde(default)]
    pub style_preferences: BTreeMap<String, Vec<String>>,
    /// Keys the native app owns that this side does not model. `phonon doctor`
    /// rewrites the file through this struct; before this field it dropped
    /// them (the sound cue, the first-run privacy answer, the retention days).
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
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
            competitor_quit_muted: Vec::new(),
            profile_prefix: default_profile_prefix(),
            edit_learning: true,
            stream_refine: None,
            live_text: default_live_text(),
            live_text_style: default_live_text_style(),
            number_pass: false,
            style_preferences: BTreeMap::new(),
            extra: BTreeMap::new(),
        }
    }
}

/// Off until the profile block earns it. Measured 2026-08-30 with
/// `scripts/eval_profile_prefix.py` (24 labeled cases, 80 corpus recordings):
/// no latency cost warm, but 21/24 exact against 22/24 without it, and on the
/// corpus its vocabulary wins (Phonon, DEVLOG, Blackwell) came with two
/// dropped sentences and one wrong term. Users can turn it on in settings.json.
pub const DEFAULT_PROFILE_PREFIX: bool = false;

fn default_profile_prefix() -> bool {
    DEFAULT_PROFILE_PREFIX
}

fn default_true() -> bool {
    true
}

fn default_microphone_priority() -> Vec<String> {
    Vec::new()
}

fn default_shortcut_mode() -> String {
    "fn".into()
}

/// Off: the owner found the text a lot to look at mid-dictation (2026-11-05); the capsule's
/// energy is the default feedback.
fn default_live_text() -> bool {
    false
}

fn default_live_text_style() -> String {
    "fade".into()
}

impl SettingsFile {
    /// Read without writing back. Missing file means defaults; the native app
    /// owns the file and this side must not rewrite it on every engine start.
    pub fn load() -> Result<Self> {
        Self::load_from(&settings_path()?)
    }

    /// Read a settings file at `path`; a missing file means defaults, an
    /// unreadable or malformed one is an error.
    pub fn load_from(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::parse(&bytes).with_context(|| format!("parse {}", path.display()))
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(bytes)?)
    }

    pub fn load_or_create() -> Result<Self> {
        let path = settings_path()?;
        if path.is_file() {
            let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            let mut settings: Self = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", path.display()))?;
            // The native app writes schema 2, where an absent privacy key
            // means "off"; stamping 1 over it would turn absent keys back on.
            settings.schema_version = settings.schema_version.max(DATA_SCHEMA_VERSION);
            write_json_atomic(&path, &settings)?;
            return Ok(settings);
        }
        let settings = Self::default();
        write_json_atomic(&path, &settings)?;
        Ok(settings)
    }
}

/// `PHONON_DATA_DIR` when set and non-empty (an isolated data root for tests, demos and
/// a second profile), else `~/Library/Application Support/Phonon`.
pub fn app_support_dir() -> Result<PathBuf> {
    app_support_dir_from(
        std::env::var_os("PHONON_DATA_DIR"),
        std::env::var_os("HOME"),
    )
}

fn app_support_dir_from(
    data_dir: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> Result<PathBuf> {
    if let Some(dir) = data_dir.filter(|dir| !dir.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    let home = home.context("HOME is not set")?;
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

/// `profile/user.md` and `profile/vocab.md` live here (SPEC, onboarding).
pub fn profile_dir() -> Result<PathBuf> {
    Ok(app_support_dir()?.join("profile"))
}

/// The user's own trained refiner, an MLX model directory. Used instead of the
/// pinned correction model when it holds `phonon_refiner.json`; its weights
/// come from the user's dictations, so they stay on this machine.
pub fn refiner_dir() -> Result<PathBuf> {
    Ok(app_support_dir()?.join("models").join("refiner"))
}

/// `dir` when it holds a refiner of the user's own (`phonon_refiner.json`).
/// It then replaces the pinned public download, and its own
/// `phonon_engine.json` keeps naming the engine it runs on; the public
/// default applies only when no such refiner is installed.
pub fn local_refiner(dir: &Path) -> Option<PathBuf> {
    dir.join("phonon_refiner.json")
        .is_file()
        .then(|| dir.to_path_buf())
}

/// The correction sidecar configuration users get: the profile when
/// `settings.json` enables it, the local refiner when one is installed, plus
/// developer environment overrides.
pub fn polish_config() -> Result<PolishConfig> {
    let settings = SettingsFile::load()?;
    let profile_dir = profile_dir()?;
    let refiner = refiner_dir()?;
    let config = PolishConfig {
        profile_dir: settings.profile_prefix.then(|| profile_dir.clone()),
        model_dir: local_refiner(&refiner),
        ..PolishConfig::default()
    };
    Ok(config.with_env_overrides(Some(&profile_dir)))
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

/// Screen terms as one dictation keeps them: trimmed, empties dropped, the first spelling
/// of each kept (case-insensitively), at most `MAX_SCREEN_TERMS`.
pub fn normalize_screen_terms<S: AsRef<str>>(terms: &[S]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for term in terms {
        let term = term.as_ref().trim();
        if term.is_empty() || out.iter().any(|seen| seen.eq_ignore_ascii_case(term)) {
            continue;
        }
        if out.len() >= MAX_SCREEN_TERMS {
            break;
        }
        out.push(term.to_string());
    }
    out
}

/// The envelope's `screen_confirmed_terms`: the dictionary's own first, unchanged, then
/// up to `MAX_SCREEN_TERMS` unfamiliar screen terms it does not already name.
fn merge_screen_terms(dictionary_terms: &[&str], extras: &[String]) -> Vec<String> {
    let mut out: Vec<String> = dictionary_terms.iter().map(|t| t.to_string()).collect();
    for term in normalize_screen_terms(extras) {
        if !out.iter().any(|seen| seen.eq_ignore_ascii_case(&term)) {
            out.push(term);
        }
    }
    out
}

/// Interface words that sit on nearly every screen and name nothing the user said.
const SCREEN_UI_LABELS: &[&str] = &[
    "Recent", "Sidebar", "Settings", "Search", "Home", "File", "Edit", "View", "Window", "Help",
    "New", "Open", "Save", "Close", "Share", "More", "Today", "INFO", "DEBUG", "ERROR", "WARN",
    "TODO", "FIXME", "NOTE", "TRACE",
];

/// Two-letter capitals that are ordinary words in a shouted heading, not acronyms.
const SCREEN_TWO_LETTER_WORDS: &[&str] = &[
    "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN", "IS", "IT", "ME", "MY", "NO",
    "OF", "OK", "ON", "OR", "PM", "SO", "TO", "UP", "US", "WE",
];

/// Suffixes that make a number a measurement, an ordinal or a time (`100ms`, `1080p`,
/// `4K`, `2nd`, `9am`), never a name.
const SCREEN_NUMBER_SUFFIXES: &[&str] = &[
    "ms", "s", "us", "ns", "sec", "secs", "min", "mins", "h", "hr", "hrs", "d", "w", "wk", "mo",
    "y", "yr", "yrs", "px", "pt", "em", "rem", "vh", "vw", "dpi", "ppi", "k", "m", "b", "g", "t",
    "p", "i", "x", "kb", "mb", "gb", "tb", "kib", "mib", "gib", "tib", "bit", "bits", "kbps",
    "mbps", "gbps", "hz", "khz", "mhz", "ghz", "fps", "mm", "cm", "km", "kg", "lb", "lbs", "ft",
    "mi", "mph", "kph", "v", "mv", "kw", "mah", "wh", "kwh", "st", "nd", "rd", "th", "am", "pm",
];

/// Unfamiliar terms in `screen_text` worth biasing a dictation toward: CamelCase and
/// mixed case (`cuBLAS`, `PyTorch`), 2-6 letter acronyms (`CUDA`), letter+digit
/// identifiers (`H100`), hyphenated or underscored identifiers (`gpt-4o`, `max_tokens`)
/// and capitalised names mid-sentence (`Infatoshi`). Interface labels, measurements,
/// ordinals, times, hashes and ordinary English are dropped. Most frequent first, ties
/// by first appearance; at most `max_terms`.
pub fn extract_screen_terms(screen_text: &str, max_terms: usize) -> Vec<String> {
    if max_terms == 0 {
        return Vec::new();
    }
    // (spelling first seen, count, position first seen), by lowercase key
    let mut found: Vec<(String, usize, usize)> = Vec::new();
    let mut by_key: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut position = 0_usize;
    for line in screen_text.lines() {
        let mut sentence_start = true;
        for chunk in line.split_whitespace() {
            let ends_sentence = chunk
                .trim_end_matches(['"', '\'', ')', ']', '\u{201d}', '\u{2019}'])
                .ends_with(['.', '!', '?', ':']);
            for token in chunk.split(|c: char| !(c.is_alphanumeric() || c == '-' || c == '_')) {
                let token = token.trim_matches(['-', '_']);
                if token.is_empty() {
                    continue;
                }
                if is_screen_term(token, sentence_start) {
                    match by_key.get(&token.to_lowercase()) {
                        Some(&slot) => found[slot].1 += 1,
                        None => {
                            by_key.insert(token.to_lowercase(), found.len());
                            found.push((token.to_string(), 1, position));
                        }
                    }
                }
                sentence_start = false;
                position += 1;
            }
            if ends_sentence {
                sentence_start = true;
            }
        }
    }
    found.sort_by_key(|(_, count, first)| (Reverse(*count), *first));
    found
        .into_iter()
        .take(max_terms)
        .map(|(term, _, _)| term)
        .collect()
}

fn is_screen_term(token: &str, sentence_start: bool) -> bool {
    let length = token.chars().count();
    if !(2..=64).contains(&length)
        || !token.chars().any(char::is_alphabetic)
        || SCREEN_UI_LABELS
            .iter()
            .any(|label| label.eq_ignore_ascii_case(token))
        || is_screen_number(token)
        || is_hex_hash(token)
    {
        return false;
    }
    let has_digit = token.chars().any(|c| c.is_ascii_digit());
    if token.contains('_') {
        return true;
    }
    if token.contains('-') {
        return has_digit
            || token
                .split('-')
                .filter(|part| !part.is_empty())
                .any(|part| {
                    capitalised_inside(part)
                        || (part.chars().count() >= 2 && part.chars().all(char::is_uppercase))
                        || !is_ordinary_english(part)
                });
    }
    if has_digit {
        return true;
    }
    if capitalised_inside(token) {
        return true;
    }
    if token.chars().all(char::is_uppercase) {
        return match length {
            2 => !SCREEN_TWO_LETTER_WORDS.contains(&token),
            3..=6 => !is_ordinary_english(token),
            _ => false,
        };
    }
    let mut chars = token.chars();
    let capitalised = chars.next().is_some_and(char::is_uppercase) && chars.all(char::is_lowercase);
    capitalised && !sentence_start && length >= 3 && !is_ordinary_english(token)
}

/// A number with a unit, ordinal or clock suffix (`100ms`, `1080p`, `2nd`, `9am`).
fn is_screen_number(token: &str) -> bool {
    if !token.starts_with(|c: char| c.is_ascii_digit()) {
        return false;
    }
    let suffix = token.trim_start_matches(|c: char| c.is_ascii_digit());
    suffix.chars().all(char::is_alphabetic)
        && SCREEN_NUMBER_SUFFIXES
            .iter()
            .any(|unit| unit.eq_ignore_ascii_case(suffix))
}

/// A commit or object hash (`a1b2c3d`, `0xdeadbeef`).
fn is_hex_hash(token: &str) -> bool {
    let digits = token
        .strip_prefix("0x")
        .or_else(|| token.strip_prefix("0X"));
    if let Some(digits) = digits {
        return !digits.is_empty() && digits.chars().all(|c| c.is_ascii_hexdigit());
    }
    token.len() >= 7
        && token.chars().all(|c| c.is_ascii_hexdigit())
        && token.chars().any(|c| c.is_ascii_digit())
        && token.chars().any(|c| c.is_ascii_alphabetic())
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

pub(crate) fn normalize(value: &str) -> String {
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

/// `normalize`'s words with their case kept, one for one.
fn written_words(value: &str) -> Vec<String> {
    value
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_string)
        .collect()
}

/// A lowercase letter followed later by a capital: `iPhone`, `GitHub`.
fn capitalised_inside(word: &str) -> bool {
    word.chars()
        .skip_while(|character| !character.is_lowercase())
        .any(char::is_uppercase)
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

/// Webster's Second word list shipped with the app (`assets/english_words.txt`).
/// Loaded once; an install missing the file falls back to no phonetic guessing,
/// which is the safe direction — a missed term repair beats a corrupted word.
fn english_words() -> &'static std::collections::HashSet<String> {
    ENGLISH_WORDS.get_or_init(|| {
        let path = crate::paths::project_root().join("assets/english_words.txt");
        fs::read_to_string(path)
            .map(|text| word_set(&text))
            .unwrap_or_default()
    })
}

static ENGLISH_WORDS: std::sync::OnceLock<std::collections::HashSet<String>> =
    std::sync::OnceLock::new();

fn word_set(text: &str) -> std::collections::HashSet<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

/// Supply the word list (`assets/english_words.txt`'s text) from somewhere other than
/// the project root: the Windows and Linux app carries it inside its executable. Only
/// the first supply, before any lookup, counts; false when the list was already set.
/// The macOS app never calls this and reads the file as before.
pub fn supply_english_words(text: &str) -> bool {
    let mut supplied = false;
    ENGLISH_WORDS.get_or_init(|| {
        supplied = true;
        word_set(text)
    });
    supplied
}

/// First names from the system's word lists (`/usr/share/dict/propernames`
/// on macOS, about 1,300 of them); empty where the file is missing. Read in
/// place, never copied into Phonon.
pub(crate) fn proper_names() -> &'static std::collections::HashSet<String> {
    static NAMES: std::sync::OnceLock<std::collections::HashSet<String>> =
        std::sync::OnceLock::new();
    NAMES.get_or_init(|| {
        fs::read_to_string("/usr/share/dict/propernames")
            .map(|text| {
                text.lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// A first name as written (`Grant`, `Frank`).
pub(crate) fn is_proper_name(word: &str) -> bool {
    proper_names().contains(word)
}

/// Whether every word of `phrase` is ordinary English, allowing for the common
/// inflections the base word list omits.
pub(crate) fn is_ordinary_english(phrase: &str) -> bool {
    let words = english_words();
    if words.is_empty() {
        // No list available: treat everything as ordinary so nothing is guessed.
        return true;
    }
    let mut any = false;
    for word in phrase.split_whitespace() {
        any = true;
        if !is_ordinary_english_word(words, word) {
            return false;
        }
    }
    any
}

/// Whether a word is real enough that a term must not be allowed to swallow it.
/// Deliberately a plain lookup: the inflection guesses below exist to decide
/// whether a word may be *rewritten*, and they are far too loose to decide
/// whether one may be *deleted* -- they read `bloss` as English by way of `blo`.
fn is_real_word(word: &str) -> bool {
    if word.chars().count() <= 2 {
        return true;
    }
    let words = english_words();
    // With no list to consult, assume every word is real so nothing is dropped.
    words.is_empty() || words.contains(&word.to_lowercase())
}

fn is_ordinary_english_word(words: &std::collections::HashSet<String>, word: &str) -> bool {
    let word = word.to_lowercase();
    if is_ordinary_english_stem(words, &word) {
        return true;
    }
    // Common English prefixes omitted from the 1934 word list (`offload`, `offloaded`,
    // `overdid`, `preloaded`, `autosave`).
    for prefix in [
        "counter", "inter", "super", "multi", "under", "cross", "over", "auto", "anti", "semi",
        "down", "back", "fore", "self", "off", "out", "pre", "non", "sub", "mis", "dis", "mid",
    ] {
        if let Some(rest) = word.strip_prefix(prefix) {
            if rest.len() >= 3 && is_ordinary_english_stem(words, rest) {
                return true;
            }
        }
    }
    for prefix in ["un", "re", "de", "up", "co"] {
        if let Some(rest) = word.strip_prefix(prefix) {
            if rest.len() >= 4 && is_ordinary_english_stem(words, rest) {
                return true;
            }
        }
    }
    false
}

fn is_ordinary_english_stem(words: &std::collections::HashSet<String>, word: &str) -> bool {
    if word.chars().count() <= 2 {
        // The list leaves out `a`, `i`, `at` and their kin, which would
        // otherwise read as exotic and license a guess. A token this short
        // carries too little sound to overrule what the recognizer heard.
        return true;
    }
    if words.contains(word) {
        return true;
    }
    for suffix in ["s", "es", "ed", "d", "ing", "ly", "er", "ers", "est"] {
        if let Some(stem) = word.strip_suffix(suffix) {
            if stem.len() < 3 {
                continue;
            }
            if words.contains(stem) {
                return true;
            }
            // running -> run, stopped -> stop
            if let Some(shortened) = stem.strip_suffix(stem.chars().last().unwrap_or(' ')) {
                if shortened.len() >= 3 && words.contains(shortened) {
                    return true;
                }
            }
            // making -> make, filed -> file
            if words.contains(&format!("{stem}e")) {
                return true;
            }
            // carries -> carry
            if let Some(without_i) = stem.strip_suffix('i') {
                if without_i.len() >= 3 && words.contains(&format!("{without_i}y")) {
                    return true;
                }
            }
        }
    }
    false
}

pub(crate) fn starts_with_vowel(value: &str) -> bool {
    value
        .chars()
        .find(|character| character.is_alphanumeric())
        .is_some_and(|character| {
            matches!(character.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u')
        })
}

fn is_short_acronym(value: &str) -> bool {
    let trimmed = value.trim();
    let len = trimmed.chars().count();
    (2..=4).contains(&len)
        && trimmed
            .chars()
            .all(|character| character.is_ascii_uppercase())
}

/// Whether two canonical spellings are the same term written differently, which
/// spacing and capitalization alone do not distinguish.
fn same_term(left: &str, right: &str) -> bool {
    let squash = |value: &str| {
        value
            .chars()
            .filter(|character| character.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    squash(left) == squash(right)
}

fn capital_count(value: &str) -> usize {
    value
        .chars()
        .filter(|character| character.is_uppercase())
        .count()
}

/// Longest run of transcript words compared against one dictionary form. Three
/// covers the terms a recognizer actually splits apart (`Claude Agent SDK`)
/// without letting a match wander across a clause.
const MAX_PHONETIC_WINDOW: usize = 3;

pub(crate) fn phonetic_key(value: &str) -> String {
    let mut output = String::new();
    // A doubled letter inside a word is silent, so `bloss` reduces to `bls`. A
    // repeat across a word boundary is not: collapsing those made `gpt 5` and
    // `GPT-5.5` indistinguishable, and turned the model version into the wrong
    // one. Restart the run at every boundary.
    for word in normalize(value).split_whitespace() {
        let mut previous = None;
        for character in word.chars().filter(|character| character.is_alphanumeric()) {
            let character = match character {
                'c' | 'q' => 'k',
                other => other,
            };
            if matches!(character, 'a' | 'e' | 'i' | 'o' | 'u' | 'y') {
                continue;
            }
            if previous != Some(character) {
                output.push(character);
            }
            previous = Some(character);
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

pub(crate) fn levenshtein(left: &str, right: &str) -> usize {
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

pub(crate) fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
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
mod local_refiner_tests {
    use super::local_refiner;

    #[test]
    fn an_installed_user_refiner_is_used_and_keeps_its_own_engine() {
        let dir = std::env::temp_dir().join(format!("phonon-local-refiner-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // No marker: the pinned public refiner applies.
        assert_eq!(local_refiner(&dir), None);
        // An engine config alone is not a refiner.
        std::fs::write(
            dir.join("phonon_engine.json"),
            r#"{"binary": "engine-ship3/bin/pe", "target": "t.gguf"}"#,
        )
        .unwrap();
        assert_eq!(local_refiner(&dir), None);
        // The owner's setup: marker plus its own engine config. The directory is
        // passed to the sidecar as it is, so its engine binary is the one used.
        std::fs::write(dir.join("phonon_refiner.json"), "{}").unwrap();
        assert_eq!(local_refiner(&dir), Some(dir.clone()));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        app_support_dir_from, extract_screen_terms, extract_transcript_payload,
        normalize_screen_terms, safe_polish_output, AsrTermEntry, DictionaryEntry, DictionaryFile,
        SettingsFile,
    };

    #[test]
    fn phonon_data_dir_overrides_the_application_support_folder() {
        assert_eq!(
            app_support_dir_from(Some("/tmp/phonon-iso".into()), Some("/Users/x".into())).unwrap(),
            std::path::PathBuf::from("/tmp/phonon-iso")
        );
        assert_eq!(
            app_support_dir_from(Some("".into()), Some("/Users/x".into())).unwrap(),
            std::path::PathBuf::from("/Users/x/Library/Application Support/Phonon")
        );
        assert_eq!(
            app_support_dir_from(None, Some("/Users/x".into())).unwrap(),
            std::path::PathBuf::from("/Users/x/Library/Application Support/Phonon")
        );
        assert!(app_support_dir_from(None, None).is_err());
    }

    #[test]
    fn asr_term_entries_carry_spoken_forms_and_replaced_phrases() {
        let mut phonon = entry("Phonon", None);
        phonon.spoken_forms = vec!["phone on".into(), "Phonon".into(), "PHONE ON".into()];
        let dictionary = DictionaryFile {
            entries: vec![
                phonon,
                entry("koo bloss", Some("cuBLAS")),
                entry("cue blas", Some("cuBLAS")),
                entry("table", None),
                entry("cublas", Some("cuBLAS")),
            ],
            ..Default::default()
        };
        let entries = dictionary.asr_term_entries();
        assert_eq!(
            entries,
            vec![
                AsrTermEntry {
                    term: "Phonon".into(),
                    spoken_forms: vec!["phone on".into()],
                },
                AsrTermEntry {
                    term: "cuBLAS".into(),
                    spoken_forms: vec!["koo bloss".into(), "cue blas".into()],
                },
            ]
        );
        let terms: Vec<String> = entries.iter().map(|e| e.term.clone()).collect();
        assert_eq!(terms, dictionary.asr_terms());
        let wire = serde_json::to_value(&entries).unwrap();
        assert_eq!(
            wire,
            serde_json::json!([
                {"term":"Phonon","spoken_forms":["phone on"]},
                {"term":"cuBLAS","spoken_forms":["koo bloss","cue blas"]}
            ])
        );
        let bare = serde_json::to_value(AsrTermEntry {
            term: "CUDA".into(),
            spoken_forms: Vec::new(),
        })
        .unwrap();
        assert_eq!(bare, serde_json::json!({"term":"CUDA"}));
    }

    #[test]
    fn screen_terms_keep_identifiers_and_names_and_drop_noise() {
        let screen = "File Edit View Window Help\n\
            Recent Settings Search\n\
            INFO loaded cuBLAS in 100ms at 1080p on the H100, commit a1b2c3d4e5\n\
            The PyTorch build uses CUDA and max_tokens with gpt-4o on the 2nd try at 9am.\n\
            Ask Infatoshi about cuBLAS. Today the well-known GPU is fine.\n\
            TODO: OK THE IT";
        let terms = extract_screen_terms(screen, 24);
        for kept in [
            "cuBLAS",
            "H100",
            "PyTorch",
            "CUDA",
            "max_tokens",
            "gpt-4o",
            "Infatoshi",
            "GPU",
        ] {
            assert!(
                terms.contains(&kept.to_string()),
                "{kept} missing: {terms:?}"
            );
        }
        for dropped in [
            "File",
            "Edit",
            "Recent",
            "Settings",
            "INFO",
            "100ms",
            "1080p",
            "a1b2c3d4e5",
            "2nd",
            "9am",
            "Today",
            "The",
            "Ask",
            "well-known",
            "TODO",
            "OK",
            "THE",
            "IT",
            "loaded",
        ] {
            assert!(
                !terms.contains(&dropped.to_string()),
                "{dropped} kept: {terms:?}"
            );
        }
        // most frequent first, then first appearance
        assert_eq!(terms[0], "cuBLAS");
        assert_eq!(terms[1], "H100");
        assert_eq!(extract_screen_terms(screen, 2), vec!["cuBLAS", "H100"]);
        assert!(extract_screen_terms(screen, 0).is_empty());
        assert!(extract_screen_terms("", 10).is_empty());
    }

    #[test]
    fn screen_terms_are_normalized_and_capped() {
        let many: Vec<String> = (0..40).map(|i| format!("Term{i}")).collect();
        assert_eq!(normalize_screen_terms(&many).len(), 24);
        assert_eq!(
            normalize_screen_terms(&[" cuBLAS ", "CUBLAS", "", "H100"]),
            vec!["cuBLAS", "H100"]
        );
    }

    #[test]
    fn unfamiliar_screen_terms_reach_the_envelope_even_without_a_dictionary() {
        let empty = DictionaryFile::default();
        assert_eq!(
            empty.prepare_polish_input_with_screen_terms("use koo bloss", "cuBLAS docs", &[]),
            "use koo bloss"
        );
        let input = empty.prepare_polish_input_with_screen_terms(
            "use koo bloss on the H one hundred",
            "cuBLAS docs H100",
            &["cuBLAS".into(), "H100".into(), "cublas".into()],
        );
        assert!(input.starts_with("<phonon_dictionary>\n"), "{input}");
        assert!(
            input.contains("screen_confirmed_terms: cuBLAS, H100\n"),
            "{input}"
        );
        assert!(input.contains("<transcript>\nuse koo bloss on the H one hundred\n</transcript>"));

        let dictionary = DictionaryFile {
            entries: vec![entry("cuBLAS", None)],
            ..Default::default()
        };
        let plain = dictionary
            .prepare_polish_input_with_context("Use Kubloss here", "CUDA Toolkit cuBLAS docs");
        assert_eq!(
            plain,
            dictionary.prepare_polish_input_with_screen_terms(
                "Use Kubloss here",
                "CUDA Toolkit cuBLAS docs",
                &[]
            )
        );
        let merged = dictionary.prepare_polish_input_with_screen_terms(
            "Use Kubloss here",
            "CUDA Toolkit cuBLAS docs",
            &["CUBLAS".into(), "Toolkit".into(), "CUDA".into()],
        );
        assert!(
            merged.contains("screen_confirmed_terms: cuBLAS, Toolkit, CUDA\n"),
            "{merged}"
        );
    }

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

    /// The phones' dictionary ports (Android Dictionary.kt, iPhone Dictionary.swift) are checked
    /// against `dictionary_cases.json`; this keeps its Rust-made fields equal to what this file
    /// does. PHONON_WRITE_DICTIONARY_CASES=1 rewrites them instead of comparing.
    #[test]
    fn dictionary_cases_match() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/dictionary_cases.json");
        let mut cases: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let write = std::env::var("PHONON_WRITE_DICTIONARY_CASES").as_deref() == Ok("1");
        let dictionary_of = |case: &serde_json::Value| DictionaryFile {
            entries: serde_json::from_value(case["entries"].clone()).unwrap(),
            ..Default::default()
        };
        let mut checked = 0;
        for case in cases["envelope"].as_array_mut().unwrap() {
            let dictionary = dictionary_of(case);
            let envelope = dictionary.prepare_polish_input(case["transcript"].as_str().unwrap());
            if write {
                case["envelope"] = serde_json::Value::String(envelope);
            } else {
                assert_eq!(
                    case["envelope"].as_str(),
                    Some(envelope.as_str()),
                    "{}",
                    case["id"]
                );
            }
            checked += 1;
        }
        for case in cases["exact"].as_array_mut().unwrap() {
            let dictionary = dictionary_of(case);
            let text = dictionary
                .apply_exact_replacements(case["text"].as_str().unwrap())
                .text;
            if write {
                case["expected"] = serde_json::Value::String(text);
            } else {
                assert_eq!(
                    case["expected"].as_str(),
                    Some(text.as_str()),
                    "{}",
                    case["id"]
                );
            }
            checked += 1;
        }
        assert!(checked >= 20, "only {checked} cases");
        if write {
            let text = serde_json::to_string_pretty(&cases).unwrap() + "\n";
            std::fs::write(&path, text).unwrap();
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
    fn phonetic_matching_reaches_multi_word_terms() {
        let dictionary = DictionaryFile {
            entries: vec![entry("Claude Code", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("ask clod kode to look at it");
        assert!(
            prepared.contains("clod kode => Claude Code"),
            "multi-word term was not reachable: {prepared}"
        );
        assert!(prepared.contains("ask Claude Code to look at it"));
    }

    #[test]
    fn phonetic_matching_leaves_a_word_capitalised_inside_alone() {
        // `iphone` and `Phonon` share the key `phn`, and the 1934 word list has
        // no `iPhone`: the owner's iPhone became Phonon in 13 dictations.
        let mut phonon = entry("Phonon", None);
        phonon.spoken_forms = vec!["phone on".into()];
        let dictionary = DictionaryFile {
            entries: vec![phonon],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("I left my iPhone on the desk");
        assert!(
            prepared.contains("I left my iPhone on the desk") && !prepared.contains("=> Phonon"),
            "iPhone was rewritten: {prepared}"
        );
        // The recognizer's lowercase mishearing of the term is still repaired.
        let prepared = dictionary.prepare_polish_input("open the phonnon settings");
        assert!(prepared.contains("open the Phonon settings"), "{prepared}");
    }

    #[test]
    fn phonetic_matching_does_not_swallow_a_neighbouring_word() {
        // Vowels are dropped, so `koo bloss is` keys the same as `koo bloss`.
        // Taking the longer window would delete `is` before the model sees it.
        let dictionary = DictionaryFile {
            entries: vec![entry("cuBLAS", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("koo bloss is slow");
        assert!(
            prepared.contains("cuBLAS is slow"),
            "a word was absorbed into the replacement: {prepared}"
        );
    }

    #[test]
    fn phonetic_matching_does_not_absorb_a_soundless_word() {
        let dictionary = DictionaryFile {
            entries: vec![entry("Speculative Decoding", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("try a speculativ decoding pass");
        assert!(
            prepared.contains("try a Speculative Decoding pass"),
            "the article was absorbed into the term: {prepared}"
        );
    }

    #[test]
    fn phonetic_matching_keeps_version_numbers_distinct() {
        // Collapsing the repeat across the boundary made `gpt 5` key the same as
        // `GPT-5.5`, quietly promoting one model version to another.
        let dictionary = DictionaryFile {
            entries: vec![entry("GPT-5.5", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("gpt 5 is out now");
        assert!(
            prepared.contains("gpt 5 is out now"),
            "one version was rewritten as another: {prepared}"
        );
    }

    #[test]
    fn phonetic_matching_does_not_swallow_a_real_word_beside_a_term() {
        // `my macbook` keys the same as `macbook`, so an unguarded match takes
        // the `my` with it and the word is gone before the model sees it.
        let dictionary = DictionaryFile {
            entries: vec![entry("MacBook", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("on my macbuk today");
        assert!(
            prepared.contains("on my"),
            "the neighbouring word was absorbed: {prepared}"
        );
    }

    #[test]
    fn phonetic_matching_treats_short_function_words_as_ordinary() {
        // `at` is missing from the word list, so without a length rule the
        // phrase reads as exotic and `even at` gets rewritten to `uv init`.
        let dictionary = DictionaryFile {
            entries: vec![entry("uv init", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("it is worse even at batch one");
        assert!(
            prepared.contains("worse even at batch one"),
            "ordinary speech was rewritten as a term: {prepared}"
        );
    }

    #[test]
    fn phonetic_matching_leaves_ordinary_english_alone() {
        let dictionary = DictionaryFile {
            entries: vec![entry("Pallas", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("please stop there");
        assert!(
            prepared.contains("please stop there"),
            "an ordinary word was guessed at: {prepared}"
        );
        assert!(!prepared.contains("=> Pallas"));
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
        assert_eq!(settings.shortcut_mode, "fn");
        assert_eq!(settings.profile_prefix, super::DEFAULT_PROFILE_PREFIX);
        assert!(settings.competitor_quit_muted.is_empty());
    }

    #[test]
    fn settings_round_trip_keeps_competitor_mutes() {
        let settings: SettingsFile = serde_json::from_str(
            r#"{"schema_version":2,"competitor_quit_muted":["com.electron.wispr-flow"]}"#,
        )
        .unwrap();
        assert_eq!(settings.competitor_quit_muted, ["com.electron.wispr-flow"]);
        let json = serde_json::to_string(&settings).unwrap();
        assert!(json.contains(r#""competitor_quit_muted":["com.electron.wispr-flow"]"#));
    }

    #[test]
    fn live_text_defaults_off_and_round_trips() {
        let absent: SettingsFile = serde_json::from_str(r#"{"schema_version":2}"#).unwrap();
        assert!(!absent.live_text);
        assert!(!SettingsFile::default().live_text);
        let on: SettingsFile =
            serde_json::from_str(r#"{"schema_version":2,"live_text":true}"#).unwrap();
        assert!(on.live_text);
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&on).unwrap()).unwrap();
        assert_eq!(json["live_text"], true);
        assert!(!json.as_object().unwrap().contains_key("extra"));
        let back: SettingsFile = serde_json::from_value(json).unwrap();
        assert_eq!(back, on);
    }

    #[test]
    fn live_text_style_defaults_to_fade_and_round_trips() {
        let absent: SettingsFile = serde_json::from_str(r#"{"schema_version":2}"#).unwrap();
        assert_eq!(absent.live_text_style, "fade");
        assert_eq!(SettingsFile::default().live_text_style, "fade");
        let blur: SettingsFile =
            serde_json::from_str(r#"{"schema_version":2,"live_text_style":"blur"}"#).unwrap();
        assert_eq!(blur.live_text_style, "blur");
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&blur).unwrap()).unwrap();
        assert_eq!(json["live_text_style"], "blur");
        assert!(blur.extra.is_empty(), "a typed field, not an unknown key");
        let back: SettingsFile = serde_json::from_value(json).unwrap();
        assert_eq!(back, blur);
    }

    #[test]
    fn settings_round_trip_keeps_keys_only_the_native_app_knows() {
        let settings: SettingsFile = serde_json::from_str(
            r#"{"schema_version":2,"sound_feedback":true,"privacy_choice_made":true,"history_retention_days":30,"style_preferences":{"chat":["lowercase_start"]}}"#,
        )
        .unwrap();
        assert!(settings.edit_learning);
        assert!(!settings.number_pass); // absent: off until the audit passes
        let on: SettingsFile =
            serde_json::from_str(r#"{"schema_version":2,"number_pass":true}"#).unwrap();
        assert!(on.number_pass);
        assert!(on.extra.is_empty(), "a typed field, not an unknown key");
        assert_eq!(settings.stream_refine, None); // absent: the engine decides by memory
        let json = serde_json::to_string(&settings).unwrap();
        assert!(!json.contains("stream_refine")); // and a rewrite does not pin it
        let off: SettingsFile =
            serde_json::from_str(r#"{"schema_version":2,"stream_refine":false}"#).unwrap();
        assert_eq!(off.stream_refine, Some(false));
        assert_eq!(settings.style_preferences["chat"], ["lowercase_start"]);
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&settings).unwrap()).unwrap();
        assert_eq!(json["sound_feedback"], true);
        assert_eq!(json["privacy_choice_made"], true);
        assert_eq!(json["history_retention_days"], 30);
        assert_eq!(json["schema_version"], 2);
        assert_eq!(json["style_preferences"]["chat"][0], "lowercase_start");
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

    #[test]
    fn phonetic_matching_protects_prefixed_english_leading_vowels_and_acronyms() {
        let dictionary = DictionaryFile {
            entries: vec![
                entry("FluidAudio", None),
                entry("Verda", None),
                entry("MCP", None),
                entry("Phonon", None),
            ],
            ..Default::default()
        };
        let transcript =
            "I offloaded the job, overdid the memcopy, and checked my iphone after offload.";
        let prepared = dictionary.prepare_polish_input(transcript);
        assert!(
            prepared.contains(transcript),
            "ordinary or unrelated words were corrupted: {prepared}"
        );
        assert!(!prepared.contains("=> FluidAudio"), "{prepared}");
        assert!(!prepared.contains("=> Verda"), "{prepared}");
        assert!(!prepared.contains("=> MCP"), "{prepared}");
        assert!(!prepared.contains("=> Phonon"), "{prepared}");
    }

    #[test]
    fn asr_term_entries_prioritize_starred_and_usage_count() {
        let mut starred = entry("ZetaTerm", None);
        starred.starred = true;
        let mut popular = entry("Tetra", None);
        popular.usage_count = 76;
        let plain = entry("AlphaTerm", None);
        let dictionary = DictionaryFile {
            entries: vec![plain, popular, starred],
            ..Default::default()
        };
        let terms = dictionary.asr_terms();
        assert_eq!(terms, vec!["ZetaTerm", "Tetra", "AlphaTerm"]);
        let entries = dictionary.asr_term_entries();
        assert_eq!(
            entries.iter().map(|e| e.term.as_str()).collect::<Vec<_>>(),
            vec!["ZetaTerm", "Tetra", "AlphaTerm"]
        );
    }
}
