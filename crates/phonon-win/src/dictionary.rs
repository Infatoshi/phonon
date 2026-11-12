//! The personal dictionary: terms and replacements the user taught.
//!
//! The file is the Mac's `dictionary.json` in the data root, the same format
//! (`phonon_core::data::DictionaryFile`), so a file copied between the two works.
//! The matching itself is the Mac's code, not a copy: the exact replacements,
//! the phonetic pre-correction and the `<phonon_dictionary>` terms all come from
//! `DictionaryFile::prepare_polish_input`.
//!
//! What the refiner sees is what the Mac sends this same refiner
//! (`sidecar/polish_server.py`, a model directory with `phonon_refiner.json`):
//! not the envelope, but the pre-corrected transcript as the user turn and the
//! envelope's terms as one "Vocabulary the speaker may use:" line at the end of
//! the system prompt, the line every training row carried. After the refiner,
//! `restore_terms` puts back a one-word term it swapped for another, and the exact
//! replacements are applied to its answer (`safe_polish_output`'s dictionary half).
//!
//! Not here: the Mac also boosts the terms inside its speech recogniser; the
//! sherpa-onnx recogniser Windows and Linux run has no such input.
//!
//! An empty or absent dictionary changes nothing: `prepare` is `None`, and the
//! correction prompt and its answer stay byte-identical to a build without this
//! module.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;
pub use phonon_core::data::{DictionaryEntry, DictionaryFile};

/// The file name in the data root, as on the Mac.
pub const FILE: &str = "dictionary.json";

/// The Mac's `source` for an entry the user typed in (bar/Sources/AppData.swift).
const SOURCE: &str = "phonon-app";

/// The Webster's word list the Mac ships (`assets/english_words.txt`). Phonetic
/// pre-correction never rewrites a word on it; without it there is no phonetic
/// guessing at all. Read only when a dictation meets a non-empty dictionary. A
/// `static`, not a `const`: each use of a const may carry its own 1.8 MB copy.
static ENGLISH_WORDS: &str = include_str!("../../../assets/english_words.txt");

/// `polish_server.py` `REFINER_VOCAB_PREFIX`, after the writing-mode line.
pub const VOCAB_PREFIX: &str = "Vocabulary the speaker may use: ";

/// The dictionary file in the data root.
pub fn path() -> PathBuf {
    crate::paths::data_root().join(FILE)
}

/// The saved dictionary; empty when there is none or it cannot be read.
pub fn load() -> DictionaryFile {
    load_at(&path())
}

/// Save atomically (a temporary file renamed over the old one), sorted and stamped
/// as the Mac saves it. (The settings window saves with `save_at` into the data
/// root it was opened on.)
#[cfg_attr(windows, allow(dead_code))]
pub fn save(dictionary: &mut DictionaryFile) -> Result<()> {
    save_at(dictionary, &path())
}

/// Add a term to spell as written (`Kubernetes`). False when it is empty or
/// already there.
pub fn add_word(dictionary: &mut DictionaryFile, phrase: &str) -> bool {
    add(dictionary, phrase, None)
}

/// Add a replacement: `phrase` heard, `replacement` written (`kates` => `k8s`).
/// False when either is empty, they are the same, or it is already there.
pub fn add_replacement(dictionary: &mut DictionaryFile, phrase: &str, replacement: &str) -> bool {
    let replacement = replacement.trim();
    if replacement.is_empty() || replacement == phrase.trim() {
        return false;
    }
    add(dictionary, phrase, Some(replacement))
}

/// Remove the entry for `phrase` (and `replacement`, `None` for a plain term),
/// matched without regard to case. False when there was none.
pub fn remove(dictionary: &mut DictionaryFile, phrase: &str, replacement: Option<&str>) -> bool {
    let key = entry_key(phrase, replacement);
    let before = dictionary.entries.len();
    dictionary
        .entries
        .retain(|entry| entry_key(&entry.phrase, entry.replacement.as_deref()) != key);
    dictionary.entries.len() != before
}

fn add(dictionary: &mut DictionaryFile, phrase: &str, replacement: Option<&str>) -> bool {
    if phrase.trim().is_empty() {
        return false;
    }
    dictionary.merge([DictionaryEntry {
        phrase: phrase.to_string(),
        replacement: replacement.map(str::to_string),
        spoken_forms: Vec::new(),
        source: SOURCE.to_string(),
        starred: false,
        usage_count: 0,
    }]) > 0
}

/// `phonon_core::data`'s merge key: the trimmed phrase and the replacement, lowercase.
fn entry_key(phrase: &str, replacement: Option<&str>) -> (String, Option<String>) {
    (
        phrase.trim().to_lowercase(),
        replacement.map(|value| value.trim().to_lowercase()),
    )
}

/// Missing file: empty. A file that is not a dictionary is moved aside to
/// `dictionary.json.corrupt-<unix ms>` (kept, never deleted) and the dictionary is
/// empty, so one bad edit cannot stop dictation.
pub fn load_at(path: &Path) -> DictionaryFile {
    match DictionaryFile::load_from(path) {
        Ok(dictionary) => dictionary,
        Err(error) => {
            let bytes_were_read = std::fs::read(path).is_ok();
            if bytes_were_read {
                let mut aside = path.as_os_str().to_owned();
                aside.push(format!(".corrupt-{}", unix_ms()));
                match std::fs::rename(path, &aside) {
                    Ok(()) => eprintln!(
                        "phonon: dictionary unreadable, moved aside to {}: {error:#}",
                        Path::new(&aside).display()
                    ),
                    Err(rename) => {
                        eprintln!("phonon: dictionary unreadable ({error:#}); not moved: {rename}")
                    }
                }
            } else {
                eprintln!("phonon: dictionary not read: {error:#}");
            }
            DictionaryFile::default()
        }
    }
}

pub fn save_at(dictionary: &mut DictionaryFile, path: &Path) -> Result<()> {
    dictionary.save_to(path)
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis())
        .unwrap_or(0)
}

/// The dictionary the dictation path uses: read again when the file's
/// modification time or size changed, so an edit (the settings window's, or the
/// file's by hand) applies from the next dictation without a restart.
pub struct Live {
    path: PathBuf,
    stamp: Option<Option<(Option<SystemTime>, u64)>>,
    dictionary: DictionaryFile,
}

impl Live {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            stamp: None,
            dictionary: DictionaryFile::default(),
        }
    }

    fn file_stamp(&self) -> Option<(Option<SystemTime>, u64)> {
        std::fs::metadata(&self.path)
            .ok()
            .map(|meta| (meta.modified().ok(), meta.len()))
    }

    /// The dictionary as of now.
    pub fn current(&mut self) -> &DictionaryFile {
        let stamp = self.file_stamp();
        if self.stamp != Some(stamp) {
            self.dictionary = load_at(&self.path);
            // a corrupt file was moved aside: stamp what is there now
            self.stamp = Some(self.file_stamp());
        }
        &self.dictionary
    }
}

/// What the refiner gets for one transcript with a dictionary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prepared {
    /// The transcript after the exact replacements and phonetic pre-correction.
    pub input: String,
    /// The terms for the vocabulary line, most confident first.
    pub vocabulary: Vec<String>,
}

/// Build the word list on a background thread when a dictionary file exists, so the
/// first dictation does not pay for it (about 130 ms in a debug build). Without a
/// file nothing is loaded.
pub fn warm() {
    if path().is_file() {
        std::thread::spawn(|| phonon_core::data::supply_english_words(ENGLISH_WORDS));
    }
}

/// `None` for an empty dictionary: the transcript goes to the refiner as before.
pub fn prepare(dictionary: &DictionaryFile, raw: &str) -> Option<Prepared> {
    if dictionary.entries.is_empty() {
        return None;
    }
    phonon_core::data::supply_english_words(ENGLISH_WORDS);
    let envelope = dictionary.prepare_polish_input(raw);
    Some(Prepared {
        input: transcript_payload(&envelope).to_string(),
        vocabulary: refiner_vocabulary(&envelope),
    })
}

/// The refiner's answer with the dictionary's exact replacements applied.
pub fn apply_replacements(dictionary: &DictionaryFile, text: &str) -> String {
    dictionary.apply_exact_replacements(text).text
}

/// `polish_server.py` `transcript_payload`.
fn transcript_payload(text: &str) -> &str {
    if let Some(start) = text.find("<transcript>") {
        let rest = &text[start + "<transcript>".len()..];
        if let Some(end) = rest.find("</transcript>") {
            return rest[..end].trim();
        }
    }
    text.trim()
}

/// Envelope lines that become vocabulary, most confident first; pair lines
/// (`spoken => canonical`, joined by `;`) give their canonical side.
const VOCAB_LINES: [(&str, bool); 5] = [
    ("screen_confirmed_terms", false),
    ("phonetic_matches", true),
    ("exact_replacements", true),
    ("likely_terms", false),
    ("canonical_terms", false),
];

/// `polish_server.py` `refiner_vocabulary`: the envelope's distinct terms, capped
/// at min(80, max(30, 3 x transcript words)).
fn refiner_vocabulary(envelope: &str) -> Vec<String> {
    let Some(start) = envelope.find("<phonon_dictionary>") else {
        return Vec::new();
    };
    let body = &envelope[start + "<phonon_dictionary>".len()..];
    let Some(end) = body.find("</phonon_dictionary>") else {
        return Vec::new();
    };
    let mut fields = std::collections::HashMap::new();
    for line in body[..end].lines() {
        if let Some((key, value)) = line.split_once(':') {
            fields.insert(key.trim(), value.trim());
        }
    }
    let mut terms = Vec::new();
    for (key, pairs) in VOCAB_LINES {
        let value = fields.get(key).copied().unwrap_or("");
        if pairs {
            terms.extend(
                value
                    .split(';')
                    .map(|pair| pair.split_once("=>").map_or("", |(_, right)| right)),
            );
        } else {
            terms.extend(value.split(','));
        }
    }
    let mut seen = std::collections::HashSet::new();
    let mut vocabulary: Vec<String> = terms
        .into_iter()
        .map(str::trim)
        .filter(|term| !term.is_empty() && seen.insert(*term))
        .map(str::to_string)
        .collect();
    let words = transcript_payload(envelope).split_whitespace().count();
    vocabulary.truncate(80.min(30.max(3 * words)));
    vocabulary
}

/// `polish_server.py` `_norm`: lowercase, letters and digits only.
fn norm(word: &str) -> String {
    word.chars()
        .flat_map(char::to_lowercase)
        .filter(|character| character.is_alphanumeric())
        .collect()
}

/// `polish_server.py` `TERM_SHAPE`: two capitals at the ends, a digit, or an inner hyphen.
fn term_shaped(core: &str) -> bool {
    let chars: Vec<char> = core.chars().collect();
    let capitals = chars.len() >= 2
        && chars[0].is_ascii_uppercase()
        && (chars[chars.len() - 1].is_ascii_uppercase() || chars[chars.len() - 1].is_ascii_digit());
    let digit = chars.iter().any(char::is_ascii_digit);
    let hyphen = chars.len() >= 3 && chars[1..chars.len() - 1].contains(&'-');
    capitals || digit || hyphen
}

/// A regex word character (`\w`).
fn word_char(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

/// `polish_server.py` `WORD_PARTS`: leading and trailing non-word characters
/// around the core.
fn word_parts(word: &str) -> (&str, &str, &str) {
    let lead_end = word
        .char_indices()
        .find(|(_, character)| word_char(*character))
        .map_or(word.len(), |(index, _)| index);
    let rest = &word[lead_end..];
    let core_end = rest
        .char_indices()
        .rev()
        .find(|(_, character)| word_char(*character))
        .map_or(0, |(index, character)| index + character.len_utf8());
    (&word[..lead_end], &rest[..core_end], &rest[core_end..])
}

/// `polish_server.py` `restore_terms`: put back a one-word dictionary term the
/// speaker said where the refiner wrote a different term-shaped word in its place
/// (a one-for-one swap in the word alignment). Everything else is left alone.
pub fn restore_terms(spoken: &str, output: &str, vocabulary: &[String]) -> String {
    let mut terms = std::collections::HashMap::new();
    for term in vocabulary {
        let term = term.trim();
        if !term.is_empty() && !term.contains(' ') {
            terms.insert(norm(term), term);
        }
    }
    if terms.is_empty() {
        return output.to_string();
    }
    let said: Vec<String> = spoken.split_whitespace().map(norm).collect();
    let said_set: std::collections::HashSet<&str> = said.iter().map(String::as_str).collect();
    let spans: Vec<(usize, usize)> = word_spans(output);
    let words: Vec<&str> = spans.iter().map(|(a, b)| &output[*a..*b]).collect();
    let normed: Vec<String> = words.iter().map(|word| norm(word)).collect();
    let mut fixes = std::collections::BTreeMap::new();
    for (i1, i2, j1, j2) in replace_opcodes(&said, &normed) {
        if i2 - i1 != j2 - j1 {
            continue;
        }
        for (i, j) in (i1..i2).zip(j1..j2) {
            let Some(term) = terms.get(&said[i]) else {
                continue;
            };
            let (lead, core, trail) = word_parts(words[j]);
            if !said_set.contains(normed[j].as_str())
                && normed[j] != norm(term)
                && term_shaped(core)
            {
                fixes.insert(j, format!("{lead}{term}{trail}"));
            }
        }
    }
    if fixes.is_empty() {
        return output.to_string();
    }
    let mut out = String::with_capacity(output.len());
    let mut last = 0;
    for (j, (a, b)) in spans.iter().enumerate() {
        if let Some(fix) = fixes.get(&j) {
            out.push_str(&output[last..*a]);
            out.push_str(fix);
            last = *b;
        }
    }
    out.push_str(&output[last..]);
    out
}

/// Byte spans of the runs of non-whitespace (`re.finditer(r"\S+")`).
fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (index, character) in text.char_indices() {
        match (character.is_whitespace(), start) {
            (true, Some(begin)) => {
                spans.push((begin, index));
                start = None;
            }
            (false, None) => start = Some(index),
            _ => {}
        }
    }
    if let Some(begin) = start {
        spans.push((begin, text.len()));
    }
    spans
}

/// The `replace` opcodes of Python's `difflib.SequenceMatcher(a, b, autojunk=False)`
/// (no junk), as (i1, i2, j1, j2).
fn replace_opcodes(a: &[String], b: &[String]) -> Vec<(usize, usize, usize, usize)> {
    let mut b2j: std::collections::HashMap<&str, Vec<usize>> = std::collections::HashMap::new();
    for (j, item) in b.iter().enumerate() {
        b2j.entry(item.as_str()).or_default().push(j);
    }
    let longest = |alo: usize, ahi: usize, blo: usize, bhi: usize| {
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0);
        let mut j2len: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for (i, item) in a.iter().enumerate().take(ahi).skip(alo) {
            let mut next = std::collections::HashMap::new();
            for &j in b2j.get(item.as_str()).map(Vec::as_slice).unwrap_or(&[]) {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = j
                    .checked_sub(1)
                    .and_then(|p| j2len.get(&p))
                    .copied()
                    .unwrap_or(0)
                    + 1;
                next.insert(j, k);
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
            j2len = next;
        }
        // difflib's extension over equal neighbours (no junk to extend over)
        while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && a[besti + bestsize] == b[bestj + bestsize]
        {
            bestsize += 1;
        }
        (besti, bestj, bestsize)
    };
    let mut queue = vec![(0, a.len(), 0, b.len())];
    let mut blocks = Vec::new();
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = longest(alo, ahi, blo, bhi);
        if k > 0 {
            blocks.push((i, j, k));
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    blocks.sort_unstable();
    // adjacent blocks merge in difflib; merging does not change where the gaps are
    blocks.push((a.len(), b.len(), 0));
    let mut opcodes = Vec::new();
    let (mut i, mut j) = (0, 0);
    for (ai, bj, size) in blocks {
        if i < ai && j < bj {
            opcodes.push((i, ai, j, bj));
        }
        i = ai + size;
        j = bj + size;
    }
    opcodes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phonon-win-dictionary-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn with(entries: &[(&str, Option<&str>)]) -> DictionaryFile {
        let mut dictionary = DictionaryFile::default();
        for (phrase, replacement) in entries {
            match replacement {
                Some(replacement) => assert!(add_replacement(&mut dictionary, phrase, replacement)),
                None => assert!(add_word(&mut dictionary, phrase)),
            }
        }
        dictionary
    }

    /// A file as the Mac app writes it (Swift JSONEncoder, pretty, sorted keys).
    const MAC_FILE: &str = r#"{
  "entries" : [
    {
      "phrase" : "Kubernetes",
      "source" : "phonon-app",
      "spoken_forms" : [
        "cooper netties"
      ],
      "starred" : true,
      "usage_count" : 3
    },
    {
      "phrase" : "kates",
      "replacement" : "k8s",
      "source" : "phonon-app",
      "spoken_forms" : [

      ],
      "starred" : false,
      "usage_count" : 0
    }
  ],
  "schema_version" : 1,
  "updated_at_unix_ms" : 1762900000000
}"#;

    #[test]
    fn the_macs_file_round_trips() {
        let dir = scratch("mac");
        let file = dir.join(FILE);
        std::fs::write(&file, MAC_FILE).unwrap();
        let mut dictionary = load_at(&file);
        assert_eq!(dictionary.entries.len(), 2);
        assert_eq!(dictionary.entries[0].spoken_forms, ["cooper netties"]);
        assert!(dictionary.entries[0].starred);
        assert_eq!(dictionary.entries[0].usage_count, 3);
        assert_eq!(dictionary.entries[1].replacement.as_deref(), Some("k8s"));
        save_at(&mut dictionary, &file).unwrap();
        let again = load_at(&file);
        assert_eq!(again.entries, dictionary.entries);
        // the keys the Mac decodes, and no temporary file left behind
        let written = std::fs::read_to_string(&file).unwrap();
        for key in [
            "schema_version",
            "updated_at_unix_ms",
            "spoken_forms",
            "usage_count",
        ] {
            assert!(written.contains(key), "{key}");
        }
        assert!(!dir.join("dictionary.json.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_is_empty_and_corrupt_is_moved_aside() {
        let dir = scratch("corrupt");
        let file = dir.join(FILE);
        assert!(load_at(&file).entries.is_empty());
        std::fs::write(&file, "{ not json").unwrap();
        assert!(load_at(&file).entries.is_empty());
        assert!(!file.exists());
        let aside: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(aside.len(), 1);
        assert!(aside[0].starts_with("dictionary.json.corrupt-"));
        assert_eq!(
            std::fs::read_to_string(dir.join(&aside[0])).unwrap(),
            "{ not json"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn add_and_remove() {
        let mut dictionary = DictionaryFile::default();
        assert!(add_word(&mut dictionary, "Kubernetes"));
        assert!(!add_word(&mut dictionary, "kubernetes "));
        assert!(!add_word(&mut dictionary, "  "));
        assert!(add_replacement(&mut dictionary, "kates", "k8s"));
        assert!(!add_replacement(&mut dictionary, "same", "same"));
        assert_eq!(dictionary.entries.len(), 2);
        assert!(!remove(&mut dictionary, "kates", None));
        assert!(remove(&mut dictionary, "KATES", Some("K8S")));
        assert!(remove(&mut dictionary, "kubernetes", None));
        assert!(dictionary.entries.is_empty());
    }

    #[test]
    fn an_edit_is_read_at_the_next_dictation() {
        let dir = scratch("live");
        let file = dir.join(FILE);
        let mut live = Live::new(file.clone());
        assert!(live.current().entries.is_empty());
        let mut dictionary = with(&[("Kubernetes", None)]);
        save_at(&mut dictionary, &file).unwrap();
        assert_eq!(live.current().entries.len(), 1);
        assert!(add_word(&mut dictionary, "Blackwell"));
        save_at(&mut dictionary, &file).unwrap();
        assert_eq!(live.current().entries.len(), 2);
        std::fs::remove_file(&file).unwrap();
        assert!(live.current().entries.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_dictionary_changes_nothing() {
        let empty = DictionaryFile::default();
        assert_eq!(prepare(&empty, "send it to cooper netties"), None);
        let raw = "okay send it to cooper netties";
        // the prompt without a vocabulary is the prompt as before
        assert_eq!(
            crate::polish::prompt_with(raw, "minimal", &[]),
            crate::polish::prompt(raw, "minimal")
        );
        // and the answer goes through the same stages as before
        for mode in ["minimal", "prompt"] {
            let answer = "Send it to cooper netties. 1. one 2. two".to_string();
            let before = phonon_llm::linebreak::layout(
                &phonon_llm::opener::restore(raw, &answer, mode),
                mode,
                false,
            );
            assert_eq!(
                crate::polish::finish(raw, answer.clone(), false, mode, None, false),
                before
            );
            assert_eq!(
                crate::polish::finish(raw, answer.clone(), true, mode, None, false),
                answer
            );
            assert_eq!(
                crate::polish::finish(raw, answer, false, mode, None, true),
                phonon_llm::numpass::apply(raw, &before)
            );
        }
    }

    #[test]
    fn a_replacement_is_applied_before_and_after_the_refiner() {
        let dictionary = with(&[("kates", Some("k8s"))]);
        let prepared = prepare(&dictionary, "deploy it on kates today").unwrap();
        assert_eq!(prepared.input, "deploy it on k8s today");
        assert_eq!(prepared.vocabulary, ["k8s"]);
        let prompt = crate::polish::prompt_with(&prepared.input, "minimal", &prepared.vocabulary);
        assert!(prompt.contains(
            "Writing mode: minimal (notes and documents).\nVocabulary the speaker may use: k8s<|im_end|>"
        ));
        assert!(prompt.contains("<|im_start|>user\ndeploy it on k8s today<|im_end|>"));
        assert_eq!(
            apply_replacements(&dictionary, "Deploy it on Kates today."),
            "Deploy it on k8s today."
        );
    }

    #[test]
    fn a_misheard_term_is_pre_corrected_but_ordinary_english_is_not() {
        let dictionary = with(&[
            ("Kubernetes", None),
            ("Claude Code", None),
            ("Pallas", None),
            ("uv init", None),
            ("MacBook", None),
        ]);
        let prepared = prepare(&dictionary, "ask clod kode about kubernetis").unwrap();
        assert_eq!(prepared.input, "ask Claude Code about Kubernetes");
        assert_eq!(prepared.vocabulary[..2], ["Claude Code", "Kubernetes"]);
        // ordinary English never moves, even where it keys like a term (please and
        // Pallas, even at and uv init, my macbook and MacBook)
        for sentence in [
            "please stop there",
            "it is worse even at batch one",
            "the cabinet is in the kitchen",
        ] {
            assert_eq!(prepare(&dictionary, sentence).unwrap().input, sentence);
        }
        assert!(prepare(&dictionary, "on my macbuk today")
            .unwrap()
            .input
            .starts_with("on my "));
    }

    #[test]
    fn the_vocabulary_is_capped_like_the_mac() {
        let mut dictionary = DictionaryFile::default();
        for index in 0..200 {
            add_word(&mut dictionary, &format!("Term{index}x"));
        }
        let envelope = dictionary.prepare_polish_input("term1x");
        let cap = refiner_vocabulary(&envelope).len();
        assert!(cap <= 30, "{cap}");
    }

    #[test]
    fn restore_terms_matches_the_sidecar() {
        let vocab = vec!["MTP".to_string(), "MLP".to_string()];
        // polish_server.py's own example
        assert_eq!(
            restore_terms("the native mtp path", "The native MTK path.", &vocab),
            "The native MTP path."
        );
        // a word that is not term-shaped stays
        assert_eq!(
            restore_terms("the native mtp path", "The native empty path.", &vocab),
            "The native empty path."
        );
        // a word the speaker said stays
        assert_eq!(
            restore_terms("mtp and MTK", "MTK and MTK", &vocab),
            "MTK and MTK"
        );
        assert_eq!(restore_terms("a b", "A B", &[]), "A B");
    }

    #[test]
    fn opcodes_match_difflib() {
        let words = |text: &str| text.split(' ').map(String::from).collect::<Vec<_>>();
        // difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes(), replace only
        assert_eq!(
            replace_opcodes(&words("a b c d e"), &words("a x c y z e")),
            [(1, 2, 1, 2), (3, 4, 3, 5)]
        );
        assert_eq!(
            replace_opcodes(&words("the native mtp path"), &words("the native mtk path")),
            [(2, 3, 2, 3)]
        );
        assert_eq!(
            replace_opcodes(&words("a b"), &words("c d")),
            [(0, 2, 0, 2)]
        );
    }
}
