//! Zero-dependency local transcript search: stemmed BM25 with tiered fuzzy matching.
//!
//! Ranking is tiered. Documents that contain every unquoted query term by exact stem always
//! rank above documents that only match some terms, or match them through a prefix, a
//! dictionary alias, a small edit distance or a Soundex code. Within a tier, BM25 scores are
//! weighted by how each term matched: exact stem 1.0, prefix or dictionary alias 0.5,
//! Levenshtein 1 to 2 edits 0.35, Soundex 0.2. Quoted phrases are exact filters.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

pub const WEIGHT_EXACT: f64 = 1.0;
pub const WEIGHT_PREFIX: f64 = 0.5;
pub const WEIGHT_ALIAS: f64 = 0.5;
pub const WEIGHT_EDIT: f64 = 0.35;
pub const WEIGHT_SOUNDEX: f64 = 0.2;

/// Fuzzy (edit distance and Soundex) expansion only runs for terms with fewer exact hits.
const FUZZY_MAX_EXACT_HITS: usize = 3;
/// Vocabulary words in more than this share of documents are "common" fuzzy targets.
const COMMON_DOC_SHARE: f64 = 0.005;
/// The common-word rule only applies to corpora at least this large.
const COMMON_MIN_DOCS: usize = 5;
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

/// Splits text into word tokens (Unicode alphanumerics) with ASCII lowercasing.
pub fn tokenize(text: &str) -> Vec<String> {
    token_spans(text)
        .into_iter()
        .map(|(start, end)| text[start..end].to_ascii_lowercase())
        .collect()
}

/// Byte spans of the word tokens in `text`.
fn token_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_alphanumeric() {
            if start.is_none() {
                start = Some(idx);
            }
        } else if let Some(s) = start.take() {
            spans.push((s, idx));
        }
    }
    if let Some(s) = start {
        spans.push((s, text.len()));
    }
    spans
}

fn is_vowel(b: u8) -> bool {
    matches!(b, b'a' | b'e' | b'i' | b'o' | b'u')
}

fn has_vowel(s: &str) -> bool {
    s.bytes().any(|b| is_vowel(b) || b == b'y')
}

/// Porter-lite English suffix stemmer. Applied identically to documents and queries, so it
/// favours consistent conflation (`ship`, `ships`, `shipped`, `shipping` -> `ship`) over
/// producing real words.
pub fn stem(word: &str) -> String {
    let w = word.to_ascii_lowercase();
    if w.len() <= 3 || !w.is_ascii() || w.bytes().any(|b| b.is_ascii_digit()) {
        return w;
    }
    let mut s = w;

    // Plurals.
    if let Some(base) = s.strip_suffix("sses") {
        s = format!("{base}ss");
    } else if let Some(base) = s.strip_suffix("ies") {
        if base.len() >= 2 {
            s = format!("{base}y");
        }
    } else if s.ends_with('s')
        && !s.ends_with("ss")
        && !s.ends_with("us")
        && !s.ends_with("is")
        && s.len() > 3
    {
        s.pop();
    }

    // Derivational suffixes (longest first).
    for (suffix, replacement) in [
        ("ational", "ate"),
        ("ization", "ize"),
        ("fulness", "ful"),
        ("iveness", "ive"),
        ("ousness", "ous"),
        ("ation", "ate"),
        ("ness", ""),
        ("ment", ""),
    ] {
        if let Some(base) = s.strip_suffix(suffix) {
            if base.len() >= 3 && has_vowel(base) {
                s = format!("{base}{replacement}");
                break;
            }
        }
    }

    // Inflections.
    let mut stripped = false;
    for suffix in ["ing", "edly", "ed", "ly"] {
        if let Some(base) = s.strip_suffix(suffix) {
            if base.len() >= 3 && has_vowel(base) {
                s = base.to_string();
                stripped = true;
                break;
            }
        }
    }
    if stripped {
        let bytes = s.as_bytes();
        let n = bytes.len();
        if n >= 2
            && bytes[n - 1] == bytes[n - 2]
            && !is_vowel(bytes[n - 1])
            && !matches!(bytes[n - 1], b'l' | b's' | b'z')
        {
            s.pop();
        }
    }

    // Trailing silent e: `hope`/`hoping` -> `hop`, `make`/`making` -> `mak`.
    if s.len() >= 4 && s.ends_with('e') {
        s.pop();
    }
    s
}

/// American Soundex code (`R163` for `Robert`). Empty when the word has no ASCII letters.
pub fn soundex(word: &str) -> String {
    fn code(c: u8) -> u8 {
        match c {
            b'b' | b'f' | b'p' | b'v' => b'1',
            b'c' | b'g' | b'j' | b'k' | b'q' | b's' | b'x' | b'z' => b'2',
            b'd' | b't' => b'3',
            b'l' => b'4',
            b'm' | b'n' => b'5',
            b'r' => b'6',
            _ => b'0',
        }
    }
    let letters: Vec<u8> = word
        .bytes()
        .filter(u8::is_ascii_alphabetic)
        .map(|b| b.to_ascii_lowercase())
        .collect();
    let Some(&first) = letters.first() else {
        return String::new();
    };
    let mut out = vec![first.to_ascii_uppercase()];
    let mut last = code(first);
    for &c in &letters[1..] {
        let d = code(c);
        if d != b'0' && d != last {
            out.push(d);
            if out.len() == 4 {
                break;
            }
        }
        // `h` and `w` do not separate equal codes; vowels do.
        if c != b'h' && c != b'w' {
            last = d;
        }
    }
    while out.len() < 4 {
        out.push(b'0');
    }
    String::from_utf8(out).unwrap_or_default()
}

/// Levenshtein distance, or `None` when it exceeds `max`.
pub fn levenshtein_bounded(a: &str, b: &str, max: usize) -> Option<usize> {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > max {
        return None;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        let mut row_min = cur[0];
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
            row_min = row_min.min(cur[j + 1]);
        }
        if row_min > max {
            return None;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let d = prev[b.len()];
    (d <= max).then_some(d)
}

/// A term a query word may also match: one stem, or a run of stems (`phone on`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Target {
    Stem(String),
    Seq(Vec<String>),
}

/// Read-only single-word alias expansion built from `dictionary.json`.
#[derive(Debug, Default, Clone)]
pub struct Aliases {
    map: HashMap<String, Vec<Target>>,
}

impl Aliases {
    /// Loads `dictionary.json` without ever writing it. Missing or unreadable files give no aliases.
    pub fn load(dictionary_path: &Path) -> Self {
        let Ok(bytes) = fs::read(dictionary_path) else {
            return Self::default();
        };
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            return Self::default();
        };
        let mut groups = Vec::new();
        for entry in value
            .get("entries")
            .and_then(|e| e.as_array())
            .into_iter()
            .flatten()
        {
            let mut forms = Vec::new();
            for key in ["phrase", "replacement"] {
                if let Some(s) = entry.get(key).and_then(|v| v.as_str()) {
                    forms.push(s.to_string());
                }
            }
            for form in entry
                .get("spoken_forms")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str())
            {
                forms.push(form.to_string());
            }
            groups.push(forms);
        }
        Self::from_groups(&groups)
    }

    /// Builds aliases from groups of equivalent forms (a phrase, its replacement and spoken forms).
    pub fn from_groups(groups: &[Vec<String>]) -> Self {
        let mut map: HashMap<String, Vec<Target>> = HashMap::new();
        for forms in groups {
            let mut singles: Vec<String> = Vec::new();
            let mut seqs: Vec<Vec<String>> = Vec::new();
            let mut keys: Vec<String> = Vec::new();
            for form in forms {
                let tokens = tokenize(form);
                match tokens.len() {
                    0 => {}
                    1 => {
                        let s = stem(&tokens[0]);
                        if !singles.contains(&s) {
                            singles.push(s.clone());
                            keys.push(s);
                        }
                    }
                    _ => {
                        let seq: Vec<String> = tokens.iter().map(|t| stem(t)).collect();
                        // Split-word form: `phone on` is also keyed by its joined spelling.
                        let joined = stem(&tokens.concat());
                        if !keys.contains(&joined) {
                            keys.push(joined);
                        }
                        if !seqs.contains(&seq) {
                            seqs.push(seq);
                        }
                    }
                }
            }
            for key in &keys {
                let entry = map.entry(key.clone()).or_default();
                for single in &singles {
                    let target = Target::Stem(single.clone());
                    if single != key && !entry.contains(&target) {
                        entry.push(target);
                    }
                }
                for seq in &seqs {
                    let target = Target::Seq(seq.clone());
                    if !entry.contains(&target) {
                        entry.push(target);
                    }
                }
            }
        }
        map.retain(|_, v| !v.is_empty());
        Self { map }
    }
}

/// A parsed query: unquoted terms plus exact quoted phrases.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Query {
    pub terms: Vec<String>,
    pub phrases: Vec<Vec<String>>,
}

pub fn parse_query(query: &str) -> Query {
    let mut parsed = Query::default();
    let mut rest = query;
    let mut unquoted = String::new();
    while let Some(open) = rest.find('"') {
        unquoted.push_str(&rest[..open]);
        unquoted.push(' ');
        let after = &rest[open + 1..];
        match after.find('"') {
            Some(close) => {
                let phrase = tokenize(&after[..close]);
                if !phrase.is_empty() {
                    parsed.phrases.push(phrase);
                }
                rest = &after[close + 1..];
            }
            None => {
                rest = after;
            }
        }
    }
    unquoted.push_str(rest);
    let mut seen = HashSet::new();
    for token in tokenize(&unquoted) {
        if seen.insert(stem(&token)) {
            parsed.terms.push(token);
        }
    }
    parsed
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    /// Index into the documents passed to [`search`].
    pub index: usize,
    pub score: f64,
    /// True when the document contains every unquoted term by exact stem.
    pub all_terms: bool,
    /// Stems in the document that matched, for highlighting.
    pub matched_stems: HashSet<String>,
}

struct DocIndex {
    words: Vec<String>,
    stems: Vec<String>,
    tf: HashMap<String, u32>,
}

impl DocIndex {
    fn new(text: &str) -> Self {
        let words = tokenize(text);
        let stems: Vec<String> = words.iter().map(|w| stem(w)).collect();
        let mut tf = HashMap::new();
        for s in &stems {
            *tf.entry(s.clone()).or_insert(0) += 1;
        }
        Self { words, stems, tf }
    }

    fn count_seq(seq: &[String], hay: &[String]) -> u32 {
        if seq.is_empty() || hay.len() < seq.len() {
            return 0;
        }
        hay.windows(seq.len()).filter(|w| *w == seq).count() as u32
    }

    fn count(&self, target: &Target) -> u32 {
        match target {
            Target::Stem(s) => self.tf.get(s).copied().unwrap_or(0),
            Target::Seq(seq) => Self::count_seq(seq, &self.stems),
        }
    }

    fn contains_phrase(&self, phrase: &[String]) -> bool {
        Self::count_seq(phrase, &self.words) > 0
    }
}

fn bm25(tf: u32, df: usize, n: usize, dl: usize, avgdl: f64) -> f64 {
    if tf == 0 {
        return 0.0;
    }
    let tf = tf as f64;
    let idf = (1.0 + (n as f64 - df as f64 + 0.5) / (df as f64 + 0.5)).ln();
    let norm = 1.0 - BM25_B + BM25_B * dl as f64 / avgdl.max(1.0);
    idf * tf * (BM25_K1 + 1.0) / (tf + BM25_K1 * norm)
}

/// Ranks `docs` against `query`. Documents without any match are left out.
pub fn search<S: AsRef<str>>(docs: &[S], query: &str, aliases: &Aliases) -> Vec<SearchHit> {
    let parsed = parse_query(query);
    if parsed.terms.is_empty() && parsed.phrases.is_empty() {
        return Vec::new();
    }
    let index: Vec<DocIndex> = docs.iter().map(|d| DocIndex::new(d.as_ref())).collect();
    let n = index.len();
    if n == 0 {
        return Vec::new();
    }
    let avgdl = index.iter().map(|d| d.stems.len()).sum::<usize>() as f64 / n as f64;
    let mut df: HashMap<&str, usize> = HashMap::new();
    for doc in &index {
        for s in doc.tf.keys() {
            *df.entry(s.as_str()).or_insert(0) += 1;
        }
    }
    let mut target_df_cache: HashMap<Target, usize> = HashMap::new();
    let mut target_df = |target: &Target| -> usize {
        if let Some(&v) = target_df_cache.get(target) {
            return v;
        }
        let v = match target {
            Target::Stem(s) => df.get(s.as_str()).copied().unwrap_or(0),
            Target::Seq(_) => index.iter().filter(|d| d.count(target) > 0).count(),
        };
        target_df_cache.insert(target.clone(), v);
        v
    };

    // Expand every unquoted term into weighted targets.
    let mut expansions: Vec<(String, Vec<(Target, f64)>)> = Vec::new();
    for term in &parsed.terms {
        let s = stem(term);
        let mut targets: HashMap<Target, f64> = HashMap::new();
        targets.insert(Target::Stem(s.clone()), WEIGHT_EXACT);
        let mut add = |t: Target, w: f64| {
            let slot = targets.entry(t).or_insert(0.0);
            if w > *slot {
                *slot = w;
            }
        };
        if let Some(alias_targets) = aliases.map.get(&s) {
            for t in alias_targets {
                add(t.clone(), WEIGHT_ALIAS);
            }
        }
        let exact_hits = df.get(s.as_str()).copied().unwrap_or(0);
        let fuzzy = exact_hits < FUZZY_MAX_EXACT_HITS && s.chars().count() >= 3;
        let max_edits = if s.chars().count() <= 4 { 1 } else { 2 };
        let query_code = soundex(&s);
        for (&vocab, &vocab_df) in &df {
            if vocab == s {
                continue;
            }
            if s.len() >= 3 && vocab.starts_with(s.as_str()) {
                add(Target::Stem(vocab.to_string()), WEIGHT_PREFIX);
                continue;
            }
            if !fuzzy {
                continue;
            }
            let common = n >= COMMON_MIN_DOCS && vocab_df as f64 > COMMON_DOC_SHARE * n as f64;
            let distance = levenshtein_bounded(&s, vocab, max_edits);
            if common {
                if exact_hits == 0 && distance == Some(1) {
                    add(Target::Stem(vocab.to_string()), WEIGHT_EDIT);
                }
                continue;
            }
            if distance.is_some() {
                add(Target::Stem(vocab.to_string()), WEIGHT_EDIT);
            } else if s.len() >= 4
                && vocab.len() >= 3
                && !query_code.is_empty()
                && soundex(vocab) == query_code
            {
                add(Target::Stem(vocab.to_string()), WEIGHT_SOUNDEX);
            }
        }
        expansions.push((s, targets.into_iter().collect()));
    }

    let mut hits = Vec::new();
    for (idx, doc) in index.iter().enumerate() {
        if !parsed.phrases.iter().all(|p| doc.contains_phrase(p)) {
            continue;
        }
        let dl = doc.stems.len();
        let mut score = 0.0;
        let mut all_terms = true;
        let mut any_term = false;
        let mut matched = HashSet::new();
        for (s, targets) in &expansions {
            if doc.tf.contains_key(s) {
                any_term = true;
            } else {
                all_terms = false;
            }
            let mut best = 0.0_f64;
            for (target, weight) in targets {
                let tf = doc.count(target);
                if tf == 0 {
                    continue;
                }
                any_term = true;
                match target {
                    Target::Stem(t) => {
                        matched.insert(t.clone());
                    }
                    Target::Seq(seq) => matched.extend(seq.iter().cloned()),
                }
                let value = weight * bm25(tf, target_df(target), n, dl, avgdl);
                best = best.max(value);
            }
            score += best;
        }
        for phrase in &parsed.phrases {
            let mut phrase_stems = HashSet::new();
            for word in phrase {
                phrase_stems.insert(stem(word));
            }
            for s in phrase_stems {
                let tf = doc.tf.get(&s).copied().unwrap_or(0);
                score +=
                    WEIGHT_EXACT * bm25(tf, df.get(s.as_str()).copied().unwrap_or(0), n, dl, avgdl);
                matched.insert(s);
            }
        }
        if !expansions.is_empty() && !any_term {
            continue;
        }
        hits.push(SearchHit {
            index: idx,
            score,
            all_terms,
            matched_stems: matched,
        });
    }
    hits.sort_by(|a, b| {
        b.all_terms
            .cmp(&a.all_terms)
            .then_with(|| b.score.total_cmp(&a.score))
            .then_with(|| a.index.cmp(&b.index))
    });
    hits
}

/// Number of tokens in `text` whose stem is in `matched`.
pub fn match_count(text: &str, matched: &HashSet<String>) -> usize {
    tokenize(text)
        .iter()
        .filter(|w| matched.contains(&stem(w)))
        .count()
}

/// A short excerpt of `text` around the first match, with matched words wrapped in `**...**`.
pub fn snippet(text: &str, matched: &HashSet<String>, max_chars: usize) -> String {
    let spans = token_spans(text);
    let is_match = |&(s, e): &(usize, usize)| matched.contains(&stem(&text[s..e]));
    let first = spans.iter().position(is_match);
    let char_len = text.chars().count();
    let (window_start, window_end) = if char_len <= max_chars {
        (0, text.len())
    } else {
        let anchor = first.map(|i| spans[i].0).unwrap_or(0);
        let anchor_chars = text[..anchor].chars().count();
        let lead = max_chars / 3;
        let start_chars = anchor_chars.saturating_sub(lead);
        let start_chars = start_chars.min(char_len.saturating_sub(max_chars));
        let byte_at = |c: usize| {
            text.char_indices()
                .nth(c)
                .map(|(b, _)| b)
                .unwrap_or(text.len())
        };
        let mut start = byte_at(start_chars);
        let mut end = byte_at(start_chars + max_chars);
        // Do not cut words in half.
        if let Some(&(s, _)) = spans.iter().find(|&&(s, e)| s < start && e > start) {
            start = s;
        }
        if let Some(&(_, e)) = spans.iter().find(|&&(s, e)| s < end && e > end) {
            end = e;
        }
        (start, end)
    };

    let mut out = String::new();
    if window_start > 0 {
        out.push_str("...");
    }
    let mut cursor = window_start;
    for span in spans
        .iter()
        .filter(|&&(s, e)| s >= window_start && e <= window_end)
    {
        if is_match(span) {
            out.push_str(&text[cursor..span.0]);
            out.push_str("**");
            out.push_str(&text[span.0..span.1]);
            out.push_str("**");
            cursor = span.1;
        }
    }
    out.push_str(&text[cursor..window_end]);
    if window_end < text.len() {
        out.push_str("...");
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stemmer_conflates_inflections() {
        for w in ["ship", "ships", "shipped", "shipping"] {
            assert_eq!(stem(w), "ship", "{w}");
        }
        assert_eq!(stem("hoping"), stem("hope"));
        assert_eq!(stem("making"), stem("make"));
        assert_eq!(stem("queries"), stem("query"));
        assert_eq!(stem("the"), "the");
        assert_eq!(stem("Phonon"), "phonon");
    }

    #[test]
    fn soundex_matches_reference_codes() {
        assert_eq!(soundex("Robert"), "R163");
        assert_eq!(soundex("Rupert"), "R163");
        assert_eq!(soundex("Ashcraft"), "A261");
        assert_eq!(soundex("Tymczak"), "T522");
        assert_eq!(soundex("Pfister"), "P236");
        assert_eq!(soundex("123"), "");
    }

    #[test]
    fn levenshtein_is_bounded() {
        assert_eq!(levenshtein_bounded("kitten", "sitten", 2), Some(1));
        assert_eq!(levenshtein_bounded("kitten", "sitting", 2), None);
        assert_eq!(levenshtein_bounded("kitten", "sitting", 3), Some(3));
        assert_eq!(levenshtein_bounded("abc", "abc", 0), Some(0));
        assert_eq!(levenshtein_bounded("a", "abcd", 2), None);
    }

    #[test]
    fn parses_quoted_phrases() {
        let q = parse_query(r#"ship "Design Review" ships now"#);
        assert_eq!(q.terms, vec!["ship", "now"]);
        assert_eq!(
            q.phrases,
            vec![vec!["design".to_string(), "review".to_string()]]
        );
    }

    #[test]
    fn all_term_exact_docs_outrank_partial_and_fuzzy() {
        let docs = [
            "we will ship the release candidate",
            "ship ship ship ship ship ship",
            "release notes only",
            "shipping the relase tomorrow",
        ];
        let hits = search(&docs, "ship release", &Aliases::default());
        assert_eq!(hits[0].index, 0);
        assert!(hits[0].all_terms);
        assert!(hits[1..].iter().all(|h| !h.all_terms));
        assert!(
            hits.iter().any(|h| h.index == 3),
            "fuzzy relase still found"
        );
    }

    #[test]
    fn quoted_phrase_filters_documents() {
        let docs = ["design review on friday", "review the design", "nothing"];
        let hits = search(&docs, r#""design review""#, &Aliases::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].index, 0);
    }

    #[test]
    fn dictionary_spoken_forms_expand_queries() {
        let aliases = Aliases::from_groups(&[vec![
            "Phonon".to_string(),
            "fonon".to_string(),
            "phone on".to_string(),
        ]]);
        let docs = [
            "the fonon app crashed",
            "turn the phone on please",
            "unrelated text",
        ];
        let hits = search(&docs, "phonon", &aliases);
        let found: Vec<usize> = hits.iter().map(|h| h.index).collect();
        assert!(found.contains(&0));
        assert!(found.contains(&1));
        assert!(!found.contains(&2));
    }

    #[test]
    fn edit_distance_fallback_is_guarded_for_common_words() {
        // Many docs containing `shop`; query `ship` has 3+ exact hits so no fuzzy expansion.
        let mut docs: Vec<String> = (0..10).map(|i| format!("shop visit {i}")).collect();
        docs.push("ship it".into());
        docs.push("ship now".into());
        docs.push("ship later".into());
        let hits = search(&docs, "ship", &Aliases::default());
        assert_eq!(hits.len(), 3);
        // Zero exact matches: common words are allowed at edit distance 1.
        let hits = search(&docs, "shap", &Aliases::default());
        assert!(!hits.is_empty());
    }

    #[test]
    fn snippet_highlights_matches() {
        let mut matched = HashSet::new();
        matched.insert(stem("shipping"));
        let s = snippet("We are shipping the CLI today.", &matched, 200);
        assert_eq!(s, "We are **shipping** the CLI today.");
        let long = format!("{} needle {}", "word ".repeat(80), "tail ".repeat(80));
        let mut m = HashSet::new();
        m.insert("needl".to_string());
        m.insert(stem("needle"));
        let s = snippet(&long, &m, 60);
        assert!(s.contains("**needle**"), "{s}");
        assert!(s.starts_with("...") && s.ends_with("..."));
    }
}
