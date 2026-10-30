//! Post-paste edit learning (SPEC: "post-insert edit capture").
//!
//! After Phonon pastes a dictation, the bar watches the text field it landed
//! in and sends back the region the user finally kept. This module aligns the
//! pasted text with that region, labels every word, and turns the edits into
//! proposals: a dictionary term ("Add `cuBLAS`? Heard as `coo blast`") or a
//! style preference (lowercase starts, no trailing period). It never writes
//! the dictionary; SPEC requires the user to approve every term.
//!
//! The alignment is a port of Wispr Flow 1.6's `text-diff-ops` (webpack
//! module 14517, `parseAlignmentSteps(pasted, edited, 4)`): tokens are split
//! on whitespace (CJK characters one per token), the common token prefix and
//! suffix are taken as matches, and the middle is aligned by a weighted edit
//! distance whose substitution cost is the normalized grapheme Levenshtein of
//! the two words times 4, so dissimilar words become a delete plus an insert
//! and similar ones a substitution. Each aligned word gets two labels, word
//! and punctuation, from `M` match, `Z` absent on both sides, `C` casing only,
//! `S` substitution, `I` insert, `D` delete and `E` edit-capture error (the
//! observation ended while the first or last word was half typed). Wispr then
//! mines contiguous `S/I/D` runs of at most four edited words
//! (`dictionary-learn-candidates`, module 23109), drops stopwords, words
//! already in the dictionary and contractions, and asks a server classifier
//! which candidates are real terms before auto-adding them with an undo
//! notification. Phonon has no server: the classifier is replaced by local
//! rules (the term must look like a term, and must sound like what was heard),
//! and nothing is added without the user's yes.

use crate::data::{self, DictionaryEntry, DictionaryFile};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

/// Substitution weight Wispr passes to `parseAlignmentSteps`.
const SUBSTITUTION_WEIGHT: f64 = 4.0;
/// Wispr's phrase learner keeps at most four edited words per candidate.
pub const MAX_TERM_WORDS: usize = 4;
/// Wispr sends at most four candidates per edit to its classifier.
pub const MAX_SUGGESTIONS: usize = 4;
/// An edit that changes more than half of the words is a rewrite, not a fix.
pub const REWRITE_RATIO: f64 = 0.5;
/// Short dictations are often fully corrected (`coo blast` to `cuBLAS` is two
/// word edits out of two), so a rewrite also needs this many word edits.
pub const REWRITE_MIN_EDITS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Label {
    Match,
    None,
    Delete,
    Insert,
    Substitution,
    CaptureError,
    Casing,
}

impl Label {
    pub fn code(self) -> char {
        match self {
            Label::Match => 'M',
            Label::None => 'Z',
            Label::Delete => 'D',
            Label::Insert => 'I',
            Label::Substitution => 'S',
            Label::CaptureError => 'E',
            Label::Casing => 'C',
        }
    }

    fn is_word_edit(self) -> bool {
        matches!(self, Label::Substitution | Label::Insert | Label::Delete)
    }
}

/// One aligned token pair. `original` is `None` for an inserted token and
/// `edited` is `None` for a deleted one; both keep their punctuation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignmentStep {
    pub word: Label,
    pub punct: Label,
    pub original: Option<String>,
    pub edited: Option<String>,
    /// The edited token starts a line (a newline precedes it): a list item
    /// or a paragraph begins there even without a period before it.
    pub line_start: bool,
}

impl AlignmentStep {
    pub fn original_word(&self) -> String {
        trim_edge_punct(self.original.as_deref().unwrap_or(""))
    }

    pub fn edited_word(&self) -> String {
        trim_edge_punct(self.edited.as_deref().unwrap_or(""))
    }

    fn same(token: &str) -> Self {
        Self {
            word: word_label(token, token),
            punct: punct_label(token, token),
            original: Some(token.to_string()),
            edited: Some(token.to_string()),
            line_start: false,
        }
    }
}

fn is_cjk(character: char) -> bool {
    // Han, Hiragana and Katakana by script extension, so the ideographic
    // full stop and comma count, as they do in Wispr's regex.
    matches!(character as u32,
        0x3001..=0x303F
        | 0x3040..=0x30FF
        | 0x31F0..=0x31FF
        | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xF900..=0xFAFF
        | 0xFF66..=0xFF9D
        | 0x20000..=0x2FA1F)
}

fn is_word_char(character: char) -> bool {
    character == '_' || character.is_alphanumeric()
}

fn is_punct_char(character: char) -> bool {
    !character.is_whitespace() && !is_word_char(character)
}

fn word_part(token: &str) -> String {
    token.chars().filter(|c| is_word_char(*c)).collect()
}

fn punct_part(token: &str) -> String {
    token.chars().filter(|c| is_punct_char(*c)).collect()
}

fn trim_edge_punct(token: &str) -> String {
    token.trim_matches(is_punct_char).to_string()
}

/// Whitespace-separated tokens; inside a chunk that holds CJK, each CJK
/// character and each punctuation run touching one is its own token.
pub fn tokenize(text: &str) -> Vec<String> {
    token_spans(text)
        .into_iter()
        .map(|(start, end)| text[start..end].to_string())
        .collect()
}

/// Byte ranges of the tokens `tokenize` returns.
pub fn token_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut chunk = Vec::new();
    for (offset, character) in text.char_indices() {
        if character.is_whitespace() {
            flush_chunk(&mut chunk, &mut spans);
        } else {
            chunk.push((offset, character));
        }
    }
    flush_chunk(&mut chunk, &mut spans);
    spans
}

fn flush_chunk(chunk: &mut Vec<(usize, char)>, spans: &mut Vec<(usize, usize)>) {
    let Some(&(first, _)) = chunk.first() else {
        return;
    };
    let end_of = |(offset, character): (usize, char)| offset + character.len_utf8();
    if !chunk.iter().any(|(_, c)| is_cjk(*c)) {
        spans.push((first, end_of(chunk[chunk.len() - 1])));
        chunk.clear();
        return;
    }
    let n = chunk.len();
    let mut split = vec![false; n];
    let mut start = 0;
    while start < n {
        if !is_punct_char(chunk[start].1) {
            start += 1;
            continue;
        }
        let mut end = start;
        while end + 1 < n && is_punct_char(chunk[end + 1].1) {
            end += 1;
        }
        if (start > 0 && is_cjk(chunk[start - 1].1)) || (end + 1 < n && is_cjk(chunk[end + 1].1)) {
            split[start..=end].fill(true);
        }
        start = end + 1;
    }
    let mut pending: Option<(usize, usize)> = None;
    for (index, &(offset, character)) in chunk.iter().enumerate() {
        if is_cjk(character) || split[index] {
            if let Some(span) = pending.take() {
                spans.push(span);
            }
            spans.push((offset, end_of((offset, character))));
        } else {
            let end = end_of((offset, character));
            pending = Some(pending.map_or((offset, end), |(begin, _)| (begin, end)));
        }
    }
    if let Some(span) = pending {
        spans.push(span);
    }
    chunk.clear();
}

/// The part of `edited` that is the dictation: text typed before or after it
/// (`prepended`/`appended` tokens, as counted by the alignment) is cut off,
/// so a corpus pair never teaches that a dictation grows a second message.
pub fn dictation_region(edited: &str, prepended_tokens: usize, appended_tokens: usize) -> String {
    let spans = token_spans(edited);
    if prepended_tokens + appended_tokens >= spans.len() {
        return String::new();
    }
    let start = spans[prepended_tokens].0;
    let end = spans[spans.len() - 1 - appended_tokens].1;
    edited[start..end].to_string()
}

fn punct_label(original: &str, edited: &str) -> Label {
    let (a, b) = (punct_part(original), punct_part(edited));
    if a == b {
        if a.is_empty() {
            Label::None
        } else {
            Label::Match
        }
    } else if !a.is_empty() && b.is_empty() {
        Label::Delete
    } else if a.is_empty() && !b.is_empty() {
        Label::Insert
    } else {
        Label::Substitution
    }
}

fn word_label(original: &str, edited: &str) -> Label {
    let (a, b) = (word_part(original), word_part(edited));
    if a == b {
        if a.is_empty() {
            Label::None
        } else {
            Label::Match
        }
    } else if a.to_lowercase() == b.to_lowercase() {
        Label::Casing
    } else if a.is_empty() && !b.is_empty() {
        Label::Insert
    } else if !a.is_empty() && b.is_empty() {
        Label::Delete
    } else {
        Label::Substitution
    }
}

fn scored(token: &str) -> Vec<char> {
    let word = word_part(token).to_lowercase();
    if word.is_empty() {
        token.chars().collect()
    } else {
        word.chars().collect()
    }
}

fn char_levenshtein(left: &[char], right: &[char]) -> usize {
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    for (row, l) in left.iter().enumerate() {
        let mut current = vec![row + 1];
        for (column, r) in right.iter().enumerate() {
            let cost = usize::from(l != r);
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

fn substitution_cost(left: &[char], right: &[char]) -> f64 {
    if left.is_empty() || right.is_empty() {
        return if left.len() == right.len() {
            0.0
        } else {
            SUBSTITUTION_WEIGHT
        };
    }
    if left.len() == 1 && right.len() == 1 {
        return if left[0] == right[0] {
            0.0
        } else {
            SUBSTITUTION_WEIGHT
        };
    }
    SUBSTITUTION_WEIGHT * char_levenshtein(left, right) as f64 / left.len().max(right.len()) as f64
}

fn cost_matrix(original: &[String], edited: &[String]) -> Vec<Vec<f64>> {
    let a: Vec<Vec<char>> = original.iter().map(|t| scored(t)).collect();
    let b: Vec<Vec<char>> = edited.iter().map(|t| scored(t)).collect();
    let mut matrix = vec![vec![0.0; b.len() + 1]; a.len() + 1];
    for (i, row) in matrix.iter_mut().enumerate() {
        row[0] = i as f64;
    }
    for (j, cell) in matrix[0].iter_mut().enumerate() {
        *cell = j as f64;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            matrix[i][j] = if a[i - 1] == b[j - 1] {
                matrix[i - 1][j - 1]
            } else {
                (matrix[i - 1][j] + 1.0)
                    .min(matrix[i][j - 1] + 1.0)
                    .min(matrix[i - 1][j - 1] + substitution_cost(&a[i - 1], &b[j - 1]))
            };
        }
    }
    matrix
}

fn backtrack(
    matrix: &[Vec<f64>],
    original: &[String],
    edited: &[String],
    offset: usize,
    total: usize,
) -> Vec<AlignmentStep> {
    let (mut a, mut c) = (original.len(), edited.len());
    let mut steps = Vec::new();
    while a > 0 || c > 0 {
        if a > 0 && c > 0 && original[a - 1] == edited[c - 1] {
            steps.push(AlignmentStep::same(&original[a - 1]));
            a -= 1;
            c -= 1;
        // Wispr's backtrack (src/main/text/text-diff-ops.js lines 165-178)
        // compares only the left and upper cells and takes the diagonal on a
        // tie, without checking that the diagonal produced matrix[a][c]; so
        // `foo` -> `bar` is one substitution, not a delete plus an insert.
        // Kept as is on purpose: the stored word_edits/punctuation_edits must
        // match Wispr's (89/89 replayed rows do), and S is also what the
        // phrase learner wants for a one-word replacement.
        } else if c > 0 && (a == 0 || matrix[a][c - 1] < matrix[a - 1][c]) {
            steps.push(AlignmentStep {
                word: word_label("", &edited[c - 1]),
                punct: punct_label("", &edited[c - 1]),
                original: None,
                edited: Some(edited[c - 1].clone()),
                line_start: false,
            });
            c -= 1;
        } else if a > 0 && (c == 0 || matrix[a - 1][c] < matrix[a][c - 1]) {
            steps.push(AlignmentStep {
                word: word_label(&original[a - 1], ""),
                punct: punct_label(&original[a - 1], ""),
                original: Some(original[a - 1].clone()),
                edited: None,
                line_start: false,
            });
            a -= 1;
        } else {
            let (from, to) = (&original[a - 1], &edited[c - 1]);
            let mut word = word_label(from, to);
            let position = offset + a;
            // A one-character word at either end of the text that starts or
            // ends the word it replaced is a half-typed word caught mid-edit.
            if (position == total || position == 1)
                && word == Label::Substitution
                && to.chars().count() == 1
                && (from.starts_with(to.as_str()) || from.ends_with(to.as_str()))
            {
                word = Label::CaptureError;
            }
            steps.push(AlignmentStep {
                word,
                punct: punct_label(from, to),
                original: Some(from.clone()),
                edited: Some(to.clone()),
                line_start: false,
            });
            a -= 1;
            c -= 1;
        }
    }
    steps.reverse();
    steps
}

/// Wispr's `parseAlignmentSteps(original, edited, 4)`.
pub fn align(original: &str, edited: &str) -> Vec<AlignmentStep> {
    let left = tokenize(original);
    let right = tokenize(edited);
    let mut suffix = 0;
    while suffix < left.len()
        && suffix < right.len()
        && left[left.len() - 1 - suffix] == right[right.len() - 1 - suffix]
    {
        suffix += 1;
    }
    let mut prefix = 0;
    while prefix < left.len() - suffix
        && prefix < right.len() - suffix
        && left[prefix] == right[prefix]
    {
        prefix += 1;
    }
    let middle_left = &left[prefix..left.len() - suffix];
    let middle_right = &right[prefix..right.len() - suffix];
    let matrix = cost_matrix(middle_left, middle_right);
    let mut steps: Vec<AlignmentStep> = left[..prefix]
        .iter()
        .map(|t| AlignmentStep::same(t))
        .collect();
    steps.extend(backtrack(
        &matrix,
        middle_left,
        middle_right,
        prefix,
        left.len(),
    ));
    steps.extend(
        left[left.len() - suffix..]
            .iter()
            .map(|t| AlignmentStep::same(t)),
    );
    // Each step with an edited token takes the next edited token in order.
    let spans = token_spans(edited);
    for (next, step) in steps.iter_mut().filter(|s| s.edited.is_some()).enumerate() {
        step.line_start = next > 0
            && spans
                .get(next)
                .is_some_and(|(start, _)| edited[spans[next - 1].1..*start].contains('\n'));
    }
    steps
}

/// The `word_edits` and `punctuation_edits` strings Wispr stores in
/// `userEditMetaData`, including its `processPunctuationVector` rule: a
/// trailing punctuation insert after a word-less token is a capture error.
pub fn edit_vectors(steps: &[AlignmentStep]) -> (String, String) {
    let words: String = steps.iter().map(|s| s.word.code()).collect();
    let mut punct: String = steps.iter().map(|s| s.punct.code()).collect();
    let bytes = punct.as_bytes();
    if bytes.len() >= 2
        && matches!(bytes[bytes.len() - 2], b'M' | b'S')
        && bytes[bytes.len() - 1] == b'I'
        && words.ends_with('Z')
    {
        punct.pop();
        punct.push('E');
    }
    (words, punct)
}

/// Text fields rewrite quotes, dashes, ellipses and spaces on their own;
/// those are never the user's correction.
pub fn normalize_typography(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201B}' | '\u{2032}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201F}' | '\u{2033}' => '"',
            '\u{2013}' | '\u{2014}' | '\u{2212}' => '-',
            '\u{00A0}' | '\u{2007}' | '\u{2009}' | '\u{202F}' => ' ',
            other => other,
        })
        .collect::<String>()
        .replace('\u{2026}', "...")
        .replace("--", "-")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditOutcome {
    /// The region still reads exactly as pasted.
    Unchanged,
    /// Only quotes, dashes, ellipses or space kinds changed.
    Typographic,
    /// The whole dictation was deleted or undone.
    Removed,
    /// The user kept typing before or after the dictation and left it alone.
    Appended,
    /// The observation ended on a half-typed word and nothing else changed.
    CaptureError,
    /// Only capitalization or punctuation changed.
    Style,
    /// Words changed; the edit is small enough to learn from.
    Correction,
    /// More than half of the words changed: a new message, not a fix.
    Rewrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TermReason {
    /// The words were replaced by something that sounds like them.
    Misheard,
    /// The word was right but its capitalization was not.
    Casing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StylePreference {
    /// The user lowercased the first letter of the dictation.
    LowercaseStart,
    /// The user lowercased every capitalized word.
    AllLowercase,
    /// The user removed the final period.
    NoTrailingPeriod,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Suggestion {
    DictionaryTerm {
        term: String,
        heard_as: String,
        reason: TermReason,
        /// The term is already in the dictionary; approving adds `heard_as`
        /// to its spoken forms.
        existing_entry: bool,
    },
    Style {
        preference: StylePreference,
    },
}

/// A candidate that was considered and dropped, kept for the corpus record
/// and for tests. `why` is a short stable code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Skipped {
    pub term: String,
    pub heard_as: String,
    pub why: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditAnalysis {
    pub outcome: EditOutcome,
    pub word_edits: String,
    pub punctuation_edits: String,
    /// Word edits inside the dictation over its length in words, 0 to 1.
    pub edit_distance: f64,
    pub word_edit_count: usize,
    /// Words typed before or after the dictation, which are not edits of it.
    pub appended_words: usize,
    pub suggestions: Vec<Suggestion>,
    pub skipped: Vec<Skipped>,
    /// Tokens of the kept text before and after the dictation region.
    #[serde(skip)]
    pub prepended_tokens: usize,
    #[serde(skip)]
    pub appended_tokens: usize,
}

impl EditAnalysis {
    fn plain(outcome: EditOutcome, steps: &[AlignmentStep]) -> Self {
        let (word_edits, punctuation_edits) = edit_vectors(steps);
        Self {
            outcome,
            word_edits,
            punctuation_edits,
            edit_distance: 0.0,
            word_edit_count: 0,
            appended_words: 0,
            suggestions: Vec::new(),
            skipped: Vec::new(),
            prepended_tokens: 0,
            appended_tokens: 0,
        }
    }
}

/// Classify what the user did to a pasted dictation.
///
/// `edited` is the region the user kept when the observation ended.
/// `within_threshold` is the last state of that region before it crossed the
/// rewrite threshold, when it did; corrections made before a rewrite are still
/// learned from it.
pub fn analyze(
    pasted: &str,
    edited: &str,
    within_threshold: Option<&str>,
    dictionary: &DictionaryFile,
) -> EditAnalysis {
    let pasted_norm = normalize_typography(pasted);
    let edited_norm = normalize_typography(edited);
    let steps = align(&pasted_norm, &edited_norm);
    if edited.trim().is_empty() {
        return EditAnalysis::plain(EditOutcome::Removed, &steps);
    }
    if pasted == edited {
        return EditAnalysis::plain(EditOutcome::Unchanged, &steps);
    }
    if pasted_norm == edited_norm {
        return EditAnalysis::plain(EditOutcome::Typographic, &steps);
    }

    let mut analysis = EditAnalysis::plain(EditOutcome::Correction, &steps);
    let Some(first) = steps.iter().position(|s| s.original.is_some()) else {
        return EditAnalysis::plain(EditOutcome::Removed, &steps);
    };
    let last = steps
        .iter()
        .rposition(|s| s.original.is_some())
        .unwrap_or(first);
    analysis.appended_words = steps[..first]
        .iter()
        .chain(steps[last + 1..].iter())
        .filter(|s| !s.edited_word().is_empty())
        .count();
    analysis.prepended_tokens = first;
    analysis.appended_tokens = steps.len() - 1 - last;
    let prepended = steps[..first].iter().any(|s| !s.edited_word().is_empty());
    let appended = steps[last + 1..]
        .iter()
        .any(|s| !s.edited_word().is_empty());
    let core = &steps[first..=last];
    let pasted_words = core.iter().filter(|s| s.original.is_some()).count();
    let kept_words = core.iter().filter(|s| s.edited.is_some()).count();
    let word_edits = core
        .iter()
        .filter(|s| s.word != Label::Match && s.word != Label::None)
        .count();
    analysis.word_edit_count = word_edits;
    analysis.edit_distance = if pasted_words.max(kept_words) == 0 {
        0.0
    } else {
        (word_edits as f64 / pasted_words.max(kept_words) as f64).min(1.0)
    };

    // Typing on past the dictation moves its final period (`it.` becomes
    // `it now.`), and typing in front of it lowercases its first word; both
    // belong to the new text, not to an edit of the dictation.
    let incidental = |index: usize| {
        (appended && index == last && steps[index].word == Label::Match)
            || (prepended && index == first && steps[index].word == Label::Casing)
    };
    let any_word_edit = core.iter().any(|s| s.word.is_word_edit());
    let any_casing = (first..=last).any(|i| steps[i].word == Label::Casing && !incidental(i));
    let any_punct = (first..=last).any(|i| {
        !matches!(
            steps[i].punct,
            Label::Match | Label::None | Label::CaptureError
        ) && !incidental(i)
    });
    let any_capture_error = core
        .iter()
        .any(|s| s.word == Label::CaptureError || s.punct == Label::CaptureError);

    if !any_word_edit && !any_casing && !any_punct {
        analysis.outcome = if any_capture_error {
            EditOutcome::CaptureError
        } else {
            EditOutcome::Appended
        };
        return analysis;
    }

    // A technical dictation can have most of its words corrected (`coo blast
    // and pie torch` to `cuBLAS and PyTorch`), so only edits that do not sound
    // like what they replaced count toward a rewrite.
    let runs = word_runs(&steps, first, last);
    let misheard_edits: usize = runs
        .iter()
        .filter(|(begin, end)| {
            let run = &steps[*begin..=*end];
            let heard: Vec<String> = run.iter().map(AlignmentStep::original_word).collect();
            let kept: Vec<String> = run.iter().map(AlignmentStep::edited_word).collect();
            sounds_alike(&heard.join(" "), &kept.join(" "))
        })
        .map(|(begin, end)| {
            steps[*begin..=*end]
                .iter()
                .filter(|s| s.word != Label::Match && s.word != Label::None)
                .count()
        })
        .sum();
    let unexplained = word_edits.saturating_sub(misheard_edits);
    let unexplained_ratio = unexplained as f64 / pasted_words.max(kept_words).max(1) as f64;
    if any_word_edit && unexplained >= REWRITE_MIN_EDITS && unexplained_ratio > REWRITE_RATIO {
        analysis.outcome = EditOutcome::Rewrite;
        if let Some(earlier) = within_threshold.filter(|text| *text != pasted && *text != edited) {
            let inner = analyze(pasted, earlier, None, dictionary);
            if inner.outcome != EditOutcome::Rewrite {
                analysis.suggestions = inner.suggestions;
                analysis.skipped = inner.skipped;
            }
        }
        return analysis;
    }

    analysis.outcome = if any_word_edit {
        EditOutcome::Correction
    } else {
        EditOutcome::Style
    };
    let mut suggestions = Vec::new();
    let mut skipped = Vec::new();
    let released = mine_word_runs(&steps, &runs, dictionary, &mut suggestions, &mut skipped);
    let in_run = |index: usize| runs.iter().any(|(b, e)| (*b..=*e).contains(&index));
    let casing: Vec<usize> = (first..=last)
        .filter(|i| {
            steps[*i].word == Label::Casing
                && (!in_run(*i) || released.contains(i))
                && !incidental(*i)
        })
        .collect();
    mine_casing(&steps, &casing, dictionary, &mut suggestions, &mut skipped);
    if !prepended {
        mine_style(
            &steps,
            first,
            last,
            any_word_edit,
            appended,
            &mut suggestions,
        );
    }

    let mut seen = std::collections::HashSet::new();
    suggestions.retain(|suggestion| {
        let key = match suggestion {
            Suggestion::DictionaryTerm { term, .. } => format!("t:{}", term.to_lowercase()),
            Suggestion::Style { preference } => format!("s:{preference:?}"),
        };
        seen.insert(key)
    });
    // Capitalized terms first, like Wispr's classifier queue.
    suggestions.sort_by_key(|suggestion| match suggestion {
        Suggestion::DictionaryTerm { term, .. } => u8::from(term == &term.to_lowercase()),
        Suggestion::Style { .. } => 2,
    });
    suggestions.truncate(MAX_SUGGESTIONS);
    analysis.suggestions = suggestions;
    analysis.skipped = skipped;
    analysis
}

fn sentence_start(steps: &[AlignmentStep], index: usize) -> bool {
    if index == 0 {
        return true;
    }
    // A kept token that starts a line (list item, new paragraph) starts a
    // sentence whether or not a period came before it.
    if steps[index..]
        .iter()
        .find(|s| s.edited.is_some())
        .is_some_and(|s| s.line_start)
    {
        return true;
    }
    // Sentence position is a property of the kept text: a deleted filler
    // (`go um then` to `go. Then`) does not hide the period before it.
    let previous = steps[..index]
        .iter()
        .rev()
        .find_map(|s| s.edited.as_deref());
    match previous {
        None => true,
        Some(token) => token.ends_with(['.', '!', '?', ':', '\n']),
    }
}

fn join_words(words: &[String]) -> String {
    if words.iter().any(|w| w.chars().any(is_cjk)) {
        words.concat()
    } else {
        words.join(" ")
    }
}

/// Wispr's phrase learner takes maximal runs of `S/I/D`. A multi-word term is
/// often one substitution plus one recased word (`clod code` to `Claude
/// Code`), so a casing step touching a run joins it; a run made only of casing
/// steps is not a run.
fn word_runs(steps: &[AlignmentStep], first: usize, last: usize) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start: Option<usize> = None;
    let close = |begin: usize, end: usize, runs: &mut Vec<(usize, usize)>| {
        if steps[begin..=end].iter().any(|s| s.word.is_word_edit()) {
            runs.push((begin, end));
        }
    };
    for (index, step) in steps.iter().enumerate().take(last + 1).skip(first) {
        if step.word.is_word_edit() || step.word == Label::Casing {
            start.get_or_insert(index);
        } else if let Some(begin) = start.take() {
            close(begin, index - 1, &mut runs);
        }
    }
    if let Some(begin) = start {
        close(begin, last, &mut runs);
    }
    runs
}

/// A count written as digits (`4x`, `8x`, `1`, `x2`): how many of a term,
/// not part of its name.
fn is_count(word: &str) -> bool {
    let core = word
        .strip_suffix(['x', '×'])
        .or_else(|| word.strip_prefix(['x', '×']))
        .unwrap_or(word);
    !core.is_empty() && core.chars().all(|c| c.is_ascii_digit())
}

/// A word at the edge of a run that is not part of the term the run
/// corrects: a count in front (`4x B300`; a number after a name is its
/// version, `Llama 3`), a stopword (`INT4 Then we`), an ordinary lowercase
/// word (`from-scratch CuTe DSL`), or a heard stopword the user deleted
/// after the term (`INT4` heard as `int four and`). A deleted stopword in
/// front is kept: it is usually part of the mishearing (`and roid` for
/// `Android`, `at las` for `Atlas`), and dropping it would teach a bare
/// fragment (`las`) as the spoken form. A deleted number word is not noise
/// either: `int four` is what `INT4` was heard as.
fn edge_noise(step: &AlignmentStep, leading: bool) -> bool {
    if step.edited.is_none() {
        return !leading && is_stopword(&step.original_word());
    }
    let kept = step.edited_word();
    let ordinary_lowercase = !kept.is_empty()
        && kept.chars().all(|c| c.is_lowercase() || c == '-')
        && kept
            .split('-')
            .all(|part| !part.is_empty() && data::is_ordinary_english(part));
    (leading && is_count(&kept)) || is_stopword(&kept) || ordinary_lowercase
}

/// The run without noise at its edges.
fn trim_run(steps: &[AlignmentStep], begin: usize, end: usize) -> (usize, usize) {
    let (mut first, mut last) = (begin, end);
    while first < last {
        let step = &steps[first];
        // A deleted stopword in front goes only together with a noise word
        // that absorbed it (`from scratch` to `from-scratch`).
        let absorbed = step.edited.is_none()
            && is_stopword(&step.original_word())
            && steps[first + 1].edited.is_some()
            && edge_noise(&steps[first + 1], true);
        if edge_noise(step, true) || absorbed {
            first += 1;
        } else {
            break;
        }
    }
    while last > first && edge_noise(&steps[last], false) {
        last -= 1;
    }
    (first, last)
}

/// Why a run is not a term, and whether its casing steps are released to
/// be mined alone (the run was words added or removed).
struct RunRejected {
    skipped: Skipped,
    release: bool,
}

/// The term and what was heard for `steps[begin..=end]`, or why not.
fn judge_run(
    steps: &[AlignmentStep],
    begin: usize,
    end: usize,
) -> Result<(String, String), RunRejected> {
    let run = &steps[begin..=end];
    let heard: Vec<String> = run
        .iter()
        .map(AlignmentStep::original_word)
        .filter(|w| !w.is_empty())
        .collect();
    let kept: Vec<String> = run
        .iter()
        .map(AlignmentStep::edited_word)
        .filter(|w| !w.is_empty())
        .collect();
    let heard_as = join_words(&heard);
    let term = join_words(&kept);
    let reject = |why: &str, release: bool| RunRejected {
        skipped: Skipped {
            term: term.clone(),
            heard_as: heard_as.clone(),
            why: why.to_string(),
        },
        release,
    };
    if heard.is_empty() {
        return Err(reject("inserted_words", false));
    }
    if kept.is_empty() {
        return Err(reject("deleted_words", false));
    }
    // Words typed in or taken out around ones that were heard right are an
    // edit of the message, not a mishearing.
    let lower =
        |words: &[String]| -> Vec<String> { words.iter().map(|w| w.to_lowercase()).collect() };
    let (heard_lower, kept_lower) = (lower(&heard), lower(&kept));
    let added = kept.len() > heard.len() && heard_lower.iter().all(|w| kept_lower.contains(w));
    let removed = heard.len() > kept.len() && kept_lower.iter().all(|w| heard_lower.contains(w));
    if added || removed {
        return Err(reject(
            if added {
                "added_words"
            } else {
                "removed_words"
            },
            true,
        ));
    }
    if kept.len() > MAX_TERM_WORDS {
        return Err(reject("too_long", false));
    }
    if is_formatting(&heard_as, &term) {
        return Err(reject("formatting", false));
    }
    let capitalized_mid_sentence = (begin..=end).any(|index| {
        let word = steps[index].edited_word();
        steps[index].edited.is_some()
            && word.chars().next().is_some_and(char::is_uppercase)
            && !is_stopword(&word)
            && !sentence_start(steps, index)
    });
    let starts_sentence = (begin..=end)
        .find(|index| steps[*index].edited.is_some())
        .is_some_and(|index| sentence_start(steps, index));
    if let Some(why) = rejection(&term, &heard_as, capitalized_mid_sentence, starts_sentence) {
        return Err(reject(why, false));
    }
    Ok((term, heard_as))
}

/// Mine each run for a term. Returns the casing steps of runs that turned
/// out to be added or removed words rather than a misheard phrase, so a
/// recased name inside one (`zainab` to `Zainab Ali`) is still offered alone.
fn mine_word_runs(
    steps: &[AlignmentStep],
    runs: &[(usize, usize)],
    dictionary: &DictionaryFile,
    suggestions: &mut Vec<Suggestion>,
    skipped: &mut Vec<Skipped>,
) -> Vec<usize> {
    let mut released = Vec::new();
    for &(begin, end) in runs {
        // The run without noise at its edges (`4x B300` is `B300`) when that
        // is a term; else the whole run, for a mishearing whose edge words
        // are part of it.
        let trimmed = trim_run(steps, begin, end);
        let core = &steps[trimmed.0..=trimmed.1];
        let verdict = if trimmed == (begin, end) {
            judge_run(steps, begin, end)
        } else if core.iter().any(|s| s.word.is_word_edit())
            && core.iter().any(|s| s.edited.is_some())
        {
            // Only a core that no longer sounds like what was heard falls
            // back to the whole run; a core rejected as formatting (`fifty
            // k` to `50k`) stays rejected.
            judge_run(steps, trimmed.0, trimmed.1).or_else(|core| {
                if core.skipped.why == "meaning_change" {
                    judge_run(steps, begin, end)
                } else {
                    Err(core)
                }
            })
        } else if core.iter().any(|s| s.word == Label::Casing) {
            // Only noise was rewritten (`from scratch` to `from-scratch`);
            // the recased words in the run are mined on their own.
            let heard: Vec<String> = steps[begin..=end]
                .iter()
                .map(AlignmentStep::original_word)
                .filter(|w| !w.is_empty())
                .collect();
            let kept: Vec<String> = steps[begin..=end]
                .iter()
                .map(AlignmentStep::edited_word)
                .filter(|w| !w.is_empty())
                .collect();
            Err(RunRejected {
                skipped: Skipped {
                    term: join_words(&kept),
                    heard_as: join_words(&heard),
                    why: "edge_noise".to_string(),
                },
                release: true,
            })
        } else {
            judge_run(steps, begin, end)
        };
        let (term, heard_as) = match verdict {
            Ok(found) => found,
            Err(rejected) => {
                if rejected.release {
                    released.extend((begin..=end).filter(|i| steps[*i].word == Label::Casing));
                }
                skipped.push(rejected.skipped);
                continue;
            }
        };
        let mut skip = |why: &str| {
            skipped.push(Skipped {
                term: term.clone(),
                heard_as: heard_as.clone(),
                why: why.to_string(),
            })
        };
        match dictionary_status(dictionary, &term, &heard_as) {
            DictionaryStatus::KnownForm => skip("known_spoken_form"),
            DictionaryStatus::Known => suggestions.push(Suggestion::DictionaryTerm {
                term: term.clone(),
                heard_as: heard_as.clone(),
                reason: TermReason::Misheard,
                existing_entry: true,
            }),
            DictionaryStatus::New => suggestions.push(Suggestion::DictionaryTerm {
                term: term.clone(),
                heard_as: heard_as.clone(),
                reason: TermReason::Misheard,
                existing_entry: false,
            }),
        }
    }
    released
}

/// Units and time suffixes the user typed onto a number (`3:30` to `3:30pm`,
/// `5` to `5k`) are formatting, not vocabulary.
fn is_formatting(heard: &str, term: &str) -> bool {
    let squash = |text: &str| -> String {
        text.chars()
            .filter(|c| c.is_alphanumeric() || *c == '%')
            .flat_map(char::to_lowercase)
            .collect()
    };
    let (heard, term) = (squash(heard), squash(term));
    if !heard.chars().any(|c| c.is_ascii_digit()) {
        return false;
    }
    let extra = term
        .strip_prefix(heard.as_str())
        .or_else(|| term.strip_suffix(heard.as_str()))
        .or_else(|| heard.strip_prefix(term.as_str()));
    matches!(
        extra,
        Some(
            "am" | "pm"
                | "st"
                | "nd"
                | "rd"
                | "th"
                | "k"
                | "m"
                | "b"
                | "ms"
                | "s"
                | "x"
                | "%"
                | "h"
                | "hr"
                | "min"
                | "gb"
                | "mb"
                | "tb"
                | "kb"
        )
    )
}

/// Why a mined phrase is not a dictionary term, or `None` when it is one.
/// Prefixes that a hyphen attaches to another word (`multi-GPU`).
const HYPHEN_PREFIXES: &[&str] = &[
    "multi", "single", "semi", "sub", "non", "pre", "post", "re", "co", "self", "cross", "inter",
    "intra", "high", "low", "mid", "ultra", "super", "anti", "auto", "de", "un", "over", "under",
    "half", "full", "dual", "quad", "tri", "bi", "mono", "poly", "pseudo", "quasi", "micro",
    "macro", "mini", "meta", "hyper", "on", "off", "in", "out", "well", "end",
];

/// `capitalized_mid_sentence`: a kept word that is not a stopword starts with
/// a capital where no sentence starts, the mark of a name.
/// `starts_sentence`: the term's first word starts a sentence, so its
/// capital says nothing.
fn rejection(
    term: &str,
    heard_as: &str,
    capitalized_mid_sentence: bool,
    starts_sentence: bool,
) -> Option<&'static str> {
    let words: Vec<&str> = term.split_whitespace().collect();
    if words.iter().all(|w| is_stopword(w)) {
        return Some("stopwords");
    }
    // `is good When` for `like when`: a reworded clause, not a name.
    if words.iter().filter(|w| is_stopword(w)).count() * 2 > words.len() {
        return Some("mostly_stopwords");
    }
    // `high level` to `high-level`, `like multi GPU` to `multi-GPU`, `NCU
    // level` to `NCU-level`: a hyphen joining heard words (fillers aside)
    // where one side is an ordinary word, a prefix or a number. Two names
    // joined (`TensorRT-LLM`, `GLM-5.3`) are the term's own spelling.
    // Case must be unchanged: `gpt 4` to `GPT-4` or `wi fi` to `Wi-Fi` is the
    // term's spelling, and a number on one side (`GPT-4`, `COVID-19`) never
    // makes a hyphen ordinary.
    let heard_content = heard_as
        .split_whitespace()
        .filter(|w| !is_stopword(w))
        .collect::<Vec<_>>()
        .join(" ");
    if term.replace('-', " ") == heard_content
        && term.contains('-')
        && term.split(['-', ' ']).enumerate().any(|(index, part)| {
            // Lowercase only: `Auto-GPT` is a name, `auto-scaling` is not. A
            // capital that only starts the sentence (`Multi-GPU servers`)
            // does not count.
            let part = if index == 0 && starts_sentence {
                let mut chars = part.chars();
                chars
                    .next()
                    .map(|c| c.to_lowercase().chain(chars).collect::<String>())
                    .unwrap_or_default()
            } else {
                part.to_string()
            };
            let lowercase =
                part.chars().all(char::is_alphabetic) && part.chars().all(char::is_lowercase);
            lowercase
                && (HYPHEN_PREFIXES.contains(&part.as_str())
                    || (part.chars().count() > 2 && data::is_ordinary_english(&part)))
        })
    {
        return Some("hyphenation");
    }
    // `2.5 G` to `2.5G`: only spaces removed. Not when the join makes a
    // name: `Git Hub` to `GitHub`, `Open AI` to `OpenAI`, `H 100` to `H100`.
    // A prefix joined on (`sub project`, `full screen`) is still spacing.
    let prefixed = heard_as
        .split_whitespace()
        .next()
        .is_some_and(|first| HYPHEN_PREFIXES.contains(&first.to_lowercase().as_str()));
    if term.contains(' ') != heard_as.contains(' ')
        && term.replace(' ', "") == heard_as.replace(' ', "")
        && !term
            .split_whitespace()
            .any(|word| joined_name(word) || (!prefixed && joined_compound(word)))
    {
        return Some("spacing");
    }
    // `on and off` to `on/off`, `src slash core` to `src/Core`,
    // `three hundred so if` to `300 If`: every word kept was already heard,
    // or is a number, a stopword or a single letter.
    let heard_words: Vec<String> = spoken_digits(heard_as)
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect();
    let nothing_new = term
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .all(|word| {
            let lower = word.to_lowercase();
            word.chars().count() <= 1
                || word.chars().all(|c| c.is_ascii_digit())
                || is_stopword(word)
                || heard_words.contains(&lower)
        });
    // `TensorRT-LLM`, `GLM-5.3`, `Auto-GPT`: names (or capitalized words)
    // joined by a hyphen, at least one of them a term.
    let name_part = |part: &str| term_like(part) || part.chars().any(|c| c.is_ascii_digit());
    let capitalized = |part: &str| part.chars().next().is_some_and(char::is_uppercase);
    let joins_names = term.split('-').count() > 1
        && term
            .split('-')
            .all(|part| name_part(part) || capitalized(part))
        && term.split('-').any(name_part);
    // `Node JS` to `Node.js`: a dot joining heard words is the term's own
    // spelling; a dot the user said (`example dot com`) is formatting.
    let said_dot = heard_as
        .split_whitespace()
        .any(|w| matches!(w.to_lowercase().as_str(), "dot" | "point"));
    let dotted_join = !said_dot
        && term.split_whitespace().any(|word| {
            let chars: Vec<char> = trim_edge_punct(word).chars().collect();
            chars
                .windows(3)
                .any(|w| w[0].is_alphanumeric() && w[1] == '.' && w[2].is_alphabetic())
        })
        && term
            .replace('.', " ")
            .to_lowercase()
            .split_whitespace()
            .eq(heard_as.to_lowercase().split_whitespace());
    // A capital the user added is a name signal; one they removed is not.
    let heard_tokens: Vec<&str> = heard_as.split(|c: char| !c.is_alphanumeric()).collect();
    let recased = term
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| {
            word.chars().count() > 1
                && !word.chars().all(|c| c.is_ascii_digit())
                && !is_stopword(word)
        })
        .any(|word| {
            let capitals = word.chars().filter(|c| c.is_uppercase()).count();
            !heard_tokens.iter().any(|heard| {
                heard.to_lowercase() == word.to_lowercase()
                    && heard.chars().filter(|c| c.is_uppercase()).count() >= capitals
            })
        });
    if nothing_new && !joins_names && !dotted_join && !recased {
        return Some("no_new_word");
    }
    let digits = term.chars().filter(char::is_ascii_digit).count();
    // A phone number or an id is not vocabulary.
    if digits >= 5 {
        return Some("number");
    }
    // `fourth` to `4th`, `16 by 16` to `16x16`: an ordinal or a size is
    // how a number is written, not a name.
    let ordinal_or_size = !words.is_empty()
        && words.iter().all(|word| {
            let core = trim_edge_punct(word).to_lowercase();
            let digits_then = |suffixes: &[&str]| {
                suffixes.iter().any(|suffix| {
                    core.strip_suffix(suffix)
                        .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
                })
            };
            let size = core.split_once(['x', '×']).is_some_and(|(a, b)| {
                !a.is_empty()
                    && !b.is_empty()
                    && a.chars().all(|c| c.is_ascii_digit())
                    && b.chars().all(|c| c.is_ascii_digit())
            });
            digits_then(&["st", "nd", "rd", "th"]) || size || is_count(&core)
        });
    if ordinal_or_size {
        return Some("number_formatting");
    }
    // `fifteen` to `15`, `eight K` to `8K`, `five point three` to `5.3`.
    let squash = |text: &str| -> String {
        text.chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect()
    };
    let said_a_number = spoken_digits(heard_as) != data::normalize(heard_as);
    if digits > 0 && said_a_number && squash(&spoken_digits(heard_as)) == squash(term) {
        // Same letters in the same case: only the number was written as
        // digits. A recased word around it (`llama three` to `Llama 3`) is
        // a name, which stays a term.
        let letters =
            |text: &str| -> String { text.chars().filter(|c| c.is_alphabetic()).collect() };
        let spoken = heard_as
            .split_whitespace()
            .filter(|w| {
                !spoken_digits(w)
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == '.')
            })
            .collect::<Vec<_>>()
            .join(" ");
        if letters(&spoken) == letters(term) {
            return Some("number_formatting");
        }
        // A number with a unit letter after it (`three hundred k` to
        // `300K`, `eight x` to `8x`), whatever its case.
        let unit_after_digits = term.split_whitespace().all(|word| {
            let core = trim_edge_punct(word);
            let unit: String = core.chars().skip_while(|c| !c.is_alphabetic()).collect();
            core.chars().next().is_some_and(|c| c.is_ascii_digit())
                && unit.chars().count() <= 2
                && unit.chars().all(char::is_alphabetic)
        });
        if unit_after_digits {
            return Some("number_formatting");
        }
    }
    // `gigabytes` to `GB`, `ten gigabytes` to `10 GB`.
    let non_numeric: Vec<String> = words
        .iter()
        .map(|w| trim_edge_punct(w))
        .filter(|w| {
            !w.chars()
                .all(|c| c.is_ascii_digit() || c == '.' || c == ',')
        })
        .collect();
    if !non_numeric.is_empty() && non_numeric.iter().all(|w| UNITS.contains(&w.as_str())) {
        return Some("unit");
    }
    if term.chars().any(is_cjk) && !term.contains(char::is_whitespace) {
        return Some("cjk_without_spaces");
    }
    if words.iter().any(|w| is_contraction(w)) {
        return Some("contraction");
    }
    if !term_like(term) && !capitalized_mid_sentence {
        return Some("ordinary_words");
    }
    // `demon` to `demo`, `speed` to `speedup`: one ordinary word respelled
    // into a close plain lowercase word is a typo fix or an inflection, not
    // a name the recognizer missed.
    let plain_lowercase = |word: &str| word.chars().all(|c| c.is_alphabetic() && c.is_lowercase());
    if !term.contains(char::is_whitespace)
        && !heard_as.contains(char::is_whitespace)
        && plain_lowercase(term)
        && plain_lowercase(heard_as)
        && data::is_ordinary_english(heard_as)
        && data::levenshtein(heard_as, term) <= 2
    {
        return Some("spelling");
    }
    if !sounds_alike(heard_as, term) {
        return Some("meaning_change");
    }
    None
}

/// A word that reads as a name once written as one: a capital right after
/// a lowercase letter (`GitHub`, `OpenAI`) or letters then digits (`H100`).
/// A number with a unit glued on (`2.5G`) is not one, and neither are
/// spelled letters joined (`X Y Z` to `XYZ`, which the Wispr replay showed
/// to be formatting: one more proposal, none in the dictionary).
fn joined_name(word: &str) -> bool {
    let chars: Vec<char> = trim_edge_punct(word).chars().collect();
    let camel = chars
        .windows(2)
        .any(|w| w[0].is_lowercase() && w[1].is_uppercase());
    let letter_led =
        chars.first().is_some_and(|c| c.is_alphabetic()) && chars.iter().any(char::is_ascii_digit);
    camel || letter_led
}

/// A long lowercase word that is not ordinary English, written as one
/// (`thread block` to `threadblock`): the field's own compound. Ten letters
/// at least: shorter joins the word list lacks are mostly everyday
/// compounds (`database`, `codebase`, `username`).
fn joined_compound(word: &str) -> bool {
    let core = trim_edge_punct(word);
    core.chars().count() >= 10
        && core.chars().all(|c| c.is_alphabetic() && c.is_lowercase())
        && !data::is_ordinary_english(&core)
}

fn is_contraction(word: &str) -> bool {
    // `O'Brien` and `D'Angelo` are names, not contractions.
    let chars: Vec<char> = word.chars().collect();
    let name_like = chars.len() >= 3
        && matches!(chars[1], '\'' | '\u{2018}' | '\u{2019}')
        && chars[2].is_uppercase()
        || chars.len() >= 2
            && matches!(chars[0], '\'' | '\u{2018}' | '\u{2019}')
            && chars[1].is_uppercase();
    if name_like {
        return false;
    }
    let lower = word.to_lowercase().replace(['\u{2018}', '\u{2019}'], "'");
    let Some(apostrophe) = lower.rfind('\'') else {
        return false;
    };
    matches!(
        &lower[apostrophe + 1..],
        "" | "s" | "t" | "ll" | "ve" | "re" | "d" | "m"
    )
}

/// The local stand-in for Wispr's `extractEditedWords` classifier: a word is
/// a term when it carries digits, inner capitals, all capitals, inner
/// punctuation (`Node.js`, `GPT-5`, `C++`) or is not ordinary English.
fn term_like(term: &str) -> bool {
    term.split_whitespace().any(|word| {
        let core = trim_edge_punct(word);
        let chars: Vec<char> = core.chars().collect();
        if chars.is_empty() {
            return false;
        }
        let letters: Vec<char> = chars
            .iter()
            .copied()
            .filter(|c| c.is_alphabetic())
            .collect();
        let inner_capital = chars.iter().skip(1).any(|c| c.is_uppercase());
        let all_capitals = letters.len() >= 2 && letters.iter().all(|c| c.is_uppercase());
        // Letters and digits in one token (`H100`, `GPT-4o`, `ZetaFlow2`); a
        // plain number is formatting, not a term.
        let digits = !letters.is_empty() && chars.iter().any(char::is_ascii_digit);
        // `high-level` is two ordinary words with a hyphen; `Node.js`,
        // `C++` and `k8s/arm` are terms.
        let hyphenated_words = core.contains('-')
            && core
                .split('-')
                .all(|part| !part.is_empty() && data::is_ordinary_english(&part.to_lowercase()));
        let inner_punct = !hyphenated_words
            && (chars.len() > 2
                && chars[1..chars.len() - 1]
                    .iter()
                    .any(|c| matches!(c, '.' | '-' | '_' | '+' | '#' | '/'))
                || word.ends_with("++")
                || word.ends_with('#') && chars.len() <= 3);
        let foreign = letters.len() == chars.len()
            && letters.len() > 2
            && !data::is_ordinary_english(&core.to_lowercase());
        inner_capital || all_capitals || digits || inner_punct || foreign
    })
}

const NUMBER_WORDS: [(&str, &str); 28] = [
    ("zero", "0"),
    ("one", "1"),
    ("two", "2"),
    ("three", "3"),
    ("four", "4"),
    ("five", "5"),
    ("six", "6"),
    ("seven", "7"),
    ("eight", "8"),
    ("nine", "9"),
    ("ten", "10"),
    ("eleven", "11"),
    ("twelve", "12"),
    ("thirteen", "13"),
    ("fourteen", "14"),
    ("fifteen", "15"),
    ("sixteen", "16"),
    ("seventeen", "17"),
    ("eighteen", "18"),
    ("nineteen", "19"),
    ("twenty", "20"),
    ("thirty", "30"),
    ("forty", "40"),
    ("fifty", "50"),
    ("sixty", "60"),
    ("seventy", "70"),
    ("eighty", "80"),
    ("ninety", "90"),
];

/// Spoken numbers as digits: `three` 3, `three hundred` 300, `twenty five`
/// 25, `two thousand twenty six` 2026, `five point three` 5 . 3.
fn spoken_digits(text: &str) -> String {
    let value_of = |word: &str| -> Option<u64> {
        NUMBER_WORDS
            .iter()
            .find(|(spoken, _)| *spoken == word)
            .and_then(|(_, digits)| digits.parse().ok())
    };
    let mut out: Vec<String> = Vec::new();
    // (total so far, current group below a thousand)
    let mut number: Option<(u64, u64)> = None;
    let flush = |number: &mut Option<(u64, u64)>, out: &mut Vec<String>| {
        if let Some((total, group)) = number.take() {
            out.push((total + group).to_string());
        }
    };
    for word in data::normalize(text).split_whitespace() {
        match (word, value_of(word)) {
            (_, Some(value)) => {
                // `twenty five`, `three hundred five` and `two thousand
                // twenty` add up; a second digit (`seven eight`, a phone
                // number) starts a new number.
                let adds = number.is_some_and(|(total, group)| {
                    (group == 0 && total > 0)
                        || (group >= 100 && group % 100 == 0 && value < 100)
                        || (group % 100 >= 20 && group % 10 == 0 && value < 10)
                });
                if number.is_some() && !adds {
                    flush(&mut number, &mut out);
                    number = Some((0, value));
                } else {
                    let (total, group) = number.unwrap_or((0, 0));
                    number = Some((total, group + value));
                }
            }
            ("hundred", None) if number.is_some() => {
                let (total, group) = number.unwrap_or((0, 0));
                number = Some((total, group.max(1) * 100));
            }
            ("thousand", None) if number.is_some() => {
                let (total, group) = number.unwrap_or((0, 0));
                number = Some((total + group.max(1) * 1_000, 0));
            }
            ("point" | "dot", None) => {
                flush(&mut number, &mut out);
                out.push(".".into());
            }
            _ => {
                flush(&mut number, &mut out);
                out.push(word.to_string());
            }
        }
    }
    flush(&mut number, &mut out);
    out.join(" ")
}

/// Unit abbreviations a user types for a spoken unit (`gigabytes` to `GB`).
const UNITS: &[&str] = &[
    "GB", "Gb", "MB", "Mb", "KB", "kb", "TB", "Tb", "PB", "ms", "us", "ns", "Hz", "kHz", "MHz",
    "GHz", "fps", "FPS", "px", "pt", "km", "kg", "mm", "cm", "ml", "mph", "kph", "GiB", "MiB",
    "TiB", "KiB", "Gbps", "Mbps", "W", "kW", "V", "mAh",
];

/// A misheard term and its correction share most of their consonant
/// skeleton; a change of mind (`Postgres` to `SQLite`) does not.
fn sounds_alike(heard: &str, term: &str) -> bool {
    let a = data::phonetic_key(&spoken_digits(heard));
    let b = data::phonetic_key(&spoken_digits(term));
    if a.is_empty() || b.is_empty() {
        return false;
    }
    let longest = a.chars().count().max(b.chars().count());
    if longest <= 2 {
        return a == b;
    }
    let distance = data::levenshtein(&a, &b);
    distance <= 1 || distance * 2 <= longest
}

enum DictionaryStatus {
    New,
    Known,
    KnownForm,
}

fn find_entry<'a>(dictionary: &'a DictionaryFile, term: &str) -> Option<&'a DictionaryEntry> {
    let wanted = term.to_lowercase();
    dictionary.entries.iter().find(|entry| {
        entry.canonical().to_lowercase() == wanted || entry.phrase.to_lowercase() == wanted
    })
}

fn dictionary_status(dictionary: &DictionaryFile, term: &str, heard_as: &str) -> DictionaryStatus {
    let Some(entry) = find_entry(dictionary, term) else {
        return DictionaryStatus::New;
    };
    let heard = data::normalize(heard_as);
    if entry.forms().any(|form| data::normalize(form) == heard) {
        DictionaryStatus::KnownForm
    } else {
        DictionaryStatus::Known
    }
}

/// Products, languages and models whose names are ordinary words: given a
/// capital mid-sentence by the user they are the name, not a title.
const NAMES_THAT_ARE_WORDS: &[&str] = &[
    "rust", "swift", "bun", "slack", "notion", "linear", "discord", "signal", "zoom", "chrome",
    "safari", "arc", "cursor", "python", "ruby", "julia", "dart", "elixir", "java", "spark",
    "flask", "rails", "react", "angular", "node", "metal", "vision", "triton", "modal", "docker",
    "excel", "outlook", "opus", "sonnet", "haiku", "gemini", "copilot", "llama", "mistral",
    "falcon", "obsidian", "homebrew", "pandas", "keynote", "codex",
];

fn mine_casing(
    steps: &[AlignmentStep],
    indices: &[usize],
    dictionary: &DictionaryFile,
    suggestions: &mut Vec<Suggestion>,
    skipped: &mut Vec<Skipped>,
) {
    for &index in indices {
        let step = &steps[index];
        let heard = step.original_word();
        let kept = step.edited_word();
        let skip = |skipped: &mut Vec<Skipped>, why: &str| {
            skipped.push(Skipped {
                term: kept.clone(),
                heard_as: heard.clone(),
                why: why.to_string(),
            })
        };
        let kept_chars: Vec<char> = kept.chars().collect();
        let inner_capital = kept_chars.iter().skip(1).any(|c| c.is_uppercase());
        let letters: Vec<char> = kept_chars
            .iter()
            .copied()
            .filter(|c| c.is_alphabetic())
            .collect();
        let all_capitals = letters.len() >= 2 && letters.iter().all(|c| c.is_uppercase());
        let starts_upper = kept_chars.first().is_some_and(|c| c.is_uppercase());
        let heard_starts_upper = heard.chars().next().is_some_and(char::is_uppercase);
        let first_letter_only = heard.chars().skip(1).eq(kept.chars().skip(1));
        if first_letter_only && sentence_start(steps, index) {
            // Sentence capitalization is style or the field's own
            // autocapitalization (`cuBLAS` to `CuBLAS`), never a term.
            skip(skipped, "sentence_case");
            continue;
        }
        if first_letter_only && (is_stopword(&kept) || is_contraction(&kept)) {
            // `we` to `We`, `it's` to `It's` after a sentence break the
            // alignment did not see.
            skip(skipped, "stopwords");
            continue;
        }
        let term = inner_capital || all_capitals || (starts_upper && !heard_starts_upper);
        if !term {
            skip(skipped, "lowercased_word");
            continue;
        }
        // A capitalized word right after a capitalized name is part of it:
        // `Claude code` to `Claude Code` is the term `Claude Code`.
        let previous = steps[..index].iter().rev().find(|s| s.edited.is_some());
        let joins_name = first_letter_only
            && previous.is_some_and(|s| {
                s.word == Label::Match
                    && s.edited_word()
                        .chars()
                        .next()
                        .is_some_and(char::is_uppercase)
                    && !is_stopword(&s.edited_word())
                    && s.edited
                        .as_deref()
                        .is_some_and(|t| !t.ends_with(['.', '!', '?', ':']))
            });
        let (kept, heard) = if joins_name {
            let name = previous.map(AlignmentStep::edited_word).unwrap_or_default();
            (format!("{name} {kept}"), format!("{name} {heard}"))
        } else {
            (kept.clone(), heard.clone())
        };
        match find_entry(dictionary, &kept) {
            Some(entry) if entry.canonical() == kept => {
                skip(skipped, "known_term");
                continue;
            }
            Some(_) => {
                skip(skipped, "known_term_other_casing");
                continue;
            }
            None => {}
        }
        // One ordinary word given a capital (`time` to `Time`) is a title or
        // the formatter's habit far more often than a name, unless it is a
        // known product or language name (`rust` to `Rust`) or a first name
        // (`grant` to `Grant`). The same capitalization made twice is a name
        // too (`RepeatedEdits`).
        let lower = kept.to_lowercase();
        if first_letter_only
            && !joins_name
            && data::is_ordinary_english(&lower)
            && !NAMES_THAT_ARE_WORDS.contains(&lower.as_str())
            && !data::is_proper_name(&kept)
        {
            skip(skipped, "common_word");
            continue;
        }
        suggestions.push(Suggestion::DictionaryTerm {
            term: kept.clone(),
            heard_as: heard.clone(),
            reason: TermReason::Casing,
            existing_entry: false,
        });
    }
}

fn mine_style(
    steps: &[AlignmentStep],
    first: usize,
    last: usize,
    any_word_edit: bool,
    appended: bool,
    suggestions: &mut Vec<Suggestion>,
) {
    let core = &steps[first..=last];
    let lowered = core
        .iter()
        .filter(|s| {
            s.word == Label::Casing
                && s.original_word().chars().any(char::is_uppercase)
                && !s.edited_word().chars().any(char::is_uppercase)
        })
        .count();
    let kept_text: String = core
        .iter()
        .filter_map(|s| s.edited.as_deref())
        .collect::<Vec<_>>()
        .join(" ");
    if !any_word_edit && lowered >= 2 && !kept_text.chars().any(char::is_uppercase) {
        suggestions.push(Suggestion::Style {
            preference: StylePreference::AllLowercase,
        });
    } else if first == 0 && steps[0].word == Label::Casing {
        let heard = steps[0].original_word();
        let kept = steps[0].edited_word();
        // A plain word only: lowercasing `App.slack.com` or `IPhone` fixes
        // the term, it says nothing about how the user starts messages.
        let plain = !kept.is_empty() && kept.chars().all(char::is_alphabetic);
        let lowered_first = plain
            && heard.chars().next().is_some_and(char::is_uppercase)
            && kept.chars().next().is_some_and(char::is_lowercase)
            && heard.chars().skip(1).eq(kept.chars().skip(1));
        if lowered_first {
            suggestions.push(Suggestion::Style {
                preference: StylePreference::LowercaseStart,
            });
        }
    }
    let final_step = &steps[last];
    if appended {
        return;
    }
    if let (Some(from), Some(to)) = (final_step.original.as_deref(), final_step.edited.as_deref()) {
        let words_kept = matches!(final_step.word, Label::Match | Label::Casing);
        if words_kept && from.ends_with('.') && !from.ends_with("..") && !to.ends_with('.') {
            suggestions.push(Suggestion::Style {
                preference: StylePreference::NoTrailingPeriod,
            });
        }
    }
}

pub const EDIT_RECORD_SCHEMA_VERSION: u32 = 1;

/// End reasons whose captured region is not the user's text. Wispr's
/// `paste-observation-registry` excludes the first four from learning; the
/// rest end before anything was captured.
pub fn end_reason_is_learnable(reason: &str) -> bool {
    !matches!(
        reason,
        "anchor_mismatch"
            | "transcript_region_removed"
            | "internal"
            | "internal_error"
            | "dictated_text_not_found"
            | "textbox_too_long"
            | "secure_field"
    )
}

/// Capitalizations of ordinary words (`mercury` to `Mercury`) the user made,
/// counted in memory for this engine session only (never written to disk).
/// One is a title or a habit; the same one twice is a name.
#[derive(Debug, Default)]
pub struct RepeatedEdits {
    seen: std::collections::HashMap<(String, String), u32>,
}

impl RepeatedEdits {
    /// Distinct capitalizations remembered before the count starts over.
    const LIMIT: usize = 1_000;

    /// Count this observation's `common_word` capitalizations and propose
    /// the ones made for the second time.
    pub fn promote(&mut self, analysis: &mut EditAnalysis) {
        if self.seen.len() >= Self::LIMIT {
            self.seen.clear();
        }
        let mut kept = Vec::new();
        // One edit counts once, however often it capitalized the word.
        let mut counted = std::collections::HashSet::new();
        for skipped in std::mem::take(&mut analysis.skipped) {
            if skipped.why != "common_word" {
                kept.push(skipped);
                continue;
            }
            let key = (skipped.term.clone(), skipped.heard_as.to_lowercase());
            if !counted.insert(key.clone()) {
                kept.push(skipped);
                continue;
            }
            let count = self.seen.entry(key).or_insert(0);
            *count += 1;
            if *count == 2 {
                analysis.suggestions.push(Suggestion::DictionaryTerm {
                    term: skipped.term,
                    heard_as: skipped.heard_as,
                    reason: TermReason::Casing,
                    existing_entry: false,
                });
            } else {
                kept.push(skipped);
            }
        }
        analysis.skipped = kept;
    }
}

/// The `edit_observed` request the bar sends when it stops watching a field.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct ObservedEdit {
    #[serde(default)]
    pub pasted: String,
    #[serde(default)]
    pub edited: String,
    #[serde(default)]
    pub edited_within_threshold: Option<String>,
    #[serde(default)]
    pub end_reason: String,
    #[serde(default)]
    pub last_keystroke: Option<String>,
    #[serde(default)]
    pub last_keystroke_age_ms: Option<u64>,
    #[serde(default)]
    pub observed_ms: u64,
    /// Local history is on, so the record may go into this dictation's
    /// corpus entry. The bar never sets it otherwise.
    #[serde(default)]
    pub record: bool,
}

impl ObservedEdit {
    pub fn analyze(&self, dictionary: &DictionaryFile) -> EditAnalysis {
        let mut analysis = analyze(
            &self.pasted,
            &self.edited,
            self.edited_within_threshold.as_deref(),
            dictionary,
        );
        if !end_reason_is_learnable(&self.end_reason) {
            analysis.suggestions.clear();
        }
        analysis
    }

    pub fn record(&self, analysis: &EditAnalysis) -> UserEditRecord {
        UserEditRecord {
            schema_version: EDIT_RECORD_SCHEMA_VERSION,
            pasted_text: self.pasted.clone(),
            edited_text: dictation_region(
                &self.edited,
                analysis.prepended_tokens,
                analysis.appended_tokens,
            ),
            appended_words: analysis.appended_words,
            skipped: analysis.skipped.iter().take(8).cloned().collect(),
            edited_text_within_threshold: self
                .edited_within_threshold
                .clone()
                .filter(|text| text != &self.edited),
            outcome: analysis.outcome,
            word_edits: analysis.word_edits.clone(),
            punctuation_edits: analysis.punctuation_edits.clone(),
            edit_distance: analysis.edit_distance,
            end_reason: self.end_reason.clone(),
            end_last_keystroke: self.last_keystroke.clone(),
            end_last_keystroke_age_ms: self.last_keystroke_age_ms,
            observed_ms: self.observed_ms,
            suggestions: analysis
                .suggestions
                .iter()
                .map(|suggestion| SuggestionRecord {
                    suggestion: suggestion.clone(),
                    decision: "pending".into(),
                })
                .collect(),
        }
    }
}

/// What the user did to one dictation after Phonon pasted it: the pair the
/// SPEC's fine-tuning plan needs (what Phonon wrote, what the user kept).
/// Stored as `user_edit` in that dictation's own corpus `metadata.json` and
/// nowhere else, and only when local history is on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserEditRecord {
    pub schema_version: u32,
    pub pasted_text: String,
    /// The dictation as the user left it. Text they typed before or after it
    /// is not kept here, only counted in `appended_words`.
    pub edited_text: String,
    #[serde(default)]
    pub appended_words: usize,
    /// Candidates considered and why each was not proposed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped: Vec<Skipped>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edited_text_within_threshold: Option<String>,
    pub outcome: EditOutcome,
    pub word_edits: String,
    pub punctuation_edits: String,
    pub edit_distance: f64,
    /// Why the bar stopped watching (Wispr's `contentObservationEndReason`).
    pub end_reason: String,
    /// Category of the last key within two seconds of the end, never the key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_last_keystroke: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_last_keystroke_age_ms: Option<u64>,
    pub observed_ms: u64,
    #[serde(default)]
    pub suggestions: Vec<SuggestionRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuggestionRecord {
    #[serde(flatten)]
    pub suggestion: Suggestion,
    /// `pending`, `approved`, `rejected` or `undone`.
    pub decision: String,
}

/// Write the record into an entry that already exists. With local history
/// off the bar deletes the entry at insertion; a late request must not bring
/// it back, so a missing entry is `Ok(false)`, never a new file.
pub fn write_record(metadata_path: &Path, record: &UserEditRecord) -> Result<bool> {
    if !metadata_path.is_file() {
        return Ok(false);
    }
    let mut metadata = data::load_recording(metadata_path)?;
    metadata.user_edit = Some(record.clone());
    data::write_json_atomic(metadata_path, &metadata)?;
    Ok(true)
}

/// Note the user's answer to one proposal in the dictation's record.
pub fn write_decision(metadata_path: &Path, index: usize, decision: &str) -> Result<bool> {
    if !matches!(decision, "approved" | "rejected" | "undone") || !metadata_path.is_file() {
        return Ok(false);
    }
    let mut metadata = data::load_recording(metadata_path)?;
    let Some(suggestion) = metadata
        .user_edit
        .as_mut()
        .and_then(|record| record.suggestions.get_mut(index))
    else {
        return Ok(false);
    };
    suggestion.decision = decision.to_string();
    data::write_json_atomic(metadata_path, &metadata)?;
    Ok(true)
}

/// The engine's reply to `edit_observed`.
pub fn observed_event(id: Option<&str>, analysis: &EditAnalysis, recorded: bool) -> Value {
    json!({
        "type": "result",
        "kind": "edit",
        "id": id,
        "outcome": analysis.outcome,
        "word_edits": analysis.word_edits,
        "punctuation_edits": analysis.punctuation_edits,
        "edit_distance": analysis.edit_distance,
        "appended_words": analysis.appended_words,
        "suggestions": analysis.suggestions,
        "recorded": recorded,
    })
}

/// Function words that are never dictionary terms on their own.
const STOPWORDS: &str = "a about above after again against all am an and any are as at be \
    because been before being below between both but by can could did do does doing down \
    during each few for from further had has have having he her here hers herself him himself \
    his how i if in into is it its itself just me more most my myself no nor not now of off on \
    once only or other our ours ourselves out over own same she should so some such than that \
    the their theirs them themselves then there these they this those through to too under \
    until up very was we were what when where which while who whom why will with would you \
    your yours yourself yourselves yes yeah ok okay um uh like well oh";

fn is_stopword(word: &str) -> bool {
    let lower = word.to_lowercase();
    STOPWORDS.split_whitespace().any(|stop| stop == lower)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DictionaryEntry, DictionaryFile};

    fn empty() -> DictionaryFile {
        DictionaryFile::default()
    }

    fn with_entry(phrase: &str, spoken: &[&str]) -> DictionaryFile {
        DictionaryFile {
            entries: vec![DictionaryEntry {
                phrase: phrase.into(),
                replacement: None,
                spoken_forms: spoken.iter().map(|s| s.to_string()).collect(),
                source: "test".into(),
                starred: false,
                usage_count: 0,
            }],
            ..Default::default()
        }
    }

    fn terms(analysis: &EditAnalysis) -> Vec<(String, String, TermReason)> {
        analysis
            .suggestions
            .iter()
            .filter_map(|s| match s {
                Suggestion::DictionaryTerm {
                    term,
                    heard_as,
                    reason,
                    ..
                } => Some((term.clone(), heard_as.clone(), *reason)),
                Suggestion::Style { .. } => None,
            })
            .collect()
    }

    fn styles(analysis: &EditAnalysis) -> Vec<StylePreference> {
        analysis
            .suggestions
            .iter()
            .filter_map(|s| match s {
                Suggestion::Style { preference } => Some(*preference),
                Suggestion::DictionaryTerm { .. } => None,
            })
            .collect()
    }

    fn skipped_why(analysis: &EditAnalysis) -> Vec<String> {
        analysis.skipped.iter().map(|s| s.why.clone()).collect()
    }

    // --- alignment, ported behaviour ------------------------------------

    #[test]
    fn tokenizer_splits_cjk_per_character_and_keeps_latin_words() {
        assert_eq!(tokenize("hello  world\n"), vec!["hello", "world"]);
        assert_eq!(tokenize("用cuBLAS。"), vec!["用", "cuBLAS", "。"]);
        assert_eq!(tokenize("東京タワー"), vec!["東", "京", "タ", "ワ", "ー"]);
    }

    #[test]
    fn vectors_label_match_casing_substitution_insert_delete() {
        let steps = align("we use coo blast today", "We use cuBLAS today");
        let (words, _) = edit_vectors(&steps);
        assert_eq!(words, "CMDSM");
        assert_eq!(steps[3].original_word(), "blast");
        assert_eq!(steps[3].edited_word(), "cuBLAS");

        let (words, punct) = edit_vectors(&align("Ship it.", "Ship it now."));
        assert_eq!((words.as_str(), punct.as_str()), ("MMI", "ZDI"));
    }

    /// Review question: a left/up tie takes the diagonal even when the
    /// diagonal is not the cheapest path. That is Wispr's backtrack too
    /// (text-diff-ops.js 165-178), so the vectors stay comparable.
    #[test]
    fn a_tie_takes_the_substitution_like_wispr() {
        let (words, _) = edit_vectors(&align("foo", "bar"));
        assert_eq!(words, "S");
        let (words, _) = edit_vectors(&align("keep foo here", "keep bar here"));
        assert_eq!(words, "MSM");
    }

    #[test]
    fn half_typed_last_word_is_a_capture_error() {
        let steps = align("Send the report today", "Send the report t");
        let (words, _) = edit_vectors(&steps);
        assert_eq!(words, "MMME");
        let analysis = analyze("Send the report today", "Send the report t", None, &empty());
        assert_eq!(analysis.outcome, EditOutcome::CaptureError);
        assert!(analysis.suggestions.is_empty());
    }

    #[test]
    fn trailing_punctuation_after_a_wordless_token_is_a_capture_error() {
        let steps = vec![
            AlignmentStep::same("ok"),
            AlignmentStep {
                word: Label::None,
                punct: Label::Insert,
                original: None,
                line_start: false,
                edited: Some("!".into()),
            },
        ];
        // Wispr's rule needs `[MS]I$` on punctuation and `Z` last on words.
        let mut steps = steps;
        steps[0].punct = Label::Match;
        assert_eq!(edit_vectors(&steps), ("MZ".into(), "ME".into()));
    }

    // --- the cases the user asked for ----------------------------------

    #[test]
    fn paste_then_retype_proposes_the_term_with_what_was_heard() {
        let pasted = "we benchmark coo blast against cutlass";
        let edited = "we benchmark cuBLAS against CUTLASS";
        let analysis = analyze(pasted, edited, None, &empty());
        assert_eq!(analysis.outcome, EditOutcome::Correction);
        let found = terms(&analysis);
        assert!(
            found.contains(&("cuBLAS".into(), "coo blast".into(), TermReason::Misheard)),
            "{found:?}"
        );
        assert!(
            found.contains(&("CUTLASS".into(), "cutlass".into(), TermReason::Casing)),
            "{found:?}"
        );
        assert!(analysis.word_edits.contains('S'));
    }

    #[test]
    fn retyping_the_whole_dictation_still_isolates_the_term() {
        // Select-all and retype produces the same final text; only the final
        // region matters, so the analysis is the same as a surgical fix.
        let analysis = analyze(
            "ask clod code to review it",
            "ask Claude Code to review it",
            None,
            &empty(),
        );
        assert_eq!(
            terms(&analysis),
            vec![(
                "Claude Code".into(),
                "clod code".into(),
                TermReason::Misheard
            )]
        );
    }

    #[test]
    fn autocorrect_noise_is_not_learned() {
        // Smart quotes and dashes from the field.
        let typographic = analyze(
            "it's done -- ship it",
            "it\u{2019}s done \u{2014} ship it",
            None,
            &empty(),
        );
        assert!(matches!(
            typographic.outcome,
            EditOutcome::Typographic | EditOutcome::Style
        ));
        assert!(typographic.suggestions.is_empty(), "{typographic:?}");
        // The field capitalized the first letter on its own.
        let capitalized = analyze("cuBLAS is fast", "CuBLAS is fast", None, &empty());
        assert!(terms(&capitalized).is_empty(), "{capitalized:?}");
        assert!(styles(&capitalized).is_empty());
        // A typo fix between two ordinary words is not vocabulary.
        let typo = analyze(
            "send it form the laptop",
            "send it from the laptop",
            None,
            &empty(),
        );
        assert!(terms(&typo).is_empty(), "{typo:?}");
        assert!(skipped_why(&typo).contains(&"stopwords".to_string()));
    }

    #[test]
    fn unrelated_typing_after_the_dictation_is_appended_not_an_edit() {
        let analysis = analyze(
            "Ship it today.",
            "Ship it today. Also ping Bob about the cutover.",
            None,
            &empty(),
        );
        assert_eq!(analysis.outcome, EditOutcome::Appended);
        assert_eq!(analysis.appended_words, 6);
        assert!(analysis.suggestions.is_empty());
        let before = analyze("ship it today", "Re: ship it today", None, &empty());
        assert_eq!(before.outcome, EditOutcome::Appended);
        // Continuing the sentence moves its period; that is not a preference.
        let continued = analyze("Ship it.", "Ship it now.", None, &empty());
        assert_eq!(continued.outcome, EditOutcome::Appended);
        assert!(continued.suggestions.is_empty(), "{continued:?}");
        let lead_in = analyze("Ship it today.", "So, ship it today.", None, &empty());
        assert_eq!(lead_in.outcome, EditOutcome::Appended, "{lead_in:?}");
    }

    #[test]
    fn undo_or_delete_removes_and_edit_then_undo_is_unchanged() {
        let removed = analyze("hello world", "", None, &empty());
        assert_eq!(removed.outcome, EditOutcome::Removed);
        assert!(removed.suggestions.is_empty());
        let restored = analyze("hello world", "hello world", None, &empty());
        assert_eq!(restored.outcome, EditOutcome::Unchanged);
    }

    #[test]
    fn rewrite_is_not_learned_but_an_earlier_fix_is() {
        let pasted = "please ask coo blast team about the gemm numbers";
        let rewritten = "never mind, I will check the dashboard myself tomorrow";
        let analysis = analyze(pasted, rewritten, None, &empty());
        assert_eq!(analysis.outcome, EditOutcome::Rewrite);
        assert!(analysis.suggestions.is_empty());
        let within = "please ask cuBLAS team about the gemm numbers";
        let analysis = analyze(pasted, rewritten, Some(within), &empty());
        assert_eq!(analysis.outcome, EditOutcome::Rewrite);
        assert_eq!(
            terms(&analysis),
            vec![("cuBLAS".into(), "coo blast".into(), TermReason::Misheard)]
        );
    }

    #[test]
    fn existing_term_gains_a_spoken_form_and_a_known_form_is_silent() {
        let analysis = analyze(
            "profile coo blast",
            "profile cuBLAS",
            None,
            &with_entry("cuBLAS", &["koo bloss"]),
        );
        assert!(matches!(
            analysis.suggestions.as_slice(),
            [Suggestion::DictionaryTerm {
                existing_entry: true,
                ..
            }]
        ));
        let known = analyze(
            "profile coo blast",
            "profile cuBLAS",
            None,
            &with_entry("cuBLAS", &["coo blast"]),
        );
        assert!(known.suggestions.is_empty());
        assert!(skipped_why(&known).contains(&"known_spoken_form".to_string()));
    }

    #[test]
    fn change_of_mind_contractions_and_stopwords_are_skipped() {
        let mind = analyze("we use Postgres here", "we use SQLite here", None, &empty());
        assert!(terms(&mind).is_empty());
        assert_eq!(skipped_why(&mind), vec!["meaning_change"]);
        let contraction = analyze("I dont know", "I don't know", None, &empty());
        assert!(terms(&contraction).is_empty());
        let pronoun = analyze("send it to them", "send it to him", None, &empty());
        assert_eq!(skipped_why(&pronoun), vec!["stopwords"]);
    }

    #[test]
    fn names_and_numbers_are_terms_when_they_sound_like_what_was_heard() {
        let name = analyze(
            "ship it to and roid users",
            "ship it to Android users",
            None,
            &empty(),
        );
        assert_eq!(
            terms(&name),
            vec![("Android".into(), "and roid".into(), TermReason::Misheard)]
        );
        let version = analyze("try llama three", "try Llama 3", None, &empty());
        assert_eq!(
            terms(&version),
            vec![("Llama 3".into(), "llama three".into(), TermReason::Misheard)]
        );
    }

    #[test]
    fn style_edits_become_preferences() {
        let chat = analyze("Sounds good.", "sounds good", None, &empty());
        assert_eq!(chat.outcome, EditOutcome::Style);
        assert_eq!(
            styles(&chat),
            vec![
                StylePreference::LowercaseStart,
                StylePreference::NoTrailingPeriod
            ]
        );
        let lower = analyze("Sounds Good To Me.", "sounds good to me.", None, &empty());
        assert_eq!(styles(&lower), vec![StylePreference::AllLowercase]);
        assert!(terms(&lower).is_empty());
    }

    #[test]
    fn suggestions_are_capped_and_deduplicated() {
        let pasted =
            "coo blast and coo blast with pie torch and tensor rt and cooda graphs and onyx";
        let edited = "cuBLAS and cuBLAS with PyTorch and TensorRT and CUDA Graphs and ONNX";
        let analysis = analyze(pasted, edited, None, &empty());
        let found = terms(&analysis);
        assert!(found.len() <= MAX_SUGGESTIONS, "{found:?}");
        let cublas = found.iter().filter(|(t, _, _)| t == "cuBLAS").count();
        assert_eq!(cublas, 1, "{analysis:#?}");
    }

    // --- the corpus record ----------------------------------------------

    struct TempEntry(std::path::PathBuf);

    impl TempEntry {
        fn new(name: &str) -> Self {
            let directory = std::env::temp_dir().join(format!(
                "phonon-edits-{name}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            std::fs::create_dir_all(&directory).unwrap();
            Self(directory)
        }

        fn metadata(&self) -> std::path::PathBuf {
            self.0.join("metadata.json")
        }
    }

    impl Drop for TempEntry {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn observed(record: bool, end_reason: &str) -> ObservedEdit {
        ObservedEdit {
            pasted: "we benchmark coo blast".into(),
            edited: "we benchmark cuBLAS".into(),
            end_reason: end_reason.into(),
            last_keystroke: Some("backspace".into()),
            last_keystroke_age_ms: Some(800),
            observed_ms: 5_300,
            record,
            ..Default::default()
        }
    }

    #[test]
    fn record_lands_in_an_existing_entry_and_keeps_its_fields() {
        let entry = TempEntry::new("record");
        std::fs::write(
            entry.metadata(),
            r#"{"schema_version":1,"id":"bar_1","created_at_unix_ms":1,"source":"bar","audio_file":"audio.wav","raw_transcript":"we benchmark coo blast","final_transcript":"we benchmark coo blast","intended_transcript":"we benchmark cuBLAS"}"#,
        )
        .unwrap();
        let request = observed(true, "quiet_after_edit");
        let analysis = request.analyze(&empty());
        assert!(write_record(&entry.metadata(), &request.record(&analysis)).unwrap());
        let stored = data::load_recording(&entry.metadata()).unwrap();
        let record = stored.user_edit.expect("user_edit written");
        assert_eq!(record.edited_text, "we benchmark cuBLAS");
        assert_eq!(record.outcome, EditOutcome::Correction);
        assert_eq!(record.end_last_keystroke.as_deref(), Some("backspace"));
        assert_eq!(record.suggestions.len(), 1);
        assert_eq!(record.suggestions[0].decision, "pending");
        assert_eq!(
            stored.intended_transcript.as_deref(),
            Some("we benchmark cuBLAS")
        );

        assert!(write_decision(&entry.metadata(), 0, "approved").unwrap());
        assert!(!write_decision(&entry.metadata(), 5, "approved").unwrap());
        assert!(!write_decision(&entry.metadata(), 0, "maybe").unwrap());
        let stored = data::load_recording(&entry.metadata()).unwrap();
        assert_eq!(
            stored.user_edit.unwrap().suggestions[0].decision,
            "approved"
        );
        let text = std::fs::read_to_string(entry.metadata()).unwrap();
        assert!(text.contains("\"kind\": \"dictionary_term\""), "{text}");
    }

    #[test]
    fn the_record_keeps_only_the_dictation_not_what_was_typed_around_it() {
        let request = ObservedEdit {
            pasted: "ship coo blast today.".into(),
            edited: "Re:  ship cuBLAS today.\n\nAlso ping Bob about the cutover.".into(),
            end_reason: "message_sent".into(),
            ..Default::default()
        };
        let analysis = request.analyze(&empty());
        let record = request.record(&analysis);
        assert_eq!(record.edited_text, "ship cuBLAS today.");
        assert_eq!(record.appended_words, 7);
        assert_eq!(terms(&analysis).len(), 1);
        // CJK and emoji spans cut on character boundaries.
        assert_eq!(dictation_region("前 東京タワー 👍 end", 1, 2), "東京タワー");
        assert_eq!(dictation_region("a b", 2, 1), "");
        assert_eq!(token_spans("a  b"), vec![(0, 1), (3, 4)]);
        for text in ["用cuBLAS。", "e\u{301} x", "東京タワー", " lead trail "] {
            let joined: Vec<String> = token_spans(text)
                .into_iter()
                .map(|(s, e)| text[s..e].to_string())
                .collect();
            assert_eq!(joined, tokenize(text), "{text}");
        }
    }

    #[test]
    fn a_deleted_entry_is_never_recreated() {
        // Local history off: the bar removed the entry at insertion.
        let entry = TempEntry::new("missing");
        let request = observed(true, "quiet_after_edit");
        let analysis = request.analyze(&empty());
        assert!(!write_record(&entry.metadata(), &request.record(&analysis)).unwrap());
        assert!(!entry.metadata().exists());
        assert!(!write_decision(&entry.metadata(), 0, "approved").unwrap());
        assert!(!entry.metadata().exists());
    }

    #[test]
    fn unreliable_end_reasons_record_but_never_propose() {
        let request = observed(false, "anchor_mismatch");
        let analysis = request.analyze(&empty());
        assert!(analysis.suggestions.is_empty());
        assert_eq!(analysis.outcome, EditOutcome::Correction);
        let event = observed_event(Some("p3"), &analysis, false);
        assert_eq!(event["kind"], "edit");
        assert_eq!(event["recorded"], false);
        let learnable = observed(false, "quiet_after_edit").analyze(&empty());
        let event = observed_event(Some("p3"), &learnable, false);
        assert_eq!(event["suggestions"][0]["kind"], "dictionary_term");
        assert_eq!(event["suggestions"][0]["term"], "cuBLAS");
        assert_eq!(event["suggestions"][0]["heard_as"], "coo blast");
    }

    /// Whatever text a field holds, the analysis must not panic and its
    /// vectors must describe every aligned token.
    #[test]
    fn arbitrary_text_never_panics() {
        let pool = [
            "the",
            "cuBLAS",
            "coo",
            "blast",
            "GPT-4o",
            "3:30pm",
            "Node.js",
            "C++",
            "\u{1F44D}",
            "e\u{301}",
            "東京",
            "タワー",
            "。",
            "—",
            "...",
            "'",
            "\"",
            "don't",
            "O'Brien",
            "\n",
            "  ",
            "I",
            "a",
            "Zoë",
            "naïve",
            "",
            "x",
            "!",
            "?",
            ",",
            "αβγ",
            "Ünïcödé",
            "-",
            "_",
        ];
        fn next(seed: &mut u64, bound: usize) -> usize {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            (*seed % bound as u64) as usize
        }
        fn make(seed: &mut u64, pool: &[&str]) -> String {
            let len = next(seed, 12);
            let words: Vec<&str> = (0..len).map(|_| pool[next(seed, pool.len())]).collect();
            words.join(if next(seed, 4) == 0 { "" } else { " " })
        }
        let mut seed: u64 = 0x5eed_1234_abcd_ef01;
        for _ in 0..2_000 {
            let pasted = make(&mut seed, &pool);
            let edited = make(&mut seed, &pool);
            let steps = align(&pasted, &edited);
            let (words, punct) = edit_vectors(&steps);
            assert_eq!(words.chars().count(), steps.len());
            assert_eq!(punct.chars().count(), steps.len());
            let analysis = analyze(&pasted, &edited, Some(&pasted), &empty());
            assert!(analysis.edit_distance >= 0.0 && analysis.edit_distance <= 1.0);
            assert!(analysis.suggestions.len() <= MAX_SUGGESTIONS);
        }
    }

    /// The behaviour cases shared with the bar's Swift tests.
    #[test]
    fn shared_behaviour_cases_hold() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/edit_learning_cases.json");
        let file: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let cases = file["cases"].as_array().unwrap();
        assert!(cases.len() >= 10);
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let dictionary = DictionaryFile {
                entries: case["dictionary"]
                    .as_array()
                    .map(|entries| {
                        entries
                            .iter()
                            .map(|entry| DictionaryEntry {
                                phrase: entry["phrase"].as_str().unwrap().into(),
                                replacement: None,
                                spoken_forms: entry["spoken_forms"]
                                    .as_array()
                                    .map(|forms| {
                                        forms
                                            .iter()
                                            .map(|f| f.as_str().unwrap().to_string())
                                            .collect()
                                    })
                                    .unwrap_or_default(),
                                source: "fixture".into(),
                                starred: false,
                                usage_count: 0,
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                ..Default::default()
            };
            let request = ObservedEdit {
                pasted: case["pasted"].as_str().unwrap().into(),
                edited: case["edited"].as_str().unwrap().into(),
                end_reason: case["end_reason"].as_str().unwrap().into(),
                ..Default::default()
            };
            let analysis = request.analyze(&dictionary);
            let event = observed_event(Some("p1"), &analysis, false);
            assert_eq!(event["outcome"], case["outcome"], "{name}: {analysis:?}");
            assert_eq!(
                event["suggestions"], case["suggestions"],
                "{name}: {analysis:?}"
            );
        }
    }

    #[test]
    fn request_parses_from_the_bar_json() {
        let request: ObservedEdit = serde_json::from_str(
            r#"{"cmd":"edit_observed","id":"p3","pasted":"a","edited":"b","end_reason":"focus_left","observed_ms":12,"record":false,"edited_within_threshold":null}"#,
        )
        .unwrap();
        assert_eq!(request.end_reason, "focus_left");
        assert!(!request.record);
    }

    // --- false positives found replaying Wispr's own observed edits -------

    #[test]
    fn units_typed_onto_a_number_are_formatting() {
        let time = analyze("meet at 3:30", "meet at 3:30pm", None, &empty());
        assert!(terms(&time).is_empty(), "{time:?}");
        assert_eq!(skipped_why(&time), vec!["formatting"]);
        let size = analyze("about 5 users", "about 5k users", None, &empty());
        assert!(terms(&size).is_empty(), "{size:?}");
        // A model name is not a unit.
        let model = analyze("try gpt 4 now", "try GPT-4o now", None, &empty());
        assert_eq!(terms(&model).len(), 1, "{model:?}");
    }

    #[test]
    fn added_words_are_not_a_mishearing_but_their_casing_still_counts() {
        let added = analyze("ping zainab today", "ping Zainab Ali today", None, &empty());
        let found = terms(&added);
        assert!(
            !found.iter().any(|(term, _, _)| term == "Zainab Ali"),
            "{added:?}"
        );
        assert_eq!(
            found,
            vec![("Zainab".into(), "zainab".into(), TermReason::Casing)]
        );
        assert!(skipped_why(&added).contains(&"added_words".to_string()));
        let removed = analyze("send to bob smith now", "send to Bob now", None, &empty());
        assert!(skipped_why(&removed).contains(&"removed_words".to_string()));
    }

    // --- false positives found replaying 1,066 Wispr ASR -> formatted pairs

    #[test]
    fn spoken_numbers_compose() {
        assert_eq!(spoken_digits("B three hundred"), "b 300");
        assert_eq!(spoken_digits("twenty five"), "25");
        assert_eq!(spoken_digits("three hundred five"), "305");
        assert_eq!(spoken_digits("two thousand twenty six"), "2026");
        assert_eq!(spoken_digits("seven eight zero"), "7 8 0");
        assert_eq!(spoken_digits("five point three flash"), "5 . 3 flash");
    }

    /// Gold misses found in the Wispr ASR -> formatted replay: compound
    /// spoken numbers and hyphenated names.
    #[test]
    fn model_names_with_numbers_and_hyphens_are_terms() {
        for (pasted, edited, term) in [
            // Third review, item 8: a case-insensitive hyphenation rule
            // dropped these.
            ("ask gpt 4 first", "ask GPT-4 first", "GPT-4"),
            ("ask GPT 4 first", "ask GPT-4 first", "GPT-4"),
            ("after covid 19 hit", "after COVID-19 hit", "COVID-19"),
            ("the wi fi died", "the Wi-Fi died", "Wi-Fi"),
            (
                "we rent B three hundred nodes",
                "we rent B300 nodes",
                "B300",
            ),
            ("try GLM 5.3 later", "try GLM-5.3 later", "GLM-5.3"),
            (
                "build TensorRT LLM engines",
                "build TensorRT-LLM engines",
                "TensorRT-LLM",
            ),
        ] {
            let analysis = analyze(pasted, edited, None, &empty());
            assert!(
                terms(&analysis).iter().any(|(t, _, _)| t == term),
                "{pasted} -> {edited}: {analysis:?}"
            );
        }
    }

    #[test]
    fn formatter_style_rewrites_are_not_terms() {
        let cases = [
            ("we need high level plans", "we need high-level plans"),
            ("go end to end now", "go end-to-end now"),
            ("about fifteen users", "about 15 users"),
            ("render at eight K", "render at 8K"),
            (
                "call seven eight zero five five five one two three four",
                "call 780-555-1234",
            ),
            ("we used ten gigabytes", "we used 10 GB"),
            ("so like when it works", "so is good When it works"),
            // Units, counts, ordinals and sizes are how numbers are written.
            (
                "a context of one hundred twenty eight k tokens",
                "a context of 128K tokens",
            ),
            ("about three hundred k users", "about 300K users"),
            ("we want eight x But then", "we want 8x then"),
            ("on the fourth try", "on the 4th try"),
            ("use 16 by 16 tiles", "use 16x16 tiles"),
            ("use sixteen by sixteen tiles", "use 16x16 tiles"),
        ];
        for (pasted, edited) in cases {
            let analysis = analyze(pasted, edited, None, &empty());
            assert!(
                terms(&analysis).is_empty(),
                "{pasted} -> {edited}: {analysis:?}"
            );
        }
        // Still terms: a mixed letter-digit name and a recased model name.
        let model = analyze("try zeta flow two now", "try ZetaFlow2 now", None, &empty());
        assert_eq!(terms(&model).len(), 1, "{model:?}");
        let llama = analyze("try llama three", "try Llama 3", None, &empty());
        assert_eq!(terms(&llama).len(), 1, "{llama:?}");
    }

    #[test]
    fn a_new_line_or_a_removed_filler_still_starts_a_sentence() {
        // `um` deleted before the capital; a list item on its own line.
        let filler = analyze(
            "we ship it um then we test",
            "We ship it. Then we test",
            None,
            &empty(),
        );
        assert!(terms(&filler).is_empty(), "{filler:?}");
        let list = analyze(
            "first item anyway the rest",
            "first item\nAnyway the rest",
            None,
            &empty(),
        );
        assert!(terms(&list).is_empty(), "{list:?}");
        // A capitalized name mid-sentence still is one.
        let name = analyze("ping ngozi today", "ping Ngozi today", None, &empty());
        assert_eq!(terms(&name).len(), 1);
    }

    #[test]
    fn lowercasing_a_url_or_term_at_the_start_is_not_a_style() {
        let url = analyze(
            "App.slack.com is down",
            "app.slack.com is down",
            None,
            &empty(),
        );
        assert!(styles(&url).is_empty(), "{url:?}");
        let word = analyze("Sounds good", "sounds good", None, &empty());
        assert_eq!(styles(&word), vec![StylePreference::LowercaseStart]);
    }

    fn proposes(pasted: &str, edited: &str, term: &str) {
        let analysis = analyze(pasted, edited, None, &empty());
        assert!(
            terms(&analysis).iter().any(|(t, _, _)| t == term),
            "{pasted} -> {edited} should propose {term}: {analysis:?}"
        );
    }

    /// Fourth review, item 4: joining two heard words into a name with an
    /// inner capital is the name's spelling, not spacing.
    #[test]
    fn joined_names_with_inner_capitals_are_terms() {
        proposes("push it to Git Hub", "push it to GitHub", "GitHub");
        proposes("ask Open AI about it", "ask OpenAI about it", "OpenAI");
        proposes("try Deep Seek today", "try DeepSeek today", "DeepSeek");
        proposes("rent an H 100 node", "rent an H100 node", "H100");
        // A unit glued to a number is still only spacing.
        let unit = analyze("at 2.5 G speeds", "at 2.5G speeds", None, &empty());
        assert!(terms(&unit).is_empty(), "{unit:?}");
        assert!(skipped_why(&unit).contains(&"spacing".to_string()));
    }

    /// Fourth review, item 5: a dot joining heard words (`Node JS`) is the
    /// term's own spelling; a spoken `dot` is formatting.
    #[test]
    fn dotted_joins_are_terms_unless_the_dot_was_said() {
        proposes("we use Node JS here", "we use Node.js here", "Node.js");
        proposes("move to Next JS soon", "move to Next.js soon", "Next.js");
        let said = analyze(
            "go to example dot com now",
            "go to example.com now",
            None,
            &empty(),
        );
        assert!(terms(&said).is_empty(), "{said:?}");
    }

    /// Fourth review, item 6: an ordinary word capitalized mid-sentence is
    /// a name when it is a known product or language name.
    #[test]
    fn known_names_that_are_ordinary_words_are_terms() {
        proposes("we write it in rust now", "we write it in Rust now", "Rust");
        proposes("the app is in swift", "the app is in Swift", "Swift");
        proposes("run it with bun please", "run it with Bun please", "Bun");
        proposes("post it on slack today", "post it on Slack today", "Slack");
        // Still a title habit for other ordinary words.
        let time = analyze("check the time now", "check the Time now", None, &empty());
        assert!(terms(&time).is_empty(), "{time:?}");
        assert!(skipped_why(&time).contains(&"common_word".to_string()));
        // Already in the dictionary: known, not a common word.
        let known = analyze(
            "ask mercury today",
            "ask Mercury today",
            None,
            &with_entry("Mercury", &[]),
        );
        assert!(terms(&known).is_empty(), "{known:?}");
        assert_eq!(skipped_why(&known), vec!["known_term".to_string()]);
    }

    /// Fourth review, item 6: the same capitalization made twice is a name.
    #[test]
    fn the_same_common_word_capitalized_twice_is_proposed() {
        let mut repeats = RepeatedEdits::default();
        let mut first = analyze("ask mercury today", "ask Mercury today", None, &empty());
        repeats.promote(&mut first);
        assert!(terms(&first).is_empty(), "{first:?}");
        let mut other = analyze("check the time now", "check the Time now", None, &empty());
        repeats.promote(&mut other);
        assert!(terms(&other).is_empty(), "{other:?}");
        let mut second = analyze("did mercury pay", "did Mercury pay", None, &empty());
        repeats.promote(&mut second);
        assert_eq!(
            terms(&second),
            vec![("Mercury".into(), "mercury".into(), TermReason::Casing)]
        );
        // Proposed once; a third sighting does not propose it again.
        let mut third = analyze("ask mercury again", "ask Mercury again", None, &empty());
        repeats.promote(&mut third);
        assert!(terms(&third).is_empty(), "{third:?}");
    }

    /// A field's own compound written as one word is a term; an ordinary
    /// word or a prefix joined on is spacing.
    #[test]
    fn joined_compounds_are_terms_unless_ordinary_or_prefixed() {
        proposes(
            "launch one thread block per row",
            "launch one threadblock per row",
            "threadblock",
        );
        proposes("ask codex to fix it", "ask Codex to fix it", "Codex");
        for (pasted, edited) in [
            ("open the data base now", "open the database now"),
            ("go full screen now", "go fullscreen now"),
            ("a sub project here", "a subproject here"),
        ] {
            let analysis = analyze(pasted, edited, None, &empty());
            assert!(
                terms(&analysis).is_empty(),
                "{pasted} -> {edited}: {analysis:?}"
            );
        }
    }

    /// One ordinary word respelled into a close lowercase word is a typo
    /// fix; a misheard word replaced by a far one is still a term.
    #[test]
    fn close_respellings_of_ordinary_words_are_spelling() {
        for (pasted, edited) in [
            ("show the demon now", "show the demo now"),
            ("a big speed here", "a big speedup here"),
        ] {
            let analysis = analyze(pasted, edited, None, &empty());
            assert!(
                terms(&analysis).is_empty(),
                "{pasted} -> {edited}: {analysis:?}"
            );
            assert!(skipped_why(&analysis).contains(&"spelling".to_string()));
        }
        proposes(
            "launch one thread block per row",
            "launch one threadblock per row",
            "threadblock",
        );
    }

    /// A first name capitalized mid-sentence is a name, where the system
    /// has a names list (macOS).
    #[test]
    fn first_names_capitalized_mid_sentence_are_terms() {
        if data::proper_names().is_empty() {
            return;
        }
        proposes("ask grant about it", "ask Grant about it", "Grant");
        proposes("send it to frank today", "send it to Frank today", "Frank");
        let title = analyze(
            "the chief officer said",
            "the Chief Officer said",
            None,
            &empty(),
        );
        assert!(terms(&title).is_empty(), "{title:?}");
    }

    /// Fifth review, item 7: one edit that capitalizes the same word twice
    /// counts once.
    #[test]
    fn one_edit_capitalizing_a_word_twice_counts_once() {
        let mut repeats = RepeatedEdits::default();
        let mut both = analyze(
            "check the time and the time again",
            "check the Time and the Time again",
            None,
            &empty(),
        );
        assert_eq!(
            skipped_why(&both)
                .iter()
                .filter(|w| *w == "common_word")
                .count(),
            2
        );
        repeats.promote(&mut both);
        assert!(terms(&both).is_empty(), "{both:?}");
        let mut later = analyze("what time is it", "what Time is it", None, &empty());
        repeats.promote(&mut later);
        assert_eq!(
            terms(&later).len(),
            1,
            "the second edit is the second sighting"
        );
    }

    /// Fifth review, item 6: a prefix capitalized only because it starts the
    /// sentence is still a prefix.
    #[test]
    fn a_sentence_initial_prefix_is_still_a_prefix() {
        let start = analyze(
            "Multi GPU servers are cheap",
            "Multi-GPU servers are cheap",
            None,
            &empty(),
        );
        assert!(terms(&start).is_empty(), "{start:?}");
        assert!(skipped_why(&start).contains(&"hyphenation".to_string()));
        let after = analyze(
            "it works. Multi GPU servers are cheap",
            "it works. Multi-GPU servers are cheap",
            None,
            &empty(),
        );
        assert!(terms(&after).is_empty(), "{after:?}");
        // Mid-sentence the capital is the name's.
        proposes("try Auto GPT now", "try Auto-GPT now", "Auto-GPT");
    }

    /// Counts, stopwords and ordinary words at the edge of a run are not
    /// part of the term; a heard word the mishearing needs stays.
    #[test]
    fn runs_lose_counts_stopwords_and_ordinary_words_at_their_edges() {
        let count = analyze(
            "we rent 4x B three hundred nodes",
            "we rent 4x B300 nodes",
            None,
            &empty(),
        );
        assert_eq!(
            terms(&count),
            vec![(
                "B300".into(),
                "B three hundred".into(),
                TermReason::Misheard
            )]
        );
        // Trailing `Then we` dropped; the deleted `four` stays in what was
        // heard.
        let stop = analyze(
            "quantize to int four and then We ship",
            "quantize to INT4 Then we ship",
            None,
            &empty(),
        );
        assert_eq!(
            terms(&stop),
            vec![("INT4".into(), "int four".into(), TermReason::Misheard)]
        );
        // Only an ordinary word was rewritten: the recased name is mined on
        // its own.
        let ordinary = analyze(
            "write a from scratch cute DSL kernel",
            "write a from-scratch CuTe DSL kernel",
            None,
            &empty(),
        );
        assert_eq!(
            terms(&ordinary),
            vec![("CuTe".into(), "cute".into(), TermReason::Casing)]
        );
        assert!(skipped_why(&ordinary).contains(&"edge_noise".to_string()));
        // A deleted stopword in front stays in what was heard: `las` alone
        // would become a spoken form of Atlas.
        let atlas = analyze(
            "we map it with at las today",
            "we map it with Atlas today",
            None,
            &empty(),
        );
        assert_eq!(
            terms(&atlas),
            vec![("Atlas".into(), "at las".into(), TermReason::Misheard)]
        );
        // `and roid` to `Android` needs its leading `and`.
        let whole = analyze(
            "ship it to and roid users",
            "ship it to Android users",
            None,
            &empty(),
        );
        assert_eq!(
            terms(&whole),
            vec![("Android".into(), "and roid".into(), TermReason::Misheard)]
        );
        assert!(is_count("4x") && is_count("8x") && is_count("12") && is_count("x2"));
        assert!(!is_count("x") && !is_count("H100") && !is_count("4k"));
    }

    /// Fourth review, item 7: only a lowercase prefix makes a hyphen
    /// ordinary; `Auto-GPT` is a name.
    #[test]
    fn capitalized_prefixes_in_names_are_terms() {
        proposes("try Auto GPT now", "try Auto-GPT now", "Auto-GPT");
        let prefix = analyze("a multi GPU box", "a multi-GPU box", None, &empty());
        assert!(terms(&prefix).is_empty(), "{prefix:?}");
    }
}
