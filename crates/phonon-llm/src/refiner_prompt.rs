//! The refiner's prompt, one definition for Rust (phonon-win builds it for llama.cpp).
//!
//! The refiner gets the exact prompt it was trained on (phonon-research, the ctx refiner
//! builders): the base instruction, the writing-mode line, then optional lines in this order:
//! `Vocabulary the speaker may use: a, b`, `On screen: <screen text>`, `Before cursor: <text>`,
//! as the system turn; the transcript as the user turn; the model's chat template with thinking
//! off. The context lines are shaped like the training data: the screen text's first
//! [`SCREEN_MAX_WORDS`] words and the last [`BEFORE_MAX_WORDS`] words before the cursor, all
//! whitespace collapsed to single spaces; a line with no text is left out. With a Before-cursor
//! line the transcript first goes through [`join_rule`]. The output is never changed by it.
//!
//! `fixtures/refiner_prompt_cases.json` holds prompts the training-side builders produced from
//! synthetic inputs; the tests below check this module against them byte for byte (as do
//! sidecar/refiner_prompt.py, ios/App/RefinerPrompt.swift and android RefinerPrompt.kt).

/// The refiner's fixed instruction.
pub const INSTRUCTION: &str = "Rewrite the raw dictation transcript into the exact text the speaker intended. \
Fix recognition errors and technical terms, keep the speaker's wording and casing style, do not add or drop content.";

/// The four writing modes and their lines (`writing_modes` in the refiner's `phonon_refiner.json`).
pub const MODES: [(&str, &str); 4] = [
    ("minimal", "Writing mode: minimal (notes and documents)."),
    ("casual", "Writing mode: casual (a chat message)."),
    ("formal", "Writing mode: formal (an email or document)."),
    (
        "prompt",
        "Writing mode: prompt (an instruction to an AI assistant).",
    ),
];

pub const VOCAB_PREFIX: &str = "Vocabulary the speaker may use: ";
pub const SCREEN_PREFIX: &str = "On screen: ";
pub const BEFORE_PREFIX: &str = "Before cursor: ";
/// Training screens were at most 350 words.
pub const SCREEN_MAX_WORDS: usize = 350;
/// Training kept the last 40 words before the cursor.
pub const BEFORE_MAX_WORDS: usize = 40;

/// The mode line for a mode name; unknown names fall back to minimal.
pub fn mode_line(mode: &str) -> &'static str {
    MODES
        .iter()
        .find(|(name, _)| *name == mode)
        .unwrap_or(&MODES[0])
        .1
}

/// Python's `str.split()` whitespace: Unicode White_Space plus the separators U+001C..U+001F.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn words(text: &str) -> Vec<&str> {
    text.split(is_space).filter(|w| !w.is_empty()).collect()
}

/// The On-screen line's text: the first [`SCREEN_MAX_WORDS`] words, whitespace collapsed.
pub fn screen_text(text: &str) -> String {
    let words = words(text);
    words[..words.len().min(SCREEN_MAX_WORDS)].join(" ")
}

/// The Before-cursor line's text: the last [`BEFORE_MAX_WORDS`] words, whitespace collapsed.
pub fn before_text(text: &str) -> String {
    let words = words(text);
    words[words.len().saturating_sub(BEFORE_MAX_WORDS)..].join(" ")
}

/// How the text before the cursor ends, for [`join_rule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ending {
    Word,
    Comma,
    SentenceEnd,
    Colon,
    Newline,
    Other,
}

/// Trailing spaces and tabs are ignored; a line break after them is a newline ending; then
/// trailing closing quotes and brackets are ignored.
pub fn end_class(before: &str) -> Ending {
    let b = before.trim_end_matches([' ', '\t']);
    if b.ends_with(['\n', '\r']) {
        return Ending::Newline;
    }
    let b = b
        .trim_end_matches(is_space)
        .trim_end_matches(['\'', '"', ')', ']']);
    match b.chars().last() {
        None => Ending::Other,
        Some('.' | '?' | '!') => Ending::SentenceEnd,
        Some(',' | ';') => Ending::Comma,
        Some(':') => Ending::Colon,
        Some(c) if c.is_alphanumeric() => Ending::Word,
        Some(_) => Ending::Other,
    }
}

/// Python's `\w` for one character: alphanumeric or `_`.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The transcript as it goes into the prompt after text before the cursor (`None`: no
/// Before-cursor line). After text that does not end a sentence (a word, `,` or `;`) its
/// first letter is lower-cased, unless the first word is "I" / I'm / I'll ..., or has another
/// capital after its first letter (API, GitHub, OK). Anything else leaves it unchanged.
pub fn join_rule(before: Option<&str>, raw: &str) -> String {
    let Some(before) = before else {
        return raw.to_string();
    };
    if !matches!(end_class(before), Ending::Word | Ending::Comma) {
        return raw.to_string();
    }
    let rest = raw.trim_start_matches(is_space);
    let lead = raw.len() - rest.len();
    let Some(first) = rest.chars().next() else {
        return raw.to_string();
    };
    if !(first.is_alphabetic() && first.is_uppercase()) {
        return raw.to_string();
    }
    let token = rest.split(is_space).next().unwrap_or("");
    let core = token.trim_end_matches(|c: char| !(is_word_char(c) || c == '\'' || c == '\u{2019}'));
    if core == "I" || core.starts_with("I'") || core.starts_with("I\u{2019}") {
        return raw.to_string();
    }
    if core.chars().skip(1).any(char::is_uppercase) {
        return raw.to_string();
    }
    let mut out = String::with_capacity(raw.len());
    out.push_str(&raw[..lead]);
    out.extend(first.to_lowercase());
    out.push_str(&rest[first.len_utf8()..]);
    out
}

/// The system turn. `screen` and `before` are the raw texts the app has (`None` or empty: no line).
pub fn system_prompt(
    mode: &str,
    vocabulary: &[String],
    screen: Option<&str>,
    before: Option<&str>,
) -> String {
    let mut system = format!("{INSTRUCTION}\n{}", mode_line(mode));
    if !vocabulary.is_empty() {
        system.push('\n');
        system.push_str(VOCAB_PREFIX);
        system.push_str(&vocabulary.join(", "));
    }
    let shown = screen.map(screen_text).unwrap_or_default();
    if !shown.is_empty() {
        system.push('\n');
        system.push_str(SCREEN_PREFIX);
        system.push_str(&shown);
    }
    let typed = before.map(before_text).unwrap_or_default();
    if !typed.is_empty() {
        system.push('\n');
        system.push_str(BEFORE_PREFIX);
        system.push_str(&typed);
    }
    system
}

/// The user turn: the trimmed transcript, through the join rule when there is a Before-cursor line.
pub fn user_turn(transcript: &str, before: Option<&str>) -> String {
    let raw = transcript.trim_matches(is_space);
    match before.filter(|b| !before_text(b).is_empty()) {
        Some(before) => join_rule(Some(before), raw),
        None => raw.to_string(),
    }
}

/// The whole prompt as the model's chat template renders it with thinking off (the evaluation
/// render, llama-server `enable_thinking: false`), for a runtime that takes raw text.
pub fn chatml(system: &str, user: &str) -> String {
    format!(
        "<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n<think></think>"
    )
}

/// [`chatml`] of [`system_prompt`] and [`user_turn`].
pub fn prompt(
    transcript: &str,
    mode: &str,
    vocabulary: &[String],
    screen: Option<&str>,
    before: Option<&str>,
) -> String {
    chatml(
        &system_prompt(mode, vocabulary, screen, before),
        &user_turn(transcript, before),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixtures() -> Value {
        serde_json::from_str(include_str!("../../../fixtures/refiner_prompt_cases.json")).unwrap()
    }

    fn text(value: &Value) -> Option<&str> {
        value.as_str()
    }

    #[test]
    fn the_fixed_strings_are_the_training_ones() {
        let f = fixtures();
        assert_eq!(f["instruction"], INSTRUCTION);
        assert_eq!(f["vocabulary_prefix"], VOCAB_PREFIX);
        assert_eq!(f["screen_max_words"], SCREEN_MAX_WORDS);
        assert_eq!(f["before_max_words"], BEFORE_MAX_WORDS);
        for (name, line) in MODES {
            assert_eq!(f["modes"][name], line);
        }
        assert_eq!(f["modes"].as_object().unwrap().len(), MODES.len());
    }

    #[test]
    fn prompts_are_byte_identical_to_the_training_builder() {
        let f = fixtures();
        let cases = f["cases"].as_array().unwrap();
        assert!(cases.len() >= 20);
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let vocabulary: Vec<String> = case["vocabulary"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            let mode = case["mode"].as_str().unwrap();
            let screen = text(&case["screen_text"]);
            let before = text(&case["before_cursor"]);
            let transcript = case["transcript"].as_str().unwrap();
            assert_eq!(
                system_prompt(mode, &vocabulary, screen, before),
                case["system"].as_str().unwrap(),
                "system: {name}"
            );
            assert_eq!(
                user_turn(transcript, before),
                case["user"].as_str().unwrap(),
                "user: {name}"
            );
            assert_eq!(
                prompt(transcript, mode, &vocabulary, screen, before),
                case["prompt"].as_str().unwrap(),
                "prompt: {name}"
            );
        }
    }

    #[test]
    fn the_31_join_rule_cases_of_the_training_rule() {
        let f = fixtures();
        let cases = f["join_rule_cases"].as_array().unwrap();
        assert_eq!(cases.len(), 31);
        for case in cases {
            let before = case[0].as_str();
            let raw = case[1].as_str().unwrap();
            assert_eq!(
                join_rule(before, raw),
                case[2].as_str().unwrap(),
                "before {before:?} raw {raw:?}"
            );
        }
    }

    #[test]
    fn unknown_modes_fall_back_to_minimal() {
        assert_eq!(mode_line("shouting"), mode_line("minimal"));
    }
}
