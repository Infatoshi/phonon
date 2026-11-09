//! Opener restore (shared output stage: macOS engine, Windows and Linux builds), run on
//! the refiner's text before the line breaks (`linebreak`).
//!
//! The refiner drops a leading discourse opener the owner keeps: "Okay, ...", "Well, ...",
//! "And ...", "Now, ...". When the transcript that went into the refiner starts with one
//! of them and the refiner's text does not, this puts it back and lowercases the old first
//! word (unless it is "I", an I-contraction, all caps, internally capitalised, or written
//! capitalised mid-sentence in the transcript, a proper noun). "So" is left alone: the
//! owner keeps it only 16 times of 28. Measured offline on 608 of the owner's dictations:
//! WER against the owner's references 9.74% -> about 8.6%.
//!
//! Per writing mode (`MODES`, from those 608 dictations: 552 prompt, 32 minimal, 24
//! casual, no formal): prompt and minimal restore; casual restores and, when the refined
//! text starts lowercase, inserts the opener lowercase too; formal (an email or document
//! should not open with "Okay,") and any other mode never restore.
//!
//! `opener_cases.json` holds the reference cases (also for the iPhone and Android ports).

/// The opener words, lowercased, and the text each one is restored as.
const OPENERS: &[(&str, &str)] = &[
    ("okay", "Okay, "),
    ("ok", "Okay, "),
    ("well", "Well, "),
    ("and", "And "),
    ("now", "Now, "),
];

/// What a writing mode does with a dropped opener.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModeRule {
    /// Leave the refined text as it is.
    Never,
    /// Restore the opener capitalised ("Okay, the build...").
    Restore,
    /// Restore it, in lowercase when the refined text starts lowercase ("okay, the build...").
    MatchCase,
}

/// Minimal mode restores too; one switch, since its per-mode data (32 items) is still
/// being measured.
const MINIMAL_RESTORES: bool = true;

/// The one per-mode table. A mode not listed (verbatim, raw passthrough, unknown) never
/// restores.
const MODES: &[(&str, ModeRule)] = &[
    ("prompt", ModeRule::Restore),
    ("casual", ModeRule::MatchCase),
    (
        "minimal",
        if MINIMAL_RESTORES {
            ModeRule::Restore
        } else {
            ModeRule::Never
        },
    ),
    ("formal", ModeRule::Never),
];

fn mode_rule(mode: &str) -> ModeRule {
    MODES
        .iter()
        .find(|(name, _)| *name == mode)
        .map_or(ModeRule::Never, |&(_, rule)| rule)
}

/// `refined` with the opener `raw` starts with restored under the writing `mode`
/// (`MODES`), or `refined` unchanged.
pub fn restore(raw: &str, refined: &str, mode: &str) -> String {
    let rule = mode_rule(mode);
    let body = refined.trim_start();
    if rule == ModeRule::Never || body.is_empty() {
        return refined.to_string();
    }
    let mut raw_words = raw.split_whitespace();
    let Some(first) = raw_words.next() else {
        return refined.to_string();
    };
    // "Okay", "Okay," or "Okay." (the opener as its own sentence)
    let opener = first.trim_end_matches([',', '.']).to_lowercase();
    let Some(&(_, prefix)) = OPENERS.iter().find(|(word, _)| *word == opener) else {
        return refined.to_string();
    };
    // a transcript that is only the opener has nothing to restore it in front of
    if !raw_words.any(|word| word.chars().any(char::is_alphanumeric)) {
        return refined.to_string();
    }
    let refined_first = body.split_whitespace().next().unwrap_or("");
    let refined_opener = core(refined_first).to_lowercase();
    let same = refined_opener == opener
        || (matches!(opener.as_str(), "okay" | "ok")
            && matches!(refined_opener.as_str(), "okay" | "ok"));
    if same {
        return refined.to_string();
    }
    // casual style written lowercase: the opener goes in lowercase, the rest as it is
    if rule == ModeRule::MatchCase && body.chars().next().is_some_and(char::is_lowercase) {
        return format!("{}{body}", prefix.to_lowercase());
    }
    let rest = if keeps_capital(refined_first, raw) {
        body.to_string()
    } else {
        let mut chars = body.chars();
        match chars.next() {
            Some(c) => c.to_lowercase().chain(chars).collect(),
            None => String::new(),
        }
    };
    format!("{prefix}{rest}")
}

/// The word without leading or trailing punctuation ("Sam's," -> "Sam's").
fn core(word: &str) -> &str {
    word.trim_matches(|c: char| !c.is_alphanumeric())
}

/// The refined text's old first word keeps its capital: "I" and I-contractions, all caps
/// ("API"), internal capitals ("iPhone", "GitHub"), or a word the transcript writes
/// capitalised mid-sentence (a proper noun).
fn keeps_capital(word: &str, raw: &str) -> bool {
    let word = core(word);
    if word == "I" || word.starts_with("I'") || word.starts_with("I\u{2019}") {
        return true;
    }
    let letters: Vec<char> = word.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.len() >= 2 && letters.iter().all(|c| c.is_uppercase()) {
        return true;
    }
    if word.chars().skip(1).any(char::is_uppercase) {
        return true;
    }
    if !word.chars().next().is_some_and(char::is_uppercase) {
        return false;
    }
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    tokens.windows(2).any(|pair| {
        let sentence_end = pair[0]
            .trim_end_matches(['"', '\'', ')', '\u{201d}', '\u{2019}'])
            .ends_with(['.', '?', '!']);
        !sentence_end && core(pair[1]) == word
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prompt mode, the one most dictations use (552 of 608); the word rules below are
    /// the same in every mode that restores.
    fn restore(raw: &str, refined: &str) -> String {
        super::restore(raw, refined, "prompt")
    }

    #[test]
    fn matches_the_reference_cases() {
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("opener_cases.json")).unwrap();
        assert!(cases.len() >= 20);
        for case in &cases {
            let raw = case["raw"].as_str().unwrap();
            let refined = case["refined"].as_str().unwrap();
            let mode = case["mode"].as_str().unwrap();
            if mode == "minimal" && !MINIMAL_RESTORES {
                continue;
            }
            let expected = case["expected"].as_str().unwrap();
            assert_eq!(
                super::restore(raw, refined, mode),
                expected,
                "{}",
                case["id"]
            );
        }
    }

    #[test]
    fn prompt_restores() {
        assert_eq!(
            super::restore("okay ship it today", "Ship it today.", "prompt"),
            "Okay, ship it today."
        );
    }

    #[test]
    fn minimal_restores_behind_its_switch() {
        let want = if MINIMAL_RESTORES {
            "Well, it depends."
        } else {
            "It depends."
        };
        assert_eq!(
            super::restore("well it depends", "It depends.", "minimal"),
            want
        );
    }

    #[test]
    fn casual_matches_the_refined_case() {
        // lowercase casual text: the opener goes in lowercase, the rest unchanged
        assert_eq!(
            super::restore("okay see you at five", "see you at five", "casual"),
            "okay, see you at five"
        );
        assert_eq!(
            super::restore("and Sam is coming", "sam is coming", "casual"),
            "and sam is coming"
        );
        // capitalised casual text: as in prompt
        assert_eq!(
            super::restore("okay see you at five", "See you at five.", "casual"),
            "Okay, see you at five."
        );
        // "So" never, in casual either
        assert_eq!(
            super::restore("so see you at five", "see you at five", "casual"),
            "see you at five"
        );
    }

    #[test]
    fn formal_never_restores() {
        assert_eq!(
            super::restore("okay the build is green", "The build is green.", "formal"),
            "The build is green."
        );
    }

    #[test]
    fn verbatim_and_unknown_modes_never_restore() {
        for mode in ["verbatim", "raw", "", "unknown"] {
            assert_eq!(
                super::restore("okay the build is green", "The build is green.", mode),
                "The build is green.",
                "{mode:?}"
            );
        }
    }

    #[test]
    fn each_opener_is_restored() {
        assert_eq!(
            restore("okay the build is green", "The build is green."),
            "Okay, the build is green."
        );
        assert_eq!(
            restore("Well, it depends on the load", "It depends on the load."),
            "Well, it depends on the load."
        );
        assert_eq!(
            restore("And then we ship it", "Then we ship it."),
            "And then we ship it."
        );
        assert_eq!(
            restore("Now, let's look at the logs", "Let's look at the logs."),
            "Now, let's look at the logs."
        );
    }

    #[test]
    fn ok_becomes_okay() {
        assert_eq!(
            restore("OK, run the tests", "Run the tests."),
            "Okay, run the tests."
        );
        assert_eq!(
            restore("ok. Next sentence", "Next sentence."),
            "Okay, next sentence."
        );
        // the refiner's "OK" counts as the opener already being there
        assert_eq!(
            restore("okay run the tests", "OK, run the tests."),
            "OK, run the tests."
        );
    }

    #[test]
    fn present_opener_is_left_alone() {
        for (raw, refined) in [
            ("Okay, the build is green", "Okay, the build is green."),
            ("okay the build is green", "Okay. The build is green."),
            ("Well it depends", "well, it depends."),
            ("And then we ship", "And then we ship."),
            ("Now, let's go", "Now let's go."),
        ] {
            assert_eq!(restore(raw, refined), refined, "{raw:?}");
        }
    }

    #[test]
    fn first_word_capital_kept_where_it_belongs() {
        // proper noun: capitalised mid-sentence in the transcript
        assert_eq!(
            restore(
                "Okay, Sam said the build is green",
                "Sam said the build is green."
            ),
            "Okay, Sam said the build is green."
        );
        assert_eq!(
            restore("okay I think so", "I think so."),
            "Okay, I think so."
        );
        assert_eq!(restore("and I'm done", "I'm done."), "And I'm done.");
        assert_eq!(
            restore("well API keys rotate", "API keys rotate."),
            "Well, API keys rotate."
        );
        assert_eq!(
            restore("okay iPhone builds pass", "iPhone builds pass."),
            "Okay, iPhone builds pass."
        );
        assert_eq!(
            restore("now GitHub is down", "GitHub is down."),
            "Now, GitHub is down."
        );
        // capitalised only at a sentence start in the transcript: not a proper noun
        assert_eq!(
            restore("Okay. The build is green", "The build is green."),
            "Okay, the build is green."
        );
    }

    #[test]
    fn so_and_other_words_are_untouched() {
        for (raw, refined) in [
            ("So the build is green", "The build is green."),
            ("Okayish results today", "Results today."),
            ("Nowhere to go", "There is nowhere to go."),
            ("the build is green", "The build is green."),
        ] {
            assert_eq!(restore(raw, refined), refined, "{raw:?}");
        }
    }

    #[test]
    fn empty_and_opener_only_inputs() {
        assert_eq!(restore("Okay, ship it", ""), "");
        assert_eq!(restore("Okay, ship it", "  "), "  ");
        assert_eq!(restore("", "Ship it."), "Ship it.");
        assert_eq!(restore("Okay.", "Okay."), "Okay.");
        assert_eq!(restore("Okay.", "Sure."), "Sure.");
        assert_eq!(restore("and", "Hmm."), "Hmm.");
        // raw passthrough (the length guard's fallback) already starts with the opener
        assert_eq!(restore("okay ship it", "okay ship it"), "okay ship it");
    }
}
