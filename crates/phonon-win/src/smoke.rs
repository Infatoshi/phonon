//! The readiness rules, ported from `crates/phonon-core/src/lib.rs`.
//!
//! macOS calls Phonon ready only after `assets/startup.wav` passes through the
//! real recogniser and the real correction model. Windows applies the same two
//! rules, in the tray app on first run and in the headless test that gates a
//! release. Weights that merely load are not readiness.

/// Lowercase words with punctuation removed.
pub fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|character| character.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word: &String| !word.is_empty())
        .collect()
}

/// Whether the recogniser produced the fixture sentence. The fixture says
/// "Hello, Fluid Voice"; the alternatives are the mishearings the macOS gate
/// already accepts, so a slightly different acoustic model still passes.
pub fn asr_smoke_passed(text: &str) -> bool {
    let words = normalized_words(text);
    words.iter().any(|word| word == "hello")
        && words
            .iter()
            .any(|word| word == "voice" || word == "fluidvoice" || word == "boys")
        && words
            .iter()
            .any(|word| word == "fluid" || word == "flamid" || word == "fluidvoice")
}

/// Whether the correction stage returned a usable sentence for this transcript.
///
/// It deliberately does not assert which words the model chose: a rewrite is a
/// legitimate correction. It catches the two real failures, a collapse to one
/// word and a runaway expansion.
pub fn llm_audio_smoke_passed(asr_text: &str, corrected: &str) -> bool {
    let spoken = normalized_words(asr_text);
    let corrected = normalized_words(corrected);
    if spoken.is_empty() || corrected.is_empty() {
        return false;
    }
    let floor = spoken.len().saturating_sub(1).max(1);
    corrected.len() >= floor && corrected.len() <= spoken.len() + 2
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same cases the macOS suite asserts, so the two gates cannot drift.
    #[test]
    fn matches_the_macos_gate() {
        assert!(asr_smoke_passed("Hello, Fluid Voice."));
        assert!(asr_smoke_passed("Hello Flamid Voice."));
        assert!(!asr_smoke_passed("Hello."));
        assert!(llm_audio_smoke_passed(
            "Hello Flamid Voice.",
            "Hello, FluidAudio."
        ));
        assert!(!llm_audio_smoke_passed("Hello Flamid Voice.", "Hello."));
        assert!(!llm_audio_smoke_passed("Hello Flamid Voice.", ""));
    }

    #[test]
    fn a_runaway_expansion_fails() {
        assert!(!llm_audio_smoke_passed(
            "Hello Fluid Voice.",
            "Hello Fluid Voice, and here is a much longer answer than was asked for."
        ));
    }

    #[test]
    fn an_empty_transcript_fails() {
        assert!(!llm_audio_smoke_passed("", "anything"));
        assert!(!asr_smoke_passed(""));
    }

    #[test]
    fn punctuation_and_case_do_not_matter() {
        assert_eq!(
            normalized_words("Hello, FLUID  voice!"),
            vec!["hello", "fluid", "voice"]
        );
    }
}
