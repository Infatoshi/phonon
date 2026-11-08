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
/// word and a runaway expansion. The correction may drop one word or 0.5% of the
/// spoken words, whichever is larger: on a long recording the refiner removes a few
/// fillers and markers (3 and 4 of 1,094 and 2,369 words on public LibriSpeech
/// recordings of 410 s and 900 s), which the one-word allowance of the 3-word
/// fixture would fail. Up to 199 words the rule is the macOS one.
pub fn llm_audio_smoke_passed(asr_text: &str, corrected: &str) -> bool {
    let spoken = normalized_words(asr_text);
    let corrected = normalized_words(corrected);
    if spoken.is_empty() || corrected.is_empty() {
        return false;
    }
    let dropped = (spoken.len() / 200).max(1);
    let floor = spoken.len().saturating_sub(dropped).max(1);
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
    fn a_long_transcript_may_lose_half_a_percent() {
        let spoken = vec!["word"; 1094].join(" ");
        let kept = |n: usize| vec!["word"; n].join(" ");
        // 1,094 words: 5 may go (0.5%), 6 may not
        assert!(llm_audio_smoke_passed(&spoken, &kept(1091)));
        assert!(llm_audio_smoke_passed(&spoken, &kept(1089)));
        assert!(!llm_audio_smoke_passed(&spoken, &kept(1088)));
        // short: still one word
        let short = vec!["word"; 199].join(" ");
        assert!(llm_audio_smoke_passed(&short, &kept(198)));
        assert!(!llm_audio_smoke_passed(&short, &kept(197)));
        // the expansion bound is unchanged
        assert!(!llm_audio_smoke_passed(&spoken, &kept(1097)));
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
