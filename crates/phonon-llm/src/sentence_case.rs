//! Capitalise the word that starts a sentence (shared output stage, run on the refiner's
//! text before the writing-style preferences and line breaks).
//!
//! A dictation is recognised in pieces, and a piece that opens after a pause often comes
//! back lowercase: "...to do two events. one was in the masjid". The refiner is told to keep
//! the speaker's casing, so it keeps that too. This puts the capital back on the first word
//! after `.`, `?` or `!`, and nowhere it could be wrong:
//!
//! * a full stop that belongs to a word does not end a sentence: abbreviations (`etc.`,
//!   `e.g.`, `Dr.`), dotted forms (`U.S.`, `a.m.`) and initials (`J.`); the one-letter `I.`
//!   does end one;
//! * an ellipsis does not end one;
//! * only a plain lowercase word is changed: file names, URLs, e-mail addresses, identifiers
//!   with digits, underscores or inner capitals (`iPhone`, `kubectl.exe`), and words the
//!   caller says are spelled lowercase on purpose (`keep_lower`: dictionary terms such as
//!   `npm`) keep their case.
//!
//! Whitespace, including line breaks, is left exactly as it was. Text with no sentence
//! boundary, and the first word of the text, are never touched.

/// Words whose trailing full stop is part of the word, lowercased, without the stop.
const ABBREVIATIONS: &[&str] = &[
    "etc", "vs", "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "inc", "ltd", "corp", "approx",
    "fig", "vol", "cf", "al", "dept", "capt", "sgt", "misc",
];

const OPENERS: &[char] = &['"', '\'', '(', '[', '{', '\u{201C}', '\u{2018}'];
const CLOSERS: &[char] = &['"', '\'', ')', ']', '}', '\u{201D}', '\u{2019}'];

/// `text` with the first word of each sentence capitalised. `keep_lower(word)` is true for a
/// word that must stay as written.
pub fn capitalize_sentence_starts(text: &str, keep_lower: &dyn Fn(&str) -> bool) -> String {
    let mut out = String::with_capacity(text.len());
    let mut sentence_ended = false;
    let mut rest = text;
    while !rest.is_empty() {
        let space = rest.len() - rest.trim_start().len();
        out.push_str(&rest[..space]);
        rest = &rest[space..];
        if rest.is_empty() {
            break;
        }
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        let token = &rest[..end];
        rest = &rest[end..];
        if sentence_ended {
            out.push_str(&capitalized(token, keep_lower));
        } else {
            out.push_str(token);
        }
        sentence_ended = ends_sentence(token);
    }
    out
}

/// Whether the whitespace-delimited `token` closes a sentence.
fn ends_sentence(token: &str) -> bool {
    let body = token.trim_end_matches(CLOSERS);
    if body.ends_with('?') || body.ends_with('!') {
        return true;
    }
    let Some(word) = body.strip_suffix('.') else {
        return false;
    };
    if word.ends_with('.') {
        return false; // "wait..." trails off; it does not end a sentence
    }
    let word = word.trim_start_matches(OPENERS);
    if word.is_empty() || word.contains('.') {
        return false; // "e.g." / "U.S." / "a.m."
    }
    let lower = word.to_lowercase();
    if ABBREVIATIONS.contains(&lower.as_str()) {
        return false;
    }
    let mut letters = word.chars();
    if let (Some(only), None) = (letters.next(), letters.next()) {
        // an initial ("J. Smith", "plan B.") unless it is the pronoun
        return only == 'I' || only.is_ascii_digit();
    }
    true
}

/// `token` with its first letter upper-cased when it is a plain lowercase word.
fn capitalized(token: &str, keep_lower: &dyn Fn(&str) -> bool) -> String {
    let lead = token.len() - token.trim_start_matches(OPENERS).len();
    let (opening, word) = token.split_at(lead);
    let core: &str = word.trim_end_matches(|c: char| !c.is_alphanumeric());
    let mut chars = core.chars();
    let plain = chars.next().is_some_and(char::is_lowercase)
        && core
            .chars()
            .all(|c| c.is_lowercase() || c == '\'' || c == '\u{2019}' || c == '-');
    if !plain || keep_lower(core) {
        return token.to_string();
    }
    let first = word.chars().next().expect("a lowercase first letter");
    let mut out = String::with_capacity(token.len());
    out.push_str(opening);
    out.extend(first.to_uppercase());
    out.push_str(&word[first.len_utf8()..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fix(text: &str) -> String {
        capitalize_sentence_starts(text, &|_| false)
    }

    #[test]
    fn the_word_after_a_full_stop_gets_a_capital() {
        assert_eq!(
            fix("I wanted to do two events. one was in the masjid. did we agree on some dates? yes! ok"),
            "I wanted to do two events. One was in the masjid. Did we agree on some dates? Yes! Ok"
        );
    }

    #[test]
    fn whitespace_and_line_breaks_are_kept() {
        assert_eq!(
            fix("Done.  next\nline. then\n\nmore"),
            "Done.  Next\nline. Then\n\nmore"
        );
        assert_eq!(fix("  lead. trail  "), "  lead. Trail  ");
    }

    #[test]
    fn the_first_word_and_text_without_a_boundary_are_left_alone() {
        assert_eq!(fix("lower start only"), "lower start only");
        assert_eq!(fix(""), "");
        assert_eq!(fix("done."), "done.");
    }

    #[test]
    fn abbreviations_initials_and_ellipses_do_not_end_a_sentence() {
        assert_eq!(
            fix("UK, US, etc. clauses apply"),
            "UK, US, etc. clauses apply"
        );
        assert_eq!(
            fix("use a tool, e.g. cargo, or i.e. make"),
            "use a tool, e.g. cargo, or i.e. make"
        );
        assert_eq!(
            fix("meet Dr. smith at 9 a.m. tomorrow"),
            "meet Dr. smith at 9 a.m. tomorrow"
        );
        assert_eq!(fix("ask J. smith about it"), "ask J. smith about it");
        assert_eq!(fix("well... maybe so"), "well... maybe so");
        assert_eq!(fix("so did I. then we left"), "so did I. Then we left");
        assert_eq!(fix("I said no. then left"), "I said no. Then left");
    }

    #[test]
    fn only_a_plain_lowercase_word_changes() {
        assert_eq!(
            fix("It ran. iPhone next. main.rs failed. v2 shipped. node_modules gone. 3 more. - item. see https://x.y"),
            "It ran. iPhone next. main.rs failed. v2 shipped. node_modules gone. 3 more. - item. See https://x.y"
        );
        assert_eq!(fix("Done. kubectl.exe ran"), "Done. kubectl.exe ran");
    }

    #[test]
    fn quotes_and_brackets_around_the_boundary_are_handled() {
        assert_eq!(
            fix("He said \"stop.\" then left"),
            "He said \"stop.\" Then left"
        );
        assert_eq!(
            fix("Done. \"next\" is (maybe) fine. (then) go"),
            "Done. \"Next\" is (maybe) fine. (Then) go"
        );
    }

    #[test]
    fn words_spelled_lowercase_on_purpose_stay() {
        let keep = |word: &str| word == "npm";
        assert_eq!(
            capitalize_sentence_starts("It built. npm ran. then it stopped", &keep),
            "It built. npm ran. Then it stopped"
        );
    }

    #[test]
    fn contractions_and_hyphenated_words_are_capitalised_whole() {
        assert_eq!(
            fix("Ok. don't stop. well-known is fine"),
            "Ok. Don't stop. Well-known is fine"
        );
    }
}
