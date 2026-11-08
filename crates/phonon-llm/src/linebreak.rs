//! Deterministic line breaks for the refiner's output (shared output stage: macOS
//! engine, Windows and Linux builds).
//!
//! Every training target was rebuilt with `' '.join`, so the refiner never emits a
//! newline: dictated steps come out as "1. a 2. b" on one line and an email's
//! greeting and sign-off run into the body. This restores the shape:
//! - lists (every mode but casual): two or more inline "N." / "N)" markers numbered
//!   1, 2, 3... in order each start a new line; text before marker 1 stays as the
//!   lead-in line;
//! - formal only: a leading greeting ("Hi Sam,") and a trailing sign-off
//!   ("Best regards, Chris") go on their own lines.
//!
//! A literal port of the reference `fmt_v0/code/linebreak_rule.py` (its regexes are
//! spelled out by hand here; `linebreak_cases.json` holds its outputs on public
//! probes). Measured with the reference on A8cP outputs: prompt step lists 0 -> 96%,
//! formal greetings 0 -> 91%, sign-offs 0 -> 97%; changed held-out outputs: minimal
//! 0/301, casual 0/297, prompt 21/285 (all real lists), formal 4/300.

/// `apply(text, mode)` of the reference.
pub fn line_breaks(text: &str, mode: &str) -> String {
    let mut t = if mode != "casual" {
        lists(text)
    } else {
        text.to_string()
    };
    if mode == "formal" {
        t = greeting(&t);
        t = sign_off(&t);
    }
    t
}

/// The app's layout: lists as in `line_breaks` (every mode but casual), but the
/// greeting and sign-off go on their own lines only for email. Slack and Teams map to
/// formal too, and "Hi Sarah," alone on a line reads wrong in a chat message.
pub fn layout(text: &str, mode: &str, email: bool) -> String {
    if email {
        return line_breaks(text, "formal");
    }
    match mode {
        "casual" => text.to_string(),
        _ => lists(text),
    }
}

/// Inline numbered markers in order become one per line (the reference's `lists`).
pub fn lists(t: &str) -> String {
    let mut seq = Vec::new();
    let mut want = 1;
    for (start, number) in markers(t) {
        if number == want {
            seq.push(start);
            want += 1;
        }
    }
    if seq.len() < 2 {
        return t.to_string();
    }
    let mut out = Vec::new();
    let mut last = 0;
    for &start in &seq {
        out.push(t[last..start].trim_end().to_string());
        last = start;
    }
    out.push(t[last..].trim().to_string());
    let lead = &out[0];
    let items = out[1..].join("\n");
    if lead.is_empty() {
        items
    } else {
        format!("{lead}\n{items}")
    }
}

/// `(?:(?<=\s)|^)(\d{1,2})[.)]\s+(?=\S)`, non-overlapping, left to right: (byte
/// offset of the digits, their value).
fn markers(t: &str) -> Vec<(usize, u32)> {
    let chars: Vec<(usize, char)> = t.char_indices().collect();
    let mut found = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let preceded = i == 0 || chars[i - 1].1.is_whitespace();
        if preceded && chars[i].1.is_ascii_digit() {
            let mut j = i;
            while j < chars.len() && j - i < 2 && chars[j].1.is_ascii_digit() {
                j += 1;
            }
            // \d{1,2} backtracks to one digit only if that leaves [.)] next, and a
            // second digit is never [.)], so the run decides.
            if j < chars.len() && matches!(chars[j].1, '.' | ')') {
                let mut k = j + 1;
                while k < chars.len() && chars[k].1.is_whitespace() {
                    k += 1;
                }
                if k > j + 1 && k < chars.len() {
                    let number = t[chars[i].0..chars[j].0].parse().unwrap_or(0);
                    found.push((chars[i].0, number));
                    i = k;
                    continue;
                }
            }
        }
        i += 1;
    }
    found
}

const GREETING_WORDS: &[&str] = &[
    "hi",
    "hello",
    "hey",
    "dear",
    "good morning",
    "good afternoon",
    "good evening",
];

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `^((?:hi|hello|hey|dear|good (?:morning|afternoon|evening))\b[^.!?\n]{0,40}?[,!])\s+(?=\S)`
/// with re.I: (end of group 1, end of the match), byte offsets.
fn greeting_match(t: &str) -> Option<(usize, usize)> {
    // The words are ASCII: compare the original's leading bytes, case-insensitively.
    for word in GREETING_WORDS {
        let Some(head) = t.get(..word.len()) else {
            continue;
        };
        if !head.eq_ignore_ascii_case(word) {
            continue;
        }
        let rest = &t[word.len()..];
        if rest.chars().next().is_some_and(is_word) {
            continue; // \b
        }
        for (taken, (offset, c)) in rest.char_indices().enumerate() {
            if c == ',' || c == '!' {
                let group_end = word.len() + offset + c.len_utf8();
                let after = &t[group_end..];
                let spaces: usize = after
                    .chars()
                    .take_while(|c| c.is_whitespace())
                    .map(char::len_utf8)
                    .sum();
                if spaces > 0 && group_end + spaces < t.len() {
                    return Some((group_end, group_end + spaces));
                }
                if c == '!' {
                    break; // "!" is outside [^.!?\n]
                }
            } else if matches!(c, '.' | '?' | '\n') {
                break;
            }
            if taken == 40 {
                break;
            }
        }
    }
    None
}

/// Formal: the greeting on its own line, and the body after it capitalized.
fn greeting(t: &str) -> String {
    let mut t = match greeting_match(t) {
        Some((group_end, end)) => format!("{}\n{}", &t[..group_end], &t[end..]),
        None => t.to_string(),
    };
    if let Some((head, body)) = t.split_once('\n') {
        let probe = format!("{head} x");
        if greeting_match(&probe).is_some() {
            let mut chars = body.chars();
            if let Some(first) = chars.next().filter(|c| c.is_lowercase()) {
                let body = format!("{}{}", first.to_uppercase(), chars.as_str());
                t = format!("{head}\n{body}");
            }
        }
    }
    t
}

const SIGN_OFF_PHRASES: &[&str] = &[
    "thanks",
    "thank you",
    "best regards",
    "kind regards",
    "best",
    "regards",
    "cheers",
    "sincerely",
    "talk soon",
    "all the best",
    "warmly",
];

/// `(?<=[.!?])\s+((?:phrases)(?:,)?\s+[A-Z][a-z]+\.?)\s*$` with re.I, leftmost match:
/// (start of the whitespace, start and end of group 1).
fn sign_off_match(t: &str) -> Option<(usize, usize, usize)> {
    let chars: Vec<(usize, char)> = t.char_indices().collect();
    for i in 1..chars.len() {
        if !matches!(chars[i - 1].1, '.' | '!' | '?') || !chars[i].1.is_whitespace() {
            continue;
        }
        let mut p = i;
        while p < chars.len() && chars[p].1.is_whitespace() {
            p += 1;
        }
        if p == chars.len() {
            continue;
        }
        let start = chars[p].0;
        for phrase in SIGN_OFF_PHRASES {
            let Some(head) = t.get(start..start + phrase.len()) else {
                continue;
            };
            if !head.eq_ignore_ascii_case(phrase) {
                continue;
            }
            let mut rest = &t[start + phrase.len()..];
            rest = rest.strip_prefix(',').unwrap_or(rest);
            let spaces = rest.len() - rest.trim_start().len();
            if spaces == 0 {
                continue;
            }
            rest = &rest[spaces..];
            // [A-Z][a-z]+ under re.I: two or more ASCII letters
            let letters = rest.chars().take_while(char::is_ascii_alphabetic).count();
            if letters < 2 {
                continue;
            }
            rest = &rest[letters..];
            rest = rest.strip_prefix('.').unwrap_or(rest);
            if !rest.trim().is_empty() {
                continue;
            }
            let group_end = t.len() - rest.len();
            return Some((chars[i].0, start, group_end));
        }
    }
    None
}

/// Formal: the sign-off (and the name after its comma) on new lines.
fn sign_off(t: &str) -> String {
    let Some((space, start, end)) = sign_off_match(t) else {
        return t.to_string();
    };
    let group = &t[start..end];
    let group = if group.contains(',') {
        group.replacen(", ", ",\n", 1)
    } else {
        group.to_string()
    };
    format!("{}\n{group}", &t[..space])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's own outputs on public fmt_v0 probes (A8cP s1) plus edge cases,
    /// generated by running `linebreak_rule.apply` on each input.
    #[test]
    fn matches_the_python_reference() {
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("linebreak_cases.json")).unwrap();
        assert!(cases.len() >= 30);
        let mut changed = 0;
        for case in &cases {
            let input = case["input"].as_str().unwrap();
            let mode = case["mode"].as_str().unwrap();
            let expected = case["expected"].as_str().unwrap();
            changed += usize::from(input != expected);
            assert_eq!(
                line_breaks(input, mode),
                expected,
                "{} ({mode})",
                case["id"]
            );
        }
        assert!(changed >= 20, "most cases exercise a change");
    }

    #[test]
    fn casual_stays_inline_and_plain_text_is_untouched() {
        assert_eq!(line_breaks("1. eggs 2. milk", "casual"), "1. eggs 2. milk");
        for mode in ["minimal", "casual", "formal", "prompt"] {
            assert_eq!(
                line_breaks("Ship it after lunch.", mode),
                "Ship it after lunch."
            );
        }
        assert_eq!(
            line_breaks("Steps 2. and 3. are optional", "prompt"),
            "Steps 2. and 3. are optional"
        );
    }

    #[test]
    fn greeting_and_sign_off_break_only_for_email() {
        let text = "Hi Sarah, the build is green. Thanks, Chris";
        // Slack / Teams: formal mode, but a chat message
        assert_eq!(layout(text, "formal", false), text);
        // email
        assert_eq!(
            layout(text, "formal", true),
            "Hi Sarah,\nThe build is green.\nThanks,\nChris"
        );
        // numbered lists break in a formal chat message as before
        assert_eq!(
            layout(
                "Hi Sarah, two things: 1. ship it 2. tag it",
                "formal",
                false
            ),
            "Hi Sarah, two things:\n1. ship it\n2. tag it"
        );
        assert_eq!(
            layout("1. ship it 2. tag it", "prompt", false),
            "1. ship it\n2. tag it"
        );
        assert_eq!(
            layout("1. eggs 2. milk", "casual", false),
            "1. eggs 2. milk"
        );
    }

    #[test]
    fn steps_and_letters() {
        assert_eq!(
            line_breaks(
                "Do this: 1. open the file 2. fix the bug 3. run tests",
                "prompt"
            ),
            "Do this:\n1. open the file\n2. fix the bug\n3. run tests"
        );
        assert_eq!(
            line_breaks("Hi Sam, the build is green. Best regards, Chris", "formal"),
            "Hi Sam,\nThe build is green.\nBest regards,\nChris"
        );
        // greeting and sign-off only in formal
        assert_eq!(
            line_breaks("Hi Sam, the build is green. Thanks, Chris", "minimal"),
            "Hi Sam, the build is green. Thanks, Chris"
        );
    }
}
