//! Number pass (shared output stage: macOS engine, Windows and Linux builds), run on the
//! final text after the line breaks (`linebreak`).
//!
//! The refiner sometimes leaves a spoken number in words, splits a phone number, or drops
//! a digit when a filler sits inside it. This deterministic pass looks at what was said
//! (the refiner's input) and rewrites, in the refiner's output, only:
//! - phone: a digit-by-digit run in the input (zero, oh, one..nine or digits, separated by
//!   spaces, commas, hyphens, dots or hesitations like "um"; at least 4 digit words and 7
//!   digits; not starting with "oh") becomes `AAA-BBB-CCCC`, `1-AAA-BBB-CCCC`, `BBB-CCCC`
//!   or the digits run together;
//! - money: "<number words> dollars|dollar|bucks|euros|euro|pounds [and <n> cents]" and
//!   "<a> point <b> million dollars" become `$N`, `$1,200`, `$N.CC`, `$A.B million`, `€N`
//!   or `£N`;
//! - dates: "<Month> [the] <ordinal>" and "the <ordinal> of <Month>" (optional
//!   ", <year words>") still in words in the output become "<Month> <D>[, YYYY]".
//!
//! Phone and money values replace the output span that holds them: after the two input
//! words before the value (the anchor must occur exactly once in the output), else before
//! the two words after it; no unique anchor, no change.
//!
//! Version: numpass **v2.2**. A literal port of the research reference
//! `passes_v2_2.numpass_v2_2` (frozen 2026-11-11, sha256 e8607047...; its regexes are
//! spelled out by hand here). v2.2 = v2 plus two bug fixes:
//! - v2.1: v2 left a hyphenated number word beside the new digits ("forty-five dollars" ->
//!   "forty-five $45"). A hyphen-joined token whose parts are all number words
//!   ("forty-five", "four-one-five", "twenty-first") now belongs to phone and money spans,
//!   a span widens over such a token directly beside it, and the date rule accepts hyphens
//!   between ordinal and year words in the output.
//! - v2.2: v2 let a phone run swallow the digit word that starts the next words ("... four
//!   two one more time" -> "41555501421 more time"). A run one digit past a complete length
//!   (7, 10, or 11 starting with 1) that ends in a single digit word followed by another
//!   word leaves that digit word out; a run right after "plus" is never trimmed.
//!
//! Percentages and plain cardinals have no rule and are never touched.
//!
//! `fixtures/numpass_cases.json` holds the v2.2 reference's own outputs on the public
//! T14/T14b refiner outputs (parity), on the public T13/T15/T16 outputs (guard), and on
//! pairs derived from the public ones that exercise the two fixes. Measured with v2 on the
//! shipped refiner: T14b (fresh set) worst-mode correct 0.807 -> 0.967 (the fixes came
//! after those sets were read). Known differences from the Python: digits are ASCII 0-9
//! (Python's `\d` also takes other scripts' decimal digits) and amounts above 10^38 are
//! left alone. Known reference behaviour kept for parity: an international number the
//! refiner wrote as "+33 1 23 45 67 89" is run together without its "+".

const DIGW: &[(&str, char)] = &[
    ("zero", '0'),
    ("oh", '0'),
    ("one", '1'),
    ("two", '2'),
    ("three", '3'),
    ("four", '4'),
    ("five", '5'),
    ("six", '6'),
    ("seven", '7'),
    ("eight", '8'),
    ("nine", '9'),
];
const HES: &[&str] = &["um", "uh", "er", "erm", "uhm", "hmm", "ah", "umm", "uhh"];
const UNITS: &[&str] = &[
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: &[&str] = &[
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
/// Ordinal words and their values, in the reference's alternation order.
const ORDW: &[(&str, u32)] = &[
    ("first", 1),
    ("second", 2),
    ("third", 3),
    ("fourth", 4),
    ("fifth", 5),
    ("sixth", 6),
    ("seventh", 7),
    ("eighth", 8),
    ("ninth", 9),
    ("tenth", 10),
    ("eleventh", 11),
    ("twelfth", 12),
    ("thirteenth", 13),
    ("fourteenth", 14),
    ("fifteenth", 15),
    ("sixteenth", 16),
    ("seventeenth", 17),
    ("eighteenth", 18),
    ("nineteenth", 19),
    ("twentieth", 20),
    ("thirtieth", 30),
];
const MONTHS: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];
/// Characters `_numish` / `_moneyish` strip from both ends of a token.
const STRIP: &[char] = &['(', ')', '[', ']', '.', ',', ';', ':', '!', '?', '"', '\''];

/// The refiner's `input` (what was said) and its `output`: the output with spoken phone
/// numbers, money amounts and word dates written as digits. Phone, then money, then dates.
pub fn apply(input: &str, output: &str) -> String {
    let text = phone_fix(input, output);
    let text = money_fix(input, &text);
    date_fix(input, &text)
}

// ---------------------------------------------------------------- word tables

fn digit_word(w: &str) -> Option<char> {
    DIGW.iter().find(|(name, _)| *name == w).map(|&(_, d)| d)
}

fn is_hes(w: &str) -> bool {
    HES.contains(&w)
}

fn unit_index(w: &str) -> Option<usize> {
    UNITS.iter().position(|u| *u == w)
}

fn tens_value(w: &str) -> Option<u128> {
    TENS.iter()
        .position(|t| *t == w)
        .map(|i| 10 * (i as u128 + 2))
}

fn is_unit_or_tens(w: &str) -> bool {
    unit_index(w).is_some() || tens_value(w).is_some()
}

/// `NUMW` of the reference.
fn is_numw(w: &str) -> bool {
    is_unit_or_tens(w) || matches!(w, "hundred" | "thousand" | "million" | "a" | "and" | "oh")
}

/// `MONEYW` of the reference.
fn is_moneyw(w: &str) -> bool {
    is_numw(w)
        || is_hes(w)
        || matches!(
            w,
            "point"
                | "dollars"
                | "dollar"
                | "bucks"
                | "cents"
                | "cent"
                | "euros"
                | "euro"
                | "pounds"
                | "usd"
        )
}

/// v2.1 `NUMWORD`: units, tens, hundred, thousand, million, oh and the ordinals.
fn is_numword(w: &str) -> bool {
    is_unit_or_tens(w)
        || matches!(w, "hundred" | "thousand" | "million" | "oh")
        || ORDW.iter().any(|(o, _)| *o == w)
}

/// v2.1: a hyphen-joined token whose every part is a number word ("forty-five",
/// "four-one-five", "twenty-first").
fn hyph_num(s: &str) -> bool {
    let low = s.to_lowercase();
    low.contains('-') && low.split('-').all(is_numword)
}

fn currency(w: &str) -> Option<&'static str> {
    match w {
        "dollars" | "dollar" | "bucks" => Some("$"),
        "euros" | "euro" => Some("€"),
        "pounds" => Some("£"),
        _ => None,
    }
}

/// Number words to a value ("one thousand two hundred" -> 1200); None for any other word
/// or no number word at all. `w2n` of the reference.
fn w2n(ws: &[&str]) -> Option<u128> {
    let (mut total, mut cur, mut seen) = (0u128, 0u128, false);
    for &w in ws {
        if w == "and" {
            continue;
        }
        if w == "a" {
            if cur == 0 {
                cur = 1;
            }
        } else if let Some(u) = unit_index(w) {
            cur = cur.checked_add(u as u128)?;
            seen = true;
        } else if let Some(t) = tens_value(w) {
            cur = cur.checked_add(t)?;
            seen = true;
        } else if w == "hundred" {
            cur = cur.max(1).checked_mul(100)?;
            seen = true;
        } else if w == "thousand" {
            total = total.checked_add(cur.max(1).checked_mul(1000)?)?;
            cur = 0;
            seen = true;
        } else {
            return None;
        }
    }
    if seen {
        total.checked_add(cur)
    } else {
        None
    }
}

// ---------------------------------------------------------------- characters

/// Python's `str.isspace` (the `\s` of a str pattern).
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python's `\w`.
fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `\b` at byte offset `p` of `s`.
fn boundary(s: &str, p: usize) -> bool {
    let before = s[..p].chars().next_back().is_some_and(is_word);
    let after = s[p..].chars().next().is_some_and(is_word);
    before != after
}

fn skip_spaces(s: &str, mut p: usize) -> usize {
    while let Some(c) = s[p..].chars().next().filter(|&c| is_space(c)) {
        p += c.len_utf8();
    }
    p
}

/// `\s+` at `p`: the offset after it.
fn spaces1(s: &str, p: usize) -> Option<usize> {
    let q = skip_spaces(s, p);
    (q > p).then_some(q)
}

/// A lowercase ASCII literal at `p`, case-insensitively as Python's `(?i)` matches it
/// (which also lets `i` match İ and ı, `k` the Kelvin sign and `s` the long s).
fn literal_ci(s: &str, p: usize, lit: &str) -> Option<usize> {
    let mut chars = s[p..].chars();
    let mut q = p;
    for l in lit.chars() {
        let c = chars.next()?;
        let same = c.to_ascii_lowercase() == l
            || (l == 'i' && matches!(c, '\u{130}' | '\u{131}'))
            || (l == 'k' && c == '\u{212a}')
            || (l == 's' && c == '\u{17f}');
        if !same {
            return None;
        }
        q += c.len_utf8();
    }
    Some(q)
}

fn literal(s: &str, p: usize, lit: &str) -> Option<usize> {
    s[p..].starts_with(lit).then_some(p + lit.len())
}

/// Python's `str.split()`.
fn split_py(s: &str) -> Vec<&str> {
    s.split(is_space).filter(|w| !w.is_empty()).collect()
}

/// `_norm` of the reference: lowercased, only a-z and apostrophes kept.
fn norm(w: &str) -> String {
    w.to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_lowercase() || *c == '\'')
        .collect()
}

/// `WTOK` of the reference: letter/apostrophe runs, digit runs, and every other
/// non-space character on its own.
fn tokens(t: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(c) = t[i..].chars().next() {
        let run = |pred: fn(char) -> bool| {
            t[i..]
                .char_indices()
                .find(|&(_, c)| !pred(c))
                .map_or(t.len(), |(k, _)| i + k)
        };
        if c.is_ascii_alphabetic() || c == '\'' {
            let j = run(|c| c.is_ascii_alphabetic() || c == '\'');
            out.push(&t[i..j]);
            i = j;
        } else if c.is_ascii_digit() {
            let j = run(|c| c.is_ascii_digit());
            out.push(&t[i..j]);
            i = j;
        } else {
            if !is_space(c) {
                out.push(&t[i..i + c.len_utf8()]);
            }
            i += c.len_utf8();
        }
    }
    out
}

fn is_word_token(w: &str) -> bool {
    w.starts_with(|c: char| c.is_ascii_alphabetic() || c == '\'')
}

fn is_digits(w: &str) -> bool {
    !w.is_empty() && w.chars().all(|c| c.is_ascii_digit())
}

// ---------------------------------------------------------------- anchors

type Anchors = (Vec<String>, Vec<String>);

/// The two words before token `i0` and the two after token `i1` (hesitations left out).
fn context(ws: &[&str], i0: usize, i1: usize) -> Anchors {
    let keep = |w: &&&str| is_word_token(w) && !is_hes(&w.to_lowercase());
    let left: Vec<String> = ws[..i0]
        .iter()
        .filter(keep)
        .map(|w| w.to_lowercase())
        .collect();
    let left = left[left.len().saturating_sub(2)..].to_vec();
    let right = ws[i1..]
        .iter()
        .filter(keep)
        .take(2)
        .map(|w| w.to_lowercase())
        .collect();
    (left, right)
}

/// Output token span holding a value: after the unique left anchor (else before the unique
/// right anchor), the maximal run of tokens accepted by `ok`; trailing "and"/"point"
/// trimmed. Tokens are the output split on single spaces.
fn anchor_span<'a>(
    out: &'a str,
    (left, right): &Anchors,
    ok: fn(&str) -> bool,
) -> (Vec<&'a str>, Option<(usize, usize)>) {
    let toks: Vec<&str> = out.split(' ').collect();
    let words: Vec<String> = toks.iter().map(|t| norm(t)).collect();
    let n = toks.len();
    let fits = |t: &str| ok(t) && !t.contains('\n');
    let mut span = None;
    if !left.is_empty() {
        let l = left.len();
        let pos: Vec<usize> = (0..n)
            .filter(|&i| i + 1 >= l && words[i + 1 - l..=i] == left[..])
            .collect();
        if let [p] = pos[..] {
            let j = p + 1;
            let mut k = j;
            while k < n && fits(toks[k]) {
                k += 1;
            }
            if k > j {
                span = Some((j, k));
            }
        }
    }
    if span.is_none() && !right.is_empty() {
        let r = right.len();
        let pos: Vec<usize> = (0..n)
            .filter(|&i| i + r <= n && words[i..i + r] == right[..])
            .collect();
        if let [p] = pos[..] {
            let k = p;
            let mut j = p;
            while j > 0 && fits(toks[j - 1]) {
                j -= 1;
            }
            if k > j {
                span = Some((j, k));
            }
        }
    }
    let Some((mut j, mut k)) = span else {
        return (toks, None);
    };
    // v2.1: a hyphenated number word directly beside the span joins it (plain number words
    // are already inside), unless punctuation ends the token before the gap
    let ends_mark = |t: &str| t.ends_with(['.', ',', ';', ':', '!', '?']);
    while j > 0 && hyph_num(toks[j - 1].trim_matches(STRIP)) && !ends_mark(toks[j - 1]) {
        j -= 1;
    }
    while k < n && hyph_num(toks[k].trim_matches(STRIP)) && !ends_mark(toks[k - 1]) {
        k += 1;
    }
    while k > j && matches!(norm(toks[k - 1]).as_str(), "and" | "point") {
        k -= 1;
    }
    (toks, (k > j).then_some((j, k)))
}

/// The tokens with `span` replaced by `new`, keeping the span's final punctuation mark.
fn replace(toks: &[&str], (j, k): (usize, usize), new: &str) -> String {
    let mut new = new.to_string();
    if let Some(c) = toks[k - 1]
        .chars()
        .next_back()
        .filter(|c| ".,;:!?".contains(*c))
    {
        new.push(c);
    }
    let mut parts: Vec<&str> = toks[..j].to_vec();
    parts.push(&new);
    parts.extend_from_slice(&toks[k..]);
    parts.join(" ")
}

// ---------------------------------------------------------------- phone

/// A complete phone length: 7, 10, or 11 starting with 1.
fn complete(d: &str) -> bool {
    matches!(d.len(), 7 | 10) || (d.len() == 11 && d.starts_with('1'))
}

/// Digit-by-digit runs in the input: (digits, anchors, cut). v2.2: a run one digit past a
/// complete length that ends in a single digit word followed by a non-digit word ("... four
/// two one more time") leaves that digit word out (`cut`); never after "plus".
fn phone_runs(inp: &str) -> Vec<(String, Anchors, bool)> {
    let ws = tokens(inp);
    let low: Vec<String> = ws.iter().map(|w| w.to_lowercase()).collect();
    let mut res = Vec::new();
    let mut i = 0;
    while i < ws.len() {
        let w = low[i].as_str();
        if (digit_word(w).is_some() && w != "oh") || is_digits(w) {
            let (mut j, mut digs, mut nw, mut last) = (i, String::new(), 0, i);
            while j < ws.len() {
                let x = low[j].as_str();
                if let Some(d) = digit_word(x) {
                    digs.push(d);
                    nw += 1;
                    last = j;
                } else if is_digits(x) {
                    digs.push_str(x);
                    last = j;
                } else if is_hes(x) || (matches!(x, "," | "-" | ".") && j + 1 < ws.len()) {
                } else {
                    break;
                }
                j += 1;
            }
            let next = ws.get(last + 1).copied().unwrap_or("");
            let intl = i > 0 && low[i - 1] == "plus";
            let cut = !intl
                && !complete(&digs)
                && complete(&digs[..digs.len() - 1])
                && digit_word(&low[last]).is_some()
                && is_word_token(next)
                && digit_word(&next.to_lowercase()).is_none();
            if cut {
                // drop the trailing digit word (and any separator or hesitation before it)
                let mut k = last - 1;
                while k > i && !(digit_word(&low[k]).is_some() || is_digits(ws[k])) {
                    k -= 1;
                }
                digs.pop();
                nw -= 1;
                last = k;
            }
            if nw >= 4 && digs.len() >= 7 {
                res.push((digs, context(&ws, i, last + 1), cut));
            }
            i = last + 1;
        } else {
            i += 1;
        }
    }
    res
}

fn fmt_digits(d: &str) -> String {
    match d.len() {
        10 => format!("{}-{}-{}", &d[..3], &d[3..6], &d[6..]),
        11 if d.starts_with('1') => format!("1-{}-{}-{}", &d[1..4], &d[4..7], &d[7..]),
        7 => format!("{}-{}", &d[..3], &d[3..]),
        _ => d.to_string(),
    }
}

fn numish(w: &str) -> bool {
    let s = w.trim_matches(STRIP);
    if s.is_empty() {
        return false;
    }
    let low = s.to_lowercase();
    digit_word(&low).is_some()
        || is_hes(&low)
        || hyph_num(s)
        || (s.chars().any(|c| c.is_ascii_digit())
            && s.chars()
                .all(|c| c.is_ascii_digit() || "-.:()+/".contains(c)))
}

/// Digits a token holds: digit characters plus hyphen-separated digit words.
fn digit_count(t: &str) -> usize {
    let s = t.trim_matches(STRIP).to_lowercase();
    s.chars().filter(char::is_ascii_digit).count()
        + s.split('-').filter(|p| digit_word(p).is_some()).count()
}

fn phone_fix(inp: &str, out: &str) -> String {
    let mut out = out.to_string();
    for (digs, anchors, cut) in phone_runs(inp) {
        let (toks, mut span) = anchor_span(&out, &anchors, numish);
        if let (Some((j, mut k)), true) = (span, cut) {
            // the left-anchored span may run on over the cut-off digit word
            // ("415-555-0142 one more"): give it back
            while k - 1 > j
                && digit_word(&toks[k - 1].trim_matches(STRIP).to_lowercase()).is_some()
                && toks[j..k - 1].iter().map(|t| digit_count(t)).sum::<usize>() >= digs.len()
            {
                k -= 1;
            }
            span = Some((j, k));
        }
        if let Some(span) = span {
            out = replace(&toks, span, &fmt_digits(&digs));
        }
    }
    out
}

// ---------------------------------------------------------------- money

fn with_commas(v: u128) -> String {
    let digits = v.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Spoken amounts in the input: (written amount, anchors).
fn money_items(inp: &str) -> Vec<(String, Anchors)> {
    let ws = tokens(inp);
    let lowned: Vec<String> = ws.iter().map(|w| w.to_lowercase()).collect();
    let low: Vec<&str> = lowned.iter().map(String::as_str).collect();
    let mut res = Vec::new();
    for (ci, &w) in low.iter().enumerate() {
        let Some(sym) = currency(w) else { continue };
        let mut i = ci;
        while i > 0 && {
            let x = low[i - 1];
            is_numw(x) || is_hes(x) || x == "point" || x == ","
        } {
            i -= 1;
        }
        let num: Vec<&str> = low[i..ci]
            .iter()
            .copied()
            .filter(|x| !is_hes(x) && *x != ",")
            .collect();
        if num.is_empty() || !num.iter().any(|x| is_unit_or_tens(x)) {
            continue;
        }
        let mut end = ci + 1;
        let mut cents = None;
        if end + 1 < low.len() && low[end] == "and" {
            let mut k = end + 1;
            while k < low.len() && (is_unit_or_tens(low[k]) || is_hes(low[k])) {
                k += 1;
            }
            if k < low.len() && matches!(low[k], "cents" | "cent") {
                let words: Vec<&str> = low[end + 1..k]
                    .iter()
                    .copied()
                    .filter(|x| !is_hes(x))
                    .collect();
                cents = w2n(&words);
                end = k + 1;
            }
        }
        let canon = match (
            num.iter().position(|x| *x == "point"),
            num.iter().position(|x| *x == "million"),
        ) {
            (Some(p), Some(m)) => {
                let a = w2n(&num[..p]);
                let b: String = num[(p + 1).min(m)..m]
                    .iter()
                    .filter_map(|x| unit_index(x))
                    .map(|u| u.to_string())
                    .collect();
                let Some(a) = a.filter(|_| !b.is_empty()) else {
                    continue;
                };
                format!("{sym}{a}.{b} million")
            }
            _ => {
                let Some(v) = w2n(&num) else { continue };
                let cents = match cents {
                    Some(c) if sym == "$" => format!(".{c:02}"),
                    _ => String::new(),
                };
                format!("{sym}{}{cents}", with_commas(v))
            }
        };
        res.push((canon, context(&ws, i, end)));
    }
    res
}

/// `[$€£]?\d[\d.,]*[kKmM]?` or a bare `[$€£]`.
fn amount_token(s: &str) -> bool {
    let cur = |c: char| matches!(c, '$' | '€' | '£');
    let mut chars = s.chars().peekable();
    if chars.next_if(|&c| cur(c)).is_some() && chars.peek().is_none() {
        return true;
    }
    if chars.next_if(|c| c.is_ascii_digit()).is_none() {
        return false;
    }
    while chars
        .next_if(|&c| c.is_ascii_digit() || c == '.' || c == ',')
        .is_some()
    {}
    chars.next_if(|c| matches!(c, 'k' | 'K' | 'm' | 'M'));
    chars.next().is_none()
}

fn moneyish(w: &str) -> bool {
    let s = w.trim_matches(STRIP);
    !s.is_empty() && (is_moneyw(&s.to_lowercase()) || hyph_num(s) || amount_token(s))
}

fn money_fix(inp: &str, out: &str) -> String {
    let mut out = out.to_string();
    for (canon, anchors) in money_items(inp) {
        let (toks, span) = anchor_span(&out, &anchors, moneyish);
        let Some((j, k)) = span else { continue };
        let has_value = toks[j..k].iter().any(|t| {
            norm(t)
                .chars()
                .chain(t.chars())
                .any(|c| c.is_ascii_digit() || c.is_ascii_lowercase())
        });
        if has_value {
            out = replace(&toks, (j, k), &canon);
        }
    }
    out
}

// ---------------------------------------------------------------- dates

/// "twenty second" -> 22, "ninth" -> 9; None if not an ordinal phrase.
fn ordinal_value(ws: &[&str]) -> Option<u32> {
    let ord = |w: &str| ORDW.iter().find(|(name, _)| *name == w).map(|&(_, v)| v);
    let last = ord(ws.last()?)?;
    match ws {
        [_] => Some(last),
        [tens, _] if last < 10 => tens_value(tens).map(|t| t as u32 + last),
        _ => None,
    }
}

/// Year words at the start of `ws` -> (year, words used): "twenty twenty six",
/// "two thousand five", "nineteen ninety eight", "twenty nineteen".
fn year(ws: &[&str]) -> Option<(u128, usize)> {
    if ws.len() >= 3 && ws[0] == "two" && ws[1] == "thousand" {
        let r = if unit_index(ws[2]).is_some() {
            w2n(&ws[2..3])
        } else {
            None
        };
        return Some(match r {
            Some(r) if r < 10 => (2000 + r, 3),
            _ => (2000, 2),
        });
    }
    if ws.len() >= 2 && matches!(ws[0], "nineteen" | "twenty") {
        let hi = if ws[0] == "nineteen" { 19 } else { 20 };
        if let Some(mut lo) = tens_value(ws[1]) {
            let mut n = 2;
            if let Some(u) = ws
                .get(2)
                .and_then(|w| unit_index(w))
                .filter(|u| (1..10).contains(u))
            {
                lo += u as u128;
                n = 3;
            }
            return Some((hi * 100 + lo, n));
        }
        if let Some(u) = unit_index(ws[1]).filter(|&u| u >= 10) {
            return Some((hi * 100 + u as u128, 2));
        }
    }
    None
}

/// One ordinal word at `p` of the lowercased input (no ordinal is a prefix of another).
fn ordinal_word_at(s: &str, p: usize) -> Option<usize> {
    ORDW.iter().find_map(|(w, _)| literal(s, p, w))
}

/// `ORDRX` at `p`, in the regex's order: with a "twenty"/"thirty" in front, then without.
fn ordinal_ends(s: &str, p: usize) -> Vec<usize> {
    let mut ends = Vec::new();
    if let Some(q) = ["twenty", "thirty"]
        .iter()
        .find_map(|w| literal(s, p, w))
        .and_then(|q| spaces1(s, q))
    {
        ends.extend(ordinal_word_at(s, q));
    }
    ends.extend(ordinal_word_at(s, p));
    ends
}

fn month_at(s: &str, p: usize) -> Option<(&'static str, usize)> {
    MONTHS
        .iter()
        .find_map(|m| literal(s, p, m).map(|q| (*m, q)))
}

/// A spoken word date in the lowercased input starting at `p`:
/// `\b(?:(MONTH)\s+(?:the\s+)?(ORDRX)|the\s+(ORDRX)\s+of\s+(MONTH))\b`.
/// Returns (end, month, ordinal text).
fn input_date_at(s: &str, p: usize) -> Option<(usize, &'static str, &str)> {
    if !boundary(s, p) {
        return None;
    }
    if let Some((month, q)) = month_at(s, p) {
        if let Some(q) = spaces1(s, q) {
            let with_the = literal(s, q, "the").and_then(|r| spaces1(s, r));
            for r in with_the.into_iter().chain([q]) {
                for e in ordinal_ends(s, r) {
                    if boundary(s, e) {
                        return Some((e, month, &s[r..e]));
                    }
                }
            }
        }
    }
    let q = literal(s, p, "the").and_then(|q| spaces1(s, q))?;
    for e in ordinal_ends(s, q) {
        let found = spaces1(s, e)
            .and_then(|r| literal(s, r, "of"))
            .and_then(|r| spaces1(s, r))
            .and_then(|r| month_at(s, r));
        if let Some((month, end)) = found {
            if boundary(s, end) {
                return Some((end, month, &s[q..e]));
            }
        }
    }
    None
}

/// Ordinal or year words separated by `[\s-]+` (v2.1: spaces or hyphens),
/// case-insensitively, at `p`.
fn words_ci(s: &str, p: usize, words: &[&str]) -> Option<usize> {
    let mut q = p;
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            let mut r = q;
            while let Some(c) = s[r..].chars().next().filter(|&c| is_space(c) || c == '-') {
                r += c.len_utf8();
            }
            if r == q {
                return None;
            }
            q = r;
        }
        q = literal_ci(s, q, w)?;
    }
    Some(q)
}

/// The date still in words in the output:
/// `(?i)\b(?:month\s+(?:the\s+)?ORD|the\s+ORD\s+of\s+month)\b`, first match.
fn output_date(s: &str, month: &str, ord: &[&str]) -> Option<(usize, usize)> {
    let at = |p: usize| -> Option<usize> {
        if !boundary(s, p) {
            return None;
        }
        if let Some(q) = literal_ci(s, p, month).and_then(|q| spaces1(s, q)) {
            let with_the = literal_ci(s, q, "the").and_then(|r| spaces1(s, r));
            for r in with_the.into_iter().chain([q]) {
                if let Some(e) = words_ci(s, r, ord).filter(|&e| boundary(s, e)) {
                    return Some(e);
                }
            }
        }
        literal_ci(s, p, "the")
            .and_then(|q| spaces1(s, q))
            .and_then(|q| words_ci(s, q, ord))
            .and_then(|q| spaces1(s, q))
            .and_then(|q| literal_ci(s, q, "of"))
            .and_then(|q| spaces1(s, q))
            .and_then(|q| literal_ci(s, q, month))
            .filter(|&e| boundary(s, e))
    };
    s.char_indices()
        .map(|(p, _)| p)
        .find_map(|p| at(p).map(|e| (p, e)))
}

/// `\s*,?\s*` at `p`.
fn comma_gap(s: &str, p: usize) -> usize {
    let q = skip_spaces(s, p);
    let q = literal(s, q, ",").unwrap_or(q);
    skip_spaces(s, q)
}

/// The year words after a date in the input: `\s*,?\s*((?:[a-z]+\s*){1,3})`.
fn year_words(s: &str) -> Option<&str> {
    let start = comma_gap(s, 0);
    let mut q = start;
    for _ in 0..3 {
        let run = s[q..]
            .find(|c: char| !c.is_ascii_lowercase())
            .unwrap_or(s.len() - q);
        if run == 0 {
            break;
        }
        q = skip_spaces(s, q + run);
    }
    (q > start).then(|| &s[start..q])
}

fn date_fix(inp: &str, out: &str) -> String {
    let low = inp.to_lowercase();
    let mut out = out.to_string();
    let mut from = 0;
    while let Some((end, month, ord)) = low[from..]
        .char_indices()
        .find_map(|(k, _)| input_date_at(&low, from + k))
    {
        from = end;
        let ord: Vec<&str> = split_py(ord);
        let Some(day) = ordinal_value(&ord) else {
            continue;
        };
        let rest = year_words(&low[end..]);
        let yr = rest.and_then(|r| year(&split_py(r)));
        let Some((start, stop)) = output_date(&out, month, &ord) else {
            continue;
        };
        let mut tail = out[stop..].to_string();
        if let (Some((y, n)), Some(rest)) = (yr, rest) {
            let ywords = &split_py(rest)[..n];
            let gap = comma_gap(&tail, 0);
            if let Some(e) = words_ci(&tail, gap, ywords).filter(|&e| boundary(&tail, e)) {
                tail = format!(", {y}{}", &tail[e..]);
            }
        }
        // a year already in digits keeps its comma: ^\s*,?\s*(\d{4})\b -> ", \1"
        let gap = comma_gap(&tail, 0);
        let four = tail[gap..]
            .bytes()
            .take(4)
            .filter(u8::is_ascii_digit)
            .count()
            == 4;
        if four && boundary(&tail, gap + 4) {
            tail = format!(", {}", &tail[gap..]);
        }
        let mut month_name = month.to_string();
        month_name[..1].make_ascii_uppercase();
        out = format!("{}{month_name} {day}{tail}", &out[..start]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cases() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/numpass_cases.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn check(set: &str) -> (usize, usize) {
        let cases = cases();
        let rows = cases[set].as_array().unwrap();
        let (mut changed, mut bad) = (0, Vec::new());
        for case in rows {
            let input = case["input"].as_str().unwrap();
            let output = case["output"].as_str().unwrap();
            let expected = case["expected"].as_str().unwrap();
            changed += usize::from(output != expected);
            let got = apply(input, output);
            if got != expected {
                bad.push(format!(
                    "in: {input}\nout: {output}\nwant: {expected}\ngot:  {got}"
                ));
            }
        }
        assert!(
            bad.is_empty(),
            "{} of {} differ:\n{}",
            bad.len(),
            rows.len(),
            bad.join("\n\n")
        );
        (rows.len(), changed)
    }

    /// The reference's own outputs on every distinct (input, refiner output) pair of the
    /// public T14 and T14b number sets (shipped and candidate refiners).
    #[test]
    fn matches_the_python_reference() {
        let (n, changed) = check("parity");
        assert_eq!(n, 1693);
        assert_eq!(changed, 571);
    }

    /// Public T13/T15/T16 outputs whose input holds a digit, a digit word, a currency word
    /// or a month (every other row cannot change): only the documented date rewrites.
    #[test]
    fn guard_rows_change_only_the_documented_dates() {
        let (n, changed) = check("guard");
        assert_eq!(n, 523);
        assert_eq!(changed, 6);
    }

    /// The fixes' own parity: pairs derived from the public ones (number words joined by
    /// hyphens, a digit word after a phone run) on which v2.2 and v2 differ.
    #[test]
    fn revision_fixes_match_the_reference() {
        let (n, changed) = check("fixes");
        assert_eq!(n, 300);
        assert!(changed > 250, "{changed}");
    }

    /// `test_numpass_v2_2.py`'s own cases beyond v2.1's: the swallowed digit word and the
    /// phone-length edges. Expected values are the v2.2 reference's outputs.
    #[test]
    fn a_digit_run_stops_at_a_complete_length() {
        let call = "call me at four one five five five five zero one four two one more time";
        let cases = [
            // v2 gave "Call me at 41555501421 more time."
            (call, "Call me at 415-555-0142 one more time.", "Call me at 415-555-0142 one more time."),
            (
                call,
                "Call me at four one five five five five zero one four two one more time.",
                "Call me at 415-555-0142 one more time.",
            ),
            (
                "my cell is six four seven two two two eight one nine three",
                "My cell is 647 222 8193.",
                "My cell is 647-222-8193.",
            ),
            (
                "the toll free line is one eight zero zero five five five two two two two",
                "The toll-free line is 1 800 555 2222.",
                "The toll-free line is 1-800-555-2222.",
            ),
            (
                "dial one four one five five five five zero one four two one more time",
                "Dial 1 415 555 0142 one more time.",
                "Dial 1-415-555-0142 one more time.",
            ),
            (
                "book the room at five five five zero one four two four people are coming",
                "Book the room at 555-0142, four people are coming.",
                "Book the room at 555-0142, four people are coming.",
            ),
            (
                "text four one five five five five zero one four two two of them are mine",
                "Text 415-555-0142, two of them are mine.",
                "Text 415-555-0142, two of them are mine.",
            ),
            // after "plus": never trimmed, as v2.1 ("dial plus" is not in the output: no change)
            (
                "dial plus one four one five five five five zero one four two",
                "Dial +1 415 555 0142.",
                "Dial +1 415 555 0142.",
            ),
            (
                "the london office is plus four four two zero seven nine four six zero nine five eight",
                "The London office is +44 20 7946 0958.",
                "The London office is +44 20 7946 0958.",
            ),
            // known reference behaviour (open issue): an international number written with
            // its "+" and groups is run together
            (
                "paris is plus three three one two three four five six seven eight nine please",
                "Paris is +33 1 23 45 67 89 please.",
                "Paris is 33123456789 please.",
            ),
        ];
        for (input, output, want) in cases {
            assert_eq!(apply(input, output), want, "{output}");
        }
    }

    /// The reference's own test file (`test_numpass_v2_1.py`), case for case: no word form
    /// of a number is left beside its digits, hyphenated or not, in any category, and
    /// percentages and plain cardinals (no rule) stay exactly as the refiner wrote them.
    #[test]
    fn no_word_form_left_beside_the_digits() {
        let tickets = "It costs forty five dollars for the tickets";
        let credits = "We spent ninety three dollars and seventy six cents on credits";
        let churn = "Churn was down forty five percent this year";
        let count = "We have forty five open tickets";
        let launch = "the launch is on march twenty first";
        let desk = "call the front desk at four one five five five five zero one four two";
        let cases = [
            // money, hyphenated and not, from the right anchor and from the left
            (
                tickets,
                "It costs forty-five dollars for the tickets",
                "It costs $45 for the tickets",
            ),
            (
                tickets,
                "It costs forty five dollars for the tickets",
                "It costs $45 for the tickets",
            ),
            (
                "It costs forty five dollars",
                "It costs forty-five dollars.",
                "It costs $45.",
            ),
            (
                credits,
                "We spent ninety-three dollars and seventy-six cents on credits",
                "We spent $93.76 on credits",
            ),
            (
                "The plumber charged seven hundred and fifty six dollars for the visit",
                "The plumber charged seven hundred fifty-six dollars for the visit",
                "The plumber charged $756 for the visit",
            ),
            (
                "We spent forty five dollars on lunch",
                "We spent $45 on lunch",
                "We spent $45 on lunch",
            ),
            // percentages and plain cardinals: untouched either way
            (
                churn,
                "Churn was down forty-five percent this year",
                "Churn was down forty-five percent this year",
            ),
            (
                churn,
                "Churn was down 45% this year",
                "Churn was down 45% this year",
            ),
            (
                count,
                "We have forty-five open tickets",
                "We have forty-five open tickets",
            ),
            (count, "We have 45 open tickets", "We have 45 open tickets"),
            // dates: hyphenated ordinal and year words
            (
                "the launch is on march twenty first, twenty twenty six",
                "The launch is on March twenty-first, twenty twenty-six.",
                "The launch is on March 21, 2026.",
            ),
            (
                launch,
                "The launch is on March twenty first.",
                "The launch is on March 21.",
            ),
            (
                launch,
                "The launch is on March 21st.",
                "The launch is on March 21st.",
            ),
            // phones: digit words hyphenated, spaced, partly digits
            (
                desk,
                "Call the front desk at four-one-five 555-0142.",
                "Call the front desk at 415-555-0142.",
            ),
            (
                desk,
                "Call the front desk at 4155, 5501, 42.",
                "Call the front desk at 415-555-0142.",
            ),
        ];
        for (input, output, want) in cases {
            assert_eq!(apply(input, output), want, "{output}");
        }
    }

    // Every expected value below is the Python reference's output on the same pair.

    #[test]
    fn phone_runs_with_fillers_and_area_codes() {
        let filler = "call me at five five five um one two three four five six seven tomorrow";
        assert_eq!(
            apply(filler, "Call me at 555 123 4567 tomorrow."),
            "Call me at 555-123-4567 tomorrow."
        );
        // the digit the refiner dropped next to the filler comes back
        assert_eq!(
            apply(filler, "Call me at 555 123 456 tomorrow."),
            "Call me at 555-123-4567 tomorrow."
        );
        assert_eq!(
            apply(
                "area code four one five, five five five, oh one nine nine thanks",
                "Area code 415 555 0199, thanks."
            ),
            "Area code 415-555-0199, thanks."
        );
        assert_eq!(
            apply(
                "plus one four one five five five five oh one nine nine is my cell",
                "+1 415 555 0199 is my cell."
            ),
            "1-415-555-0199 is my cell."
        );
        // seven digits, "oh" inside the run
        assert_eq!(
            apply(
                "my extension is five five five uh oh one two three ok",
                "My extension is 555 0123 ok"
            ),
            "My extension is 555-0123 ok"
        );
        // no left anchor: the span before the unique right anchor
        assert_eq!(
            apply(
                "at five five five one two three four at noon",
                "at 5551234 at noon at 5"
            ),
            "at 555-1234 at noon at 5"
        );
        // under four digit words: not a run
        assert_eq!(apply("room one two three", "Room 1 2 3"), "Room 1 2 3");
    }

    #[test]
    fn money_amounts() {
        // v2 left "forty-five $45" here; v2.1 takes the hyphenated word into the span
        assert_eq!(
            apply(
                "it costs forty five dollars for the room",
                "It costs forty-five dollars for the room."
            ),
            "It costs $45 for the room."
        );
        assert_eq!(
            apply(
                "it costs forty five dollars for the room",
                "It costs forty five dollars for the room."
            ),
            "It costs $45 for the room."
        );
        assert_eq!(
            apply(
                "pay twelve dollars and fifty cents today",
                "Pay 12 dollars and 50 cents today."
            ),
            "Pay $12.50 today."
        );
        assert_eq!(
            apply(
                "raised two point five million dollars last year",
                "Raised 2.5 million dollars last year."
            ),
            "Raised $2.5 million last year."
        );
        assert_eq!(
            apply(
                "it was one thousand two hundred bucks total",
                "It was 1200 bucks total."
            ),
            "It was $1,200 total."
        );
        assert_eq!(
            apply("about thirty euros each", "About 30 euros each."),
            "About €30 each."
        );
        assert_eq!(
            apply("about thirty pounds each", "About 30 pounds each."),
            "About £30 each."
        );
        // no number word before the currency: nothing to write
        assert_eq!(
            apply("a few dollars more", "A few dollars more."),
            "A few dollars more."
        );
    }

    #[test]
    fn word_dates() {
        assert_eq!(
            apply(
                "due may the fifth, twenty twenty six",
                "Due May the fifth, twenty twenty six."
            ),
            "Due May 5, 2026."
        );
        assert_eq!(
            apply(
                "on the tenth of april two thousand five we met",
                "On the tenth of April 2005 we met."
            ),
            "On April 10, 2005 we met."
        );
        assert_eq!(
            apply(
                "block september the twenty fourth please",
                "Block September the 24th please."
            ),
            "Block September the 24th please."
        );
        assert_eq!(
            apply(
                "block september twenty fourth please",
                "Block September twenty-fourth please."
            ),
            "Block September 24 please." // v2 left the hyphenated ordinal in words
        );
        assert_eq!(
            apply(
                "block september twenty fourth please",
                "Block September twenty fourth please."
            ),
            "Block September 24 please."
        );
        assert_eq!(
            apply("set it to april first", "Set it to April first."),
            "Set it to April 1."
        );
        // a word that is not a date stays
        assert_eq!(
            apply("may i go first", "May I go first?"),
            "May I go first?"
        );
    }

    #[test]
    fn untouched_without_numbers() {
        for text in ["", "Hello there.", "Line one\nLine two"] {
            assert_eq!(apply(text, text), text);
        }
        assert_eq!(apply("hello", "Hi  there"), "Hi  there");
    }

    #[test]
    fn helpers() {
        assert_eq!(w2n(&["one", "thousand", "two", "hundred"]), Some(1200));
        assert_eq!(w2n(&["a", "hundred"]), Some(100));
        assert_eq!(w2n(&["oh"]), None);
        assert_eq!(with_commas(1234567), "1,234,567");
        assert_eq!(fmt_digits("14155550199"), "1-415-555-0199");
        assert_eq!(year(&["twenty", "twenty", "six"]), Some((2026, 3)));
        assert_eq!(year(&["two", "thousand", "and"]), Some((2000, 2)));
        assert_eq!(year(&["nineteen", "eighteen"]), Some((1918, 2)));
        assert!(amount_token("$1,200") && amount_token("3.5M") && amount_token("€"));
        assert!(!amount_token("1x") && !amount_token("$k") && !amount_token(""));
    }
}
