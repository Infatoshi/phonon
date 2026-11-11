//! Number pass (shared output stage: macOS engine, Windows and Linux builds), run on the
//! final text after the line breaks (`linebreak`).
//!
//! The refiner sometimes leaves a spoken number in words. This deterministic pass looks at
//! what was said (the refiner's input) and converts, in the refiner's output, only spoken
//! number WORDS it left as words:
//! - phone: a digit-by-digit run in the input (zero, oh, one..nine or digits, "double" /
//!   "triple" before a digit word, separated by spaces, commas, hyphens, dots or
//!   hesitations like "um"; at least 4 digit words and 7 digits; not starting with "oh")
//!   becomes `AAA-BBB-CCCC`, `1-AAA-BBB-CCCC`, `BBB-CCCC` or the digits run together;
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
//! # Invariant
//!
//! The rule converts spoken number words. It never rewrites, regroups or strips what the
//! refiner already wrote in digits or symbols (a leading "+", grouped digits, a "$" amount,
//! a formatted date, a version, a time). Enforced in code: each replacement (one phone
//! run, one money amount, the date stage) is kept only if INV-1..3 hold between before and
//! after, and the whole result is checked once more against the refiner's output, which is
//! returned unchanged on any failure.
//! - INV-1, digits: the output's ASCII digits, read in order, are a subsequence of the
//!   digits after the rule. New digits come only from converted number words; no existing
//!   digit is removed, changed or reordered.
//! - INV-2, symbols: no `+ $ € £ % : . / ( )` inside or directly beside an existing digit is
//!   removed (per-symbol count may not drop).
//! - INV-3, regrouping: spaces, commas or hyphens between existing digits change only when
//!   the input's spoken words give exactly the same digit string (phone: the span reads as
//!   the spoken run; money with no number word left: same digits as the spoken amount).
//!   A run with a leading "+" is never touched: no span holds or directly follows a "+"
//!   token, and every `+<digits>` run survives verbatim.
//! - INV-4, identity: if the INPUT holds no spoken number word (zero..nineteen, twenty..
//!   ninety, hundred, thousand, million, billion, oh, double, triple, the ordinals
//!   first..twentieth and thirtieth; hyphen-joined forms count by their parts), the output
//!   is returned unchanged.
//! - INV-5, whitespace (v2.3.1): whitespace outside a replaced span is byte-identical. The
//!   output is split on any whitespace (space, tab, `\n`, `\r`, ...), every run is kept, and
//!   a span never crosses a run holding a line break; checked as: the whitespace runs that
//!   hold a line break are the same, in order.
//! - INV-6, words (v2.3.1): no word outside the number vocabulary is removed. On whitespace
//!   tokens (punctuation stripped, lowercased; a token with a digit or currency sign is a
//!   value), the words that are not removable are the same, in order. Removable: number
//!   words, "point", unit words (dollars, bucks, cents, euros, pounds, ...), hesitations;
//!   "a" right before a scale word ("a thousand"); "and" between a number or unit token and
//!   a number token; "the" before an ordinal; "of" between an ordinal and a month. A span
//!   also stops at a token that ends a sentence, and its edges are trimmed to number, unit
//!   or digit tokens ("$2,500 a night" keeps "a").
//!
//! So a wrong or missing digit the refiner wrote stays as written: that belongs to the
//! refiner's training, not to a post-rule.
//!
//! # Version
//!
//! The final rule, numpass **v2.3.1 + v4** (frozen as v4.3.1): a literal port of the
//! research reference `passes_v4_3_1.numpass_v4_3_1` (frozen 2026-11-11, sha256
//! fef013da...; on `passes_v2_3_1`, sha256 12ebecfc...; regexes spelled out by hand here).
//! It is v2 plus:
//! - v2.1: a hyphen-joined token whose parts are all number words ("forty-five",
//!   "four-one-five", "twenty-first") belongs to phone and money spans and a span widens
//!   over one directly beside it; dates accept hyphens between ordinal and year words. v2
//!   wrote "forty-five $45".
//! - v2.2: a phone run one digit past a complete length (7, 10, or 11 starting with 1)
//!   that ends in a single digit word followed by another word leaves that word out, never
//!   after "plus" ("... four two one more time"; v2 wrote "41555501421 more time").
//! - v2.3: INV-1..INV-4.
//! - v2.3.1: INV-5 and INV-6. v2.3 split the output on single spaces, so after the line
//!   breaks a value ending a list item shared a token with the next line and the new digits
//!   went in beside the old words ("$45 dollars", "forty $45"); and its money span took in
//!   "a" and ran past a sentence end ("$2,500 night", "$52,000 step closer").
//! - v4: "double" / "triple" before a digit word in a phone run.
//!
//! Percentages and plain cardinals have no rule and are never touched.
//!
//! `fixtures/numpass_cases.json` holds the reference's own outputs on the public T14/T14b
//! refiner outputs (parity), the public T13/T15/T16 outputs (guard), pairs derived from the
//! public ones that exercise each revision, and the reference's 66 test cases. Known
//! differences from the Python: digits are ASCII 0-9 (Python's `\d` also takes other
//! scripts' decimal digits) and amounts above 10^38 are left alone.

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
    // INV-4: nothing spoken as a number word, nothing to convert
    if !has_number_words(input) {
        return output.to_string();
    }
    let text = phone_fix(input, output);
    let text = money_fix(input, &text);
    // the date stage is kept only if it holds INV-1..3 on its own
    let dated = date_fix(input, &text);
    let text = if inv_ok(&text, &dated) { dated } else { text };
    // and the whole result once more against what the refiner wrote
    if inv_ok(output, &text) {
        text
    } else {
        output.to_string()
    }
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

/// The output split on any whitespace (v2.3.1): `toks[i]` and `toks[i + 1]` are separated by
/// `seps[i]`, kept byte for byte. Text that starts or ends with whitespace has an empty
/// first or last token.
struct Split<'a> {
    toks: Vec<&'a str>,
    seps: Vec<&'a str>,
}

fn split_ws(s: &str) -> Split<'_> {
    let (mut toks, mut seps) = (Vec::new(), Vec::new());
    let (mut start, mut chars) = (0, s.char_indices().peekable());
    while let Some((i, c)) = chars.next() {
        if !is_space(c) {
            continue;
        }
        let mut end = i + c.len_utf8();
        while let Some(&(k, d)) = chars.peek() {
            if !is_space(d) {
                break;
            }
            end = k + d.len_utf8();
            chars.next();
        }
        toks.push(&s[start..i]);
        seps.push(&s[i..end]);
        start = end;
    }
    toks.push(&s[start..]);
    Split { toks, seps }
}

/// A whitespace run that holds a line break.
fn line_break(sep: &str) -> bool {
    sep.contains(['\n', '\r'])
}

fn ends_mark(t: &str) -> bool {
    t.ends_with(['.', ',', ';', ':', '!', '?'])
}

fn ends_sentence(t: &str) -> bool {
    t.ends_with(['.', '!', '?'])
}

/// Output token span holding a value: after the unique left anchor (else before the unique
/// right anchor), the maximal run of tokens accepted by `ok`, never across a line break or
/// past a token that ends a sentence (v2.3.1); hyphenated number words beside it join it;
/// trailing "and"/"point" trimmed; then its edges trimmed to number, unit or digit tokens
/// and its inner words checked (INV-6), or no span.
fn anchor_span<'a>(
    out: &'a str,
    (left, right): &Anchors,
    ok: fn(&str) -> bool,
) -> (Split<'a>, Option<(usize, usize)>) {
    let split = split_ws(out);
    let (toks, seps) = (&split.toks, &split.seps);
    let words: Vec<String> = toks.iter().map(|t| norm(t)).collect();
    let n = toks.len();
    // may token `i` join a span whose next token is `i + 1` / previous token is `i - 1`
    let joins_back = |i: usize| !(line_break(seps[i - 1]) || ends_sentence(toks[i - 1]));
    let mut span = None;
    if !left.is_empty() {
        let l = left.len();
        let pos: Vec<usize> = (0..n)
            .filter(|&i| i + 1 >= l && words[i + 1 - l..=i] == left[..])
            .collect();
        if let [p] = pos[..] {
            let j = p + 1;
            let mut k = j;
            while k < n && ok(toks[k]) && (k == j || joins_back(k)) {
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
            while j > 0 && ok(toks[j - 1]) && (j == k || joins_back(j)) {
                j -= 1;
            }
            if k > j {
                span = Some((j, k));
            }
        }
    }
    let Some((mut j, mut k)) = span else {
        return (split, None);
    };
    // v2.1: a hyphenated number word directly beside the span joins it (plain number words
    // are already inside), unless punctuation or a line break comes between
    while j > 0
        && hyph_num(toks[j - 1].trim_matches(STRIP))
        && !ends_mark(toks[j - 1])
        && !line_break(seps[j - 1])
    {
        j -= 1;
    }
    while k < n
        && hyph_num(toks[k].trim_matches(STRIP))
        && !ends_mark(toks[k - 1])
        && !line_break(seps[k - 1])
    {
        k += 1;
    }
    while k > j && matches!(norm(toks[k - 1]).as_str(), "and" | "point") {
        k -= 1;
    }
    let (j, k) = trim_span(toks, j, k);
    let span = (k > j && span_words_ok(&toks[j..k])).then_some((j, k));
    (split, span)
}

/// The text with `span` replaced by `new`, keeping the span's final punctuation mark and
/// every whitespace run outside the span (INV-5).
fn replace(split: &Split, (j, k): (usize, usize), new: &str) -> String {
    let mut new = new.to_string();
    if let Some(c) = split.toks[k - 1]
        .chars()
        .next_back()
        .filter(|c| ".,;:!?".contains(*c))
    {
        new.push(c);
    }
    let mut out = String::new();
    for i in 0..j {
        out.push_str(split.toks[i]);
        out.push_str(split.seps[i]);
    }
    out.push_str(&new);
    for i in k..split.toks.len() {
        out.push_str(split.seps[i - 1]);
        out.push_str(split.toks[i]);
    }
    out
}

// ---------------------------------------------------------------- words (INV-6, v2.3.1)

const SCALES: [&str; 4] = ["hundred", "thousand", "million", "billion"];
const UNIT_WORDS: [&str; 9] = [
    "dollars", "dollar", "bucks", "cents", "cent", "euros", "euro", "pounds", "usd",
];

/// A whitespace token as a word: punctuation stripped, lowercased; None for a value (a
/// digit or currency sign in it) or nothing.
fn word_of(t: &str) -> Option<String> {
    let s = t.trim_matches(STRIP).to_lowercase();
    (!s.is_empty() && !s.contains(|c: char| c.is_ascii_digit() || matches!(c, '$' | '€' | '£')))
        .then_some(s)
}

/// A number word: units, tens, scales, oh, double, triple, ordinals, hyphen-joined.
fn is_number_word(w: &str) -> bool {
    is_unit_or_tens(w)
        || SCALES.contains(&w)
        || matches!(w, "oh" | "double" | "triple")
        || ORDW.iter().any(|(o, _)| *o == w)
        || hyph_num(w)
}

fn is_unit_word(w: &str) -> bool {
    UNIT_WORDS.contains(&w)
}

/// A number token: a number word, or a value holding a digit.
fn is_number_token(t: &str) -> bool {
    match word_of(t) {
        Some(w) => is_number_word(&w),
        None => t.contains(|c: char| c.is_ascii_digit()),
    }
}

/// Inside a span: "a" only right before a scale word, "and" only between a number or unit
/// token and a number token, "point" and hesitations only between number tokens.
fn span_words_ok(toks: &[&str]) -> bool {
    (0..toks.len()).all(|i| {
        let Some(w) = word_of(toks[i]) else {
            return true;
        };
        if is_number_word(&w) || is_unit_word(&w) {
            return true;
        }
        let prev = i.checked_sub(1).map(|p| toks[p]);
        let next = toks.get(i + 1).copied();
        let num_prev = prev.is_some_and(is_number_token);
        let num_next = next.is_some_and(is_number_token);
        match w.as_str() {
            "a" => next
                .and_then(word_of)
                .is_some_and(|n| SCALES.contains(&n.as_str())),
            "and" => {
                (num_prev || prev.and_then(word_of).is_some_and(|p| is_unit_word(&p))) && num_next
            }
            w if w == "point" || is_hes(w) => num_prev && num_next,
            _ => false,
        }
    })
}

/// Span edges start and end on a number, unit or digit token ("a" may start it right
/// before a scale word).
fn trim_span(toks: &[&str], mut j: usize, mut k: usize) -> (usize, usize) {
    let edge_ok = |i: usize, start: bool, k: usize| match word_of(toks[i]) {
        None => true,
        Some(w) => {
            is_number_word(&w)
                || is_unit_word(&w)
                || (start
                    && w == "a"
                    && i + 1 < k
                    && word_of(toks[i + 1]).is_some_and(|n| SCALES.contains(&n.as_str())))
        }
    };
    while k > j && !edge_ok(k - 1, false, k) {
        k -= 1;
    }
    while k > j && !edge_ok(j, true, k) {
        j += 1;
    }
    (j, k)
}

/// INV-6: the words of `text` no number conversion may remove, in order.
fn kept_words(text: &str) -> Vec<String> {
    let split = split_ws(text);
    let toks: Vec<&str> = split.toks.into_iter().filter(|t| !t.is_empty()).collect();
    let words: Vec<Option<String>> = toks.iter().map(|t| word_of(t)).collect();
    let at = |i: usize| words.get(i).cloned().flatten();
    let is_ord = |w: &Option<String>| {
        w.as_deref()
            .is_some_and(|w| ORDW.iter().any(|(o, _)| *o == w))
    };
    let mut kept = Vec::new();
    for (i, w) in words.iter().enumerate() {
        let Some(w) = w else { continue };
        if is_number_word(w) || is_unit_word(w) || is_hes(w) || w == "point" {
            continue;
        }
        let prev = i.checked_sub(1).and_then(at);
        let (next, next2) = (at(i + 1), at(i + 2));
        let removable = match w.as_str() {
            "a" => next.as_deref().is_some_and(|n| SCALES.contains(&n)),
            "and" => {
                i > 0
                    && i + 1 < toks.len()
                    && (is_number_token(toks[i - 1]) || prev.as_deref().is_some_and(is_unit_word))
                    && is_number_token(toks[i + 1])
            }
            "the" => {
                is_ord(&next)
                    || (matches!(next.as_deref(), Some("twenty" | "thirty")) && is_ord(&next2))
            }
            "of" => is_ord(&prev) && next.as_deref().is_some_and(|n| MONTHS.contains(&n)),
            _ => false,
        };
        if !removable {
            kept.push(w.clone());
        }
    }
    kept
}

/// INV-5: the whitespace runs that hold a line break, in order.
fn line_breaks(text: &str) -> Vec<&str> {
    split_ws(text)
        .seps
        .into_iter()
        .filter(|s| line_break(s))
        .collect()
}

// ---------------------------------------------------------------- phone

/// A complete phone length: 7, 10, or 11 starting with 1.
fn complete(d: &str) -> bool {
    matches!(d.len(), 7 | 10) || (d.len() == 11 && d.starts_with('1'))
}

/// v4: "double" / "triple" before a digit word repeat it.
fn repeat(w: &str) -> Option<usize> {
    match w {
        "double" => Some(2),
        "triple" => Some(3),
        _ => None,
    }
}

/// Digit-by-digit runs in the input: (digits, anchors, cut). v4: "double five", "triple
/// oh" read as 55, 000. v2.2: a run one digit past a complete length that ends in a single
/// digit word followed by a non-digit word ("... four two one more time") leaves that digit
/// word out (`cut`); never after "plus" or when the word is the target of double/triple.
fn phone_runs(inp: &str) -> Vec<(String, Anchors, bool)> {
    let ws = tokens(inp);
    let low: Vec<String> = ws.iter().map(|w| w.to_lowercase()).collect();
    let rep_at = |j: usize| {
        repeat(&low[j]).filter(|_| low.get(j + 1).is_some_and(|n| digit_word(n).is_some()))
    };
    let mut res = Vec::new();
    let mut i = 0;
    while i < ws.len() {
        let w = low[i].as_str();
        if rep_at(i).is_some() || (digit_word(w).is_some() && w != "oh") || is_digits(w) {
            let (mut j, mut digs, mut nw, mut last) = (i, String::new(), 0, i);
            while j < ws.len() {
                let x = low[j].as_str();
                if let Some(times) = rep_at(j) {
                    let d = digit_word(&low[j + 1]).unwrap();
                    digs.extend(std::iter::repeat_n(d, times));
                    nw += times;
                    last = j + 1;
                    j += 2;
                    continue;
                }
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
            let by_rep = last > 0 && repeat(&low[last - 1]).is_some();
            let cut = !intl
                && !by_rep
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
        || repeat(&low).is_some()
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
        let (split, mut span) = anchor_span(&out, &anchors, numish);
        let toks = &split.toks;
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
        let Some((j, k)) = span else { continue };
        // INV-3: never touch a "+" run; the span must read exactly as the spoken run
        if plus_near(toks, j, k) || span_digits(&toks[j..k]).as_deref() != Some(digs.as_str()) {
            continue;
        }
        // INV-2: a phone span may not hold $ % : . / ( ) beside its digits
        if prot_counts(strip_tail(&toks[j..k].join(" ")))
            .iter()
            .any(|&n| n > 0)
        {
            continue;
        }
        let new = replace(&split, (j, k), &fmt_digits(&digs));
        if inv_ok(&out, &new) {
            out = new;
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
        let (split, span) = anchor_span(&out, &anchors, moneyish);
        let toks = &split.toks;
        let Some((j, k)) = span else { continue };
        let has_value = toks[j..k].iter().any(|t| {
            norm(t)
                .chars()
                .chain(t.chars())
                .any(|c| c.is_ascii_digit() || c.is_ascii_lowercase())
        });
        if !has_value || plus_near(toks, j, k) {
            continue;
        }
        let old = strip_tail(&toks[j..k].join(" ")).to_string();
        let has_number_word = toks[j..k].iter().any(|t| {
            ascii_words(&t.to_lowercase()).iter().any(|w| {
                is_unit_or_tens(w) || matches!(*w, "hundred" | "thousand" | "million" | "billion")
            }) || hyph_num(t.trim_matches(STRIP))
        });
        // INV-3: a pure regroup only when the spoken amount has the same digits
        if !has_number_word && digits(&old) != digits(&canon) {
            continue;
        }
        // INV-2 inside the span
        let (pc, po) = (prot_counts(&canon), prot_counts(&old));
        if pc.iter().zip(po.iter()).any(|(c, o)| c < o) {
            continue;
        }
        let new = replace(&split, (j, k), &canon);
        if inv_ok(&out, &new) {
            out = new;
        }
    }
    out
}

// ---------------------------------------------------------------- invariant (v2.3)

/// INV-2's protected symbols.
const PROT: [char; 10] = ['+', '$', '€', '£', '%', ':', '.', '/', '(', ')'];

/// INV-4's spoken number words: units, tens, hundred, thousand, million, billion, oh,
/// double, triple and the ordinals (hyphen-joined forms count by their parts).
fn has_number_words(inp: &str) -> bool {
    ascii_words(&inp.to_lowercase()).iter().any(|w| {
        is_unit_or_tens(w)
            || matches!(
                *w,
                "hundred" | "thousand" | "million" | "billion" | "oh" | "double" | "triple"
            )
            || ORDW.iter().any(|(o, _)| o == w)
    })
}

/// The runs of a-z in `s`.
fn ascii_words(s: &str) -> Vec<&str> {
    s.split(|c: char| !c.is_ascii_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

fn digits(s: &str) -> String {
    s.chars().filter(char::is_ascii_digit).collect()
}

/// How many of each protected symbol sit directly beside a digit.
fn prot_counts(s: &str) -> [usize; PROT.len()] {
    let chars: Vec<char> = s.chars().collect();
    let mut counts = [0; PROT.len()];
    for (i, c) in chars.iter().enumerate() {
        let Some(p) = PROT.iter().position(|x| x == c) else {
            continue;
        };
        let beside = (i > 0 && chars[i - 1].is_ascii_digit())
            || chars.get(i + 1).is_some_and(char::is_ascii_digit);
        if beside {
            counts[p] += 1;
        }
    }
    counts
}

/// The `+<digits>` runs of `s`: `\+\d(?:[\d\s().\-]*\d)?`.
fn plus_runs(s: &str) -> Vec<&str> {
    let mut runs = Vec::new();
    let mut from = 0;
    while let Some(p) = s[from..].find('+').map(|k| from + k) {
        let mut chars = s[p + 1..].char_indices();
        if !chars.next().is_some_and(|(_, c)| c.is_ascii_digit()) {
            from = p + 1;
            continue;
        }
        let mut end = p + 2;
        for (k, c) in chars {
            if c.is_ascii_digit() {
                end = p + 1 + k + 1;
            } else if !(is_space(c) || "().-".contains(c)) {
                break;
            }
        }
        runs.push(&s[p..end]);
        from = end;
    }
    runs
}

/// INV-1, INV-2 and the "+" part of INV-3 between two versions of one output: the old
/// digits are a subsequence of the new, no protected symbol beside a digit is lost, and
/// every "+<digits>" run survives verbatim.
fn inv_ok(before: &str, after: &str) -> bool {
    let new = digits(after);
    let mut rest = new.chars();
    if !digits(before).chars().all(|d| rest.any(|n| n == d)) {
        return false;
    }
    let (pb, pa) = (prot_counts(before), prot_counts(after));
    if pa.iter().zip(pb.iter()).any(|(a, b)| a < b) {
        return false;
    }
    plus_runs(before).iter().all(|run| after.contains(run))
        // v2.3.1: INV-5 line breaks kept, INV-6 words kept
        && line_breaks(before) == line_breaks(after)
        && kept_words(before) == kept_words(after)
}

/// A "+" in the span or in the token right before it.
fn plus_near(toks: &[&str], j: usize, k: usize) -> bool {
    toks[j..k].iter().any(|t| t.contains('+')) || (j > 0 && toks[j - 1].contains('+'))
}

/// One trailing `.,;:!?` removed.
fn strip_tail(s: &str) -> &str {
    s.strip_suffix(['.', ',', ';', ':', '!', '?']).unwrap_or(s)
}

/// The digits output tokens read as, in order: digits as written, digit words, and
/// double/triple before a digit word; hesitations skipped. None if anything else is there.
fn span_digits(toks: &[&str]) -> Option<String> {
    let (mut out, mut times) = (String::new(), 1);
    for t in toks {
        let mut i = 0;
        while let Some(c) = t[i..].chars().next() {
            if c.is_ascii_digit() {
                out.extend(std::iter::repeat_n(c, times));
                times = 1;
                i += 1;
            } else if c.is_ascii_alphabetic() || c == '\'' {
                let len = t[i..]
                    .find(|c: char| !(c.is_ascii_alphabetic() || c == '\''))
                    .unwrap_or(t.len() - i);
                let word = t[i..i + len].to_lowercase();
                if let Some(d) = digit_word(&word) {
                    out.extend(std::iter::repeat_n(d, times));
                    times = 1;
                } else if let Some(r) = repeat(&word) {
                    times = r;
                } else if !is_hes(&word) {
                    return None;
                }
                i += len;
            } else {
                i += c.len_utf8();
            }
        }
    }
    Some(out)
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
        // v2 changed 571; the invariant leaves wrong or missing digits the refiner wrote alone
        // (v2.3: 236; v2.3.1 also keeps "a" in "$2,500 a night" and a filler at a span edge)
        assert_eq!(changed, 228);
    }

    /// Public T13/T15/T16 outputs whose input holds a digit, a digit word, a currency word
    /// or a month (every other row cannot change): only the documented date rewrites.
    #[test]
    fn guard_rows_change_only_the_documented_dates() {
        let (n, changed) = check("guard");
        assert_eq!(n, 523);
        assert_eq!(changed, 6);
    }

    /// The revisions' own parity: pairs derived from the public ones (hyphenated number
    /// words, a digit word after a phone run, double/triple, "+", brackets, dots, % and :
    /// beside digits, line breaks, tabs, CRLF, words after amounts, sentence ends) on which
    /// the final rule and v2 differ, 100 per deciding revision.
    #[test]
    fn revision_fixes_match_the_reference() {
        let (n, _) = check("fixes");
        assert_eq!(n, 400);
    }

    /// The reference's own test file (`test_numpass_v2_3_1.py`), all 66 cases.
    #[test]
    fn reference_test_cases() {
        let (n, _) = check("reference_tests");
        assert_eq!(n, 66);
    }

    /// The swallowed digit word and the phone-length edges (v2.2), double/triple (v4).
    /// Expected values are the final reference's outputs.
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
            // INV-3: a "+" run stays exactly as written (v2.2 ran it together)
            (
                "paris is plus three three one two three four five six seven eight nine please",
                "Paris is +33 1 23 45 67 89 please.",
                "Paris is +33 1 23 45 67 89 please.",
            ),
            // v4: double/triple read as repeated digits
            (
                "call four one five triple five zero one four two now",
                "Call four one five triple five zero one four two now.",
                "Call 415-555-0142 now.",
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
        // INV-1/3: a digit the refiner dropped is not restored (the written digits stay)
        assert_eq!(
            apply(filler, "Call me at 555 123 456 tomorrow."),
            "Call me at 555 123 456 tomorrow."
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
            "+1 415 555 0199 is my cell." // INV-3: never a "+" run
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

    /// v2.3.1, INV-6: no ordinary word is lost. "a" and "and" go only inside a number
    /// ("a thousand", "two hundred and five"), never after the unit; a span stops at a
    /// sentence end. ("a hundred dollars" alone is not an amount the rule reads, as in v2.)
    #[test]
    fn words_beside_an_amount_are_kept() {
        let cases = [
            (
                "the hotel was twenty five hundred dollars a night",
                "The hotel was 2,500 dollars a night.",
                "The hotel was $2,500 a night.",
            ),
            (
                "the hotel was twenty five hundred dollars a night",
                "The hotel was $2,500 a night.",
                "The hotel was $2,500 a night.",
            ),
            (
                "it costs a thousand two hundred dollars total",
                "It costs a thousand two hundred dollars total.",
                "It costs $1,200 total.",
            ),
            (
                "it costs a hundred dollars total",
                "It costs a hundred dollars total.",
                "It costs a hundred dollars total.",
            ),
            (
                "it costs two hundred and five dollars total",
                "It costs two hundred and five dollars total.",
                "It costs $205 total.",
            ),
            (
                "we raised fifty two thousand dollars one step closer",
                "We raised fifty two thousand dollars. One step closer.",
                "We raised $52,000. One step closer.",
            ),
            (
                "it was five dollars and change",
                "It was five dollars and change.",
                "It was $5 and change.",
            ),
        ];
        for (input, output, want) in cases {
            assert_eq!(apply(input, output), want, "{output}");
        }
    }

    /// v2.3.1: the pass runs after the line breaks, so a value can end a list item or a
    /// paragraph, or sit beside a tab or CRLF. The whole value is replaced and every line
    /// break, tab and CRLF outside it stays byte for byte (INV-5).
    #[test]
    fn values_beside_line_breaks_tabs_and_crlf() {
        let cases = [
            // a list item ending in the amount
            (
                "pay forty five dollars then ship it",
                "1. Pay forty five dollars\n2. Ship it.",
                "1. Pay $45\n2. Ship it.",
            ),
            // a phone number ending a list item
            (
                "one call the desk at five five five zero one four two then ship it",
                "1. Call the desk at five five five zero one four two\n2. Ship it.",
                "1. Call the desk at 555-0142\n2. Ship it.",
            ),
            // the end of a paragraph
            (
                "it costs forty five dollars next we ship",
                "It costs forty five dollars.\n\nNext we ship.",
                "It costs $45.\n\nNext we ship.",
            ),
            // a blank line before the value
            (
                "pay forty five dollars now",
                "Pay\n\nforty five dollars now",
                "Pay\n\n$45 now",
            ),
            // a tab before it
            (
                "it costs forty five dollars for the room",
                "It costs\tforty five dollars for the room.",
                "It costs\t$45 for the room.",
            ),
            // CRLF after it
            (
                "it costs forty five dollars thanks",
                "It costs forty five dollars.\r\nThanks",
                "It costs $45.\r\nThanks",
            ),
        ];
        let mut failed = Vec::new();
        for (input, output, want) in cases {
            let got = apply(input, output);
            if got != want {
                failed.push(format!("{output:?}\n  want {want:?}\n  got  {got:?}"));
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    // ------------------------------------------------------------ invariant properties
    // Written apart from the rule's own guard (`inv_ok`), as the reference's tests are.

    /// The invariant violations of `new` against the refiner's `out` for `input`:
    /// "INV-4" a change with no spoken number word in the input, "INV-1" an existing digit
    /// removed, changed or reordered, "INV-2" a protected symbol beside a digit lost or a
    /// "+<digits>" run not kept verbatim.
    fn violations(input: &str, out: &str, new: &str) -> Vec<&'static str> {
        const WORDS: &str = "zero one two three four five six seven eight nine ten eleven twelve \
            thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty \
            sixty seventy eighty ninety hundred thousand million billion oh double triple first \
            second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth \
            fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth thirtieth";
        let words: Vec<&str> = WORDS.split_whitespace().collect();
        let lower = input.to_lowercase();
        let spoken = lower
            .split(|c: char| !c.is_ascii_lowercase())
            .any(|w| words.contains(&w));
        let mut bad = Vec::new();
        if !spoken && new != out {
            bad.push("INV-4");
        }
        let after: Vec<char> = new.chars().filter(char::is_ascii_digit).collect();
        let mut at = 0;
        for d in out.chars().filter(char::is_ascii_digit) {
            match after[at..].iter().position(|&c| c == d) {
                Some(k) => at += k + 1,
                None => {
                    bad.push("INV-1");
                    break;
                }
            }
        }
        let count = |s: &str, sym: char| {
            let c: Vec<char> = s.chars().collect();
            (0..c.len())
                .filter(|&i| {
                    c[i] == sym
                        && ((i > 0 && c[i - 1].is_ascii_digit())
                            || (i + 1 < c.len() && c[i + 1].is_ascii_digit()))
                })
                .count()
        };
        let symbols_lost = "+$€£%:./()".chars().any(|s| count(new, s) < count(out, s));
        // every "+" followed by a digit, up to its last digit before anything outside
        // digits, whitespace, ( ) . -
        let plus_lost = out.match_indices('+').any(|(p, _)| {
            let tail: Vec<char> = out[p + 1..].chars().collect();
            if !tail.first().is_some_and(char::is_ascii_digit) {
                return false;
            }
            let mut last = 0;
            for (i, c) in tail.iter().enumerate() {
                if c.is_ascii_digit() {
                    last = i;
                } else if !(c.is_whitespace() || "().-".contains(*c)) {
                    break;
                }
            }
            let run: String = std::iter::once('+')
                .chain(tail[..=last].iter().copied())
                .collect();
            !new.contains(&run)
        });
        if symbols_lost || plus_lost {
            bad.push("INV-2");
        }
        // INV-5: every whitespace run holding a line break, in order
        let breaks = |s: &str| -> Vec<String> {
            let mut runs = Vec::new();
            let mut cur = String::new();
            for c in s.chars().chain(std::iter::once('x')) {
                if c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c) {
                    cur.push(c);
                } else {
                    if cur.contains(['\n', '\r']) {
                        runs.push(cur.clone());
                    }
                    cur.clear();
                }
            }
            runs
        };
        if breaks(out) != breaks(new) {
            bad.push("INV-5");
        }
        // INV-6: the plain words no conversion may remove, in order
        let kept = |s: &str| -> Vec<String> {
            let toks: Vec<String> = s
                .split(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
                .filter(|t| !t.is_empty())
                .map(|t| t.trim_matches(STRIP).to_lowercase())
                .collect();
            let numw = |t: &str| !t.is_empty() && t.split('-').all(|p| words.contains(&p));
            let num = |t: &str| t.contains(|c: char| c.is_ascii_digit()) || numw(t);
            let losable = |t: &str| {
                numw(t)
                    || [
                        "point", "dollars", "dollar", "bucks", "cents", "cent", "euros", "euro",
                        "pounds", "usd", "um", "uh", "er", "erm", "uhm", "hmm", "ah", "umm", "uhh",
                    ]
                    .contains(&t)
            };
            let ord = |t: &str| ORDW.iter().any(|(o, _)| *o == t);
            let get = |i: usize| toks.get(i).map(String::as_str).unwrap_or("");
            let mut res = Vec::new();
            for (i, w) in toks.iter().enumerate() {
                if w.is_empty()
                    || w.contains(['$', '€', '£'])
                    || w.contains(|c: char| c.is_ascii_digit())
                {
                    continue;
                }
                let (p, n, n2) = (if i > 0 { get(i - 1) } else { "" }, get(i + 1), get(i + 2));
                let skip = losable(w)
                    || (w == "a" && ["hundred", "thousand", "million", "billion"].contains(&n))
                    || (w == "and" && (num(p) || losable(p)) && num(n))
                    || (w == "the" && (ord(n) || (["twenty", "thirty"].contains(&n) && ord(n2))))
                    || (w == "of" && ord(p) && MONTHS.contains(&n));
                if !skip {
                    res.push(w.clone());
                }
            }
            res
        };
        if kept(out) != kept(new) {
            bad.push("INV-6");
        }
        bad
    }

    /// The invariant holds on every row of the fixture (public outputs, derived pairs and
    /// the reference's test cases), for the rule's actual output.
    #[test]
    fn invariant_holds_on_every_fixture_row() {
        // the checker itself catches each kind of harm
        assert_eq!(
            violations("hi there", "Room 5", "Room 6"),
            ["INV-4", "INV-1"]
        );
        assert_eq!(violations("one", "at +1 415", "at 1-415"), ["INV-2"]);
        assert_eq!(violations("one", "45% off", "45 off"), ["INV-2"]);
        assert_eq!(violations("one", "415 555", "555 415"), ["INV-1"]);
        assert!(violations("one", "415 555", "415-555 1").is_empty());
        assert_eq!(violations("five", "five\n2. go", "5 2. go"), ["INV-5"]);
        assert_eq!(violations("five", "$5 a night", "$5 night"), ["INV-6"]);
        assert!(violations("five", "a hundred and five dollars", "$105").is_empty());
        assert!(violations("one", "the first of May", "May 1").is_empty());
        let cases = cases();
        let mut n = 0;
        for set in ["parity", "guard", "fixes", "reference_tests"] {
            for case in cases[set].as_array().unwrap() {
                let input = case["input"].as_str().unwrap();
                let output = case["output"].as_str().unwrap();
                let new = apply(input, output);
                assert_eq!(
                    violations(input, output, &new),
                    Vec::<&str>::new(),
                    "{output}"
                );
                n += 1;
            }
        }
        assert_eq!(n, 1693 + 523 + 400 + 66);
    }

    /// Seeded random dictations (number words, digits already written, "+", $ % : . / ( ),
    /// hyphens, dates, money, fillers) and random refiner outputs, part derived from the
    /// input: the invariant holds on every one, and an input with no number word leaves the
    /// output exactly as it was.
    #[test]
    fn invariant_holds_on_seeded_random_text() {
        const SPOKEN: &[&str] = &[
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "oh",
            "ten", "twelve", "fifteen", "twenty", "thirty", "forty", "fifty", "ninety", "hundred",
            "thousand", "million", "first", "second", "fifth", "twenty", "double", "triple",
            "point", "and", "cents", "dollars", "bucks", "euros", "pounds", "plus", "percent",
            "april", "may", "the", "of", "um", "uh", "call", "me", "at", "room", "costs", "on",
            "is", "more", "time", "people", "please", ",", ".",
        ];
        const WRITTEN: &[&str] = &[
            "415",
            "555",
            "0142",
            "1",
            "+1",
            "+44",
            "(415)",
            "555-0142",
            "415.555.0142",
            "$45",
            "$1,200",
            "€30",
            "£64",
            "45%",
            "10:30",
            "3/4",
            "v2.10.1",
            "2026",
            "1200",
            "12.50",
            "forty-five",
            "four-one-five",
            "twenty-first",
            "one",
            "five",
            "two",
            "four",
            "triple",
            "double",
            "dollars",
            "April",
            "May",
            "the",
            "of",
            "call",
            "me",
            "at",
            "Room",
            "costs",
            "is",
            "more",
            "time",
            "people",
            "please",
            "um",
            ",",
            ".",
            "-",
            "+",
            "(",
            ")",
            "a",
            "night",
            "and",
            "change",
            "dollars.",
            "One",
            "2.",
        ];
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n as u64) as usize
        };
        let (mut changed, mut identity_cases) = (0, 0);
        for case in 0..20_000 {
            let len = 3 + next(14);
            let input: Vec<&str> = (0..len).map(|_| SPOKEN[next(SPOKEN.len())]).collect();
            let input = if case % 5 == 0 {
                // no number word at all: INV-4 must return the output untouched
                identity_cases += 1;
                "call me at the room please".to_string()
            } else {
                input.join(" ")
            };
            // an output that keeps part of the input's words and writes other tokens
            let mut out: Vec<&str> = Vec::new();
            for w in input.split(' ') {
                match next(4) {
                    0 => out.push(WRITTEN[next(WRITTEN.len())]),
                    1 => {}
                    _ => out.push(w),
                }
            }
            for _ in 0..next(3) {
                out.insert(next(out.len() + 1), WRITTEN[next(WRITTEN.len())]);
            }
            // separators, with line breaks, blank lines, tabs and CRLF around and inside
            // spans (v2.3.1)
            let seps = [
                " ", " ", " ", " ", ", ", "-", "\n", "\n\n", "\t", "\r\n", ".\n",
            ];
            let mut text = String::new();
            for (i, t) in out.iter().enumerate() {
                if i > 0 {
                    text.push_str(seps[next(seps.len())]);
                }
                text.push_str(t);
            }
            let out = text + [".", "", "?", "\n"][next(4)];
            let new = apply(&input, &out);
            changed += usize::from(new != out);
            assert_eq!(
                violations(&input, &out, &new),
                Vec::<&str>::new(),
                "input: {input}\nout: {out}\nnew: {new}"
            );
        }
        // the generator exercises real rewrites, not only the identity
        assert!(changed > 200, "{changed}");
        assert_eq!(identity_cases, 4_000);
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
