//! Source-code identifier extract matching `profile_miner.extract.extract_code`.
//!
//! Identifiers in a file are unique and sorted (not first-seen order). Repo and
//! file order follow CPython 3.12 `os.walk` (all files in a directory, then
//! subdirs in `readdir` order). `walkdir`'s interleaved DFS would cut different
//! files at the per-repo / total caps.

mod extract;

use std::collections::{BTreeSet, HashSet};
use std::sync::OnceLock;

use regex::bytes::Regex;

pub use extract::{
    extract_code, find_repos, write_extract_code, ExtractLimits, ExtractStats, StatVal,
};

/// Same pattern as Python `RE_CODE_TOKEN`. `[A-Za-z0-9]` is ASCII, not Unicode.
pub const RE_CODE_TOKEN: &str = r"[A-Za-z0-9]+(?:[\-_.][A-Za-z0-9]+)*";

pub const CODE_MAX_BYTES: u64 = 400_000;
pub const CODE_MAX_FILES_PER_REPO: usize = 4000;
pub const CODE_MAX_FILES: usize = 120_000;
pub const FIND_REPOS_MAX_DEPTH: usize = 3;

/// Repo basename (case-insensitive) skipped because its docs quote the held-out set.
pub const REPO_DOC_EXCLUDE: &str = "phonon";

pub const CODE_SKIP_SUFFIX: [&str; 4] = [".min.js", ".lock", "-lock.json", ".ipynb"];

/// Copied from `profile_miner.extract.CODE_STOP`; Python `str.split()` on this string.
const CODE_STOP_SRC: &str = r#"
def class return self none true false import from as with pass break continue lambda yield raise global del assert
print len range int str float bool list dict set tuple super isinstance type object elif else while for if in is not
and or async await try except finally nonlocal
fn let mut pub impl struct enum match use mod const static crate where dyn ref move unsafe trait loop
some ok err vec string option result box usize isize u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64 char
void long double unsigned signed sizeof typedef include define ifdef ifndef endif pragma template typename namespace
using public private protected virtual override nullptr auto constexpr inline extern static_cast dynamic_cast
reinterpret_cast const_cast noexcept
function export default new this null undefined interface implements extends throw catch switch case do
instanceof typeof console log var require module exports
guard func init deinit weak strong protocol extension throws rethrows nil self any
package go chan defer map make range select
elsif unless begin end then rescue ensure nil
"#;

pub fn code_stop() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&str>> = OnceLock::new();
    SET.get_or_init(|| CODE_STOP_SRC.split_whitespace().collect())
}

pub fn is_tree_skip(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | "node_modules"
            | "target"
            | ".venv"
            | "venv"
            | "__pycache__"
            | ".build"
            | "dist"
            | "build"
            | ".cache"
    )
}

pub fn is_code_ext(ext: &str) -> bool {
    matches!(
        ext,
        "py" | "rs"
            | "swift"
            | "ts"
            | "tsx"
            | "js"
            | "jsx"
            | "go"
            | "c"
            | "cc"
            | "cpp"
            | "cu"
            | "cuh"
            | "h"
            | "hpp"
            | "sh"
            | "zsh"
            | "toml"
            | "yaml"
            | "yml"
            | "cmake"
            | "mm"
            | "kt"
            | "java"
            | "rb"
            | "lua"
            | "sql"
            | "proto"
            | "md"
            | "txt"
    )
}

/// File stem extension after the last dot, lowercased. Empty if `fname` has no dot.
pub fn file_ext(fname: &str) -> String {
    if !fname.contains('.') {
        String::new()
    } else {
        fname
            .rsplit_once('.')
            .map(|(_, e)| e)
            .unwrap_or("")
            .to_ascii_lowercase()
    }
}

/// Same file selection as `extract_code`: extension, skip suffixes, leading dot.
pub fn select_code_file(fname: &str) -> bool {
    if fname.starts_with('.') {
        return false;
    }
    if CODE_SKIP_SUFFIX.iter().any(|s| fname.ends_with(s)) {
        return false;
    }
    is_code_ext(&file_ext(fname))
}

fn code_token_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(RE_CODE_TOKEN).expect("RE_CODE_TOKEN"))
}

/// Unique identifiers from one file, sorted (Python `set` then `sorted`).
pub fn tokenize(text: &[u8]) -> BTreeSet<String> {
    let stop = code_stop();
    let mut toks = BTreeSet::new();
    for m in code_token_re().find_iter(text) {
        let t = m.as_bytes();
        if t.len() < 2 || t.len() > 40 {
            continue;
        }
        if !t.iter().any(u8::is_ascii_alphabetic) {
            continue;
        }
        let s = std::str::from_utf8(t).expect("regex token is ASCII");
        let mut buf = [0u8; 40];
        let lower = ascii_lower(s, &mut buf);
        if stop.contains(lower) {
            continue;
        }
        toks.insert(s.to_string());
    }
    toks
}

fn ascii_lower<'a>(s: &'a str, buf: &'a mut [u8; 40]) -> &'a str {
    if s.bytes().all(|b| !b.is_ascii_uppercase()) {
        return s;
    }
    let n = s.len();
    buf[..n].copy_from_slice(s.as_bytes());
    for b in &mut buf[..n] {
        b.make_ascii_lowercase();
    }
    std::str::from_utf8(&buf[..n]).expect("ascii")
}

pub fn join_tokens(toks: &BTreeSet<String>) -> String {
    let mut out = String::new();
    for (i, t) in toks.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(t);
    }
    out
}

pub fn default_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME")
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from)
}

pub fn phonon_support_dir(home: &std::path::Path) -> std::path::PathBuf {
    home.join("Library")
        .join("Application Support")
        .join("Phonon")
}

/// Whether `path` starts with `prefix` by path components.
pub fn path_str_prefix(path: &std::path::Path, prefix: &std::path::Path) -> bool {
    path.starts_with(prefix)
}

pub fn repo_docs_excluded(name: &str) -> bool {
    name.eq_ignore_ascii_case(REPO_DOC_EXCLUDE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str) -> String {
        join_tokens(&tokenize(text.as_bytes()))
    }

    #[test]
    fn tokenize_unique_sorted_not_first_seen() {
        assert_eq!(line("foo bar foo"), "bar foo");
        assert_eq!(line("zzz aaa MMM"), "MMM aaa zzz");
    }

    #[test]
    fn tokenize_compound_and_split_separators() {
        assert_eq!(line("foo-bar foo_bar foo.bar"), "foo-bar foo.bar foo_bar");
        assert_eq!(line("foo--bar foo-.bar"), "bar foo");
        assert_eq!(line("a-b_c.d e"), "a-b_c.d");
        assert_eq!(line("_foo foo_ foo__bar"), "bar foo");
    }

    #[test]
    fn tokenize_length_and_alpha() {
        let long41 = "x".repeat(41);
        let long40 = "y".repeat(40);
        let s = format!("a ab 123 123a {long41} {long40}");
        assert_eq!(line(&s), format!("123a ab {long40}"));
    }

    #[test]
    fn tokenize_ascii_not_unicode_letters() {
        assert_eq!(line("café naive"), "caf naive");
    }

    #[test]
    fn tokenize_replacement_splits() {
        assert_eq!(line("hello\u{FFFD}world"), "hello world");
        assert_eq!(join_tokens(&tokenize(b"hello\xffworld")), "hello world");
    }

    #[test]
    fn stop_list_filters_keywords_keeps_idents() {
        assert_eq!(
            line("def foo_bar():\n    return HELLO_TOKEN\n"),
            "HELLO_TOKEN foo_bar"
        );
        assert_eq!(line("fn let ident_alpha"), "ident_alpha");
        // `OK`.lower() is `ok`, which is in CODE_STOP (Rust Result).
        assert_eq!(line("self Self OK I"), "");
        // `FOR`.lower() is `for` (Python keyword), so both drop.
        assert_eq!(line("FOR for Foo foo"), "Foo foo");
        assert!(code_stop().contains("fn"));
        assert!(code_stop().contains("self"));
        assert!(code_stop().contains("nil"));
        assert!(!code_stop().contains("phonon"));
        assert_eq!(code_stop().len(), 175);
    }

    #[test]
    fn skip_rules_extension_suffix_dotfile() {
        assert!(select_code_file("foo.py"));
        assert!(select_code_file("FOO.RS"));
        assert!(select_code_file("CMakeLists.txt"));
        assert!(select_code_file("a.cu"));
        assert!(!select_code_file("foo.json"));
        assert!(!select_code_file("foo.min.js"));
        assert!(!select_code_file("Cargo.lock"));
        assert!(!select_code_file("package-lock.json"));
        assert!(!select_code_file("notes.ipynb"));
        assert!(!select_code_file(".hidden.rs"));
        assert!(!select_code_file("Makefile"));
        assert!(!select_code_file("foo."));
    }

    #[test]
    fn tree_skip_names() {
        assert!(is_tree_skip("node_modules"));
        assert!(is_tree_skip("target"));
        assert!(is_tree_skip("build"));
        assert!(!is_tree_skip("src"));
    }

    #[test]
    fn path_str_prefix_uses_path_components() {
        use std::path::Path;
        let prefix = Path::new("/Users/alice/Library/Application Support/Phonon");
        assert!(path_str_prefix(
            Path::new("/Users/alice/Library/Application Support/Phonon/sub"),
            prefix
        ));
        assert!(!path_str_prefix(
            Path::new("/Users/alice/Library/Application Support/PhononExtra"),
            prefix
        ));
    }
}
