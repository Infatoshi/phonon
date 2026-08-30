//! Quick dependency / model presence check for the local dictation stack.

use anyhow::Result;
use phonon_core::data::{list_recordings, DictionaryFile, SettingsFile};
use phonon_core::project_root;
use phonon_llm::{
    fluid1_available, polish_available, POLISH_MODEL_ID, POLISH_MODEL_REVISION,
    POLISH_RUNTIME_REQUIREMENT,
};

pub fn run_doctor() -> Result<()> {
    let root = project_root();
    println!("phonon doctor");
    println!("root: {}", root.display());
    println!();

    check(
        "sidecar/asr_server.py",
        root.join("sidecar/asr_server.py").is_file(),
    );
    check(
        "sidecar/polish_server.py",
        root.join("sidecar/polish_server.py").is_file(),
    );
    check(
        "prompts/polish_v2.txt",
        root.join("prompts/polish_v2.txt").is_file(),
    );
    // Without this list the phonetic guard cannot tell an ordinary English word
    // from a technical term, so retrieval turns itself off rather than guess.
    check(
        "assets/english_words.txt (phonetic guard)",
        root.join("assets/english_words.txt").is_file(),
    );
    let installed_app = root
        .parent()
        .map(|contents| contents.join("MacOS/PhononBar"))
        .filter(|binary| binary.is_file());
    if installed_app.is_some() {
        check("native macOS app", true);
    } else {
        check(
            "bar/Package.swift",
            root.join("bar/Package.swift").is_file(),
        );
        check(
            "signed bar app (bar/dist/Phonon.app)",
            root.join("bar/dist/Phonon.app/Contents/MacOS/PhononBar")
                .is_file(),
        );
    }
    check("uv", crate::resolve_runtime_tool("uv").is_some());
    // Both sidecars start from uv's cache alone once these pass, so a Mac with
    // no network still dictates. A fresh install fails them until first launch.
    if let Some(uv) = crate::resolve_runtime_tool("uv") {
        optional(
            "ASR runtime cached for offline start",
            phonon_asr::uv_offline_ready(&uv, phonon_asr::ASR_RUNTIME_REQUIREMENT),
        );
        optional(
            "correction runtime cached for offline start",
            phonon_asr::uv_offline_ready(&uv, POLISH_RUNTIME_REQUIREMENT),
        );
    }
    // The native app records through CoreAudio. SoX only backs `phonon` runs
    // from a terminal, so a fresh Mac without it is not a broken install.
    optional(
        "terminal recording (SoX, optional)",
        which::which("rec").is_ok(),
    );
    check("pbcopy", which::which("pbcopy").is_ok());
    check("swift", which::which("swift").is_ok());

    check(
        "required correction stage installed",
        polish_available(&root),
    );
    println!(
        "  correction model: {POLISH_MODEL_ID}@{}",
        &POLISH_MODEL_REVISION[..7]
    );
    println!("  correction runtime: {POLISH_RUNTIME_REQUIREMENT}");
    println!(
        "  fluid-1 baseline (developer comparison only): {}",
        if fluid1_available() {
            "installed"
        } else {
            "absent"
        }
    );

    let dictionary = DictionaryFile::load();
    let dictionary_count = dictionary
        .as_ref()
        .map(|value| value.entries.len())
        .unwrap_or(0);
    // Entries are earned by use; empty is correct on a fresh install. Only a
    // dictionary that fails to load is a problem.
    check(
        &format!("JSON dictionary ({dictionary_count} entries)"),
        dictionary.is_ok(),
    );
    check("JSON settings", SettingsFile::load_or_create().is_ok());
    let recordings = list_recordings();
    let corpus_count = recordings.as_ref().map(|value| value.len()).unwrap_or(0);
    check(
        &format!("paired WAV corpus ({corpus_count} entries)"),
        recordings.is_ok(),
    );

    println!();
    println!("ok paths:");
    println!("  phonon           # native macOS app — Right ⌥ push-to-talk");
    println!("  phonon bar       # explicit native-app launcher");
    println!("  phonon bar --rebuild");
    println!("  phonon engine    # warm backend for the bar");
    println!("  phonon bench");
    println!("  phonon dictionary --help");
    println!("  phonon corpus --help");
    Ok(())
}

fn check(label: &str, ok: bool) {
    let mark = if ok { "ok  " } else { "MISS" };
    println!("  [{mark}] {label}");
}

/// Something whose absence is fine, so it never reads as a failed install.
fn optional(label: &str, present: bool) {
    let mark = if present { "ok  " } else { "  - " };
    println!("  [{mark}] {label}");
}
