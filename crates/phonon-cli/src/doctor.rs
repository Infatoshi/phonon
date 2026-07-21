//! Quick dependency / model presence check for the local dictation stack.

use anyhow::Result;
use phonon_core::data::{list_recordings, DictionaryFile, SettingsFile};
use phonon_core::project_root;
use phonon_llm::{fluid1_available, fluid_drafter_dir, fluid_helper, fluid_model_dir};

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
        "prompts/polish_v1.txt",
        root.join("prompts/polish_v1.txt").is_file(),
    );
    check(
        "bar/Package.swift",
        root.join("bar/Package.swift").is_file(),
    );
    check(
        "signed bar app (bar/dist/Phonon.app)",
        root.join("bar/dist/Phonon.app/Contents/MacOS/PhononBar")
            .is_file(),
    );
    check("uv", which::which("uv").is_ok());
    check("rec (SoX)", which::which("rec").is_ok());
    check("pbcopy", which::which("pbcopy").is_ok());
    check("swift", which::which("swift").is_ok());

    let helper = fluid_helper();
    let model = fluid_model_dir();
    let drafter = fluid_drafter_dir();
    check(
        &format!("fluid-intelligence-mlx ({})", helper.display()),
        helper.is_file(),
    );
    check(
        &format!("fluid-1 model ({})", model.display()),
        model.is_dir(),
    );
    check(
        &format!("MTP drafter ({})", drafter.display()),
        drafter.is_dir(),
    );
    check("fluid-1 polish usable", fluid1_available());

    let dictionary = DictionaryFile::load();
    let dictionary_count = dictionary
        .as_ref()
        .map(|value| value.entries.len())
        .unwrap_or(0);
    check(
        &format!("JSON dictionary ({dictionary_count} entries)"),
        dictionary.is_ok() && dictionary_count > 0,
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
