//! `phonon data export-pairs` end to end, on a fixture corpus under a
//! temporary HOME.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn temp_home(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let home = std::env::temp_dir().join(format!(
        "phonon-export-{name}-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&home).unwrap();
    home
}

fn entry(corpus: &Path, id: &str, unix_ms: u64, refined: &str, edited: Option<&str>) {
    let directory = corpus.join(id);
    fs::create_dir_all(&directory).unwrap();
    let user_edit = edited.map(|text| {
        serde_json::json!({
            "schema_version": 1, "pasted_text": refined, "edited_text": text,
            "outcome": "correction", "word_edits": "MS", "punctuation_edits": "ZZ",
            "edit_distance": 0.5, "end_reason": "observation_window_elapsed", "observed_ms": 60000,
        })
    });
    let metadata = serde_json::json!({
        "schema_version": 2, "id": id, "created_at_unix_ms": unix_ms, "source": "bar",
        "audio_file": "audio.wav", "raw_transcript": refined.to_lowercase(),
        "final_transcript": refined, "user_edit": user_edit,
    });
    fs::write(
        directory.join("metadata.json"),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
}

const OCT_01: u64 = 1_790_812_800_000;
const DAY: u64 = 86_400_000;

fn export(home: &Path, extra: &[&str]) -> (Vec<serde_json::Value>, String) {
    let out = home.join("pairs.jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_phonon"))
        .env("HOME", home)
        .args(["data", "export-pairs", "--eval-from", "2026-10-10", "--out"])
        .arg(&out)
        .args(extra)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(output.status.success(), "{stderr}");
    let rows = fs::read_to_string(&out)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    (rows, stderr)
}

#[test]
fn exports_corpus_pairs_with_a_date_split() {
    let home = temp_home("split");
    let corpus = home.join("Library/Application Support/Phonon/Corpus");
    entry(
        &corpus,
        "early",
        OCT_01,
        "Ask clod code.",
        Some("Ask Claude Code."),
    );
    entry(&corpus, "late", OCT_01 + 12 * DAY, "Ship it today.", None);

    let (rows, stderr) = export(&home, &[]);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["id"], "early");
    assert_eq!(rows[0]["refined_text"], "Ask clod code.");
    assert_eq!(rows[0]["edited_text"], "Ask Claude Code.");
    assert_eq!(rows[0]["edited"], true);
    assert_eq!(rows[0]["split"], "train");
    assert_eq!(rows[1]["edited_text"], "Ship it today.");
    assert_eq!(rows[1]["edited"], false);
    assert_eq!(rows[1]["split"], "eval");
    assert!(
        stderr.contains("2 rows (2 phonon, 0 wispr): 1 edited, 1 train, 1 eval from 2026-10-10")
    );

    let (edited, _) = export(&home, &["--edited-only"]);
    assert_eq!(edited.len(), 1);
    let (recent, _) = export(&home, &["--since", "2026-10-05"]);
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0]["id"], "late");

    let ids = home.join("holdout.txt");
    fs::write(&ids, "early\n").unwrap();
    let (held, _) = export(&home, &["--holdout-ids", ids.to_str().unwrap()]);
    assert_eq!(held[0]["split"], "eval");
    fs::remove_dir_all(home).unwrap();
}

#[test]
fn audio_needs_a_wispr_source() {
    let home = temp_home("audio");
    let output = Command::new(env!("CARGO_BIN_EXE_phonon"))
        .env("HOME", &home)
        .args(["data", "export-pairs", "--with-audio"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    fs::remove_dir_all(home).unwrap();
}
