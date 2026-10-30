//! Replay observed edits through Phonon's edit classifier.
//!
//! Input: a JSON array (or JSONL) of objects with `pasted` and `edited`, and
//! optionally `id` and `edited_unbounded`. Output, one JSON line per row: the
//! word and punctuation edit vectors, the outcome and the proposals. Used to
//! check the alignment against Wispr Flow's own stored `userEditMetaData`
//! (`sqlite3 -json flow.sqlite "select ... from History"`) without any of the
//! text leaving the machine.
//!
//! cargo run -p phonon-core --example edit_replay -- rows.json > replay.jsonl
//!
//! `--repeat` counts repeated capitalizations across rows the way the
//! engine does across one session (`edits::RepeatedEdits`).

use phonon_core::data::DictionaryFile;
use phonon_core::edits;
use serde_json::{json, Value};
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut repeats = args
        .iter()
        .any(|arg| arg == "--repeat")
        .then(edits::RepeatedEdits::default);
    let path = args
        .iter()
        .find(|arg| !arg.starts_with("--"))
        .ok_or_else(|| anyhow::anyhow!("usage: edit_replay [--repeat] ROWS.json"))?;
    let text = std::fs::read_to_string(path)?;
    let rows: Vec<Value> = match serde_json::from_str::<Value>(&text) {
        Ok(Value::Array(rows)) => rows,
        _ => text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?,
    };
    let dictionary = DictionaryFile::default();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for row in rows {
        let field = |key: &str| row.get(key).and_then(Value::as_str).unwrap_or("");
        let pasted = strip_markup(field("pasted"));
        let edited = strip_markup(field("edited"));
        let steps = edits::align(&pasted, &edited);
        let (word_edits, punctuation_edits) = edits::edit_vectors(&steps);
        let mut analysis = edits::analyze(&pasted, &edited, None, &dictionary);
        if let Some(repeats) = repeats.as_mut() {
            repeats.promote(&mut analysis);
        }
        writeln!(
            out,
            "{}",
            json!({
                "id": row.get("id"),
                "word_edits": word_edits,
                "punctuation_edits": punctuation_edits,
                "outcome": analysis.outcome,
                "suggestions": analysis.suggestions,
                "skipped": analysis.skipped,
            })
        )?;
    }
    Ok(())
}

/// Wispr's `stripFormattingMarkupTags`: list tags out, then trim.
fn strip_markup(text: &str) -> String {
    let mut output = text.to_string();
    for tag in ["<ol>", "</ol>", "<li>", "</li>", "<ul>", "</ul>"] {
        output = output.replace(tag, "");
    }
    output.trim().to_string()
}
