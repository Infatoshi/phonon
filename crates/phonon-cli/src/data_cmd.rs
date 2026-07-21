use anyhow::{bail, Context, Result};
use phonon_core::data::{
    app_support_dir, corpus_dir, delete_recording, dictionary_path, import_recording,
    list_recordings, load_recording_by_id, safe_polish_output, set_intended_transcript,
    settings_path, usage_stats, DictionaryEntry, DictionaryFile, SettingsFile,
};
use phonon_llm::ServeJson;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Deserialize)]
struct WisprEntry {
    phrase: String,
    replacement: Option<String>,
    #[serde(default)]
    source: String,
    #[serde(default, rename = "isStarred")]
    is_starred: u64,
    #[serde(default, rename = "frequencyUsed")]
    frequency_used: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorrectionCase {
    id: String,
    raw: String,
    intended: String,
}

#[derive(Debug, Serialize)]
struct CorrectionEvaluation {
    id: String,
    raw: String,
    intended: String,
    output: String,
    passed: bool,
    latency_ms: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LegacyHistoryEntry {
    audio_path: Option<PathBuf>,
    timestamp: Option<String>,
    transcript: String,
}

pub fn default_wispr_database() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join("Library/Application Support/Wispr Flow/flow.sqlite"))
}

pub fn default_dictionary_terms_path() -> Result<PathBuf> {
    Ok(app_support_dir()?.join("dictionary_new_terms.txt"))
}

pub fn import_txt(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!("dictionary terms file not found: {}", path.display());
    }
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("read dictionary terms from {}", path.display()))?;
    let entries = contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|phrase| DictionaryEntry {
            phrase: phrase.to_string(),
            replacement: None,
            spoken_forms: Vec::new(),
            source: "phonon-term-list".into(),
            starred: false,
            usage_count: 0,
        });
    let mut dictionary = DictionaryFile::load()?;
    let imported = dictionary.merge(entries);
    dictionary.save()?;
    println!(
        "imported {imported} new entries; {} total\n{}",
        dictionary.entries.len(),
        dictionary_path()?.display()
    );
    Ok(())
}

pub fn import_wispr(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!("Wispr Flow database not found: {}", path.display());
    }
    let query = "SELECT phrase,replacement,source,isStarred,frequencyUsed FROM Dictionary WHERE isDeleted=0 AND isSnippet=0 ORDER BY phrase COLLATE NOCASE";
    let output = Command::new("sqlite3")
        .arg("-json")
        .arg(path)
        .arg(query)
        .output()
        .context("run sqlite3 to read Wispr Flow dictionary")?;
    if !output.status.success() {
        bail!(
            "sqlite3 failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let wispr: Vec<WisprEntry> =
        serde_json::from_slice(&output.stdout).context("parse Wispr Flow dictionary export")?;
    let mut dictionary = DictionaryFile::load()?;
    let imported = dictionary.merge(wispr.iter().map(|entry| DictionaryEntry {
        phrase: entry.phrase.clone(),
        replacement: entry.replacement.clone(),
        spoken_forms: Vec::new(),
        source: if entry.source.is_empty() {
            "wispr-flow".into()
        } else {
            format!("wispr-flow:{}", entry.source)
        },
        starred: entry.is_starred != 0,
        usage_count: entry.frequency_used,
    }));
    dictionary.save()?;
    println!(
        "imported {imported} new entries; {} total\n{}",
        dictionary.entries.len(),
        dictionary_path()?.display()
    );
    Ok(())
}

pub fn list_dictionary(json: bool) -> Result<()> {
    let dictionary = DictionaryFile::load()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&dictionary)?);
        return Ok(());
    }
    println!("{} dictionary entries", dictionary.entries.len());
    for entry in dictionary.entries {
        match entry.replacement {
            Some(replacement) => println!("{} -> {replacement}", entry.phrase),
            None => println!("{}", entry.phrase),
        }
    }
    Ok(())
}

pub fn add_dictionary(
    phrase: String,
    replacement: Option<String>,
    spoken_forms: Vec<String>,
) -> Result<()> {
    let mut dictionary = DictionaryFile::load()?;
    let added = dictionary.merge([DictionaryEntry {
        phrase,
        replacement,
        spoken_forms,
        source: "phonon-manual".into(),
        starred: true,
        usage_count: 0,
    }]);
    dictionary.save()?;
    println!(
        "{}; {} total",
        if added == 1 { "added" } else { "updated" },
        dictionary.entries.len()
    );
    Ok(())
}

pub fn learn_from_recording(id: &str, from: &str, to: &str) -> Result<()> {
    let recording = load_recording_by_id(id)?;
    if !recording
        .raw_transcript
        .to_lowercase()
        .contains(&from.to_lowercase())
    {
        bail!("{from:?} does not appear in recording {id}'s raw transcript");
    }
    let mut dictionary = DictionaryFile::load()?;
    dictionary.merge([DictionaryEntry {
        phrase: from.to_string(),
        replacement: Some(to.to_string()),
        spoken_forms: Vec::new(),
        source: format!("phonon-history:{id}"),
        starred: true,
        usage_count: 0,
    }]);
    dictionary.save()?;
    println!("learned {from} -> {to} from {id}");
    Ok(())
}

pub fn test_dictionary(text: &str, screen_context: &str, json: bool) -> Result<()> {
    let dictionary = DictionaryFile::load()?;
    let candidates = dictionary
        .relevant_entries(text)
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    let corrected = dictionary.apply_exact_replacements(text);
    let screen_confirmed_terms = dictionary.screen_confirmed_terms(text, screen_context);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "input": text,
                "candidates": candidates,
                "screen_confirmed_terms": screen_confirmed_terms,
                "polish_input": dictionary.prepare_polish_input_with_context(text, screen_context),
                "output": corrected.text,
                "applied": corrected.applied,
            }))?
        );
    } else {
        println!("input:  {text}");
        println!("output: {}", corrected.text);
        println!("candidates: {}", candidates.len());
        for entry in candidates {
            println!("  {} -> {}", entry.phrase, entry.canonical());
        }
        if !screen_confirmed_terms.is_empty() {
            println!("screen confirmed: {}", screen_confirmed_terms.join(", "));
        }
    }
    Ok(())
}

pub fn evaluate_dictionary(root: &Path, fixtures: &Path, json: bool) -> Result<()> {
    let cases: Vec<CorrectionCase> = serde_json::from_slice(
        &std::fs::read(fixtures).with_context(|| format!("read {}", fixtures.display()))?,
    )
    .with_context(|| format!("parse {}", fixtures.display()))?;
    let dictionary = DictionaryFile::load()?;
    let mut helper = ServeJson::spawn(root, true)?;
    let (_, warmup) = helper.warmup()?;
    if !warmup.ok {
        bail!("LLM warmup failed: {:?}", warmup.error);
    }
    let mut evaluations = Vec::new();
    for case in cases {
        let input = dictionary.prepare_polish_input(&case.raw);
        let (wall, response) = helper.run_text(&input)?;
        if !response.ok {
            bail!("correction case {} failed: {:?}", case.id, response.error);
        }
        let model_output = response.output_text.unwrap_or_default();
        let output = safe_polish_output(&dictionary, &case.raw, &model_output).text;
        evaluations.push(CorrectionEvaluation {
            passed: comparable(&output) == comparable(&case.intended),
            id: case.id,
            raw: case.raw,
            intended: case.intended,
            output,
            latency_ms: wall.as_secs_f64() * 1_000.0,
        });
    }
    helper.shutdown();
    if json {
        println!("{}", serde_json::to_string_pretty(&evaluations)?);
    } else {
        let passed = evaluations
            .iter()
            .filter(|evaluation| evaluation.passed)
            .count();
        println!(
            "dictionary correction: {passed}/{} passed",
            evaluations.len()
        );
        for evaluation in evaluations {
            let status = if evaluation.passed { "PASS" } else { "FAIL" };
            println!(
                "{status} {} ({:.0}ms)",
                evaluation.id, evaluation.latency_ms
            );
            if !evaluation.passed {
                println!("  raw:      {}", evaluation.raw);
                println!("  intended: {}", evaluation.intended);
                println!("  output:   {}", evaluation.output);
            }
        }
    }
    Ok(())
}

pub fn show_paths() -> Result<()> {
    std::fs::create_dir_all(corpus_dir()?)?;
    SettingsFile::load_or_create()?;
    println!("data:       {}", app_support_dir()?.display());
    println!("settings:   {}", settings_path()?.display());
    println!("dictionary: {}", dictionary_path()?.display());
    println!("corpus:     {}", corpus_dir()?.display());
    Ok(())
}

pub fn list_corpus(search: Option<&str>, json: bool) -> Result<()> {
    let search = search.map(str::to_lowercase);
    let recordings = list_recordings()?
        .into_iter()
        .filter(|recording| {
            search.as_ref().is_none_or(|query| {
                recording.raw_transcript.to_lowercase().contains(query)
                    || recording.final_transcript.to_lowercase().contains(query)
                    || recording
                        .intended_transcript
                        .as_deref()
                        .unwrap_or_default()
                        .to_lowercase()
                        .contains(query)
            })
        })
        .collect::<Vec<_>>();
    if json {
        println!("{}", serde_json::to_string_pretty(&recordings)?);
    } else {
        for recording in recordings {
            let transcript = if recording.final_transcript.is_empty() {
                &recording.raw_transcript
            } else {
                &recording.final_transcript
            };
            println!(
                "{}  {}  {}",
                recording.id,
                recording.source,
                transcript.chars().take(100).collect::<String>()
            );
        }
    }
    Ok(())
}

pub fn show_recording(id: &str) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&load_recording_by_id(id)?)?
    );
    Ok(())
}

pub fn migrate_legacy_history() -> Result<()> {
    let path = app_support_dir()?.join("History.json");
    if !path.is_file() {
        bail!("legacy history not found: {}", path.display());
    }
    let entries: Vec<LegacyHistoryEntry> = serde_json::from_slice(&std::fs::read(&path)?)
        .with_context(|| format!("parse {}", path.display()))?;
    let mut imported = 0;
    let mut missing_audio = 0;
    for entry in entries {
        let Some(audio_path) = entry.audio_path else {
            missing_audio += 1;
            continue;
        };
        if !audio_path.is_file() {
            missing_audio += 1;
            continue;
        }
        let stem = audio_path
            .file_stem()
            .and_then(|value| value.to_str())
            .context("legacy WAV has no UTF-8 filename")?;
        let id = format!("legacy_{stem}");
        let (_, created) = import_recording(
            &audio_path,
            &id,
            "legacy-macos-app",
            &entry.transcript,
            entry.timestamp,
        )?;
        imported += usize::from(created);
    }
    let recordings_dir = app_support_dir()?.join("Recordings");
    let mut orphan_wavs = 0;
    if recordings_dir.is_dir() {
        for directory_entry in std::fs::read_dir(&recordings_dir)? {
            let audio_path = directory_entry?.path();
            if audio_path.extension().and_then(|value| value.to_str()) != Some("wav") {
                continue;
            }
            let stem = audio_path
                .file_stem()
                .and_then(|value| value.to_str())
                .context("legacy WAV has no UTF-8 filename")?;
            let id = format!("legacy_{stem}");
            let (_, created) = import_recording(&audio_path, &id, "legacy-macos-app", "", None)?;
            if created {
                imported += 1;
                orphan_wavs += 1;
            }
        }
    }
    println!(
        "migrated {imported} WAV-backed entries ({orphan_wavs} without history text); {missing_audio} history rows had no available WAV"
    );
    Ok(())
}

pub fn set_intended(id: &str, text: &str) -> Result<()> {
    let recording = set_intended_transcript(id, text)?;
    println!(
        "{} intended transcript saved to {}",
        recording.id,
        corpus_dir()?
            .join(&recording.id)
            .join("metadata.json")
            .display()
    );
    Ok(())
}

pub fn delete_corpus_recording(id: &str) -> Result<()> {
    delete_recording(id)?;
    println!("deleted recording {id}");
    Ok(())
}

pub fn print_stats(json: bool) -> Result<()> {
    let stats = usage_stats()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        println!("recordings:       {}", stats.recordings);
        println!("words:            {}", stats.words);
        println!(
            "speaking seconds: {:.1}",
            stats.speaking_ms as f64 / 1_000.0
        );
        println!("dictionary fixes: {}", stats.dictionary_fixes);
    }
    Ok(())
}

fn comparable(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_alphanumeric() || character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::comparable;

    #[test]
    fn comparison_ignores_punctuation_but_not_words() {
        assert_eq!(comparable("Use CUDA, now."), "use cuda now");
        assert_ne!(comparable("Use CUDA"), comparable("Use CUDA now"));
    }
}
