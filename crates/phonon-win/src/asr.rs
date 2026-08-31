//! Speech recognition through sherpa-onnx.
//!
//! `sherpa-onnx-offline.exe` writes one JSON object per wave file to standard
//! output and everything else to standard error, so the parser only has to read
//! standard output. The model is the ONNX int8 export of Parakeet TDT 0.6b v2,
//! the same acoustic model the macOS build runs on MLX.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};

use crate::fetch;
use crate::manifest;

/// A ready recogniser: the tool and the weights it needs.
#[derive(Debug, Clone)]
pub struct Recognizer {
    tool: PathBuf,
    model_dir: PathBuf,
    threads: usize,
}

/// One transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub text: String,
}

/// Pull the transcript out of one `sherpa-onnx-offline` standard-output block.
///
/// Kept separate from the process call so the parser is provable without Windows.
pub fn parse_output(stdout: &str) -> Result<Transcript> {
    for line in stdout.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_str(line).with_context(|| format!("parse sherpa result {line}"))?;
        let text = value
            .get("text")
            .and_then(|text| text.as_str())
            .ok_or_else(|| anyhow!("sherpa result has no text field: {line}"))?;
        return Ok(Transcript {
            text: text.trim().to_string(),
        });
    }
    bail!("sherpa-onnx-offline printed no result object")
}

/// The path, made absolute against the current directory. Nothing is resolved:
/// a symbolic link stays a symbolic link, and no extended-length prefix is added.
pub fn absolute(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(std::env::current_dir()
        .context("read the current directory")?
        .join(path))
}

/// How many decoding threads to use. sherpa-onnx gains little past four, and
/// leaving cores free keeps the machine responsive while the user dictates.
pub fn thread_count(logical_cpus: usize) -> usize {
    logical_cpus.saturating_sub(1).clamp(1, 4)
}

impl Recognizer {
    /// Locate an installed recogniser. Call after `fetch::ensure_all`.
    pub fn installed() -> Result<Self> {
        let tool = fetch::sentinel_path(&manifest::SHERPA_RUNTIME);
        if !tool.is_file() {
            bail!("{} is missing; run the first-run download", tool.display());
        }
        let model_dir = fetch::component_dir(&manifest::ASR_MODEL);
        if !model_dir.join("encoder.int8.onnx").is_file() {
            bail!("{} is missing the encoder", model_dir.display());
        }
        let threads = thread_count(
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(2),
        );
        Ok(Self {
            tool,
            model_dir,
            threads,
        })
    }

    /// The command line, so a failure can be reported exactly as it was run.
    pub fn args(&self, wav: &Path) -> Vec<String> {
        vec![
            format!(
                "--encoder={}",
                self.model_dir.join("encoder.int8.onnx").display()
            ),
            format!(
                "--decoder={}",
                self.model_dir.join("decoder.int8.onnx").display()
            ),
            format!(
                "--joiner={}",
                self.model_dir.join("joiner.int8.onnx").display()
            ),
            format!("--tokens={}", self.model_dir.join("tokens.txt").display()),
            "--model-type=nemo_transducer".into(),
            format!("--num-threads={}", self.threads),
            wav.display().to_string(),
        ]
    }

    /// Transcribe one 16-bit single-channel wave file.
    pub fn transcribe(&self, wav: &Path) -> Result<Transcript> {
        if !wav.is_file() {
            bail!("{} is not a file", wav.display());
        }
        // The tool runs from its own directory so it finds onnxruntime.dll, which
        // means a relative wave path would resolve against that directory
        // instead of the caller's. Make it absolute here. `canonicalize` would
        // do it too, but on Windows it returns an extended-length `\\?\` path
        // that not every C++ file reader accepts.
        let wav = absolute(wav)?;
        let mut command = Command::new(&self.tool);
        command.args(self.args(&wav));
        if let Some(bin) = self.tool.parent() {
            command.current_dir(bin);
        }
        let output = command
            .output()
            .with_context(|| format!("run {}", self.tool.display()))?;
        if !output.status.success() {
            bail!(
                "sherpa-onnx-offline exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        parse_output(&String::from_utf8_lossy(&output.stdout))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One real standard-output block, copied from the sherpa-onnx manual.
    const SAMPLE: &str = r#"{"lang": "", "emotion": "", "event": "", "text": " Well, I don't wish to see it any more, observed Phebe.", "timestamps": [0.32, 0.64], "tokens":[" Well", ","], "words": []}"#;

    #[test]
    fn reads_the_text_field() {
        let parsed = parse_output(SAMPLE).unwrap();
        assert_eq!(
            parsed.text,
            "Well, I don't wish to see it any more, observed Phebe."
        );
    }

    /// Standard error is redirected into the same stream by some shells. The
    /// parser must skip the noise and find the object.
    #[test]
    fn skips_lines_that_are_not_the_result() {
        let mixed =
            format!("Creating recognizer ...\nStarted\nDone!\n\n./foo.wav\n{SAMPLE}\n----\n");
        assert!(parse_output(&mixed).unwrap().text.starts_with("Well"));
    }

    #[test]
    fn an_empty_transcript_is_still_a_result() {
        let empty = r#"{"text": "", "tokens":[]}"#;
        assert_eq!(parse_output(empty).unwrap().text, "");
    }

    #[test]
    fn no_result_object_is_an_error() {
        assert!(parse_output("Creating recognizer ...\nStarted\n").is_err());
    }

    #[test]
    fn a_relative_path_becomes_absolute() {
        let made = absolute(Path::new("assets/startup.wav")).unwrap();
        assert!(made.is_absolute());
        assert!(made.ends_with("assets/startup.wav"));
        // An absolute path is handed back untouched, prefix and all.
        let already = std::env::temp_dir().join("x.wav");
        assert_eq!(absolute(&already).unwrap(), already);
    }

    #[test]
    fn thread_count_leaves_a_core_free_and_stops_at_four() {
        assert_eq!(thread_count(1), 1);
        assert_eq!(thread_count(2), 1);
        assert_eq!(thread_count(4), 3);
        assert_eq!(thread_count(8), 4);
        assert_eq!(thread_count(32), 4);
    }
}
