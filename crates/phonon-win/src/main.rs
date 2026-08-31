//! Phonon for Windows.
//!
//! One executable. It carries the correction prompt and the startup fixture and
//! nothing else; the speech runtime, the correction runtime, and both models are
//! downloaded on first run and checked against pinned SHA-256 hashes.
//!
//! The crate builds on every platform so the portable half, the manifest, the
//! download and verify path, the output parsers, and the readiness rules, is
//! covered by the macOS test suite as well. Only `run` needs Windows.

// Half of this crate exists for the Windows application, which is not built on
// other platforms. Off Windows that half is unreachable, and saying so here is
// clearer than an attribute on every item.
#![cfg_attr(not(windows), allow(dead_code))]

mod asr;
mod fetch;
mod http;
mod manifest;
mod paths;
mod pipeline;
mod polish;
mod smoke;

#[cfg(windows)]
mod win;

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use pipeline::PassResult;

/// The startup fixture, shared with the macOS build. It is small enough to ship
/// inside the executable, and first run has to prove the audio path before
/// Phonon claims to be ready.
pub const STARTUP_WAV: &[u8] = include_bytes!("../../../assets/startup.wav");

#[derive(Parser)]
#[command(name = "phonon-win", about = "Phonon dictation for Windows", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the tray application. This is the default.
    Run,
    /// Download and verify the runtimes and models, then stop.
    Fetch,
    /// Print what first run downloads and where it goes.
    Info,
    /// Run one wave file through the real recogniser and the real correction
    /// model, then check the result. This is the release gate.
    Selftest {
        /// The wave file. Defaults to the fixture inside the executable.
        #[arg(long)]
        wav: Option<PathBuf>,
        /// Skip the fixture word check. Use for a file that is not the fixture.
        #[arg(long)]
        any_audio: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Run) {
        Command::Run => run(),
        Command::Fetch => fetch_all(),
        Command::Info => {
            info();
            Ok(())
        }
        Command::Selftest { wav, any_audio } => selftest(wav, any_audio),
    }
}

#[cfg(windows)]
fn run() -> Result<()> {
    win::run()
}

#[cfg(not(windows))]
fn run() -> Result<()> {
    bail!(
        "phonon-win runs on Windows. On macOS use `phonon`. \
         `phonon-win info`, `fetch`, and `selftest` work anywhere."
    )
}

fn info() {
    println!("Phonon for Windows");
    println!("  data root:        {}", paths::data_root().display());
    println!(
        "  first run downloads {}",
        manifest::human_bytes(manifest::total_bytes())
    );
    for component in manifest::ALL {
        println!(
            "  {:<28} {:>9}  {}  {}",
            component.label,
            manifest::human_bytes(component.download_bytes()),
            if fetch::installed(component) {
                "installed"
            } else {
                "missing  "
            },
            component.dir
        );
    }
    println!(
        "  speech runtime:   sherpa-onnx {}",
        manifest::SHERPA_VERSION
    );
    println!("  correction runtime: llama.cpp {}", manifest::LLAMA_BUILD);
    println!(
        "  speech model:     {}@{}",
        manifest::ASR_MODEL_ID,
        &manifest::ASR_MODEL_REVISION[..7]
    );
    println!(
        "  correction model: {}@{}",
        manifest::POLISH_MODEL_ID,
        &manifest::POLISH_MODEL_REVISION[..7]
    );
}

fn fetch_all() -> Result<()> {
    let mut last = String::new();
    fetch::ensure_all(&mut |progress| {
        let line = progress.line();
        if line != last {
            println!("{line}");
            last = line;
        }
    })?;
    println!("every runtime and model is installed and verified");
    Ok(())
}

/// The release gate. It downloads what is missing, starts the real correction
/// server, transcribes real audio, corrects it, and applies the same two rules
/// the macOS readiness contract applies.
fn selftest(wav: Option<PathBuf>, any_audio: bool) -> Result<()> {
    fetch_all()?;
    let wav = match wav {
        Some(path) => path,
        None => {
            let path = paths::data_root().join("startup.wav");
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, STARTUP_WAV).context("write the startup fixture")?;
            path
        }
    };
    println!("selftest: {}", wav.display());

    let engine = pipeline::Engine::start()?;
    let result = engine.run(&wav)?;
    println!("  raw:      {:?}", result.raw);
    println!("  polished: {:?}", result.polished);
    println!(
        "  asr {:.0} ms, correction {:.0} ms",
        result.asr_ms, result.polish_ms
    );

    if any_audio {
        assert_usable(&result)?;
    } else {
        assert_startup(&result)?;
    }
    println!("selftest passed");
    Ok(())
}

/// Both readiness rules, for the known fixture.
pub fn assert_startup(result: &PassResult) -> Result<()> {
    if !smoke::asr_smoke_passed(&result.raw) {
        bail!(
            "the recogniser did not produce the fixture sentence; got {:?}",
            result.raw
        );
    }
    assert_usable(result)
}

/// The correction rule alone, for audio whose words are not known ahead of time.
pub fn assert_usable(result: &PassResult) -> Result<()> {
    if result.raw.trim().is_empty() {
        bail!("the recogniser returned nothing");
    }
    if !smoke::llm_audio_smoke_passed(&result.raw, &result.polished) {
        bail!(
            "correction did not return a usable sentence: {:?} became {:?}",
            result.raw,
            result.polished
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixture_ships_inside_the_executable() {
        // A RIFF header and about one second of 16 kHz mono audio.
        assert_eq!(&STARTUP_WAV[..4], b"RIFF");
        assert_eq!(&STARTUP_WAV[8..12], b"WAVE");
        assert!(STARTUP_WAV.len() > 20_000, "the fixture looks truncated");
    }

    fn result(raw: &str, polished: &str) -> PassResult {
        PassResult {
            raw: raw.into(),
            polished: polished.into(),
            asr_ms: 1.0,
            polish_ms: 1.0,
        }
    }

    #[test]
    fn the_gate_accepts_a_good_pass() {
        assert!(assert_startup(&result("hello fluid voice", "Hello, Fluid Voice.")).is_ok());
    }

    #[test]
    fn the_gate_rejects_a_wrong_transcript() {
        let error = assert_startup(&result("goodbye", "Goodbye."))
            .unwrap_err()
            .to_string();
        assert!(error.contains("fixture sentence"));
    }

    #[test]
    fn the_gate_rejects_a_collapsed_correction() {
        let error = assert_startup(&result("hello fluid voice", "Hello."))
            .unwrap_err()
            .to_string();
        assert!(error.contains("usable sentence"));
    }

    /// Recognition without correction is not a supported mode, so an empty
    /// correction must fail even when the transcript was right.
    #[test]
    fn the_gate_rejects_recognition_alone() {
        assert!(assert_startup(&result("hello fluid voice", "")).is_err());
        assert!(assert_usable(&result("", "anything")).is_err());
    }
}
