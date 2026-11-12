//! One dictation pass: wave file in, finished text out.
//!
//! The recogniser runs per pass. The correction model stays resident, because
//! loading three gigabytes of weights for every utterance would dominate the
//! latency a user feels.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::asr::Recognizer;
use crate::polish::{Corrector, Device};

/// What one pass produced, and what each stage cost.
#[derive(Debug, Clone)]
pub struct PassResult {
    pub raw: String,
    pub polished: String,
    pub asr_ms: f64,
    pub polish_ms: f64,
}

impl PassResult {
    /// One line for the log: timings and lengths, never the text. On Linux the
    /// autostarted app's stderr lands in the systemd user journal, so a summary
    /// that quoted the dictation wrote every one of them to disk (Ubuntu QA VM,
    /// 2026-11-12: 391 dictations in the persistent journal).
    pub fn summary(&self) -> String {
        format!(
            "asr {:.0} ms, {} chars; correction {:.0} ms, {} chars",
            self.asr_ms,
            self.raw.chars().count(),
            self.polish_ms,
            self.polished.chars().count()
        )
    }
}

/// The warm engine: a located recogniser and a running correction server.
pub struct Engine {
    recognizer: Recognizer,
    corrector: Corrector,
    /// The personal dictionary dictations use (not the startup check).
    dictionary: std::sync::Mutex<crate::dictionary::Live>,
}

impl Engine {
    /// Locate the recogniser and start the correction server. Both must succeed;
    /// recognition without correction is not a supported mode. `backends` is the
    /// tuned-runtime manifest URL (see `backends::DEFAULT_MANIFEST`).
    pub fn start(backends: Option<&str>) -> Result<Self> {
        let mut recognizer =
            Recognizer::installed().context("speech recognition is not installed")?;
        recognizer.load_resident();
        let corrector = Corrector::start(backends).context("correction model did not start")?;
        crate::dictionary::warm();
        Ok(Self {
            recognizer,
            corrector,
            dictionary: std::sync::Mutex::new(crate::dictionary::Live::new(
                crate::dictionary::path(),
            )),
        })
    }

    /// Where the correction model runs.
    pub fn device(&self) -> Device {
        self.corrector.device()
    }

    /// Whether the next correction goes to the fallback server (saturated CPU).
    pub fn lean(&self) -> bool {
        self.corrector.lean()
    }

    /// Start the saturated-CPU fallback corrector now (see `Corrector::start_lean`).
    #[cfg_attr(not(windows), allow(dead_code))]
    pub fn start_lean(&self) {
        self.corrector.start_lean();
    }

    /// Call while idle with the machine's busy fraction (see `Corrector::warm`).
    pub fn warm(&self, busy: Option<f64>) {
        self.corrector.warm(busy);
    }

    /// Call at key-down with the machine's busy fraction (see `Corrector::prepare`).
    pub fn prepare(&self, busy: Option<f64>) {
        self.corrector.prepare(busy);
    }

    /// The startup check passed on this engine: keep only the tuned build in use.
    pub fn confirm_backend(&mut self) {
        self.corrector.confirm();
    }

    /// Move the correction model from the GPU to the CPU.
    pub fn fall_back_to_cpu(&mut self) -> Result<()> {
        self.corrector.fall_back_to_cpu()
    }

    /// Whether `run_samples` can skip the file (the speech model is resident).
    pub fn resident(&self) -> bool {
        self.recognizer.is_resident()
    }

    /// Run a dictation's captured audio (mono, -1..1) through both stages without
    /// writing a file. Requires a resident speech model (`resident`). A correction
    /// that fails leaves the raw transcript (see `correct_or_raw`).
    pub fn run_samples(&self, sample_rate: u32, samples: &[f32]) -> Result<PassResult> {
        let started = Instant::now();
        let transcript = self
            .recognizer
            .transcribe_samples(sample_rate, samples)
            .ok_or_else(|| anyhow::anyhow!("the speech model is not resident"))?
            .context("transcribe the recording")?;
        let asr_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(self.correct_or_raw(transcript.text, asr_ms, crate::mode::current()))
    }

    /// Run a dictation's wave file through both stages in the writing mode picked
    /// from the tray; like `run`, except that a correction that fails leaves the raw
    /// transcript.
    pub fn dictate(&self, wav: &Path) -> Result<PassResult> {
        let started = Instant::now();
        let transcript = self
            .recognizer
            .transcribe(wav)
            .with_context(|| format!("transcribe {}", wav.display()))?;
        let asr_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(self.correct_or_raw(transcript.text, asr_ms, crate::mode::current()))
    }

    /// The corrected text in `mode`, or the raw transcript when correction fails: a
    /// dictation the user spoke is inserted either way, never dropped (the correction
    /// server rejected a 900 s transcript with HTTP 400 before sentence groups,
    /// 2026-11-08).
    fn correct_or_raw(&self, raw: String, asr_ms: f64, mode: &str) -> PassResult {
        // the mode's name only, never the text
        eprintln!("phonon: correcting in writing mode {mode}");
        let started = Instant::now();
        // the personal dictionary as saved now (re-read when the file changed)
        let mut dictionary = self
            .dictionary
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let polished = match self
            .corrector
            .correct_with(&raw, mode, dictionary.current())
        {
            Ok(polished) => polished,
            Err(error) => {
                eprintln!("phonon: correction failed, inserting the raw transcript: {error:#}");
                raw.clone()
            }
        };
        PassResult {
            raw,
            polished,
            asr_ms,
            polish_ms: started.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Run one wave file through both stages in the default mode (minimal), whatever
    /// the tray says: the startup check compares against a fixed answer.
    pub fn run(&self, wav: &Path) -> Result<PassResult> {
        let started = Instant::now();
        let transcript = self
            .recognizer
            .transcribe(wav)
            .with_context(|| format!("transcribe {}", wav.display()))?;
        let asr_ms = started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        let polished = self
            .corrector
            .correct(&transcript.text)
            .context("correct the transcript")?;
        let polish_ms = started.elapsed().as_secs_f64() * 1000.0;

        Ok(PassResult {
            raw: transcript.text,
            polished,
            asr_ms,
            polish_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_summary_names_both_stages() {
        let result = PassResult {
            raw: "hello fluid voice".into(),
            polished: "Hello, Fluid Voice.".into(),
            asr_ms: 412.4,
            polish_ms: 980.6,
        };
        let line = result.summary();
        assert_eq!(line, "asr 412 ms, 17 chars; correction 981 ms, 19 chars");
    }

    #[test]
    fn the_summary_carries_no_dictated_text() {
        let result = PassResult {
            raw: "zebra quartz dictation".into(),
            polished: "Zebra quartz dictation, polished.".into(),
            asr_ms: 1.0,
            polish_ms: 2.0,
        };
        let line = result.summary().to_lowercase();
        for word in ["zebra", "quartz", "dictation", "polished"] {
            assert!(!line.contains(word), "{word} reached the log line: {line}");
        }
    }
}
