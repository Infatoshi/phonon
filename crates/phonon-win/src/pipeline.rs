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
    /// One line for the log.
    pub fn summary(&self) -> String {
        format!(
            "asr {:.0} ms -> {:?}; correction {:.0} ms -> {:?}",
            self.asr_ms, self.raw, self.polish_ms, self.polished
        )
    }
}

/// The warm engine: a located recogniser and a running correction server.
pub struct Engine {
    recognizer: Recognizer,
    corrector: Corrector,
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
        Ok(Self {
            recognizer,
            corrector,
        })
    }

    /// Where the correction model runs.
    pub fn device(&self) -> Device {
        self.corrector.device()
    }

    /// Whether the next correction goes to the one-thread server (saturated CPU).
    pub fn lean(&self) -> bool {
        self.corrector.lean()
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
        Ok(self.correct_or_raw(transcript.text, asr_ms))
    }

    /// Run a dictation's wave file through both stages; like `run`, except that a
    /// correction that fails leaves the raw transcript.
    pub fn dictate(&self, wav: &Path) -> Result<PassResult> {
        let started = Instant::now();
        let transcript = self
            .recognizer
            .transcribe(wav)
            .with_context(|| format!("transcribe {}", wav.display()))?;
        let asr_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(self.correct_or_raw(transcript.text, asr_ms))
    }

    /// The corrected text, or the raw transcript when correction fails: a dictation
    /// the user spoke is inserted either way, never dropped (the correction server
    /// rejected a 900 s transcript with HTTP 400 before sentence groups, 2026-11-08).
    fn correct_or_raw(&self, raw: String, asr_ms: f64) -> PassResult {
        let started = Instant::now();
        let polished = match self.corrector.correct(&raw) {
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

    /// Run one wave file through both stages.
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
        assert!(line.contains("asr 412 ms"));
        assert!(line.contains("correction 981 ms"));
        assert!(line.contains("Hello, Fluid Voice."));
    }
}
