//! One dictation pass: wave file in, finished text out.
//!
//! The recogniser runs per pass. The correction model stays resident, because
//! loading three gigabytes of weights for every utterance would dominate the
//! latency a user feels.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::asr::Recognizer;
use crate::polish::Corrector;

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
    /// recognition without correction is not a supported mode.
    pub fn start() -> Result<Self> {
        let recognizer = Recognizer::installed().context("speech recognition is not installed")?;
        let corrector = Corrector::start().context("correction model did not start")?;
        Ok(Self {
            recognizer,
            corrector,
        })
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
