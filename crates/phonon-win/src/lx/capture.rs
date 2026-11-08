//! Microphone capture on Linux through the sound server's own recorder.
//!
//! `pw-record` (PipeWire, default on current desktops), then `parecord`
//! (PulseAudio), then `arecord` (bare ALSA). Each records the default input as
//! 16 kHz mono 16-bit WAV and finalizes the header when interrupted, so Phonon
//! links no audio library and follows whatever device the desktop has selected.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

pub struct Recording {
    child: Child,
    tool: &'static str,
    path: PathBuf,
    started: Instant,
}

fn command(tool: &str, path: &Path) -> Command {
    let mut c = Command::new(tool);
    match tool {
        "pw-record" => c.args(["--rate", "16000", "--channels", "1", "--format", "s16"]),
        "parecord" => c.args([
            "--rate=16000",
            "--channels=1",
            "--format=s16le",
            "--file-format=wav",
        ]),
        _ => c.args(["-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav"]),
    };
    c.arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    c
}

/// Start recording the default input into `path`.
pub fn start(path: &Path) -> Result<Recording> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    for tool in ["pw-record", "parecord", "arecord"] {
        let Ok(mut child) = command(tool, path).spawn() else {
            continue;
        };
        // a recorder that cannot open the device exits at once
        std::thread::sleep(Duration::from_millis(30));
        if let Ok(Some(status)) = child.try_wait() {
            eprintln!("phonon: {tool} exited at start ({status})");
            continue;
        }
        return Ok(Recording {
            child,
            tool,
            path: path.to_path_buf(),
            started: Instant::now(),
        });
    }
    bail!("no recorder could open the microphone (tried pw-record, parecord, arecord)")
}

impl Recording {
    pub fn device(&self) -> &str {
        self.tool
    }

    pub fn seconds(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// Stop with SIGINT, so the recorder writes the final WAV header.
    pub fn finish(mut self) -> Result<PathBuf> {
        unsafe { libc::kill(self.child.id() as libc::pid_t, libc::SIGINT) };
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if self.child.try_wait()?.is_some() {
                break;
            }
            if Instant::now() > deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let size = std::fs::metadata(&self.path)
            .with_context(|| format!("{} wrote nothing", self.tool))?
            .len();
        if size <= 44 {
            bail!("{} recorded no audio", self.tool);
        }
        Ok(self.path)
    }
}
