//! Microphone capture on Linux through the sound server's own recorder.
//!
//! `pw-record` (PipeWire, default on current desktops), then `parecord`
//! (PulseAudio), then `arecord` (bare ALSA). Each records the default input as
//! 16 kHz mono 16-bit WAV and finalizes the header when interrupted, so Phonon
//! links no audio library and follows whatever device the desktop has selected.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use phonon_hotkey::autostop::{AutoStop, Decision};

/// Every recorder is asked for this rate, mono, 16-bit.
const RATE: u32 = 16_000;

pub struct Recording {
    child: Child,
    tool: &'static str,
    path: PathBuf,
    started: Instant,
    auto: AutoStop,
    /// Where the samples start in the file, once its header has been read.
    data_offset: Option<u64>,
    wave: [f32; super::indicator::POINTS],
    last_wave: Instant,
}

/// Where the sample data begins in a WAV file's first bytes: after the `data` chunk's
/// header. The recorders write the header before any audio and leave the offset alone.
fn wav_data_offset(head: &[u8]) -> Option<u64> {
    if head.len() < 12 || &head[..4] != b"RIFF" || &head[8..12] != b"WAVE" {
        return None;
    }
    let mut at = 12usize;
    while at + 8 <= head.len() {
        let size = u32::from_le_bytes(head[at + 4..at + 8].try_into().ok()?) as usize;
        if &head[at..at + 4] == b"data" {
            return Some((at + 8) as u64);
        }
        // chunks are padded to an even length
        at = at.checked_add(8 + size + size % 2)?;
    }
    None
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
            auto: AutoStop::new(RATE),
            data_offset: None,
            wave: [0.0; super::indicator::POINTS],
            last_wave: Instant::now(),
        });
    }
    bail!("no recorder could open the microphone (tried pw-record, parecord, arecord)")
}

impl Recording {
    pub fn device(&self) -> &str {
        self.tool
    }

    /// The file the recorder is writing.
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn seconds(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// Whether a latched recording should stop by itself now (`phonon_hotkey::autostop`).
    /// The recorder writes to a file, so the meter reads what it appended since the last
    /// call. The clock is wall time since the start, so a recorder that writes nothing is
    /// still discarded.
    pub fn auto_stop(&mut self, latched: bool) -> Decision {
        let fresh = self.appended().unwrap_or_default();
        if !fresh.is_empty() {
            self.wave = super::indicator::waveform(&fresh);
            self.last_wave = Instant::now();
        } else if self.last_wave.elapsed() >= Duration::from_millis(200) {
            self.wave.fill(0.0);
        }
        self.auto.feed(&fresh);
        self.auto.decision(latched, self.seconds())
    }

    /// Visualization of the same samples already fed to auto-stop. No second capture.
    pub fn waveform(&self) -> [f32; super::indicator::POINTS] {
        self.wave
    }

    /// The samples the recorder wrote after the ones already fed to the meter.
    fn appended(&mut self) -> Option<Vec<i16>> {
        let mut file = File::open(&self.path).ok()?;
        let offset = match self.data_offset {
            Some(offset) => offset,
            None => {
                let mut head = Vec::with_capacity(4096);
                (&mut file).take(4096).read_to_end(&mut head).ok()?;
                let offset = wav_data_offset(&head)?;
                self.data_offset = Some(offset);
                offset
            }
        };
        file.seek(SeekFrom::Start(offset + 2 * self.auto.fed() as u64))
            .ok()?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).ok()?;
        // a sample cut in half by the writer is read whole next time
        Some(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| i16::from_le_bytes(*pair))
                .collect(),
        )
    }

    /// Stop the recorder and delete its file: no transcription, no paste, nothing kept.
    pub fn discard(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.path);
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

#[cfg(test)]
mod tests {
    use super::wav_data_offset;

    fn header(extra: &[u8]) -> Vec<u8> {
        let mut h = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&[0; 16]);
        h.extend_from_slice(extra);
        h.extend_from_slice(b"data\xff\xff\xff\xff");
        h
    }

    #[test]
    fn the_plain_44_byte_header() {
        assert_eq!(wav_data_offset(&header(&[])), Some(44));
    }

    /// libsndfile (pw-record, parecord) can put another chunk before the data; odd sizes pad.
    #[test]
    fn chunks_before_the_data_are_skipped() {
        let mut list = b"LIST".to_vec();
        list.extend_from_slice(&3u32.to_le_bytes());
        list.extend_from_slice(&[1, 2, 3, 0]);
        assert_eq!(wav_data_offset(&header(&list)), Some(44 + 12));
    }

    #[test]
    fn no_header_yet_is_no_offset() {
        assert_eq!(wav_data_offset(b""), None);
        assert_eq!(wav_data_offset(b"RIFF\0\0\0\0WAVEfmt "), None);
        assert_eq!(wav_data_offset(b"not a wave file at all"), None);
    }

    #[test]
    fn the_wave_and_auto_stop_share_each_appended_sample_once() {
        use super::*;
        use std::io::Write;

        let path =
            std::env::temp_dir().join(format!("phonon-wave-capture-{}.wav", std::process::id()));
        let pcm: Vec<i16> = (0..480)
            .map(|i| if i % 2 == 0 { 4096 } else { -4096 })
            .collect();
        let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut file = File::create(&path).unwrap();
        file.write_all(&header(&[])).unwrap();
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
        let mut recording = Recording {
            child: Command::new("true").spawn().unwrap(),
            tool: "fixture",
            path: path.clone(),
            started: Instant::now(),
            auto: AutoStop::new(RATE),
            data_offset: None,
            wave: [0.0; super::super::indicator::POINTS],
            last_wave: Instant::now(),
        };
        assert_eq!(recording.auto_stop(false), Decision::KeepRecording);
        assert_eq!(recording.auto.fed(), pcm.len());
        assert_eq!(
            recording.waveform(),
            super::super::indicator::waveform(&pcm)
        );
        let wave = recording.waveform();
        recording.auto_stop(false);
        assert_eq!(recording.auto.fed(), pcm.len());
        assert_eq!(recording.waveform(), wave);
        recording.last_wave = Instant::now() - Duration::from_millis(250);
        recording.auto_stop(false);
        assert_eq!(recording.waveform(), [0.0; super::super::indicator::POINTS]);
        assert_eq!(recording.auto.fed(), pcm.len());
        file.write_all(&vec![0; bytes.len()]).unwrap();
        file.flush().unwrap();
        recording.auto_stop(false);
        assert_eq!(recording.auto.fed(), 2 * pcm.len());
        assert_eq!(recording.waveform(), [0.0; super::super::indicator::POINTS]);
        recording.discard();
    }
}
