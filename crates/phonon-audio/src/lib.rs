use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

static RECORDING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicrophoneStatus {
    pub priority: usize,
    pub label: String,
    pub device_name: Option<String>,
    pub selected: bool,
}

impl MicrophoneStatus {
    pub fn available(&self) -> bool {
        self.device_name.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct MicrophoneSelection {
    pub devices: Vec<MicrophoneStatus>,
    pub selected_device: Option<String>,
}

const DEFAULT_MICROPHONE_PRIORITY: &[&str] = &[];

#[derive(Debug, Deserialize)]
struct AudioSettings {
    #[serde(default)]
    microphone_priority: Vec<String>,
}

pub fn resolve_microphone_priority() -> Result<MicrophoneSelection> {
    let output = Command::new("system_profiler")
        .args(["SPAudioDataType", "-json"])
        .output()
        .context("enumerate CoreAudio devices")?;
    if !output.status.success() {
        bail!(
            "system_profiler audio enumeration failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let value: Value = serde_json::from_slice(&output.stdout).context("parse CoreAudio devices")?;
    let priority = configured_microphone_priority();
    Ok(resolve_from_names_with_priority(
        &input_device_names(&value),
        &priority,
    ))
}

fn input_device_names(value: &Value) -> Vec<String> {
    let mut names = Vec::new();
    collect_input_device_names(value, &mut names);
    names
}

fn collect_input_device_names(value: &Value, names: &mut Vec<String>) {
    match value {
        Value::Array(values) => {
            for value in values {
                collect_input_device_names(value, names);
            }
        }
        Value::Object(object) => {
            if object.contains_key("coreaudio_device_input") {
                if let Some(name) = object.get("_name").and_then(Value::as_str) {
                    names.push(name.to_string());
                }
            }
            for value in object.values() {
                collect_input_device_names(value, names);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
fn resolve_from_names(names: &[String]) -> MicrophoneSelection {
    resolve_from_names_with_priority(
        names,
        &DEFAULT_MICROPHONE_PRIORITY
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>(),
    )
}

fn resolve_from_names_with_priority(names: &[String], priority: &[String]) -> MicrophoneSelection {
    let mut selected_device = None;
    let mut devices = Vec::new();
    for (index, label) in priority.iter().enumerate() {
        let pattern = label.to_lowercase();
        let device_name = names
            .iter()
            .find(|name| name.to_lowercase().contains(&pattern))
            .cloned();
        let selected = selected_device.is_none() && device_name.is_some();
        if selected {
            selected_device.clone_from(&device_name);
        }
        devices.push(MicrophoneStatus {
            priority: index + 1,
            label: label.clone(),
            device_name,
            selected,
        });
    }
    MicrophoneSelection {
        devices,
        selected_device,
    }
}

fn configured_microphone_priority() -> Vec<String> {
    let Some(home) = std::env::var_os("HOME") else {
        return DEFAULT_MICROPHONE_PRIORITY
            .iter()
            .map(|value| (*value).to_string())
            .collect();
    };
    let path = PathBuf::from(home).join("Library/Application Support/Phonon/settings.json");
    let configured = std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<AudioSettings>(&bytes).ok())
        .map(|settings| settings.microphone_priority)
        .unwrap_or_default();
    if configured.is_empty() {
        DEFAULT_MICROPHONE_PRIORITY
            .iter()
            .map(|value| (*value).to_string())
            .collect()
    } else {
        configured
    }
}

pub fn unique_wav_path() -> PathBuf {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let sequence = RECORDING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = home
        .join("Library")
        .join("Application Support")
        .join("Phonon")
        .join("Corpus")
        .join(format!("terminal_{ms}_{}_{sequence}", std::process::id()));
    dir.join("audio.wav")
}

pub fn start_recording(wav: &Path, device_name: Option<&str>) -> Result<Child> {
    if let Some(parent) = wav.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create recording directory {}", parent.display()))?;
    }
    let sox = which::which("sox").context("install SoX: brew install sox")?;
    let mut command = Command::new(sox);
    command.arg("-q");
    if let Some(device_name) = device_name {
        command.args(["-t", "coreaudio"]).arg(device_name);
    } else {
        command.arg("-d");
    }
    command
        .args([
            "-r",
            "16000",
            "-c",
            "1",
            "-b",
            "16",
            "-e",
            "signed-integer",
            wav.to_str().unwrap_or_default(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("start recording (is mic free?)")
}

pub fn stop_recording(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

pub fn wav_has_speech(path: &Path) -> Result<bool> {
    let bytes = std::fs::read(path).with_context(|| format!("read WAV {}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("unsupported WAV header: {}", path.display());
    }
    let mut cursor = 12;
    let mut format = None;
    let mut data = None;
    while cursor + 8 <= bytes.len() {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_len = u32::from_le_bytes([
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]) as usize;
        let start = cursor + 8;
        let end = start.saturating_add(chunk_len).min(bytes.len());
        if chunk_id == b"fmt " && end >= start + 16 {
            format = Some((
                u16::from_le_bytes([bytes[start], bytes[start + 1]]),
                u16::from_le_bytes([bytes[start + 2], bytes[start + 3]]),
                u32::from_le_bytes([
                    bytes[start + 4],
                    bytes[start + 5],
                    bytes[start + 6],
                    bytes[start + 7],
                ]),
                u16::from_le_bytes([bytes[start + 14], bytes[start + 15]]),
            ));
        } else if chunk_id == b"data" {
            data = Some(&bytes[start..end]);
        }
        cursor = start + chunk_len + (chunk_len % 2);
    }
    let Some((audio_format, channels, sample_rate, bits_per_sample)) = format else {
        bail!("WAV has no format chunk: {}", path.display());
    };
    if audio_format != 1 || channels != 1 || bits_per_sample != 16 {
        bail!("speech gate requires mono PCM16 WAV: {}", path.display());
    }
    let data = data.with_context(|| format!("WAV has no data chunk: {}", path.display()))?;
    let samples = data
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    Ok(samples_have_speech(&samples, sample_rate))
}

fn samples_have_speech(samples: &[i16], sample_rate: u32) -> bool {
    const FRAME_RATE_HZ: usize = 50;
    const MIN_CONSECUTIVE_VOICED_FRAMES: usize = 4;
    const NOISE_MULTIPLIER: f64 = 3.0;
    const ABSOLUTE_RMS_FLOOR: f64 = 180.0;
    const MAX_VOICED_ZERO_CROSSING_RATE: f64 = 0.20;

    let frame_samples = (sample_rate as usize / FRAME_RATE_HZ).max(1);
    if samples.len() < frame_samples * MIN_CONSECUTIVE_VOICED_FRAMES {
        return false;
    }
    let energies = samples
        .chunks_exact(frame_samples)
        .map(|frame| {
            let mean_square = frame
                .iter()
                .map(|sample| f64::from(*sample).powi(2))
                .sum::<f64>()
                / frame.len() as f64;
            mean_square.sqrt()
        })
        .collect::<Vec<_>>();
    let mut sorted = energies.clone();
    sorted.sort_by(f64::total_cmp);
    let noise_floor = sorted[sorted.len() / 5];
    let voiced_threshold = (noise_floor * NOISE_MULTIPLIER).max(ABSOLUTE_RMS_FLOOR);
    let mut consecutive = 0;
    for (frame, energy) in samples.chunks_exact(frame_samples).zip(energies) {
        let zero_crossings = frame
            .windows(2)
            .filter(|pair| pair[0].is_negative() != pair[1].is_negative())
            .count();
        let zero_crossing_rate = zero_crossings as f64 / (frame.len() - 1) as f64;
        if energy > voiced_threshold && zero_crossing_rate < MAX_VOICED_ZERO_CROSSING_RATE {
            consecutive += 1;
            if consecutive >= MIN_CONSECUTIVE_VOICED_FRAMES {
                return true;
            }
        } else {
            consecutive = 0;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{
        input_device_names, resolve_from_names, samples_have_speech, unique_wav_path,
        wav_has_speech,
    };
    use serde_json::json;

    #[test]
    fn recording_paths_are_unique() {
        assert_ne!(unique_wav_path(), unique_wav_path());
    }

    #[test]
    fn microphone_priority_is_unconfigured_by_default() {
        let selection = resolve_from_names(&["Built-in Microphone".into()]);
        assert!(selection.devices.is_empty());
        assert!(selection.selected_device.is_none());
    }

    #[test]
    fn system_profiler_parser_only_returns_inputs() {
        let value = json!({"SPAudioDataType": [{"_items": [
            {"_name": "MacBook Pro Microphone", "coreaudio_device_input": 3},
            {"_name": "MacBook Pro Speakers", "coreaudio_device_output": 2}
        ]}]});
        assert_eq!(input_device_names(&value), vec!["MacBook Pro Microphone"]);
    }

    #[test]
    fn speech_gate_rejects_silence_steady_noise_and_a_click() {
        let sample_rate = 16_000;
        assert!(!samples_have_speech(
            &vec![0; sample_rate],
            sample_rate as u32
        ));
        assert!(!samples_have_speech(
            &vec![120; sample_rate],
            sample_rate as u32
        ));
        let mut click = vec![0; sample_rate];
        for sample in &mut click[4_000..4_320] {
            *sample = 5_000;
        }
        assert!(!samples_have_speech(&click, sample_rate as u32));

        let mut sustained_broadband_buzz = vec![0; sample_rate];
        for (index, sample) in sustained_broadband_buzz[4_000..7_200]
            .iter_mut()
            .enumerate()
        {
            *sample = if index % 2 == 0 { 2_000 } else { -2_000 };
        }
        assert!(!samples_have_speech(
            &sustained_broadband_buzz,
            sample_rate as u32
        ));
    }

    #[test]
    fn speech_gate_accepts_sustained_voiced_energy() {
        let sample_rate = 16_000;
        let mut samples = vec![40; sample_rate];
        for (index, sample) in samples[4_000..7_200].iter_mut().enumerate() {
            *sample = if index % 32 < 16 { 2_000 } else { -2_000 };
        }
        assert!(samples_have_speech(&samples, sample_rate as u32));
    }

    #[test]
    fn wav_parser_feeds_the_same_speech_gate() {
        let sample_rate = 16_000_u32;
        let mut samples = vec![0_i16; sample_rate as usize];
        for (index, sample) in samples[4_000..7_200].iter_mut().enumerate() {
            *sample = if index % 32 < 16 { 2_000 } else { -2_000 };
        }
        let data_len = (samples.len() * 2) as u32;
        let mut wav = Vec::with_capacity(44 + data_len as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }
        let path =
            std::env::temp_dir().join(format!("phonon-speech-gate-{}.wav", std::process::id()));
        std::fs::write(&path, wav).unwrap();
        assert!(wav_has_speech(&path).unwrap());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn startup_voice_fixture_passes_the_gate() {
        let path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/startup.wav");
        assert!(wav_has_speech(&path).unwrap());
    }
}
