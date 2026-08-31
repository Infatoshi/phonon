//! Microphone capture through WASAPI, by way of cpal.
//!
//! The recorder follows the system default input device and takes it in whatever
//! format the device offers, then writes one single-channel 16-bit wave file per
//! pass. sherpa-onnx resamples internally, so the file keeps the device rate and
//! no resampler is needed here.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;

/// Single-channel samples collected so far.
struct Buffer {
    samples: Vec<i16>,
}

/// A capture in progress. It must stay on the thread that started it: a cpal
/// stream is not safe to move between threads on Windows.
pub struct Recording {
    stream: cpal::Stream,
    buffer: Arc<Mutex<Buffer>>,
    sample_rate: u32,
    device_name: String,
}

/// Mix interleaved frames down to one channel.
fn downmix(frame: &[f32], channels: u16) -> i16 {
    let sum: f32 = frame.iter().copied().sum();
    let mean = sum / channels.max(1) as f32;
    (mean.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

/// Start recording from the system default input device.
pub fn start() -> Result<Recording> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no microphone is available"))?;
    let device_name = device.name().unwrap_or_else(|_| "input".into());
    let supported = device
        .default_input_config()
        .context("read the microphone's default format")?;
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let channels = config.channels;
    let sample_rate = config.sample_rate.0;

    let buffer = Arc::new(Mutex::new(Buffer {
        samples: Vec::with_capacity(sample_rate as usize * 8),
    }));
    let sink = Arc::clone(&buffer);
    let on_error = |error| eprintln!("phonon: microphone error: {error}");

    macro_rules! build {
        ($sample:ty, $to_f32:expr) => {{
            let sink = Arc::clone(&sink);
            device.build_input_stream(
                &config,
                move |data: &[$sample], _: &cpal::InputCallbackInfo| {
                    let Ok(mut buffer) = sink.lock() else { return };
                    for frame in data.chunks(channels as usize) {
                        let mixed: Vec<f32> = frame.iter().copied().map($to_f32).collect();
                        buffer.samples.push(downmix(&mixed, channels));
                    }
                },
                on_error,
                None,
            )
        }};
    }

    let stream = match sample_format {
        SampleFormat::F32 => build!(f32, |value: f32| value),
        SampleFormat::I16 => build!(i16, |value: i16| value as f32 / i16::MAX as f32),
        SampleFormat::U16 => {
            build!(u16, |value: u16| (value as f32 - 32768.0) / 32768.0)
        }
        SampleFormat::I32 => {
            build!(i32, |value: i32| value as f32 / i32::MAX as f32)
        }
        other => bail!("the microphone offers {other:?}, which Phonon cannot read"),
    }
    .context("open the microphone")?;
    stream.play().context("start the microphone")?;

    Ok(Recording {
        stream,
        buffer,
        sample_rate,
        device_name,
    })
}

impl Recording {
    /// Which device is being recorded.
    pub fn device(&self) -> &str {
        &self.device_name
    }

    /// Stop, and write what was captured to `path`.
    pub fn finish(self, path: &Path) -> Result<PathBuf> {
        drop(self.stream);
        let samples = {
            let buffer = self
                .buffer
                .lock()
                .map_err(|_| anyhow!("the capture buffer was poisoned"))?;
            buffer.samples.clone()
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec)
            .with_context(|| format!("create {}", path.display()))?;
        for sample in &samples {
            writer.write_sample(*sample)?;
        }
        writer.finalize().context("close the wave file")?;
        Ok(path.to_path_buf())
    }

    /// Seconds captured so far.
    pub fn seconds(&self) -> f64 {
        self.buffer
            .lock()
            .map(|buffer| buffer.samples.len() as f64 / self.sample_rate.max(1) as f64)
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::downmix;

    #[test]
    fn a_centred_stereo_frame_becomes_one_sample() {
        assert_eq!(downmix(&[1.0, 1.0], 2), i16::MAX);
        assert_eq!(downmix(&[0.0, 0.0], 2), 0);
        assert_eq!(downmix(&[1.0, -1.0], 2), 0);
    }

    /// A device that clips must not wrap around to the opposite sign.
    #[test]
    fn clipping_saturates() {
        assert_eq!(downmix(&[4.0, 4.0], 2), i16::MAX);
        assert_eq!(downmix(&[-4.0, -4.0], 2), -i16::MAX);
    }
}
