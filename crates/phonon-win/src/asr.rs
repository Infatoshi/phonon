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

/// `CREATE_NO_WINDOW`: run a console child without a console window.
#[cfg(windows)]
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// A ready recogniser: the tool and the weights it needs, and the resident
/// model when the sherpa-onnx library loaded (otherwise every pass spawns the
/// command-line tool, which reloads 0.6 GB of weights each time).
#[derive(Debug, Clone)]
pub struct Recognizer {
    tool: PathBuf,
    model_dir: PathBuf,
    threads: usize,
    resident: Option<std::sync::Arc<resident::Resident>>,
}

/// Parakeet held in memory through the sherpa-onnx C API (csrc/sherpa_shim.c).
mod resident {
    use std::ffi::{c_char, c_int, c_void, CStr, CString};
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    use anyhow::{anyhow, bail, Result};

    extern "C" {
        fn phonon_asr_bind(lookup: extern "C" fn(*const c_char) -> *mut c_void) -> c_int;
        fn phonon_asr_create(
            encoder: *const c_char,
            decoder: *const c_char,
            joiner: *const c_char,
            tokens: *const c_char,
            threads: c_int,
            provider: *const c_char,
        ) -> *const c_void;
        fn phonon_asr_file(recognizer: *const c_void, wav: *const c_char) -> *mut c_char;
        fn phonon_asr_samples(
            recognizer: *const c_void,
            sample_rate: c_int,
            samples: *const f32,
            num_samples: c_int,
        ) -> *mut c_char;
        fn phonon_asr_free(text: *mut c_char);
        fn phonon_asr_last_error() -> *const c_char;
    }

    /// The message of the last decode that ONNX Runtime failed (csrc/guard.cc).
    fn last_error() -> String {
        let text = unsafe { phonon_asr_last_error() };
        if text.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    }

    /// The loaded library; it stays loaded for the life of the process.
    static LIBRARY: OnceLock<libloading::Library> = OnceLock::new();

    extern "C" fn lookup(name: *const c_char) -> *mut c_void {
        let Some(lib) = LIBRARY.get() else {
            return std::ptr::null_mut();
        };
        let name = unsafe { CStr::from_ptr(name) }.to_bytes_with_nul();
        unsafe { lib.get::<*mut c_void>(name) }
            .map(|s| *s)
            .unwrap_or(std::ptr::null_mut())
    }

    pub struct Resident {
        handle: Mutex<usize>,
    }

    impl std::fmt::Debug for Resident {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("Resident")
        }
    }

    #[cfg(windows)]
    fn open(runtime: &Path) -> Result<libloading::Library> {
        use libloading::os::windows::{Library, LOAD_WITH_ALTERED_SEARCH_PATH};
        // altered search path: onnxruntime.dll is found beside the C API dll
        let path = runtime.join("lib").join("sherpa-onnx-c-api.dll");
        Ok(
            unsafe { Library::load_with_flags(&path, LOAD_WITH_ALTERED_SEARCH_PATH) }
                .map_err(|e| anyhow!("load {}: {e}", path.display()))?
                .into(),
        )
    }

    #[cfg(not(windows))]
    fn open(runtime: &Path) -> Result<libloading::Library> {
        let lib = runtime.join("lib");
        // load onnxruntime by full path first, so the C API's dependency on it
        // resolves without LD_LIBRARY_PATH in Phonon's own process
        for dep in ["libonnxruntime.so", "libonnxruntime.dylib"] {
            if lib.join(dep).is_file() {
                std::mem::forget(
                    unsafe { libloading::Library::new(lib.join(dep)) }
                        .map_err(|e| anyhow!("load {dep}: {e}"))?,
                );
            }
        }
        let path = ["libsherpa-onnx-c-api.so", "libsherpa-onnx-c-api.dylib"]
            .iter()
            .map(|n| lib.join(n))
            .find(|p| p.is_file())
            .ok_or_else(|| anyhow!("no sherpa-onnx C API library in {}", lib.display()))?;
        unsafe { libloading::Library::new(&path) }
            .map_err(|e| anyhow!("load {}: {e}", path.display()))
    }

    /// ONNX Runtime session options for the resident model, read by sherpa-onnx from the file named after
    /// "cpu:" in the provider string. With its CPU arena and memory patterns on (the default), ONNX Runtime
    /// keeps the activation memory of the longest pass so far for the life of the process: on the Ubuntu QA VM
    /// Phonon grew from 0.70 GB to 1.24 GB RSS after one 68 s dictation and never came back (2026-11-10).
    const SESSION_CONFIG: &str = "EnableCpuMemArena=0\nEnableMemPattern=0\n";

    /// "cpu:<config>", or plain "cpu" (the old behaviour) when the file cannot be written.
    fn provider(runtime: &Path) -> String {
        let path = runtime.join("phonon-session.cfg");
        let current = std::fs::read_to_string(&path).ok();
        if current.as_deref() == Some(SESSION_CONFIG)
            || std::fs::write(&path, SESSION_CONFIG).is_ok()
        {
            format!("cpu:{}", path.display())
        } else {
            "cpu".to_string()
        }
    }

    impl Resident {
        pub fn load(runtime: &Path, model_dir: &Path, threads: usize) -> Result<Self> {
            if LIBRARY.get().is_none() {
                let _ = LIBRARY.set(open(runtime)?);
            }
            let missing = unsafe { phonon_asr_bind(lookup) };
            if missing != 0 {
                bail!("sherpa-onnx C API is missing symbol #{missing}");
            }
            let c = |name: &str| {
                CString::new(model_dir.join(name).to_string_lossy().as_bytes()).unwrap()
            };
            let (enc, dec, joi, tok) = (
                c("encoder.int8.onnx"),
                c("decoder.int8.onnx"),
                c("joiner.int8.onnx"),
                c("tokens.txt"),
            );
            let provider = CString::new(provider(runtime).as_bytes()).unwrap();
            let handle = unsafe {
                phonon_asr_create(
                    enc.as_ptr(),
                    dec.as_ptr(),
                    joi.as_ptr(),
                    tok.as_ptr(),
                    threads as c_int,
                    provider.as_ptr(),
                )
            };
            if handle.is_null() {
                bail!("sherpa-onnx could not create the recogniser");
            }
            Ok(Self {
                handle: Mutex::new(handle as usize),
            })
        }

        /// Transcribe mono samples in -1..1 at `sample_rate`, without a file.
        pub fn transcribe_samples(&self, sample_rate: u32, samples: &[f32]) -> Result<String> {
            let handle = self
                .handle
                .lock()
                .map_err(|_| anyhow!("recogniser lock poisoned"))?;
            let text = unsafe {
                phonon_asr_samples(
                    *handle as *const c_void,
                    sample_rate as c_int,
                    samples.as_ptr(),
                    samples.len() as c_int,
                )
            };
            if text.is_null() {
                bail!("sherpa-onnx could not transcribe: {}", last_error());
            }
            let out = unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .trim()
                .to_string();
            unsafe { phonon_asr_free(text) };
            release_freed_memory();
            Ok(out)
        }

        pub fn transcribe(&self, wav: &Path) -> Result<String> {
            let handle = self
                .handle
                .lock()
                .map_err(|_| anyhow!("recogniser lock poisoned"))?;
            let path = CString::new(wav.to_string_lossy().as_bytes())?;
            let text = unsafe { phonon_asr_file(*handle as *const c_void, path.as_ptr()) };
            if text.is_null() {
                let error = last_error();
                if error.is_empty() {
                    bail!("sherpa-onnx could not read {}", wav.display());
                }
                bail!(
                    "sherpa-onnx could not transcribe {}: {error}",
                    wav.display()
                );
            }
            let out = unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .trim()
                .to_string();
            unsafe { phonon_asr_free(text) };
            release_freed_memory();
            Ok(out)
        }
    }

    /// Give the pages a pass freed back to the system. With the arena off, ONNX Runtime frees a pass's
    /// activations, but glibc keeps the freed heap: after one 68 s dictation Phonon stayed at 0.90-1.05 GB
    /// RSS instead of 0.71 GB (Ubuntu QA VM, 2026-11-10). A few ms, once per pass.
    fn release_freed_memory() {
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        unsafe {
            libc::malloc_trim(0);
        }
    }
}

/// The longest audio one recogniser pass gets. The exported encoder takes at most
/// 400 s (its relative-position table is 5,000 frames of 80 ms: 399.7 s decodes,
/// 400.2 s makes ONNX Runtime throw, which terminated Phonon: exit 134, Linux build,
/// 2026-11-08), and long passes also lose words. LibriSpeech test-clean, 915 s, WER
/// by pass length: 1.31 (30 s), 1.35 (60 s), 2.03 (120 s), 2.92 (240 s), 2.83 (360 s),
/// 1.35 one utterance at a time. Owner dictations over 60 s (77 clips, 155 min):
/// 21.20 in 60 s passes vs 21.66 in one pass (-0.46 [-0.84, +0.01]). x86 CPU, sherpa
/// int8, 2026-11-08. Android uses the same rule (android Engines.kt `Passes`).
pub const MAX_PASS_SECONDS: usize = 60;

/// [start, end) sample ranges of at most `MAX_PASS_SECONDS` that cover `samples`,
/// each cut at the quietest 100 ms of the last quarter of its window (15 s), so a cut
/// lands in a pause rather than inside a word where one exists.
pub fn passes(samples: &[f32], sample_rate: u32) -> Vec<(usize, usize)> {
    let rate = sample_rate.max(10) as usize;
    let max = MAX_PASS_SECONDS * rate;
    let frame = rate / 10;
    let search = max / 4;
    let mut out = Vec::new();
    let mut start = 0;
    while samples.len() - start > max {
        let limit = start + max;
        let (mut cut, mut quietest) = (limit, f32::INFINITY);
        let mut at = limit - search;
        while at + frame <= limit {
            let energy: f32 = samples[at..at + frame].iter().map(|x| x * x).sum();
            if energy < quietest {
                quietest = energy;
                cut = at + frame / 2;
            }
            at += frame;
        }
        out.push((start, cut));
        start = cut;
    }
    out.push((start, samples.len()));
    out
}

/// A wave file's mono samples in -1..1 and its rate, when it is longer than one
/// pass; `None` for anything shorter (the caller hands the file over as it is).
fn read_if_long(wav: &Path) -> Result<Option<(u32, Vec<f32>)>> {
    let mut reader =
        hound::WavReader::open(wav).with_context(|| format!("open {}", wav.display()))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    if (reader.duration() as usize) <= MAX_PASS_SECONDS * spec.sample_rate as usize {
        return Ok(None);
    }
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1_i64 << (spec.bits_per_sample.max(1) - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok(Some((spec.sample_rate, mono)))
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
            resident: None,
        })
    }

    /// Load Parakeet into this process once, so passes skip the per-call model
    /// load. On failure the command-line tool keeps working, per pass.
    pub fn load_resident(&mut self) {
        let runtime = fetch::component_dir(&manifest::SHERPA_RUNTIME);
        let started = std::time::Instant::now();
        match resident::Resident::load(&runtime, &self.model_dir, self.threads) {
            Ok(r) => {
                eprintln!(
                    "phonon: speech model resident in {} ms",
                    started.elapsed().as_millis()
                );
                self.resident = Some(std::sync::Arc::new(r));
            }
            Err(error) => eprintln!(
                "phonon: resident speech model unavailable, using the tool per pass: {error:#}"
            ),
        }
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

    /// Whether Parakeet is loaded in this process (`transcribe_samples` works).
    pub fn is_resident(&self) -> bool {
        self.resident.is_some()
    }

    /// Transcribe samples in memory with the resident model; `None` when it is not
    /// resident and the caller must write a file for the per-pass tool.
    pub fn transcribe_samples(
        &self,
        sample_rate: u32,
        samples: &[f32],
    ) -> Option<Result<Transcript>> {
        let r = self.resident.as_ref()?;
        let mut parts = Vec::new();
        for (start, end) in passes(samples, sample_rate) {
            match r.transcribe_samples(sample_rate, &samples[start..end]) {
                Ok(text) if !text.is_empty() => parts.push(text),
                Ok(_) => {}
                Err(error) => return Some(Err(error)),
            }
        }
        Some(Ok(Transcript {
            text: parts.join(" "),
        }))
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
        if let Some(r) = &self.resident {
            // Too long for one pass: read it here and split it (see `passes`).
            if let Some((rate, samples)) = read_if_long(&wav)? {
                if let Some(result) = self.transcribe_samples(rate, &samples) {
                    return result;
                }
            }
            return Ok(Transcript {
                text: r.transcribe(&wav)?,
            });
        }
        let mut command = Command::new(&self.tool);
        #[cfg(windows)]
        {
            // A console tool started from a tray app gets its own console
            // window unless told otherwise; it flashed on the desktop on every
            // utterance on a real Windows Server 2022 box.
            use std::os::windows::process::CommandExt;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        #[cfg(target_os = "linux")]
        crate::linux::runtime_libraries(
            &mut command,
            &fetch::component_dir(&manifest::SHERPA_RUNTIME),
        )?;
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

    #[test]
    fn short_audio_is_one_pass() {
        let samples = vec![0.1_f32; 16_000 * 45];
        assert_eq!(passes(&samples, 16_000), vec![(0, samples.len())]);
        assert_eq!(passes(&[], 16_000), vec![(0, 0)]);
    }

    #[test]
    fn long_audio_splits_in_the_pauses_and_every_pass_fits() {
        let rate = 16_000;
        // 150 s of "speech" with a 200 ms pause at 55 s and another at 110 s
        let mut samples = vec![0.5_f32; rate * 150];
        for pause in [55, 110] {
            samples[pause * rate..pause * rate + rate / 5].fill(0.0);
        }
        let windows = passes(&samples, rate as u32);
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].0, 0);
        assert_eq!(windows.last().unwrap().1, samples.len());
        for pair in windows.windows(2) {
            assert_eq!(pair[0].1, pair[1].0, "passes must be contiguous");
        }
        for (start, end) in &windows {
            assert!(end - start <= MAX_PASS_SECONDS * rate);
        }
        // each cut falls inside its pause
        for (cut, pause) in [(windows[0].1, 55), (windows[1].1, 110)] {
            assert!(
                (pause * rate..pause * rate + rate / 5).contains(&cut),
                "cut {cut}"
            );
        }
    }

    #[test]
    fn audio_without_a_pause_still_fits() {
        let samples = vec![0.5_f32; 16_000 * 1000];
        let windows = passes(&samples, 16_000);
        assert!(windows
            .iter()
            .all(|(start, end)| end - start <= MAX_PASS_SECONDS * 16_000));
        assert_eq!(windows.last().unwrap().1, samples.len());
    }

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
