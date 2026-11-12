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
mod autostart;
mod backends;
mod dictionary;
mod fetch;
mod history;
mod http;
mod manifest;
mod mode;
mod paths;
mod pipeline;
mod polish;
mod settings;
mod single;
mod smoke;
#[cfg(any(windows, target_os = "linux"))]
mod window;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
mod lx;

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
#[command(name = "phonon-win", version)]
#[cfg_attr(windows, command(about = "Phonon dictation for Windows"))]
#[cfg_attr(
    not(windows),
    command(about = "Phonon speech recognition and correction")
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the tray application. This is the default.
    Run,
    /// Open the settings window (the tray's "Settings..." does this).
    Settings {
        /// The page to open: general or history.
        #[arg(long)]
        page: Option<String>,
    },
    /// Download and verify the runtimes and models, then stop.
    Fetch,
    /// Linux: check keyboard and paste access and print the one-time fix.
    Setup,
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
        /// Backend manifest (https://, http:// or file://) to pick a tuned GPU
        /// runtime from. Defaults to the one the app reads.
        #[arg(long)]
        backends: Option<String>,
    },
    /// Internal: run one wave file through the dictation path (the tray's writing
    /// mode and the personal dictionary) and print both texts.
    #[command(hide = true)]
    Dictate {
        #[arg(long)]
        wav: PathBuf,
    },
    /// Internal: list the CUDA devices this environment shows (see backends.rs).
    #[command(hide = true)]
    CudaDevices,
    /// Release tooling: sign a backend manifest, writing <manifest>.sig.
    #[command(hide = true)]
    SignBackends {
        /// File holding the ed25519 seed as 64 hex characters.
        #[arg(long)]
        key: PathBuf,
        manifest: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Run) {
        Command::CudaDevices => backends::print_cuda_devices(),
        Command::Dictate { wav } => dictate(&wav),
        Command::SignBackends { key, manifest } => sign_backends(&key, &manifest),
        Command::Run => {
            let Some(_guard) = single::claim(&paths::data_root())? else {
                eprintln!("phonon: Phonon is already running; this copy exits");
                return Ok(());
            };
            run()
        }
        Command::Settings { page } => settings_window(page.as_deref()),
        Command::Fetch => fetch_all(),
        Command::Setup => setup(),
        Command::Info => {
            info();
            Ok(())
        }
        Command::Selftest {
            wav,
            any_audio,
            backends,
        } => selftest(
            wav,
            any_audio,
            backends.as_deref().or(backends::DEFAULT_MANIFEST),
        ),
    }
}

#[cfg(windows)]
fn run() -> Result<()> {
    win::run()
}

#[cfg(target_os = "linux")]
fn run() -> Result<()> {
    lx::run()
}

#[cfg(any(windows, target_os = "linux"))]
fn settings_window(page: Option<&str>) -> Result<()> {
    let page =
        match page {
            None => None,
            Some(text) => Some(window::Page::parse(text).ok_or_else(|| {
                anyhow::anyhow!("no settings page {text:?}; use general or history")
            })?),
        };
    window::run(page)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn settings_window(_page: Option<&str>) -> Result<()> {
    bail!("the settings window is for Windows and Linux. On macOS use the Phonon app.")
}

#[cfg(target_os = "linux")]
fn setup() -> Result<()> {
    lx::setup()
}

#[cfg(not(target_os = "linux"))]
fn setup() -> Result<()> {
    println!("Nothing to set up on this platform.");
    Ok(())
}

#[cfg(not(any(windows, target_os = "linux")))]
fn run() -> Result<()> {
    bail!("run is not supported on this platform. On macOS use `phonon`.")
}

fn info() {
    #[cfg(not(target_os = "linux"))]
    println!("Phonon for Windows");
    #[cfg(target_os = "linux")]
    println!("Phonon for Linux x86_64");
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

/// Sign a manifest with the release key and say whether this build trusts it.
fn sign_backends(key: &std::path::Path, manifest: &std::path::Path) -> Result<()> {
    let seed = std::fs::read_to_string(key).with_context(|| format!("read {}", key.display()))?;
    let bytes = std::fs::read(manifest).with_context(|| format!("read {}", manifest.display()))?;
    let (signature, public) = backends::sign_manifest(&bytes, &seed)?;
    let mut out = manifest.as_os_str().to_owned();
    out.push(".sig");
    std::fs::write(&out, format!("{signature}\n"))?;
    println!("signed {} with public key {public}", manifest.display());
    if !backends::MANIFEST_KEYS.contains(&public.as_str()) {
        bail!("this key is not in backends::MANIFEST_KEYS; the app will refuse the manifest");
    }
    Ok(())
}

fn fetch_all() -> Result<()> {
    use std::io::{IsTerminal, Write};
    // A terminal gets one line rewritten in place; a log gets one line per file and
    // per ten percent, instead of one per 4 MB chunk.
    let tty = std::io::stdout().is_terminal();
    let mut last = String::new();
    let mut last_step: (&str, u32) = ("", u32::MAX);
    fetch::ensure_all(&mut |progress| {
        let line = progress.line();
        if line == last {
            return;
        }
        last = line.clone();
        if tty {
            print!("\r\x1b[2K{line}");
            let _ = std::io::stdout().flush();
        } else {
            let step = (progress.file, progress.percent() / 10);
            if step != last_step {
                println!("{line}");
                last_step = step;
            }
        }
    })?;
    if tty && !last.is_empty() {
        println!();
    }
    println!("every runtime and model is installed and verified");
    Ok(())
}

/// The release gate. It downloads what is missing, starts the real correction
/// server, transcribes real audio, corrects it, and applies the same two rules
/// the macOS readiness contract applies.
fn selftest(wav: Option<PathBuf>, any_audio: bool, backends: Option<&str>) -> Result<()> {
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

    let started = std::time::Instant::now();
    let mut engine = pipeline::Engine::start(backends)?;
    println!(
        "  correction device: {:?} (engine ready in {} ms)",
        engine.device(),
        started.elapsed().as_millis()
    );
    // as at key-down: a saturated CPU (Windows) moves the correction to one thread
    #[cfg(windows)]
    {
        crate::win::load::start();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let busy = crate::win::load::busy();
        let started = std::time::Instant::now();
        engine.prepare(busy);
        println!(
            "  cpu busy {:.0}%, one-thread corrector: {} (prepare {} ms)",
            busy.unwrap_or(0.0) * 100.0,
            engine.lean(),
            started.elapsed().as_millis()
        );
    }
    let result = startup_check_with_fallback(&mut engine, &wav, any_audio)?;
    println!("  raw:      {:?}", result.raw);
    println!("  polished: {:?}", result.polished);
    println!(
        "  asr {:.0} ms, correction {:.0} ms on {:?}",
        result.asr_ms,
        result.polish_ms,
        engine.device()
    );
    println!("selftest passed");
    Ok(())
}

/// One wave file through the dictation path, as a dictation from the hotkey runs it.
fn dictate(wav: &std::path::Path) -> Result<()> {
    fetch_all()?;
    let engine = pipeline::Engine::start(backends::DEFAULT_MANIFEST)?;
    let entries = dictionary::load().entries.len();
    let result = engine.dictate(wav)?;
    println!("  dictionary entries: {entries}");
    println!("  raw:      {:?}", result.raw);
    println!("  polished: {:?}", result.polished);
    println!(
        "  asr {:.0} ms, correction {:.0} ms",
        result.asr_ms, result.polish_ms
    );
    Ok(())
}

/// Run the readiness check; a GPU that fails it (a driver that answers wrongly)
/// moves the correction model to the CPU and the check runs again, as the
/// Windows app does.
pub fn startup_check_with_fallback(
    engine: &mut pipeline::Engine,
    wav: &std::path::Path,
    any_audio: bool,
) -> Result<PassResult> {
    let check = |engine: &pipeline::Engine| -> Result<PassResult> {
        let result = engine.run(wav)?;
        if any_audio {
            assert_usable(&result)?;
        } else {
            assert_startup(&result)?;
        }
        Ok(result)
    };
    check_with_fallback(engine, check)
}

/// What the startup check needs from an engine (a trait so tests can use a fake).
trait Fallback {
    fn device(&self) -> polish::Device;
    fn confirm_backend(&mut self);
    fn fall_back_to_cpu(&mut self) -> Result<()>;
}

impl Fallback for pipeline::Engine {
    fn device(&self) -> polish::Device {
        pipeline::Engine::device(self)
    }
    fn confirm_backend(&mut self) {
        pipeline::Engine::confirm_backend(self)
    }
    fn fall_back_to_cpu(&mut self) -> Result<()> {
        pipeline::Engine::fall_back_to_cpu(self)
    }
}

/// `check` on the engine; passing confirms the tuned build in use, and failing on
/// the GPU moves the correction to the CPU and checks again there.
fn check_with_fallback<E: Fallback, R>(
    engine: &mut E,
    check: impl Fn(&E) -> Result<R>,
) -> Result<R> {
    match check(engine) {
        Ok(result) => {
            engine.confirm_backend();
            Ok(result)
        }
        Err(error) if engine.device() == polish::Device::Gpu => {
            eprintln!("phonon: startup check failed on the GPU ({error:#}); retrying on the CPU");
            engine.fall_back_to_cpu()?;
            check(engine)
        }
        Err(error) => Err(error),
    }
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

    /// An engine whose check passes or fails per device, recording what happened.
    struct FakeEngine {
        device: polish::Device,
        gpu_passes: bool,
        cpu_starts: bool,
        confirmed: u32,
        fallbacks: u32,
    }

    impl FakeEngine {
        fn on(device: polish::Device, gpu_passes: bool) -> Self {
            Self {
                device,
                gpu_passes,
                cpu_starts: true,
                confirmed: 0,
                fallbacks: 0,
            }
        }
    }

    impl Fallback for FakeEngine {
        fn device(&self) -> polish::Device {
            self.device
        }
        fn confirm_backend(&mut self) {
            self.confirmed += 1;
        }
        fn fall_back_to_cpu(&mut self) -> Result<()> {
            self.fallbacks += 1;
            if !self.cpu_starts {
                bail!("the CPU server did not start");
            }
            self.device = polish::Device::Cpu;
            Ok(())
        }
    }

    fn fake_check(engine: &FakeEngine) -> Result<polish::Device> {
        match engine.device {
            polish::Device::Gpu if !engine.gpu_passes => bail!("the GPU answered wrongly"),
            device => Ok(device),
        }
    }

    #[test]
    fn a_gpu_that_passes_the_startup_check_is_confirmed() {
        let mut engine = FakeEngine::on(polish::Device::Gpu, true);
        let device = check_with_fallback(&mut engine, fake_check).unwrap();
        assert_eq!(device, polish::Device::Gpu);
        assert_eq!((engine.confirmed, engine.fallbacks), (1, 0));
    }

    /// A GPU that answers wrongly ends on the CPU, and its build is not confirmed
    /// (the family's earlier build stays on disk for the next start).
    #[test]
    fn a_gpu_that_fails_the_startup_check_ends_on_the_cpu() {
        let mut engine = FakeEngine::on(polish::Device::Gpu, false);
        let device = check_with_fallback(&mut engine, fake_check).unwrap();
        assert_eq!(device, polish::Device::Cpu);
        assert_eq!((engine.confirmed, engine.fallbacks), (0, 1));
    }

    #[test]
    fn a_failing_cpu_check_is_an_error_without_a_fallback() {
        let mut engine = FakeEngine::on(polish::Device::Cpu, true);
        let failing = |_: &FakeEngine| -> Result<()> { bail!("wrong transcript") };
        assert!(check_with_fallback(&mut engine, failing).is_err());
        assert_eq!((engine.confirmed, engine.fallbacks), (0, 0));
        // the GPU fails and the CPU server cannot start: the error surfaces
        let mut engine = FakeEngine::on(polish::Device::Gpu, false);
        engine.cpu_starts = false;
        assert!(check_with_fallback(&mut engine, fake_check).is_err());
        assert_eq!(engine.fallbacks, 1);
    }
}
