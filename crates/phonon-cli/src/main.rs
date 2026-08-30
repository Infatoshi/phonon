//! phonon — personal local dictation CLI (Mac MLX).
//!
//! ```text
//! phonon              # native macOS app
//! phonon bar          # floating macOS pill (Wispr-style)
//! phonon engine       # warm JSONL backend for the bar
//! phonon bench
//! phonon doctor
//! ```

mod bench;
mod data_cmd;
mod doctor;
use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::Command;

pub(crate) fn resolve_runtime_tool(name: &str) -> Option<PathBuf> {
    let bundled = std::env::current_exe().ok().and_then(|executable| {
        executable
            .parent()
            .map(|directory| directory.join(name))
            .filter(|path| path.is_file())
    });
    bundled.or_else(|| which::which(name).ok())
}

#[derive(Parser, Debug)]
#[command(
    name = "phonon",
    version,
    about = "Personal dictation: Parakeet + fluid-1 + MTP (auto-polish, floating bar)",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Floating macOS pill (NSPanel) — hold Globe (fn) to talk, types into frontmost app.
    Bar {
        /// Rebuild the Swift bar before launch.
        #[arg(long)]
        rebuild: bool,
    },
    /// Long-lived warm engines (JSONL stdin/stdout). Used by `phonon bar`.
    Engine,
    /// Profile both shipped weight streams (asr / llm).
    Bench {
        #[arg(short = 'n', long, default_value_t = 5)]
        iters: usize,
        #[arg(long)]
        text: Option<String>,
        /// Also measure the old fluid-1 helper, when installed.
        #[arg(long)]
        baseline: bool,
        #[arg(long)]
        skip_cold: bool,
        #[arg(long)]
        wav: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Check uv / SoX / correction model / project files / bar binary.
    Doctor,
    /// Profile Phonon, from model internals to real keyboard-to-insertion latency.
    Profile {
        #[command(subcommand)]
        command: ProfileCommands,
    },
    /// Inspect and edit the local terms and exact replacements used by ASR/LLM correction.
    Dictionary {
        #[command(subcommand)]
        command: DictionaryCommands,
    },
    /// Inspect the paired WAV + metadata corpus and set intended transcripts.
    Corpus {
        #[command(subcommand)]
        command: CorpusCommands,
    },
    /// Show local dictation usage totals.
    Stats {
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum DictionaryCommands {
    /// List Phonon's JSON dictionary.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Add a canonical term or exact replacement, with optional spoken forms.
    Add {
        phrase: String,
        #[arg(long)]
        replacement: Option<String>,
        #[arg(long = "spoken-form")]
        spoken_forms: Vec<String>,
    },
    /// Learn an exact correction from a stored recording's raw transcript.
    Learn {
        id: String,
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: String,
    },
    /// Import active non-snippet entries from Wispr Flow's local database.
    ImportWispr {
        #[arg(long)]
        database: Option<PathBuf>,
    },
    /// Import one canonical term per line from a UTF-8 text file.
    ImportTxt { path: Option<PathBuf> },
    /// Show candidate retrieval and deterministic replacements for text.
    Test {
        text: String,
        /// OCR text to exercise screen-confirmed retrieval.
        #[arg(long)]
        context: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Run the warmed Fluid/MTP correction pass against difficult fixtures.
    Evaluate {
        #[arg(long)]
        fixtures: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum CorpusCommands {
    /// Print data, dictionary, and corpus paths.
    Path,
    /// List recordings, optionally searching all transcript stages.
    List {
        #[arg(long)]
        search: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Show one recording's raw, final, intended, and inference metadata.
    Show { id: String },
    /// Set the ground-truth/intended transcript for a recording.
    SetIntended { id: String, text: String },
    /// Copy legacy Phonon WAV-backed History.json rows into the paired corpus.
    MigrateLegacy,
    /// Delete one recording directory and its paired WAV/metadata.
    Delete { id: String },
}

#[derive(Subcommand, Debug)]
enum ProfileCommands {
    /// Top literal MLX Metal kernels in one warmed MTP request.
    Kernel {
        #[arg(long, default_value_t = 10)]
        top: usize,
        #[arg(long)]
        text: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Warm LLM prefill, decode, prefix-cache, and MTP phase breakdown.
    Model {
        #[arg(long, default_value_t = 3)]
        iters: usize,
        #[arg(long, value_delimiter = ',', default_value = "8,32,128,256")]
        words: Vec<usize>,
    },
    /// Summarize real keyboard-to-insertion traces recorded by the floating bar.
    E2e {
        #[arg(long, default_value_t = 20)]
        last: usize,
        #[arg(long)]
        json: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Commands::Bar { rebuild: false }) {
        Commands::Bar { rebuild } => run_bar(rebuild),
        Commands::Engine => phonon_core::run_engine_serve(),
        Commands::Bench {
            iters,
            text,
            baseline,
            skip_cold,
            wav,
            json,
        } => bench::run_bench(bench::BenchArgs {
            iters,
            text,
            baseline,
            skip_cold,
            json,
            wav,
        }),
        Commands::Doctor => doctor::run_doctor(),
        Commands::Profile { command } => run_profile(command),
        Commands::Dictionary { command } => run_dictionary(command),
        Commands::Corpus { command } => run_corpus(command),
        Commands::Stats { json } => data_cmd::print_stats(json),
    }
}

fn run_dictionary(command: DictionaryCommands) -> Result<()> {
    match command {
        DictionaryCommands::List { json } => data_cmd::list_dictionary(json),
        DictionaryCommands::Add {
            phrase,
            replacement,
            spoken_forms,
        } => data_cmd::add_dictionary(phrase, replacement, spoken_forms),
        DictionaryCommands::Learn { id, from, to } => {
            data_cmd::learn_from_recording(&id, &from, &to)
        }
        DictionaryCommands::ImportWispr { database } => {
            let database = database
                .map(Ok)
                .unwrap_or_else(data_cmd::default_wispr_database)?;
            data_cmd::import_wispr(&database)
        }
        DictionaryCommands::ImportTxt { path } => {
            let path = path
                .map(Ok)
                .unwrap_or_else(data_cmd::default_dictionary_terms_path)?;
            data_cmd::import_txt(&path)
        }
        DictionaryCommands::Test {
            text,
            context,
            json,
        } => data_cmd::test_dictionary(&text, context.as_deref().unwrap_or_default(), json),
        DictionaryCommands::Evaluate { fixtures, json } => {
            let root = phonon_core::project_root();
            let fixtures = fixtures.unwrap_or_else(|| root.join("fixtures/correction_cases.json"));
            data_cmd::evaluate_dictionary(&root, &fixtures, json)
        }
    }
}

fn run_corpus(command: CorpusCommands) -> Result<()> {
    match command {
        CorpusCommands::Path => data_cmd::show_paths(),
        CorpusCommands::List { search, json } => data_cmd::list_corpus(search.as_deref(), json),
        CorpusCommands::Show { id } => data_cmd::show_recording(&id),
        CorpusCommands::SetIntended { id, text } => data_cmd::set_intended(&id, &text),
        CorpusCommands::MigrateLegacy => data_cmd::migrate_legacy_history(),
        CorpusCommands::Delete { id } => data_cmd::delete_corpus_recording(&id),
    }
}

fn run_profile(command: ProfileCommands) -> Result<()> {
    match command {
        ProfileCommands::Kernel { top, text, json } => {
            let report = phonon_profile::profile_kernels(
                &phonon_core::project_root(),
                &phonon_profile::KernelProfileConfig {
                    top,
                    text: text
                        .unwrap_or_else(|| phonon_profile::KernelProfileConfig::default().text),
                },
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("phonon literal MLX Metal kernels");
                println!("  metric: {}", report.metric);
                println!(
                    "  dispatches: {} across {} unique kernels",
                    report.total_invocations, report.unique_kernels
                );
                println!("  {:>3}  {:>8}  {:>12}  kernel", "#", "calls", "encode ms");
                for (index, row) in report.kernels.iter().enumerate() {
                    println!(
                        "  {:>3}  {:>8}  {:>12.3}  {}",
                        index + 1,
                        row.invocations,
                        row.cpu_encode_ms,
                        row.kernel
                    );
                }
            }
        }
        ProfileCommands::Model { iters, words } => {
            let report = phonon_profile::profile_llm(
                &phonon_core::project_root(),
                &phonon_profile::LlmProfileConfig {
                    input_words: words,
                    steady_iterations: iters,
                    ..Default::default()
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        ProfileCommands::E2e { last, json } => {
            let report =
                phonon_profile::profile_e2e(&phonon_profile::default_e2e_log_path()?, last)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("phonon keyboard → insertion profile");
                println!("  log: {}", report.log_path.display());
                println!(
                    "  samples: {} selected, {} successful",
                    report.samples, report.successful_samples
                );
                println!(
                    "  median key-down → recording: {:.1}ms",
                    report.median.key_down_to_recording_ms
                );
                println!(
                    "  median key-down → panel:     {:.1}ms",
                    report.median.key_down_to_panel_ms
                );
                if let Some(value) = report.median.key_down_to_first_preview_ms {
                    println!("  median key-down → preview:   {value:.1}ms");
                }
                println!(
                    "  median key-up → insertion:  {:.1}ms",
                    report.median.key_up_to_insert_ms
                );
                println!(
                    "    key event → callback:     {:.1}ms",
                    report.median.key_event_to_callback_ms
                );
                println!(
                    "    callback → main actor:    {:.1}ms",
                    report.median.callback_to_main_ms
                );
                println!(
                    "    key-up → capture end:     {:.1}ms",
                    report.median.key_up_to_capture_end_ms
                );
                println!(
                    "    WAV encode:               {:.1}ms",
                    report.median.wav_encode_ms
                );
                println!(
                    "    final ASR:                {:.1}ms",
                    report.median.asr_ms
                );
                println!(
                    "    MTP polish:               {:.1}ms",
                    report.median.polish_ms
                );
                println!(
                    "    insertion event posting:  {:.1}ms",
                    report.median.insertion_post_ms
                );
                println!(
                    "  latest: {} via {}",
                    report.latest.pass_id, report.latest.source
                );
            }
        }
    }
    Ok(())
}

fn bar_root() -> PathBuf {
    phonon_core::project_root().join("bar")
}

fn bar_bin() -> PathBuf {
    bar_root().join("dist/Phonon.app/Contents/MacOS/PhononBar")
}

fn build_bar() -> Result<()> {
    let root = phonon_core::project_root();
    let bar = bar_root();
    if !bar.join("Package.swift").is_file() {
        bail!("missing bar package at {}", bar.display());
    }
    let packager = root.join("scripts/package-bar.sh");
    let status = Command::new(&packager)
        .status()
        .with_context(|| format!("package Swift bar with {}", packager.display()))?;
    if !status.success() {
        bail!("Phonon.app packaging failed");
    }
    Ok(())
}

fn run_bar(rebuild: bool) -> Result<()> {
    if let Some(app) = bundled_app() {
        if rebuild {
            bail!("the installed app cannot rebuild itself; reinstall Phonon instead");
        }
        let status = Command::new("open")
            .arg(&app)
            .status()
            .with_context(|| format!("open {}", app.display()))?;
        if !status.success() {
            bail!("open {} failed with {status}", app.display());
        }
        return Ok(());
    }

    let bin = bar_bin();
    if rebuild || !bin.is_file() {
        eprintln!("building signed Phonon.app…");
        build_bar()?;
    }
    if !bin.is_file() {
        bail!(
            "bar app executable missing at {}\n  run: phonon bar --rebuild",
            bin.display()
        );
    }
    // Ensure phonon is findable for engine subprocess.
    let phonon = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("phonon"));
    let status = Command::new(&bin)
        .env("PHONON_BIN", &phonon)
        .status()
        .context("launch PhononBar")?;
    if !status.success() {
        bail!("PhononBar exited with {status}");
    }
    Ok(())
}

fn bundled_app() -> Option<PathBuf> {
    let executable = std::env::current_exe().ok()?.canonicalize().ok()?;
    let contents = executable.parent()?.parent()?;
    let app = contents.parent()?.to_path_buf();
    (app.extension().and_then(|value| value.to_str()) == Some("app")
        && contents.join("MacOS/PhononBar").is_file())
    .then_some(app)
}
