//! The `phonon` command: launches the macOS app and inspects its local data.
//!
//! ```text
//! phonon              # native macOS app
//! phonon engine       # warm JSONL backend the app starts (hidden)
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
    about = "Local voice typing for macOS: Parakeet speech recognition and an on-device correction model",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Open the Phonon app (what `phonon` with no command does). Hold Globe (fn) to talk.
    Bar {
        /// Rebuild the app from this checkout before opening it.
        #[arg(long)]
        rebuild: bool,
    },
    /// Warm speech and correction engines over JSONL on stdin and stdout. The app starts this.
    #[command(hide = true)]
    Engine,
    /// Time speech recognition and correction, cold and warm.
    Bench {
        /// Warm correction runs, cycling through the clips.
        #[arg(short = 'n', long, default_value_t = 5)]
        iters: usize,
        /// Correct this text instead of the built-in clips.
        #[arg(long)]
        text: Option<String>,
        /// Also time the old fluid-1 helper when it is installed (developer comparison).
        #[arg(long, hide = true)]
        baseline: bool,
        /// Skip the cold-start runs.
        #[arg(long)]
        skip_cold: bool,
        /// Also transcribe this WAV file; without it only the speech model load is timed.
        #[arg(long)]
        wav: Option<PathBuf>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Check that the runtime, the models and the local data files are in place.
    Doctor,
    /// Measure latency, from model phases to real key-press-to-insertion timings.
    Profile {
        #[command(subcommand)]
        command: ProfileCommands,
    },
    /// List and edit dictionary terms and exact replacements.
    Dictionary {
        #[command(subcommand)]
        command: DictionaryCommands,
    },
    /// Browse stored recordings and their transcripts.
    Corpus {
        #[command(subcommand)]
        command: CorpusCommands,
    },
    /// Show local dictation totals.
    Stats {
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Export stored dictations for training and evaluation.
    Data {
        #[command(subcommand)]
        command: DataCommands,
    },
}

#[derive(Subcommand, Debug)]
enum DataCommands {
    /// Write one JSONL row per dictation, with a train/eval date split.
    ///
    /// Each row holds the raw transcript, the corrected text, the text you
    /// kept, and its split: train before --eval-from, eval from then on.
    ExportPairs {
        /// Only dictations on or after this UTC date (YYYY-MM-DD).
        #[arg(long)]
        since: Option<String>,
        /// First held-out day (UTC, YYYY-MM-DD); earlier days are train.
        /// Default: 14 days ago.
        #[arg(long)]
        eval_from: Option<String>,
        /// Write the JSONL here instead of stdout.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Only dictations the user edited, with an observation that can
        /// serve as a label.
        #[arg(long)]
        edited_only: bool,
        /// Also export Wispr Flow's history from this flow.sqlite, read
        /// through a temporary copy.
        #[arg(long)]
        wispr_db: Option<PathBuf>,
        /// With --wispr-db: write each exported row's audio as a WAV.
        #[arg(long)]
        with_audio: bool,
        /// Where --with-audio writes the WAVs (default: a temporary directory).
        #[arg(long)]
        audio_dir: Option<PathBuf>,
        /// Ids that are always eval: one per line, or JSONL rows with `id`
        /// or `transcriptEntityId` (existing holdout sets). Repeatable.
        #[arg(long)]
        holdout_ids: Vec<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum DictionaryCommands {
    /// List the dictionary.
    List {
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Add a term, or an exact replacement, with optional spoken forms.
    Add {
        /// The term as it should be written, or the spoken phrase when --replacement is set.
        phrase: String,
        /// Always write this text instead of the phrase.
        #[arg(long)]
        replacement: Option<String>,
        /// Another way the term is heard. Repeatable.
        #[arg(long = "spoken-form")]
        spoken_forms: Vec<String>,
    },
    /// Learn an exact replacement from a stored recording's raw transcript.
    Learn {
        /// Recording id, from `phonon corpus list`.
        id: String,
        /// Text as it appears in the raw transcript.
        #[arg(long)]
        from: String,
        /// Text it should become.
        #[arg(long)]
        to: String,
    },
    /// Import active entries (not snippets) from Wispr Flow's local database.
    ImportWispr {
        /// Path to Wispr Flow's flow.sqlite (default: its usual location).
        #[arg(long)]
        database: Option<PathBuf>,
    },
    /// Import one term per line from a UTF-8 text file, skipping terms already present.
    ImportTxt {
        /// Text file to read (default: dictionary_new_terms.txt in Phonon's data folder).
        path: Option<PathBuf>,
    },
    /// Show which dictionary terms and replacements would apply to some text.
    Test {
        /// Transcript text to test.
        text: String,
        /// Screen text, to test the screen-context path.
        #[arg(long)]
        context: Option<String>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Run the warmed correction model over the hard-case fixtures and score it.
    Evaluate {
        /// Fixture file (default: fixtures/correction_cases.json).
        #[arg(long)]
        fixtures: Option<PathBuf>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum CorpusCommands {
    /// Print the data, dictionary and recordings paths.
    Path,
    /// List stored recordings.
    List {
        /// Only recordings whose transcripts contain this text.
        #[arg(long)]
        search: Option<String>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Show one recording: raw and corrected transcripts, what you meant, and timings.
    Show {
        /// Recording id, from `phonon corpus list`.
        id: String,
    },
    /// Save what you meant to say for a recording (used to measure accuracy).
    SetIntended {
        /// Recording id, from `phonon corpus list`.
        id: String,
        /// The transcript you intended.
        text: String,
    },
    /// Copy recordings from the old History.json format into the recordings folder.
    #[command(hide = true)]
    MigrateLegacy,
    /// Delete one recording, its audio and its metadata.
    Delete {
        /// Recording id, from `phonon corpus list`.
        id: String,
    },
    /// Re-run correction over recordings that have an intended transcript.
    ///
    /// Runs the warmed correction model over every recording with an intended
    /// transcript (plus any fixture files) and reports text and timing per
    /// case. Prompt options come from settings.json and PHONON_PROFILE_*
    /// variables.
    PolishEval {
        /// Fixture JSON files of {id, raw, intended} cases to add.
        #[arg(long)]
        fixtures: Vec<PathBuf>,
        /// Also include recordings without an intended transcript.
        #[arg(long)]
        unlabeled: bool,
        /// Newest corpus recordings to include.
        #[arg(long)]
        limit: Option<usize>,
        /// Runs per case after warmup; all are reported.
        #[arg(long, default_value_t = 2)]
        passes: usize,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum ProfileCommands {
    /// Top Metal kernels in one warmed request. Needs the fluid-1 helper (developer only).
    #[command(hide = true)]
    Kernel {
        /// Kernels to list.
        #[arg(long, default_value_t = 10)]
        top: usize,
        /// Text to correct instead of the built-in sample.
        #[arg(long)]
        text: Option<String>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Warm correction-model timings (prefill and decode) by input length. Prints JSON.
    Model {
        /// Timed runs per input length after warmup.
        #[arg(long, default_value_t = 3)]
        iters: usize,
        /// Input lengths in words, comma separated.
        #[arg(long, value_delimiter = ',', default_value = "8,32,128,256")]
        words: Vec<usize>,
    },
    /// Summarize key-press-to-insertion timings from your recent dictations.
    E2e {
        /// Number of recent dictations to include.
        #[arg(long, default_value_t = 20)]
        last: usize,
        /// Print JSON instead of text.
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
        Commands::Data { command } => run_data(command),
    }
}

fn run_data(command: DataCommands) -> Result<()> {
    match command {
        DataCommands::ExportPairs {
            since,
            eval_from,
            out,
            edited_only,
            wispr_db,
            with_audio,
            audio_dir,
            holdout_ids,
        } => data_cmd::export_pairs(data_cmd::ExportPairsArgs {
            since,
            eval_from,
            out,
            edited_only,
            wispr_db,
            with_audio,
            audio_dir,
            holdout_ids,
        }),
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
        CorpusCommands::PolishEval {
            fixtures,
            unlabeled,
            limit,
            passes,
            json,
        } => data_cmd::polish_eval(
            &phonon_core::project_root(),
            data_cmd::PolishEvalArgs {
                fixtures,
                unlabeled,
                limit,
                passes,
                json,
            },
        ),
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
                println!("phonon Metal kernels");
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
                    "    correction:               {:.1}ms",
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
