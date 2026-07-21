use anyhow::{bail, Context, Result};
use phonon_llm::{
    fluid_drafter_dir, fluid_helper, fluid_model_dir, timing_from_resp, RunTiming, ServeJson,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct LlmProfileConfig {
    pub input_words: Vec<usize>,
    pub steady_iterations: usize,
    pub include_plain: bool,
    pub include_mtp: bool,
}

impl Default for LlmProfileConfig {
    fn default() -> Self {
        Self {
            input_words: vec![8, 32, 128, 256],
            steady_iterations: 3,
            include_plain: true,
            include_mtp: true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfileSample {
    pub wall_ms: f64,
    pub ttft_ms: f64,
    pub decode_ms: f64,
    pub generated_tokens: u64,
    pub tokens_per_second: f64,
    pub prompt_tokens: Option<u64>,
    pub prefix_tokens: Option<u64>,
    pub prefill_tokens: Option<u64>,
    pub prefix_cache_hit: Option<bool>,
    pub mtp_accept_percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfileRow {
    pub mode: String,
    pub input_words: usize,
    pub model_load_ms: f64,
    pub startup_demo_ms: f64,
    pub warm_median: ProfileSample,
}

#[derive(Debug, Clone, Serialize)]
pub struct LlmProfileReport {
    pub rows: Vec<ProfileRow>,
}

#[derive(Debug, Clone)]
pub struct KernelProfileConfig {
    pub top: usize,
    pub text: String,
}

impl Default for KernelProfileConfig {
    fn default() -> Self {
        Self {
            top: 10,
            text: "please clean this spoken developer dictation while keeping every technical detail and the original intent intact".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct KernelProfileRow {
    pub kernel: String,
    pub invocations: u64,
    pub cpu_encode_ms: f64,
    pub mean_cpu_encode_us: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct KernelProfileReport {
    pub metric: String,
    pub total_invocations: u64,
    pub unique_kernels: usize,
    pub kernels: Vec<KernelProfileRow>,
}

pub fn profile_kernels(root: &Path, config: &KernelProfileConfig) -> Result<KernelProfileReport> {
    #[cfg(not(target_os = "macos"))]
    bail!("literal MLX kernel profiling requires macOS Metal");

    let helper = fluid_helper();
    let model = fluid_model_dir();
    let drafter = fluid_drafter_dir();
    if !helper.is_file() || !model.is_dir() {
        bail!("fluid-1 MLX helper or model is not installed");
    }

    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let temp = std::env::temp_dir().join(format!("phonon-kernel-{}-{nonce}", std::process::id()));
    let helpers = temp.join("Contents/Helpers");
    fs::create_dir_all(&helpers)?;
    let result = profile_kernels_in_temp(root, config, &helper, &model, &drafter, &temp, &helpers);
    let _ = fs::remove_dir_all(&temp);
    result
}

fn profile_kernels_in_temp(
    root: &Path,
    config: &KernelProfileConfig,
    helper: &Path,
    model: &Path,
    drafter: &Path,
    temp: &Path,
    helpers: &Path,
) -> Result<KernelProfileReport> {
    let source = root.join("crates/phonon-profile/assets/metal_dispatch_profile.m");
    let hook = temp.join("libphonon_metal_dispatch_profile.dylib");
    run_checked(
        Command::new("clang")
            .args([
                "-dynamiclib",
                "-fobjc-arc",
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
            ])
            .arg(&source)
            .arg("-o")
            .arg(&hook),
        "compile Metal dispatch profiler",
    )?;

    let helper_name = helper.file_name().context("MLX helper has no filename")?;
    let copied_helper = helpers.join(helper_name);
    fs::copy(helper, &copied_helper)?;
    let installed_helpers = helper.parent().context("MLX helper has no parent")?;
    for entry in fs::read_dir(installed_helpers)? {
        let entry = entry?;
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &helpers.join(entry.file_name()))?;
        }
    }
    let _ = Command::new("codesign")
        .args(["--remove-signature"])
        .arg(&copied_helper)
        .status();
    run_checked(
        Command::new("codesign")
            .args(["-s", "-"])
            .arg(&copied_helper),
        "ad-hoc sign profiling copy of MLX helper",
    )?;

    let flag = temp.join("record-dispatches");
    let mut command = Command::new(&copied_helper);
    command
        .arg("serve-json")
        .arg("--model-dir")
        .arg(model)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("DYLD_INSERT_LIBRARIES", &hook)
        .env("PHONON_PROFILE_FLAG", &flag);
    if drafter.is_dir() {
        command
            .arg("--mtp-drafter-dir")
            .arg(drafter)
            .args(["--draft-block-size", "6"]);
    }
    let prompt = root.join("prompts/polish_v1.txt");
    if prompt.is_file() {
        command.arg("--system-prompt-file").arg(prompt);
    }

    let mut child = command.spawn().context("spawn instrumented MLX helper")?;
    let mut stdin = child.stdin.take().context("instrumented helper stdin")?;
    let stdout = child.stdout.take().context("instrumented helper stdout")?;
    let stderr = child.stderr.take().context("instrumented helper stderr")?;
    let stderr_reader = std::thread::spawn(move || {
        let mut output = String::new();
        for line in BufReader::new(stderr).lines().map_while(|line| line.ok()) {
            output.push_str(&line);
            output.push('\n');
        }
        output
    });
    let mut stdout = BufReader::new(stdout);

    request(
        &mut stdin,
        &mut stdout,
        &serde_json::json!({"command": "warmup"}),
    )?;
    fs::write(&flag, b"record")?;
    let response = request(
        &mut stdin,
        &mut stdout,
        &serde_json::json!({"command": "run", "inputText": config.text}),
    )?;
    let _ = fs::remove_file(&flag);
    if response.get("ok").and_then(Value::as_bool) != Some(true) {
        bail!("instrumented MLX request failed: {response}");
    }
    let _ = request(
        &mut stdin,
        &mut stdout,
        &serde_json::json!({"command": "shutdown"}),
    );
    drop(stdin);
    let status = child.wait()?;
    let stderr = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("stderr reader panicked"))?;
    if !status.success() {
        bail!(
            "instrumented MLX helper exited with {status}: {}",
            tail(&stderr, 12)
        );
    }
    parse_kernel_profile(&stderr, config.top)
}

fn request(stdin: &mut impl Write, stdout: &mut impl BufRead, payload: &Value) -> Result<Value> {
    writeln!(stdin, "{payload}")?;
    stdin.flush()?;
    let mut line = String::new();
    loop {
        line.clear();
        if stdout.read_line(&mut line)? == 0 {
            bail!("instrumented MLX helper reached EOF");
        }
        if let Ok(value) = serde_json::from_str(line.trim()) {
            return Ok(value);
        }
    }
}

fn parse_kernel_profile(stderr: &str, top: usize) -> Result<KernelProfileReport> {
    let mut totals: HashMap<String, (u64, u64)> = HashMap::new();
    for line in stderr.lines() {
        let Some(payload) = line.strip_prefix("PHONON_KERNEL\t") else {
            continue;
        };
        let Some((name, nanoseconds)) = payload.rsplit_once('\t') else {
            continue;
        };
        let nanoseconds: u64 = nanoseconds.parse()?;
        let entry = totals.entry(name.to_string()).or_default();
        entry.0 += 1;
        entry.1 += nanoseconds;
    }
    if totals.is_empty() {
        bail!("the warmed MLX request emitted no named Metal dispatches");
    }
    let total_invocations = totals.values().map(|(calls, _)| calls).sum();
    let unique_kernels = totals.len();
    let mut kernels: Vec<_> = totals
        .into_iter()
        .map(|(kernel, (invocations, nanoseconds))| KernelProfileRow {
            kernel,
            invocations,
            cpu_encode_ms: nanoseconds as f64 / 1_000_000.0,
            mean_cpu_encode_us: nanoseconds as f64 / invocations as f64 / 1_000.0,
        })
        .collect();
    kernels.sort_by(|left, right| {
        right
            .cpu_encode_ms
            .total_cmp(&left.cpu_encode_ms)
            .then_with(|| left.kernel.cmp(&right.kernel))
    });
    kernels.truncate(top.max(1));
    Ok(KernelProfileReport {
        metric: "CPU time encoding literal MLX Metal dispatches; GPU per-kernel counters are unavailable on this M4 Max".into(),
        total_invocations,
        unique_kernels,
        kernels,
    })
}

fn copy_dir(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn run_checked(command: &mut Command, context: &str) -> Result<()> {
    let output = command.output().with_context(|| context.to_string())?;
    if !output.status.success() {
        bail!(
            "{context}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

fn tail(text: &str, lines: usize) -> String {
    text.lines()
        .rev()
        .take(lines)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct E2EProfileRecord {
    pub timestamp: String,
    pub pass_id: String,
    pub source: String,
    pub text_characters: usize,
    pub insertion_succeeded: bool,
    pub key_event_to_callback_ms: f64,
    pub callback_to_main_ms: f64,
    pub key_down_to_recording_ms: f64,
    pub key_down_to_panel_ms: f64,
    pub key_down_to_first_stream_chunk_ms: Option<f64>,
    pub key_down_to_first_partial_ms: Option<f64>,
    pub key_down_to_first_preview_ms: Option<f64>,
    pub dictation_ms: f64,
    pub key_up_to_main_ms: f64,
    pub key_up_to_capture_end_ms: f64,
    pub wav_encode_ms: f64,
    pub asr_ms: f64,
    pub polish_ms: f64,
    pub insertion_post_ms: f64,
    pub key_up_to_insert_ms: f64,
    pub key_down_to_insert_ms: f64,
    pub asr_reported_ms: Option<f64>,
    pub llm_reported_wall_ms: Option<f64>,
    pub llm_ttft_ms: Option<f64>,
    pub llm_tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct E2EMedian {
    pub key_event_to_callback_ms: f64,
    pub callback_to_main_ms: f64,
    pub key_down_to_recording_ms: f64,
    pub key_down_to_panel_ms: f64,
    pub key_down_to_first_stream_chunk_ms: Option<f64>,
    pub key_down_to_first_partial_ms: Option<f64>,
    pub key_down_to_first_preview_ms: Option<f64>,
    pub key_up_to_main_ms: f64,
    pub key_up_to_capture_end_ms: f64,
    pub wav_encode_ms: f64,
    pub asr_ms: f64,
    pub polish_ms: f64,
    pub insertion_post_ms: f64,
    pub key_up_to_insert_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct E2EProfileReport {
    pub log_path: PathBuf,
    pub samples: usize,
    pub successful_samples: usize,
    pub median: E2EMedian,
    pub latest: E2EProfileRecord,
}

pub fn default_e2e_log_path() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join("Library/Logs/Phonon/e2e.jsonl"))
}

pub fn profile_e2e(path: &Path, last: usize) -> Result<E2EProfileReport> {
    let input = std::fs::read_to_string(path).with_context(|| {
        format!(
            "read E2E profile log {}; record a dictation first",
            path.display()
        )
    })?;
    let mut records = Vec::new();
    for (index, line) in input
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        records.push(
            serde_json::from_str::<E2EProfileRecord>(line)
                .with_context(|| format!("parse E2E profile line {}", index + 1))?,
        );
    }
    if records.is_empty() {
        bail!("E2E profile log contains no samples");
    }
    let take = last.max(1).min(records.len());
    let selected = &records[records.len() - take..];
    let successful: Vec<&E2EProfileRecord> = selected
        .iter()
        .filter(|record| record.insertion_succeeded)
        .collect();
    if successful.is_empty() {
        bail!("selected E2E traces contain no successful insertions");
    }
    let pick = |field: fn(&E2EProfileRecord) -> f64| {
        median(successful.iter().map(|record| field(record)).collect())
    };
    Ok(E2EProfileReport {
        log_path: path.to_path_buf(),
        samples: selected.len(),
        successful_samples: successful.len(),
        median: E2EMedian {
            key_event_to_callback_ms: pick(|record| record.key_event_to_callback_ms),
            callback_to_main_ms: pick(|record| record.callback_to_main_ms),
            key_down_to_recording_ms: pick(|record| record.key_down_to_recording_ms),
            key_down_to_panel_ms: pick(|record| record.key_down_to_panel_ms),
            key_down_to_first_stream_chunk_ms: optional_median(
                successful
                    .iter()
                    .filter_map(|record| record.key_down_to_first_stream_chunk_ms)
                    .collect(),
            ),
            key_down_to_first_partial_ms: optional_median(
                successful
                    .iter()
                    .filter_map(|record| record.key_down_to_first_partial_ms)
                    .collect(),
            ),
            key_down_to_first_preview_ms: optional_median(
                successful
                    .iter()
                    .filter_map(|record| record.key_down_to_first_preview_ms)
                    .collect(),
            ),
            key_up_to_main_ms: pick(|record| record.key_up_to_main_ms),
            key_up_to_capture_end_ms: pick(|record| record.key_up_to_capture_end_ms),
            wav_encode_ms: pick(|record| record.wav_encode_ms),
            asr_ms: pick(|record| record.asr_ms),
            polish_ms: pick(|record| record.polish_ms),
            insertion_post_ms: pick(|record| record.insertion_post_ms),
            key_up_to_insert_ms: pick(|record| record.key_up_to_insert_ms),
        },
        latest: selected.last().expect("selected is non-empty").clone(),
    })
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    values[values.len() / 2]
}

fn optional_median(values: Vec<f64>) -> Option<f64> {
    (!values.is_empty()).then(|| median(values))
}

pub fn profile_llm(root: &Path, config: &LlmProfileConfig) -> Result<LlmProfileReport> {
    let mut rows = Vec::new();
    let mut modes = Vec::new();
    if config.include_plain {
        modes.push(("fluid-1", false));
    }
    if config.include_mtp && fluid_drafter_dir().is_dir() {
        modes.push(("fluid-1+mtp", true));
    }

    for (mode, use_mtp) in modes {
        for &input_words in &config.input_words {
            let input = profile_input(input_words);
            let mut serve = ServeJson::spawn(root, use_mtp)?;
            let (warmup, _) = serve.warmup()?;
            // Pay first-request graph/shape specialization before collecting
            // any profile samples. Application readiness follows the same rule.
            let (demo_wall, demo_response) = serve.run_text(&input)?;
            let _ = timing_from_resp(demo_wall, &demo_response)?;

            let mut steady = Vec::new();
            for _ in 0..config.steady_iterations.max(1) {
                let (wall, response) = serve.run_text(&input)?;
                steady.push(sample(&timing_from_resp(wall, &response)?));
            }
            serve.shutdown();
            rows.push(ProfileRow {
                mode: mode.into(),
                input_words,
                model_load_ms: warmup.as_secs_f64() * 1000.0,
                startup_demo_ms: demo_wall.as_secs_f64() * 1000.0,
                warm_median: median_sample(&steady),
            });
        }
    }
    Ok(LlmProfileReport { rows })
}

fn profile_input(words: usize) -> String {
    const VOCABULARY: &[&str] = &[
        "please",
        "clean",
        "this",
        "spoken",
        "developer",
        "dictation",
        "while",
        "keeping",
        "every",
        "technical",
        "detail",
        "and",
        "the",
        "original",
        "intent",
        "intact",
    ];
    (0..words)
        .map(|index| VOCABULARY[index % VOCABULARY.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

fn sample(timing: &RunTiming) -> ProfileSample {
    let stage = timing.stage_timings.as_ref();
    ProfileSample {
        wall_ms: timing.wall_ms,
        ttft_ms: timing.ttft_ms,
        decode_ms: timing.decode_ms,
        generated_tokens: timing.tokens,
        tokens_per_second: timing.tok_s,
        prompt_tokens: integer(stage, "mlx_prompt_tokens"),
        prefix_tokens: integer(stage, "mlx_prefix_tokens"),
        prefill_tokens: integer(stage, "mlx_prefill_tokens"),
        prefix_cache_hit: integer(stage, "mlx_prefix_cache_hit").map(|value| value != 0),
        mtp_accept_percent: number(stage, "mtp_accept_percent"),
    }
}

fn median_sample(samples: &[ProfileSample]) -> ProfileSample {
    let middle = samples.len() / 2;
    let pick = |mut values: Vec<f64>| -> f64 {
        values.sort_by(|a, b| a.total_cmp(b));
        values[middle]
    };
    let representative = &samples[middle];
    ProfileSample {
        wall_ms: pick(samples.iter().map(|x| x.wall_ms).collect()),
        ttft_ms: pick(samples.iter().map(|x| x.ttft_ms).collect()),
        decode_ms: pick(samples.iter().map(|x| x.decode_ms).collect()),
        generated_tokens: pick(samples.iter().map(|x| x.generated_tokens as f64).collect()) as u64,
        tokens_per_second: pick(samples.iter().map(|x| x.tokens_per_second).collect()),
        prompt_tokens: representative.prompt_tokens,
        prefix_tokens: representative.prefix_tokens,
        prefill_tokens: representative.prefill_tokens,
        prefix_cache_hit: representative.prefix_cache_hit,
        mtp_accept_percent: representative.mtp_accept_percent,
    }
}

fn integer(stage: Option<&Value>, key: &str) -> Option<u64> {
    stage?.get(key)?.as_u64()
}

fn number(stage: Option<&Value>, key: &str) -> Option<f64> {
    stage?.get(key)?.as_f64()
}

#[cfg(test)]
mod tests {
    use super::{parse_kernel_profile, profile_e2e, profile_input};
    use std::fs;

    #[test]
    fn generated_inputs_have_requested_word_count() {
        for words in [1, 8, 32, 128] {
            assert_eq!(profile_input(words).split_whitespace().count(), words);
        }
    }

    #[test]
    fn e2e_report_keeps_dictation_time_out_of_release_latency() {
        let path = std::env::temp_dir().join(format!(
            "phonon-e2e-test-{}-{}.jsonl",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let line = r#"{"timestamp":"2026-07-20T00:00:00Z","pass_id":"p1","source":"right-option","text_characters":12,"insertion_succeeded":true,"key_event_to_callback_ms":0.1,"callback_to_main_ms":0.2,"key_down_to_recording_ms":2.0,"key_down_to_panel_ms":1.0,"dictation_ms":5000.0,"key_up_to_main_ms":0.3,"key_up_to_capture_end_ms":1.0,"wav_encode_ms":3.0,"asr_ms":80.0,"polish_ms":100.0,"insertion_post_ms":1.0,"key_up_to_insert_ms":185.0,"key_down_to_insert_ms":5185.0,"asr_reported_ms":75.0,"llm_reported_wall_ms":95.0,"llm_ttft_ms":35.0,"llm_tokens_per_second":250.0}"#;
        fs::write(&path, format!("{line}\n")).unwrap();
        let report = profile_e2e(&path, 20).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(report.median.key_up_to_insert_ms, 185.0);
        assert_eq!(report.latest.dictation_ms, 5000.0);
        assert_eq!(report.successful_samples, 1);
    }

    #[test]
    fn literal_kernels_are_ranked_by_aggregate_encode_time() {
        let report = parse_kernel_profile(
            "PHONON_KERNEL\tsteel_gemm\t4000\nPHONON_KERNEL\trms\t9000\nPHONON_KERNEL\tsteel_gemm\t7000\n",
            10,
        )
        .unwrap();
        assert_eq!(report.total_invocations, 3);
        assert_eq!(report.unique_kernels, 2);
        assert_eq!(report.kernels[0].kernel, "steel_gemm");
        assert_eq!(report.kernels[0].invocations, 2);
        assert_eq!(report.kernels[0].cpu_encode_ms, 0.011);
    }
}
