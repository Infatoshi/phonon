//! Bench all three weight streams + end-to-end polish latency.
//!
//! Streams:
//!   1. asr      — Parakeet load + optional wav transcribe
//!   2. fluid-1  — target LLM cold/warm TTFT/ITL (no MTP)
//!   3. mtp      — same path with Gemma MTP speculative head (TTFT/ITL + accept)

use anyhow::{bail, Context, Result};
use phonon_core::project_root;
use phonon_llm::{
    cold_run, fluid1_available, fluid1_paths, fluid_drafter_dir, fluid_helper, fluid_model_dir,
    timing_from_resp, RunTiming, ServeJson,
};
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const DEFAULT_CLIPS: &[&str] = &[
    "numba cuda use nvidia binding for tdt decoding on the three oh nine oh",
    "open the qwen three point five four b four bit model with m l x",
    "check nvidia smi on the blackwell server",
    "fix the typo in train polish lora and re run the eval",
    "push the branch and open a draft p r against main",
];

pub struct BenchArgs {
    pub iters: usize,
    pub text: Option<String>,
    pub no_mtp: bool,
    pub skip_cold: bool,
    pub json: bool,
    /// Optional wav for ASR stream timing. If missing, only measures ASR load.
    pub wav: Option<PathBuf>,
}

fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

pub fn run_bench(args: BenchArgs) -> Result<()> {
    let root = project_root();
    let clips: Vec<String> = if let Some(t) = args.text {
        vec![t]
    } else {
        DEFAULT_CLIPS.iter().map(|s| (*s).to_string()).collect()
    };
    let iters = args.iters.max(1);

    if args.json {
        return run_bench_json(
            &root,
            &clips,
            iters,
            !args.no_mtp,
            args.skip_cold,
            args.wav.as_deref(),
        );
    }

    println!("phonon bench — three weight streams");
    println!("root={}", root.display());
    println!();

    // ─── 1. ASR ───────────────────────────────────────────────
    println!("== stream 1: asr (parakeet-tdt-0.6b-v2) ==");
    match bench_asr_full(&root, args.wav.as_deref()) {
        Ok(a) => {
            println!("  load_ms={:.0}  model={}", a.load_ms, a.model);
            if let Some(t) = a.transcribe_ms {
                println!(
                    "  transcribe_ms={t:.0}  out={:?}",
                    a.transcript
                        .as_deref()
                        .unwrap_or("")
                        .chars()
                        .take(60)
                        .collect::<String>()
                );
            } else {
                println!("  transcribe: skipped (pass --wav path.wav to measure)");
            }
        }
        Err(e) => println!("  FAIL: {e:#}"),
    }
    println!();

    if !fluid1_available() {
        println!("== stream 2/3: fluid-1 + mtp ==");
        println!(
            "  SKIP — fluid-1 missing\n  helper: {}\n  model:  {}",
            fluid_helper().display(),
            fluid_model_dir().display()
        );
        return Ok(());
    }

    // ─── 2. fluid-1 without MTP ───────────────────────────────
    println!("== stream 2: fluid-1 (target LLM, no MTP) ==");
    if !args.skip_cold {
        let cold = cold_run(&root, &clips[0], false)?;
        let lat = cold.latency_ms.unwrap_or(cold.wall_ms);
        let ttft = cold.ttft_ms.unwrap_or(0.0);
        let toks = cold.generated_tokens.unwrap_or(0);
        let decode = (lat - ttft).max(0.0);
        let itl = if toks > 1 {
            decode / (toks as f64 - 1.0)
        } else {
            0.0
        };
        println!(
            "  cold run: wall={:.0}ms  ttft={:.0}ms  decode={:.0}ms  itl={:.1}ms  tokens={}  tok/s={:.0}",
            cold.wall_ms,
            ttft,
            decode,
            itl,
            toks,
            cold.tokens_per_second.unwrap_or(0.0)
        );
        println!("  out: {}", cold.output);
    }
    {
        let mut serve = ServeJson::spawn_fluid(&root, false)?;
        let (wu, _) = serve.warmup()?;
        println!(
            "  warmup_ms={:.0}  (fluid-1 weights only)",
            wu.as_secs_f64() * 1000.0
        );
        let mut timings = Vec::new();
        for i in 0..iters {
            let clip = &clips[i % clips.len()];
            let (wall, resp) = serve.run_text(clip)?;
            let t = timing_from_resp(wall, &resp)?;
            println!(
                "  iter{i}: wall={:.0}ms  ttft={:.0}ms  itl={:.1}ms  tok/s={:.0}  tokens={}",
                t.wall_ms, t.ttft_ms, t.itl_ms, t.tok_s, t.tokens
            );
            timings.push(t);
        }
        serve.shutdown();
        print_steady_summary("fluid-1", &timings);
    }
    println!();

    // ─── 3. MTP speculative head ──────────────────────────────
    println!("== stream 3: mtp (gemma-4-E2B speculative decoder head) ==");
    let has_drafter = fluid1_paths().map(|(_, _, d)| d.is_some()).unwrap_or(false);
    if args.no_mtp {
        println!("  SKIP — --no-mtp");
    } else if !has_drafter {
        println!(
            "  SKIP — drafter missing at {}",
            fluid_drafter_dir().display()
        );
    } else {
        if !args.skip_cold {
            let cold = cold_run(&root, &clips[0], true)?;
            let lat = cold.latency_ms.unwrap_or(cold.wall_ms);
            let ttft = cold.ttft_ms.unwrap_or(0.0);
            let toks = cold.generated_tokens.unwrap_or(0);
            let decode = (lat - ttft).max(0.0);
            let itl = if toks > 1 {
                decode / (toks as f64 - 1.0)
            } else {
                0.0
            };
            println!(
                "  cold run (+MTP load): wall={:.0}ms  ttft={:.0}ms  decode={:.0}ms  itl={:.1}ms  tokens={}  tok/s={:.0}",
                cold.wall_ms,
                ttft,
                decode,
                itl,
                toks,
                cold.tokens_per_second.unwrap_or(0.0)
            );
        }
        let mut serve = ServeJson::spawn_fluid(&root, true)?;
        let (wu, wu_resp) = serve.warmup()?;
        let msg = wu_resp
            .status
            .as_ref()
            .and_then(|s| s.message.as_deref())
            .unwrap_or("");
        println!("  warmup_ms={:.0}  msg={msg}", wu.as_secs_f64() * 1000.0);
        let mut timings = Vec::new();
        for i in 0..iters {
            let clip = &clips[i % clips.len()];
            let (wall, resp) = serve.run_text(clip)?;
            let t = timing_from_resp(wall, &resp)?;
            let mtp_note = t
                .stage_timings
                .as_ref()
                .map(|s| {
                    let acc = s.get("mtp_accept_percent").and_then(|v| v.as_f64());
                    let en = s.get("mtp_enabled").and_then(|v| v.as_f64());
                    format!(
                        "  mtp_enabled={} accept={}%",
                        en.unwrap_or(0.0) as i64,
                        acc.unwrap_or(0.0) as i64
                    )
                })
                .unwrap_or_default();
            println!(
                "  iter{i}: wall={:.0}ms  ttft={:.0}ms  itl={:.1}ms  tok/s={:.0}  tokens={}{mtp_note}",
                t.wall_ms, t.ttft_ms, t.itl_ms, t.tok_s, t.tokens
            );
            if i == 0 {
                if let Some(st) = &t.stage_timings {
                    println!("         stageTimings: {st}");
                }
            }
            timings.push(t);
        }
        serve.shutdown();
        print_steady_summary("fluid-1+mtp", &timings);
    }

    println!();
    println!("== e2e picture (dictation path) ==");
    println!("  boot: asr load ∥ fluid-1 load ∥ mtp load  (parallel in native app)");
    println!("  request: record → asr.transcribe → auto polish (warm fluid-1+mtp)");
    println!("  polish always-on; no manual step");

    Ok(())
}

fn print_steady_summary(label: &str, timings: &[RunTiming]) {
    let steady: &[RunTiming] = if timings.len() >= 3 {
        &timings[1..]
    } else {
        timings
    };
    let mut walls: Vec<f64> = steady.iter().map(|t| t.wall_ms).collect();
    let mut ttfts: Vec<f64> = steady.iter().map(|t| t.ttft_ms).collect();
    let mut itls: Vec<f64> = steady.iter().map(|t| t.itl_ms).collect();
    let mut tps: Vec<f64> = steady.iter().map(|t| t.tok_s).collect();
    let mut decodes: Vec<f64> = steady.iter().map(|t| t.decode_ms).collect();

    let med_wall = median(&mut walls);
    let med_ttft = median(&mut ttfts);
    let med_itl = median(&mut itls);
    let med_tps = median(&mut tps);
    let med_decode = median(&mut decodes);
    let total = med_wall.max(1.0);

    println!(
        "  [{label} steady n={}] wall_med={:.0}ms  ttft_med={:.0}ms  itl_med={:.1}ms  tok/s_med={:.0}",
        steady.len(),
        med_wall,
        med_ttft,
        med_itl,
        med_tps
    );
    println!(
        "  breakdown: ttft={:.0}%  decode={:.0}%  (mean wall={:.0}ms)",
        100.0 * med_ttft / total,
        100.0 * med_decode / total,
        mean(&walls)
    );
}

struct AsrBench {
    load_ms: f64,
    model: String,
    transcribe_ms: Option<f64>,
    transcript: Option<String>,
}

fn bench_asr_full(root: &Path, wav: Option<&Path>) -> Result<AsrBench> {
    let script = root.join("sidecar/asr_server.py");
    if !script.is_file() {
        bail!("missing {}", script.display());
    }
    let uv = crate::resolve_runtime_tool("uv").context("uv not found")?;
    let mut child = Command::new(uv)
        .args([
            "run",
            "--with",
            "parakeet-mlx",
            "python",
            script.to_str().unwrap(),
        ])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("spawn ASR")?;

    let mut stdout = BufReader::new(child.stdout.take().context("stdout")?);
    let mut stdin = child.stdin.take().context("stdin")?;
    let t0 = Instant::now();
    let mut model = String::from("parakeet");
    let mut load_ms = 0.0;
    let mut ready = false;
    let mut line = String::new();

    while !ready {
        line.clear();
        if stdout.read_line(&mut line)? == 0 {
            bail!("ASR EOF before ready");
        }
        let v: serde_json::Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        match v.get("type").and_then(|x| x.as_str()).unwrap_or("") {
            "ready" => {
                load_ms = t0.elapsed().as_secs_f64() * 1000.0;
                if let Some(m) = v.get("model").and_then(|x| x.as_str()) {
                    model = m.to_string();
                }
                ready = true;
            }
            "error" => {
                bail!(
                    "{}",
                    v.get("msg").and_then(|x| x.as_str()).unwrap_or("asr error")
                );
            }
            _ => {
                if t0.elapsed() > Duration::from_secs(180) {
                    bail!("ASR load timeout");
                }
            }
        }
    }

    let mut transcribe_ms = None;
    let mut transcript = None;
    if let Some(wav) = wav {
        if !wav.is_file() {
            bail!("wav not found: {}", wav.display());
        }
        let t1 = Instant::now();
        writeln!(
            stdin,
            "{}",
            json!({"cmd": "transcribe", "path": wav.to_str().unwrap()})
        )?;
        stdin.flush()?;
        loop {
            line.clear();
            if stdout.read_line(&mut line)? == 0 {
                bail!("ASR EOF waiting result");
            }
            let v: serde_json::Value = match serde_json::from_str(line.trim()) {
                Ok(v) => v,
                Err(_) => continue,
            };
            match v.get("type").and_then(|x| x.as_str()).unwrap_or("") {
                "result" => {
                    transcribe_ms = Some(
                        v.get("seconds")
                            .and_then(|x| x.as_f64())
                            .unwrap_or_else(|| t1.elapsed().as_secs_f64())
                            * 1000.0,
                    );
                    // sidecar reports seconds already; use that as ms
                    if let Some(s) = v.get("seconds").and_then(|x| x.as_f64()) {
                        transcribe_ms = Some(s * 1000.0);
                    }
                    transcript = v
                        .get("text")
                        .and_then(|x| x.as_str())
                        .map(|s| s.to_string());
                    break;
                }
                "error" => {
                    bail!(
                        "transcribe: {}",
                        v.get("msg").and_then(|x| x.as_str()).unwrap_or("?")
                    );
                }
                _ => {
                    if t1.elapsed() > Duration::from_secs(120) {
                        bail!("transcribe timeout");
                    }
                }
            }
        }
    }

    let _ = writeln!(stdin, "{}", json!({"cmd": "shutdown"}));
    let _ = child.kill();
    let _ = child.wait();

    Ok(AsrBench {
        load_ms,
        model,
        transcribe_ms,
        transcript,
    })
}

fn run_bench_json(
    root: &Path,
    clips: &[String],
    iters: usize,
    use_mtp: bool,
    skip_cold: bool,
    wav: Option<&Path>,
) -> Result<()> {
    let mut out = serde_json::Map::new();

    match bench_asr_full(root, wav) {
        Ok(a) => {
            out.insert(
                "asr".into(),
                json!({
                    "load_ms": a.load_ms,
                    "model": a.model,
                    "transcribe_ms": a.transcribe_ms,
                    "transcript": a.transcript,
                }),
            );
        }
        Err(e) => {
            out.insert("asr".into(), json!({"error": format!("{e:#}")}));
        }
    }

    if fluid1_available() {
        if !skip_cold {
            let cold = cold_run(root, &clips[0], false)?;
            out.insert(
                "fluid1_cold".into(),
                json!({
                    "wall_ms": cold.wall_ms,
                    "latency_ms": cold.latency_ms,
                    "ttft_ms": cold.ttft_ms,
                    "tokens_per_second": cold.tokens_per_second,
                    "generated_tokens": cold.generated_tokens,
                }),
            );
        }
        let mut serve = ServeJson::spawn_fluid(root, false)?;
        let (wu, _) = serve.warmup()?;
        let mut runs = Vec::new();
        for i in 0..iters {
            let (wall, resp) = serve.run_text(&clips[i % clips.len()])?;
            let t = timing_from_resp(wall, &resp)?;
            runs.push(json!({
                "wall_ms": t.wall_ms,
                "ttft_ms": t.ttft_ms,
                "itl_ms": t.itl_ms,
                "tok_s": t.tok_s,
                "tokens": t.tokens,
            }));
        }
        serve.shutdown();
        out.insert(
            "fluid1_warm".into(),
            json!({
                "warmup_ms": wu.as_secs_f64() * 1000.0,
                "runs": runs,
            }),
        );

        if use_mtp && fluid_drafter_dir().is_dir() {
            if !skip_cold {
                let cold = cold_run(root, &clips[0], true)?;
                out.insert(
                    "mtp_cold".into(),
                    json!({
                        "wall_ms": cold.wall_ms,
                        "ttft_ms": cold.ttft_ms,
                        "tokens_per_second": cold.tokens_per_second,
                    }),
                );
            }
            let mut serve = ServeJson::spawn_fluid(root, true)?;
            let (wu, _) = serve.warmup()?;
            let mut runs = Vec::new();
            for i in 0..iters {
                let (wall, resp) = serve.run_text(&clips[i % clips.len()])?;
                let t = timing_from_resp(wall, &resp)?;
                runs.push(json!({
                    "wall_ms": t.wall_ms,
                    "ttft_ms": t.ttft_ms,
                    "itl_ms": t.itl_ms,
                    "tok_s": t.tok_s,
                    "tokens": t.tokens,
                    "stage_timings": t.stage_timings,
                }));
            }
            serve.shutdown();
            out.insert(
                "mtp_warm".into(),
                json!({
                    "warmup_ms": wu.as_secs_f64() * 1000.0,
                    "runs": runs,
                }),
            );
        }
    }

    println!("{}", serde_json::Value::Object(out));
    Ok(())
}
