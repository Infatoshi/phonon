//! Transcript correction through llama.cpp, with Phonon's public refiner.
//!
//! The refiner (fine-tuned LFM3-1.7B, Q4_K_S GGUF) is the model the iPhone and
//! Android builds run, and the one macOS falls back to. Windows and Linux run it
//! inside a resident `llama-server`, with the prompt built exactly as
//! `ios/App/Refiner.swift` and `android/.../refiner_jni.cpp` build it: the refiner
//! system prompt plus the writing-mode line, the bare transcript as the user
//! turn, the chat template with thinking closed, greedy decoding without a
//! repetition penalty, at most 1.5 x the transcript's tokens + 32, and the
//! length guard that keeps the raw transcript when the answer runs away.
//!
//! Correction is mandatory. A recogniser result that never reaches this stage is
//! a failure, not a degraded mode.

#[cfg(not(target_os = "linux"))]
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};

use crate::fetch;
use crate::http;
use crate::manifest;

/// The refiner's system prompt (sidecar/polish_server.py, ios/App/Refiner.swift).
pub const SYSTEM_PROMPT: &str = "Rewrite the raw dictation transcript into the exact text the speaker intended. \
Fix recognition errors and technical terms, keep the speaker's wording and casing style, do not add or drop content.";

/// `writing_modes` from the refiner's `phonon_refiner.json` at the pinned revision.
pub const MODES: [(&str, &str); 4] = [
    ("minimal", "Writing mode: minimal (notes and documents)."),
    ("casual", "Writing mode: casual (a chat message)."),
    ("formal", "Writing mode: formal (an email or document)."),
    (
        "prompt",
        "Writing mode: prompt (an instruction to an AI assistant).",
    ),
];

/// The mode a dictation uses until the user picks another from the tray
/// (`crate::mode`), and the one the startup check, the warm-up and the speed probe
/// always use. Nothing tells Windows and Linux yet which app has focus.
pub const DEFAULT_MODE: &str = "minimal";

/// The model may take this long to load before the server is called dead.
const READY_TIMEOUT: Duration = Duration::from_secs(600);

/// Every GPU try of one runtime together (listing, start, warm-up, speed probe)
/// gets this long; then the next path in the fallback order starts. A GPU server
/// on a 3090 is ready in a few seconds; a hung driver must not hold up the CPU
/// fallback for `READY_TIMEOUT`.
const GPU_BUDGET: Duration = Duration::from_secs(120);

/// One GPU listing, warm-up or speed probe request may take at most this long.
const GPU_STEP: Duration = Duration::from_secs(30);

/// The warm-up and the corrections themselves (CPU or a GPU that passed its
/// checks) may take this long.
const CORRECTION_TIMEOUT: Duration = Duration::from_secs(300);

/// What is left of `deadline`, at most `cap`.
fn left(deadline: Instant, cap: Duration) -> Duration {
    deadline.saturating_duration_since(Instant::now()).min(cap)
}

/// The mode line for a mode name; unknown names fall back to minimal.
pub fn mode_line(mode: &str) -> &'static str {
    MODES
        .iter()
        .find(|(name, _)| *name == mode)
        .unwrap_or(&MODES[0])
        .1
}

/// The whole prompt for one transcript, as the iPhone builds it.
pub fn prompt(raw: &str, mode: &str) -> String {
    format!(
        "<|startoftext|><|im_start|>system\n{SYSTEM_PROMPT}\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n<think></think>",
        mode_line(mode),
        raw.trim()
    )
}

/// How many tokens the refiner may generate: 1.5 x the transcript's + 32.
pub fn output_budget(raw_tokens: u32) -> u32 {
    (1.5 * raw_tokens as f64) as u32 + 32
}

/// The length guard: an empty answer, or one with more than 1.2 x the
/// transcript's words + 20, is a runaway; the raw transcript is kept instead.
pub fn guarded(raw: &str, answer: &str) -> (String, bool) {
    let answer = answer.trim();
    let raw_words = raw.split_whitespace().count() as f64;
    let fired =
        answer.is_empty() || answer.split_whitespace().count() as f64 > 1.2 * raw_words + 20.0;
    if fired {
        (raw.trim().to_string(), true)
    } else {
        (answer.to_string(), false)
    }
}

/// How many threads llama.cpp should use: about one per physical core, never
/// every logical CPU. Decoding is memory-bound, and ggml's threads wait on each
/// other every layer, so one thread that the scheduler parks (an SMT sibling, a
/// browser, a VM) stalls them all. On a 16-core 9950X3D with another 8-thread
/// load running, 31 threads decoded 4.9-21.8 tokens/s and 16 or 8 threads
/// 35-37 (DEVLOG 2026-11-05). Small machines keep one CPU free for the tray,
/// the hook, and whatever the user is typing into.
pub fn thread_count(logical_cpus: usize) -> usize {
    if logical_cpus <= 4 {
        return logical_cpus.saturating_sub(1).max(1);
    }
    (logical_cpus / 2).min(16)
}

/// A free loopback port. The server binds it a moment later; nothing else on the
/// machine is looking for a port in that instant.
fn free_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").context("reserve a local port")?;
    Ok(listener.local_addr()?.port())
}

/// The generated text in one `/completion` reply.
pub fn parse_reply(value: &serde_json::Value) -> Result<String> {
    value
        .get("content")
        .and_then(|c| c.as_str())
        .map(str::to_string)
        .ok_or_else(|| anyhow!("correction reply has no content: {value}"))
}

/// What the correction server computes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// llama.cpp's CPU backend; no layer or op leaves the CPU.
    Cpu,
    /// A GPU, every layer offloaded: a tuned CUDA runtime from the backend
    /// manifest, or (Windows) a discrete GPU through Vulkan.
    Gpu,
}

/// A CPU at or above this busy fraction is "saturated": ggml's threads meet at a
/// barrier on every graph node, and on Windows a preempted thread stalls the rest
/// for a scheduler quantum. Bare-metal 9950X3D with a busy loop on every CPU:
/// 8 threads 0.6-2.3 t/s prefill and no 32-token decode within 120 s, 1 thread
/// 29 / 5.7 t/s; QA VM (8 vCPU): 4 threads 21.9 / 1.7, 1 thread 96.6 / 12.7
/// (DEVLOG 2026-11-07 9:50 pm and 10:30 pm).
pub const SATURATED: f64 = 0.85;

/// Threads of the fallback corrector used while the CPU is saturated. Windows 11
/// VM, 8 vCPU, a Normal-priority busy loop on every vCPU, llama-bench pp16+tg32 at
/// AboveNormal, two rounds (2026-11-10): 1 thread 16.0 / 15.6 t/s, 2 threads
/// 17.2 / 17.1, 3 threads 16.7 / 3.5 (one round stalled). Two looked stable on
/// that one VM, but bare metal was never measured, so it stays at one until a
/// bare-metal comparison.
const LEAN_THREADS: usize = 1;

/// Threads of the fallback corrector on a machine with `logical_cpus` logical
/// CPUs: `LEAN_THREADS`, never more than the main server gets.
fn lean_threads(logical_cpus: usize) -> usize {
    LEAN_THREADS.min(thread_count(logical_cpus))
}

/// PCI vendor ids of GPU makers that ship a Vulkan driver on Windows: NVIDIA,
/// AMD, Intel. A machine with none of them (a VM's virtual adapter, Microsoft
/// Basic Display) has no device `--list-devices` could offer, and the listing is
/// not free: cold, with real-time scanning on and every vCPU busy, it hit its
/// 20 s timeout (W11, Windows 11 VM, 2026-11-10). Integrated GPUs keep the
/// listing: registry memory sizes cannot tell them from discrete ones (DEVLOG
/// 2026-11-07), and `MIN_GPU_DECODE` still decides.
const GPU_VENDORS: &[&str] = &["VEN_10DE", "VEN_1002", "VEN_8086"];

/// Whether any display adapter id ("PCI\VEN_10DE&DEV_2204&...") is from a GPU vendor.
#[cfg(any(windows, test))]
fn any_gpu_vendor(adapter_ids: &[String]) -> bool {
    adapter_ids.iter().any(|id| {
        let id = id.to_ascii_uppercase();
        id.strip_prefix("PCI\\")
            .is_some_and(|rest| GPU_VENDORS.iter().any(|vendor| rest.starts_with(vendor)))
    })
}

/// The PnP ids of the display adapters Windows knows (EnumDisplayDevicesW).
#[cfg(windows)]
pub fn display_adapter_ids() -> Vec<String> {
    use windows_sys::Win32::Graphics::Gdi::{EnumDisplayDevicesW, DISPLAY_DEVICEW};
    let mut ids = Vec::new();
    for index in 0..32 {
        // SAFETY: a zeroed DISPLAY_DEVICEW with `cb` set is what the call expects.
        let mut device: DISPLAY_DEVICEW = unsafe { std::mem::zeroed() };
        device.cb = std::mem::size_of::<DISPLAY_DEVICEW>() as u32;
        if unsafe { EnumDisplayDevicesW(std::ptr::null(), index, &mut device, 0) } == 0 {
            break;
        }
        let end = device.DeviceID.iter().position(|&c| c == 0).unwrap_or(128);
        ids.push(String::from_utf16_lossy(&device.DeviceID[..end]));
    }
    ids
}

/// The `VulkanN` device to offload to: the first one `llama-server --list-devices`
/// prints ("  Vulkan0: NVIDIA GeForce RTX 3090 (24322 MiB, 23554 MiB free)").
/// llama.cpp leaves integrated GPUs out of that list when a discrete one exists
/// (an RTX 3090 + Ryzen 9950X3D desktop lists the 3090, not the Radeon iGPU); an iGPU-only
/// machine lists its iGPU, and `MIN_GPU_DECODE` decides whether it is worth using.
/// `kind` is the device prefix: "Vulkan", or "CUDA" for a tuned CUDA runtime
/// ("  CUDA0: NVIDIA GeForce RTX 3090 (24135 MiB, 23700 MiB free)").
pub fn pick_device(listing: &str, kind: &str) -> Option<String> {
    listing.lines().find_map(|line| {
        let (device, _) = line.trim().split_once(':')?;
        let index = device.strip_prefix(kind)?;
        index.parse::<u32>().ok()?;
        Some(device.to_string())
    })
}

/// A GPU that decodes slower than this stays unused: a CPU gives the refiner 15-57
/// tokens/s (laptop 4-core to 9950X3D), and an integrated GPU sharing the CPU's
/// memory was not measured. The 3090 on Vulkan decodes about 400 (b12132, 2026-11-07).
const MIN_GPU_DECODE: f64 = 60.0;

/// Decode speed of the server at `base`, from llama-server's own timings for a
/// 32-token greedy completion.
fn decode_rate(base: &str, timeout: Duration) -> Option<f64> {
    if timeout.is_zero() {
        return None;
    }
    let tokens = tokenize(
        base,
        &prompt(
            "one two three four five six seven eight nine ten",
            DEFAULT_MODE,
        ),
    )
    .ok()?;
    let value: serde_json::Value = http::agent()
        .post(&format!("{base}/completion"))
        .timeout(timeout)
        .send_json(serde_json::json!({
            "prompt": tokens, "n_predict": 32, "temperature": 0.0, "top_k": 1,
            "cache_prompt": false, "stream": false,
        }))
        .ok()?
        .into_json()
        .ok()?;
    value.get("timings")?.get("predicted_per_second")?.as_f64()
}

/// One running `llama-server`.
struct Server {
    child: Child,
    base: String,
    /// The kill-on-close job that holds the server (Windows), kept open for the
    /// life of the process.
    #[cfg(windows)]
    job: isize,
}

/// Put the server in a job that Windows kills when its last handle closes. Phonon
/// holds that handle until it exits, so a crash or an End Task from Task Manager
/// takes the server (about 1 GB) with it instead of leaving it running.
#[cfg(windows)]
fn kill_with_parent(child: &Child) -> isize {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            eprintln!("phonon: CreateJobObjectW failed; the correction server may outlive a crash");
            return 0;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let set = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if set == 0 || AssignProcessToJobObject(job, child.as_raw_handle() as _) == 0 {
            eprintln!("phonon: could not tie the correction server to Phonon's lifetime");
        }
        job as isize
    }
}

/// The server executable and the model file, checked to exist.
fn runtime_files() -> Result<(PathBuf, PathBuf)> {
    let server = fetch::sentinel_path(&manifest::LLAMA_RUNTIME);
    if !server.is_file() {
        bail!(
            "{} is missing; run the first-run download",
            server.display()
        );
    }
    let model: PathBuf = fetch::component_dir(&manifest::POLISH_MODEL).join(manifest::POLISH_GGUF);
    if !model.is_file() {
        bail!("{} is missing", model.display());
    }
    Ok((server, model))
}

/// A command for the bundled server that shows no console window (Windows) and
/// finds its shared libraries (Linux). `env` is set on the child only (a tuned
/// runtime pins its device with CUDA_VISIBLE_DEVICES). `workdir` is the CPU
/// runtime's directory for every server: ggml loads its backends from the
/// executable's directory and the working directory, so a tuned CUDA-only
/// runtime (no libggml-cpu-*.so of its own, which keeps its download near 15 MB)
/// takes its CPU variant from the CPU runtime every install already has.
fn server_command(server: &Path, env: &[(String, String)], workdir: &Path) -> Result<Command> {
    let mut command = Command::new(server);
    command.stdin(Stdio::null());
    command.envs(env.iter().map(|(k, v)| (k, v)));
    command.current_dir(workdir);
    #[cfg(windows)]
    {
        // Without CREATE_NO_WINDOW the server owns a console window that sits on
        // the desktop for the life of the tray app. Above normal like Phonon itself
        // (win/mod.rs); a child does not inherit that class.
        use std::os::windows::process::CommandExt;
        use windows_sys::Win32::System::Threading::ABOVE_NORMAL_PRIORITY_CLASS;
        command.creation_flags(crate::asr::CREATE_NO_WINDOW | ABOVE_NORMAL_PRIORITY_CLASS);
    }
    // every runtime (the CPU one, a tuned GPU one) keeps its libraries beside its server
    #[cfg(target_os = "linux")]
    if let Some(dir) = server.parent() {
        crate::linux::runtime_libraries(&mut command, dir)?;
    }
    Ok(command)
}

/// What the server prints for `--list-devices` (stdout and stderr, which carries
/// ggml's per-device lines). A machine without the driver lists no device; a
/// module that fails to load (wrong driver, missing library) lists none either.
fn device_listing(
    server: &Path,
    env: &[(String, String)],
    workdir: &Path,
    timeout: Duration,
) -> Option<String> {
    let mut command = server_command(server, env, workdir).ok()?;
    command
        .arg("--list-devices")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().ok()?;
    let started = Instant::now();
    // A driver can take seconds to initialise; a hung one must not hold up startup.
    while child.try_wait().ok()?.is_none() {
        if started.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            eprintln!("phonon: listing GPUs took over {timeout:?}; not using them");
            return None;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let output = child.wait_with_output().ok()?;
    Some(format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

/// Which GPU a runtime should offload to, and how to make it see that GPU.
struct Target<'a> {
    /// "Vulkan" or "CUDA": the device name prefix the server lists.
    kind: &'a str,
    /// Set on every child of this runtime (the tuned runtime's device pin).
    env: Vec<(String, String)>,
    /// For a pinned CUDA device: its name; the listing must hold that device alone.
    pinned: Option<&'a str>,
}

/// How long a server may take to become ready and to answer its warm-up.
#[derive(Debug, Clone, Copy)]
struct Limits {
    ready_by: Instant,
    warm_up: Duration,
}

impl Limits {
    /// The CPU server: the model may load slowly (real-time scanning, a busy CPU).
    fn cpu() -> Self {
        Self {
            ready_by: Instant::now() + READY_TIMEOUT,
            warm_up: CORRECTION_TIMEOUT,
        }
    }

    /// A GPU try that must finish by `deadline`.
    fn gpu(deadline: Instant) -> Self {
        Self {
            ready_by: deadline,
            warm_up: left(deadline, GPU_STEP),
        }
    }
}

/// A server on the target GPU of the runtime at `server`, if it lists it, starts,
/// and decodes fast enough, all by `deadline`. `label` names the runtime in the log.
#[allow(clippy::too_many_arguments)]
fn gpu_server(
    server: &Path,
    target: &Target,
    label: &str,
    model: &Path,
    threads: usize,
    workdir: &Path,
    deadline: Instant,
) -> Option<Server> {
    let kind = target.kind;
    let budget = left(deadline, GPU_STEP);
    if budget.is_zero() {
        eprintln!("phonon: no time left to try {label}");
        return None;
    }
    let listing = device_listing(server, &target.env, workdir, budget).unwrap_or_default();
    if let Some(name) = target.pinned {
        if !crate::backends::listing_matches(&listing, name) {
            eprintln!("phonon: {label} does not list the chosen {name} alone; not using it");
            return None;
        }
    }
    let Some(gpu) = pick_device(&listing, kind) else {
        eprintln!("phonon: {label} lists no {kind} device");
        return None;
    };
    let limits = Limits::gpu(deadline);
    match Server::spawn(
        server,
        model,
        threads,
        Some(&gpu),
        &target.env,
        workdir,
        limits,
    ) {
        // Twice: the first Vulkan run after an install compiles the GPU's shaders
        // (3090: 66 tokens/s cold, 300 warm).
        Ok(main) => match [
            decode_rate(&main.base, left(deadline, GPU_STEP)),
            decode_rate(&main.base, left(deadline, GPU_STEP)),
        ]
        .into_iter()
        .flatten()
        .reduce(f64::max)
        {
            Some(rate) if rate >= MIN_GPU_DECODE => {
                eprintln!("phonon: correction runs on {gpu} with {label} ({rate:.0} tokens/s)");
                Some(main)
            }
            rate => {
                eprintln!(
                    "phonon: {gpu} with {label} decodes {:.0} tokens/s; not using it",
                    rate.unwrap_or(0.0)
                );
                None
            }
        },
        Err(error) => {
            eprintln!("phonon: {gpu} with {label} failed ({error:#})");
            None
        }
    }
}

impl Server {
    /// The CPU runtime's server on the CPU, from its own directory.
    fn cpu(server: &Path, model: &Path, threads: usize) -> Result<Self> {
        let workdir = server.parent().unwrap_or(Path::new("."));
        Self::spawn(server, model, threads, None, &[], workdir, Limits::cpu())
    }

    /// Start a server and wait until the model answers its health check.
    fn spawn(
        server: &Path,
        model: &Path,
        threads: usize,
        gpu: Option<&str>,
        env: &[(String, String)],
        workdir: &Path,
        limits: Limits,
    ) -> Result<Self> {
        let port = free_port()?;
        let mut command = server_command(server, env, workdir)?;
        command
            .arg("-m")
            .arg(model)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("-c")
            .arg("2048")
            .arg("-t")
            .arg(threads.to_string())
            // No repacked second copy of the weights: it held 1.14 GB of private
            // memory next to the 943 MB mapped model (win-beta.1 report: 2.7 GB,
            // laptop slowed; reproduced 2.58 GB peak on the clean VM, 2026-11-07).
            // Measured on a 9950X3D, 8 threads, b12132: decode 52.6 vs 52.5 t/s,
            // 16-token prefill 480 vs 548 t/s (about 4 ms per dictation), 64-token
            // prefill 1072 vs 704 t/s.
            .arg("--no-repack")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        match gpu {
            Some(device) => {
                command.arg("--device").arg(device).arg("-ngl").arg("99");
            }
            // Nothing leaves the CPU: no layers, and no ops offloaded for big batches.
            None => {
                command.arg("--device").arg("none").arg("-ngl").arg("0");
            }
        }
        #[cfg(target_os = "linux")]
        let log_path = {
            // No host-memory prompt cache (default up to 8 GiB): one client and one
            // prompt prefix, which the slot's own KV cache already reuses. The server's
            // private memory grew 125 -> 174 MB over ten dictations without this
            // (Ubuntu QA VM, 2026-11-10). Linux only: Windows keeps its server as it was.
            command.arg("--cache-ram").arg("0");
            // A pipe that nobody drains fills during model loading on Linux.
            // Keep diagnostics on disk so startup and later requests cannot block.
            let logs = crate::paths::data_root().join("logs");
            std::fs::create_dir_all(&logs)?;
            let path = logs.join(format!("llama-server-{port}.log"));
            let log = std::fs::File::create(&path).context("create the correction server log")?;
            command.stdout(Stdio::null()).stderr(log);
            // Kill the server when Phonon dies, however it dies: a `pkill phonon`
            // left the old server running with 2 GB (QA VM, 2026-11-07). The signal
            // follows the spawning thread, which is the main thread here. SIGKILL, not
            // SIGTERM: llama-server's own SIGTERM shutdown can hang, and one did after a
            // `pkill phonon` (main thread inside the handler, SIGTERM blocked, 1.1 GB
            // left for 13 minutes; QA VM, 2026-11-10). The server keeps no state to save.
            use std::os::unix::process::CommandExt;
            unsafe {
                command.pre_exec(|| {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            path
        };
        let mut child = command
            .spawn()
            .with_context(|| format!("start {}", server.display()))?;
        #[cfg(windows)]
        let job = kill_with_parent(&child);

        let base = format!("http://127.0.0.1:{port}");
        let started = Instant::now();
        loop {
            if let Some(status) = child.try_wait()? {
                #[cfg(target_os = "linux")]
                let stderr = std::fs::read_to_string(&log_path).unwrap_or_default();
                #[cfg(not(target_os = "linux"))]
                let mut stderr = String::new();
                #[cfg(not(target_os = "linux"))]
                if let Some(mut pipe) = child.stderr.take() {
                    let _ = pipe.read_to_string(&mut stderr);
                }
                bail!(
                    "llama-server exited with {status} before it was ready: {}",
                    stderr
                        .lines()
                        .rev()
                        .take(12)
                        .collect::<Vec<_>>()
                        .join(" | ")
                );
            }
            if http::agent()
                .get(&format!("{base}/health"))
                .timeout(Duration::from_secs(5))
                .call()
                .is_ok()
            {
                break;
            }
            if Instant::now() >= limits.ready_by {
                let _ = child.kill();
                let _ = child.wait();
                bail!(
                    "llama-server did not become ready within {:?}",
                    started.elapsed()
                );
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        // Windows pipes are never drained after startup; a chatty server would fill
        // them and block. The server logs little after it is ready, but drop them.
        #[cfg(not(target_os = "linux"))]
        {
            child.stdout.take();
            child.stderr.take();
        }
        let server = Self {
            child,
            base,
            #[cfg(windows)]
            job,
        };
        // warm-up, as on the iPhone and Android: loads the weights' pages and caches the prompt head
        if !limits.warm_up.is_zero() {
            let _ = correct_one_within(&server.base, "warm up", DEFAULT_MODE, limits.warm_up);
        }
        Ok(server)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        #[cfg(windows)]
        if self.job != 0 {
            unsafe { windows_sys::Win32::Foundation::CloseHandle(self.job as _) };
        }
    }
}

/// Token ids from the model's own tokeniser. Special tokens in the text are
/// parsed; no BOS is added, because the prompt carries `<|startoftext|>`.
fn tokenize(base: &str, text: &str) -> Result<Vec<serde_json::Value>> {
    let value: serde_json::Value = http::agent()
        .post(&format!("{base}/tokenize"))
        .timeout(Duration::from_secs(10))
        .send_json(serde_json::json!({ "content": text, "add_special": false }))?
        .into_json()
        .context("read tokenize reply")?;
    value
        .get("tokens")
        .and_then(|t| t.as_array())
        .cloned()
        .ok_or_else(|| anyhow!("tokenize reply has no tokens: {value}"))
}

/// Correct one transcript on the server at `base`.
fn correct_at(base: &str, raw: &str, mode: &str) -> Result<String> {
    if raw.trim().is_empty() {
        return Ok(String::new());
    }
    // A long transcript is corrected in sentence groups: the prompt plus its output
    // budget must fit the server's context, or the answer stops mid-text (410 s
    // dictation: "n_tokens = 2047, truncated = 1", the tail lost) or the request
    // fails (900 s: "request (3239 tokens) exceeds the available context size").
    let mut parts = Vec::new();
    let mut guard_fired = false;
    for group in sentence_groups(raw, GROUP_WORDS) {
        let (text, fired) = correct_one(base, &group, mode)?;
        guard_fired |= fired;
        parts.push(text);
    }
    let text = parts.join(" ");
    // a length-guard fallback is the raw transcript and stays as it is, as on the Mac
    if guard_fired {
        return Ok(text);
    }
    // the shared output stage, same as the Mac engine: the opener the refiner dropped
    // back in front (phonon_llm::opener), then numbered steps one per line
    // (phonon_llm::linebreak); greeting and sign-off lines are for email only, and
    // this corrector does not know the target app, so it never splits them
    let text = phonon_llm::opener::restore(raw, &text, mode);
    let text = phonon_llm::linebreak::layout(&text, mode, false);
    // then spoken phone numbers, money and dates as digits (phonon_llm::numpass)
    if number_pass_on_in(&crate::paths::data_root()) {
        return Ok(phonon_llm::numpass::apply(raw, &text));
    }
    Ok(text)
}

/// The number pass switch, the Mac's settings.json `number_pass`: one word in this file in
/// the data root, next to `writing_mode.txt`. "on" turns it on; no file or anything else
/// leaves it off (the default until the do-no-harm audit passes). Read at each dictation,
/// so an edit applies to the next one.
const NUMBER_PASS_FILE: &str = "number_pass.txt";

fn number_pass_on_in(root: &Path) -> bool {
    std::fs::read_to_string(root.join(NUMBER_PASS_FILE))
        .is_ok_and(|text| text.trim().eq_ignore_ascii_case("on"))
}

/// Words per correction request. With the 2048-token context: about 330 transcript
/// tokens, a 68-token prompt head and a 1.5 x 330 + 32 output budget, under 1,000.
const GROUP_WORDS: usize = 250;

/// The transcript split after sentence ends (. ? !) into groups of at most
/// `max_words` words; a run-on sentence longer than that is cut at the word limit.
/// One group, the whole text, when it is short enough.
fn sentence_groups(raw: &str, max_words: usize) -> Vec<String> {
    let words: Vec<&str> = raw.split_whitespace().collect();
    if words.len() <= max_words {
        return vec![raw.trim().to_string()];
    }
    let mut groups = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    let mut sentence: Vec<&str> = Vec::new();
    for word in words {
        sentence.push(word);
        let ends = word.ends_with(['.', '?', '!']);
        if ends || sentence.len() >= max_words {
            if current.len() + sentence.len() > max_words && !current.is_empty() {
                groups.push(current.join(" "));
                current.clear();
            }
            current.append(&mut sentence);
        }
    }
    if current.len() + sentence.len() > max_words && !current.is_empty() {
        groups.push(current.join(" "));
        current.clear();
    }
    current.append(&mut sentence);
    if !current.is_empty() {
        groups.push(current.join(" "));
    }
    groups
}

/// One correction request, with the length guard (true when it kept the raw
/// text); no layout.
fn correct_one(base: &str, raw: &str, mode: &str) -> Result<(String, bool)> {
    correct_one_within(base, raw, mode, CORRECTION_TIMEOUT)
}

/// `correct_one` with the completion request bounded by `timeout`.
fn correct_one_within(
    base: &str,
    raw: &str,
    mode: &str,
    timeout: Duration,
) -> Result<(String, bool)> {
    let tokens = tokenize(base, &prompt(raw, mode))?;
    let budget = output_budget(tokenize(base, raw.trim())?.len() as u32);
    let value: serde_json::Value = http::agent()
        .post(&format!("{base}/completion"))
        .timeout(timeout)
        .send_json(serde_json::json!({
            "prompt": tokens,
            "n_predict": budget,
            "temperature": 0.0,
            "top_k": 1,
            // the GGUF can carry sampling defaults (general.sampling.*) that
            // llama-server applies; the refiner is trained for plain greedy
            "repeat_penalty": 1.0,
            "cache_prompt": true,
            "stream": false,
        }))
        .context("correction request failed")?
        .into_json()
        .context("read correction reply")?;
    let (text, fired) = guarded(raw, &parse_reply(&value)?);
    if fired {
        eprintln!("phonon: length guard kept the raw transcript");
    }
    Ok((text, fired))
}

/// The correction model, resident: a server on the GPU when the machine has a
/// discrete one (Windows), otherwise on the CPU, plus (Windows, CPU only) a
/// small fallback server (`LEAN_THREADS` threads) for dictations that start while
/// every core is busy.
pub struct Corrector {
    main: Server,
    device: Device,
    server: PathBuf,
    model: PathBuf,
    lean: Arc<Mutex<Lean>>,
    use_lean: AtomicBool,
    /// The tuned build in use, until the startup check confirms it.
    pending: Option<crate::backends::Installed>,
}

/// The fallback server's state. It starts on a background thread: on the QA VM
/// with every vCPU busy, starting it took about 12 s (2026-11-07), and started at
/// key-down that held up the dictation's text by about 7 s.
enum Lean {
    Absent,
    Starting,
    Ready(Server),
    Failed,
}

impl Corrector {
    /// Start the server and wait until the model answers. Call after
    /// `fetch::ensure_all`.
    ///
    /// The fallback order: a tuned GPU runtime from the backend manifest at
    /// `backends` (when given and this machine has a build), then (Windows) any
    /// Vulkan GPU, then the CPU. Each GPU path has to list its device, start, and
    /// decode at `MIN_GPU_DECODE` or faster before it is used.
    pub fn start(backends: Option<&str>) -> Result<Self> {
        let (server, model) = runtime_files()?;
        let threads = thread_count(
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(2),
        );
        // every server runs from the CPU runtime's directory (see `server_command`)
        let workdir = server.parent().unwrap_or(Path::new(".")).to_path_buf();
        if let Some(url) = backends {
            let started = Instant::now();
            let prepared = crate::backends::prepare_within(url, crate::backends::PREPARE_BUDGET);
            eprintln!(
                "phonon: tuned GPU runtime prepared in {} ms",
                started.elapsed().as_millis()
            );
            match prepared {
                Ok(Some(tuned)) => {
                    let target = Target {
                        kind: "CUDA",
                        env: tuned.env(),
                        pinned: Some(&tuned.device.name),
                    };
                    let deadline = Instant::now() + GPU_BUDGET;
                    // the manifest's build first, then earlier builds still on disk
                    for candidate in &tuned.servers {
                        let label = candidate
                            .parent()
                            .and_then(|dir| dir.file_name())
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_else(|| tuned.id.clone());
                        if let Some(main) = gpu_server(
                            candidate, &target, &label, &model, threads, &workdir, deadline,
                        ) {
                            let mut corrector = Self::new(main, Device::Gpu, server, model);
                            corrector.pending =
                                candidate.parent().map(|dir| crate::backends::Installed {
                                    id: tuned.id.clone(),
                                    dir: dir.to_path_buf(),
                                });
                            return Ok(corrector);
                        }
                    }
                }
                Ok(None) => eprintln!("phonon: no tuned GPU runtime for this machine"),
                Err(error) => eprintln!("phonon: tuned GPU runtime unavailable ({error:#})"),
            }
        }
        #[cfg(windows)]
        {
            let adapters = display_adapter_ids();
            if any_gpu_vendor(&adapters) {
                let vulkan = Target {
                    kind: "Vulkan",
                    env: Vec::new(),
                    pinned: None,
                };
                let deadline = Instant::now() + GPU_BUDGET;
                if let Some(main) = gpu_server(
                    &server, &vulkan, "vulkan", &model, threads, &workdir, deadline,
                ) {
                    return Ok(Self::new(main, Device::Gpu, server, model));
                }
            } else {
                eprintln!(
                    "phonon: no NVIDIA, AMD or Intel display adapter; not listing Vulkan devices"
                );
            }
        }
        let main = Server::cpu(&server, &model, threads)?;
        eprintln!("phonon: correction runs on the CPU");
        Ok(Self::new(main, Device::Cpu, server, model))
    }

    fn new(main: Server, device: Device, server: PathBuf, model: PathBuf) -> Self {
        Self {
            main,
            device,
            server,
            model,
            lean: Arc::new(Mutex::new(Lean::Absent)),
            use_lean: AtomicBool::new(false),
            pending: None,
        }
    }

    /// Where the correction model runs.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The startup check passed: a tuned build in use becomes the family's only
    /// build on disk (earlier ones were kept in case this one failed).
    pub fn confirm(&mut self) {
        if self.device == Device::Gpu {
            if let Some(installed) = self.pending.take() {
                installed.confirm();
            }
        }
    }

    /// Move from the GPU to the CPU, after the GPU gave a wrong answer. Earlier
    /// tuned builds stay on disk.
    pub fn fall_back_to_cpu(&mut self) -> Result<()> {
        let threads = thread_count(
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(2),
        );
        self.main = Server::cpu(&self.server, &self.model, threads)?;
        self.device = Device::Cpu;
        self.pending = None;
        Ok(())
    }

    /// Call while idle and at key-down with the machine's busy fraction. The first
    /// time the CPU is saturated this starts the fallback server in the
    /// background (it shares the mapped weights; about 240 MB of its own), so it is
    /// usually ready before the user dictates. Returns whether a saturated
    /// dictation can use it. Windows only: Linux's scheduler degrades gracefully
    /// (QA VM, all 8 vCPUs busy: 1.9x slower, no stall).
    pub fn warm(&self, busy: Option<f64>) -> bool {
        if self.device != Device::Cpu || !cfg!(windows) {
            return false;
        }
        if !busy.is_some_and(|busy| busy >= SATURATED) {
            return false;
        }
        self.spawn_lean()
    }

    /// Start the fallback corrector now, whatever the load (Windows, CPU only):
    /// on a machine that is already saturated when the user first dictates, the
    /// first-busy-moment start in `warm` left that dictation waiting for it
    /// (W2, Windows 11 VM, 2026-11-10: ready 8.1 s and 25.4 s after the load began).
    pub fn start_lean(&self) {
        if self.device == Device::Cpu && cfg!(windows) {
            self.spawn_lean();
        }
    }

    /// Start the fallback server in the background unless it already started.
    /// Returns whether a saturated dictation can use it.
    fn spawn_lean(&self) -> bool {
        let Ok(mut lean) = self.lean.lock() else {
            return false;
        };
        if matches!(*lean, Lean::Absent) {
            *lean = Lean::Starting;
            let (slot, server, model) = (
                Arc::clone(&self.lean),
                self.server.clone(),
                self.model.clone(),
            );
            let threads = lean_threads(
                std::thread::available_parallelism()
                    .map(|count| count.get())
                    .unwrap_or(2),
            );
            std::thread::spawn(move || {
                let started = Instant::now();
                let next = match Server::cpu(&server, &model, threads) {
                    Ok(server) => {
                        eprintln!(
                            "phonon: {threads}-thread corrector ready in {} ms",
                            started.elapsed().as_millis()
                        );
                        Lean::Ready(server)
                    }
                    Err(error) => {
                        eprintln!("phonon: {threads}-thread corrector did not start: {error:#}");
                        Lean::Failed
                    }
                };
                if let Ok(mut lean) = slot.lock() {
                    *lean = next;
                }
            });
        }
        !matches!(*lean, Lean::Failed)
    }

    /// Call at key-down with the machine's busy fraction: a saturated CPU sends
    /// this dictation's correction to the fallback server.
    pub fn prepare(&self, busy: Option<f64>) {
        let lean = self.warm(busy);
        self.use_lean.store(lean, Ordering::Release);
    }

    /// Whether the next correction goes to the fallback server.
    pub fn lean(&self) -> bool {
        self.use_lean.load(Ordering::Acquire)
    }

    fn base(&self) -> String {
        if self.lean() {
            // Still starting: wait for it. On a saturated CPU the main server's
            // threads stall each other far longer (9950X3D: no 32-token decode in
            // 120 s; DEVLOG 2026-11-07 9:50 pm).
            let started = Instant::now();
            loop {
                match self.lean.lock().as_deref() {
                    Ok(Lean::Ready(server)) => return server.base.clone(),
                    Ok(Lean::Starting) if started.elapsed() < READY_TIMEOUT => {}
                    _ => break,
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        self.main.base.clone()
    }

    /// Correct one transcript in the default writing mode.
    pub fn correct(&self, raw: &str) -> Result<String> {
        self.correct_in(raw, DEFAULT_MODE)
    }

    /// Correct one transcript in a writing mode (minimal, casual, formal, prompt).
    pub fn correct_in(&self, raw: &str, mode: &str) -> Result<String> {
        correct_at(&self.base(), raw, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_pass_is_off_unless_the_file_says_on() {
        let root = std::env::temp_dir().join(format!("phonon-win-numpass-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        assert!(!number_pass_on_in(&root)); // no file
        for (text, on) in [
            ("off\n", false),
            ("", false),
            ("yes", false),
            ("on\n", true),
            (" ON ", true),
        ] {
            std::fs::write(root.join(NUMBER_PASS_FILE), text).unwrap();
            assert_eq!(number_pass_on_in(&root), on, "{text:?}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn vulkan_is_listed_only_with_a_gpu_vendors_adapter() {
        let ids = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        // RTX 3090 desktop with a Radeon iGPU; an Intel laptop; an AMD APU alone
        assert!(any_gpu_vendor(&ids(&[
            "PCI\\VEN_10DE&DEV_2204&SUBSYS_38811462&REV_A1",
            "PCI\\VEN_1002&DEV_13C0&SUBSYS_00000000&REV_C1",
        ])));
        assert!(any_gpu_vendor(&ids(&[
            "PCI\\VEN_8086&DEV_A7A0&SUBSYS_0B141028&REV_04"
        ])));
        assert!(any_gpu_vendor(&ids(&["pci\\ven_1002&dev_15bf&rev_c4"])));
        // a QEMU VM (virtio-vga, Bochs/stdvga), Microsoft Basic Display, Hyper-V, nothing
        assert!(!any_gpu_vendor(&ids(&[
            "PCI\\VEN_1AF4&DEV_1050&SUBSYS_11001AF4&REV_01"
        ])));
        assert!(!any_gpu_vendor(&ids(&[
            "PCI\\VEN_1234&DEV_1111&SUBSYS_11001AF4&REV_02"
        ])));
        assert!(!any_gpu_vendor(&ids(&[
            "ROOT\\BasicDisplay",
            "ROOT\\BasicRender"
        ])));
        assert!(!any_gpu_vendor(&ids(&[
            "VMBUS\\{DA0A7802-E377-4AAC-8E77-0558EB1073F8}"
        ])));
        assert!(!any_gpu_vendor(&[]));
        // a vendor id later in the string (a subsystem id) is not the adapter's vendor
        assert!(!any_gpu_vendor(&ids(&[
            "PCI\\VEN_1AF4&DEV_1050&SUBSYS_VEN_10DE"
        ])));
    }

    #[test]
    fn a_short_transcript_is_one_group() {
        assert_eq!(
            sentence_groups("  Hello there.  ", 250),
            vec!["Hello there."]
        );
    }

    #[test]
    fn long_transcripts_split_after_sentences_and_keep_every_word() {
        let sentence = "one two three four five six seven eight nine ten.";
        let raw = vec![sentence; 60].join(" "); // 600 words
        let groups = sentence_groups(&raw, 250);
        assert_eq!(groups.len(), 3);
        for group in &groups {
            assert!(group.split_whitespace().count() <= 250);
            assert!(group.ends_with('.'), "cut mid-sentence: {group}");
        }
        assert_eq!(groups.join(" "), raw);
    }

    #[test]
    fn a_run_on_sentence_is_cut_at_the_word_limit() {
        let raw = vec!["word"; 610].join(" ");
        let groups = sentence_groups(&raw, 250);
        assert!(groups.iter().all(|g| g.split_whitespace().count() <= 250));
        assert_eq!(groups.join(" "), raw);
    }

    #[test]
    fn the_prompt_matches_the_iphone_build() {
        let p = prompt("  hello there  ", "minimal");
        assert!(p.starts_with(
            "<|startoftext|><|im_start|>system\nRewrite the raw dictation transcript"
        ));
        assert!(p.contains(
            "do not add or drop content.\nWriting mode: minimal (notes and documents).<|im_end|>"
        ));
        assert!(p.ends_with(
            "<|im_start|>user\nhello there<|im_end|>\n<|im_start|>assistant\n<think></think>"
        ));
    }

    #[test]
    fn unknown_modes_fall_back_to_minimal() {
        assert_eq!(
            mode_line("prompt"),
            "Writing mode: prompt (an instruction to an AI assistant)."
        );
        assert_eq!(mode_line("shouting"), mode_line("minimal"));
    }

    #[test]
    fn budget_is_one_and_a_half_times_plus_32() {
        assert_eq!(output_budget(0), 32);
        assert_eq!(output_budget(20), 62);
        assert_eq!(output_budget(401), 633);
    }

    #[test]
    fn the_length_guard_keeps_the_raw_text_on_a_runaway() {
        assert_eq!(
            guarded("one two three", "One, two, three."),
            ("One, two, three.".into(), false)
        );
        assert_eq!(
            guarded("one two three", "  "),
            ("one two three".into(), true)
        );
        let runaway = "word ".repeat(30);
        assert!(guarded("one two three", &runaway).1);
    }

    #[test]
    fn reads_the_completion_content() {
        let reply = serde_json::json!({ "content": "Hello, Fluid Voice.", "stop": true });
        assert_eq!(parse_reply(&reply).unwrap(), "Hello, Fluid Voice.");
        assert!(parse_reply(&serde_json::json!({})).is_err());
    }

    #[test]
    fn picks_the_listed_vulkan_device() {
        let listing = "load_backend: loaded Vulkan backend from ggml-vulkan.dll\n\
            Available devices:\n  Vulkan0: NVIDIA GeForce RTX 3090 (24322 MiB, 23554 MiB free)\n";
        assert_eq!(pick_device(listing, "Vulkan").as_deref(), Some("Vulkan0"));
        let two = "Available devices:\n  Vulkan1: AMD Radeon RX 7600 (8176 MiB, 8000 MiB free)\n";
        assert_eq!(pick_device(two, "Vulkan").as_deref(), Some("Vulkan1"));
        // no Vulkan driver (the QA VM): the CPU
        assert_eq!(
            pick_device("Available devices:\n  (none)\n", "Vulkan"),
            None
        );
        assert_eq!(
            pick_device(
                "load_backend: loaded CPU backend from ggml-cpu-zen4.dll",
                "Vulkan"
            ),
            None
        );
    }

    #[test]
    fn picks_the_listed_cuda_device() {
        let listing = "load_backend: loaded CUDA backend from /x/libggml-cuda.so\n\
            load_backend: loaded CPU backend from /x/libggml-cpu-zen4.so\n\
            Available devices:\n  CUDA0: NVIDIA GeForce RTX 3090 (24135 MiB, 18634 MiB free)\n";
        assert_eq!(pick_device(listing, "CUDA").as_deref(), Some("CUDA0"));
        assert_eq!(pick_device(listing, "Vulkan"), None);
        // the module did not load (no driver, CUDA_VISIBLE_DEVICES empty): no device
        let none =
            "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected\n\
            Available devices:\n";
        assert_eq!(pick_device(none, "CUDA"), None);
    }

    #[test]
    fn correction_threads_follow_physical_cores() {
        assert_eq!(thread_count(1), 1);
        assert_eq!(thread_count(2), 1);
        assert_eq!(thread_count(4), 3);
        assert_eq!(thread_count(8), 4);
        assert_eq!(thread_count(16), 8);
        assert_eq!(thread_count(32), 16);
        assert_eq!(thread_count(64), 16);
    }

    #[test]
    fn fallback_threads_never_exceed_the_main_server() {
        for logical_cpus in [1, 2, 3, 4, 8, 16, 64] {
            let lean = lean_threads(logical_cpus);
            assert!(lean >= 1);
            assert!(lean <= LEAN_THREADS);
            assert!(lean <= thread_count(logical_cpus));
        }
        assert_eq!(lean_threads(2), 1);
    }

    /// A fake `llama-server`: a shell script that answers `--list-devices` with
    /// `listing` and otherwise runs `serve`. No GPU and no model are involved.
    #[cfg(unix)]
    fn fake_server(dir: &Path, listing: &str, serve: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join("llama-server");
        let script = format!(
            "#!/bin/sh\nif [ \"$1\" = --list-devices ]; then\n{listing}\nexit 0\nfi\n{serve}\n"
        );
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    /// Every way a tuned CUDA server can fail at start ends in None (the caller
    /// goes on to the next build, Vulkan or the CPU) within the GPU deadline.
    #[cfg(unix)]
    #[test]
    fn a_failing_gpu_server_is_given_up_within_its_deadline() {
        let root = std::env::temp_dir().join(format!("phonon-win-fakegpu-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let rtx3090 = "echo '  CUDA0: NVIDIA GeForce RTX 3090 (24145 MiB, 21037 MiB free)'";
        let target = Target {
            kind: "CUDA",
            env: vec![("CUDA_VISIBLE_DEVICES".into(), "GPU-test".into())],
            pinned: Some("NVIDIA GeForce RTX 3090"),
        };
        let cases = [
            // the pin did not take: another card, or two cards
            ("wrong-device", "echo '  CUDA0: NVIDIA GeForce RTX 2080 Ti (11000 MiB, 10000 MiB free)'", "exit 1"),
            ("two-devices", "echo '  CUDA0: NVIDIA GeForce RTX 3090 (1 MiB, 1 MiB free)'; echo '  CUDA1: NVIDIA GeForce RTX 3090 (1 MiB, 1 MiB free)'", "exit 1"),
            // the module did not load (old driver, missing library): no device
            ("no-device", "echo 'Available devices:'", "exit 1"),
            // a hung driver during the listing
            ("listing-hangs", "exec sleep 60", "exit 1"),
            // the server crashes while loading
            ("server-crashes", rtx3090, "echo 'CUDA error: no kernel image' >&2; exit 134"),
            // the server never answers its health check
            ("server-hangs", rtx3090, "exec sleep 60"),
        ];
        for (name, listing, serve) in cases {
            let dir = root.join(name);
            let server = fake_server(&dir, listing, serve);
            let started = Instant::now();
            let deadline = started + Duration::from_secs(3);
            let main = gpu_server(
                &server,
                &target,
                name,
                &dir.join("model.gguf"),
                1,
                &dir,
                deadline,
            );
            assert!(main.is_none(), "{name}");
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "{name} took {:?}",
                started.elapsed()
            );
        }
        // no time left: nothing is started at all
        let server = fake_server(&root.join("late"), rtx3090, "exit 1");
        let started = Instant::now();
        assert!(gpu_server(&server, &target, "late", &root.join("m"), 1, &root, started).is_none());
        assert!(started.elapsed() < Duration::from_secs(1));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn deadlines_never_go_negative() {
        let now = Instant::now();
        assert_eq!(left(now, GPU_STEP), Duration::ZERO);
        assert!(left(now + Duration::from_secs(5), GPU_STEP) <= Duration::from_secs(5));
        assert_eq!(left(now + Duration::from_secs(500), GPU_STEP), GPU_STEP);
        let gpu = Limits::gpu(now + Duration::from_secs(5));
        assert!(gpu.warm_up <= Duration::from_secs(5));
        assert!(Limits::cpu().ready_by > now + GPU_BUDGET);
    }
}
