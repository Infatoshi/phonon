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

/// The mode Windows and Linux use: nothing tells them yet which app has focus.
pub const DEFAULT_MODE: &str = "minimal";

/// The model may take this long to load before the server is called dead.
const READY_TIMEOUT: Duration = Duration::from_secs(600);

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
fn decode_rate(base: &str) -> Option<f64> {
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
        .timeout(Duration::from_secs(60))
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
/// runtime pins its device with CUDA_VISIBLE_DEVICES).
fn server_command(server: &Path, env: &[(String, String)]) -> Result<Command> {
    let mut command = Command::new(server);
    command.stdin(Stdio::null());
    command.envs(env.iter().map(|(k, v)| (k, v)));
    if let Some(dir) = server.parent() {
        command.current_dir(dir);
    }
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
fn device_listing(server: &Path, env: &[(String, String)]) -> Option<String> {
    let mut command = server_command(server, env).ok()?;
    command
        .arg("--list-devices")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().ok()?;
    let started = Instant::now();
    // A driver can take seconds to initialise; a hung one must not hold up startup.
    while child.try_wait().ok()?.is_none() {
        if started.elapsed() > Duration::from_secs(20) {
            let _ = child.kill();
            eprintln!("phonon: listing GPUs took over 20 s; using the CPU");
            return None;
        }
        std::thread::sleep(Duration::from_millis(100));
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

/// A server on the target GPU of the runtime at `server`, if it lists it, starts,
/// and decodes fast enough. `label` names the runtime in the log.
fn gpu_server(
    server: &Path,
    target: &Target,
    label: &str,
    model: &Path,
    threads: usize,
) -> Option<Server> {
    let kind = target.kind;
    let listing = device_listing(server, &target.env).unwrap_or_default();
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
    match Server::spawn(server, model, threads, Some(&gpu), &target.env) {
        // Twice: the first Vulkan run after an install compiles the GPU's shaders
        // (3090: 66 tokens/s cold, 300 warm).
        Ok(main) => match [decode_rate(&main.base), decode_rate(&main.base)]
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
    /// Start a server and wait until the model answers its health check.
    fn spawn(
        server: &Path,
        model: &Path,
        threads: usize,
        gpu: Option<&str>,
        env: &[(String, String)],
    ) -> Result<Self> {
        let port = free_port()?;
        let mut command = server_command(server, env)?;
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
            // A pipe that nobody drains fills during model loading on Linux.
            // Keep diagnostics on disk so startup and later requests cannot block.
            let logs = crate::paths::data_root().join("logs");
            std::fs::create_dir_all(&logs)?;
            let path = logs.join(format!("llama-server-{port}.log"));
            let log = std::fs::File::create(&path).context("create the correction server log")?;
            command.stdout(Stdio::null()).stderr(log);
            // SIGTERM the server when Phonon dies, however it dies: a `pkill phonon`
            // left the old server running with 2 GB (QA VM, 2026-11-07). The signal
            // follows the spawning thread, which is the main thread here.
            use std::os::unix::process::CommandExt;
            unsafe {
                command.pre_exec(|| {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
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
            if started.elapsed() > READY_TIMEOUT {
                let _ = child.kill();
                bail!("llama-server did not become ready within {READY_TIMEOUT:?}");
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
        let _ = correct_at(&server.base, "warm up", DEFAULT_MODE);
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
    Ok(phonon_llm::linebreak::layout(&text, mode, false))
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
    let tokens = tokenize(base, &prompt(raw, mode))?;
    let budget = output_budget(tokenize(base, raw.trim())?.len() as u32);
    let value: serde_json::Value = http::agent()
        .post(&format!("{base}/completion"))
        .timeout(Duration::from_secs(300))
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
/// one-thread server for dictations that start while every core is busy.
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

/// The one-thread server's state. It starts on a background thread: on the QA VM
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
        if let Some(url) = backends {
            match crate::backends::prepare(url) {
                Ok(Some(tuned)) => {
                    let target = Target {
                        kind: "CUDA",
                        env: tuned.env(),
                        pinned: Some(&tuned.device.name),
                    };
                    // the manifest's build first, then earlier builds still on disk
                    for candidate in &tuned.servers {
                        let label = candidate
                            .parent()
                            .and_then(|dir| dir.file_name())
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_else(|| tuned.id.clone());
                        if let Some(main) = gpu_server(candidate, &target, &label, &model, threads)
                        {
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
            let vulkan = Target {
                kind: "Vulkan",
                env: Vec::new(),
                pinned: None,
            };
            if let Some(main) = gpu_server(&server, &vulkan, "vulkan", &model, threads) {
                return Ok(Self::new(main, Device::Gpu, server, model));
            }
        }
        let main = Server::spawn(&server, &model, threads, None, &[])?;
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
        self.main = Server::spawn(&self.server, &self.model, threads, None, &[])?;
        self.device = Device::Cpu;
        self.pending = None;
        Ok(())
    }

    /// Call while idle and at key-down with the machine's busy fraction. The first
    /// time the CPU is saturated this starts the one-thread server in the
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
            std::thread::spawn(move || {
                let started = Instant::now();
                let next = match Server::spawn(&server, &model, 1, None, &[]) {
                    Ok(server) => {
                        eprintln!(
                            "phonon: one-thread corrector ready in {} ms",
                            started.elapsed().as_millis()
                        );
                        Lean::Ready(server)
                    }
                    Err(error) => {
                        eprintln!("phonon: one-thread corrector did not start: {error:#}");
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
    /// this dictation's correction to the one-thread server.
    pub fn prepare(&self, busy: Option<f64>) {
        let lean = self.warm(busy);
        self.use_lean.store(lean, Ordering::Release);
    }

    /// Whether the next correction goes to the one-thread server.
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
}
