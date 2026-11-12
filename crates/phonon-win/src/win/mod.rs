//! The Windows application: tray, hook, capture, correction, insertion.
//!
//! Two threads. The main thread owns the message-only window, the tray icon, and
//! the keyboard hook, and it does nothing else, because a low-level hook that
//! blocks is removed by Windows. The worker thread owns everything slow: the
//! first-run download, the resident correction server, the microphone, and the
//! insertion.

pub mod capture;
pub mod hook;
pub mod inject;
pub mod load;
pub mod tray;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use phonon_hotkey::autostop::{self, Decision};
use phonon_hotkey::Action;

use crate::fetch;
use crate::history;
use crate::manifest;
use crate::paths;
use crate::pipeline::Engine;
use crate::polish::Device;

/// Everything the worker thread reacts to.
enum Event {
    Key(Action),
    Menu(tray::Command),
}

/// Start the application. Returns when the user quits.
pub fn run() -> Result<()> {
    // The executable is a console program so that `info`, `fetch`, and
    // `selftest` print to a terminal and to continuous integration. The tray
    // application has no use for a console, so it gives the one it inherited
    // back. Started from Explorer, a console window appears for an instant.
    unsafe { windows_sys::Win32::System::Console::FreeConsole() };

    // Above normal (polish.rs starts the correction server the same way: Windows
    // does not pass this class on to child processes by itself). With a
    // busy loop on every CPU at normal priority, a Normal Phonon never started the
    // recording within 120 s of the key press; AboveNormal dictated in 12-14 s
    // (QA VM, 2026-11-07). Phonon is idle between dictations, so this only orders
    // the second or two of a dictation ahead of background work.
    unsafe {
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess, SetPriorityClass, ABOVE_NORMAL_PRIORITY_CLASS,
        };
        SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
    }

    let (events, inbox) = channel::<Event>();

    let (menu_tx, menu_rx) = channel::<tray::Command>();
    tray::create(menu_tx).context("create the tray icon")?;

    let (key_tx, key_rx) = channel::<Action>();
    let (name, code) = hook::configured_key();
    hook::install(code, key_tx).context("install the keyboard hook")?;
    // This thread runs the keyboard hook and the tray, and nothing slow. Windows
    // skips a low-level hook that does not answer within its timeout: with a busy
    // loop on every CPU at normal priority the hook starved, a key-up arrived
    // 100 s late (a 104 s recording, then an empty one; QA VM, 2026-11-07).
    // Time-critical keeps the hook answering on a saturated machine; it uses
    // microseconds per key, so it takes nothing from anyone.
    unsafe {
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_TIME_CRITICAL,
        };
        if SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL) == 0 {
            eprintln!("phonon: could not raise the keyboard thread's priority");
        }
    }
    eprintln!("phonon: dictation key is {name}");

    // Two channels feed one worker; forward both. Keys pressed before the models
    // are ready are dropped with a hint instead of queueing silently: on a first
    // run the download and warmup take minutes, and a press then used to do
    // nothing visible and replay later (win-beta.1 report: "Right Ctrl did nothing").
    forward_keys(key_rx, events.clone());
    forward(menu_rx, events.clone(), Event::Menu);

    let worker = std::thread::spawn(move || {
        if let Err(error) = work(inbox) {
            eprintln!("phonon: {error:#}");
            // A balloon dies with the tray icon, which goes away as soon as the
            // loop quits, so a fatal error also gets a dialog the user must close.
            fatal_dialog(&format!("{error:#}"));
        }
        tray::post_quit();
    });

    tray::run_message_loop();
    hook::uninstall();
    tray::destroy();
    let _ = worker.join();
    Ok(())
}

/// Set once the startup check passes; the key forwarder drops presses until then.
static READY: AtomicBool = AtomicBool::new(false);

/// Forward dictation-key actions to the worker once Phonon is ready. Before that,
/// a key-down shows (at most every 15 s) what Phonon is still doing.
fn forward_keys(source: Receiver<Action>, sink: Sender<Event>) {
    std::thread::spawn(move || {
        let mut last_hint: Option<Instant> = None;
        while let Ok(action) = source.recv() {
            if !READY.load(Ordering::Acquire) {
                let due = last_hint.is_none_or(|t| t.elapsed() >= Duration::from_secs(15));
                if matches!(action, Action::Start) && due {
                    let doing = tray::status();
                    let body = if doing.is_empty() {
                        "Dictation starts as soon as the models are ready.".to_string()
                    } else {
                        format!("Dictation starts as soon as the models are ready (now: {doing}).")
                    };
                    tray::notify("Phonon is still setting up", &body, false);
                    last_hint = Some(Instant::now());
                }
                continue;
            }
            if sink.send(Event::Key(action)).is_err() {
                break;
            }
        }
    });
}

/// Move items from one channel to another, wrapping them.
fn forward<T: Send + 'static>(source: Receiver<T>, sink: Sender<Event>, wrap: fn(T) -> Event) {
    std::thread::spawn(move || {
        while let Ok(item) = source.recv() {
            if sink.send(wrap(item)).is_err() {
                break;
            }
        }
    });
}

/// Show "Phonon stopped" with the reason, and wait until the user closes it.
fn fatal_dialog(text: &str) {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        MessageBoxW, MB_ICONERROR, MB_OK, MB_SETFOREGROUND,
    };
    let wide = |s: &str| s.encode_utf16().chain(Some(0)).collect::<Vec<u16>>();
    let (body, title) = (wide(text), wide("Phonon stopped"));
    unsafe {
        MessageBoxW(
            std::ptr::null_mut(),
            body.as_ptr(),
            title.as_ptr(),
            MB_OK | MB_ICONERROR | MB_SETFOREGROUND,
        )
    };
}

/// The Visual C++ runtime DLLs the downloaded llama.cpp and sherpa-onnx builds
/// link against. Phonon itself links the C runtime statically, but those two do
/// not, and a clean Windows 11 does not ship them: llama-server then dies with a
/// "VCRUNTIME140.dll was not found" system dialog after the 3.8 GB download.
const VC_RUNTIME_DLLS: [&str; 3] = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"];
const VC_REDIST_URL: &str = "https://aka.ms/vs/17/release/vc_redist.x64.exe";

/// The runtime DLLs that System32 does not have.
fn missing_vc_runtime() -> Vec<&'static str> {
    use windows_sys::Win32::Foundation::FreeLibrary;
    use windows_sys::Win32::System::LibraryLoader::{LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32};
    VC_RUNTIME_DLLS
        .into_iter()
        .filter(|name| {
            let wide: Vec<u16> = name.encode_utf16().chain(Some(0)).collect();
            let module = unsafe {
                LoadLibraryExW(
                    wide.as_ptr(),
                    std::ptr::null_mut(),
                    LOAD_LIBRARY_SEARCH_SYSTEM32,
                )
            };
            if module.is_null() {
                return true;
            }
            unsafe { FreeLibrary(module) };
            false
        })
        .collect()
}

/// First-run download, warmup, then the dictation loop.
fn work(inbox: Receiver<Event>) -> Result<()> {
    refresh_history();
    // Check before the download, not after it: without the runtime nothing that
    // the download brings can start.
    let missing_runtime = missing_vc_runtime();
    if !missing_runtime.is_empty() {
        anyhow::bail!(
            "Phonon needs the Microsoft Visual C++ Redistributable (x64), which this PC does not have \
             ({} missing). Install it from {VC_REDIST_URL}, then start Phonon again.",
            missing_runtime.join(", ")
        );
    }

    let missing: Vec<_> = manifest::ALL
        .iter()
        .filter(|component| !fetch::installed(component))
        .collect();
    if !missing.is_empty() {
        let total: u64 = missing.iter().map(|c| c.download_bytes()).sum();
        tray::notify(
            "Phonon is setting up",
            &format!(
                "Downloading {} of speech and correction models. Phonon works offline afterwards.",
                manifest::human_bytes(total)
            ),
            false,
        );
    }
    fetch::ensure_all(&mut |progress| tray::set_status(&progress.line()))
        .context("download the speech and correction models")?;

    tray::set_status("loading the correction model");
    let mut engine = Engine::start(crate::backends::DEFAULT_MANIFEST)?;
    load::start();

    // The same readiness gate macOS applies: prove the real audio path before
    // telling the user Phonon is ready.
    tray::set_status("running the startup check");
    let mut check = startup_check(&engine);
    if check.is_ok() {
        engine.confirm_backend();
    }
    if check.is_err() && engine.device() == Device::Gpu {
        // A GPU driver that answers wrongly must not cost the user dictation.
        eprintln!("phonon: startup check failed on the GPU; retrying on the CPU");
        tray::set_status("loading the correction model on the CPU");
        engine.fall_back_to_cpu()?;
        check = startup_check(&engine);
    }
    match check {
        Ok(text) => eprintln!("phonon: startup check passed: {text}"),
        Err(error) => {
            tray::notify("Phonon startup check failed", &format!("{error:#}"), true);
            return Err(error);
        }
    }
    // On the CPU, the fallback corrector for a saturated machine starts now, not
    // at the first busy moment, so the first dictation under load does not wait.
    engine.start_lean();
    crate::settings::record_engine(engine.device());

    let (name, _) = hook::current_key();
    tray::set_status(&idle_text());
    READY.store(true, Ordering::Release);
    tray::notify(
        "Phonon is ready",
        &format!("Hold {name} to dictate, or tap to toggle."),
        false,
    );
    // the tray menu reads `update::available` each time it opens
    crate::update::start(|found| {
        tray::notify(
            &format!("Phonon {} is out", found.title),
            "Download it from the tray menu.",
            false,
        )
    });

    let mut recording: Option<capture::Recording> = None;
    let mut settings_watch = crate::settings::Watch::of(crate::settings::FILE);
    let _ = settings_watch.changed_in(&paths::data_root());
    // the settings window can clear the history; the tray's recent list follows
    let mut history_watch = crate::settings::Watch::of(history::LOG_FILE);
    let _ = history_watch.changed_in(&paths::data_root());
    loop {
        let event = match inbox.recv_timeout(Duration::from_millis(500)) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => {
                // The settings window writes settings.json; a new key applies here,
                // between dictations. Mode, history and the clipboard rule are read
                // at each dictation.
                if recording.is_none() {
                    let root = paths::data_root();
                    let settings_changed = settings_watch.changed_in(&root);
                    if settings_changed && hook::follow_settings().is_some() {
                        tray::set_status(&idle_text());
                    }
                    if history_watch.changed_in(&root) || settings_changed {
                        refresh_history();
                    }
                }
                match &mut recording {
                    Some(active) => match active.auto_stop(hook::latched()) {
                        Decision::KeepRecording => {
                            tray::set_status(&format!("recording {:.1} s", active.seconds()))
                        }
                        // The key is up (it latched), so the next press must start, not
                        // stop; then the same stop the next tap would make.
                        Decision::Commit => {
                            eprintln!(
                                "phonon: latched dictation silent for {} s after speech; stopping",
                                autostop::SILENCE_AFTER_SPEECH_SECONDS
                            );
                            hook::reset_latch();
                            finish(&engine, recording.take());
                        }
                        Decision::Discard => {
                            eprintln!(
                                "phonon: latched dictation heard no speech in {} s; discarding",
                                autostop::NO_SPEECH_SECONDS
                            );
                            hook::reset_latch();
                            if let Some(active) = recording.take() {
                                active.discard();
                            }
                            tray::set_status(&idle_text());
                        }
                    },
                    // The fallback corrector (`polish::LEAN_THREADS` threads) started
                    // after the startup check; if it has not, start it now that the
                    // CPU saturates, in the background, rather than at the next key-down.
                    None => engine.warm(load::busy()),
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        };
        match event {
            Event::Menu(tray::Command::Quit) => {
                tray::set_last(None);
                break;
            }
            Event::Menu(tray::Command::OpenFolder) => {
                let root = paths::data_root();
                let _ = std::process::Command::new("explorer").arg(&root).spawn();
            }
            Event::Menu(tray::Command::Toggle) => {
                // The menu and the key drive the same two states. A menu stop
                // must also clear a latch, or the next key press would restart.
                hook::reset_latch();
                if recording.is_some() {
                    finish(&engine, recording.take());
                } else {
                    recording = begin();
                    if recording.is_some() {
                        engine.prepare(load::busy());
                    }
                }
            }
            Event::Key(Action::Start) => {
                if recording.is_none() {
                    recording = begin();
                    // After the microphone is recording. Returns at once; a
                    // fallback corrector it starts comes up in the background.
                    if recording.is_some() {
                        engine.prepare(load::busy());
                    }
                }
            }
            Event::Key(Action::Stop) => {
                finish(&engine, recording.take());
            }
            Event::Key(Action::None) => {}
            Event::Menu(tray::Command::KeyChanged) => {
                if recording.is_none() {
                    tray::set_status(&idle_text());
                }
            }
            Event::Menu(tray::Command::ToggleHistory) => toggle_history(),
            // A recording in progress is still in memory, so everything on disk can go.
            Event::Menu(tray::Command::ClearHistory) => clear_history(),
            Event::Menu(tray::Command::CopyRecent(text)) => copy_recent(text),
        }
    }
    Ok(())
}

/// Run the embedded fixture through both stages, exactly as macOS does.
fn startup_check(engine: &Engine) -> Result<String> {
    let wav = paths::data_root().join("startup.wav");
    if let Some(parent) = wav.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&wav, crate::STARTUP_WAV).context("write the startup fixture")?;
    let result = engine.run(&wav)?;
    crate::assert_startup(&result)?;
    Ok(result.polished)
}

fn begin() -> Option<capture::Recording> {
    match capture::start() {
        Ok(recording) => {
            tray::set_status(&format!("recording from {}", recording.device()));
            Some(recording)
        }
        Err(error) => {
            let mut text = format!("{error:#}");
            // E_ACCESSDENIED from WASAPI: Windows privacy settings block the
            // microphone for desktop apps (seen on a stock Server 2022).
            if text.contains("0x80070005") {
                text.push_str(
                    ". Turn on Settings > Privacy & security > Microphone, including \"Let desktop apps access your microphone\"",
                );
            }
            eprintln!("phonon: cannot hear you: {text}");
            tray::notify("Phonon cannot hear you", &text, true);
            None
        }
    }
}

/// The resting tooltip. The key can change from the tray menu, so it is read each time.
fn idle_text() -> String {
    format!("ready. Hold {} to dictate", hook::current_key().0)
}

fn finish(engine: &Engine, recording: Option<capture::Recording>) {
    let Some(recording) = recording else { return };
    // A tap, or a recording a starved key-up cut to nothing: no word fits in it.
    if recording.seconds() < 0.25 {
        tray::set_status("nothing was said");
        return;
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_millis())
        .unwrap_or(0);
    let path = paths::recordings().join(format!("{stamp}.wav"));
    tray::set_status("transcribing");
    let captured = match recording.stop() {
        Ok(captured) => captured,
        Err(error) => {
            eprintln!("phonon: {error:#}");
            tray::set_status(&idle_text());
            return;
        }
    };
    // No file on the way to the text: with every core busy, Defender's scan of
    // the wave file Phonon wrote and read back cost 4.7 s, and its first-use
    // scans of the speech model's files over a minute (QA VM, 2026-11-07). The
    // history copy is written after the text is in, except for a recording over a
    // minute: that one is saved first, so nothing long is lost if transcription
    // fails or Phonon dies (a 410 s pass terminated the process, 2026-11-08).
    // With local history off that file is deleted once the text is in (see
    // `history::keep_wav`).
    let seconds = captured.seconds();
    let saved_first = engine.resident() && seconds > history::SAVE_FIRST_SECONDS;
    if saved_first {
        if let Err(error) = captured.write(&path) {
            eprintln!("phonon: could not keep the recording: {error:#}");
        }
    }
    let outcome = if engine.resident() {
        engine.run_samples(captured.sample_rate, &captured.to_f32())
    } else {
        captured.write(&path).and_then(|wav| engine.dictate(&wav))
    };
    let keep_history = history::enabled();
    let keep = keep_history && engine.resident() && !saved_first;
    match &outcome {
        Ok(result) => {
            eprintln!("phonon: {}", result.summary());
            if result.polished.trim().is_empty() {
                tray::set_status("nothing was said");
            } else {
                // Before the paste, so the tray has it whatever happens next.
                tray::set_last(Some(result.polished.trim().to_string()));
                // Every insertion ends with a space, as on macOS, so the next
                // dictation does not run into this one ("afternoon?I think").
                let text = format!("{} ", result.polished.trim());
                if inject::nowhere_to_paste() {
                    eprintln!("phonon: no window to paste into; keeping the text");
                    keep_unpasted(result.polished.trim());
                } else {
                    let after = crate::settings::Settings::load().clipboard_after_paste();
                    match inject::insert(&text, after) {
                        Ok(path) => tray::set_status(&format!("inserted ({path})")),
                        Err(error) => {
                            // the error names the step that failed, never the text
                            eprintln!("phonon: could not insert the text: {error:#}");
                            keep_unpasted(result.polished.trim());
                        }
                    }
                }
            }
        }
        Err(error) => {
            eprintln!("phonon: {error:#}");
            tray::notify(
                "Phonon could not transcribe that",
                &format!("{error:#}"),
                true,
            );
        }
    }
    if keep {
        if let Err(error) = captured.write(&path) {
            eprintln!("phonon: could not keep the recording: {error:#}");
        }
    }
    if keep_history {
        let (raw, final_text) = match &outcome {
            Ok(result) => (result.raw.as_str(), result.polished.trim()),
            Err(_) => ("", ""),
        };
        let entry = history::Entry::new(history::now_ms(), seconds, raw, final_text, &path);
        if let Err(error) = history::append(&entry) {
            eprintln!("phonon: could not save the dictation to history: {error:#}");
        }
        refresh_history();
    } else if !history::keep_wav(false, outcome.is_ok(), seconds) {
        history::remove_if_present(&path);
    }
    tray::set_status(&idle_text());
}

/// Show the history switch and the recent list as they are on disk.
fn refresh_history() {
    tray::set_history(history::enabled(), history::recent_now());
}

fn toggle_history() {
    let on = !history::enabled();
    match history::set_enabled_in(&paths::data_root(), on) {
        Ok(()) => {
            eprintln!("phonon: local history {}", if on { "on" } else { "off" });
            refresh_history();
        }
        Err(error) => tray::notify(
            "Phonon could not change Keep history",
            &format!("{error:#}"),
            true,
        ),
    }
}

/// No confirmation dialog: a tray has nowhere to put one but a modal box, and
/// the menu item says what it does.
fn clear_history() {
    match history::clear(None) {
        Ok(removed) => {
            eprintln!("phonon: history cleared ({removed} recordings removed)");
            refresh_history();
            tray::notify(
                "History cleared",
                &format!(
                    "Removed the dictation history and {removed} recording{}.",
                    if removed == 1 { "" } else { "s" }
                ),
                false,
            );
        }
        Err(error) => tray::notify(
            "Phonon could not clear history",
            &format!("{error:#}"),
            true,
        ),
    }
}

/// A dictation that could not be pasted stays on the clipboard (no restore) and
/// in the tray's "Copy last dictation", and the balloon says where it is.
fn keep_unpasted(text: &str) {
    match inject::copy(text) {
        Ok(()) => {
            tray::set_status("kept on the clipboard");
            tray::notify(
                crate::pipeline::NOT_PASTED_TITLE,
                crate::pipeline::NOT_PASTED_BODY,
                false,
            );
        }
        Err(error) => tray::notify(
            "Phonon could not insert the text",
            &format!("{error:#}. It is in the tray: Copy last dictation."),
            true,
        ),
    }
}

fn copy_recent(text: String) {
    match inject::copy(&text) {
        Ok(()) => tray::notify("Copied to the clipboard", "Paste it with Ctrl+V.", false),
        Err(error) => tray::notify("Phonon could not copy that", &format!("{error:#}"), true),
    }
}
