//! The Linux desktop application: tray, dictation key, capture, correction, insertion.
//!
//! The same flow as Windows (`win/mod.rs`): first-run download, a startup check
//! on the embedded fixture, then hold the key to dictate, or tap it to start and tap again to stop.
//! Works on X11 and Wayland alike because the key and the paste go through the
//! kernel (evdev and uinput), not through the display server.

pub mod capture;
pub mod hook;
pub mod inject;
pub mod tray;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use phonon_hotkey::autostop::{self, Decision};
use phonon_hotkey::Action;

use crate::fetch;
use crate::history;
use crate::manifest;
use crate::paths;
use crate::pipeline::Engine;

/// A key action carries the number of the dictation the indicator lit for its press (`indicator::press`).
enum Event {
    Key(Action, u64),
    Menu(tray::Command),
}

/// Set once the startup check passes; the key thread drops presses until then (as on Windows), so a
/// press during the first-run download neither lights the tray nor replays minutes later.
static READY: AtomicBool = AtomicBool::new(false);

/// While recording, the worker reads the recorder's file this often: the indicator's voice level
/// and the latched auto-stop both follow it.
const LEVEL_POLL: Duration = Duration::from_millis(40);
/// Between dictations (and for the tray's "recording 3.5 s"), the worker's slower beat.
const IDLE_POLL: Duration = Duration::from_millis(500);

/// The udev rule the .deb installs (linux/70-phonon.rules): the signed-in user may read keyboards and
/// write /dev/uinput, through logind's uaccess, so no group change and no logging out.
pub const UDEV_RULES: &str = include_str!("../../linux/70-phonon.rules");

/// The one-time root setup for a copy installed without the package.
pub fn setup_script() -> String {
    let rules: String = UDEV_RULES
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| format!("{l}\n"))
        .collect();
    format!(
        "sudo tee /etc/udev/rules.d/70-phonon.rules > /dev/null <<'RULES'\n{rules}RULES\n\
         echo uinput | sudo tee /etc/modules-load.d/phonon-uinput.conf > /dev/null\n\
         sudo modprobe uinput && sudo udevadm control --reload && sudo udevadm trigger"
    )
}

/// The command as the user installed it: `phonon` from the .deb, `phonon-win` from a bare download.
pub fn command_name() -> String {
    std::env::args()
        .next()
        .and_then(|a| {
            std::path::Path::new(&a)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "phonon".into())
}

/// What is missing for `run`, in words, or nothing.
pub fn missing_access() -> Vec<String> {
    let mut missing = Vec::new();
    let (readable, denied) = hook::devices();
    if readable.is_empty() {
        missing.push(format!(
            "cannot read any keyboard ({} found, permission denied)",
            denied.len()
        ));
    }
    if std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/uinput")
        .is_err()
    {
        missing.push("cannot write /dev/uinput (needed to paste)".into());
    }
    missing
}

pub fn setup() -> Result<()> {
    let missing = missing_access();
    if missing.is_empty() {
        println!(
            "Phonon has keyboard and paste access. Start it with `{} run`.",
            command_name()
        );
        return Ok(());
    }
    for m in &missing {
        println!("  missing: {m}");
    }
    println!(
        "\nRun this once (no need to log out), then start Phonon again:\n\n{}\n",
        setup_script()
    );
    Ok(())
}

/// Start the application. A fatal error is also shown as a notification: started from the desktop's
/// autostart, Phonon has no terminal, and an error only on stderr is invisible.
pub fn run() -> Result<()> {
    let result = app();
    if let Err(error) = &result {
        tray::notify("Phonon stopped", &format!("{error:#}"), true);
    }
    result
}

fn app() -> Result<()> {
    let (events, inbox) = channel::<Event>();
    let (menu_tx, menu_rx) = channel::<tray::Command>();
    tray::create(menu_tx);
    refresh_history();
    {
        let events = events.clone();
        std::thread::spawn(move || {
            while let Ok(c) = menu_rx.recv() {
                if events.send(Event::Menu(c)).is_err() {
                    break;
                }
            }
        });
    }

    let missing = missing_access();
    if !missing.is_empty() {
        let text = format!(
            "{}. Run `{} setup` in a terminal for the one-time fix.",
            missing.join("; "),
            command_name()
        );
        anyhow::bail!("Phonon needs keyboard access: {text}");
    }
    inject::prepare()?;
    let (key_tx, key_rx) = channel::<Action>();
    let (name, code) = hook::configured_key();
    hook::install(code, key_tx)?;
    std::thread::spawn(move || {
        let mut last_hint: Option<Instant> = None;
        // the dictation the last press lit; its key-up releases that one and no other
        let mut held = 0;
        while let Ok(a) = key_rx.recv() {
            if !READY.load(Ordering::Acquire) {
                let due = last_hint.is_none_or(|t| t.elapsed() >= Duration::from_secs(15));
                if matches!(a, Action::Start) && due {
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
            // The indicator moves on the key itself, on this thread, not after the worker has
            // started the recorder: the press must be felt within a frame.
            match a {
                Action::Start => held = crate::indicator::press(),
                Action::Stop => crate::indicator::release(held),
                Action::None => {}
            }
            if events.send(Event::Key(a, held)).is_err() {
                break;
            }
        }
    });

    let pending: Vec<_> = manifest::ALL
        .iter()
        .filter(|c| !fetch::installed(c))
        .collect();
    if !pending.is_empty() {
        let total: u64 = pending.iter().map(|c| c.download_bytes()).sum();
        tray::notify(
            "Phonon is setting up",
            &format!(
                "Downloading {} of speech and correction models. Phonon works offline afterwards.",
                manifest::human_bytes(total)
            ),
            false,
        );
    }
    fetch::ensure_all(&mut |p| tray::set_status(&p.line()))
        .context("download the speech and correction models")?;
    tray::set_status("loading the correction model");
    let mut engine = Engine::start(crate::backends::DEFAULT_MANIFEST)?;
    tray::set_status("running the startup check");
    let wav = paths::data_root().join("startup.wav");
    std::fs::write(&wav, crate::STARTUP_WAV).context("write the startup fixture")?;
    let check = crate::startup_check_with_fallback(&mut engine, &wav, false)?;
    eprintln!("phonon: startup check passed: {}", check.polished);
    crate::settings::record_engine(&engine.route());

    let mut idle = format!("ready. Hold {name} to dictate");
    tray::set_status(&idle);
    READY.store(true, Ordering::Release);
    tray::notify(
        "Phonon is ready",
        // one line in a GNOME notification (about 45 characters): the longer
        // "or tap it to start and tap again to stop" was cut off mid-word
        &format!("Hold {name} to dictate, or tap to toggle."),
        false,
    );
    crate::update::start(|found| {
        tray::notify(
            &format!("Phonon {} is out", found.title),
            "Download it from the tray menu.",
            false,
        );
        tray::refresh();
    });

    let mut recording: Option<capture::Recording> = None;
    // The indicator's number for the dictation `recording` belongs to: every indicator call for it
    // names it, so a dictation that finishes after the next key press leaves that one's light alone.
    let mut lit: u64 = 0;
    let mut settings_watch = crate::settings::Watch::of(crate::settings::FILE);
    let _ = settings_watch.changed_in(&paths::data_root());
    // the settings window can clear the history; the tray's recent list follows
    let mut history_watch = crate::settings::Watch::of(history::LOG_FILE);
    let _ = history_watch.changed_in(&paths::data_root());
    // The last key, menu or recording activity: a long dictation's recogniser memory is given
    // back only after `asr::SHRINK_AFTER_IDLE` without any (`Engine::shrink_speech_model`).
    let mut last_active = Instant::now();
    // the last slow beat: status text, settings and history watches
    let mut last_beat = Instant::now();
    loop {
        let wait = if recording.is_some() {
            LEVEL_POLL
        } else {
            IDLE_POLL
        };
        let event = match inbox.recv_timeout(wait) {
            Ok(e) => {
                last_active = Instant::now();
                e
            }
            Err(RecvTimeoutError::Timeout) => {
                let beat = recording.is_none() || last_beat.elapsed() >= IDLE_POLL;
                if beat {
                    last_beat = Instant::now();
                }
                if recording.is_none() && last_active.elapsed() >= crate::asr::SHRINK_AFTER_IDLE {
                    engine.shrink_speech_model();
                }
                // The settings window writes settings.json; a new key applies here,
                // between dictations. Mode, history and the clipboard rule are read
                // at each dictation.
                if recording.is_none() && beat {
                    let root = paths::data_root();
                    let settings_changed = settings_watch.changed_in(&root);
                    if settings_changed {
                        if let Some(label) = hook::follow_settings() {
                            idle = format!("ready. Hold {label} to dictate");
                            tray::set_status(&idle);
                        }
                        tray::set_mode(crate::mode::current());
                    }
                    if history_watch.changed_in(&root) || settings_changed {
                        refresh_history();
                    }
                }
                if let Some(r) = &mut recording {
                    last_active = Instant::now();
                    // every LEVEL_POLL: also feeds the indicator's voice level (capture.rs)
                    match r.auto_stop(hook::latched()) {
                        Decision::KeepRecording => {
                            if beat {
                                tray::set_status(&format!("recording {:.1} s", r.seconds()))
                            }
                        }
                        // The key is up (it latched), so the next press must start, not
                        // stop; then the same stop the next tap would make.
                        Decision::Commit => {
                            eprintln!(
                                "phonon: latched dictation silent for {} s after speech; stopping",
                                autostop::SILENCE_AFTER_SPEECH_SECONDS
                            );
                            hook::reset_latch();
                            finish(&engine, recording.take(), lit, &idle);
                        }
                        Decision::Discard => {
                            eprintln!(
                                "phonon: latched dictation heard no speech in {} s; discarding",
                                autostop::NO_SPEECH_SECONDS
                            );
                            hook::reset_latch();
                            if let Some(r) = recording.take() {
                                r.discard();
                            }
                            crate::indicator::dropped(lit);
                            tray::set_status(&idle);
                        }
                    }
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
                let _ = std::process::Command::new("xdg-open")
                    .arg(paths::data_root())
                    .spawn();
            }
            Event::Menu(tray::Command::Toggle) => {
                hook::reset_latch();
                if recording.is_some() {
                    finish(&engine, recording.take(), lit, &idle);
                } else {
                    lit = crate::indicator::press();
                    recording = begin(lit);
                }
            }
            // The key thread lit this dictation at the press; the worker never lights it again, so
            // a Start that waited here behind a slow dictation, its key long up, stays processing.
            Event::Key(Action::Start, dictation) => {
                // a press during a recording (a menu start, then the key) takes its light over
                lit = dictation;
                if recording.is_none() {
                    recording = begin(lit);
                }
            }
            Event::Key(Action::Stop, dictation) => {
                // with no recording (the microphone failed to open), the key's own dictation
                let dictation = if recording.is_some() { lit } else { dictation };
                finish(&engine, recording.take(), dictation, &idle)
            }
            Event::Key(Action::None, _) => {}
            Event::Menu(tray::Command::ToggleHistory) => toggle_history(),
            // The recorder is still writing the active recording's file; leave it.
            Event::Menu(tray::Command::ClearHistory) => {
                clear_history(recording.as_ref().map(|r| r.path()))
            }
            Event::Menu(tray::Command::CopyRecent(text)) => copy_recent(text),
        }
    }
    Ok(())
}

/// Whether the desktop session is locked (logind's LockedHint, which GNOME, KDE and other lockers set).
/// The key reader sees keys on the lock screen too: on 2026-11-06 a VM's screen locked during a test and
/// Phonon typed eight dictations into the password field. While locked, Phonon neither records nor pastes.
/// Unknown (no logind) counts as unlocked.
fn session_locked() -> bool {
    // the user's graphical session first: a Phonon started from a terminal or ssh inherits that session's id
    let uid = unsafe { libc::getuid() }.to_string();
    let display = std::process::Command::new("loginctl")
        .args(["show-user", &uid, "-p", "Display", "--value"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    let session = display.or_else(|| std::env::var("XDG_SESSION_ID").ok());
    let Some(session) = session.filter(|s| !s.is_empty()) else {
        return false;
    };
    std::process::Command::new("loginctl")
        .args(["show-session", &session, "-p", "LockedHint", "--value"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "yes")
        .unwrap_or(false)
}

/// Start the recorder for dictation `lit` (the indicator's number), or fade its light.
fn begin(lit: u64) -> Option<capture::Recording> {
    if session_locked() {
        eprintln!("phonon: screen is locked; not recording");
        crate::indicator::dropped(lit);
        return None;
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    match capture::start(&paths::recordings().join(format!("{stamp}.wav"))) {
        Ok(r) => {
            tray::set_status(&format!("recording from {}", r.device()));
            Some(r)
        }
        Err(error) => {
            crate::indicator::dropped(lit);
            tray::notify("Phonon cannot hear you", &format!("{error:#}"), true);
            None
        }
    }
}

/// End dictation `lit` (the indicator's number for it): transcribe, correct, insert.
fn finish(engine: &Engine, recording: Option<capture::Recording>, lit: u64, idle: &str) {
    // Whatever way this returns, a panic included, the light does not stay on (a no-op once the
    // dictation was pasted or dropped, or once a newer dictation owns the indicator).
    let light = crate::indicator::Lit(lit);
    let Some(recording) = recording else {
        return;
    };
    // Processing starts at key-up. The key thread already released this dictation; a menu stop, an
    // auto-stop, or a key stopping a menu-started dictation did not come through it. Only this
    // dictation's listening can end here, never a newer one's.
    crate::indicator::release(lit);
    let key_up = std::time::Instant::now();
    let seconds = recording.seconds();
    // The recorder writes this file as it records, so a long dictation survives
    // a crash; with local history off it is deleted once the text is in.
    let wav = recording.path().to_path_buf();
    tray::set_status("transcribing");
    let outcome = recording.finish().and_then(|wav| engine.dictate(&wav));
    match &outcome {
        Ok(result) => {
            eprintln!("phonon: {}", result.summary());
            if result.polished.trim().is_empty() {
                crate::indicator::dropped(lit);
                tray::set_status("nothing was said");
            } else {
                // Before the paste, so the tray has it whatever happens next,
                // including a screen that locked while recording.
                tray::set_last(Some(result.polished.trim().to_string()));
                if session_locked() {
                    eprintln!("phonon: screen locked while recording; text not inserted");
                    crate::indicator::dropped(lit);
                } else {
                    let text = format!("{} ", result.polished.trim());
                    let after = crate::settings::Settings::load().clipboard_after_paste();
                    match inject::insert(&text, after) {
                        Ok(path) => {
                            crate::indicator::dropped(lit);
                            eprintln!(
                                "phonon: {path} {} chars, key-up to insert {} ms",
                                text.len(),
                                key_up.elapsed().as_millis()
                            )
                        }
                        Err(error) => {
                            // the error names the step that failed, never the text
                            eprintln!("phonon: could not insert the text: {error:#}");
                            keep_unpasted(result.polished.trim(), lit);
                        }
                    }
                }
            }
        }
        Err(error) => {
            crate::indicator::dropped(lit);
            tray::notify(
                "Phonon could not transcribe that",
                &format!("{error:#}"),
                true,
            )
        }
    }
    // Whatever path got here, never leave the core lit (a no-op once pasted or dropped).
    drop(light);
    if history::enabled() {
        let (raw, final_text) = match &outcome {
            Ok(result) => (result.raw.as_str(), result.polished.trim()),
            Err(_) => ("", ""),
        };
        let entry = history::Entry::new(history::now_ms(), seconds, raw, final_text, &wav);
        if let Err(error) = history::append(&entry) {
            eprintln!("phonon: could not save the dictation to history: {error:#}");
        }
        refresh_history();
    } else if !history::keep_wav(false, outcome.is_ok(), seconds) {
        history::remove_if_present(&wav);
    }
    tray::set_status(idle);
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

/// No confirmation dialog: a tray menu has nowhere to put one, and the item
/// says what it does. `active` is a recording in progress, which stays.
fn clear_history(active: Option<&std::path::Path>) {
    match history::clear(active) {
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

/// A dictation that could not be pasted stays on the clipboard and in the
/// tray's "Copy last dictation", and the notification says where it is.
fn keep_unpasted(text: &str, lit: u64) {
    crate::indicator::dropped(lit);
    match inject::copy(text) {
        Ok(()) => tray::notify_private(
            crate::pipeline::NOT_PASTED_TITLE,
            crate::pipeline::NOT_PASTED_BODY,
            false,
        ),
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
