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

use std::sync::mpsc::{channel, RecvTimeoutError};
use std::time::Duration;

use anyhow::{Context, Result};
use phonon_hotkey::Action;

use crate::fetch;
use crate::manifest;
use crate::paths;
use crate::pipeline::Engine;

enum Event {
    Key(Action),
    Menu(tray::Command),
}

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
        while let Ok(a) = key_rx.recv() {
            if events.send(Event::Key(a)).is_err() {
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

    let idle = format!("ready. Hold {name} to dictate");
    tray::set_status(&idle);
    tray::notify(
        "Phonon is ready",
        &format!("Hold {name} to dictate, or tap it to start and tap again to stop."),
        false,
    );

    let mut recording: Option<capture::Recording> = None;
    loop {
        let event = match inbox.recv_timeout(Duration::from_millis(500)) {
            Ok(e) => e,
            Err(RecvTimeoutError::Timeout) => {
                if let Some(r) = &recording {
                    tray::set_status(&format!("recording {:.1} s", r.seconds()));
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        };
        match event {
            Event::Menu(tray::Command::Quit) => break,
            Event::Menu(tray::Command::OpenFolder) => {
                let _ = std::process::Command::new("xdg-open")
                    .arg(paths::data_root())
                    .spawn();
            }
            Event::Menu(tray::Command::Toggle) => {
                hook::reset_latch();
                if recording.is_some() {
                    finish(&engine, recording.take(), &idle);
                } else {
                    recording = begin();
                }
            }
            Event::Key(Action::Start) if recording.is_none() => recording = begin(),
            Event::Key(Action::Stop) => finish(&engine, recording.take(), &idle),
            Event::Key(_) => {}
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

fn begin() -> Option<capture::Recording> {
    if session_locked() {
        eprintln!("phonon: screen is locked; not recording");
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
            tray::notify("Phonon cannot hear you", &format!("{error:#}"), true);
            None
        }
    }
}

fn finish(engine: &Engine, recording: Option<capture::Recording>, idle: &str) {
    let Some(recording) = recording else { return };
    let key_up = std::time::Instant::now();
    tray::set_status("transcribing");
    match recording.finish().and_then(|wav| engine.dictate(&wav)) {
        Ok(result) => {
            eprintln!("phonon: {}", result.summary());
            if result.polished.trim().is_empty() {
                tray::set_status("nothing was said");
            } else if session_locked() {
                eprintln!("phonon: screen locked while recording; text not inserted");
            } else {
                let text = format!("{} ", result.polished.trim());
                match inject::insert(&text) {
                    Ok(path) => eprintln!(
                        "phonon: {path} {} chars, key-up to insert {} ms",
                        text.len(),
                        key_up.elapsed().as_millis()
                    ),
                    Err(error) => tray::notify(
                        "Phonon could not insert the text",
                        &format!("{error:#}"),
                        true,
                    ),
                }
            }
        }
        Err(error) => tray::notify(
            "Phonon could not transcribe that",
            &format!("{error:#}"),
            true,
        ),
    }
    tray::set_status(idle);
}
