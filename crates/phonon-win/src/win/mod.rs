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
pub mod tray;

use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::time::Duration;

use anyhow::{Context, Result};
use phonon_hotkey::Action;

use crate::fetch;
use crate::manifest;
use crate::paths;
use crate::pipeline::Engine;

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

    let (events, inbox) = channel::<Event>();

    let (menu_tx, menu_rx) = channel::<tray::Command>();
    tray::create(menu_tx).context("create the tray icon")?;

    let (key_tx, key_rx) = channel::<Action>();
    let (name, code) = hook::configured_key();
    hook::install(code, key_tx).context("install the keyboard hook")?;
    eprintln!("phonon: dictation key is {name}");

    // Two channels feed one worker; forward both.
    forward(key_rx, events.clone(), Event::Key);
    forward(menu_rx, events.clone(), Event::Menu);

    let worker = std::thread::spawn(move || {
        if let Err(error) = work(inbox) {
            eprintln!("phonon: {error:#}");
            tray::notify("Phonon stopped", &format!("{error:#}"), true);
        }
        tray::post_quit();
    });

    tray::run_message_loop();
    hook::uninstall();
    tray::destroy();
    let _ = worker.join();
    Ok(())
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

/// First-run download, warmup, then the dictation loop.
fn work(inbox: Receiver<Event>) -> Result<()> {
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
    let engine = Engine::start()?;

    // The same readiness gate macOS applies: prove the real audio path before
    // telling the user Phonon is ready.
    tray::set_status("running the startup check");
    match startup_check(&engine) {
        Ok(text) => eprintln!("phonon: startup check passed: {text}"),
        Err(error) => {
            tray::notify("Phonon startup check failed", &format!("{error:#}"), true);
            return Err(error);
        }
    }

    let (name, _) = hook::configured_key();
    let idle = format!("ready. Hold {name} to dictate");
    tray::set_status(&idle);
    tray::notify(
        "Phonon is ready",
        &format!("Hold {name} to dictate. Double-tap it to latch."),
        false,
    );

    let mut recording: Option<capture::Recording> = None;
    loop {
        let event = match inbox.recv_timeout(Duration::from_millis(500)) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => {
                if let Some(active) = &recording {
                    tray::set_status(&format!("recording {:.1} s", active.seconds()));
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        };
        match event {
            Event::Menu(tray::Command::Quit) => break,
            Event::Menu(tray::Command::OpenFolder) => {
                let root = paths::data_root();
                let _ = std::process::Command::new("explorer").arg(&root).spawn();
            }
            Event::Menu(tray::Command::Toggle) => {
                // The menu and the key drive the same two states. A menu stop
                // must also clear a latch, or the next key press would restart.
                hook::reset_latch();
                if recording.is_some() {
                    finish(&engine, recording.take(), &idle);
                } else {
                    recording = begin();
                }
            }
            Event::Key(Action::Start) => {
                if recording.is_none() {
                    recording = begin();
                }
            }
            Event::Key(Action::Stop) => {
                finish(&engine, recording.take(), &idle);
            }
            Event::Key(Action::None) => {}
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
            tray::notify("Phonon cannot hear you", &format!("{error:#}"), true);
            None
        }
    }
}

fn finish(engine: &Engine, recording: Option<capture::Recording>, idle: &str) {
    let Some(recording) = recording else { return };
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_millis())
        .unwrap_or(0);
    let path = paths::recordings().join(format!("{stamp}.wav"));
    tray::set_status("transcribing");
    let outcome = recording
        .finish(&path)
        .and_then(|wav| engine.run(&wav).map(|result| (wav, result)));
    match outcome {
        Ok((_, result)) => {
            eprintln!("phonon: {}", result.summary());
            if result.polished.trim().is_empty() {
                tray::set_status("nothing was said");
            } else {
                match inject::insert(&result.polished) {
                    Ok(path) => tray::set_status(&format!("inserted ({path})")),
                    Err(error) => {
                        tray::notify(
                            "Phonon could not insert the text",
                            &format!("{error:#}"),
                            true,
                        );
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
    tray::set_status(idle);
}
