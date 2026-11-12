//! Linux's dictation capsule, in a separate process like the settings window.
//!
//! The recorder supplies a bounded waveform over stdin, never a second mic
//! stream, transcript, or file path. A latest-frame mailbox keeps a slow or
//! failed graphics driver off the dictation thread. No helper remains at idle.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{bail, Context, Result};
use eframe::egui::{self, pos2, vec2, Color32, Stroke};
use serde::{Deserialize, Serialize};

pub const POINTS: usize = 24;
const MAX_FRAME: usize = 4096;
const TITLE: &str = "Phonon dictation";
const SIZE: egui::Vec2 = vec2(256.0, 48.0);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case", deny_unknown_fields)]
enum Frame {
    Listening { wave: [f32; POINTS] },
    Transcribing {},
}

impl Frame {
    fn valid(&self) -> bool {
        match self {
            Self::Listening { wave } => wave.iter().all(|n| n.is_finite() && n.abs() <= 1.0),
            Self::Transcribing {} => true,
        }
    }
}

/// Signed peaks from each bin of the actual PCM, not an animated sine wave.
/// Square-root compression makes quiet speech visible without clipping louder
/// speech. Near-silent samples remain exactly zero.
pub fn waveform(pcm: &[i16]) -> [f32; POINTS] {
    let mut wave = [0.0; POINTS];
    for (i, point) in wave.iter_mut().enumerate() {
        let from = i * pcm.len() / POINTS;
        let to = (i + 1) * pcm.len() / POINTS;
        let sample = pcm[from..to]
            .iter()
            .max_by_key(|s| s.unsigned_abs())
            .copied()
            .unwrap_or(0);
        let value = sample as f32 / 32768.0;
        if value.abs() > 0.002 {
            *point = value.signum() * value.abs().sqrt();
        }
    }
    wave
}

struct Mailbox {
    revision: u64,
    frame: Option<Frame>,
}

impl Mailbox {
    fn replace(&mut self, frame: Option<Frame>) -> bool {
        if self.frame == frame {
            return false;
        }
        self.frame = frame;
        self.revision = self.revision.wrapping_add(1);
        true
    }
}

type Mail = Arc<(Mutex<Mailbox>, Condvar)>;

pub struct Indicator {
    enabled: bool,
    child: Option<Child>,
    mail: Option<Mail>,
}

impl Indicator {
    pub fn new() -> Self {
        let disabled = std::env::var("PHONON_INDICATOR")
            .map(|s| {
                matches!(
                    s.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off"
                )
            })
            .unwrap_or(false);
        Self {
            enabled: !disabled && std::env::var_os("DISPLAY").is_some_and(|s| !s.is_empty()),
            child: None,
            mail: None,
        }
    }

    pub fn begin(&mut self) {
        self.hide();
        if !self.enabled {
            return;
        }
        if let Err(error) = self.spawn() {
            eprintln!("phonon: indicator unavailable (dictation continues): {error}");
        }
    }

    fn spawn(&mut self) -> Result<()> {
        let mut child = Command::new(std::env::current_exe()?)
            .arg("indicator")
            // xdg-shell cannot position an overlay or guarantee click-through.
            // Only this UI process uses X11/XWayland, never the tray or recorder.
            .env_remove("WAYLAND_DISPLAY")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .context("start the indicator helper")?;
        let mut input = child.stdin.take().context("open the indicator pipe")?;
        let mail = Arc::new((
            Mutex::new(Mailbox {
                revision: 0,
                frame: Some(Frame::Listening {
                    wave: [0.0; POINTS],
                }),
            }),
            Condvar::new(),
        ));
        let worker = mail.clone();
        std::thread::spawn(move || {
            let (lock, wake) = &*worker;
            let mut sent = u64::MAX;
            loop {
                let frame = {
                    let state = lock.lock().unwrap();
                    let state = wake.wait_while(state, |s| s.revision == sent).unwrap();
                    sent = state.revision;
                    state.frame.clone()
                };
                let Some(frame) = frame else { break };
                if serde_json::to_writer(&mut input, &frame).is_err()
                    || input.write_all(b"\n").is_err()
                {
                    break;
                }
            }
        });
        self.child = Some(child);
        self.mail = Some(mail);
        Ok(())
    }

    fn publish(&self, frame: Option<Frame>) {
        if let Some(mail) = &self.mail {
            let (lock, wake) = &**mail;
            if lock.lock().unwrap().replace(frame) {
                wake.notify_one();
            }
        }
    }

    pub fn listening(&self, wave: [f32; POINTS]) {
        self.publish(Some(Frame::Listening { wave }));
    }

    pub fn transcribing(&self) {
        self.publish(Some(Frame::Transcribing {}));
    }

    pub fn hide(&mut self) {
        self.publish(None);
        self.mail = None;
        if let Some(mut child) = self.child.take() {
            // Also handles a graphics driver stuck in its event loop.
            let _ = child.kill();
            std::thread::spawn(move || {
                let _ = child.wait();
            });
        }
    }
}

impl Drop for Indicator {
    fn drop(&mut self) {
        self.hide();
    }
}

fn read_frame(input: &mut impl BufRead) -> Result<Option<Frame>> {
    let mut bytes = Vec::new();
    let count = input
        .take((MAX_FRAME + 1) as u64)
        .read_until(b'\n', &mut bytes)?;
    if count == 0 {
        return Ok(None);
    }
    if count > MAX_FRAME || !bytes.ends_with(b"\n") {
        bail!("invalid indicator frame size");
    }
    let frame: Frame = serde_json::from_slice(&bytes).context("invalid indicator frame")?;
    if !frame.valid() {
        bail!("invalid indicator waveform");
    }
    Ok(Some(frame))
}

/// Run only in the internal `indicator` child. EOF closes the capsule even
/// when the resident process crashed; no persistent state is written.
pub fn run() -> Result<()> {
    let mut input = BufReader::new(std::io::stdin());
    let Some(first) = read_frame(&mut input)? else {
        return Ok(());
    };
    std::env::remove_var("WAYLAND_DISPLAY");
    let shared = Arc::new(Mutex::new(Some(first)));
    let state = shared.clone();
    let viewport = egui::ViewportBuilder::default()
        .with_title(TITLE)
        .with_app_id("phonon-indicator")
        .with_inner_size(SIZE)
        .with_decorations(false)
        .with_resizable(false)
        .with_transparent(true)
        .with_active(false)
        .with_visible(false)
        .with_taskbar(false)
        .with_always_on_top()
        .with_mouse_passthrough(true)
        .with_window_type(egui::X11WindowType::Notification)
        .with_override_redirect(true);
    let options = eframe::NativeOptions {
        viewport,
        persist_window: false,
        ..Default::default()
    };
    eframe::run_native(
        TITLE,
        options,
        Box::new(move |cc| {
            cc.egui_ctx
                .all_styles_mut(|style| style.visuals.panel_fill = Color32::TRANSPARENT);
            if let Some(screen) = pointer_screen() {
                cc.egui_ctx
                    .send_viewport_cmd(egui::ViewportCommand::OuterPosition(placement(
                        screen,
                        cc.egui_ctx.pixels_per_point(),
                    )));
            }
            let context = cc.egui_ctx.clone();
            let reader = state.clone();
            std::thread::spawn(move || {
                while let Ok(Some(frame)) = read_frame(&mut input) {
                    *reader.lock().unwrap() = Some(frame);
                    context.request_repaint();
                }
                *reader.lock().unwrap() = None;
                context.send_viewport_cmd(egui::ViewportCommand::Close);
                context.request_repaint();
            });
            Ok(Box::new(Capsule {
                state,
                shown: false,
            }))
        }),
    )
    .map_err(|error| anyhow::anyhow!("open indicator: {error}"))?;
    Ok(())
}

struct Capsule {
    state: Arc<Mutex<Option<Frame>>>,
    shown: bool,
}

impl eframe::App for Capsule {
    fn clear_color(&self, _: &egui::Visuals) -> [f32; 4] {
        [0.0; 4]
    }

    fn logic(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        if self.state.lock().unwrap().is_none() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        // eframe skips `ui` while hidden. Reveal from logic, after placement,
        // or the initially transparent window can stay invisible forever.
        if !self.shown {
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
            ctx.request_repaint();
            self.shown = true;
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        let Some(frame) = self.state.lock().unwrap().clone() else {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        };
        let painter = ui.painter();
        let rect = egui::Rect::from_min_size(pos2(0.0, 0.0), SIZE).shrink(1.0);
        let ember = Color32::from_rgb(241, 151, 89);
        let ink = Color32::from_rgb(244, 237, 229);
        painter.rect_filled(rect, 24, Color32::from_rgb(26, 21, 18));
        painter.rect_stroke(rect, 24, Stroke::new(1.0, ember), egui::StrokeKind::Inside);
        painter.circle_filled(pos2(23.0, 24.0), 13.0, Color32::from_rgb(52, 35, 24));
        painter.text(
            pos2(23.0, 23.0),
            egui::Align2::CENTER_CENTER,
            "φ",
            egui::FontId::proportional(21.0),
            ember,
        );
        let label = match &frame {
            Frame::Listening { .. } => "Listening",
            Frame::Transcribing {} => "Transcribing",
        };
        painter.text(
            pos2(45.0, 24.0),
            egui::Align2::LEFT_CENTER,
            label,
            egui::FontId::proportional(13.0),
            ink,
        );
        match frame {
            Frame::Listening { wave } => {
                let knots: Vec<_> = wave
                    .iter()
                    .enumerate()
                    .map(|(i, sample)| {
                        let t = i as f32 / (POINTS - 1) as f32;
                        pos2(
                            158.0 + 80.0 * t,
                            24.0 - sample * 11.0 * (t * std::f32::consts::PI).sin(),
                        )
                    })
                    .collect();
                painter.add(egui::Shape::line(smooth(&knots), Stroke::new(1.6, ink)));
            }
            Frame::Transcribing {} => {
                painter.text(
                    pos2(198.0, 22.0),
                    egui::Align2::CENTER_CENTER,
                    "…",
                    egui::FontId::proportional(22.0),
                    ember,
                );
            }
        }
    }
}

fn smooth(knots: &[egui::Pos2]) -> Vec<egui::Pos2> {
    let mut points = Vec::with_capacity((knots.len() - 1) * 4 + 1);
    for i in 0..knots.len() - 1 {
        let before = knots[i.saturating_sub(1)].to_vec2();
        let from = knots[i].to_vec2();
        let to = knots[i + 1].to_vec2();
        let after = knots[(i + 2).min(knots.len() - 1)].to_vec2();
        for step in 0..4 {
            let t = step as f32 / 4.0;
            let value = (2.0 * from
                + (to - before) * t
                + (2.0 * before - 5.0 * from + 4.0 * to - after) * t * t
                + (-before + 3.0 * from - 3.0 * to + after) * t * t * t)
                * 0.5;
            points.push(value.to_pos2());
        }
    }
    points.push(*knots.last().unwrap());
    points
}

fn placement(screen: egui::Rect, scale: f32) -> egui::Pos2 {
    let screen = egui::Rect::from_min_max(screen.min / scale, screen.max / scale);
    pos2(
        screen.min.x + ((screen.width() - SIZE.x) / 2.0).max(0.0),
        (screen.max.y - SIZE.y - 64.0).max(screen.min.y),
    )
}

/// XRandR includes negative origins and the pointer's output. X11 is loaded
/// only by this child; these are already winit's runtime libraries.
fn pointer_screen() -> Option<egui::Rect> {
    use x11_dl::{xlib, xrandr};
    let x = xlib::Xlib::open().ok()?;
    // SAFETY: the display is local to this function; Xlib owns returned data,
    // which is bounded by the server's monitor count and freed before close.
    unsafe {
        let display = (x.XOpenDisplay)(std::ptr::null());
        if display.is_null() {
            return None;
        }
        let root = (x.XDefaultRootWindow)(display);
        let (mut r, mut child, mut px, mut py, mut wx, mut wy, mut mask) = (0, 0, 0, 0, 0, 0, 0);
        (x.XQueryPointer)(
            display, root, &mut r, &mut child, &mut px, &mut py, &mut wx, &mut wy, &mut mask,
        );
        let mut result = None;
        if let Ok(randr) = xrandr::Xrandr::open() {
            let mut count = 0;
            let monitors = (randr.XRRGetMonitors)(display, root, 1, &mut count);
            if !monitors.is_null() {
                if (1..=256).contains(&count) {
                    let monitors_slice = std::slice::from_raw_parts(monitors, count as usize);
                    let chosen = monitors_slice
                        .iter()
                        .find(|m| {
                            px >= m.x && py >= m.y && px < m.x + m.width && py < m.y + m.height
                        })
                        .or_else(|| monitors_slice.iter().find(|m| m.primary != 0))
                        .unwrap_or(&monitors_slice[0]);
                    result = Some(egui::Rect::from_min_size(
                        pos2(chosen.x as f32, chosen.y as f32),
                        vec2(chosen.width as f32, chosen.height as f32),
                    ));
                }
                (randr.XRRFreeMonitors)(monitors);
            }
        }
        if result.is_none() {
            let screen = (x.XDefaultScreen)(display);
            result = Some(egui::Rect::from_min_size(
                pos2(0.0, 0.0),
                vec2(
                    (x.XDisplayWidth)(display, screen) as f32,
                    (x.XDisplayHeight)(display, screen) as f32,
                ),
            ));
        }
        (x.XCloseDisplay)(display);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_and_the_noise_floor_are_flat() {
        assert_eq!(waveform(&[]), [0.0; POINTS]);
        assert_eq!(waveform(&[0; 1600]), [0.0; POINTS]);
        assert_eq!(waveform(&[32; 1600]), [0.0; POINTS]);
    }

    #[test]
    fn louder_pcm_is_larger_without_flat_topping() {
        let soft = waveform(&[1000; 1600]);
        let loud = waveform(&[16000; 1600]);
        assert!(loud[12] > soft[12] * 2.0);
        assert!(loud[12] < 1.0);
        assert_eq!(waveform(&[i16::MIN; 1600]), [-1.0; POINTS]);
    }

    #[test]
    fn waveform_preserves_real_signed_sample_peaks() {
        let pcm: Vec<_> = (0..POINTS)
            .flat_map(|i| [if i % 2 == 0 { 4096 } else { -4096 }; 20])
            .collect();
        let wave = waveform(&pcm);
        for (i, point) in wave.iter().enumerate() {
            assert_eq!(point.is_sign_positive(), i % 2 == 0);
        }
    }

    #[test]
    fn mailbox_coalesces_frames_but_keeps_the_final_state() {
        let mut mail = Mailbox {
            revision: 0,
            frame: None,
        };
        assert!(mail.replace(Some(Frame::Listening {
            wave: [0.0; POINTS]
        })));
        assert!(!mail.replace(Some(Frame::Listening {
            wave: [0.0; POINTS]
        })));
        for _ in 0..1000 {
            mail.replace(Some(Frame::Listening {
                wave: [0.5; POINTS],
            }));
        }
        mail.replace(Some(Frame::Transcribing {}));
        assert_eq!(mail.frame, Some(Frame::Transcribing {}));
        mail.replace(None);
        assert_eq!(mail.frame, None);
    }

    #[test]
    fn protocol_is_bounded_and_rejects_unexpected_data() {
        assert!(read_frame(&mut b"".as_slice()).unwrap().is_none());
        assert!(
            read_frame(&mut b"{\"phase\":\"transcribing\"}\n".as_slice())
                .unwrap()
                .is_some()
        );
        assert!(
            read_frame(&mut b"{\"phase\":\"transcribing\",\"text\":\"private\"}\n".as_slice())
                .is_err()
        );
        assert!(read_frame(&mut vec![b'x'; MAX_FRAME + 1].as_slice()).is_err());
        assert!(read_frame(&mut b"{\"phase\":\"listening\",\"wave\":[]}\n".as_slice()).is_err());
        let invalid = Frame::Listening {
            wave: [2.0; POINTS],
        };
        let mut bytes = serde_json::to_vec(&invalid).unwrap();
        bytes.push(b'\n');
        assert!(read_frame(&mut bytes.as_slice()).is_err());
    }

    #[test]
    fn placement_handles_scaled_and_negative_monitor_origins() {
        let screen = egui::Rect::from_min_size(pos2(-2560.0, 0.0), vec2(2560.0, 1440.0));
        assert_eq!(placement(screen, 2.0), pos2(-768.0, 608.0));
    }

    #[test]
    fn smooth_curve_is_continuous_and_flat_for_silence() {
        let knots: Vec<_> = (0..POINTS).map(|i| pos2(i as f32, 0.0)).collect();
        let curve = smooth(&knots);
        assert_eq!(curve.len(), (POINTS - 1) * 4 + 1);
        assert!(curve.iter().all(|p| p.y == 0.0));
        assert!(curve.windows(2).all(|p| p[1].x > p[0].x));
    }

    #[test]
    fn a_hidden_capsule_is_revealed_without_waiting_for_a_ui_pass() {
        let ctx = egui::Context::default();
        let mut capsule = Capsule {
            state: Arc::new(Mutex::new(Some(Frame::Listening {
                wave: [0.0; POINTS],
            }))),
            shown: false,
        };
        let output = ctx.run_logic(&Default::default(), |ctx| {
            eframe::App::logic(&mut capsule, ctx, &mut eframe::Frame::_new_kittest());
        });
        assert!(output.viewport_commands[&egui::ViewportId::ROOT]
            .contains(&egui::ViewportCommand::Visible(true)));
        *capsule.state.lock().unwrap() = None;
        let output = ctx.run_logic(&Default::default(), |ctx| {
            eframe::App::logic(&mut capsule, ctx, &mut eframe::Frame::_new_kittest());
        });
        let commands = &output.viewport_commands[&egui::ViewportId::ROOT];
        assert!(commands.contains(&egui::ViewportCommand::Close));
        assert!(!commands.contains(&egui::ViewportCommand::Visible(true)));
    }

    #[test]
    fn disabled_indicator_has_no_process_or_persistent_state() {
        let mut indicator = Indicator {
            enabled: false,
            child: None,
            mail: None,
        };
        indicator.begin();
        indicator.listening([0.5; POINTS]);
        indicator.transcribing();
        indicator.hide();
        assert!(indicator.child.is_none());
        assert!(indicator.mail.is_none());
    }
}
