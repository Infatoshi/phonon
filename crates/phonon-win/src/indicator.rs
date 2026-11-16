//! The dictation indicator's four stages (ignite, listening, processing, pasted), drawn into the tray icon.
//!
//! Windows and Linux have no floating bar: the tray icon is the whole interface. The icon keeps its
//! square slot; while a dictation is under way it shows the Mac bar's gradient in a capsule (black at
//! both ends, one warm light in the middle), and between dictations it is the normal Phonon icon and
//! nothing repaints.
//!
//! The gradient is the Mac capsule's and the iPhone's: black at 0.5-w and 0.5+w, colour at 0.5.
//!  1. Ignite, the instant the key goes down: full strength, falling to the live voice level in 0.18 s.
//!  2. Listening: w = 0.5, strength follows the smoothed voice level.
//!  3. Processing, from key-up until the text is delivered: over 1.2 s w narrows from 0.5 to 0.08 and the
//!     light warms and brightens; then the tight core holds, still.
//!  4. Pasted: over 0.6 s w opens from where it was back to 0.5 (sqrt of progress) while a light orange
//!     fades to nothing.
//!  5. Ended with nothing pasted: no burst, the core fades out in 0.2 s.
//!  6. Reduced motion: the width never moves; processing is a steady medium core, pasted a 0.3 s fade.
//!
//! `look` is the pure stage maths; `Machine` is the state the hooks drive; `start` runs the one thread
//! that turns frames into icons for the platform's tray (the sink).

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The Mac capsule's ember (0.42, 0.12, 0.035).
pub const BASE: [f32; 3] = [107.0, 31.0, 9.0];
/// What the light warms to while gathering.
pub const BRIGHT: [f32; 3] = [227.0, 101.0, 29.0];
/// The burst's light orange.
pub const ORANGE: [f32; 3] = [255.0, 170.0, 100.0];

pub const IGNITE_SECONDS: f32 = 0.18;
pub const GATHER_SECONDS: f32 = 1.2;
pub const BURST_SECONDS: f32 = 0.6;
pub const DROP_SECONDS: f32 = 0.2;
pub const REDUCED_FADE_SECONDS: f32 = 0.3;
pub const TIGHT_WIDTH: f32 = 0.08;
pub const WIDE_WIDTH: f32 = 0.5;
/// Reduced motion's processing core: steady, between wide and tight.
pub const REDUCED_WIDTH: f32 = 0.25;
pub const LEVEL_FLOOR: f32 = 0.08;

/// Where in a dictation the indicator is. The variants that end one carry what they fade from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stage {
    /// Key down until key up. Elapsed counts from the press.
    Listening,
    /// Key up until the text is delivered or the dictation ends. Elapsed counts from the release.
    Processing,
    /// The text went in. Elapsed counts from that moment; `from_width` is the gather's width then.
    Pasted { from_width: f32 },
    /// Ended with nothing pasted (empty, error, cancelled).
    Dropped { from_width: f32, from_strength: f32, from_heat: f32 },
}

/// What one frame looks like. The centre colour is `lerp(BASE, BRIGHT, heat) * strength` over black,
/// plus `ORANGE * burst`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    /// Half the width of the lit part, as a fraction of the whole (0.5 = the full width).
    pub half_width: f32,
    /// Opacity of the warm light over black.
    pub strength: f32,
    /// 0 = the base ember, 1 = the bright processing colour.
    pub heat: f32,
    /// Opacity of the burst's orange light.
    pub burst: f32,
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// The stage maths: stage, seconds since it began, smoothed voice level (0..1), reduced motion.
pub fn look(stage: Stage, elapsed: f32, level: f32, reduced: bool) -> Look {
    let elapsed = elapsed.max(0.0);
    let level = level.clamp(LEVEL_FLOOR, 1.0);
    match stage {
        Stage::Listening => {
            let ignite = (1.0 - elapsed / IGNITE_SECONDS).clamp(0.0, 1.0);
            // 1.0 at the press, the live level 0.18 s later, linearly between
            let energy = lerp(level, 1.0, ignite);
            Look { half_width: WIDE_WIDTH, strength: 0.50 + 0.38 * energy, heat: 0.0, burst: 0.0 }
        }
        Stage::Processing if reduced => {
            Look { half_width: REDUCED_WIDTH, strength: 0.80, heat: 0.5, burst: 0.0 }
        }
        Stage::Processing => {
            let k = (elapsed / GATHER_SECONDS).clamp(0.0, 1.0);
            Look {
                half_width: lerp(WIDE_WIDTH, TIGHT_WIDTH, k),
                strength: lerp(0.65, 1.0, k),
                heat: k,
                burst: 0.0,
            }
        }
        Stage::Pasted { from_width } if reduced => Look {
            half_width: from_width,
            strength: 0.0,
            heat: 0.0,
            burst: 1.0 - (elapsed / REDUCED_FADE_SECONDS).clamp(0.0, 1.0),
        },
        Stage::Pasted { from_width } => {
            let p = (elapsed / BURST_SECONDS).clamp(0.0, 1.0);
            Look {
                half_width: lerp(from_width, WIDE_WIDTH, p.sqrt()),
                strength: 0.0,
                heat: 0.0,
                burst: 1.0 - p,
            }
        }
        Stage::Dropped { from_width, from_strength, from_heat } => Look {
            half_width: from_width,
            strength: from_strength * (1.0 - (elapsed / DROP_SECONDS).clamp(0.0, 1.0)),
            heat: from_heat,
            burst: 0.0,
        },
    }
}

/// Whether a stage that ends (pasted, dropped) has finished and the icon is at rest again.
pub fn finished(stage: Stage, elapsed: f32, reduced: bool) -> bool {
    match stage {
        Stage::Pasted { .. } if reduced => elapsed >= REDUCED_FADE_SECONDS,
        Stage::Pasted { .. } => elapsed >= BURST_SECONDS,
        Stage::Dropped { .. } => elapsed >= DROP_SECONDS,
        Stage::Listening | Stage::Processing => false,
    }
}

/// A microphone's RMS (0..1 of full scale; speech sits around 0.02 to 0.2) as a 0..1 voice level.
pub fn level_from_rms(rms: f32) -> f32 {
    (rms * 7.0).clamp(0.0, 1.0).sqrt().max(LEVEL_FLOOR)
}

/// Fast attack, slow release, floor 0.08.
#[derive(Clone, Copy, Debug)]
pub struct Smoother {
    shown: f32,
}

impl Smoother {
    pub const ATTACK_SECONDS: f32 = 0.04;
    pub const RELEASE_SECONDS: f32 = 0.30;

    pub fn new() -> Self {
        Self { shown: LEVEL_FLOOR }
    }

    pub fn step(&mut self, target: f32, dt: f32) -> f32 {
        let target = target.clamp(LEVEL_FLOOR, 1.0);
        let tau = if target > self.shown { Self::ATTACK_SECONDS } else { Self::RELEASE_SECONDS };
        self.shown += (target - self.shown) * (1.0 - (-dt.max(0.0) / tau).exp());
        self.shown = self.shown.clamp(LEVEL_FLOOR, 1.0);
        self.shown
    }

    pub fn value(&self) -> f32 {
        self.shown
    }
}

impl Default for Smoother {
    fn default() -> Self {
        Self::new()
    }
}

/// The indicator's state, in seconds on any steady clock. Events call these from any thread;
/// each says what it does when it is not the right moment (then nothing happens).
#[derive(Clone, Copy, Debug, Default)]
pub struct Machine {
    active: Option<(Stage, f32)>,
}

impl Machine {
    pub fn is_rest(&self) -> bool {
        self.active.is_none()
    }

    pub fn stage(&self) -> Option<Stage> {
        self.active.map(|(stage, _)| stage)
    }

    /// Key down or tap. Lights at once, whatever the indicator was doing (a press while the last
    /// dictation is still processing starts the new one; that one's late paste is then ignored).
    pub fn press(&mut self, now: f32) {
        if !matches!(self.active, Some((Stage::Listening, _))) {
            self.active = Some((Stage::Listening, now));
        }
    }

    /// Key up or stop: listening becomes processing.
    pub fn release(&mut self, now: f32) {
        if matches!(self.active, Some((Stage::Listening, _))) {
            self.active = Some((Stage::Processing, now));
        }
    }

    /// The text was inserted: burst from wherever the gather is.
    pub fn pasted(&mut self, now: f32, reduced: bool) {
        if let Some((Stage::Processing, since)) = self.active {
            let from_width = look(Stage::Processing, now - since, LEVEL_FLOOR, reduced).half_width;
            self.active = Some((Stage::Pasted { from_width }, now));
        }
    }

    /// The dictation ended with nothing pasted: no burst, the core fades out.
    pub fn dropped(&mut self, now: f32, level: f32, reduced: bool) {
        if let Some((stage @ (Stage::Listening | Stage::Processing), since)) = self.active {
            let current = look(stage, now - since, level, reduced);
            self.active = Some((
                Stage::Dropped {
                    from_width: current.half_width,
                    from_strength: current.strength,
                    from_heat: current.heat,
                },
                now,
            ));
        }
    }

    /// The frame to draw now, or `None` at rest. A stage that ends hands back to rest here.
    pub fn frame(&mut self, now: f32, level: f32, reduced: bool) -> Option<Look> {
        let (stage, since) = self.active?;
        let elapsed = now - since;
        if finished(stage, elapsed, reduced) {
            self.active = None;
            return None;
        }
        Some(look(stage, elapsed, level, reduced))
    }
}

/// A frame as the tray wants it: straight-alpha RGBA, `size` x `size`, top row first.
pub fn render_rgba(look: Look, size: u32) -> Vec<u8> {
    let n = size as usize;
    let mut out = vec![0u8; n * n * 4];
    let size_f = size as f32;
    // the capsule: full width, half the height, round ends
    let half_h = size_f * 0.25;
    let (cx, cy) = (size_f / 2.0, size_f / 2.0);
    let core = [
        lerp(BASE[0], BRIGHT[0], look.heat) * look.strength,
        lerp(BASE[1], BRIGHT[1], look.heat) * look.strength,
        lerp(BASE[2], BRIGHT[2], look.heat) * look.strength,
    ];
    for y in 0..n {
        for x in 0..n {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            // signed distance to a capsule whose end circles sit half_h in from the edges
            let dx = ((px - cx).abs() - (size_f / 2.0 - half_h)).max(0.0);
            let sd = (dx * dx + (py - cy) * (py - cy)).sqrt() - half_h;
            let coverage = (0.5 - sd).clamp(0.0, 1.0);
            if coverage <= 0.0 {
                continue;
            }
            let u = px / size_f;
            let g = (1.0 - (u - 0.5).abs() / look.half_width.max(0.001)).max(0.0);
            let i = (y * n + x) * 4;
            for c in 0..3 {
                let v = (core[c] + ORANGE[c] * look.burst) * g;
                out[i + c] = v.clamp(0.0, 255.0).round() as u8;
            }
            out[i + 3] = (coverage * 255.0).round() as u8;
        }
    }
    out
}

/// What a platform's tray provides. `show(Some(rgba), size)` replaces the icon with a frame;
/// `show(None, _)` puts the normal icon back. `size` is the pixel size to render at.
pub struct Sink {
    pub size: fn() -> u32,
    pub show: fn(Option<&[u8]>, u32),
    /// Whether the system asks for reduced motion (read when a dictation ends and at start).
    pub reduced_motion: fn() -> bool,
}

struct Shared {
    machine: Mutex<Machine>,
    wake: Condvar,
    level: AtomicU32,
    reduced: AtomicBool,
    origin: Instant,
}

fn shared() -> &'static Shared {
    static SHARED: OnceLock<Shared> = OnceLock::new();
    SHARED.get_or_init(|| Shared {
        machine: Mutex::new(Machine::default()),
        wake: Condvar::new(),
        level: AtomicU32::new(LEVEL_FLOOR.to_bits()),
        reduced: AtomicBool::new(false),
        origin: Instant::now(),
    })
}

fn now() -> f32 {
    shared().origin.elapsed().as_secs_f32()
}

fn with_machine(f: impl FnOnce(&mut Machine, f32)) {
    let s = shared();
    if let Ok(mut machine) = s.machine.lock() {
        f(&mut machine, now());
    }
    s.wake.notify_all();
}

fn reduced() -> bool {
    shared().reduced.load(Ordering::Relaxed)
}

/// The dictation key went down (or the menu started a dictation). Returns at once; the
/// animation thread draws the next frame without waiting for the microphone or the models.
pub fn press() {
    with_machine(|m, t| m.press(t));
}

/// The key came up (or a stop): processing begins.
pub fn release() {
    with_machine(|m, t| m.release(t));
}

/// The text was inserted.
pub fn pasted() {
    let reduced = reduced();
    with_machine(|m, t| m.pasted(t, reduced));
}

/// The dictation ended with nothing pasted: empty, error, cancelled, kept on the clipboard.
pub fn dropped() {
    let reduced = reduced();
    let level = f32::from_bits(shared().level.load(Ordering::Relaxed));
    with_machine(|m, t| m.dropped(t, level, reduced));
}

/// Like `dropped`, for the end of a dictation's processing: a newer dictation already listening
/// (its key came down while this one finished) is left alone.
pub fn abandon() {
    let reduced = reduced();
    let level = f32::from_bits(shared().level.load(Ordering::Relaxed));
    with_machine(|m, t| {
        if m.stage() == Some(Stage::Processing) {
            m.dropped(t, level, reduced)
        }
    });
}

/// The microphone's RMS for the latest buffer (0..1 of full scale).
pub fn push_level(rms: f32) {
    shared().level.store(level_from_rms(rms).to_bits(), Ordering::Relaxed);
}

/// Start the animation thread. It sleeps until the first press and sleeps again at rest, so an idle
/// Phonon repaints nothing.
pub fn start(sink: Sink) {
    shared().reduced.store((sink.reduced_motion)(), Ordering::Relaxed);
    let _ = std::thread::Builder::new().name("indicator".into()).spawn(move || run(sink));
}

fn run(sink: Sink) {
    let s = shared();
    let mut smoother = Smoother::new();
    let mut last = now();
    let mut drawing = false;
    let frame_time = Duration::from_millis(33);
    loop {
        let Ok(guard) = s.machine.lock() else { return };
        let mut guard = match s.wake.wait_while(guard, |m| m.is_rest() && !drawing) {
            Ok(guard) => guard,
            Err(_) => return,
        };
        let t = now();
        let dt = t - last;
        last = t;
        let target = f32::from_bits(s.level.load(Ordering::Relaxed));
        let level = smoother.step(target, dt);
        let look = guard.frame(t, level, reduced());
        drop(guard);
        match look {
            Some(look) => {
                drawing = true;
                let size = (sink.size)();
                (sink.show)(Some(&render_rgba(look, size)), size);
                std::thread::sleep(frame_time);
            }
            None if drawing => {
                drawing = false;
                (sink.show)(None, 0);
                // a setting changed in the last dictation applies from the next
                s.reduced.store((sink.reduced_motion)(), Ordering::Relaxed);
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn ignite_is_full_strength_at_the_press_before_any_level_arrives() {
        let at_press = look(Stage::Listening, 0.0, 0.0, false);
        assert!((at_press.strength - 0.88).abs() < EPS, "{at_press:?}");
        // whatever the level, the first frame is the same
        assert_eq!(at_press, look(Stage::Listening, 0.0, 0.3, false));
        assert_eq!(at_press.half_width, WIDE_WIDTH);
    }

    #[test]
    fn ignite_decays_linearly_to_the_live_level_in_0_18_seconds() {
        let level = 0.4;
        let live = 0.50 + 0.38 * level;
        let mid = look(Stage::Listening, 0.09, level, false).strength;
        let want = 0.50 + 0.38 * (0.5 * 1.0 + 0.5 * level);
        assert!((mid - want).abs() < EPS);
        assert!((look(Stage::Listening, 0.18, level, false).strength - live).abs() < EPS);
        assert!((look(Stage::Listening, 2.0, level, false).strength - live).abs() < EPS);
        // the floor holds in silence
        let quiet = look(Stage::Listening, 5.0, 0.0, false).strength;
        assert!((quiet - (0.50 + 0.38 * LEVEL_FLOOR)).abs() < EPS);
    }

    #[test]
    fn gather_reaches_0_08_at_1_2_seconds_and_holds() {
        let start = look(Stage::Processing, 0.0, 0.5, false);
        assert!((start.half_width - 0.5).abs() < EPS);
        assert!((start.strength - 0.65).abs() < EPS);
        assert!((look(Stage::Processing, 0.6, 0.5, false).half_width - 0.29).abs() < EPS);
        let end = look(Stage::Processing, 1.2, 0.5, false);
        assert!((end.half_width - 0.08).abs() < EPS);
        assert!((end.strength - 1.0).abs() < EPS);
        assert!((end.heat - 1.0).abs() < EPS);
        for t in [1.3, 5.0, 60.0] {
            assert_eq!(look(Stage::Processing, t, 0.9, false), end, "no loop, no breathing at {t}");
        }
    }

    #[test]
    fn burst_follows_sqrt_and_ends_at_zero() {
        let s = Stage::Pasted { from_width: 0.08 };
        let start = look(s, 0.0, 0.5, false);
        assert!((start.half_width - 0.08).abs() < EPS && (start.burst - 1.0).abs() < EPS);
        let quarter = look(s, 0.15, 0.5, false); // p = 0.25
        assert!((quarter.half_width - (0.08 + 0.42 * 0.5)).abs() < EPS);
        assert!((quarter.burst - 0.75).abs() < EPS);
        let end = look(s, 0.6, 0.5, false);
        assert!((end.half_width - 0.5).abs() < EPS);
        assert!(end.burst.abs() < EPS && end.strength == 0.0);
        assert!(finished(s, 0.6, false) && !finished(s, 0.59, false));
    }

    #[test]
    fn a_paste_before_the_gather_ends_bursts_from_where_it_is() {
        let mut m = Machine::default();
        m.press(0.0);
        m.release(1.0);
        m.pasted(1.6, false); // half way through the gather: w = 0.29
        let Some(Stage::Pasted { from_width }) = m.stage() else { panic!("{:?}", m.stage()) };
        assert!((from_width - 0.29).abs() < EPS);
        let first = m.frame(1.6, 0.5, false).unwrap();
        assert!((first.half_width - 0.29).abs() < EPS);
        let last = m.frame(2.19, 0.5, false).unwrap();
        assert!(last.half_width > 0.49 && last.burst < 0.02);
        assert!(m.frame(2.2, 0.5, false).is_none() && m.is_rest());
    }

    #[test]
    fn ending_with_nothing_pasted_skips_the_burst() {
        let mut m = Machine::default();
        m.press(0.0);
        m.release(1.0);
        m.dropped(3.0, 0.5, false); // gather done: the tight bright core
        let first = m.frame(3.0, 0.5, false).unwrap();
        assert!((first.half_width - 0.08).abs() < EPS && (first.strength - 1.0).abs() < EPS);
        for t in [3.05, 3.1, 3.19] {
            let f = m.frame(t, 0.5, false).unwrap();
            assert_eq!(f.burst, 0.0);
            assert!((f.half_width - 0.08).abs() < EPS, "the core only fades, it does not open");
        }
        let mid = look(m.stage().unwrap(), 0.1, 0.5, false);
        assert!((mid.strength - 0.5).abs() < EPS);
        assert!(m.frame(3.2, 0.5, false).is_none() && m.is_rest());
    }

    #[test]
    fn a_failure_while_listening_fades_too() {
        let mut m = Machine::default();
        m.press(0.0);
        m.dropped(0.05, 0.3, false);
        assert!(matches!(m.stage(), Some(Stage::Dropped { .. })));
        assert!(m.frame(0.30, 0.3, false).is_none());
    }

    #[test]
    fn reduced_motion_keeps_the_width_constant() {
        let widths: Vec<f32> = [0.0, 0.3, 1.2, 4.0]
            .into_iter()
            .map(|t| look(Stage::Processing, t, 0.5, true).half_width)
            .collect();
        assert!(widths.iter().all(|w| (*w - REDUCED_WIDTH).abs() < EPS), "{widths:?}");
        let mut m = Machine::default();
        m.press(0.0);
        m.release(1.0);
        m.pasted(2.0, true);
        let s = m.stage().unwrap();
        for t in [0.0, 0.1, 0.29] {
            assert!((look(s, t, 0.5, true).half_width - REDUCED_WIDTH).abs() < EPS);
        }
        assert!((look(s, 0.15, 0.5, true).burst - 0.5).abs() < EPS);
        assert!(finished(s, 0.3, true) && !finished(s, 0.29, true));
    }

    #[test]
    fn events_out_of_order_do_nothing() {
        let mut m = Machine::default();
        m.pasted(0.0, false);
        m.release(0.0);
        m.dropped(0.0, 0.5, false);
        assert!(m.is_rest());
        m.press(1.0);
        m.pasted(1.1, false); // not processing yet
        assert_eq!(m.stage(), Some(Stage::Listening));
        m.release(2.0);
        m.press(2.5); // a new dictation while the last one still processes
        assert_eq!(m.stage(), Some(Stage::Listening));
        m.pasted(3.0, false); // the old one's late paste must not light the new one
        assert_eq!(m.stage(), Some(Stage::Listening));
    }

    #[test]
    fn the_smoother_attacks_fast_and_releases_slowly() {
        let mut s = Smoother::new();
        let up = s.step(1.0, 0.04);
        assert!(up > 0.6, "{up}");
        let down = s.step(0.0, 0.04);
        assert!(down > 0.55 && down < up, "release is slow: {down} after {up}");
        for _ in 0..200 {
            s.step(0.0, 0.05);
        }
        assert!((s.value() - LEVEL_FLOOR).abs() < 1e-3);
    }

    #[test]
    fn the_frame_is_black_at_the_ends_and_lit_in_the_middle() {
        let wide = render_rgba(Look { half_width: 0.5, strength: 0.88, heat: 0.0, burst: 0.0 }, 32);
        let px = |buf: &[u8], x: usize, y: usize| {
            let i = (y * 32 + x) * 4;
            [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
        };
        let middle = px(&wide, 16, 16);
        let end = px(&wide, 2, 16);
        assert!(middle[0] > 80 && middle[3] == 255, "{middle:?}");
        assert!(end[0] < 20 && end[3] > 0, "{end:?}");
        // outside the capsule is transparent: the icon keeps its square slot
        assert_eq!(px(&wide, 16, 0)[3], 0);
        // the tight core lights less of the width
        let tight = render_rgba(Look { half_width: 0.08, strength: 1.0, heat: 1.0, burst: 0.0 }, 32);
        assert_eq!(px(&tight, 6, 16)[0], 0);
        assert!(px(&tight, 16, 16)[0] > 150);
    }
}
