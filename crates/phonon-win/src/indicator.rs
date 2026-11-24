//! The dictation indicator's stages (ignite, listening, processing, then a fade back to rest), drawn into
//! the tray icon.
//!
//! Windows and Linux have no floating bar: the tray icon is the whole interface. The icon keeps its
//! square slot; while a dictation is under way it shows the Mac bar's gradient in a capsule (black at
//! both ends, one warm light in the middle), and between dictations it is the normal Phonon icon and
//! nothing repaints.
//!
//! The gradient is the Mac capsule's and the iPhone's: black at both ends, one warm ember light at 0.5
//! (`half_width` stays at 0.5 in every stage; there is no sliding-together core):
//!  1. Ignite, the instant the key goes down: full strength, falling to the live voice level in 0.10 s.
//!  2. Listening: w = 0.5, strength follows the smoothed voice level.
//!  3. Processing, from key-up until the text is delivered: steady warm ember at full width (0.65 strength,
//!     no repaint while waiting).
//!  4. Ended (text pasted or not): no pulse, fades straight out at full width in 0.10 s with quadratic ease-out.
//!
//! Every press starts a new dictation with its own number (`press` returns it). The events that move a
//! dictation on (`release`, `dropped`) carry that number and do nothing when the indicator has moved on
//! to a newer dictation, so a slow dictation's late end never fades the next one.
//!
//! `look` is the pure stage maths; `Machine` is the state the hooks drive; `start` runs the one thread
//! that turns frames into icons for the platform's tray (the sink).

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The Mac capsule's warm ember (0.42, 0.12, 0.035).
pub const BASE: [f32; 3] = [107.0, 31.0, 9.0];

pub const IGNITE_SECONDS: f32 = 0.10;
pub const DROP_SECONDS: f32 = 0.10;
pub const WIDE_WIDTH: f32 = 0.5;
/// Width never moves in either normal or reduced-motion mode.
pub const REDUCED_WIDTH: f32 = WIDE_WIDTH;
pub const LEVEL_FLOOR: f32 = 0.08;

/// Listening's light is this opaque over black in silence...
pub const LISTEN_STRENGTH: f32 = 0.50;
/// ...plus this much times the voice energy (0..1). The Mac capsule's numbers (macos/Sources/PhononBar.swift
/// `warmth`: 0.50 + energy * 0.38).
pub const LISTEN_STRENGTH_RANGE: f32 = 0.38;
/// Full strength: energy 1, the ignite flash and the loudest voice, 0.50 + 0.38 = 0.88 opacity of the ember
/// over black.
pub const FULL_STRENGTH: f32 = LISTEN_STRENGTH + LISTEN_STRENGTH_RANGE;
/// Steady warm ember strength while processing (key-up until text arrives).
pub const PROCESSING_STRENGTH: f32 = 0.65;

/// A dictation still listening with no audio arriving for this long is dropped: the worker died, the
/// microphone never opened, or a key-up was lost. A latched dictation that keeps receiving audio is never
/// cut, however long it runs.
pub const SILENT_LISTENING_LIMIT: f64 = 180.0;
/// Processing (no progress signal reaches the indicator while the models run) is dropped after this long
/// plus `PROCESSING_PER_SECOND_HEARD` times the seconds it listened, so a long recording's long
/// transcription is never cut while a lost dictation does not stay lit for good.
pub const PROCESSING_LIMIT: f64 = 300.0;
pub const PROCESSING_PER_SECOND_HEARD: f64 = 2.0;

/// The tray changes at most this often (about 15 frames a second).
pub const FRAME_TIME: Duration = Duration::from_millis(66);
/// With nothing moving, the thread wakes this often (or at once on an event or a new level) to check the
/// watchdog; it paints nothing unless the frame changed.
const IDLE_CHECK: Duration = Duration::from_secs(1);
/// A voice level closer than this to the last one is not news (the frame cannot visibly change).
const LEVEL_STEP: f32 = 0.01;
/// The smoother is settled (the listening frame holds still) within this of its target.
const SETTLED: f32 = 0.004;
/// The reduced-motion setting is re-read at most this often, after a dictation, off the drawing thread.
const REDUCED_MOTION_REREAD: Duration = Duration::from_secs(60);

/// Where in a dictation the indicator is. The variants that end one carry what they fade from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stage {
    /// Key down until key up. Elapsed counts from the press.
    Listening,
    /// Key up until the text is delivered or the dictation ends. Elapsed counts from the release.
    Processing,
    /// Ended, text pasted or not (empty, error, cancelled).
    Dropped {
        from_width: f32,
        from_strength: f32,
        from_heat: f32,
    },
}

/// What one frame looks like. The centre colour is `BASE * strength` over black.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    /// Half the width of the lit part, as a fraction of the whole (always 0.5 = full width).
    pub half_width: f32,
    /// Opacity of the warm light over black.
    pub strength: f32,
    /// Kept at 0.0: single warm ember colour.
    pub heat: f32,
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// The stage maths: stage, seconds since it began, smoothed voice level (0..1), reduced motion.
pub fn look(stage: Stage, elapsed: f32, level: f32, _reduced: bool) -> Look {
    let elapsed = elapsed.max(0.0);
    let level = level.clamp(LEVEL_FLOOR, 1.0);
    match stage {
        Stage::Listening => {
            let ignite = (1.0 - elapsed / IGNITE_SECONDS).clamp(0.0, 1.0);
            // full strength at the press, the live level 0.10 s later, linearly between
            let live = LISTEN_STRENGTH + LISTEN_STRENGTH_RANGE * level;
            Look {
                half_width: WIDE_WIDTH,
                strength: lerp(live, FULL_STRENGTH, ignite),
                heat: 0.0,
            }
        }
        Stage::Processing => Look {
            half_width: WIDE_WIDTH,
            strength: PROCESSING_STRENGTH,
            heat: 0.0,
        },
        Stage::Dropped {
            from_width,
            from_strength,
            from_heat,
        } => {
            let p = (elapsed / DROP_SECONDS).clamp(0.0, 1.0);
            let remain = (1.0 - p) * (1.0 - p);
            Look {
                half_width: from_width,
                strength: from_strength * remain,
                heat: from_heat,
            }
        }
    }
}

/// Whether the ending stage has finished and the icon is at rest again.
pub fn finished(stage: Stage, elapsed: f32) -> bool {
    match stage {
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
        let tau = if target > self.shown {
            Self::ATTACK_SECONDS
        } else {
            Self::RELEASE_SECONDS
        };
        self.shown += (target - self.shown) * (1.0 - (-dt.max(0.0) / tau).exp());
        self.shown = self.shown.clamp(LEVEL_FLOOR, 1.0);
        self.shown
    }

    /// Whether the shown level has reached `target` (nothing left to animate).
    pub fn settled(&self, target: f32) -> bool {
        (self.shown - target.clamp(LEVEL_FLOOR, 1.0)).abs() < SETTLED
    }

    #[cfg(test)]
    pub fn value(&self) -> f32 {
        self.shown
    }
}

impl Default for Smoother {
    fn default() -> Self {
        Self::new()
    }
}

/// The indicator's state. Times are seconds on any steady clock, as f64 so that a Phonon running for weeks
/// still times a 0.18 s ignite exactly (an f32 clock is 0.125 s coarse after 12 days). Events call these
/// from any thread; each says what it does when it is not the right moment (then nothing happens).
#[derive(Clone, Copy, Debug, Default)]
pub struct Machine {
    /// The stage and when it began.
    active: Option<(Stage, f64)>,
    /// The number of the last press; `owner` is the dictation the stage belongs to.
    generation: u64,
    owner: u64,
    /// How long the dictation listened, for the processing watchdog.
    listened: f64,
}

impl Machine {
    pub fn is_rest(&self) -> bool {
        self.active.is_none()
    }

    pub fn stage(&self) -> Option<Stage> {
        self.active.map(|(stage, _)| stage)
    }

    /// The last press's number: it changes on every press.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Key down, tap, or a menu start: a new dictation, lit at once whatever the indicator was doing (a
    /// press while the last dictation is still processing starts the new one; that one's late events are
    /// then ignored). Returns the new dictation's number. A press while another dictation still listens
    /// takes that light over without a second flash.
    pub fn press(&mut self, now: f64) -> u64 {
        self.generation += 1;
        self.owner = self.generation;
        if !matches!(self.active, Some((Stage::Listening, _))) {
            self.active = Some((Stage::Listening, now));
        }
        self.generation
    }

    fn owns(&self, dictation: u64) -> bool {
        dictation != 0 && dictation == self.owner
    }

    /// Key up or stop: dictation `dictation`'s listening becomes processing.
    pub fn release(&mut self, dictation: u64, now: f64) {
        if !self.owns(dictation) {
            return;
        }
        if let Some((Stage::Listening, since)) = self.active {
            self.listened = now - since;
            self.active = Some((Stage::Processing, now));
        }
    }

    /// Dictation `dictation` ended (text pasted or not): no pulse, the light fades out from where it is
    /// (`level` is the smoothed voice level it was showing).
    pub fn dropped(&mut self, dictation: u64, now: f64, level: f32, reduced: bool) {
        if self.owns(dictation) {
            self.fade(now, level, reduced);
        }
    }

    fn fade(&mut self, now: f64, level: f32, reduced: bool) {
        if let Some((stage @ (Stage::Listening | Stage::Processing), since)) = self.active {
            let current = look(stage, (now - since) as f32, level, reduced);
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

    /// Whether a lit stage has outlived any real dictation: listening with no audio since `heard` (or the
    /// press) for `SILENT_LISTENING_LIMIT`, or processing past its bound.
    pub fn stale(&self, now: f64, heard: f64) -> bool {
        match self.active {
            Some((Stage::Listening, since)) => now - heard.max(since) > SILENT_LISTENING_LIMIT,
            Some((Stage::Processing, since)) => {
                now - since > PROCESSING_LIMIT + PROCESSING_PER_SECOND_HEARD * self.listened
            }
            _ => false,
        }
    }

    /// Fade out a stale stage (see `stale`). Returns whether it did.
    pub fn watchdog(&mut self, now: f64, heard: f64, level: f32, reduced: bool) -> bool {
        let stale = self.stale(now, heard);
        if stale {
            self.fade(now, level, reduced);
        }
        stale
    }

    /// The frame to draw now, or `None` at rest. A stage that ends hands back to rest here.
    pub fn frame(&mut self, now: f64, level: f32, reduced: bool) -> Option<Look> {
        let (stage, since) = self.active?;
        let elapsed = (now - since) as f32;
        if finished(stage, elapsed) {
            self.active = None;
            return None;
        }
        Some(look(stage, elapsed, level, reduced))
    }

    /// Whether the next frame can differ from this one for any reason but the voice level: false at rest,
    /// while processing holds steady, and for listening after the ignite.
    pub fn moving(&self, now: f64, _reduced: bool) -> bool {
        match self.active {
            None | Some((Stage::Processing, _)) => false,
            Some((Stage::Listening, since)) => now - since < f64::from(IGNITE_SECONDS),
            Some((Stage::Dropped { .. }, _)) => true,
        }
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
        BASE[0] * look.strength,
        BASE[1] * look.strength,
        BASE[2] * look.strength,
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
            for (slot, light) in out[i..i + 3].iter_mut().zip(core) {
                *slot = (light * g).clamp(0.0, 255.0).round() as u8;
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
    /// Whether the system asks for reduced motion. Called off the drawing thread: at start, then at most
    /// once a minute after a dictation ends, so a slow answer never holds a frame.
    pub reduced_motion: fn() -> bool,
}

/// The machine and a count of the events that changed it, so the drawing thread can sleep until one does.
struct State {
    machine: Machine,
    revision: u64,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    /// The latest voice level (f32 bits), as pushed by the microphone.
    level: AtomicU32,
    /// The smoothed level the drawing thread last showed (f32 bits): a fade starts from it.
    shown: AtomicU32,
    /// When audio last arrived (f64 seconds on `now`'s clock, as bits), for the watchdog.
    heard: AtomicU64,
    reduced: AtomicBool,
    /// Tells the reduced-motion reader a dictation ended.
    reread: OnceLock<Mutex<mpsc::Sender<()>>>,
    origin: Instant,
}

fn shared() -> &'static Shared {
    static SHARED: OnceLock<Shared> = OnceLock::new();
    SHARED.get_or_init(|| Shared {
        state: Mutex::new(State {
            machine: Machine::default(),
            revision: 0,
        }),
        wake: Condvar::new(),
        level: AtomicU32::new(LEVEL_FLOOR.to_bits()),
        shown: AtomicU32::new(LEVEL_FLOOR.to_bits()),
        heard: AtomicU64::new(0f64.to_bits()),
        reduced: AtomicBool::new(false),
        reread: OnceLock::new(),
        origin: Instant::now(),
    })
}

fn now() -> f64 {
    shared().origin.elapsed().as_secs_f64()
}

fn with_machine<R>(f: impl FnOnce(&mut Machine, f64) -> R) -> Option<R> {
    let s = shared();
    let result = s.state.lock().ok().map(|mut state| {
        state.revision = state.revision.wrapping_add(1);
        f(&mut state.machine, now())
    });
    s.wake.notify_all();
    result
}

fn reduced() -> bool {
    shared().reduced.load(Ordering::Relaxed)
}

fn shown_level() -> f32 {
    f32::from_bits(shared().shown.load(Ordering::Relaxed))
}

/// The dictation key went down (or the menu started a dictation): a new dictation, lit within a frame.
/// Returns its number for `release` and `dropped`. Returns at once; the animation thread draws
/// without waiting for the microphone or the models. The last dictation's voice level is forgotten, so
/// the ignite falls to this one's voice, not to that one's last word.
pub fn press() -> u64 {
    let s = shared();
    s.level.store(LEVEL_FLOOR.to_bits(), Ordering::Relaxed);
    s.shown.store(LEVEL_FLOOR.to_bits(), Ordering::Relaxed);
    with_machine(|m, t| m.press(t)).unwrap_or(0)
}

/// Dictation `dictation`'s key came up (or a stop): processing begins. Nothing for an older dictation.
pub fn release(dictation: u64) {
    with_machine(|m, t| m.release(dictation, t));
}

/// Dictation `dictation` ended (pasted, empty, error, cancelled, kept on the clipboard): the light fades.
/// Nothing once it was dropped, or for an older dictation.
pub fn dropped(dictation: u64) {
    let reduced = reduced();
    let level = shown_level();
    with_machine(|m, t| m.dropped(dictation, t, level, reduced));
}

/// Drops its dictation when it goes out of scope, so no path out of a dictation's processing, a panic
/// included, leaves the light on. A no-op once the dictation was dropped.
pub struct Lit(pub u64);

impl Drop for Lit {
    fn drop(&mut self) {
        dropped(self.0);
    }
}

/// The microphone's RMS for the latest buffer (0..1 of full scale). Wakes the drawing thread only when
/// the level moved enough to show.
pub fn push_level(rms: f32) {
    let s = shared();
    s.heard.store(now().to_bits(), Ordering::Relaxed);
    let level = level_from_rms(rms);
    let old = f32::from_bits(s.level.load(Ordering::Relaxed));
    if (level - old).abs() < LEVEL_STEP {
        return;
    }
    s.level.store(level.to_bits(), Ordering::Relaxed);
    // through the lock, so a drawing thread about to sleep either sees the new level or this wake-up
    drop(s.state.lock());
    s.wake.notify_all();
}

/// Start the animation thread. It sleeps until the first press and sleeps again at rest, so an idle
/// Phonon repaints nothing.
pub fn start(sink: Sink) {
    let (reread, requests) = mpsc::channel::<()>();
    let _ = shared().reread.set(Mutex::new(reread));
    let read = sink.reduced_motion;
    let _ = std::thread::Builder::new()
        .name("reduced-motion".into())
        .spawn(move || follow_reduced_motion(read, requests));
    let _ = std::thread::Builder::new()
        .name("indicator".into())
        .spawn(move || run(sink));
}

/// Read the reduced-motion setting at start, then again after a dictation ends, at most once a minute: a
/// setting changed during one dictation applies from a later one.
fn follow_reduced_motion(read: fn() -> bool, requests: mpsc::Receiver<()>) {
    let s = shared();
    s.reduced.store(read(), Ordering::Relaxed);
    while requests.recv().is_ok() {
        s.reduced.store(read(), Ordering::Relaxed);
        std::thread::sleep(REDUCED_MOTION_REREAD);
        while requests.try_recv().is_ok() {}
    }
}

fn ask_reread() {
    if let Some(Ok(sender)) = shared().reread.get().map(|s| s.lock()) {
        let _ = sender.send(());
    }
}

fn run(sink: Sink) {
    let s = shared();
    let mut smoother = Smoother::new();
    let mut last = now();
    let mut seen_generation = 0;
    // the frame in the tray now, `None` with the normal icon there
    let mut showing: Option<Vec<u8>> = None;
    loop {
        let Ok(state) = s.state.lock() else { return };
        let drawing = showing.is_some();
        let Ok(mut state) = s
            .wake
            .wait_while(state, |st| st.machine.is_rest() && !drawing)
        else {
            return;
        };
        let t = now();
        if state.machine.generation() != seen_generation {
            // a new dictation: its own level from the floor, no time carried over from the last
            seen_generation = state.machine.generation();
            smoother = Smoother::new();
            last = t;
        }
        let dt = (t - last) as f32;
        last = t;
        let target_bits = s.level.load(Ordering::Relaxed);
        let target = f32::from_bits(target_bits);
        let level = smoother.step(target, dt);
        s.shown.store(level.to_bits(), Ordering::Relaxed);
        let reduced = reduced();
        let heard = f64::from_bits(s.heard.load(Ordering::Relaxed));
        if state.machine.watchdog(t, heard, level, reduced) {
            eprintln!("phonon: the dictation indicator outlived its dictation; fading it");
        }
        let look = state.machine.frame(t, level, reduced);
        let listening = state.machine.stage() == Some(Stage::Listening);
        let moving = state.machine.moving(t, reduced) || (listening && !smoother.settled(target));
        let revision = state.revision;
        drop(state);
        match look {
            Some(look) => {
                let size = (sink.size)();
                let rgba = render_rgba(look, size);
                // the same pixels again (the held core, a steady level) are not sent to the tray
                if showing.as_deref() != Some(rgba.as_slice()) {
                    (sink.show)(Some(&rgba), size);
                    showing = Some(rgba);
                }
            }
            None => {
                if showing.take().is_some() {
                    (sink.show)(None, 0);
                    ask_reread();
                }
                continue;
            }
        }
        if moving {
            std::thread::sleep(FRAME_TIME);
        } else {
            // Still: sleep until an event, a new voice level, or the watchdog's next look.
            let Ok(state) = s.state.lock() else { return };
            let _ = s.wake.wait_timeout_while(state, IDLE_CHECK, |st| {
                st.revision == revision && s.level.load(Ordering::Relaxed) == target_bits
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    /// Listening's strength at a voice energy (0..1).
    fn listening(energy: f32) -> f32 {
        LISTEN_STRENGTH + LISTEN_STRENGTH_RANGE * energy
    }

    #[test]
    fn full_strength_is_the_mac_capsules_listening_maximum() {
        // 0.50 + 0.38 opacity over black, the loudest voice and the ignite alike
        let loudest = look(Stage::Listening, 5.0, 1.0, false).strength;
        assert!((loudest - 0.88).abs() < EPS && (loudest - FULL_STRENGTH).abs() < EPS);
        let processing = look(Stage::Processing, 0.0, 0.5, false).strength;
        assert!((processing - PROCESSING_STRENGTH).abs() < EPS);
    }

    #[test]
    fn ignite_is_full_strength_at_the_press_before_any_level_arrives() {
        let at_press = look(Stage::Listening, 0.0, 0.0, false);
        assert!(
            (at_press.strength - FULL_STRENGTH).abs() < EPS,
            "{at_press:?}"
        );
        // whatever the level, the first frame is the same
        assert_eq!(at_press, look(Stage::Listening, 0.0, 0.3, false));
        assert_eq!(at_press.half_width, WIDE_WIDTH);
    }

    #[test]
    fn ignite_decays_linearly_to_the_live_level_in_0_10_seconds() {
        let level = 0.4;
        let mid = look(Stage::Listening, 0.05, level, false).strength;
        // half way through the ignite: half full energy, half the live level
        assert!((mid - listening(0.5 + 0.5 * level)).abs() < EPS);
        assert!(
            (look(Stage::Listening, 0.10, level, false).strength - listening(level)).abs() < EPS
        );
        assert!(
            (look(Stage::Listening, 2.0, level, false).strength - listening(level)).abs() < EPS
        );
        // the floor holds in silence
        let quiet = look(Stage::Listening, 5.0, 0.0, false).strength;
        assert!((quiet - listening(LEVEL_FLOOR)).abs() < EPS);
    }

    #[test]
    fn processing_is_steady_full_width_warm_ember() {
        let steady = look(Stage::Processing, 0.0, 0.5, false);
        assert!((steady.half_width - WIDE_WIDTH).abs() < EPS);
        assert!((steady.strength - PROCESSING_STRENGTH).abs() < EPS);
        assert!((steady.heat - 0.0).abs() < EPS);
        for t in [0.02, 0.1, 0.6, 1.2, 5.0, 60.0] {
            assert_eq!(
                look(Stage::Processing, t, 0.9, false),
                steady,
                "no sliding or breathing at {t}"
            );
            assert_eq!(
                look(Stage::Processing, t, 0.9, true),
                steady,
                "identical under reduced motion at {t}"
            );
        }
    }

    #[test]
    fn fast_dictation_ends_and_fades_cleanly_at_full_width() {
        let mut m = Machine::default();
        let a = m.press(0.0);
        m.release(a, 1.0);
        m.dropped(a, 1.02, 0.5, false);
        let Some(Stage::Dropped {
            from_width,
            from_strength,
            ..
        }) = m.stage()
        else {
            panic!("{:?}", m.stage())
        };
        assert!((from_width - WIDE_WIDTH).abs() < EPS);
        assert!((from_strength - PROCESSING_STRENGTH).abs() < EPS);
        let first = m.frame(1.02, 0.5, false).unwrap();
        assert!((first.half_width - WIDE_WIDTH).abs() < EPS);
        assert!(
            m.frame(1.02 + f64::from(DROP_SECONDS), 0.5, false)
                .is_none()
                && m.is_rest()
        );
    }

    #[test]
    fn the_end_fades_with_no_pulse_or_width_change() {
        let mut m = Machine::default();
        let a = m.press(0.0);
        m.release(a, 1.0);
        m.dropped(a, 3.0, 0.5, false);
        let first = m.frame(3.0, 0.5, false).unwrap();
        assert!(
            (first.half_width - WIDE_WIDTH).abs() < EPS
                && (first.strength - PROCESSING_STRENGTH).abs() < EPS
        );
        for t in [3.02, 3.05, 3.09] {
            let f = m.frame(t, 0.5, false).unwrap();
            assert_eq!(f.heat, 0.0, "no colour change");
            assert!(f.strength <= first.strength, "never brighter");
            assert!((f.half_width - WIDE_WIDTH).abs() < EPS, "width stays full");
        }
        let mid = look(m.stage().unwrap(), DROP_SECONDS / 2.0, 0.5, false);
        assert!((mid.strength - PROCESSING_STRENGTH * 0.25).abs() < EPS);
        assert!(m.frame(3.0 + f64::from(DROP_SECONDS), 0.5, false).is_none() && m.is_rest());
    }

    #[test]
    fn a_failure_while_listening_fades_from_the_level_it_showed() {
        let mut m = Machine::default();
        let a = m.press(0.0);
        m.dropped(a, 1.0, 0.3, false);
        let Some(Stage::Dropped { from_strength, .. }) = m.stage() else {
            panic!("{:?}", m.stage())
        };
        assert!((from_strength - listening(0.3)).abs() < EPS);
        assert!(m.frame(1.0 + f64::from(DROP_SECONDS), 0.3, false).is_none());
    }

    #[test]
    fn width_never_moves_normal_or_reduced_motion() {
        for reduced in [false, true] {
            let widths: Vec<f32> = [0.0, 0.05, 0.3, 1.2, 4.0]
                .into_iter()
                .map(|t| look(Stage::Processing, t, 0.5, reduced).half_width)
                .collect();
            assert!(
                widths.iter().all(|w| (*w - WIDE_WIDTH).abs() < EPS),
                "{widths:?}"
            );
            let mut m = Machine::default();
            let a = m.press(0.0);
            m.release(a, 1.0);
            assert!(
                !m.moving(1.0, reduced),
                "steady processing does not repaint"
            );
            m.dropped(a, 2.0, 0.5, reduced);
            let s = m.stage().unwrap();
            for t in [0.0, 0.05, 0.09] {
                assert!((look(s, t, 0.5, reduced).half_width - WIDE_WIDTH).abs() < EPS);
            }
            assert!(finished(s, DROP_SECONDS) && !finished(s, DROP_SECONDS - 0.01));
        }
    }

    #[test]
    fn events_out_of_order_do_nothing() {
        let mut m = Machine::default();
        m.release(0, 0.0);
        m.dropped(0, 0.0, 0.5, false);
        assert!(m.is_rest());
        let a = m.press(1.0);
        m.release(a, 2.0);
        m.release(a, 2.2); // a second stop restarts nothing
        assert_eq!(m.stage(), Some(Stage::Processing));
        assert!(!m.moving(2.2, false));
    }

    #[test]
    fn a_late_end_of_a_does_not_fade_b_while_b_processes() {
        let mut m = Machine::default();
        let a = m.press(0.0);
        m.release(a, 1.0);
        let b = m.press(2.0); // B's key while A is still processing
        m.release(b, 3.0);
        m.dropped(a, 3.6, 0.5, false);
        assert_eq!(m.stage(), Some(Stage::Processing), "A's end is not B's");
        m.dropped(b, 4.0, 0.5, false);
        assert!(matches!(m.stage(), Some(Stage::Dropped { .. })));
    }

    #[test]
    fn a_abandoning_does_not_fade_b_while_b_listens() {
        let mut m = Machine::default();
        let a = m.press(0.0);
        m.release(a, 1.0);
        let b = m.press(2.0);
        m.dropped(a, 2.5, 0.5, false); // A ended empty while B is held
        assert_eq!(m.stage(), Some(Stage::Listening));
        m.release(b, 4.0);
        assert_eq!(m.stage(), Some(Stage::Processing));
    }

    #[test]
    fn releasing_a_after_b_was_pressed_leaves_b_listening() {
        let mut m = Machine::default();
        let a = m.press(0.0);
        let b = m.press(0.5); // a menu start, or a press the hook saw before A's stop reached the worker
        m.release(a, 1.0);
        assert_eq!(m.stage(), Some(Stage::Listening), "A's stop is not B's");
        m.release(b, 2.0);
        assert_eq!(m.stage(), Some(Stage::Processing));
    }

    #[test]
    fn a_press_while_listening_does_not_flash_again() {
        let mut m = Machine::default();
        m.press(0.0);
        m.press(1.0);
        let f = m.frame(1.0, LEVEL_FLOOR, false).unwrap();
        assert!((f.strength - listening(LEVEL_FLOOR)).abs() < EPS);
    }

    #[test]
    fn a_long_latched_dictation_is_never_cut_while_audio_arrives() {
        let mut m = Machine::default();
        m.press(0.0);
        // an hour of dictation, audio arriving all along
        for minute in 1..=60u32 {
            let t = f64::from(minute) * 60.0;
            assert!(!m.watchdog(t, t - 0.05, 0.5, false), "cut at {t} s");
        }
        // the key-up never came and the audio stopped: dropped after the limit
        let quiet_from = 3600.0;
        assert!(!m.stale(quiet_from + SILENT_LISTENING_LIMIT - 1.0, quiet_from));
        assert!(m.watchdog(
            quiet_from + SILENT_LISTENING_LIMIT + 1.0,
            quiet_from,
            0.5,
            false
        ));
        assert!(matches!(m.stage(), Some(Stage::Dropped { .. })));
    }

    #[test]
    fn processing_is_given_time_in_proportion_to_what_was_heard() {
        let mut m = Machine::default();
        let a = m.press(0.0);
        m.release(a, 600.0); // a ten-minute dictation
        let bound = PROCESSING_LIMIT + PROCESSING_PER_SECOND_HEARD * 600.0;
        assert!(!m.stale(600.0 + bound - 1.0, 600.0));
        assert!(m.watchdog(600.0 + bound + 1.0, 600.0, 0.5, false));
        assert!(matches!(m.stage(), Some(Stage::Dropped { .. })));
    }

    #[test]
    fn steady_processing_never_repaints() {
        let mut m = Machine::default();
        assert!(!m.moving(0.0, false), "rest");
        let a = m.press(0.0);
        assert!(m.moving(0.0, false), "the ignite");
        assert!(!m.moving(f64::from(IGNITE_SECONDS) + 0.01, false));
        m.release(a, 1.0);
        assert!(!m.moving(1.0, false), "steady processing");
        assert!(!m.moving(1.5, false), "steady processing");
        m.dropped(a, 5.0, 0.5, false);
        assert!(m.moving(5.05, false), "the fade");
    }

    #[test]
    fn an_old_clock_still_times_the_ignite() {
        // twelve days of uptime
        let start = 12.0 * 24.0 * 3600.0;
        let mut m = Machine::default();
        m.press(start);
        let mid = m
            .frame(start + f64::from(IGNITE_SECONDS) / 2.0, 0.0, false)
            .unwrap();
        assert!((mid.strength - listening(0.5 + 0.5 * LEVEL_FLOOR)).abs() < 1e-3);
    }

    #[test]
    fn the_smoother_attacks_fast_and_releases_slowly() {
        let mut s = Smoother::new();
        let up = s.step(1.0, 0.04);
        assert!(up > 0.6, "{up}");
        let down = s.step(0.0, 0.04);
        assert!(
            down > 0.55 && down < up,
            "release is slow: {down} after {up}"
        );
        assert!(!s.settled(0.0));
        for _ in 0..200 {
            s.step(0.0, 0.05);
        }
        assert!((s.value() - LEVEL_FLOOR).abs() < 1e-3);
        assert!(s.settled(0.0));
    }

    #[test]
    fn the_frame_is_black_at_the_ends_and_lit_in_the_middle() {
        let wide = render_rgba(
            Look {
                half_width: WIDE_WIDTH,
                strength: FULL_STRENGTH,
                heat: 0.0,
            },
            32,
        );
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
        // steady processing uses the same full-width warm ember at lower strength
        let steady = render_rgba(
            Look {
                half_width: WIDE_WIDTH,
                strength: PROCESSING_STRENGTH,
                heat: 0.0,
            },
            32,
        );
        assert!(px(&steady, 16, 16)[0] > 50 && px(&steady, 16, 16)[0] < middle[0]);
        assert!(px(&steady, 6, 16)[0] > 0);
    }
}
