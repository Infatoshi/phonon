//! Hold, tap, and double-tap latch for one dictation key.
//!
//! This is the Rust port of the Swift `HoldKeyTapLatch` in `bar/Sources/PhononBar.swift`.
//! macOS drives it from the Globe (fn) key. Windows drives it from Right Ctrl through
//! a `WH_KEYBOARD_LL` hook. Both platforms must behave the same, so the rules live
//! here and both test suites cover the same cases.
//!
//! Rules:
//! - Press starts a capture. Release stops it. That is hold-to-talk.
//! - A short press and release is a tap. It still records, so a tap-tap-stop user
//!   gets a short capture.
//! - Two taps inside the double-tap window latch the capture. The next press stops it.
//! - A release while latched does nothing.

/// What the caller must do after a key event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Begin recording.
    Start,
    /// End recording.
    Stop,
    /// Do nothing.
    None,
}

/// A press is a tap when it lasts no longer than this.
pub const DEFAULT_TAP_MAX_NS: u64 = 250_000_000;
/// Two taps latch when the second press starts within this time of the first release.
pub const DEFAULT_DOUBLE_TAP_WINDOW_NS: u64 = 350_000_000;

/// The state machine. Timestamps are monotonic nanoseconds; the caller supplies them.
#[derive(Debug, Clone)]
pub struct HoldKeyTapLatch {
    tap_max_ns: u64,
    double_tap_window_ns: u64,
    latched: bool,
    down_at: Option<u64>,
    last_tap_up_at: Option<u64>,
    second_tap_candidate: bool,
}

impl Default for HoldKeyTapLatch {
    fn default() -> Self {
        Self::new(DEFAULT_TAP_MAX_NS, DEFAULT_DOUBLE_TAP_WINDOW_NS)
    }
}

impl HoldKeyTapLatch {
    /// A latch with explicit thresholds. Both bounds are inclusive.
    pub fn new(tap_max_ns: u64, double_tap_window_ns: u64) -> Self {
        Self {
            tap_max_ns,
            double_tap_window_ns,
            latched: false,
            down_at: None,
            last_tap_up_at: None,
            second_tap_candidate: false,
        }
    }

    /// Whether the capture is latched on.
    pub fn latched(&self) -> bool {
        self.latched
    }

    /// Feed a key-down at `now` nanoseconds.
    pub fn key_down(&mut self, now: u64) -> Action {
        if self.latched {
            self.reset();
            return Action::Stop;
        }
        self.second_tap_candidate = self
            .last_tap_up_at
            .is_some_and(|up| now >= up && now - up <= self.double_tap_window_ns);
        self.last_tap_up_at = None;
        self.down_at = Some(now);
        Action::Start
    }

    /// Feed a key-up at `now` nanoseconds.
    pub fn key_up(&mut self, now: u64) -> Action {
        // A key-up with no matching key-down follows a latched stop.
        let Some(down_at) = self.down_at.take() else {
            return Action::None;
        };
        let is_tap = now >= down_at && now - down_at <= self.tap_max_ns;
        if is_tap && self.second_tap_candidate {
            self.second_tap_candidate = false;
            self.latched = true;
            return Action::None;
        }
        self.second_tap_candidate = false;
        self.last_tap_up_at = if is_tap { Some(now) } else { None };
        Action::Stop
    }

    /// Forget the latch and any pending tap. Call this when something else ended
    /// the recording, such as a tray menu item or an error.
    pub fn reset(&mut self) {
        self.latched = false;
        self.down_at = None;
        self.last_tap_up_at = None;
        self.second_tap_candidate = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Milliseconds as nanoseconds, to keep the cases readable.
    fn ms(value: u64) -> u64 {
        value * 1_000_000
    }

    #[test]
    fn hold_starts_on_down_and_stops_on_release() {
        let mut latch = HoldKeyTapLatch::default();
        assert_eq!(latch.key_down(ms(0)), Action::Start);
        assert_eq!(latch.key_up(ms(900)), Action::Stop);
        assert!(!latch.latched());
    }

    #[test]
    fn lone_tap_is_a_short_capture_without_latch() {
        let mut latch = HoldKeyTapLatch::default();
        assert_eq!(latch.key_down(ms(0)), Action::Start);
        assert_eq!(latch.key_up(ms(200)), Action::Stop);
        assert!(!latch.latched());
        // A press well after the window is a plain hold again.
        assert_eq!(latch.key_down(ms(2_000)), Action::Start);
        assert_eq!(latch.key_up(ms(2_800)), Action::Stop);
        assert!(!latch.latched());
    }

    #[test]
    fn double_tap_latches_and_next_press_stops() {
        let mut latch = HoldKeyTapLatch::default();
        assert_eq!(latch.key_down(ms(0)), Action::Start);
        assert_eq!(latch.key_up(ms(150)), Action::Stop);
        assert_eq!(latch.key_down(ms(400)), Action::Start);
        assert_eq!(latch.key_up(ms(550)), Action::None);
        assert!(latch.latched());
        // While latched the press itself stops; its release is ignored.
        assert_eq!(latch.key_down(ms(5_000)), Action::Stop);
        assert!(!latch.latched());
        assert_eq!(latch.key_up(ms(5_100)), Action::None);
        // The stop press does not seed a new double-tap.
        assert_eq!(latch.key_down(ms(5_200)), Action::Start);
        assert_eq!(latch.key_up(ms(5_300)), Action::Stop);
        assert!(!latch.latched());
    }

    #[test]
    fn second_press_held_long_is_a_hold_not_a_latch() {
        let mut latch = HoldKeyTapLatch::default();
        assert_eq!(latch.key_down(ms(0)), Action::Start);
        assert_eq!(latch.key_up(ms(150)), Action::Stop);
        assert_eq!(latch.key_down(ms(400)), Action::Start);
        assert_eq!(latch.key_up(ms(1_200)), Action::Stop);
        assert!(!latch.latched());
    }

    #[test]
    fn second_tap_outside_window_does_not_latch() {
        let mut latch = HoldKeyTapLatch::default();
        assert_eq!(latch.key_down(ms(0)), Action::Start);
        assert_eq!(latch.key_up(ms(150)), Action::Stop);
        // The window runs from the first release to the second press.
        assert_eq!(latch.key_down(ms(150 + 351)), Action::Start);
        assert_eq!(latch.key_up(ms(650)), Action::Stop);
        assert!(!latch.latched());
    }

    #[test]
    fn first_press_held_long_does_not_seed_a_double_tap() {
        let mut latch = HoldKeyTapLatch::default();
        assert_eq!(latch.key_down(ms(0)), Action::Start);
        assert_eq!(latch.key_up(ms(600)), Action::Stop);
        assert_eq!(latch.key_down(ms(700)), Action::Start);
        assert_eq!(latch.key_up(ms(800)), Action::Stop);
        assert!(!latch.latched());
    }

    #[test]
    fn thresholds_are_inclusive() {
        let mut latch = HoldKeyTapLatch::new(ms(250), ms(350));
        assert_eq!(latch.key_down(ms(0)), Action::Start);
        assert_eq!(latch.key_up(ms(250)), Action::Stop);
        assert_eq!(latch.key_down(ms(600)), Action::Start);
        assert_eq!(latch.key_up(ms(850)), Action::None);
        assert!(latch.latched());
    }

    #[test]
    fn reset_clears_latch_and_pending_tap() {
        let mut latch = HoldKeyTapLatch::default();
        assert_eq!(latch.key_down(ms(0)), Action::Start);
        assert_eq!(latch.key_up(ms(100)), Action::Stop);
        latch.reset();
        assert_eq!(latch.key_down(ms(200)), Action::Start);
        assert_eq!(latch.key_up(ms(300)), Action::Stop);
        assert!(!latch.latched());
        // Latched, then reset because something else ended the recording.
        assert_eq!(latch.key_down(ms(400)), Action::Start);
        assert_eq!(latch.key_up(ms(500)), Action::None);
        assert!(latch.latched());
        latch.reset();
        assert!(!latch.latched());
        assert_eq!(latch.key_down(ms(9_000)), Action::Start);
    }

    /// The hook can deliver a repeat key-down while the key is held. It must not
    /// restart the capture or lose the original press time.
    #[test]
    fn a_clock_that_goes_backwards_is_not_a_tap() {
        let mut latch = HoldKeyTapLatch::default();
        assert_eq!(latch.key_down(ms(1_000)), Action::Start);
        assert_eq!(latch.key_up(ms(900)), Action::Stop);
        assert!(!latch.latched());
    }
}
