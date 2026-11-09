//! Hold or tap for one dictation key.
//!
//! This is the Rust port of the Swift `HoldKeyTapLatch` in `bar/Sources/PhononBar.swift`.
//! macOS drives it from the Globe (fn) key. Windows drives it from Right Ctrl through
//! a `WH_KEYBOARD_LL` hook. Both platforms must behave the same, so the rules live
//! here and both test suites cover the same cases.
//!
//! Rules:
//! - Press starts a capture. Release stops it. That is hold-to-talk.
//! - A short press and release is a tap. A tap latches the capture on (press to
//!   toggle): the recording that began at the press keeps going.
//! - While latched, the next press stops the capture. Its release does nothing.
//!
//! There is no double-tap: one tap starts, the next tap stops.

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

/// The state machine. Timestamps are monotonic nanoseconds; the caller supplies them.
#[derive(Debug, Clone)]
pub struct HoldKeyTapLatch {
    tap_max_ns: u64,
    latched: bool,
    down_at: Option<u64>,
}

impl Default for HoldKeyTapLatch {
    fn default() -> Self {
        Self::new(DEFAULT_TAP_MAX_NS)
    }
}

impl HoldKeyTapLatch {
    /// A latch with an explicit tap threshold. The bound is inclusive.
    pub fn new(tap_max_ns: u64) -> Self {
        Self {
            tap_max_ns,
            latched: false,
            down_at: None,
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
        if is_tap {
            self.latched = true;
            return Action::None;
        }
        Action::Stop
    }

    /// Forget the latch and any pending press. Call this when something else ended
    /// the recording, such as a tray menu item or an error.
    pub fn reset(&mut self) {
        self.latched = false;
        self.down_at = None;
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
    fn one_tap_latches_and_the_next_press_stops() {
        let mut latch = HoldKeyTapLatch::default();
        assert_eq!(latch.key_down(ms(0)), Action::Start);
        assert_eq!(latch.key_up(ms(120)), Action::None);
        assert!(latch.latched());
        // While latched the press itself stops; its release is ignored.
        assert_eq!(latch.key_down(ms(4_000)), Action::Stop);
        assert!(!latch.latched());
        assert_eq!(latch.key_up(ms(4_100)), Action::None);
        // And again: the stop press leaves nothing pending.
        assert_eq!(latch.key_down(ms(4_300)), Action::Start);
        assert_eq!(latch.key_up(ms(4_400)), Action::None);
        assert!(latch.latched());
    }

    /// Two quick taps used to latch; now the first latches and the second stops.
    #[test]
    fn a_double_tap_is_a_start_then_a_stop() {
        let mut latch = HoldKeyTapLatch::default();
        assert_eq!(latch.key_down(ms(0)), Action::Start);
        assert_eq!(latch.key_up(ms(150)), Action::None);
        assert_eq!(latch.key_down(ms(400)), Action::Stop);
        assert_eq!(latch.key_up(ms(550)), Action::None);
        assert!(!latch.latched());
    }

    #[test]
    fn a_hold_after_a_hold_is_push_to_talk() {
        let mut latch = HoldKeyTapLatch::default();
        assert_eq!(latch.key_down(ms(0)), Action::Start);
        assert_eq!(latch.key_up(ms(600)), Action::Stop);
        assert_eq!(latch.key_down(ms(700)), Action::Start);
        assert_eq!(latch.key_up(ms(2_700)), Action::Stop);
        assert!(!latch.latched());
    }

    #[test]
    fn threshold_is_inclusive() {
        let mut latch = HoldKeyTapLatch::new(ms(250));
        assert_eq!(latch.key_down(ms(0)), Action::Start);
        assert_eq!(latch.key_up(ms(250)), Action::None);
        assert!(latch.latched());
        latch.reset();
        assert_eq!(latch.key_down(ms(1_000)), Action::Start);
        assert_eq!(latch.key_up(ms(1_251)), Action::Stop);
        assert!(!latch.latched());
    }

    #[test]
    fn reset_clears_the_latch() {
        let mut latch = HoldKeyTapLatch::default();
        assert_eq!(latch.key_down(ms(400)), Action::Start);
        assert_eq!(latch.key_up(ms(500)), Action::None);
        assert!(latch.latched());
        // Something else ended the recording: the next press starts, not stops.
        latch.reset();
        assert!(!latch.latched());
        assert_eq!(latch.key_down(ms(9_000)), Action::Start);
    }

    #[test]
    fn a_clock_that_goes_backwards_is_not_a_tap() {
        let mut latch = HoldKeyTapLatch::default();
        assert_eq!(latch.key_down(ms(1_000)), Action::Start);
        assert_eq!(latch.key_up(ms(900)), Action::Stop);
        assert!(!latch.latched());
    }
}
