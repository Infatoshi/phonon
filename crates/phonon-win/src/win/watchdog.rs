//! Pure decisions behind the keyboard hook's watchdog (`hook::watchdog_tick`).
//!
//! Windows drops a low-level keyboard hook without telling anyone: when the hook
//! misses its timeout, when another app (Tauri 2, raw-input games) installs its
//! own hook in front, or after a session switch. The symptom is a dictation key
//! that does nothing while the rest of the keyboard works. Everything here is
//! plain arithmetic on millisecond stamps so it is tested on every platform;
//! `hook.rs` feeds it `GetTickCount`, `GetLastInputInfo` and the foreground PID.
//!
//! This file builds without Windows: `paths.rs` includes it in non-Windows test
//! builds, so it must not name `windows_sys` or anything else in `win`.

/// Stamped into `dwExtraInfo` of the inputs Phonon sends itself (the menu-bar
/// mask), so the hook passes them through untouched. ASCII "PHON".
pub const PHONON_INJECTED_EXTRA_INFO: usize = 0x5048_4F4E;

/// An unassigned virtual key. Tapping it between an Alt or Win press and its
/// release tells Windows the modifier was used, so the release does not open a
/// window's menu bar (Chrome, Electron) or the Start menu.
pub const VK_MENU_MASK: u16 = 0xE8;

/// Input in the session counts as "now" within this window.
const RECENT_INPUT_MS: u64 = 1_000;
/// The first wait before re-arming a silent hook; doubles per failed re-arm.
const BASE_BACKOFF_MS: u64 = 3_000;
const MAX_BACKOFF_MS: u64 = 60_000;
const MAX_BACKOFF_SHIFT: u32 = 5;
/// Past the keyboard's repeat delay, a held key has sent a repeat.
const REPEAT_SLACK_MS: u64 = 250;
const MIN_REPEAT_DELAY_MS: u64 = 250;
const MAX_REPEAT_DELAY_MS: u64 = 1_200;

const VK_MENU: u32 = 0x12;
const VK_LWIN: u32 = 0x5B;
const VK_RWIN: u32 = 0x5C;
const VK_LCONTROL: u32 = 0xA2;
const VK_LMENU: u32 = 0xA4;
const VK_RMENU: u32 = 0xA5;
/// `LLKHF_EXTENDED` in `KBDLLHOOKSTRUCT::flags`.
const LLKHF_EXTENDED: u32 = 0x01;

/// Milliseconds from `earlier` to `now` on the 32-bit tick clock
/// (`GetTickCount`, `LASTINPUTINFO::dwTime`), which wraps every 49.7 days.
pub fn tick_distance_u32(now: u32, earlier: u32) -> u32 {
    now.wrapping_sub(earlier)
}

/// The keyboard repeat delay from `SPI_GETKEYBOARDDELAY` (0 to 3, a quarter
/// second each past the first), clamped to what the heal expects.
pub fn repeat_delay_from_spi(level: u32) -> u64 {
    (u64::from(level) + 1)
        .saturating_mul(250)
        .clamp(MIN_REPEAT_DELAY_MS, MAX_REPEAT_DELAY_MS)
}

/// Whether a hook event is the Left Control that Windows synthesizes in front
/// of Right Alt on AltGr layouts. It is not a key the user pressed.
pub fn is_altgr_synthetic_lctrl(vk_code: u32, scan_code: u32, flags: u32) -> bool {
    vk_code == VK_LCONTROL
        && (scan_code == 0x21D || (scan_code == 0x1D && flags & LLKHF_EXTENDED != 0))
}

/// Whether swallowing this key's release needs the menu-bar mask: a lone Alt
/// release activates the focused window's menu bar, a lone Win release opens Start.
pub fn needs_menu_bar_mask(watched_vk: u32) -> bool {
    matches!(
        watched_vk,
        VK_MENU | VK_LMENU | VK_RMENU | VK_LWIN | VK_RWIN
    )
}

/// When to re-install a hook that has gone quiet while the user is typing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookWatchdog {
    last_hook_ms: u64,
    silent_streak: u32,
    last_rearm_ms: Option<u64>,
}

impl HookWatchdog {
    /// A watchdog for a hook installed at `now_ms`.
    pub fn new(now_ms: u64) -> Self {
        Self {
            last_hook_ms: now_ms,
            silent_streak: 0,
            last_rearm_ms: None,
        }
    }

    pub fn last_hook_ms(&self) -> u64 {
        self.last_hook_ms
    }

    /// Re-arms since the hook last answered.
    pub fn silent_streak(&self) -> u32 {
        self.silent_streak
    }

    /// The hook answered at `now_ms`: it is alive, so the backoff starts over.
    pub fn record_hook_event(&mut self, now_ms: u64) {
        self.last_hook_ms = now_ms;
        self.silent_streak = 0;
        self.last_rearm_ms = None;
    }

    /// A re-arm was attempted at `now_ms`; the next waits twice as long.
    pub fn record_rearm(&mut self, now_ms: u64) {
        self.silent_streak = self.silent_streak.saturating_add(1);
        self.last_rearm_ms = Some(now_ms);
    }

    /// 3 s, 6 s, 12 s, 24 s, 48 s, then 60 s.
    pub fn backoff_ms(&self) -> u64 {
        (BASE_BACKOFF_MS << self.silent_streak.min(MAX_BACKOFF_SHIFT)).min(MAX_BACKOFF_MS)
    }

    /// Whether the hook should be re-installed: someone is typing in the session,
    /// the hook has heard nothing for the backoff, and the last re-arm is at least
    /// that old. Never while Phonon's own window is in front.
    pub fn should_rearm(
        &self,
        now_ms: u64,
        last_session_input_ms: u64,
        own_window_in_front: bool,
    ) -> bool {
        if own_window_in_front {
            return false;
        }
        let backoff = self.backoff_ms();
        let input_recent = now_ms.saturating_sub(last_session_input_ms) <= RECENT_INPUT_MS;
        let hook_silent = now_ms.saturating_sub(self.last_hook_ms) >= backoff;
        let rearm_old = self
            .last_rearm_ms
            .is_none_or(|at| now_ms.saturating_sub(at) >= backoff);
        input_recent && hook_silent && rearm_old
    }

    /// Whether a dictation key Phonon thinks is held was really let go with the
    /// key-up lost. A held key repeats after `repeat_delay_ms`, so a hook quiet
    /// for longer than that (plus slack) with the key not down missed the release.
    pub fn should_heal_stuck_hold(
        now_ms: u64,
        held: bool,
        last_hook_ms: u64,
        key_physically_down: bool,
        repeat_delay_ms: u64,
    ) -> bool {
        let delay = repeat_delay_ms.clamp(MIN_REPEAT_DELAY_MS, MAX_REPEAT_DELAY_MS);
        held && !key_physically_down
            && now_ms.saturating_sub(last_hook_ms) > delay + REPEAT_SLACK_MS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_distance_survives_the_wrap() {
        assert_eq!(tick_distance_u32(5_000, 4_000), 1_000);
        assert_eq!(tick_distance_u32(500, u32::MAX - 499), 1_000);
        assert_eq!(tick_distance_u32(7, 7), 0);
    }

    #[test]
    fn the_backoff_doubles_to_a_minute() {
        let mut watchdog = HookWatchdog::new(0);
        let mut seen = Vec::new();
        for _ in 0..8 {
            seen.push(watchdog.backoff_ms());
            watchdog.record_rearm(0);
        }
        assert_eq!(
            seen,
            [3_000, 6_000, 12_000, 24_000, 48_000, 60_000, 60_000, 60_000]
        );
    }

    #[test]
    fn a_silent_hook_rearms_only_while_someone_types() {
        let watchdog = HookWatchdog::new(0);
        // Typing 200 ms ago, hook quiet for 3 s.
        assert!(watchdog.should_rearm(3_000, 2_800, false));
        // Nobody typed for 5 s: a quiet hook is just a quiet keyboard.
        assert!(!watchdog.should_rearm(10_000, 5_000, false));
        // Hook heard 2 s ago: not silent long enough.
        assert!(!watchdog.should_rearm(2_000, 2_000, false));
        // Phonon's own window in front.
        assert!(!watchdog.should_rearm(3_000, 3_000, true));
    }

    #[test]
    fn rearms_back_off_until_the_hook_answers() {
        let mut watchdog = HookWatchdog::new(0);
        assert!(watchdog.should_rearm(3_000, 3_000, false));
        watchdog.record_rearm(3_000);
        // Next wait is 6 s from the re-arm.
        assert!(!watchdog.should_rearm(8_000, 8_000, false));
        assert!(watchdog.should_rearm(9_000, 9_000, false));
        watchdog.record_rearm(9_000);
        assert_eq!(watchdog.silent_streak(), 2);
        assert!(!watchdog.should_rearm(20_000, 20_000, false));
        assert!(watchdog.should_rearm(21_000, 21_000, false));
        // A key reaches the hook: the streak and the wait reset.
        watchdog.record_hook_event(21_500);
        assert_eq!(watchdog.silent_streak(), 0);
        assert_eq!(watchdog.last_hook_ms(), 21_500);
        assert!(!watchdog.should_rearm(24_000, 24_000, false));
        assert!(watchdog.should_rearm(24_500, 24_500, false));
    }

    #[test]
    fn a_lost_key_up_heals_after_the_repeat_delay() {
        // Default delay (level 1 = 500 ms): heal past 750 ms of quiet.
        let delay = repeat_delay_from_spi(1);
        assert_eq!(delay, 500);
        assert!(!HookWatchdog::should_heal_stuck_hold(
            10_750, true, 10_000, false, delay
        ));
        assert!(HookWatchdog::should_heal_stuck_hold(
            10_751, true, 10_000, false, delay
        ));
        // Still down, or not held: leave it.
        assert!(!HookWatchdog::should_heal_stuck_hold(
            20_000, true, 10_000, true, delay
        ));
        assert!(!HookWatchdog::should_heal_stuck_hold(
            20_000, false, 10_000, false, delay
        ));
        // Odd delays are clamped to 250..1200 ms.
        assert!(HookWatchdog::should_heal_stuck_hold(
            1_001, true, 500, false, 0
        ));
        assert!(!HookWatchdog::should_heal_stuck_hold(
            1_900, true, 500, false, 5_000
        ));
        assert!(HookWatchdog::should_heal_stuck_hold(
            1_951, true, 500, false, 5_000
        ));
    }

    #[test]
    fn repeat_delay_levels_map_to_quarter_seconds() {
        assert_eq!(repeat_delay_from_spi(0), 250);
        assert_eq!(repeat_delay_from_spi(3), 1_000);
        assert_eq!(repeat_delay_from_spi(99), 1_200);
    }

    #[test]
    fn altgr_left_control_is_recognized() {
        assert!(is_altgr_synthetic_lctrl(0xA2, 0x21D, 0));
        assert!(is_altgr_synthetic_lctrl(0xA2, 0x1D, LLKHF_EXTENDED));
        // A real Left Control press.
        assert!(!is_altgr_synthetic_lctrl(0xA2, 0x1D, 0));
        // Right Control shares the scan code but not the key.
        assert!(!is_altgr_synthetic_lctrl(0xA3, 0x1D, LLKHF_EXTENDED));
    }

    #[test]
    fn alt_and_win_need_the_menu_mask() {
        for vk in [0x12, 0xA4, 0xA5, 0x5B, 0x5C] {
            assert!(needs_menu_bar_mask(vk), "{vk:#x}");
        }
        // Right Ctrl, Right Shift, Caps Lock, F13, Ctrl+Space.
        for vk in [0xA3, 0xA1, 0x14, 0x7C, 0x1_0020] {
            assert!(!needs_menu_bar_mask(vk), "{vk:#x}");
        }
    }

    #[test]
    fn the_injected_tag_spells_phon() {
        assert_eq!((PHONON_INJECTED_EXTRA_INFO as u32).to_be_bytes(), *b"PHON");
    }
}
