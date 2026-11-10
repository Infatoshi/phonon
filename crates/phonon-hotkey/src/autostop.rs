//! When a latched dictation stops on its own.
//!
//! The Rust port of the Swift `LatchAutoStop` and `VoiceDetector` in `bar/Sources`, so a stray
//! tap cannot leave the mic recording until the next press on any desktop. A held key never
//! stops this way (its release does): [`AutoStop::decision`] answers only for a latched
//! recording.
//!
//! - No speech at all [`NO_SPEECH_SECONDS`] after the start: discard (no paste, no history).
//! - Speech, then [`SILENCE_AFTER_SPEECH_SECONDS`] of silence: commit, as the next tap would.
//!
//! Speech is the level meter against the room's floor ([`VoiceDetector`]), the same signal the
//! Mac capsule answers to, but only once it has lasted [`SPEECH_RUN_SECONDS`] without a break:
//! a keystroke or a click near the mic is one or two meter readings. Timestamps are seconds on
//! any monotonic clock; the caller supplies them.

/// No speech at all this long after the start. Someone who tapped on purpose starts talking
/// within a second or two (the start cue says it is listening); 8 s leaves room to gather a
/// thought and still frees the mic soon after a stray tap.
pub const NO_SPEECH_SECONDS: f64 = 8.0;
/// Silence this long after speech. Someone dictating a long prompt stops to read or think for
/// tens of seconds, and a commit pastes into whatever is focused, so this is a full minute;
/// someone who has walked away still gets their words committed.
pub const SILENCE_AFTER_SPEECH_SECONDS: f64 = 60.0;
/// Voice this long without a break counts as speech: about a syllable, and longer than a
/// keystroke. Time, not a count of readings, because the reading rate follows the device.
pub const SPEECH_RUN_SECONDS: f64 = 0.15;
/// Voice that has begun but not yet lasted [`SPEECH_RUN_SECONDS`] holds a limit off while
/// readings keep arriving this recently, so talking right at a limit is not cut. A mic that
/// stopped mid-voice must not hold it off for ever.
pub const VOICE_HOLD_SECONDS: f64 = 1.0;

/// What to do with a latched recording now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    KeepRecording,
    /// Nothing was said since the start: drop it, with no paste and no history entry.
    Discard,
    /// Speech, then a long silence: stop and commit as the next press would.
    Commit,
}

/// The decision rule, fed one voice reading at a time.
#[derive(Debug, Clone)]
pub struct LatchAutoStop {
    started_at: f64,
    last_speech_at: Option<f64>,
    voice_since: Option<f64>,
    last_voice_at: Option<f64>,
}

impl LatchAutoStop {
    pub fn new(started_at: f64) -> Self {
        Self {
            started_at,
            last_speech_at: None,
            voice_since: None,
            last_voice_at: None,
        }
    }

    pub fn heard_speech(&self) -> bool {
        self.last_speech_at.is_some()
    }

    pub fn last_speech_at(&self) -> Option<f64> {
        self.last_speech_at
    }

    /// One reading of the voice detector.
    pub fn note(&mut self, voice: bool, now: f64) {
        if !voice {
            self.voice_since = None;
            return;
        }
        let since = *self.voice_since.get_or_insert(now);
        self.last_voice_at = Some(now);
        if now - since >= SPEECH_RUN_SECONDS {
            self.last_speech_at = Some(now);
        }
    }

    pub fn decision(&self, now: f64) -> Decision {
        if self.voice_since.is_some()
            && self
                .last_voice_at
                .is_some_and(|at| now - at < VOICE_HOLD_SECONDS)
        {
            return Decision::KeepRecording;
        }
        match self.last_speech_at {
            Some(at) if now - at >= SILENCE_AFTER_SPEECH_SECONDS => Decision::Commit,
            Some(_) => Decision::KeepRecording,
            None if now - self.started_at >= NO_SPEECH_SECONDS => Decision::Discard,
            None => Decision::KeepRecording,
        }
    }
}

/// Speech vs room noise on a level meter (0 = -55 dBFS, 1 = 0 dBFS), as on the Mac: a level is
/// voice when it clears an absolute floor and stands well above this dictation's noise floor,
/// which follows the quietest levels down at once and drifts up slowly.
#[derive(Debug, Clone)]
pub struct VoiceDetector {
    floor: f64,
}

impl Default for VoiceDetector {
    /// Starts at a quiet room (-47 dBFS), so speech from the first instant counts. The gaps
    /// between words pull it back down, so steady speech never becomes the floor.
    fn default() -> Self {
        Self { floor: 0.15 }
    }
}

impl VoiceDetector {
    /// About -41 dBFS.
    pub const MINIMUM: f64 = 0.25;
    /// About 10 dB above the room.
    pub const MARGIN: f64 = 0.18;

    pub fn is_voice(&mut self, level: f64) -> bool {
        self.floor = if level < self.floor {
            level
        } else {
            self.floor + (level - self.floor) * 0.002
        };
        level >= Self::MINIMUM.max(self.floor + Self::MARGIN)
    }
}

/// The Mac meter's level for one block of samples: RMS in dBFS, -55..0 mapped to 0..1.
pub fn level(samples: &[i16]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples
        .iter()
        .map(|&s| {
            let x = f64::from(s) / 32768.0;
            x * x
        })
        .sum();
    let rms = (sum / samples.len() as f64).sqrt();
    let db = if rms > 1e-9 {
        20.0 * rms.log10()
    } else {
        -60.0
    };
    ((db + 55.0) / 55.0).clamp(0.0, 1.0)
}

/// Captured samples in, decisions out: the Mac's meter path for any capture. The Mac meter
/// takes one 1,024-frame buffer at 48 kHz (21.3 ms), smooths it into the meter (0.18 old,
/// 0.82 new) and hands every second value to the voice detector, so a reading is about 43 ms
/// of audio. This cuts the samples into blocks of the same duration at the capture's rate and
/// stamps each reading with its audio position, so readings that arrive in a burst (a file
/// tailed every half second) still time the speech run correctly.
#[derive(Debug, Clone)]
pub struct AutoStop {
    stop: LatchAutoStop,
    detector: VoiceDetector,
    sample_rate: u32,
    block: usize,
    pending: Vec<i16>,
    fed: usize,
    blocks: u64,
    meter: f64,
}

impl AutoStop {
    /// A recording that started at 0 s on the caller's clock, captured at `sample_rate`.
    pub fn new(sample_rate: u32) -> Self {
        let block = (sample_rate as usize * 1024 / 48_000).max(1);
        Self {
            stop: LatchAutoStop::new(0.0),
            detector: VoiceDetector::default(),
            sample_rate: sample_rate.max(1),
            block,
            pending: Vec::with_capacity(block),
            fed: 0,
            blocks: 0,
            meter: 0.0,
        }
    }

    /// How many samples have been fed so far.
    pub fn fed(&self) -> usize {
        self.fed
    }

    pub fn heard_speech(&self) -> bool {
        self.stop.heard_speech()
    }

    /// Feed captured mono samples, in order.
    pub fn feed(&mut self, samples: &[i16]) {
        self.fed += samples.len();
        let mut rest = samples;
        while !rest.is_empty() {
            let take = (self.block - self.pending.len()).min(rest.len());
            self.pending.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.pending.len() == self.block {
                self.meter = 0.18 * self.meter + 0.82 * level(&self.pending);
                self.pending.clear();
                self.blocks += 1;
                if self.blocks.is_multiple_of(2) {
                    let at = (self.blocks * self.block as u64) as f64 / f64::from(self.sample_rate);
                    let voice = self.detector.is_voice(self.meter);
                    self.stop.note(voice, at);
                }
            }
        }
    }

    /// The decision at `now` seconds since the recording started. Only a latched recording
    /// stops on its own: a held key (or a recording the latch does not own) always keeps going.
    pub fn decision(&self, latched: bool, now: f64) -> Decision {
        if !latched {
            return Decision::KeepRecording;
        }
        self.stop.decision(now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Meter readings arrive about every 43 ms.
    const READING: f64 = 0.043;

    /// Voice on every reading from `from` for `seconds`.
    fn speak(stop: &mut LatchAutoStop, from: f64, seconds: f64) {
        let mut t = from;
        while t <= from + seconds {
            stop.note(true, t);
            t += READING;
        }
        stop.note(false, t);
    }

    #[test]
    fn a_stray_tap_with_no_speech_is_discarded_after_the_limit() {
        let mut stop = LatchAutoStop::new(100.0);
        let mut t = 100.0;
        while t < 100.0 + NO_SPEECH_SECONDS - 0.1 {
            stop.note(false, t);
            assert_eq!(stop.decision(t), Decision::KeepRecording);
            t += READING;
        }
        assert_eq!(stop.decision(100.0 + NO_SPEECH_SECONDS), Decision::Discard);
        assert!(!stop.heard_speech());
    }

    #[test]
    fn no_meter_readings_at_all_still_discards() {
        // A mic that delivers no buffers never feeds a reading; the timer still decides.
        let stop = LatchAutoStop::new(0.0);
        assert_eq!(stop.decision(7.9), Decision::KeepRecording);
        assert_eq!(stop.decision(8.0), Decision::Discard);
    }

    #[test]
    fn keystrokes_are_not_speech() {
        let mut stop = LatchAutoStop::new(0.0);
        // Typing: one voiced reading per key, quiet between, for the whole window.
        let mut t = 0.0;
        while t < 10.0 {
            stop.note(true, t);
            stop.note(false, t + READING);
            t += 0.2;
        }
        assert!(!stop.heard_speech());
        assert_eq!(stop.decision(10.0), Decision::Discard);
    }

    #[test]
    fn speech_then_long_silence_commits() {
        let mut stop = LatchAutoStop::new(0.0);
        speak(&mut stop, 1.0, 3.0);
        assert!(stop.heard_speech());
        let last = stop.last_speech_at().unwrap();
        // Past the no-speech limit, but speech was heard, so it keeps going.
        assert_eq!(stop.decision(12.0), Decision::KeepRecording);
        assert_eq!(
            stop.decision(last + SILENCE_AFTER_SPEECH_SECONDS - 0.1),
            Decision::KeepRecording
        );
        assert_eq!(
            stop.decision(last + SILENCE_AFTER_SPEECH_SECONDS),
            Decision::Commit
        );
    }

    #[test]
    fn speech_after_a_pause_restarts_the_silence_clock() {
        let mut stop = LatchAutoStop::new(0.0);
        speak(&mut stop, 1.0, 2.0);
        speak(&mut stop, 25.0, 2.0);
        // 33 s after the first speech, but only about 6 s after the second.
        assert_eq!(stop.decision(33.0), Decision::KeepRecording);
        assert_eq!(
            stop.decision(27.0 + SILENCE_AFTER_SPEECH_SECONDS + 0.1),
            Decision::Commit
        );
    }

    #[test]
    fn speech_just_before_the_no_speech_limit_keeps_recording() {
        let mut stop = LatchAutoStop::new(0.0);
        speak(&mut stop, 7.5, 0.4);
        assert!(stop.heard_speech());
        assert_eq!(stop.decision(8.5), Decision::KeepRecording);
    }

    /// Voice that began just before a limit holds it off while readings keep arriving, and
    /// stops holding it off once they stop (a mic that died mid-voice).
    #[test]
    fn voice_just_beginning_holds_the_limit_off() {
        let mut stop = LatchAutoStop::new(0.0);
        stop.note(true, 7.95);
        assert!(!stop.heard_speech());
        assert_eq!(stop.decision(8.0), Decision::KeepRecording);
        assert_eq!(
            stop.decision(9.5),
            Decision::Discard,
            "no reading for over a second"
        );
    }

    #[test]
    fn the_limits_are_the_chosen_defaults() {
        assert_eq!(NO_SPEECH_SECONDS, 8.0);
        assert_eq!(SILENCE_AFTER_SPEECH_SECONDS, 60.0);
        assert_eq!(SPEECH_RUN_SECONDS, 0.15);
        assert_eq!(VOICE_HOLD_SECONDS, 1.0);
        const { assert!(NO_SPEECH_SECONDS < SILENCE_AFTER_SPEECH_SECONDS) };
    }

    #[test]
    fn voice_detector_tells_speech_from_room_noise() {
        let mut detector = VoiceDetector::default();
        // A fan at about -44 dBFS (0.2) for 2 s.
        for _ in 0..100 {
            assert!(!detector.is_voice(0.2));
        }
        assert!(detector.is_voice(0.55), "speech at about -25 dBFS");
        let mut fresh = VoiceDetector::default();
        assert!(fresh.is_voice(0.5), "speech from the first instant");
    }

    const RATE: u32 = 16_000;

    fn seconds(s: f64) -> usize {
        (s * f64::from(RATE)) as usize
    }

    /// A 220 Hz tone at `dbfs`, standing in for a voice.
    fn tone(s: f64, dbfs: f64) -> Vec<i16> {
        let amp = 10f64.powf(dbfs / 20.0) * 32767.0 * std::f64::consts::SQRT_2;
        (0..seconds(s))
            .map(|i| {
                let t = i as f64 / f64::from(RATE);
                (amp * (2.0 * std::f64::consts::PI * 220.0 * t).sin()) as i16
            })
            .collect()
    }

    /// Uniform white noise with an RMS of about `dbfs`, from a fixed seed.
    fn noise(s: f64, dbfs: f64) -> Vec<i16> {
        let amp = 10f64.powf(dbfs / 20.0) * 32767.0 * 3f64.sqrt();
        let mut x: u32 = 0x1234_5678;
        (0..seconds(s))
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                let u = f64::from(x) / f64::from(u32::MAX) * 2.0 - 1.0;
                (amp * u) as i16
            })
            .collect()
    }

    #[test]
    fn level_maps_dbfs_like_the_mac_meter() {
        assert_eq!(level(&[]), 0.0);
        assert_eq!(level(&[0; 512]), 0.0);
        let half = level(&tone(0.1, -27.5));
        assert!((half - 0.5).abs() < 0.01, "{half}");
        assert!(level(&[i16::MAX; 64]) > 0.99);
    }

    #[test]
    fn a_tone_counts_as_speech() {
        let mut auto = AutoStop::new(RATE);
        auto.feed(&tone(1.0, -20.0));
        assert!(auto.heard_speech());
        assert_eq!(auto.fed(), seconds(1.0));
    }

    #[test]
    fn silence_and_quiet_room_noise_are_not_speech() {
        let mut quiet = AutoStop::new(RATE);
        quiet.feed(&vec![0; seconds(9.0)]);
        assert!(!quiet.heard_speech());
        assert_eq!(quiet.decision(true, 9.0), Decision::Discard);

        let mut fan = AutoStop::new(48_000);
        let room = noise(9.0 * 3.0, -46.0); // 9 s at 48 kHz
        fan.feed(&room);
        assert!(!fan.heard_speech());
        assert_eq!(fan.decision(true, 9.0), Decision::Discard);
    }

    #[test]
    fn a_click_is_not_speech() {
        let mut auto = AutoStop::new(RATE);
        auto.feed(&vec![0; seconds(2.0)]);
        auto.feed(&tone(0.05, -10.0));
        auto.feed(&vec![0; seconds(7.0)]);
        assert!(!auto.heard_speech());
        assert_eq!(auto.decision(true, 9.0), Decision::Discard);
    }

    #[test]
    fn speech_then_a_minute_of_silence_commits_from_samples() {
        let mut auto = AutoStop::new(RATE);
        auto.feed(&vec![0; seconds(1.0)]);
        auto.feed(&tone(2.0, -20.0));
        // Fed in uneven pieces, as a capture delivers them.
        let silence = vec![0; seconds(61.0)];
        for piece in silence.chunks(777) {
            auto.feed(piece);
        }
        assert!(auto.heard_speech());
        assert_eq!(auto.decision(true, 30.0), Decision::KeepRecording);
        assert_eq!(auto.decision(true, 64.0), Decision::Commit);
    }

    /// Hold-to-talk: whatever the audio and however long, the key's release ends it.
    #[test]
    fn a_held_recording_is_never_stopped() {
        let mut silent = AutoStop::new(RATE);
        silent.feed(&vec![0; seconds(20.0)]);
        let mut spoke = AutoStop::new(RATE);
        spoke.feed(&tone(1.0, -20.0));
        spoke.feed(&vec![0; seconds(70.0)]);
        for now in [0.0, 8.0, 9.0, 20.0, 71.0, 600.0, 3_600.0] {
            assert_eq!(silent.decision(false, now), Decision::KeepRecording);
            assert_eq!(spoke.decision(false, now), Decision::KeepRecording);
        }
        // The same recordings, latched, would have stopped.
        assert_eq!(silent.decision(true, 20.0), Decision::Discard);
        assert_eq!(spoke.decision(true, 71.0), Decision::Commit);
    }
}
