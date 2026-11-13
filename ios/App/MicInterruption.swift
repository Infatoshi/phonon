import Foundation

/// What the microphone session does about audio-session events: a phone call or Siri interrupting the mic, the
/// interruption ending, or iOS resetting its media services (which invalidates the audio engine). Plain logic so the
/// hostless unit tests cover it; Session.swift turns the actions into AVAudioSession / AVAudioEngine calls.
///
/// While `down`, the session must not report alive (no heartbeat, no pong): the keyboard then opens the app, whose
/// start() restarts the engine, instead of showing ready over a dead microphone.
struct MicInterruption: Equatable {
    enum Event: Equatable {
        case began
        case ended(shouldResume: Bool)
        case mediaServicesReset
    }

    enum Action: Equatable {
        /// nothing to do now
        case none
        /// the mic is gone: finish any utterance with what was heard, stop claiming alive
        case pause
        /// restart the engine (it is still valid)
        case restart
        /// build a new engine, then start it (media services were reset: the old one is unusable)
        case rebuildAndRestart
    }

    private(set) var down = false
    private(set) var needsNewEngine = false

    mutating func handle(_ event: Event, running: Bool) -> Action {
        switch event {
        case .began:
            guard running else { return .none }
            down = true
            return .pause
        case .ended(let shouldResume):
            // without shouldResume another app still holds audio: stay down, the next start() restarts
            guard running, down, shouldResume else { return .none }
            return needsNewEngine ? .rebuildAndRestart : .restart
        case .mediaServicesReset:
            needsNewEngine = true
            guard running else { return .none }
            down = true
            return .rebuildAndRestart
        }
    }

    /// The engine started again: the session is live.
    mutating func restarted() {
        down = false
        needsNewEngine = false
    }

    /// Alive only with the session on, no interruption pending, and the engine actually running (iOS also stops the
    /// engine on its own, e.g. on an audio configuration change).
    func alive(running: Bool, engineRunning: Bool) -> Bool { running && !down && engineRunning }
}
