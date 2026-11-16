import Foundation

/// Ties a result to the dictation it answers. The keyboard writes a fresh token to IPC before it signals start, the
/// app reads it at the start and writes it back beside the result, and the keyboard takes a result only when its token
/// is the one the keyboard is waiting for. A late result from a cancelled dictation (its transcription was already
/// running) is then never inserted into the next dictation, nor does it eat that dictation's real result.
enum DictationToken {
    static func make() -> String { UUID().uuidString }

    /// True when `result` (the token written with the result) answers the dictation waiting for `expected`.
    static func accepts(result: String?, waitingFor expected: String?) -> Bool {
        guard let expected, !expected.isEmpty, let result else { return false }
        return result == expected
    }
}

/// The app's side of the same rule: each utterance gets a generation; a cancel or the next start moves it on, and a
/// delivery from an earlier generation is dropped (no result written, no history row, no status change).
struct UtteranceGate {
    private(set) var generation = 0
    /// The keyboard's token for the current utterance, echoed with its result.
    private(set) var token: String?

    /// A new utterance starts: anything still in flight from the previous one is stale from now on.
    mutating func begin(token: String?) -> Int {
        generation += 1
        self.token = token
        return generation
    }

    /// The keyboard cancelled: the utterance's transcription may still finish, but its delivery is dropped.
    mutating func cancel() {
        generation += 1
        token = nil
    }

    func isCurrent(_ generation: Int) -> Bool { generation == self.generation }
}

extension PendingStart {
    /// `shouldResume`, except that a wish this keyboard wrote while it is still on screen is not acted on there: the
    /// user is on the way to Phonon, and a late pong in that same appearance must not light the bar and start a
    /// dictation behind them. The wish is acted on when the keyboard appears again, with Phonon's session up.
    static func shouldResume(wantedAt: Double, now: Double, alive: Bool, wroteThisAppearance: Bool) -> Bool {
        !wroteThisAppearance && shouldResume(wantedAt: wantedAt, now: now, alive: alive)
    }
}
