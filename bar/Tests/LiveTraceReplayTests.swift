import XCTest

@testable import PhononBar

/// Replays recorded dictations (live.jsonl lines) through the card and counts real word
/// changes by how far back they reach, words that vanished, and spots that flipped back.
/// Runs only with PHONON_LIVE_REPLAY=<path to a live.jsonl>, PHONON_LIVE_REPLAY_LINE=<1-based
/// line, or "all"; default the last>.
final class LiveTraceReplayTests: XCTestCase {
    /// A change starting more than this many words from the line's end counts as far back.
    static let farBack = 4

    private struct Line: Decodable {
        let started: String
        let events: [Event]
    }

    private struct Event: Decodable {
        let ms: Int
        let kind: String
        let applied: Bool?
        let at: Int?
        let words: String?
        let covered: Int?
    }

    @MainActor
    func testReplayRecordedDictation() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["PHONON_LIVE_REPLAY"] else {
            throw XCTSkip("set PHONON_LIVE_REPLAY to a live.jsonl")
        }
        let lines = try String(contentsOfFile: path, encoding: .utf8)
            .split(separator: "\n")
        let pick = env["PHONON_LIVE_REPLAY_LINE"] ?? ""
        let indices = pick == "all" ? Array(1...lines.count) : [Int(pick) ?? lines.count]
        var totals = (near: 0, far: 0, frozen: 0, vanished: 0, flips: 0)
        for index in indices {
            let line = try JSONDecoder().decode(Line.self, from: Data(lines[index - 1].utf8))
            let r = replay(line, debugMs: env["PHONON_LIVE_REPLAY_DEBUG_MS"])
            totals.near += r.near
            totals.far += r.far.count
            totals.frozen += r.frozen
            totals.vanished += r.vanished
            totals.flips += r.flips
            print(
                "replay \(index) \(line.started): \(r.near) changes within \(Self.farBack) words of the end, "
                    + "\(r.far.count) further back (\(r.frozen) past \(LiveTextMotion.freezeReach)), "
                    + "\(r.vanished) vanished, \(r.flips) flipped back")
            for row in r.far { print("  " + row) }
        }
        print(
            "replay total: near \(totals.near), far \(totals.far), past freeze \(totals.frozen), "
                + "vanished \(totals.vanished), flipped back \(totals.flips)")
    }

    @MainActor
    private func replay(_ line: Line, debugMs: String?) -> (
        near: Int, far: [String], frozen: Int, vanished: Int, flips: Int
    ) {
        var clock: TimeInterval = 1000
        let state = LiveTextState(clock: { clock })
        var inputs: [String: [String]] = [:]
        var shown: [String] = []
        var near = 0
        var far: [String] = []
        var frozen = 0
        var vanished = 0
        var flips = 0
        // what each card position held before its last change
        var before: [Int: String] = [:]
        for event in line.events {
            clock = 1000 + Double(event.ms) / 1000
            let family = event.kind.hasPrefix("draft") ? "draft" : event.kind
            guard let at = event.at else { continue }
            var words = Array((inputs[family] ?? []).prefix(at))
            words += LiveTextPolicy.words(event.words ?? "")
            inputs[family] = words
            let text = words.joined(separator: " ")
            switch event.kind {
            case "partial" where event.applied == true:
                state.noteVoice()
                state.setTranscript(text)
            case "draft" where event.applied == true:
                state.noteVoice()
                let before = state.correctedWords.joined(separator: " ")
                // the draft's source: draftMatches let it through only if it agrees with the
                // partial shown, so it is that partial's first `covered` words
                let partial = LiveTextPolicy.openWords((inputs["partial"] ?? []).joined(separator: " "))
                let source = Array(partial.prefix(event.covered ?? 0))
                state.applyDraft(
                    text, coveredWords: event.covered ?? 0, complete: true,
                    source: ProcessInfo.processInfo.environment["PHONON_LIVE_REPLAY_COUNT"] == nil
                        ? source : nil)
                if debugMs == String(event.ms) {
                    print("debug before: \(before)\ndebug draft:  \(text)\ndebug after:  \(state.correctedText)")
                }
            default:
                continue
            }
            let now = state.shownWords
            if now.count < shown.count {
                vanished += shown.count - now.count
                if ProcessInfo.processInfo.environment["PHONON_LIVE_REPLAY_VANISH"] != nil {
                    let k = max(shown.count - 10, 0)
                    print(
                        String(format: "  vanish %5.1f s %@ -%d: ", Double(event.ms) / 1000, event.kind, shown.count - now.count)
                            + "'\(shown[k...].joined(separator: " "))' -> '\(now[min(k, now.count)...].joined(separator: " "))'")
                }
            }
            let n = min(shown.count, now.count)
            let differ = (0..<n).filter {
                LiveTextPolicy.normalized(shown[$0]) != LiveTextPolicy.normalized(now[$0])
            }
            for i in differ {
                let new = LiveTextPolicy.normalized(now[i])
                if before[i] == new {
                    flips += 1
                    if ProcessInfo.processInfo.environment["PHONON_LIVE_REPLAY_VANISH"] != nil {
                        let k = max(i - 4, 0)
                        print(
                            String(format: "  flip %5.1f s %@ @%d of %d: ", Double(event.ms) / 1000, event.kind, i, now.count)
                                + "'\(shown[k..<min(i + 3, shown.count)].joined(separator: " "))' -> '\(now[k..<min(i + 3, now.count)].joined(separator: " "))'")
                    }
                }
                before[i] = LiveTextPolicy.normalized(shown[i])
            }
            if let first = differ.first {
                let back = shown.count - first
                if back > Self.farBack {
                    if back > LiveTextMotion.freezeReach { frozen += 1 }
                    far.append(
                        String(format: "%5.1f s  %d back  ", Double(event.ms) / 1000, back)
                            + "\(event.kind) cov \(event.covered ?? 0)  "
                            + "'\(shown[first..<min(first + 6, shown.count)].joined(separator: " "))' -> "
                            + "'\(now[first..<min(first + 6, now.count)].joined(separator: " "))'")
                } else {
                    near += 1
                }
            }
            shown = now
        }
        return (near, far, frozen, vanished, flips)
    }
}
