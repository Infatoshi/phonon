import Foundation

/// Dictations still being processed when the next one starts. Starting a new dictation used
/// to drop the earlier one's text on the floor (and, with history off, leave its audio on
/// disk). Now each is finished in the background with the same engine commands it would have
/// had, and its text goes to the "not pasted" card (copy button) instead of being pasted into
/// whatever has focus by then. With history off its WAV is deleted when it is done, whatever
/// the outcome.
///
/// Pure state: the controller feeds in engine events and carries out the returned steps.
struct EarlierDictations {
    enum Stage: Equatable {
        /// The WAV is still being written off the main thread.
        case writingWav
        /// The WAV is on disk; the transcribe waits for the engine.
        case awaitingTranscribe
        case transcribing
        /// The transcript is known and its correction was sent.
        case polishing(raw: String)
    }

    enum Step: Equatable {
        /// An engine command, as `[String: Any]` after `command(for:)`.
        case transcribe(path: String, id: String)
        case polish(raw: String, id: String)
        case dictationEnd(id: String)
        /// Text to show on the "not pasted" card: `raw` is the transcript, `candidate` the
        /// correction (the controller applies the same safety check and macros as a paste).
        case show(raw: String, candidate: String)
        /// Delete this WAV (history is off).
        case deleteWav(String)
    }

    struct Item {
        var id: String
        var stage: Stage
        var wavPath: String?
        /// Builds this dictation's correction request with its own app, field and screen
        /// context, captured when the next dictation started.
        var polishCommand: (String) -> [String: Any]
    }

    private(set) var items: [Item] = []

    var isEmpty: Bool { items.isEmpty }

    func contains(_ id: String?) -> Bool {
        guard let id else { return false }
        return items.contains { $0.id == id }
    }

    func polishCommand(raw: String, id: String) -> [String: Any]? {
        items.first { $0.id == id }?.polishCommand(raw)
    }

    /// The dictation `id` is set aside at `stage`. A transcript known but not yet sent for
    /// correction (`stage` polishing) returns its polish step now, without waiting on screen
    /// context; a WAV waiting for the engine goes out once it is ready.
    mutating func add(
        id: String, stage: Stage, wavPath: String?, polishSent: Bool, engineReady: Bool,
        polishCommand: @escaping (String) -> [String: Any]
    ) -> [Step] {
        items.append(Item(id: id, stage: stage, wavPath: wavPath, polishCommand: polishCommand))
        switch stage {
        case let .polishing(raw) where !polishSent:
            return [.polish(raw: raw, id: id)]
        case .awaitingTranscribe where engineReady:
            return sendTranscribe(id: id)
        default:
            return []
        }
    }

    /// The WAV write finished; nil when `id` is not set aside.
    mutating func wavWritten(id: String, path: String?, engineReady: Bool) -> [Step]? {
        guard let index = index(id) else { return nil }
        guard let path else {
            // Too short to transcribe: the engine still hears the dictation ended.
            items.remove(at: index)
            return [.dictationEnd(id: id)]
        }
        items[index].wavPath = path
        items[index].stage = .awaitingTranscribe
        return engineReady ? sendTranscribe(id: id) : []
    }

    /// The final transcript of `id`; nil when it is not set aside.
    mutating func transcript(id: String, text: String, keepAudio: Bool) -> [Step]? {
        guard let index = index(id) else { return nil }
        guard !text.isEmpty else { return finish(index, keepAudio: keepAudio) }
        items[index].stage = .polishing(raw: text)
        return [.polish(raw: text, id: id)]
    }

    /// The correction of `id`; nil when it is not set aside.
    mutating func corrected(id: String, text: String?, keepAudio: Bool) -> [Step]? {
        guard let index = index(id) else { return nil }
        guard case let .polishing(raw) = items[index].stage else { return [] }
        return [.show(raw: raw, candidate: text ?? raw)] + finish(index, keepAudio: keepAudio)
    }

    /// The engine reported an error for `id`: a known transcript is still shown.
    mutating func failed(id: String, keepAudio: Bool) -> [Step]? {
        guard let index = index(id) else { return nil }
        var steps: [Step] = []
        if case let .polishing(raw) = items[index].stage { steps.append(.show(raw: raw, candidate: raw)) }
        return steps + finish(index, keepAudio: keepAudio)
    }

    /// The engine (re)started: whatever was in flight there is gone, so send it again, in order.
    mutating func engineReady() -> [Step] {
        var steps: [Step] = []
        for item in items {
            switch item.stage {
            case .writingWav: break
            case .awaitingTranscribe, .transcribing:
                steps += sendTranscribe(id: item.id)
            case let .polishing(raw):
                steps.append(.polish(raw: raw, id: item.id))
            }
        }
        return steps
    }

    /// The card's text: an earlier dictation's text still on the card is kept ahead of the next.
    static func cardText(onCard: String?, adding text: String) -> String {
        guard let onCard, !onCard.isEmpty else { return text }
        return onCard + " " + text
    }

    private func index(_ id: String) -> Int? { items.firstIndex { $0.id == id } }

    private mutating func sendTranscribe(id: String) -> [Step] {
        guard let index = index(id), let path = items[index].wavPath else { return [] }
        items[index].stage = .transcribing
        return [.transcribe(path: path, id: id)]
    }

    private mutating func finish(_ index: Int, keepAudio: Bool) -> [Step] {
        let item = items.remove(at: index)
        guard !keepAudio, let path = item.wavPath else { return [] }
        return [.deleteWav(path)]
    }
}
