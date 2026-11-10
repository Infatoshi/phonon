import Foundation

// Meeting transcripts (SPEC "Meeting notes"). After a recording stops, each track is fed to the
// one resident speech model in spans of about a minute, one at a time and only while no
// dictation is under way (the engine's meeting queue holds them back). The sentences of the two
// tracks are merged by time on the shared timeline into turns: "me" from the mic track, "call"
// from the system track (no diarization of the call side yet). Progress is kept in
// transcript.partial.json after every span, so a quit or crash resumes where it stopped; the
// finished transcript is transcript.json and the partial file goes away.

enum TranscriptSpeaker: String, Codable, Equatable {
    case me, call

    var label: String { self == .me ? "You" : "Call" }
}

/// Meetings/<id>/transcript.json.
struct MeetingTranscript: Codable, Equatable {
    static let fileName = "transcript.json"
    static let currentSchema = 1

    var schemaVersion = MeetingTranscript.currentSchema
    var model: String
    var createdAt: String
    var turns: [Turn]

    struct Turn: Codable, Equatable {
        var startSeconds: Double
        var endSeconds: Double
        var speaker: TranscriptSpeaker
        var text: String

        enum CodingKeys: String, CodingKey {
            case startSeconds = "start_seconds"
            case endSeconds = "end_seconds"
            case speaker, text
        }
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case model
        case createdAt = "created_at"
        case turns
    }

    var wordCount: Int {
        turns.reduce(0) { $0 + $1.text.split(whereSeparator: \.isWhitespace).count }
    }

    /// The whole transcript as text, for Copy: one paragraph per turn.
    var plainText: String {
        turns.map { "\(FeedTime.clock($0.startSeconds)) \($0.speaker.label): \($0.text)" }
            .joined(separator: "\n\n")
    }

    static func load(from directory: URL) -> MeetingTranscript? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(fileName)) else {
            return nil
        }
        return try? JSONDecoder().decode(MeetingTranscript.self, from: data)
    }

    func save(to directory: URL) throws {
        try MeetingTranscriptFiles.write(self, to: directory.appendingPathComponent(Self.fileName))
    }
}

enum MeetingTranscriptFiles {
    static func write<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(value).write(to: url, options: .atomic)
    }
}

/// A sentence from the speech model with its time in seconds (from the start of the span it
/// was heard in, or of the track, or of the shared timeline, as the caller says).
struct TimedSentence: Codable, Equatable {
    var start: Double
    var end: Double
    var text: String

    func shifted(by seconds: Double) -> TimedSentence {
        TimedSentence(start: start + seconds, end: end + seconds, text: text)
    }
}

/// Meetings/<id>/transcript.partial.json: where each track's transcription stands. Sentence
/// times are seconds from the track's sample 0.
struct MeetingTranscriptProgress: Codable, Equatable {
    static let fileName = "transcript.partial.json"

    var schemaVersion = 1
    var model: String?
    var mic = TrackProgress()
    var system = TrackProgress()
    /// The last failure, shown on Notes with the Transcribe button; cleared on a retry.
    var error: String?

    struct TrackProgress: Codable, Equatable {
        /// Track samples before this are transcribed.
        var cursor: Int64 = 0
        var done = false
        var sentences: [TimedSentence] = []
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case model, mic, system, error
    }

    subscript(track: MeetingTrack) -> TrackProgress {
        get { track == .mic ? mic : system }
        set {
            if track == .mic { mic = newValue } else { system = newValue }
        }
    }

    static func load(from directory: URL) -> MeetingTranscriptProgress? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(fileName)) else {
            return nil
        }
        return try? JSONDecoder().decode(MeetingTranscriptProgress.self, from: data)
    }

    func save(to directory: URL) throws {
        try MeetingTranscriptFiles.write(self, to: directory.appendingPathComponent(Self.fileName))
    }
}

/// One span of one track for the speech model: track samples [startSample, endSample) and the
/// chunks that hold them.
struct MeetingSpan: Equatable {
    let track: MeetingTrack
    let startSample: Int64
    let endSample: Int64
    let chunks: [MeetingChunk]
    /// The span reaches the end of the track's run of contiguous chunks: nothing after it can
    /// complete a sentence it cuts.
    let reachesRunEnd: Bool
}

/// How a track is cut into spans. A span is about `targetSeconds`, never over `maxSeconds`
/// (Metal Parakeet answers windows up to 82 s; a longer one would load the MLX copy of the
/// model). A short remainder joins the span before it. A span that is not the last of its run
/// gives back its last sentence when that sentence runs into the cut: the next span starts just
/// before it, so no sentence is ever split between two spans.
enum MeetingSpanPlanner {
    static let targetSeconds = 60.0
    static let maxSeconds = 75.0
    /// A last sentence ending this close to the cut may be cut off.
    static let cutTouchSeconds = 1.0
    /// Each span moves the cursor at least this far (a span that is all one sentence keeps it).
    static let minAdvanceSeconds = 5.0
    /// The next span starts this much before the sentence it re-transcribes.
    static let restartPadSeconds = 0.15

    /// The next span of `track` at or after `cursor`, or nil when the track is done.
    static func next(
        track: MeetingTrack, chunks: [MeetingChunk], cursor: Int64, rate: Int
    ) -> MeetingSpan? {
        let sorted = chunks.filter { $0.sampleCount > 0 }.sorted { $0.startSample < $1.startSample }
        guard
            let first = sorted.firstIndex(where: { $0.startSample + $0.sampleCount > cursor })
        else { return nil }
        let start = max(cursor, sorted[first].startSample)
        // The run of contiguous chunks from there (a gap ends it).
        var runEnd = sorted[first].startSample + sorted[first].sampleCount
        var last = first
        let rate = Double(max(1, rate))
        let maxSamples = Int64(maxSeconds * rate)
        while last + 1 < sorted.count, sorted[last + 1].startSample == runEnd,
            runEnd - start <= maxSamples
        {
            last += 1
            runEnd = sorted[last].startSample + sorted[last].sampleCount
        }
        let reachesRunEnd = runEnd - start <= maxSamples
        let end = reachesRunEnd ? runEnd : start + Int64((targetSeconds * rate).rounded())
        let covering = sorted[first...last].filter {
            $0.startSample < end && $0.startSample + $0.sampleCount > start
        }
        return MeetingSpan(
            track: track, startSample: start, endSample: end, chunks: Array(covering),
            reachesRunEnd: reachesRunEnd)
    }

    /// The sentences to keep from a span's answer (in track seconds) and the cursor after it.
    static func advance(
        _ span: MeetingSpan, sentences: [TimedSentence], rate: Int
    ) -> (kept: [TimedSentence], cursor: Int64) {
        let rate = Double(max(1, rate))
        let spanStart = Double(span.startSample) / rate
        let duration = Double(span.endSample - span.startSample) / rate
        let ordered = sentences.sorted { $0.start < $1.start }
        if !span.reachesRunEnd, let last = ordered.last,
            last.end >= duration - cutTouchSeconds, last.start >= minAdvanceSeconds
        {
            let restart = max(0, last.start - restartPadSeconds)
            return (
                ordered.dropLast().map { $0.shifted(by: spanStart) },
                span.startSample + Int64((restart * rate).rounded())
            )
        }
        return (ordered.map { $0.shifted(by: spanStart) }, span.endSample)
    }
}

enum TurnMerger {
    /// Both tracks' sentences (in seconds on the shared timeline) as turns in time order;
    /// consecutive sentences of the same speaker make one turn.
    static func turns(mic: [TimedSentence], call: [TimedSentence]) -> [MeetingTranscript.Turn] {
        let tagged =
            mic.map { ($0, TranscriptSpeaker.me) } + call.map { ($0, TranscriptSpeaker.call) }
        let ordered = tagged.enumerated().sorted { a, b in
            a.element.0.start != b.element.0.start
                ? a.element.0.start < b.element.0.start : a.offset < b.offset
        }
        var turns: [MeetingTranscript.Turn] = []
        for (_, (sentence, speaker)) in ordered {
            let text = sentence.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            if var turn = turns.last, turn.speaker == speaker {
                turn.text += " " + text
                turn.endSeconds = max(turn.endSeconds, sentence.end)
                turns[turns.count - 1] = turn
            } else {
                turns.append(
                    MeetingTranscript.Turn(
                        startSeconds: sentence.start, endSeconds: sentence.end, speaker: speaker,
                        text: text))
            }
        }
        return turns
    }

    /// The transcript of a finished progress file: each track's sentences moved onto the shared
    /// timeline by its start offset (the timeline the Notes player uses).
    static func transcript(
        _ progress: MeetingTranscriptProgress, metadata: MeetingMetadata, createdAt: Date = Date()
    ) -> MeetingTranscript {
        let offsets = MeetingNote.timelineOffsets(metadata)
        let rate = Double(max(1, metadata.sampleRate))
        let mic = progress.mic.sentences.map { $0.shifted(by: Double(offsets.mic) / rate) }
        let call = progress.system.sentences.map { $0.shifted(by: Double(offsets.system) / rate) }
        return MeetingTranscript(
            model: progress.model ?? "unknown",
            createdAt: ISO8601DateFormatter().string(from: createdAt),
            turns: turns(mic: mic, call: call))
    }
}

/// What the Notes page shows for a meeting's transcript.
enum NoteTranscriptState: Equatable {
    case ready(MeetingTranscript)
    /// Running, or waiting its turn (`progress` nil): for the speech model or another meeting.
    case transcribing(progress: Double?)
    /// Still being recorded: nothing to transcribe yet.
    case recording
    /// Never transcribed (a meeting from before transcripts, or one never started).
    case notTranscribed
    case failed(String)

    static func resolve(
        transcript: MeetingTranscript?, job: MeetingTranscriptionJob?, savedError: String?,
        recording: Bool
    ) -> NoteTranscriptState {
        if let transcript { return .ready(transcript) }
        if recording { return .recording }
        switch job {
        case .queued: return .transcribing(progress: nil)
        case .running(let progress): return .transcribing(progress: progress)
        case .failed(let message): return .failed(message)
        case nil: return savedError.map { .failed($0) } ?? .notTranscribed
        }
    }
}

/// A meeting's place in the transcriber.
enum MeetingTranscriptionJob: Equatable {
    case queued
    case running(Double)
    case failed(String)
}
