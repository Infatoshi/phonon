import CryptoKit
import Foundation

// Meeting transcript cleanup (SPEC "Meeting notes", cleanup). After transcript.json is written,
// its raw speech-model turns are cut into chunks the way the meeting adapter was trained
// (meetings_v2: 120-220 words of whole speaker turns), each chunk is cleaned by the correction
// model plus the meeting LoRA adapter on a CPU llama.cpp process (MeetingCleanupRuntime.swift),
// and every answer is checked before it is used: the same speakers in the same order, no
// speaker line lost, the raw chunk's content words kept. A chunk that fails keeps its raw turns.
// The model's TL;DR line is always dropped (its attribution failed the bar). The result is
// transcript.cleaned.json next to transcript.json, which is never rewritten.

/// One speaker line of a chunk: a raw turn, or a part of a long one.
struct CleanupLine: Codable, Equatable {
    /// Index of the raw turn in transcript.json.
    var turn: Int
    /// "Speaker N", numbered by first appearance in the meeting, as in training.
    var label: String
    var text: String

    var wordCount: Int { text.split(whereSeparator: \.isWhitespace).count }
}

struct CleanupChunk: Equatable {
    var index: Int
    var lines: [CleanupLine]

    var wordCount: Int { lines.reduce(0) { $0 + $1.wordCount } }

    /// The user turn the adapter was trained on: one "Speaker N: words" line per line.
    var prompt: String { lines.map { "\($0.label): \($0.text)" }.joined(separator: "\n") }

    /// The training generator's output cap: room for the cleaned lines and the TL;DR.
    var maxTokens: Int { Int(2 * Double(wordCount) * 1.4) + 96 }
}

/// The meetings_v2 prompt (spec_v2.MEET_SYSTEM): the meeting system prompt plus the mode line.
enum MeetingCleanupPrompt {
    static let system =
        "Clean one chunk of a speaker-labelled meeting transcript for meeting notes. Keep every speaker line in "
        + "order with its label, remove fillers and false starts, fix misrecognized words and keep each speaker's "
        + "wording; then write one TL;DR line.\n"
        + "Writing mode: meeting (a speaker-labelled meeting transcript: keep every speaker line, "
        + "then one TL;DR line)."
}

/// How turns become chunks: the training chunker (meetings_v1 chunks.py) on the app's turns.
enum MeetingCleanupChunker {
    static let minWords = 120
    static let maxWords = 220
    /// A turn is split at the first sentence end at or after this many words...
    static let splitAtSentenceWords = 120
    /// ...or here, with no sentence end.
    static let splitHardWords = 160

    static func labels(_ turns: [MeetingTranscript.Turn]) -> [TranscriptSpeaker: String] {
        var names: [TranscriptSpeaker: String] = [:]
        for turn in turns where names[turn.speaker] == nil {
            names[turn.speaker] = "Speaker \(names.count + 1)"
        }
        return names
    }

    /// Every turn as one or more lines; long turns split as training split them.
    static func lines(_ turns: [MeetingTranscript.Turn]) -> [CleanupLine] {
        let names = labels(turns)
        var out: [CleanupLine] = []
        for (index, turn) in turns.enumerated() {
            let words = turn.text.split(whereSeparator: \.isWhitespace).map(String.init)
            guard !words.isEmpty else { continue }
            var current: [String] = []
            for word in words {
                current.append(word)
                if (current.count >= splitAtSentenceWords && endsSentence(word))
                    || current.count >= splitHardWords
                {
                    out.append(CleanupLine(turn: index, label: names[turn.speaker]!, text: current.joined(separator: " ")))
                    current = []
                }
            }
            if !current.isEmpty {
                out.append(CleanupLine(turn: index, label: names[turn.speaker]!, text: current.joined(separator: " ")))
            }
        }
        return out
    }

    /// Whole lines, 120-220 words, closed at a sentence end once 120 is reached. Training dropped
    /// a run that could not reach 120 words before 220; here it becomes a short chunk, so no
    /// line is left out, and so does the remainder at the end.
    static func chunks(_ turns: [MeetingTranscript.Turn]) -> [CleanupChunk] {
        var out: [CleanupChunk] = []
        var current: [CleanupLine] = []
        var words = 0
        func close() {
            if !current.isEmpty { out.append(CleanupChunk(index: out.count, lines: current)) }
            current = []
            words = 0
        }
        for line in lines(turns) {
            let count = line.wordCount
            if !current.isEmpty && words + count > maxWords { close() }
            current.append(line)
            words += count
            if words >= minWords && words <= maxWords && endsSentence(line.text) { close() }
        }
        close()
        return out
    }

    /// A rejected chunk's second pass (GOAL N13): the same lines as two chunks, cut at the line
    /// boundary nearest half its words. Nil for a one-line chunk.
    static func halves(_ chunk: CleanupChunk) -> (CleanupChunk, CleanupChunk)? {
        guard chunk.lines.count >= 2 else { return nil }
        let total = chunk.wordCount
        var best = 1
        var bestGap = Int.max
        var words = 0
        for cut in 1..<chunk.lines.count {
            words += chunk.lines[cut - 1].wordCount
            let gap = abs(2 * words - total)
            if gap < bestGap {
                bestGap = gap
                best = cut
            }
        }
        return (
            CleanupChunk(index: chunk.index, lines: Array(chunk.lines[..<best])),
            CleanupChunk(index: chunk.index, lines: Array(chunk.lines[best...]))
        )
    }

    static func endsSentence(_ text: String) -> Bool {
        guard let last = text.last else { return false }
        return last == "." || last == "?" || last == "!"
    }
}

/// What a chunk's answer becomes.
enum CleanupOutcome: Equatable {
    /// Per chunk line, its cleaned text; nil for a line that held only fillers and was dropped.
    case cleaned([String?])
    /// The answer failed a check; the chunk keeps its raw lines.
    case rejected(String)
    /// After the second pass one half passed and one did not: per chunk line its text, the raw
    /// text at the offsets in `raw`, and why that half was rejected.
    case mixed([String?], raw: [Int], reason: String)

    /// A rejected chunk's two halves, each checked by itself, as one outcome for the chunk: both
    /// cleaned is cleaned, both rejected keeps the first answer's reason, one of each is mixed.
    static func combine(
        _ first: CleanupOutcome, _ second: CleanupOutcome, halves: (CleanupChunk, CleanupChunk), original: String
    ) -> CleanupOutcome {
        switch (first, second) {
        case (.cleaned(let a), .cleaned(let b)):
            return .cleaned(a + b)
        case (.cleaned(let a), .rejected(let why)):
            let raw = halves.1.lines.map { Optional($0.text) }
            return .mixed(a + raw, raw: Array(a.count..<(a.count + raw.count)), reason: why)
        case (.rejected(let why), .cleaned(let b)):
            return .mixed(halves.0.lines.map { Optional($0.text) } + b, raw: Array(0..<halves.0.lines.count), reason: why)
        default:
            return .rejected(original)
        }
    }
}

/// The checks on an answer, before anything of it is used. Thresholds from the MB adapters'
/// outputs on the 190 public eval2 chunks (2026-11-10): among structurally valid answers the
/// lowest content recall was 0.867 / 0.844 (false starts removed), the lowest recall of a line
/// with 12+ content words 0.58, the most new words 5.7 per 100. With the line alignment below
/// (not greedy) 3/190 (s1) and 4/190 (s2) are rejected; tests/meeting_cleanup_check.py is the
/// Python reference and gives the same verdict on all 380 answers.
enum MeetingCleanupValidator {
    static let minChunkRecall = 0.85
    static let longLineWords = 12
    static let minLongLineRecall = 0.5
    static let maxNewWordsPer100 = 10.0
    static let maxLengthRatio = 2.2

    /// Words that a line may consist of and still be dropped (obj_meet.py FILL).
    static let fillers: Set<String> = [
        "um", "uh", "uhm", "hmm", "mm", "mhm", "mmhmm", "mm-hmm", "uh-huh", "huh", "ah", "eh", "er", "erm", "hm",
        "mmm", "oh", "yeah", "okay", "ok", "right", "yes", "so", "like", "well", "mm-mm", "yep", "sure", "alright",
    ]

    /// Lower case, hyphens split, everything but a-z 0-9 and apostrophes a space (obj_meet.py toks).
    static func tokens(_ text: String) -> [String] {
        var out = ""
        out.unicodeScalars.reserveCapacity(text.unicodeScalars.count)
        for scalar in text.lowercased().unicodeScalars {
            switch scalar {
            case "a"..."z", "0"..."9", "'", " ": out.unicodeScalars.append(scalar)
            default: out.unicodeScalars.append(" ")
            }
        }
        return out.split(separator: " ").map(String.init)
    }

    static func fillerOnly(_ text: String) -> Bool { tokens(text).allSatisfy { fillers.contains($0) } }

    static func isTLDR(_ line: String) -> Bool {
        line.range(of: #"^\s*\**\s*TL;?\s*DR"#, options: [.regularExpression, .caseInsensitive]) != nil
    }

    /// "Speaker N: words" (bold markers allowed), or nil.
    static func labelled(_ line: String) -> (label: String, text: String)? {
        guard
            let match = line.range(of: #"^\s*\**(Speaker \d+)\**\s*:\s*\**"#, options: .regularExpression)
        else { return nil }
        let head = String(line[match])
        guard let label = head.range(of: #"Speaker \d+"#, options: .regularExpression) else { return nil }
        return (String(head[label]), line[match.upperBound...].trimmingCharacters(in: .whitespaces))
    }

    /// The answer's speaker lines, TL;DR lines taken out; nil when a line is neither.
    static func parse(_ output: String) -> [(label: String, text: String)]? {
        var lines: [(label: String, text: String)] = []
        for raw in output.components(separatedBy: .newlines) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || isTLDR(line) { continue }
            guard let labelled = labelled(line) else { return nil }
            lines.append(labelled)
        }
        return lines
    }

    static func recall(of source: [String], in output: [String]) -> Double {
        guard !source.isEmpty else { return 1 }
        var have: [String: Int] = [:]
        for word in output { have[word, default: 0] += 1 }
        var kept = 0
        for word in source where (have[word] ?? 0) > 0 {
            have[word]! -= 1
            kept += 1
        }
        return Double(kept) / Double(source.count)
    }

    /// Words two lines share (multiset).
    static func overlap(_ a: String, _ b: String) -> Int {
        var have: [String: Int] = [:]
        for word in tokens(b) { have[word, default: 0] += 1 }
        var shared = 0
        for word in tokens(a) where (have[word] ?? 0) > 0 {
            have[word]! -= 1
            shared += 1
        }
        return shared
    }

    /// Same label, and a non-empty answer line of fillers only never stands for a line that held
    /// content (that line was dropped, not cleaned).
    static func canStandFor(_ answer: (label: String, text: String), _ input: CleanupLine) -> Bool {
        guard answer.label == input.label else { return false }
        return !(!answer.text.isEmpty && fillerOnly(answer.text) && !fillerOnly(input.text))
    }

    /// Answer line k -> input line, strictly increasing, each able to stand for its input line, every input line left out
    /// filler-only; of those, the one sharing the most words with the input. Nil when there is
    /// none. The greedy walk this replaces matched each answer line to the first input line of
    /// its label, so a kept line could land on an earlier filler line of the same speaker
    /// ("Speaker 2: Right." before "Speaker 2: <the real line>") and a correct answer was
    /// rejected: 3 of 6 (s1) and 4 of 8 (s2) eval2 fallbacks. Ties take the match over the skip
    /// from the end, as tests/meeting_cleanup_check.py does.
    static func align(_ input: [CleanupLine], _ answer: [(label: String, text: String)]) -> [Int]? {
        let n = input.count
        let m = answer.count
        let filler = input.map { fillerOnly($0.text) }
        var best = [[Int?]](repeating: [Int?](repeating: nil, count: n + 1), count: m + 1)
        best[0][0] = 0
        if n > 0 {
            for j in 1...n { best[0][j] = filler[j - 1] ? best[0][j - 1] : nil }
        }
        func score(_ i: Int, _ j: Int) -> Int? {
            guard canStandFor(answer[i - 1], input[j - 1]), let before = best[i - 1][j - 1] else { return nil }
            return before + overlap(answer[i - 1].text, input[j - 1].text)
        }
        if m > 0, n > 0 {
            for i in 1...m {
                for j in 1...n {
                    let skip = filler[j - 1] ? best[i][j - 1] : nil
                    let match = score(i, j)
                    best[i][j] = [skip, match].compactMap { $0 }.max()
                }
            }
        }
        guard let total = best[m][n] else { return nil }
        var at = [Int](repeating: 0, count: m)
        var i = m
        var j = n
        var remaining = total
        while i > 0 {
            if let match = score(i, j), match == remaining {
                at[i - 1] = j - 1
                remaining = best[i - 1][j - 1]!
                i -= 1
            } else {
                remaining = best[i][j - 1]!
            }
            j -= 1
        }
        return at
    }

    /// Why no alignment exists, named by the greedy walk the check used before.
    static func structuralReason(_ input: [CleanupLine], _ answer: [(label: String, text: String)]) -> String {
        var answered: Set<Int> = []
        var next = 0
        for (label, _) in answer {
            while next < input.count && input[next].label != label {
                if !fillerOnly(input[next].text) { return "speaker order" }
                next += 1
            }
            guard next < input.count else { return "extra speaker line" }
            answered.insert(next)
            next += 1
        }
        for (index, line) in input.enumerated() where !answered.contains(index) && !fillerOnly(line.text) {
            return "speaker line lost"
        }
        return "speaker order"
    }

    static func check(_ chunk: CleanupChunk, output: String) -> CleanupOutcome {
        guard let answer = parse(output) else { return .rejected("unlabelled line") }
        let input = chunk.lines
        guard let at = align(input, answer) else { return .rejected(structuralReason(input, answer)) }
        var cleaned = [String?](repeating: nil, count: input.count)
        // A label kept with nothing after it ("Speaker 4:" for "you know,") is a dropped filler
        // line; the recall checks below catch one that held real words.
        for (line, index) in zip(answer, at) where !line.text.isEmpty { cleaned[index] = line.text }
        let inputWords = input.flatMap { tokens($0.text) }
        let outputWords = cleaned.compactMap { $0 }.flatMap(tokens)
        if Double(outputWords.count) > maxLengthRatio * Double(max(1, inputWords.count)) {
            return .rejected("too long")
        }
        let content = inputWords.filter { !fillers.contains($0) }
        if recall(of: content, in: outputWords) < minChunkRecall { return .rejected("content lost") }
        for (index, line) in input.enumerated() {
            let words = tokens(line.text).filter { !fillers.contains($0) }
            guard words.count >= longLineWords else { continue }
            if recall(of: words, in: tokens(cleaned[index] ?? "")) < minLongLineRecall {
                return .rejected("line cut short")
            }
        }
        var inputCounts: [String: Int] = [:]
        for word in inputWords { inputCounts[word, default: 0] += 1 }
        var outputCounts: [String: Int] = [:]
        for word in outputWords { outputCounts[word, default: 0] += 1 }
        let added = outputCounts.reduce(0) { $0 + max(0, $1.value - (inputCounts[$1.key] ?? 0)) }
        if 100 * Double(added) / Double(max(1, outputWords.count)) > maxNewWordsPer100 {
            return .rejected("words added")
        }
        return .cleaned(cleaned)
    }
}

/// Which raw transcript a cleanup belongs to: turn count and a hash of speakers and texts.
struct CleanupSource: Codable, Equatable {
    var turns: Int
    var sha256: String

    init(_ transcript: MeetingTranscript) {
        turns = transcript.turns.count
        var hasher = SHA256()
        for turn in transcript.turns {
            hasher.update(data: Data("\(turn.speaker.rawValue)\t\(turn.text)\n".utf8))
        }
        sha256 = hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// One chunk's result as saved: its cleaned lines, or why it kept the raw ones.
struct CleanupChunkResult: Codable, Equatable {
    var chunk: Int
    /// Per chunk line (nil: dropped as filler only); absent when the chunk keeps its raw lines.
    var lines: [String?]?
    var reason: String?
    var seconds: Double?
    /// Lines (offsets in the chunk) kept raw while the rest is cleaned: a half the second pass
    /// could not clean.
    var raw: [Int]?
    /// Why the first answer was rejected, when the chunk went through the second pass.
    var retried: String?

    init(chunk: Int, outcome: CleanupOutcome, seconds: Double? = nil, retried: String? = nil) {
        self.chunk = chunk
        self.seconds = seconds
        self.retried = retried
        switch outcome {
        case .cleaned(let lines): self.lines = lines
        case .rejected(let reason): self.reason = reason
        case .mixed(let lines, let raw, let reason):
            self.lines = lines
            self.raw = raw
            self.reason = reason
        }
    }

    /// Whether any of the chunk's lines fell back to raw (the N13 count).
    var keptRaw: Bool { lines == nil || !(raw ?? []).isEmpty }
}

/// Meetings/<id>/transcript.cleaned.partial.json: the chunks cleaned so far, so a quit
/// resumes at the next chunk. `error` after a failure or a skip; cleared on a retry.
struct MeetingCleanupProgress: Codable, Equatable {
    static let fileName = "transcript.cleaned.partial.json"

    var schemaVersion = 1
    var source: CleanupSource?
    var results: [CleanupChunkResult] = []
    var error: String?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case source, results, error
    }

    static func load(from directory: URL) -> MeetingCleanupProgress? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(fileName)) else { return nil }
        return try? JSONDecoder().decode(MeetingCleanupProgress.self, from: data)
    }

    func save(to directory: URL) throws {
        try MeetingTranscriptFiles.write(self, to: directory.appendingPathComponent(Self.fileName))
    }
}

/// Meetings/<id>/transcript.cleaned.json: one entry per raw turn, by index.
struct CleanedTranscript: Codable, Equatable {
    static let fileName = "transcript.cleaned.json"
    static let currentSchema = 1

    var schemaVersion = CleanedTranscript.currentSchema
    var source: CleanupSource
    var runtime: String
    var adapterSHA256: String
    var createdAt: String
    var chunks: Int
    var rawChunks: Int
    var turns: [Turn]

    struct Turn: Codable, Equatable {
        /// nil when every part of the turn held only fillers and was dropped.
        var text: String?
        var source: Source
    }

    enum Source: String, Codable, Equatable {
        /// Every part of the turn cleaned.
        case cleaned
        /// Every part kept raw: its chunk failed a check.
        case raw
        /// A long turn split over chunks of which some failed.
        case mixed
        case dropped
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case source, runtime
        case adapterSHA256 = "adapter_sha256"
        case createdAt = "created_at"
        case chunks
        case rawChunks = "raw_chunks"
        case turns
    }

    static func load(from directory: URL) -> CleanedTranscript? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(fileName)) else { return nil }
        return try? JSONDecoder().decode(CleanedTranscript.self, from: data)
    }

    func save(to directory: URL) throws {
        try MeetingTranscriptFiles.write(self, to: directory.appendingPathComponent(Self.fileName))
    }

    /// Whether this cleanup was made from `transcript` as it is now.
    func matches(_ transcript: MeetingTranscript) -> Bool {
        source == CleanupSource(transcript) && turns.count == transcript.turns.count
    }

    /// The turns put back together from the chunk results: a turn's parts joined, raw text
    /// for every line of a chunk that failed (or never got a result).
    static func assemble(
        transcript: MeetingTranscript, chunks: [CleanupChunk], results: [CleanupChunkResult],
        runtime: String, adapterSHA256: String, createdAt: Date = Date()
    ) -> CleanedTranscript {
        var parts = [[(text: String?, cleaned: Bool)]](repeating: [], count: transcript.turns.count)
        let byChunk = Dictionary(results.map { ($0.chunk, $0) }, uniquingKeysWith: { _, last in last })
        var rawChunks = 0
        for chunk in chunks {
            let lines = byChunk[chunk.index]?.lines
            let raw = Set(byChunk[chunk.index]?.raw ?? [])
            if lines == nil || lines!.count != chunk.lines.count || !raw.isEmpty { rawChunks += 1 }
            for (offset, line) in chunk.lines.enumerated() {
                if let lines, lines.count == chunk.lines.count, !raw.contains(offset) {
                    parts[line.turn].append((lines[offset], true))
                } else {
                    parts[line.turn].append((line.text, false))
                }
            }
        }
        let turns = transcript.turns.enumerated().map { index, turn -> Turn in
            let mine = parts[index]
            if mine.isEmpty { return Turn(text: turn.text.isEmpty ? nil : turn.text, source: .raw) }
            let texts = mine.compactMap(\.text)
            if texts.isEmpty { return Turn(text: nil, source: .dropped) }
            let cleaned = mine.filter(\.cleaned).count
            let source: Source = cleaned == mine.count ? .cleaned : cleaned == 0 ? .raw : .mixed
            return Turn(text: texts.joined(separator: " "), source: source)
        }
        return CleanedTranscript(
            source: CleanupSource(transcript), runtime: runtime, adapterSHA256: adapterSHA256,
            createdAt: ISO8601DateFormatter().string(from: createdAt), chunks: chunks.count,
            rawChunks: rawChunks, turns: turns)
    }
}

/// Memory gate. The cleanup process maps the 988 MB correction model a second time (clean
/// file pages the system can drop) and holds about 0.6 GB of its own (measured: peak footprint
/// 0.58 GB, peak resident 1.6 GB with --no-repack on an M4 Max). Like Android's refiner gate:
/// never on a Mac under 12 GiB, and not when under 2.5 GB is available when a meeting starts.
enum MeetingCleanupMemory {
    static let minimumPhysical: UInt64 = 12 << 30
    static let minimumAvailable: UInt64 = 2_500_000_000

    /// Why cleanup must not run now, or nil.
    static func refusal(physical: UInt64, available: UInt64?) -> String? {
        if physical < minimumPhysical {
            return "Cleaning transcripts needs a Mac with 16 GB of memory."
        }
        if let available, available < minimumAvailable {
            return String(
                format: "Not cleaned: only %.1f GB of memory was free (%.1f GB needed).",
                Double(available) / 1e9, Double(minimumAvailable) / 1e9)
        }
        return nil
    }

    /// Free, inactive, purgeable and speculative pages: what the system can hand out now.
    static func availableBytes() -> UInt64? {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        let pages =
            UInt64(stats.free_count) + UInt64(stats.inactive_count) + UInt64(stats.purgeable_count)
            + UInt64(stats.speculative_count)
        return pages * UInt64(vm_kernel_page_size)
    }
}

/// A meeting's place in the cleaner.
enum MeetingCleanupJob: Equatable {
    case queued
    /// Fraction of chunks done; `paused` while a dictation runs.
    case running(Double, paused: Bool)
    case failed(String)
}

/// What the Notes page shows of a meeting's cleanup.
enum NoteCleanupState: Equatable {
    case cleaned(CleanedTranscript)
    case cleaning(progress: Double?, paused: Bool)
    /// Stopped or skipped (memory), with the reason; the button retries.
    case failed(String)
    /// Not cleaned and nothing running: the button offers it when cleanup is possible.
    case none

    static func resolve(
        transcript: MeetingTranscript?, cleaned: CleanedTranscript?, job: MeetingCleanupJob?,
        savedError: String?
    ) -> NoteCleanupState {
        guard let transcript, !transcript.turns.isEmpty else { return .none }
        if let cleaned, cleaned.matches(transcript) { return .cleaned(cleaned) }
        switch job {
        case .queued: return .cleaning(progress: nil, paused: false)
        case .running(let progress, let paused): return .cleaning(progress: progress, paused: paused)
        case .failed(let message): return .failed(message)
        case nil: return savedError.map { .failed($0) } ?? .none
        }
    }
}

/// The transcript lines Notes lists: cleaned text when there is a cleanup and raw is not asked
/// for (dropped filler-only turns left out), else the raw turns.
enum NoteTranscriptRows {
    struct Row: Equatable {
        var startSeconds: Double
        var speaker: TranscriptSpeaker
        var text: String
    }

    static func rows(_ transcript: MeetingTranscript, cleanup: NoteCleanupState, showRaw: Bool) -> [Row] {
        guard !showRaw, case .cleaned(let cleaned) = cleanup else {
            return transcript.turns.map { Row(startSeconds: $0.startSeconds, speaker: $0.speaker, text: $0.text) }
        }
        return zip(transcript.turns, cleaned.turns).compactMap { turn, clean in
            clean.text.map { Row(startSeconds: turn.startSeconds, speaker: turn.speaker, text: $0) }
        }
    }

    /// For Copy: one "m:ss You: text" paragraph per row.
    static func plainText(_ rows: [Row]) -> String {
        rows.map { "\(FeedTime.clock($0.startSeconds)) \($0.speaker.label): \($0.text)" }
            .joined(separator: "\n\n")
    }
}
