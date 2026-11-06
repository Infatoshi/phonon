import Foundation

/// Crash and kill recovery. Audio is appended to `<id>.pcm` (16 kHz mono Float32) while it is
/// recorded, and `journal.json` records how far the pipeline got. A dictation whose journal
/// entry is not `delivered` when the app starts is finished from what is on disk: the raw
/// transcript if ASR had finished, else the audio file.
public final class RecordingJournal: @unchecked Sendable {
    public enum Stage: String, Codable, Sendable { case recording, transcribed, delivered, discarded }

    public struct Entry: Codable, Equatable, Sendable {
        public var id: String
        public var startedAt: Date
        public var stage: Stage
        public var raw: String?
        public var mode: WritingMode?
    }

    public let directory: URL
    private var handle: FileHandle?
    private let lock = NSLock()

    public init(directory: URL) {
        self.directory = directory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    private var journalURL: URL { directory.appendingPathComponent("journal.json") }
    public func audioURL(_ id: String) -> URL { directory.appendingPathComponent("\(id).pcm") }

    public var entries: [Entry] {
        (try? JSONDecoder().decode([Entry].self, from: Data(contentsOf: journalURL))) ?? []
    }

    private func save(_ entries: [Entry]) {
        if let data = try? JSONEncoder().encode(Array(entries.suffix(20))) {
            try? data.write(to: journalURL, options: .atomic)
        }
    }

    private func update(_ id: String, _ change: (inout Entry) -> Void) {
        var all = entries
        if let i = all.firstIndex(where: { $0.id == id }) {
            change(&all[i])
            save(all)
        }
    }

    public func begin(id: String, mode: WritingMode?) {
        lock.lock(); defer { lock.unlock() }
        FileManager.default.createFile(atPath: audioURL(id).path, contents: nil)
        handle = try? FileHandle(forWritingTo: audioURL(id))
        var all = entries
        all.append(Entry(id: id, startedAt: Date(), stage: .recording, raw: nil, mode: mode))
        save(all)
    }

    public func append(_ samples: [Float]) {
        lock.lock(); defer { lock.unlock() }
        guard let handle, !samples.isEmpty else { return }
        samples.withUnsafeBufferPointer { buf in
            handle.write(Data(buffer: buf))
        }
    }

    public func endRecording() {
        lock.lock(); defer { lock.unlock() }
        try? handle?.synchronize()
        try? handle?.close()
        handle = nil
    }

    public func transcribed(_ id: String, raw: String) { update(id) { $0.stage = .transcribed; $0.raw = raw } }

    /// Delivered or discarded: the audio is no longer needed.
    public func finish(_ id: String, stage: Stage) {
        update(id) { $0.stage = stage }
        try? FileManager.default.removeItem(at: audioURL(id))
    }

    /// Entries a previous run left unfinished.
    public func unfinished() -> [Entry] {
        entries.filter { $0.stage == .recording || $0.stage == .transcribed }
    }

    public static func readSamples(_ url: URL) -> [Float] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        let count = data.count / MemoryLayout<Float>.size
        return data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self).prefix(count))
        }
    }
}
