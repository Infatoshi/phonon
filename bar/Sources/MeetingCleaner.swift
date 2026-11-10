import Foundation

/// Cleans finished meeting transcripts one at a time in the background (MeetingCleanup.swift):
/// chunk, clean each chunk on the CPU runtime, check it, save progress after every chunk, and
/// write transcript.cleaned.json at the end. Dictation goes first: the runtime is paused
/// (SIGSTOP) whenever the engine says a dictation is under way (`{"type":"dictation"}`, its
/// meeting holds) or the engine is not up, and no chunk starts until it is quiet again. The
/// runtime process lives only while a meeting is being cleaned. Logs carry meeting ids, counts
/// and timings only, never what was said.
@MainActor
final class MeetingCleaner {
    struct Environment {
        /// Whether cleanup can run on this Mac at all: the adapter installed or downloadable.
        var available: () -> Bool
        var physicalMemory: () -> UInt64 = { ProcessInfo.processInfo.physicalMemory }
        var availableMemory: () -> UInt64? = { MeetingCleanupMemory.availableBytes() }
        /// Downloads what is missing (runtime, adapter), finds the base model, and returns a
        /// runtime that is not started yet, plus the adapter's sha256 for the cleaned file.
        var makeRuntime: () async throws -> (runtime: MeetingCleanupRuntime, adapterSHA256: String)
    }

    var onJobsChanged: (([String: MeetingCleanupJob]) -> Void)?
    /// A meeting's transcript.cleaned.json is written.
    var onFinished: ((String) -> Void)?
    /// Where a meeting's transcript files are (a check can point elsewhere).
    var outputDirectory: (URL) -> URL = { $0 }

    private(set) var jobs: [String: MeetingCleanupJob] = [:] {
        didSet { if jobs != oldValue { onJobsChanged?(jobs) } }
    }
    private let environment: Environment
    private var waiting: [URL] = []
    private var currentID: String?
    private var task: Task<Void, Never>?
    private var runtime: MeetingCleanupRuntime?
    private var dictationQuiet = false
    private var engineReady = false
    private var gateWaiters: [CheckedContinuation<Void, Never>] = []

    init(environment: Environment) {
        self.environment = environment
    }

    var isIdle: Bool { currentID == nil && waiting.isEmpty }
    var isAvailable: Bool { environment.available() }
    private var runnable: Bool { dictationQuiet && engineReady }

    /// Clean the meeting in `directory` (its transcript was just written, the Notes button,
    /// or a resume), from its progress file when there is one.
    func enqueue(directory: URL) {
        let id = directory.lastPathComponent
        guard currentID != id, !waiting.contains(where: { $0.lastPathComponent == id }), isAvailable else { return }
        let output = outputDirectory(directory)
        guard let transcript = MeetingTranscript.load(from: output), !transcript.turns.isEmpty else { return }
        if let cleaned = CleanedTranscript.load(from: output), cleaned.matches(transcript) { return }
        // a quit from here on resumes at the next launch
        var progress = MeetingCleanupProgress.load(from: output) ?? MeetingCleanupProgress()
        progress.error = nil
        try? progress.save(to: output)
        waiting.append(directory)
        jobs[id] = .queued
        startNext()
    }

    /// Meetings whose cleanup a quit or crash interrupted (a progress file without an error).
    func resumePending(in root: URL, fileManager: FileManager = .default) {
        let folders =
            (try? fileManager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
        for folder in folders.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let output = outputDirectory(folder)
            guard let progress = MeetingCleanupProgress.load(from: output), progress.error == nil else { continue }
            enqueue(directory: folder)
        }
    }

    /// The engine's `{"type":"dictation","quiet":..}`.
    func setDictationQuiet(_ quiet: Bool) {
        dictationQuiet = quiet
        applyGate()
    }

    /// The engine is up (`ready`) or gone; while it is gone nothing tells us about dictation,
    /// so cleanup holds. A new engine sends its dictation state at its first poll.
    func setEngineReady(_ ready: Bool) {
        engineReady = ready
        if !ready { dictationQuiet = false }
        applyGate()
    }

    /// The app is quitting: stop the runtime; the progress file resumes it at the next launch.
    func shutdown() {
        task?.cancel()
        task = nil
        runtime?.setPaused(false)
        runtime?.stop()
        runtime = nil
        waiting.removeAll()
        currentID = nil
        releaseWaiters()
    }

    private func applyGate() {
        runtime?.setPaused(!runnable)
        if let id = currentID, case .running(let fraction, _) = jobs[id] {
            jobs[id] = .running(fraction, paused: !runnable)
        }
        if runnable { releaseWaiters() }
    }

    private func releaseWaiters() {
        let waiters = gateWaiters
        gateWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }

    /// Returns once no dictation is under way (or the task is cancelled).
    private func gate() async {
        while !runnable && !Task.isCancelled {
            await withCheckedContinuation { gateWaiters.append($0) }
        }
    }

    private func startNext() {
        guard currentID == nil, !waiting.isEmpty else { return }
        let directory = waiting.removeFirst()
        currentID = directory.lastPathComponent
        task = Task { [weak self] in
            await self?.run(directory)
        }
    }

    private func done(_ id: String, job: MeetingCleanupJob?) {
        runtime?.stop()
        runtime = nil
        jobs[id] = job
        currentID = nil
        task = nil
        startNext()
    }

    private func fail(_ id: String, output: URL, progress: MeetingCleanupProgress, message: String) {
        var progress = progress
        progress.error = message
        try? progress.save(to: output)
        NSLog("phonon meeting cleanup: \(id) stopped after \(progress.results.count) chunks")
        done(id, job: .failed(message))
    }

    private func run(_ directory: URL) async {
        let id = directory.lastPathComponent
        let output = outputDirectory(directory)
        let started = Date()
        guard let transcript = MeetingTranscript.load(from: output) else {
            return done(id, job: .failed("This meeting's transcript could not be read."))
        }
        let source = CleanupSource(transcript)
        var progress = MeetingCleanupProgress.load(from: output) ?? MeetingCleanupProgress()
        if progress.source != source { progress = MeetingCleanupProgress(source: source) }
        progress.error = nil
        if let refusal = MeetingCleanupMemory.refusal(
            physical: environment.physicalMemory(), available: environment.availableMemory())
        {
            NSLog("phonon meeting cleanup: \(id) skipped (memory)")
            return fail(id, output: output, progress: progress, message: refusal)
        }
        let chunks = MeetingCleanupChunker.chunks(transcript.turns)
        let total = Double(max(1, chunks.count))
        jobs[id] = .running(Double(progress.results.count) / total, paused: !runnable)
        do {
            try progress.save(to: output)
        } catch {
            return done(id, job: .failed("Could not save cleanup progress: \(error.localizedDescription)"))
        }

        let made: (runtime: MeetingCleanupRuntime, adapterSHA256: String)
        do {
            made = try await environment.makeRuntime()
            if Task.isCancelled { return }
            runtime = made.runtime
            await gate()
            if Task.isCancelled { return }
            try await made.runtime.start()
            made.runtime.setPaused(!runnable)
        } catch {
            if Task.isCancelled { return }
            return fail(id, output: output, progress: progress, message: error.localizedDescription)
        }
        NSLog(
            "phonon meeting cleanup: \(id) started (\(chunks.count) chunks, \(progress.results.count) done) "
                + String(format: "after %.1f s", Date().timeIntervalSince(started)))

        let finished = Set(progress.results.map(\.chunk))
        for chunk in chunks where !finished.contains(chunk.index) {
            var outcome: CleanupOutcome?
            let chunkStarted = Date()
            for attempt in 0..<2 where outcome == nil {
                await gate()
                if Task.isCancelled { return }
                do {
                    let answer = try await made.runtime.complete(
                        system: MeetingCleanupPrompt.system, user: chunk.prompt, maxTokens: chunk.maxTokens)
                    outcome = MeetingCleanupValidator.check(chunk, output: answer)
                } catch {
                    if Task.isCancelled { return }
                    // once more on a fresh process; a chunk that breaks it twice keeps its raw lines
                    made.runtime.stop()
                    do {
                        try await made.runtime.start()
                        made.runtime.setPaused(!runnable)
                    } catch {
                        if Task.isCancelled { return }
                        return fail(id, output: output, progress: progress, message: error.localizedDescription)
                    }
                    if attempt == 1 { outcome = .rejected("runtime error") }
                }
            }
            let result = CleanupChunkResult(
                chunk: chunk.index, outcome: outcome ?? .rejected("runtime error"),
                seconds: (Date().timeIntervalSince(chunkStarted) * 10).rounded() / 10)
            progress.results.append(result)
            do {
                try progress.save(to: output)
            } catch {
                return fail(id, output: output, progress: progress, message: error.localizedDescription)
            }
            jobs[id] = .running(Double(progress.results.count) / total, paused: !runnable)
        }

        let cleaned = CleanedTranscript.assemble(
            transcript: transcript, chunks: chunks, results: progress.results, runtime: made.runtime.label,
            adapterSHA256: made.adapterSHA256)
        do {
            try cleaned.save(to: output)
            try? FileManager.default.removeItem(at: output.appendingPathComponent(MeetingCleanupProgress.fileName))
        } catch {
            return fail(id, output: output, progress: progress, message: error.localizedDescription)
        }
        let reasons = Dictionary(grouping: progress.results.compactMap(\.reason), by: { $0 }).mapValues(\.count)
        NSLog(
            "phonon meeting cleanup: \(id) done: \(chunks.count) chunks, \(cleaned.rawChunks) kept raw "
                + "\(reasons.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }), "
                + String(format: "%.1f s", Date().timeIntervalSince(started)))
        done(id, job: nil)
        onFinished?(id)
    }
}

extension MeetingCleaner {
    /// The app's cleaner: assets under Application Support/Phonon, the base model from the
    /// installed refiner or the pinned snapshot, llama-server as the runtime.
    static func live(support: URL) -> MeetingCleaner {
        let downloader = PinnedDownloader(root: support)
        let adapter = MeetingCleanupManifest.adapter
        return MeetingCleaner(
            environment: Environment(
                available: {
                    adapter.asset.url != nil
                        || FileManager.default.fileExists(
                            atPath: downloader.directory(adapter).appendingPathComponent(adapter.sentinel).path)
                },
                makeRuntime: {
                    let candidates = MeetingCleanupManifest.baseCandidates(support: support)
                    let base = await Task.detached(priority: .utility) {
                        PinnedFiles.firstMatching(
                            candidates, sha256: MeetingCleanupManifest.baseSHA256,
                            bytes: MeetingCleanupManifest.baseBytes)
                    }.value
                    guard let base else {
                        throw CleanupSetupError.baseModel
                    }
                    let adapterFile = try await downloader.ensure(adapter)
                    let server = try await downloader.ensure(MeetingCleanupManifest.runtime)
                    let setup = LlamaServerRuntime.Setup(
                        server: server, model: base, adapter: adapterFile,
                        adapterSHA256: adapter.asset.sha256, threads: LlamaServerRuntime.threads(),
                        pidFile: support.appendingPathComponent("runtime/meeting-cleanup.pid"))
                    return (LlamaServerRuntime(setup: setup), adapter.asset.sha256)
                }))
    }
}

enum CleanupSetupError: LocalizedError {
    case baseModel

    var errorDescription: String? {
        "The correction model the meeting adapter was made for is not installed."
    }
}
