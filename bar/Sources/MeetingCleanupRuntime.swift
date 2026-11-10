import CryptoKit
import Darwin
import Foundation

// The meeting cleanup runtime: stock llama.cpp (the release Windows and Linux already run),
// its llama-server on the CPU at low priority with the meeting LoRA adapter on the shipped
// correction model. Separate from `pe`, the dictation engine, which cannot load adapters and
// keeps the GPU to itself. Started for a meeting and stopped when it is done; paused with
// SIGSTOP while a dictation runs. Listens on 127.0.0.1 only; its output goes nowhere.

/// What the cleaner needs of a runtime (a fake in tests).
@MainActor
protocol MeetingCleanupRuntime: AnyObject {
    func start() async throws
    func complete(system: String, user: String, maxTokens: Int) async throws -> String
    /// Stop (true) or continue (false) the process; a request in flight just waits.
    func setPaused(_ paused: Bool)
    func stop()
    /// For the cleaned file: what produced it.
    var label: String { get }
}

/// One pinned download (MeetingCleanupManifest).
struct PinnedAsset: Equatable {
    var name: String
    /// nil until the file is published; then only an installed copy can be used.
    var url: URL?
    var sha256: String
    var bytes: Int64
}

struct PinnedComponent: Equatable {
    enum Unpack: Equatable {
        case none
        /// A .tar.gz whose `stripTo` folder becomes the component folder.
        case tarGz(stripTo: String)
    }

    /// Folder under Application Support/Phonon.
    var dir: String
    var label: String
    var asset: PinnedAsset
    var unpack: Unpack
    /// The file that must exist for the component to count as installed.
    var sentinel: String
}

/// What meeting cleanup fetches on first use, pinned like the Windows and Linux manifest
/// (crates/phonon-win/src/manifest.rs): a release tag or a repository revision plus the exact
/// SHA-256. Fetched only when a meeting is cleaned, never at launch.
enum MeetingCleanupManifest {
    static let llamaBuild = "b12132"

    /// llama.cpp b12132 macos-arm64, the GitHub release asset (sha256 from the release's
    /// asset digest, checked against a download on 2026-11-10).
    static let runtime = PinnedComponent(
        dir: "runtime/llama-b12132-macos-arm64", label: "meeting cleanup runtime",
        asset: PinnedAsset(
            name: "llama-b12132-bin-macos-arm64.tar.gz",
            url: URL(
                string:
                    "https://github.com/ggml-org/llama.cpp/releases/download/b12132/llama-b12132-bin-macos-arm64.tar.gz"
            ),
            sha256: "0f7b107b860d3e9b7066ddc2b90e2f417641a4fd5f37355a603a1e52a98dc461",
            bytes: 16_671_755),
        unpack: .tarGz(stripTo: "llama-b12132"), sentinel: "llama-server")

    /// The meetings_v2 MB s1 adapter (GGUF LoRA, q8_0) for the shipped correction model.
    /// RELEASE: not published yet. Upload it to Infatoshi/phonon-refiner-1.7b as
    /// `phonon-meeting-lora-q8_0.gguf` and set `url` to its resolve URL at that commit.
    /// Until then it is used only when a verified copy is already installed here
    /// (Application Support/Phonon/models/meeting-cleanup/phonon-meeting-lora-q8_0.gguf).
    static let adapterPlaceholderURL =
        "https://huggingface.co/Infatoshi/phonon-refiner-1.7b/resolve/<revision>/phonon-meeting-lora-q8_0.gguf"
    static let adapter = PinnedComponent(
        dir: "models/meeting-cleanup", label: "meeting cleanup adapter",
        asset: PinnedAsset(
            name: "phonon-meeting-lora-q8_0.gguf", url: nil,
            sha256: "eb04d7f0cd25bc45e0e1b4c73447a72b5d388bf1cd82638ff11dab1df6c39596",
            bytes: 26_231_840),
        unpack: .none, sentinel: "phonon-meeting-lora-q8_0.gguf")

    /// The correction model the adapter was trained on (RLQv4U Q4_K_S-e4): the pinned
    /// refiner's GGUF (`phonon_llm::POLISH_MODEL_REVISION`).
    static let baseSHA256 = "0439996bda5886b92c86db49249be7d7ad2175ccf35bf2630c09433684e502e5"
    static let baseBytes: Int64 = 988_453_696
    static let refinerRepo = "Infatoshi/phonon-refiner-1.7b"
    static let refinerRevision = "b20b28075046af94056ab5d5bff89a7abc0f6f50"
    static let refinerGGUF = "phonon-refiner-1.7b-Q4_K_S-e4.gguf"

    /// Where the base model may be: an installed refiner folder (models/refiner, its
    /// phonon_engine.json target), then the pinned snapshot in the Hugging Face cache.
    static func baseCandidates(support: URL, environment: [String: String] = ProcessInfo.processInfo.environment)
        -> [URL]
    {
        var out: [URL] = []
        let refiner = support.appendingPathComponent("models/refiner", isDirectory: true)
        if let data = try? Data(contentsOf: refiner.appendingPathComponent("phonon_engine.json")),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let target = object["target"] as? String, !target.contains("/")
        {
            out.append(refiner.appendingPathComponent(target))
        }
        let hub: URL
        if let cache = environment["HF_HUB_CACHE"], !cache.isEmpty {
            hub = URL(fileURLWithPath: cache, isDirectory: true)
        } else if let home = environment["HF_HOME"], !home.isEmpty {
            hub = URL(fileURLWithPath: home, isDirectory: true).appendingPathComponent("hub")
        } else {
            hub = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub")
        }
        out.append(
            hub.appendingPathComponent("models--" + refinerRepo.replacingOccurrences(of: "/", with: "--"))
                .appendingPathComponent("snapshots/\(refinerRevision)/\(refinerGGUF)"))
        return out
    }
}

enum PinnedFiles {
    /// SHA-256 of a file, lowercase hex, read 4 MB at a time.
    static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 4 << 20), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// The first candidate whose size and hash match, hashing only files of the right size.
    /// A hash already checked this session (same path, size and modification date) is reused.
    static func firstMatching(_ candidates: [URL], sha256 expected: String, bytes: Int64) -> URL? {
        for url in candidates {
            let resolved = url.resolvingSymlinksInPath()
            guard
                let attributes = try? FileManager.default.attributesOfItem(atPath: resolved.path),
                (attributes[.size] as? NSNumber)?.int64Value == bytes
            else { continue }
            let key = "\(resolved.path)|\(bytes)|\((attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)"
            if verified.withLock({ $0.contains(key) }) { return url }
            if (try? sha256(resolved)) == expected {
                verified.withLock { _ = $0.insert(key) }
                return url
            }
        }
        return nil
    }

    private static let verified = NSLockedSet()
}

/// A set of strings behind a lock.
final class NSLockedSet: @unchecked Sendable {
    private let lock = NSLock()
    private var set: Set<String> = []

    func withLock<T>(_ body: (inout Set<String>) -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body(&set)
    }
}

/// Fetch, verify and install a pinned component, by the rules of the Windows and Linux
/// first-run download (crates/phonon-win/src/fetch.rs): bytes go to downloads/<name>.part, an
/// interrupted download resumes from the bytes on disk (HTTP Range), the file is moved into
/// place only once its SHA-256 matches, and a component is unpacked into a staging folder that
/// is renamed into place at the end, so a half-written one never looks installed.
final class PinnedDownloader: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    enum Failure: LocalizedError, Equatable {
        case unpublished(String)
        case http(Int)
        case verification(String)
        case unpack(String)

        var errorDescription: String? {
            switch self {
            case .unpublished(let label): return "The \(label) is not available for download yet."
            case .http(let status): return "The download failed (HTTP \(status))."
            case .verification(let name): return "\(name) failed verification."
            case .unpack(let name): return "\(name) could not be unpacked."
            }
        }
    }

    let root: URL
    private let configuration: URLSessionConfiguration

    init(root: URL, configuration: URLSessionConfiguration = .default) {
        self.root = root
        self.configuration = configuration
    }

    var downloads: URL { root.appendingPathComponent("downloads", isDirectory: true) }

    func directory(_ component: PinnedComponent) -> URL {
        root.appendingPathComponent(component.dir, isDirectory: true)
    }

    func installed(_ component: PinnedComponent) -> URL? {
        let sentinel = directory(component).appendingPathComponent(component.sentinel)
        guard FileManager.default.fileExists(atPath: sentinel.path) else { return nil }
        if component.unpack == .none {
            // a single file is checked every time: it may have been put there by hand
            return PinnedFiles.firstMatching(
                [sentinel], sha256: component.asset.sha256, bytes: component.asset.bytes)
        }
        return sentinel
    }

    /// The installed component's sentinel file, downloading it first when needed.
    func ensure(_ component: PinnedComponent) async throws -> URL {
        if let installed = installed(component) { return installed }
        guard let url = component.asset.url else { throw Failure.unpublished(component.label) }
        let file = try await fetch(component.asset, from: url)
        try install(component, from: file)
        guard let installed = installed(component) else { throw Failure.unpack(component.asset.name) }
        try? FileManager.default.removeItem(at: file)
        return installed
    }

    private func fetch(_ asset: PinnedAsset, from url: URL) async throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(at: downloads, withIntermediateDirectories: true)
        let target = downloads.appendingPathComponent(asset.name)
        if fm.fileExists(atPath: target.path) {
            if (try? PinnedFiles.sha256(target)) == asset.sha256 { return target }
            try? fm.removeItem(at: target)
        }
        let partial = downloads.appendingPathComponent(asset.name + ".part")
        var have = ((try? fm.attributesOfItem(atPath: partial.path))?[.size] as? NSNumber)?.int64Value ?? 0
        if have > asset.bytes {
            try? fm.removeItem(at: partial)
            have = 0
        }
        if have < asset.bytes {
            try await receive(url, into: partial, from: have)
        }
        guard (try? PinnedFiles.sha256(partial)) == asset.sha256 else {
            // a stale or corrupt partial is worth nothing; the next try starts over
            try? fm.removeItem(at: partial)
            throw Failure.verification(asset.name)
        }
        try fm.moveItem(at: partial, to: target)
        return target
    }

    private func install(_ component: PinnedComponent, from file: URL) throws {
        let fm = FileManager.default
        let final = directory(component)
        try fm.createDirectory(at: final.deletingLastPathComponent(), withIntermediateDirectories: true)
        let staging = final.deletingLastPathComponent().appendingPathComponent(".\(final.lastPathComponent).staging")
        try? fm.removeItem(at: staging)
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: staging) }
        switch component.unpack {
        case .none:
            try fm.moveItem(at: file, to: staging.appendingPathComponent(component.asset.name))
        case .tarGz(let stripTo):
            let raw = staging.appendingPathComponent(".raw", isDirectory: true)
            try fm.createDirectory(at: raw, withIntermediateDirectories: true)
            let tar = Process()
            tar.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
            tar.arguments = ["-xzf", file.path, "-C", raw.path]
            tar.standardOutput = FileHandle.nullDevice
            tar.standardError = FileHandle.nullDevice
            try tar.run()
            tar.waitUntilExit()
            let top = raw.appendingPathComponent(stripTo, isDirectory: true)
            guard tar.terminationStatus == 0, fm.fileExists(atPath: top.path) else {
                throw Failure.unpack(component.asset.name)
            }
            for name in try fm.contentsOfDirectory(atPath: top.path) {
                try fm.moveItem(at: top.appendingPathComponent(name), to: staging.appendingPathComponent(name))
            }
            try? fm.removeItem(at: raw)
        }
        guard fm.fileExists(atPath: staging.appendingPathComponent(component.sentinel).path) else {
            throw Failure.unpack(component.asset.name)
        }
        try? fm.removeItem(at: final)
        try fm.moveItem(at: staging, to: final)
    }

    // MARK: Resumable transfer

    private var output: FileHandle?
    private var continuation: CheckedContinuation<Void, Error>?
    private var offset: Int64 = 0

    private func receive(_ url: URL, into partial: URL, from offset: Int64) async throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: partial.path) { fm.createFile(atPath: partial.path, contents: nil) }
        var request = URLRequest(url: url)
        if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
        let handle = try FileHandle(forWritingTo: partial)
        self.offset = offset
        let session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            self.output = handle
            self.continuation = continuation
            session.dataTask(with: request).resume()
        }
    }

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        let status = (response as? HTTPURLResponse)?.statusCode ?? 200
        guard let output else { return completionHandler(.cancel) }
        if status == 206 {
            output.seekToEndOfFile()
        } else if status == 200 {
            // the server sent the whole file: start the partial over
            try? output.truncate(atOffset: 0)
        } else {
            completionHandler(.cancel)
            finish(Failure.http(status))
            return
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        output?.write(data)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        finish(error)
    }

    private func finish(_ error: Error?) {
        guard let continuation else { return }
        self.continuation = nil
        try? output?.synchronize()
        try? output?.close()
        output = nil
        if let error { continuation.resume(throwing: error) } else { continuation.resume() }
    }
}

/// Running time: wall time minus the time spent paused. `mark()` then `active(since:)` gives
/// the running seconds of one operation, however many pauses fell inside it.
struct ActiveClock {
    let now: () -> Date
    private var pausedTotal: TimeInterval = 0
    private var pausedSince: Date?

    init(now: @escaping () -> Date) { self.now = now }

    struct Mark {
        var at: Date
        var pausedTotal: TimeInterval
    }

    private func pausedSoFar(_ at: Date) -> TimeInterval {
        pausedTotal + (pausedSince.map { max(0, at.timeIntervalSince($0)) } ?? 0)
    }

    func mark() -> Mark {
        let at = now()
        return Mark(at: at, pausedTotal: pausedSoFar(at))
    }

    func active(since mark: Mark) -> TimeInterval {
        let at = now()
        return max(0, at.timeIntervalSince(mark.at) - (pausedSoFar(at) - mark.pausedTotal))
    }

    var isPaused: Bool { pausedSince != nil }

    mutating func pause() {
        if pausedSince == nil { pausedSince = now() }
    }

    mutating func resume() {
        guard let since = pausedSince else { return }
        pausedTotal += max(0, now().timeIntervalSince(since))
        pausedSince = nil
    }
}

/// llama-server as the runtime.
@MainActor
final class LlamaServerRuntime: MeetingCleanupRuntime {
    struct Setup: Equatable {
        var server: URL
        var model: URL
        var adapter: URL
        var adapterSHA256: String
        var threads: Int
        /// The process id is kept here so a crash's leftover is stopped at the next start.
        var pidFile: URL
    }

    enum Failure: LocalizedError {
        case startTimeout
        case requestTimeout
        case exited(Int32)
        case badReply

        var errorDescription: String? {
            switch self {
            case .startTimeout: return "The cleanup runtime did not start."
            case .requestTimeout: return "The cleanup runtime did not answer."
            case .exited(let status): return "The cleanup runtime stopped (status \(status))."
            case .badReply: return "The cleanup runtime gave an unreadable answer."
            }
        }
    }

    let setup: Setup
    /// Running (not paused) seconds the server may take to answer /health after launch, and to
    /// answer one chunk. Time spent stopped for a dictation never counts against either.
    let startBudget: TimeInterval
    let requestBudget: TimeInterval
    private var process: Process?
    private var port = 0
    /// Whether the process should be stopped; kept even before it launches.
    private var paused = false
    private var clock: ActiveClock
    /// Stand-ins for the HTTP calls (tests): /health on a port, and one chat request.
    private let probe: ((Int) async -> Bool)?
    private let send: (@Sendable (URLRequest) async throws -> Data)?

    init(
        setup: Setup, startBudget: TimeInterval = 120, requestBudget: TimeInterval = 600,
        now: @escaping () -> Date = Date.init, probe: ((Int) async -> Bool)? = nil,
        send: (@Sendable (URLRequest) async throws -> Data)? = nil
    ) {
        self.setup = setup
        self.startBudget = startBudget
        self.requestBudget = requestBudget
        self.clock = ActiveClock(now: now)
        self.probe = probe
        self.send = send
    }

    var label: String { "llama.cpp \(MeetingCleanupManifest.llamaBuild) CPU" }

    /// Threads for the CPU process: 4 (measured on an M4 Max), never more than half the cores.
    static func threads(cores: Int = ProcessInfo.processInfo.activeProcessorCount) -> Int {
        max(1, min(4, cores / 2))
    }

    /// The command line: CPU only (no Metal device, so it never shares the GPU with `pe`),
    /// weights mapped and not repacked (0.58 GB instead of 1.45 GB of its own memory, 12%
    /// slower), one slot, greedy as in training, loopback only, at nice 15.
    static func arguments(_ setup: Setup, port: Int) -> [String] {
        [
            "-n", "15", setup.server.path,
            "-m", setup.model.path, "--lora", setup.adapter.path,
            "--device", "none", "-ngl", "0", "--no-repack",
            "-t", "\(setup.threads)", "-tb", "\(setup.threads)",
            "-c", "4096", "-np", "1", "--repeat-penalty", "1.0",
            "--host", "127.0.0.1", "--port", "\(port)", "--no-webui",
        ]
    }

    func start() async throws {
        guard process == nil else { return }
        Self.stopLeftover(pidFile: setup.pidFile, server: setup.server)
        port = Self.freePort()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/nice")
        process.arguments = Self.arguments(setup, port: port)
        process.environment = ["PATH": "/usr/bin:/bin"]
        // nothing it prints is kept: its log can quote the meeting
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        // nice only, no background QoS: that confines it to the efficiency cores, 30x slower
        // on an M4 Max (3.3 against 99 tokens/s, `taskpolicy -b`, 2026-11-10)
        try process.run()
        self.process = process
        // a pause asked for before the launch holds from the first instruction
        if paused { kill(process.processIdentifier, SIGSTOP) }
        try? FileManager.default.createDirectory(
            at: setup.pidFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? "\(process.processIdentifier)".write(to: setup.pidFile, atomically: true, encoding: .utf8)
        // The model loads while running only: a dictation that stops the process mid-load
        // stops this clock too, so a long pause is a wait, never a start failure.
        let started = clock.mark()
        while clock.active(since: started) < startBudget {
            if Task.isCancelled {
                stop()
                throw CancellationError()
            }
            guard process.isRunning else {
                self.stop()
                throw Failure.exited(process.terminationStatus)
            }
            if !paused, await healthy(), !paused { return }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        stop()
        throw Failure.startTimeout
    }

    private func healthy() async -> Bool {
        if let probe { return await probe(port) }
        guard let url = URL(string: "http://127.0.0.1:\(port)/health") else { return false }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        guard let (_, response) = try? await URLSession.shared.data(for: request) else { return false }
        return (response as? HTTPURLResponse)?.statusCode == 200
    }

    func complete(system: String, user: String, maxTokens: Int) async throws -> String {
        guard let process, process.isRunning else { throw Failure.exited(self.process?.terminationStatus ?? -1) }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        // A dictation stops the process mid-answer and the request just waits: no wall-clock
        // timeout on it (a day, in effect); the budget below counts running time only.
        request.timeoutInterval = 86_400
        let body: [String: Any] = [
            "messages": [["role": "system", "content": system], ["role": "user", "content": user]],
            "temperature": 0, "max_tokens": maxTokens, "cache_prompt": false,
            "chat_template_kwargs": ["enable_thinking": false],
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let send = self.send ?? { request in try await URLSession.shared.data(for: request).0 }
        let started = clock.mark()
        let data = try await withThrowingTaskGroup(of: Data?.self) { group -> Data in
            group.addTask { try await send(request) }
            group.addTask { @MainActor [weak self] in
                // the watchdog: running time only
                while let self, self.clock.active(since: started) < self.requestBudget {
                    try await Task.sleep(nanoseconds: 100_000_000)
                }
                return nil
            }
            defer { group.cancelAll() }
            guard let first = try await group.next(), let data = first else { throw Failure.requestTimeout }
            return data
        }
        guard
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let choices = object["choices"] as? [[String: Any]],
            let message = choices.first?["message"] as? [String: Any],
            let content = message["content"] as? String
        else { throw Failure.badReply }
        return content
    }

    func setPaused(_ paused: Bool) {
        guard paused != self.paused else { return }
        self.paused = paused
        if paused { clock.pause() } else { clock.resume() }
        guard let process, process.isRunning else { return }
        kill(process.processIdentifier, paused ? SIGSTOP : SIGCONT)
    }

    func stop() {
        if let process, process.isRunning {
            if paused { kill(process.processIdentifier, SIGCONT) }
            process.terminate()
        }
        // `paused` stays as asked: a restart during a dictation launches stopped
        process = nil
        try? FileManager.default.removeItem(at: setup.pidFile)
    }

    /// A server left by a crash: stopped only when the recorded pid still runs our binary.
    static func stopLeftover(pidFile: URL, server: URL) {
        guard
            let text = try? String(contentsOf: pidFile, encoding: .utf8),
            let pid = pid_t(text.trimmingCharacters(in: .whitespacesAndNewlines)), pid > 0
        else { return }
        var buffer = [CChar](repeating: 0, count: Int(MAXPATHLEN) * 4)
        if proc_pidpath(pid, &buffer, UInt32(buffer.count)) > 0,
            URL(fileURLWithPath: String(cString: buffer)).resolvingSymlinksInPath().path
                == server.resolvingSymlinksInPath().path
        {
            kill(pid, SIGCONT)
            kill(pid, SIGTERM)
        }
        try? FileManager.default.removeItem(at: pidFile)
    }

    static func freePort() -> Int {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return 19_480 }
        defer { close(fd) }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, length) == 0 && getsockname(fd, $0, &length) == 0
            }
        }
        return bound ? Int(UInt16(bigEndian: address.sin_port)) : 19_480
    }
}
