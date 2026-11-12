import CryptoKit
import Foundation

/// One model file: where it goes under the store's root, where it comes from, and what it must hash to.
struct ModelFile: Sendable, Equatable {
    let path: String
    let url: URL
    let sha256: String
    let size: Int64
}

/// Downloads model files into `root` with resume, retry and sha256 checks: the iPhone twin of
/// android/app/src/main/java/ai/phonon/keyboard/ModelStore.kt, and the same policy as the Mac's
/// sidecar/model_pins.py (a pinned commit, every file's size and sha256, a file that fails the check is removed).
/// Plain Foundation + CryptoKit, so the unit tests run it against an in-process HTTP server.
///
/// A file is ready when it is at its full size and its marker `.verified/<path>` holds "<sha256> <size>", written
/// only after the whole file hashed correctly, so readiness never rehashes 460 MB on each start. A download goes to
/// `.partial/<path>` and resumes from that file's length with an HTTP Range request (Hugging Face answers 206 through
/// its CDN redirect; the Range header is carried across redirects by hand); a server that ignores Range (200)
/// restarts the file. The hash is checked once the part is complete: a mismatch deletes it and the next attempt
/// starts over. Markers and parts sit beside the model's folders, never inside a `.mlmodelc`.
final class ModelStore: @unchecked Sendable {
    let root: URL
    private let log: @Sendable (String) -> Void
    private let sleep: @Sendable (UInt64) async -> Void
    private let session: URLSessionConfiguration
    private let fm = FileManager.default

    static let margin: Int64 = 64 << 20

    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    private struct MismatchAgain: Error { let failure: Failure }

    init(root: URL,
         log: @escaping @Sendable (String) -> Void = { _ in },
         sleep: @escaping @Sendable (UInt64) async -> Void = { try? await Task.sleep(nanoseconds: $0) },
         session: URLSessionConfiguration = .default) {
        self.root = root
        self.log = log
        self.sleep = sleep
        self.session = session
    }

    func file(_ f: ModelFile) -> URL { root.appendingPathComponent(f.path) }
    private func part(_ f: ModelFile) -> URL { root.appendingPathComponent(".partial").appendingPathComponent(f.path) }
    private func marker(_ f: ModelFile) -> URL { root.appendingPathComponent(".verified").appendingPathComponent(f.path) }

    private func length(_ url: URL) -> Int64 {
        ((try? fm.attributesOfItem(atPath: url.path))?[.size] as? NSNumber)?.int64Value ?? -1
    }

    private func stamp(_ f: ModelFile) -> String { "\(f.sha256) \(f.size)" }

    func isReady(_ f: ModelFile) -> Bool {
        guard length(file(f)) == f.size, let m = try? String(contentsOf: marker(f), encoding: .utf8) else { return false }
        return m.trimmingCharacters(in: .whitespacesAndNewlines) == stamp(f)
    }

    func allReady(_ files: [ModelFile]) -> Bool { files.allSatisfy(isReady) }

    /// Bytes still to fetch for `files` (a resumable part counts as fetched).
    func remaining(_ files: [ModelFile]) -> Int64 {
        files.filter { !isReady($0) }.reduce(0) { $0 + $1.size - min(max(length(part($1)), 0), $1.size) }
    }

    /// Take over files an older build left in another folder (FluidAudio's own cache): each one that is at its
    /// pinned size and hashes to its pinned sha256 is moved into place and marked; any other is left for the
    /// download. Returns how many were adopted.
    @discardableResult
    func adopt(from legacy: URL, files: [ModelFile]) -> Int {
        var adopted = 0
        for f in files where !isReady(f) {
            let src = legacy.appendingPathComponent(f.path)
            guard length(src) == f.size, (try? Self.sha256(of: src)) == f.sha256 else { continue }
            do {
                try fm.createDirectory(at: file(f).deletingLastPathComponent(), withIntermediateDirectories: true)
                try? fm.removeItem(at: file(f))
                try fm.moveItem(at: src, to: file(f))
                try writeMarker(f)
                adopted += 1
            } catch {
                log("ADOPT failed file=\(f.path) error=\(error.localizedDescription)")
            }
        }
        if adopted > 0 { log("ADOPT files=\(adopted) from=\(legacy.lastPathComponent)") }
        return adopted
    }

    /// Fetch every file of `files` that is not ready. `progress` gets (bytes done, bytes total) over the whole list.
    /// Throws `Failure` when the disk lacks space or a file still fails after `attempts` tries.
    func ensure(_ files: [ModelFile], attempts: Int = 8,
                progress: @escaping @Sendable (Int64, Int64) -> Void = { _, _ in }) async throws {
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        let need = remaining(files)
        if need > 0, let free = freeSpace(), free < need + Self.margin {
            throw Failure(message: "not enough storage: need \(need >> 20) MB, \(free >> 20) MB free")
        }
        let total = files.reduce(0) { $0 + $1.size }
        var before = files.filter(isReady).reduce(0) { $0 + $1.size }
        for f in files where !isReady(f) {
            let base = before
            try await fetch(f, attempts: attempts) { have in progress(base + have, total) }
            before += f.size
            progress(before, total)
        }
    }

    private func freeSpace() -> Int64? {
        let values = try? root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    private func writeMarker(_ f: ModelFile) throws {
        let m = marker(f)
        try fm.createDirectory(at: m.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data((stamp(f) + "\n").utf8).write(to: m, options: .atomic)
    }

    private func fetch(_ f: ModelFile, attempts: Int, progress: @escaping @Sendable (Int64) -> Void) async throws {
        try? fm.removeItem(at: marker(f))
        var last: Error?
        var mismatches = 0
        for attempt in 1...max(attempts, 1) {
            do {
                try await download(f, progress: progress)
                let p = part(f)
                let got = try Self.sha256(of: p)
                if got != f.sha256 {
                    try? fm.removeItem(at: p)
                    mismatches += 1
                    let failure = Failure(message: "sha256 mismatch for \(f.path): got \(got)")
                    // as on the Mac: one fresh fetch after a bad file, failing again is an error
                    if mismatches >= 2 { throw MismatchAgain(failure: failure) }
                    throw failure
                }
                let dst = file(f)
                try fm.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? fm.removeItem(at: dst)
                try fm.moveItem(at: p, to: dst)
                try writeMarker(f)
                log("DOWNLOAD ok file=\(f.path) bytes=\(f.size) attempt=\(attempt)")
                return
            } catch is CancellationError {
                throw CancellationError()
            } catch let again as MismatchAgain {
                log("DOWNLOAD failed file=\(f.path) error=\(again.failure.message)")
                throw again.failure
            } catch {
                last = error
                log("DOWNLOAD retry file=\(f.path) attempt=\(attempt) have=\(max(length(part(f)), 0)) "
                    + "error=\(error.localizedDescription)")
                if attempt < attempts { await sleep(UInt64(min(30, 1 << attempt)) * 1_000_000_000) }
            }
        }
        throw Failure(message: "download failed for \(f.path): \(last?.localizedDescription ?? "unknown error")")
    }

    /// Bring the part to the full size, resuming from its current length.
    private func download(_ f: ModelFile, progress: @escaping @Sendable (Int64) -> Void) async throws {
        let p = part(f)
        try fm.createDirectory(at: p.deletingLastPathComponent(), withIntermediateDirectories: true)
        var have = max(length(p), 0)
        if have > f.size { try? fm.removeItem(at: p); have = 0 }
        if have == f.size { return }
        if !fm.fileExists(atPath: p.path) { fm.createFile(atPath: p.path, contents: nil) }
        var request = URLRequest(url: f.url, timeoutInterval: 60)
        request.setValue("phonon-ios", forHTTPHeaderField: "User-Agent")
        if have > 0 { request.setValue("bytes=\(have)-", forHTTPHeaderField: "Range") }
        let sink = PartWriter(part: p, offset: have, size: f.size, path: f.path, progress: progress)
        let s = URLSession(configuration: session, delegate: sink, delegateQueue: sink.queue)
        defer { s.invalidateAndCancel() }
        let got = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Int64, Error>) in
                sink.done = c
                s.dataTask(with: request).resume()
            }
        } onCancel: {
            s.invalidateAndCancel()
        }
        if got != f.size { throw Failure(message: "short read for \(f.path): \(got) of \(f.size)") }
    }

    static func sha256(of url: URL) throws -> String {
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        var digest = SHA256()
        while let chunk = try h.read(upToCount: 1 << 20), !chunk.isEmpty {
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// One request's delegate: checks the status and Content-Range, appends (206 from the part's length) or truncates
/// (200), writes each chunk to the part, and finishes with the part's final length.
private final class PartWriter: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    let queue: OperationQueue = {
        let q = OperationQueue()
        q.maxConcurrentOperationCount = 1
        return q
    }()
    var done: CheckedContinuation<Int64, Error>?
    private let part: URL
    private let size: Int64
    private let path: String
    private let progress: @Sendable (Int64) -> Void
    private var have: Int64
    private var handle: FileHandle?
    private var failure: Error?
    private var lastReport: Int64 = 0

    init(part: URL, offset: Int64, size: Int64, path: String, progress: @escaping @Sendable (Int64) -> Void) {
        self.part = part
        self.have = offset
        self.size = size
        self.path = path
        self.progress = progress
    }

    private func fail(_ message: String) -> URLSession.ResponseDisposition {
        failure = ModelStore.Failure(message: message)
        return .cancel
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        // carry the Range header to the CDN
        var next = request
        if let range = task.originalRequest?.value(forHTTPHeaderField: "Range") {
            next.setValue(range, forHTTPHeaderField: "Range")
        }
        completionHandler(next)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse else { return completionHandler(fail("not an HTTP response")) }
        let code = http.statusCode
        let append: Bool
        if code == 206 && have > 0 {
            let range = http.value(forHTTPHeaderField: "Content-Range") ?? ""
            let start = range.split(separator: " ").last?.split(separator: "-").first.flatMap { Int64($0) }
            guard start == have else {
                return completionHandler(fail("bad Content-Range \(range) for offset \(have)"))
            }
            append = true
        } else if code == 200 || code == 206 {
            have = 0
            append = false
        } else if code == 416 && have > 0 {
            try? FileManager.default.removeItem(at: part)
            return completionHandler(fail("HTTP 416 at offset \(have)"))
        } else {
            return completionHandler(fail("HTTP \(code) for \(path)"))
        }
        do {
            let h = try FileHandle(forWritingTo: part)
            if append { try h.seekToEnd() } else { try h.truncate(atOffset: 0) }
            handle = h
        } catch {
            failure = error
            return completionHandler(.cancel)
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard failure == nil, let handle else { return }
        if have + Int64(data.count) > size {
            failure = ModelStore.Failure(message: "more bytes than expected for \(path)")
            dataTask.cancel()
            return
        }
        do {
            try handle.write(contentsOf: data)
            have += Int64(data.count)
            if have - lastReport >= 4 << 20 {
                lastReport = have
                progress(have)
            }
        } catch {
            failure = error
            dataTask.cancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        try? handle?.close()
        handle = nil
        guard let c = done else { return }
        done = nil
        if let e = failure ?? error {
            c.resume(throwing: e)
        } else {
            progress(have)
            c.resume(returning: have)
        }
    }
}
