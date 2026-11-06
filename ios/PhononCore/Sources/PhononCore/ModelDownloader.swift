import CryptoKit
import Foundation

/// Downloads the pinned model files with HTTP Range resume and a streaming sha256 check.
/// A partial file survives a kill or a lost network (`<file>.part`); the next run hashes what
/// is already on disk and asks only for the rest. A file is moved into place only after its
/// size and sha256 match the manifest. This is the app's only network use.
public final class ModelDownloader: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    public struct Progress: Equatable, Sendable {
        public var bytesDone: Int64
        public var bytesTotal: Int64
        public var file: String
        public var fraction: Double { bytesTotal > 0 ? Double(bytesDone) / Double(bytesTotal) : 0 }
    }

    public enum Failure: LocalizedError, Equatable {
        case badStatus(Int, String)
        case hashMismatch(String)
        case sizeMismatch(String, Int64, Int64)
        case notEnoughSpace(needed: Int64, free: Int64)
        case cancelled
        case network(String)

        public var errorDescription: String? {
            switch self {
            case .badStatus(let code, let file): return "Server answered \(code) for \(file)."
            case .hashMismatch(let file): return "\(file) failed its integrity check and was deleted. Try again."
            case .sizeMismatch(let file, let got, let want): return "\(file) is \(got) bytes, expected \(want)."
            case .notEnoughSpace(let needed, let free):
                let f = ByteCountFormatter.string(fromByteCount: free, countStyle: .file)
                let n = ByteCountFormatter.string(fromByteCount: needed, countStyle: .file)
                return "Needs \(n) free, \(f) available."
            case .cancelled: return "Download paused."
            case .network(let message): return message
            }
        }
    }

    public let root: URL
    private let configuration: URLSessionConfiguration
    private var session: URLSession!
    private let queue = OperationQueue()
    private var active: Transfer?
    private var cancelled = false

    public init(root: URL, configuration: URLSessionConfiguration = .default) {
        self.root = root
        self.configuration = configuration
        super.init()
        queue.maxConcurrentOperationCount = 1
        configuration.timeoutIntervalForRequest = 60
        configuration.waitsForConnectivity = false
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var r = root
        try? r.setResourceValues(values)
    }

    // MARK: Status

    private var verifiedURL: URL { root.appendingPathComponent("verified.json") }

    private func loadVerified() -> [String: String] {
        (try? JSONDecoder().decode([String: String].self, from: Data(contentsOf: verifiedURL))) ?? [:]
    }

    private func markVerified(_ file: ModelFile) {
        var v = loadVerified()
        v[file.folder + "/" + file.path] = file.sha256
        if let data = try? JSONEncoder().encode(v) { try? data.write(to: verifiedURL, options: .atomic) }
    }

    /// True when the file is in place, the right size, and was hashed once after download.
    public func isReady(_ file: ModelFile) -> Bool {
        let url = file.localURL(root: root)
        guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize, Int64(size) == file.size else {
            return false
        }
        return loadVerified()[file.folder + "/" + file.path] == file.sha256
    }

    public func isReady(_ pack: ModelPack) -> Bool { pack.files.allSatisfy(isReady) }

    /// Bytes already on disk for a pack, counting partial files.
    public func bytesOnDisk(_ pack: ModelPack) -> Int64 {
        pack.files.reduce(0) { $0 + bytesOnDisk($1) }
    }

    func bytesOnDisk(_ file: ModelFile) -> Int64 {
        if isReady(file) { return file.size }
        let size = (try? partURL(file).resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return min(Int64(size), file.size)
    }

    public func delete(_ pack: ModelPack) {
        let folder = root.appendingPathComponent(pack.files.first?.folder ?? "", isDirectory: true)
        try? FileManager.default.removeItem(at: folder)
        var v = loadVerified()
        for f in pack.files { v.removeValue(forKey: f.folder + "/" + f.path) }
        if let data = try? JSONEncoder().encode(v) { try? data.write(to: verifiedURL, options: .atomic) }
    }

    private func partURL(_ file: ModelFile) -> URL {
        file.localURL(root: root).appendingPathExtension("part")
    }

    // MARK: Download

    public func cancel() {
        queue.addOperation {
            self.cancelled = true
            self.active?.task?.cancel()
        }
    }

    /// Seconds between retries, times the attempt number.
    public var retryDelay: Double = 1.5

    /// Downloads every missing file of `packs`, reporting progress over all of them.
    public func download(_ packs: [ModelPack], progress: @escaping @Sendable (Progress) -> Void) async throws {
        try await download(files: packs.flatMap(\.files), progress: progress)
    }

    public func download(files: [ModelFile], progress: @escaping @Sendable (Progress) -> Void) async throws {
        queue.addOperation { self.cancelled = false }
        let total = files.reduce(Int64(0)) { $0 + $1.size }
        let onDisk = files.reduce(Int64(0)) { $0 + bytesOnDisk($1) }
        if let free = freeSpace(), free < total - onDisk + 300_000_000 {
            throw Failure.notEnoughSpace(needed: total - onDisk + 300_000_000, free: free)
        }
        var doneBefore = files.filter(isReady).reduce(Int64(0)) { $0 + $1.size }
        progress(Progress(bytesDone: doneBefore, bytesTotal: total, file: ""))
        for file in files where !isReady(file) {
            var attempt = 0
            while true {
                do {
                    try await fetch(file) { fileBytes in
                        progress(Progress(bytesDone: doneBefore + fileBytes, bytesTotal: total, file: file.path))
                    }
                    break
                } catch let failure as Failure {
                    if failure == .cancelled { throw failure }
                    attempt += 1
                    if attempt >= 4 { throw failure }
                    try await Task.sleep(nanoseconds: UInt64(retryDelay * Double(attempt) * 1_000_000_000))
                }
            }
            doneBefore += file.size
            progress(Progress(bytesDone: doneBefore, bytesTotal: total, file: file.path))
        }
    }

    private func freeSpace() -> Int64? {
        #if os(iOS)
        let values = try? root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
        #else
        let values = try? root.resourceValues(forKeys: [.volumeAvailableCapacityKey])
        return values?.volumeAvailableCapacity.map(Int64.init)
        #endif
    }

    private final class Transfer {
        let file: ModelFile
        let handle: FileHandle
        var hasher = SHA256()
        var written: Int64
        let resumedFrom: Int64
        var task: URLSessionDataTask?
        var status = 0
        let progress: (Int64) -> Void
        var continuation: CheckedContinuation<Void, Error>?
        var lastReport = Date.distantPast

        init(file: ModelFile, handle: FileHandle, written: Int64, progress: @escaping (Int64) -> Void) {
            self.file = file
            self.handle = handle
            self.written = written
            self.resumedFrom = written
            self.progress = progress
        }
    }

    private func fetch(_ file: ModelFile, progress: @escaping (Int64) -> Void) async throws {
        let final = file.localURL(root: root)
        let part = partURL(file)
        let fm = FileManager.default
        try fm.createDirectory(at: final.deletingLastPathComponent(), withIntermediateDirectories: true)
        var existing = Int64((try? part.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        if existing > file.size {
            try? fm.removeItem(at: part)
            existing = 0
        }
        if !fm.fileExists(atPath: part.path) { fm.createFile(atPath: part.path, contents: nil) }
        let handle = try FileHandle(forUpdating: part)
        let transfer = Transfer(file: file, handle: handle, written: existing, progress: progress)
        // Hash the bytes a previous run already wrote.
        if existing > 0 {
            try handle.seek(toOffset: 0)
            var remaining = existing
            while remaining > 0 {
                let chunk = try handle.read(upToCount: Int(min(remaining, 4 << 20))) ?? Data()
                if chunk.isEmpty { break }
                transfer.hasher.update(data: chunk)
                remaining -= Int64(chunk.count)
            }
        }
        try handle.seek(toOffset: UInt64(existing))
        progress(existing)

        if existing < file.size {
            var request = URLRequest(url: file.url)
            if existing > 0 { request.setValue("bytes=\(existing)-", forHTTPHeaderField: "Range") }
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                queue.addOperation {
                    if self.cancelled {
                        cont.resume(throwing: Failure.cancelled)
                        return
                    }
                    transfer.continuation = cont
                    let task = self.session.dataTask(with: request)
                    transfer.task = task
                    self.active = transfer
                    task.resume()
                }
            }
        }
        try? handle.close()
        if transfer.written != file.size {
            throw Failure.sizeMismatch(file.path, transfer.written, file.size)
        }
        let digest = transfer.hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == file.sha256 else {
            try? fm.removeItem(at: part)
            throw Failure.hashMismatch(file.path)
        }
        try? fm.removeItem(at: final)
        try fm.moveItem(at: part, to: final)
        markVerified(file)
    }

    // MARK: URLSessionDataDelegate (runs on `queue`)

    public func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                           completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let t = active, let http = response as? HTTPURLResponse else {
            completionHandler(.cancel)
            return
        }
        t.status = http.statusCode
        if http.statusCode == 200 && t.written > 0 {
            // The server ignored the Range header: start the file over.
            try? t.handle.truncate(atOffset: 0)
            try? t.handle.seek(toOffset: 0)
            t.hasher = SHA256()
            t.written = 0
        } else if http.statusCode != 200 && http.statusCode != 206 {
            completionHandler(.cancel)
            return
        }
        completionHandler(.allow)
    }

    public func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard let t = active else { return }
        do {
            try t.handle.write(contentsOf: data)
        } catch {
            dataTask.cancel()
            return
        }
        t.hasher.update(data: data)
        t.written += Int64(data.count)
        let now = Date()
        if now.timeIntervalSince(t.lastReport) > 0.1 {
            t.lastReport = now
            t.progress(t.written)
        }
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let t = active, let cont = t.continuation else { return }
        t.continuation = nil
        active = nil
        try? t.handle.synchronize()
        if cancelled {
            cont.resume(throwing: Failure.cancelled)
        } else if t.status != 200 && t.status != 206 && t.status != 0 {
            cont.resume(throwing: Failure.badStatus(t.status, t.file.path))
        } else if let error {
            cont.resume(throwing: Failure.network(error.localizedDescription))
        } else {
            cont.resume()
        }
    }
}
