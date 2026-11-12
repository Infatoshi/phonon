import Foundation
import Network

/// A tiny HTTP/1.1 server on 127.0.0.1 for the downloader tests: serves `files` by path, honours `Range: bytes=N-`
/// with 206 (unless `ignoreRange`), answers `/redirect/<path>` with a 302 to `/<path>` (as Hugging Face sends a
/// resolve URL to its CDN), and can cut the next responses short to simulate a dropped connection.
final class TestServer: @unchecked Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "phonon.testserver")
    private let lock = NSLock()
    private var _files: [String: Data] = [:]
    private var _requests: [(path: String, range: String?)] = []
    private var _cutAfter: [Int] = []
    private var _ignoreRange = false
    private(set) var port: UInt16 = 0

    init() throws {
        let params = NWParameters.tcp
        params.requiredLocalEndpoint = NWEndpoint.hostPort(host: "127.0.0.1", port: 0)
        listener = try NWListener(using: params)
        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { state in
            if case .ready = state { ready.signal() }
            if case .failed = state { ready.signal() }
        }
        listener.newConnectionHandler = { [weak self] c in self?.serve(c) }
        listener.start(queue: queue)
        _ = ready.wait(timeout: .now() + 5)
        port = listener.port?.rawValue ?? 0
    }

    deinit { listener.cancel() }

    func url(_ path: String) -> URL { URL(string: "http://127.0.0.1:\(port)/\(path)")! }

    func set(_ path: String, _ data: Data) { lock.withLock { _files[path] = data } }
    var requests: [(path: String, range: String?)] { lock.withLock { _requests } }
    /// The next responses send only this many body bytes each, then close the connection.
    func cut(after bytes: [Int]) { lock.withLock { _cutAfter = bytes } }
    var ignoreRange: Bool {
        get { lock.withLock { _ignoreRange } }
        set { lock.withLock { _ignoreRange = newValue } }
    }

    private func serve(_ c: NWConnection) {
        c.start(queue: queue)
        read(c, Data())
    }

    private func read(_ c: NWConnection, _ buffer: Data) {
        c.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, done, error in
            guard let self else { return }
            var buf = buffer
            if let data { buf.append(data) }
            if let end = buf.range(of: Data("\r\n\r\n".utf8)) {
                self.respond(c, String(decoding: buf[..<end.lowerBound], as: UTF8.self))
            } else if done || error != nil {
                c.cancel()
            } else {
                self.read(c, buf)
            }
        }
    }

    private func respond(_ c: NWConnection, _ head: String) {
        let lines = head.components(separatedBy: "\r\n")
        let parts = lines.first?.split(separator: " ") ?? []
        let path = parts.count > 1 ? String(parts[1].dropFirst()) : ""
        var range: String?
        for line in lines.dropFirst() {
            let kv = line.split(separator: ":", maxSplits: 1)
            if kv.count == 2, kv[0].lowercased() == "range" { range = kv[1].trimmingCharacters(in: .whitespaces) }
        }
        let (body, cut, ignore): (Data?, Int?, Bool) = lock.withLock {
            _requests.append((path, range))
            let cut = _cutAfter.isEmpty || path.hasPrefix("redirect/") ? nil : _cutAfter.removeFirst()
            return (_files[path], cut, _ignoreRange)
        }
        if path.hasPrefix("redirect/") {
            let to = "/" + path.dropFirst("redirect/".count)
            send(c, "HTTP/1.1 302 Found\r\nLocation: \(to)\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", Data())
            return
        }
        guard let body else {
            send(c, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", Data())
            return
        }
        var start = 0
        if !ignore, let range, range.hasPrefix("bytes="), range.hasSuffix("-"),
           let from = Int(range.dropFirst(6).dropLast()) {
            start = from
        }
        if start >= body.count && start > 0 {
            send(c, "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", Data())
            return
        }
        let slice = body[start...]
        var header = start > 0
            ? "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes \(start)-\(body.count - 1)/\(body.count)\r\n"
            : "HTTP/1.1 200 OK\r\n"
        header += "Content-Length: \(slice.count)\r\nConnection: close\r\n\r\n"
        let sent = cut.map { Data(slice.prefix($0)) } ?? Data(slice)
        send(c, header, sent)
    }

    private func send(_ c: NWConnection, _ header: String, _ body: Data) {
        c.send(content: Data(header.utf8) + body, completion: .contentProcessed { _ in c.cancel() })
    }
}
