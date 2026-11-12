import AVFoundation
import Foundation
import Speech
import UIKit

/// Buffers from the input tap go to the current utterance's request, if any. Touched from the audio thread.
final class Feed: @unchecked Sendable {
    private let lock = NSLock()
    private var request: SFSpeechAudioBufferRecognitionRequest?

    func set(_ r: SFSpeechAudioBufferRecognitionRequest?) {
        lock.lock()
        request = r
        lock.unlock()
    }

    func push(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        let r = request
        lock.unlock()
        r?.append(buffer)
    }
}

/// The app side: one microphone session kept running (in the background too, UIBackgroundModes audio), and an
/// on-device recognition per utterance between the keyboard's start and stop signals.
@MainActor
final class Session: ObservableObject {
    @Published var running = false
    /// set when Phonon could not jump back to the app that opened it; the home screen then says to tap the breadcrumb
    @Published var backHint = false
    @Published var status = "Not started"
    @Published var last = ""
    @Published var lastMs: Int?
    @Published var count = 0

    private let engine = AVAudioEngine()
    private let feed = Feed()
    private let sink = SampleSink()
    private let parakeet = Parakeet()
    @Published var asrStatus = "Speech model: not loaded"
    private var useParakeet = false {
        didSet { parakeetReady = useParakeet }
    }
    @Published var parakeetReady = false
    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var heartbeat: Timer?
    private var utteranceStart = 0.0

    /// One finished dictation, newest first in `history` (kept on this iPhone only, last 50).
    struct Dictation: Codable, Identifiable, Equatable {
        var id = UUID()
        var at: Date
        var text: String
        var raw: String
        var mode: String
        var ms: Int?
    }
    @Published var history: [Dictation] = {
        guard let d = UserDefaults.standard.data(forKey: "history"),
            let h = try? JSONDecoder().decode([Dictation].self, from: d)
        else { return [] }
        return h
    }()
    @Published var sessionStarted: Date?

    // MARK: dictionary

    /// The personal dictionary (Application Support/dictionary.json, the desktop's schema). Empty: dictation is
    /// exactly as without one.
    @Published private(set) var dictionary = UserDictionary()

    static var dictionaryDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    }

    func addToDictionary(phrase: String, replacement: String?) {
        var d = dictionary
        if d.words == nil { d = d.withWords(WordList.bundled()) }
        saveDictionary(d.with(.init(phrase: phrase, replacement: replacement, source: "phone")))
    }

    func removeFromDictionary(_ entry: UserDictionary.Entry) { saveDictionary(dictionary.without(entry)) }

    private func saveDictionary(_ d: UserDictionary) {
        dictionary = d
        do { try UserDictionary.save(dir: Self.dictionaryDir, d) } catch { print("dictionary save failed", error) }
    }

    init() {
        IPC.observe(IPC.start) { [weak self] in self?.beginUtterance() }
        IPC.observe(IPC.stop) { [weak self] in self?.endUtterance() }
        IPC.observe(IPC.cancel) { [weak self] in self?.cancelUtterance() }
        IPC.observe(IPC.ping) { [weak self] in
            guard let self, self.running else { return }  // only a live mic session counts as alive
            self.beat()
            IPC.post(IPC.pong)
        }
        sink.onLevel = { level in
            DispatchQueue.main.async {
                IPC.set(level, IPC.levelValue)
                IPC.post(IPC.level)
            }
        }
        if let m = IPC.string(IPC.modeKey), WritingMode(rawValue: m) != nil { mode = m }
        dictionary = UserDictionary.load(dir: Self.dictionaryDir) { WordList.bundled() }
        // a model downloaded earlier: load it right away so the first dictation uses it
        if FileManager.default.fileExists(atPath: Self.parakeetMarker.path) { loadParakeet() }
    }

    private func cancelUtterance() {
        _ = sink.end()
        feed.set(nil)
        task?.cancel()
        task = nil
        request = nil
        setStatus("Ready")
    }

    func permissions() async -> String? {
        let mic = await AVAudioApplication.requestRecordPermission()
        if !mic { return "Microphone access is off: Settings > Phonon > Microphone." }
        // Apple's recognizer is only the fallback while Parakeet downloads; its permission is asked, never required
        if FileManager.default.fileExists(atPath: Self.parakeetMarker.path) { return nil }
        _ = await withCheckedContinuation { c in
            SFSpeechRecognizer.requestAuthorization { c.resume(returning: $0) }
        }
        return nil
    }

    func start() async {
        guard !running else { return }
        if let problem = await permissions() {
            status = problem
            return
        }
        do {
            let s = AVAudioSession.sharedInstance()
            try s.setCategory(.playAndRecord, mode: .default, options: [.mixWithOthers, .defaultToSpeaker])
            try s.setActive(true)
            let input = engine.inputNode
            let format = input.outputFormat(forBus: 0)
            let feed = self.feed
            let sink = self.sink
            input.removeTap(onBus: 0)
            input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
                feed.push(buffer)
                sink.push(buffer)
            }
            engine.prepare()
            try engine.start()
        } catch {
            status = "Microphone did not start: \(error.localizedDescription)"
            return
        }
        running = true
        sessionStarted = Date()
        loadParakeet()
        loadRefiner()
        setStatus("Ready")
        beat()
        heartbeat = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            Task { @MainActor in self.beat() }
        }
    }

    func stopSession() {
        heartbeat?.invalidate()
        heartbeat = nil
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        running = false
        sessionStarted = nil
        IPC.set(0, IPC.heartbeat)
        setStatus("Stopped")
    }

    private func beat() { IPC.set(IPC.now(), IPC.heartbeat) }

    private func setStatus(_ s: String) {
        status = s
        IPC.set(s, IPC.stateText)
        IPC.post(IPC.state)
    }

    /// One line per dictation in Documents/timings.jsonl, copied off the phone with devicectl.
    private func logTiming(_ row: [String: Any]) {
        guard let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
            let data = try? JSONSerialization.data(withJSONObject: row)
        else { return }
        let url = dir.appendingPathComponent("timings.jsonl")
        if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
        guard let h = try? FileHandle(forWritingTo: url) else { return }
        h.seekToEndOfFile()
        h.write(data + Data("\n".utf8))
        try? h.close()
    }

    static var parakeetMarker: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("parakeet.ready")
    }

    /// Parakeet replaces Apple's recognizer once it is loaded (first start downloads it, about 0.6 GB).
    private var warmAsrMs = -1

    func loadParakeet() {
        guard !useParakeet else { return }
        asrStatus = "Speech model: downloading Parakeet…"
        Task {
            do {
                let ms = try await parakeet.load { done, total in
                    Task { @MainActor in
                        guard !self.useParakeet, done < total else { return }
                        self.asrStatus = "Speech model: downloading Parakeet \(done >> 20) of \(total >> 20) MB…"
                    }
                }
                // warm-up, as on the Mac: one throwaway decode compiles the CoreML graphs and pages the weights in,
                // so the first real dictation is not the slow one
                let w0 = Date()
                _ = try? await parakeet.transcribe((0..<16000).map { Float(sin(Double($0) * 0.05) * 0.01) })
                warmAsrMs = Int(Date().timeIntervalSince(w0) * 1000)
                useParakeet = true
                FileManager.default.createFile(atPath: Self.parakeetMarker.path, contents: nil)
                asrStatus = "Speech model: Parakeet v2 on device (loaded in \(ms) ms, warm-up \(warmAsrMs) ms)"
            } catch {
                asrStatus = "Speech model: Apple on-device (Parakeet failed: \(error.localizedDescription))"
            }
        }
    }

    private func beginUtterance() {
        guard running else { return }
        utteranceStart = IPC.now()
        sink.begin()  // also drives the keyboard's level meter on the Apple-recognizer fallback
        if useParakeet {
            request = nil
            setStatus("Listening")
            return
        }
        guard let recognizer else { return }
        task?.cancel()
        let r = SFSpeechAudioBufferRecognitionRequest()
        r.requiresOnDeviceRecognition = true
        r.shouldReportPartialResults = false
        r.addsPunctuation = true
        r.taskHint = .dictation
        request = r
        utteranceStart = IPC.now()
        task = recognizer.recognitionTask(with: r) { [weak self] result, error in
            let text = result?.bestTranscription.formattedString
            let final = result?.isFinal ?? false
            let failed = error != nil
            Task { @MainActor in
                guard let self, self.request === r else { return }
                if final, let text {
                    self.refineAndDeliver(text)
                } else if failed {
                    self.deliver("", raw: "", refined: nil)
                }
            }
        }
        feed.set(r)
        setStatus("Listening")
    }

    private func endUtterance() {
        if useParakeet {
            let samples = sink.end()
            setStatus("Transcribing")
            Task {
                let text = (try? await parakeet.transcribe(samples)) ?? ""
                if text.isEmpty { deliver("", raw: "", refined: nil) } else { refineAndDeliver(text) }
            }
            return
        }
        _ = sink.end()
        guard let r = request else { return }
        feed.set(nil)
        r.endAudio()
        setStatus("Transcribing")
    }

    // MARK: refiner

    @Published var mode = UserDefaults.standard.string(forKey: "mode") ?? "minimal" {
        didSet {
            UserDefaults.standard.set(mode, forKey: "mode")
            if IPC.string(IPC.modeKey) != mode { IPC.set(mode, IPC.modeKey) }
        }
    }
    @Published var refinerStatus = "Refiner: not loaded"
    @Published var useRefiner = UserDefaults.standard.object(forKey: "use_refiner") as? Bool ?? true {
        didSet { UserDefaults.standard.set(useRefiner, forKey: "use_refiner") }
    }
    private var refiner: Refiner?
    private let refineQueue = DispatchQueue(label: "sh.phonon.ios.refine", qos: .userInitiated)
    private var asrDoneAt = 0.0

    static var modelURL: URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("refiner.gguf")
    }

    func loadRefiner() {
        guard refiner == nil, let url = Self.modelURL else { return }
        guard FileManager.default.fileExists(atPath: url.path) else {
            refinerStatus = "Refiner: no model at Documents/refiner.gguf"
            return
        }
        refinerStatus = "Refiner: loading…"
        let t0 = Date()
        refineQueue.async {
            let r = Refiner(path: url.path)
            let ms = Int(Date().timeIntervalSince(t0) * 1000)
            DispatchQueue.main.async {
                self.refiner = r
                self.refinerStatus = r == nil ? "Refiner: failed to load" : "Refiner: LFM3-1.7B on CPU (loaded in \(ms) ms)"
                if let r {  // warm it so the first dictation is not the slow one
                    self.refineQueue.async { _ = r.refine("warm up", mode: "minimal") }
                }
            }
        }
    }

    /// The self-test transcripts (mode, raw), shared with Tests/SelftestTests.swift (Firebase Test Lab).
    nonisolated static let selftestCases = [
        ("minimal", "um so the first thing we do is call could a malloc to to allocate memory on the GPU"),
        ("prompt", "okay so I want you to add a retry to the fetch function like three retries no actually make it five with exponential backoff and um also log every failure to the console"),
        ("casual", "yeah I think we should uh grab dinner around seven if that works for you"),
        ("formal", "hi sarah just wanted to let you know the the deploy is gonna slip to thursday because the tests are failing on the arm runners"),
    ]

    /// Launched with --selftest: load the refiner, time it on fixed transcripts (each mode), write
    /// Documents/selftest.jsonl. No microphone, no recognizer.
    func selftest() {
        guard let url = Self.modelURL else { return }
        refinerStatus = "Self-test running"
        let cases = Self.selftestCases
        refineQueue.async {
            var rows: [[String: Any]] = []
            let t0 = Date()
            let args = ProcessInfo.processInfo.arguments
            func arg(_ name: String) -> String? {
                args.firstIndex(of: name).flatMap { i in i + 1 < args.count ? args[i + 1] : nil }
            }
            let threads = arg("--threads").flatMap { Int32($0) } ?? Refiner.defaultThreads
            // A/B flags: --pool 0 = ggml's per-token threads, --stock 1 = stock worker QoS and repack chunking
            let usePool = arg("--pool") != "0"
            let stock = arg("--stock") == "1"
            Refiner.setStock(stock)
            let before = Self.threadPriorities()
            // shortlist-head experiment: --model NAME loads Documents/NAME (the product always loads refiner.gguf)
            let path = arg("--model").map { url.deletingLastPathComponent().appendingPathComponent($0).path } ?? url.path
            guard let r = self.refiner ?? Refiner(path: path, threads: threads, persistentPool: usePool) else {
                DispatchQueue.main.async { self.refinerStatus = "Self-test: model failed to load" }
                return
            }
            let stamp = arg("--stamp") ?? ""
            rows.append(["load_ms": Int(Date().timeIntervalSince(t0) * 1000), "threads": r.threads, "stamp": stamp,
                         "model": (path as NSString).lastPathComponent, "shortlist": r.shortlist,
                         "pool": r.persistentPool, "stock": stock, "caller_qos": Int(qos_class_self().rawValue),
                         "thermal": ProcessInfo.processInfo.thermalState.rawValue, "threads_by_pri_before": before])
            // each mode twice in a row: rep 0 prefills the fixed head (a mode switch), rep 1 reuses it
            for (mode, raw) in cases {
                for rep in 0..<2 {
                    let t = Date()
                    let o = r.refine(raw, mode: mode)
                    rows.append(["rep": rep, "mode": mode, "total_ms": Int(Date().timeIntervalSince(t) * 1000),
                                 "prefill_ms": o.prefillMs, "decode_ms": o.decodeMs, "prompt_tok": o.promptTokens,
                                 "gen_tok": o.genTokens, "guard": o.guardFired, "out": o.text,
                                 "tok_hash": String(o.tokenHash, radix: 16), "fallback": o.fallbacks])
                }
            }
            // live threads by base priority after the run (a persistent pool's workers are still alive, paused)
            rows[0]["threads_by_pri_after"] = Self.threadPriorities()
            rows[0]["thermal_after"] = ProcessInfo.processInfo.thermalState.rawValue
            if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                let lines = rows.compactMap { try? JSONSerialization.data(withJSONObject: $0) }
                    .map { String(decoding: $0, as: UTF8.self) }.joined(separator: "\n") + "\n"
                try? lines.write(to: dir.appendingPathComponent("selftest.jsonl"), atomically: true, encoding: .utf8)
            }
            DispatchQueue.main.async {
                self.refiner = self.refiner ?? r
                self.refinerStatus = "Self-test done"
            }
        }
    }

    /// Live threads of this process counted by base scheduling priority (31 = QoS Default, 37 = User Initiated,
    /// 47 = User Interactive), so the selftest can check what priority the refiner's workers really got.
    nonisolated static func threadPriorities() -> [String: Int] {
        var list: thread_act_array_t?
        var count: mach_msg_type_number_t = 0
        guard task_threads(mach_task_self_, &list, &count) == KERN_SUCCESS, let list else { return [:] }
        var hist: [String: Int] = [:]
        for i in 0..<Int(count) {
            var info = thread_extended_info()
            var n = mach_msg_type_number_t(MemoryLayout<thread_extended_info>.size / MemoryLayout<natural_t>.size)
            let kr = withUnsafeMutablePointer(to: &info) { p in
                p.withMemoryRebound(to: integer_t.self, capacity: Int(n)) {
                    thread_info(list[i], thread_flavor_t(THREAD_EXTENDED_INFO), $0, &n)
                }
            }
            if kr == KERN_SUCCESS { hist[String(info.pth_priority), default: 0] += 1 }
            mach_port_deallocate(mach_task_self_, list[i])
        }
        vm_deallocate(mach_task_self_, vm_address_t(bitPattern: list),
                      vm_size_t(Int(count) * MemoryLayout<thread_t>.size))
        return hist
    }

    private func refineAndDeliver(_ raw: String) {
        asrDoneAt = IPC.now()
        // the dictionary, as on the Mac and Android: exact replacements and phonetic pre-corrections before the
        // refiner, its vocabulary line in the refiner's prompt; an empty dictionary leaves raw and the prompt as they are
        let dict = dictionary
        let prepared = raw.isEmpty ? UserDictionary.Prepared(text: raw, vocab: []) : dict.prepare(raw)
        guard useRefiner, let r = refiner, !raw.isEmpty else {
            deliver(dict.isEmpty ? raw : dict.applyExact(prepared.text), raw: raw, refined: nil)
            return
        }
        setStatus("Refining")
        // the keyboard's mode chip wins: it is what the user picked for this text field
        if let m = IPC.string(IPC.modeKey), WritingMode(rawValue: m) != nil, m != self.mode { self.mode = m }
        let mode = self.mode
        refineQueue.async {
            let out = r.refine(prepared.text, mode: mode, vocab: prepared.vocab, spoken: raw,
                               dictionary: dict.isEmpty ? nil : dict)
            DispatchQueue.main.async { self.deliver(out.text, raw: raw, refined: out) }
        }
    }

    private func deliver(_ text: String, raw: String? = nil, refined: Refiner.Output? = nil) {
        request = nil
        task = nil
        let stopAt = IPC.double(IPC.stopAt)
        let ms = stopAt > 0 ? Int((IPC.now() - stopAt) * 1000) : nil
        IPC.set(text, IPC.resultText)
        IPC.set(IPC.double(IPC.resultSeq) + 1, IPC.resultSeq)
        IPC.post(IPC.result)
        var row: [String: Any] = [
            "at": IPC.now(), "ms": ms ?? -1, "words": text.split(separator: " ").count,
            "held_s": stopAt > 0 ? stopAt - utteranceStart : -1,
            "background": UIApplication.shared.applicationState != .active,
            "asr_ms": stopAt > 0 && asrDoneAt > 0 ? Int((asrDoneAt - stopAt) * 1000) : -1,
            "raw": raw ?? text, "text": text, "mode": mode,
        ]
        if let refined {
            row["prompt_tok"] = refined.promptTokens
            row["gen_tok"] = refined.genTokens
            row["prefill_ms"] = refined.prefillMs
            row["decode_ms"] = refined.decodeMs
            row["guard"] = refined.guardFired
        }
        logTiming(row)
        asrDoneAt = 0
        last = text
        lastMs = ms
        count += 1
        if !text.isEmpty {
            history.insert(Dictation(at: Date(), text: text, raw: raw ?? text, mode: mode, ms: ms), at: 0)
            if history.count > 50 { history.removeLast(history.count - 50) }
            if let d = try? JSONEncoder().encode(history) { UserDefaults.standard.set(d, forKey: "history") }
        }
        setStatus("Ready")
    }
}
