import Foundation
import llama
import PhononCore

/// The Phonon refiner on llama.cpp: the trained prompt, greedy decoding, the token budget and
/// the word-count guard of `sidecar/polish_server.py`.
///
/// It runs on the CPU. Keyboard dictation always finishes while Phonon is in the background
/// (the user is back in Messages), and iOS refuses GPU work from background apps: Metal
/// command buffers fail, ggml-metal flags the error and llama_decode fails, so a GPU model
/// would fail every keyboard dictation.
final class LlamaRefiner: @unchecked Sendable {
    struct Stats: Codable, Sendable {
        var promptTokens = 0
        var reusedTokens = 0
        var outputTokens = 0
        var budget = 0
        var prefillSeconds = 0.0
        var decodeSeconds = 0.0
        var prefillTokPerSec: Double { prefillSeconds > 0 ? Double(promptTokens - reusedTokens) / prefillSeconds : 0 }
        var decodeTokPerSec: Double { decodeSeconds > 0 ? Double(outputTokens) / decodeSeconds : 0 }
    }

    struct Output: Sendable {
        var text: String
        var raw: String
        var guardFired: Bool
        var stats: Stats
    }

    enum Failure: LocalizedError {
        case load(String), tokenize, decode(Int32), contextFull, unsupportedChip
        var errorDescription: String? {
            switch self {
            case .load(let path): return "Could not load the cleanup model at \(path)."
            case .tokenize: return "Could not tokenize the transcript."
            case .decode(let code): return "The cleanup model stopped (code \(code))."
            case .contextFull: return "The dictation is longer than the cleanup model's window."
            case .unsupportedChip: return "Cleanup needs an A15 chip or newer (iPhone 13 or later). Dictation inserts the raw transcript on this phone."
            }
        }
    }

    private let model: OpaquePointer
    private let ctx: OpaquePointer
    private let vocab: OpaquePointer
    private let sampler: UnsafeMutablePointer<llama_sampler>
    private let nCtx: Int32
    private let nBatch: Int32
    private var cached: [llama_token] = []
    private let lock = NSLock()
    let loadSeconds: Double
    let threads: Int32

    private static let backendOnce: Void = {
        llama_backend_init()
        llama_log_set({ _, _, _ in }, nil)
    }()

    /// The engine is compiled for A15 and newer (dot product, fp16, int8 matmul); an older
    /// chip would fault on the first matrix multiply, so refuse cleanly instead.
    static var chipSupported: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return sysctlInt("hw.optional.arm.FEAT_I8MM") == 1
        #endif
    }

    static func sysctlInt(_ name: String) -> Int {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        return sysctlbyname(name, &value, &size, nil, 0) == 0 ? Int(value) : 0
    }

    init(modelURL: URL) throws {
        guard Self.chipSupported else { throw Failure.unsupportedChip }
        _ = Self.backendOnce
        let started = Date()
        var mparams = llama_model_default_params()
        mparams.n_gpu_layers = 0
        mparams.load_mode = LLAMA_LOAD_MODE_MMAP
        guard let model = llama_model_load_from_file(modelURL.path, mparams) else {
            throw Failure.load(modelURL.lastPathComponent)
        }
        // Performance cores only: efficiency cores straggle and stall every token (FACT on an
        // M-series CPU, llama-bench: decode 129 tok/s on 8 P-cores, 42 tok/s on all 12).
        let pcores = Int32(Self.sysctlInt("hw.perflevel0.physicalcpu"))
        threads = pcores > 0 ? pcores : 2
        var cparams = llama_context_default_params()
        cparams.n_ctx = 4096
        cparams.n_batch = 512
        cparams.n_ubatch = 512
        cparams.n_threads = threads
        cparams.n_threads_batch = threads
        cparams.no_perf = true
        guard let ctx = llama_init_from_model(model, cparams) else {
            llama_model_free(model)
            throw Failure.load(modelURL.lastPathComponent)
        }
        self.model = model
        self.ctx = ctx
        self.vocab = llama_model_get_vocab(model)
        self.nCtx = Int32(cparams.n_ctx)
        self.nBatch = Int32(cparams.n_batch)
        let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())!
        llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        self.sampler = chain
        loadSeconds = Date().timeIntervalSince(started)
    }

    deinit {
        llama_sampler_free(sampler)
        llama_free(ctx)
        llama_model_free(model)
    }

    func tokenize(_ text: String, parseSpecial: Bool) -> [llama_token] {
        let utf8 = Array(text.utf8CString)
        var capacity = Int32(utf8.count + 16)
        var tokens = [llama_token](repeating: 0, count: Int(capacity))
        var n = llama_tokenize(vocab, text, Int32(utf8.count - 1), &tokens, capacity, false, parseSpecial)
        if n < 0 {
            capacity = -n
            tokens = [llama_token](repeating: 0, count: Int(capacity))
            n = llama_tokenize(vocab, text, Int32(utf8.count - 1), &tokens, capacity, false, parseSpecial)
        }
        return n > 0 ? Array(tokens.prefix(Int(n))) : []
    }

    private func piece(_ token: llama_token) -> [UInt8] {
        var buf = [CChar](repeating: 0, count: 64)
        var n = llama_token_to_piece(vocab, token, &buf, 64, 0, false)
        if n < 0 {
            buf = [CChar](repeating: 0, count: Int(-n))
            n = llama_token_to_piece(vocab, token, &buf, -n, 0, false)
        }
        return buf.prefix(Int(max(0, n))).map { UInt8(bitPattern: $0) }
    }

    /// Feeds `tokens` from position `from`, in batches.
    private func prefill(_ tokens: [llama_token], from start: Int) throws {
        var i = start
        while i < tokens.count {
            let end = min(tokens.count, i + Int(nBatch))
            var slice = Array(tokens[i..<end])
            let rc = slice.withUnsafeMutableBufferPointer { buf in
                llama_decode(ctx, llama_batch_get_one(buf.baseAddress, Int32(buf.count)))
            }
            if rc != 0 { throw Failure.decode(rc) }
            i = end
        }
    }

    /// All llama.cpp work runs here. ggml's worker threads inherit the QoS of the thread that
    /// starts them, and a low QoS confines them to the efficiency cores.
    private static let queue = DispatchQueue(label: "phonon.refiner", qos: .userInteractive)

    func refineAsync(transcript: String, mode: WritingMode?, vocabulary: [String],
                     isCancelled: @escaping @Sendable () -> Bool = { false }) async throws -> Output {
        try await withCheckedThrowingContinuation { cont in
            Self.queue.async {
                cont.resume(with: Result { try self.refine(transcript: transcript, mode: mode, vocabulary: vocabulary, isCancelled: isCancelled) })
            }
        }
    }

    /// Refines one transcript on the calling thread. `isCancelled` is polled between tokens.
    func refine(transcript: String, mode: WritingMode?, vocabulary: [String],
                isCancelled: () -> Bool = { false }) throws -> Output {
        lock.lock()
        defer { lock.unlock() }
        var stats = Stats()
        let terms = RefinerPrompt.capVocabulary(RefinerPrompt.relevantVocabulary(vocabulary, transcript: transcript), transcript: transcript)
        let system = RefinerPrompt.system(vocabulary: terms, modeLine: mode?.line ?? "")
        let prompt = RefinerPrompt.render(system: system, user: transcript)
        let tokens = tokenize(prompt, parseSpecial: true)
        guard !tokens.isEmpty else { throw Failure.tokenize }
        let budget = RefinerPrompt.tokenBudget(inputTokens: tokenize(transcript, parseSpecial: false).count)
        stats.promptTokens = tokens.count
        stats.budget = budget
        if tokens.count + budget > Int(nCtx) { throw Failure.contextFull }

        // Reuse the KV state of the shared prefix (the system prompt) when the memory allows a
        // partial removal; LFM3's short-conv layers may not, and then the prompt is fed whole.
        let memory = llama_get_memory(ctx)
        var common = 0
        while common < min(cached.count, tokens.count) - 1 && cached[common] == tokens[common] { common += 1 }
        if common > 0 && llama_memory_seq_rm(memory, 0, Int32(common), -1) {
            stats.reusedTokens = common
        } else {
            llama_memory_clear(memory, true)
            common = 0
        }
        cached = []
        let prefillStart = Date()
        try prefill(tokens, from: common)
        stats.prefillSeconds = Date().timeIntervalSince(prefillStart)
        cached = tokens

        let decodeStart = Date()
        var bytes: [UInt8] = []
        var produced = 0
        while produced < budget {
            if isCancelled() { break }
            var token = llama_sampler_sample(sampler, ctx, -1)
            if llama_vocab_is_eog(vocab, token) { break }
            bytes += piece(token)
            produced += 1
            cached.append(token)
            let rc = llama_decode(ctx, llama_batch_get_one(&token, 1))
            if rc != 0 { throw Failure.decode(rc) }
        }
        stats.decodeSeconds = Date().timeIntervalSince(decodeStart)
        stats.outputTokens = produced

        let raw = String(decoding: bytes, as: UTF8.self)
        let cleaned = RefinerPrompt.clean(raw)
        var (text, fired) = RefinerPrompt.guardOutput(spoken: transcript, output: cleaned)
        if !fired { text = RefinerPrompt.restoreTerms(spoken: transcript, output: text, vocabulary: terms) }
        return Output(text: text, raw: raw, guardFired: fired, stats: stats)
    }
}
