import Foundation
import llama

/// The Phonon refiner (fine-tuned LFM3-1.7B, Q4_K_S GGUF) through llama.cpp on the CPU: a backgrounded iPhone app
/// may not use the GPU, and the refiner runs while the user is in another app. Same prompt as the Mac app and
/// intent_v0's gen_llama.py: the refiner system prompt + the writing-mode line, the raw transcript as the user turn,
/// the chat template with thinking off, greedy, output capped at 1.5 x input tokens + 32, and the length guard.
final class Refiner: @unchecked Sendable {
    static let systemPrompt =
        "Rewrite the raw dictation transcript into the exact text the speaker intended. Fix recognition errors "
        + "and technical terms, keep the speaker's wording and casing style, do not add or drop content."
    static let modeLines = [
        "minimal": "Writing mode: minimal (notes and documents).",
        "casual": "Writing mode: casual (a chat message).",
        "formal": "Writing mode: formal (an email or document).",
        "prompt": "Writing mode: prompt (an instruction to an AI assistant).",
    ]

    struct Output {
        let text: String
        let promptTokens: Int
        let genTokens: Int
        let prefillMs: Int
        let decodeMs: Int
        let guardFired: Bool
    }

    private let model: OpaquePointer
    private let ctx: OpaquePointer
    private let vocab: OpaquePointer
    private let sampler: UnsafeMutablePointer<llama_sampler>
    let threads: Int32

    init?(path: String, threads: Int32 = 4) {
        llama_backend_init()
        var mp = llama_model_default_params()
        mp.n_gpu_layers = 0
        guard let m = llama_model_load_from_file(path, mp) else { return nil }
        var cp = llama_context_default_params()
        cp.n_ctx = 2048
        cp.n_batch = 512
        cp.n_ubatch = 512
        cp.n_threads = threads
        cp.n_threads_batch = threads
        guard let c = llama_init_from_model(m, cp) else {
            llama_model_free(m)
            return nil
        }
        model = m
        ctx = c
        vocab = llama_model_get_vocab(m)
        sampler = llama_sampler_init_greedy()
        self.threads = threads
    }

    deinit {
        llama_sampler_free(sampler)
        llama_free(ctx)
        llama_model_free(model)
    }

    private func tokenize(_ s: String) -> [llama_token] {
        let n = Int32(s.utf8.count)
        var toks = [llama_token](repeating: 0, count: Int(n) + 16)
        var r = llama_tokenize(vocab, s, n, &toks, Int32(toks.count), false, true)
        if r < 0 {
            toks = [llama_token](repeating: 0, count: Int(-r))
            r = llama_tokenize(vocab, s, n, &toks, Int32(toks.count), false, true)
        }
        return Array(toks.prefix(Int(max(r, 0))))
    }

    private func piece(_ t: llama_token) -> [UInt8] {
        var buf = [CChar](repeating: 0, count: 64)
        let n = llama_token_to_piece(vocab, t, &buf, Int32(buf.count), 0, false)
        guard n > 0 else { return [] }
        return buf.prefix(Int(n)).map { UInt8(bitPattern: $0) }
    }

    /// A snapshot of the model state after the fixed head of the prompt (system prompt + mode line + "user"
    /// header): a dictation in the same mode restores it and only prefills its own words. LFM3 is a hybrid
    /// (attention + short convolutions): its conv state cannot be rolled back with seq_rm, so the whole sequence
    /// state is saved and restored instead.
    private var cachedHead: [llama_token] = []
    private var headState: [UInt8] = []

    func refine(_ raw: String, mode: String) -> Output {
        let system = Self.systemPrompt + "\n" + (Self.modeLines[mode] ?? Self.modeLines["minimal"]!)
        let head = tokenize("<|startoftext|><|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n")
        var tail = tokenize("\(raw)<|im_end|>\n<|im_start|>assistant\n<think></think>")
        let tokens = head + tail
        let inputTokens = tokenize(raw).count
        let cap = Int(1.5 * Double(inputTokens)) + 32
        let mem = llama_get_memory(ctx)
        llama_sampler_reset(sampler)
        let t0 = Date()
        var ok = true
        llama_memory_clear(mem, true)
        if head == cachedHead && !headState.isEmpty {
            let n = headState.withUnsafeBufferPointer { p in
                llama_state_seq_set_data(ctx, p.baseAddress, p.count, 0)
            }
            ok = n > 0
        } else {
            ok = false
        }
        if !ok {
            llama_memory_clear(mem, true)
            var h = head
            ok = h.withUnsafeMutableBufferPointer { p in
                llama_decode(ctx, llama_batch_get_one(p.baseAddress, Int32(p.count))) == 0
            }
            if ok {
                let size = llama_state_seq_get_size(ctx, 0)
                var buf = [UInt8](repeating: 0, count: size)
                let got = buf.withUnsafeMutableBufferPointer { p in llama_state_seq_get_data(ctx, p.baseAddress, size, 0) }
                headState = got > 0 ? Array(buf.prefix(got)) : []
                cachedHead = got > 0 ? head : []
            }
        }
        if ok {
            ok = tail.withUnsafeMutableBufferPointer { p in
                llama_decode(ctx, llama_batch_get_one(p.baseAddress, Int32(p.count))) == 0
            }
        }
        let t1 = Date()
        var out: [UInt8] = []
        var gen = 0
        while ok && gen < cap {
            var tok = llama_sampler_sample(sampler, ctx, -1)
            if llama_vocab_is_eog(vocab, tok) { break }
            out += piece(tok)
            gen += 1
            ok = llama_decode(ctx, llama_batch_get_one(&tok, 1)) == 0
        }
        let t2 = Date()
        var text = String(decoding: out, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        // the length guard: a runaway or a failed decode falls back to the raw transcript
        let rawWords = raw.split(separator: " ").count
        let fired = !ok || text.isEmpty || Double(text.split(separator: " ").count) > 1.2 * Double(rawWords) + 20
        // the shared output stage (crates/phonon-llm linebreak): numbered steps one per line, and in formal mode
        // the greeting and sign-off on their own lines; the refiner itself never writes a newline
        if fired { text = raw } else { text = LineBreaks.apply(text, mode: mode) }
        return Output(text: text, promptTokens: tokens.count, genTokens: gen,
                      prefillMs: Int(t1.timeIntervalSince(t0) * 1000), decodeMs: Int(t2.timeIntervalSince(t1) * 1000),
                      guardFired: fired)
    }
}
