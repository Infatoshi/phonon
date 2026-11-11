// Shortlist-head evaluation (Mac CPU). Built against llama.cpp b12132 + ios/llama-b12132-phonon.patch +
// ios/llama-b12132-shortlist.patch (see ios/build_shortlist_eval.sh).
//
// For each item (JSONL: id, system, user) it decodes the shortlist model greedily along the full-head trajectory
// (debug mode: every step also computes every other row), and records per step whether rule R1 fell back and
// whether the shortlist alone would have chosen another token. Under deterministic greedy decoding, the shortlist
// head with fallback produces a different text iff some step is "missed" (no fallback and the shortlist choice
// differs from the full argmax), so this measures the real decoder's difference rate exactly.
// For the first `n_check` items it also decodes the stock full-head model (tied output) and checks that the token
// sequences are equal and that every logit (all 128,000 ids, every step) is bit-identical.
// Output: one JSON line per item (no text is printed for owner data; this tool is run on public items only).
//
// usage: shortlist_eval FULL.gguf SL.gguf ITEMS.jsonl OUT.jsonl THRESHOLD N_THREADS N_CHECK [MODE]
//   MODE "debug" (default) or "real" (the decoder as it would ship: R1 fallback only, compares to the full model)
#include "llama.h"
#include "ggml-cpu.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static std::vector<llama_token> tokenize(const llama_vocab * v, const std::string & s, bool special) {
    std::vector<llama_token> t(s.size() + 16);
    int n = llama_tokenize(v, s.c_str(), (int) s.size(), t.data(), (int) t.size(), false, special);
    if (n < 0) {
        t.resize(-n);
        n = llama_tokenize(v, s.c_str(), (int) s.size(), t.data(), (int) t.size(), false, special);
    }
    t.resize(n > 0 ? n : 0);
    return t;
}

static std::string piece(const llama_vocab * v, llama_token t) {
    char buf[256];
    int n = llama_token_to_piece(v, t, buf, sizeof(buf), 0, false);
    return n > 0 ? std::string(buf, n) : std::string();
}

static bool prefill(llama_context * ctx, std::vector<llama_token> toks) {
    llama_memory_clear(llama_get_memory(ctx), true);
    return llama_decode(ctx, llama_batch_get_one(toks.data(), (int) toks.size())) == 0;
}

static llama_context * make_ctx(llama_model * m, bool embd, int n_threads, ggml_threadpool_t tp) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 4096;
    cp.n_ubatch = 512;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    if (embd) {
        cp.embeddings = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    }
    llama_context * c = llama_init_from_model(m, cp);
    if (c && tp) llama_attach_threadpool(c, tp, tp);
    return c;
}

int main(int argc, char ** argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: %s FULL SL ITEMS OUT THRESHOLD N_THREADS N_CHECK [debug|real]\n", argv[0]);
        return 1;
    }
    const float T = std::stof(argv[5]);
    const int n_threads = std::stoi(argv[6]);
    const int n_check = std::stoi(argv[7]);
    const bool real = argc > 8 && std::string(argv[8]) == "real";
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * mfull = llama_model_load_from_file(argv[1], mp);
    llama_model * msl = llama_model_load_from_file(argv[2], mp);
    if (!mfull || !msl) { fprintf(stderr, "load failed\n"); return 1; }
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    ggml_threadpool_t tp = ggml_threadpool_new(&tpp);
    llama_context * cfull = make_ctx(mfull, false, n_threads, tp);
    llama_context * csl = make_ctx(msl, false, n_threads, tp);
    if (!cfull || !csl || llama_phonon_sl_k(csl) <= 0) { fprintf(stderr, "context or shortlist missing\n"); return 1; }
    const llama_vocab * v = llama_model_get_vocab(msl);
    const int n_vocab = llama_vocab_n_tokens(v);
    std::vector<float> lsl(n_vocab);
    std::ifstream in(argv[3]);
    std::ofstream out(argv[4]);
    std::string line;
    int idx = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        json it = json::parse(line);
        const std::string sys = it["system"], user = it["user"];
        std::string prompt = "<|startoftext|><|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\n" + user +
                             "<|im_end|>\n<|im_start|>assistant\n<think></think>";
        auto toks = tokenize(v, prompt, true);
        auto inp = tokenize(v, user, false);
        const int cap = (int) (1.5 * inp.size()) + 32;
        std::vector<llama_token> extras = inp;
        const std::string vk = "Vocabulary the speaker may use:";
        size_t vp = sys.find(vk);
        bool has_vocab = vp != std::string::npos;
        if (has_vocab) {
            size_t e = sys.find('\n', vp);
            std::string terms = sys.substr(vp + vk.size(), e == std::string::npos ? std::string::npos : e - vp - vk.size());
            for (auto t : tokenize(v, terms, false)) extras.push_back(t);
            for (auto t : tokenize(v, " " + terms, false)) extras.push_back(t);
        }
        const int groups = llama_phonon_sl_set_extras(csl, extras.data(), (int) extras.size());
        const bool check = idx < n_check;
        json r;
        r["id"] = it["id"];
        r["prompt_tok"] = toks.size();
        r["extra_groups"] = groups;
        r["has_vocab"] = has_vocab;
        bool ok = prefill(csl, toks) && (!check && !real ? true : prefill(cfull, toks));
        int steps = 0, fb = 0, missed = 0, first_miss = -1, logit_bad = 0, seq_diff = -1;
        std::vector<json> misses;
        std::string text;
        uint64_t h_sl = 1469598103934665603ULL, h_full = 1469598103934665603ULL;
        bool full_done = false, sl_done = false;
        while (ok && steps < cap) {
            int32_t f = 0; float best = 0; llama_token sl_tok = -1;
            llama_token tok;
            if (real) {
                // the shipped decoder: shortlist + R1; compared to the full model decoded alongside
                tok = sl_done ? -1 : llama_phonon_sl_sample(csl, T, 0, &f, &best, &sl_tok);
                llama_token ft = -1;
                if (!full_done) {
                    const float * lf = llama_get_logits_ith(cfull, -1);
                    ft = 0;
                    for (int i = 1; i < n_vocab; ++i) if (lf[i] > lf[ft]) ft = i;
                }
                if (seq_diff < 0 && tok != ft) seq_diff = steps;
                fb += f;
                steps++;
                if (!sl_done) { h_sl = (h_sl ^ (uint32_t) tok) * 1099511628211ULL; sl_done = llama_vocab_is_eog(v, tok); }
                if (!full_done) { h_full = (h_full ^ (uint32_t) ft) * 1099511628211ULL; full_done = llama_vocab_is_eog(v, ft); }
                if (sl_done && full_done) break;
                if (!sl_done) ok = llama_decode(csl, llama_batch_get_one(&tok, 1)) == 0;
                if (!full_done && ok) ok = llama_decode(cfull, llama_batch_get_one(&ft, 1)) == 0;
                continue;
            }
            tok = llama_phonon_sl_sample(csl, T, 1, &f, &best, &sl_tok);  // full argmax; sl_tok = shortlist choice
            if (check) {
                const float * lf = llama_get_logits_ith(cfull, -1);
                llama_token ft = 0;
                for (int i = 1; i < n_vocab; ++i) if (lf[i] > lf[ft]) ft = i;
                llama_phonon_sl_full_logits(csl, lsl.data());
                int bad = 0;
                for (int i = 0; i < n_vocab; ++i) if (memcmp(&lf[i], &lsl[i], sizeof(float)) != 0) bad++;
                logit_bad += bad;
                if (seq_diff < 0 && ft != tok) seq_diff = steps;
            }
            steps++;
            fb += f;
            if (!f && sl_tok != tok) {
                missed++;
                if (first_miss < 0) {
                    first_miss = steps - 1;
                    misses.push_back({{"step", steps - 1}, {"full", piece(v, tok)}, {"sl", piece(v, sl_tok)},
                                      {"best", best}, {"off", text.size()}});
                }
            }
            h_full = (h_full ^ (uint32_t) tok) * 1099511628211ULL;
            if (llama_vocab_is_eog(v, tok)) break;
            text += piece(v, tok);
            ok = llama_decode(csl, llama_batch_get_one(&tok, 1)) == 0;
            if (check && ok) ok = llama_decode(cfull, llama_batch_get_one(&tok, 1)) == 0;
        }
        r["ok"] = ok;
        r["steps"] = steps;
        r["fallback"] = fb;
        if (real) {
            r["first_diff"] = seq_diff;
            r["hash_sl"] = h_sl;
            r["hash_full"] = h_full;
        } else {
            r["missed"] = missed;
            r["first_miss"] = first_miss;
            r["misses"] = misses;
            r["full_text"] = text;
            if (check) { r["logit_bitdiff"] = logit_bad; r["seq_vs_tied_first_diff"] = seq_diff; }
        }
        out << r.dump() << "\n";
        out.flush();
        idx++;
    }
    llama_free(csl);
    llama_free(cfull);
    ggml_threadpool_free(tp);
    llama_model_free(msl);
    llama_model_free(mfull);
    return 0;
}
