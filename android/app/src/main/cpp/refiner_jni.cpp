// Phonon refiner (LFM3-1.7B GGUF) through llama.cpp on the CPU, inside the IME process.
// A line-for-line port of ios/App/Refiner.swift: system prompt + mode line, the raw transcript as the user turn,
// "<|startoftext|><|im_start|>...<think></think>", greedy, output capped at 1.5 x input tokens + 32, the length guard,
// and the KV-cache reuse of the fixed prompt head.
//
// CPU threads (auto mode, the keyboard's path since 2026-11-06 evening): llama.cpp without an attached threadpool
// spawns and joins its worker threads on every graph, i.e. every token, unpinned; on Test Lab phones that cost
// 30-80% of decode speed and sometimes collapsed to seconds per token (DEVLOG 4:10 pm). Auto mode attaches one
// persistent ggml threadpool pinned (strict, one thread per core) to the K highest-cpu_capacity cores. K comes from a
// short probe at first load among 2/4/6 plus the big-core count and one fewer (never all cores), cached per device in
// a small text file: the fastest K for
// single-token decode, and, if a different K prefills a 32-token batch faster, a second pool for prefill. A K whose
// probe token runs over 3x the best so far is abandoned; at runtime two decode tokens over 4x the fastest seen move
// decode to the next K (pinned pools collapsed to seconds per token on some phones). The pools are paused after every
// load and dictation so their threads sleep on a condition variable while the keyboard is idle, and the calling
// thread's own CPU affinity is restored (ggml pins the thread that kicks a graph to one core).
// RAM gate: MemTotal < 4000 MB or MemAvailable < 1500 MB -> no refiner (load returns 0, the IME commits the raw
// transcript); MemTotal < 5000 MB -> no repacked weight copy and K = 2.
#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>
#include <cstring>
#include <dlfcn.h>
#include <link.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#define TAG "PhononRefiner"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGP(...) __android_log_print(ANDROID_LOG_INFO, "Phonon", __VA_ARGS__)

namespace {

const char *kSystemPrompt =
    "Rewrite the raw dictation transcript into the exact text the speaker intended. Fix recognition errors "
    "and technical terms, keep the speaker's wording and casing style, do not add or drop content.";

std::string modeLine(const std::string &mode) {
    if (mode == "casual") return "Writing mode: casual (a chat message).";
    if (mode == "formal") return "Writing mode: formal (an email or document).";
    if (mode == "prompt") return "Writing mode: prompt (an instruction to an AI assistant).";
    return "Writing mode: minimal (notes and documents).";
}

// ---------- threadpool plumbing (the CPU backend is a dlopen'd variant module, so everything goes by address) ----------

std::string g_variantPath;  // full path of the libggml-cpu-*.so ggml picked
using TpNew = ggml_threadpool_t (*)(ggml_threadpool_params *);
using TpVoid = void (*)(ggml_threadpool_t);
TpNew g_tpNew = nullptr;
TpVoid g_tpFree = nullptr, g_tpPause = nullptr;

void resolvePool() {
    static bool done = false;
    if (done) return;
    done = true;
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    if (reg) {
        g_tpNew = (TpNew)ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
        g_tpFree = (TpVoid)ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
    }
    // ggml_threadpool_pause is exported by the module but not offered through get_proc_address
    if (!g_variantPath.empty()) {
        void *h = dlopen(g_variantPath.c_str(), RTLD_NOW | RTLD_NOLOAD);
        if (h) g_tpPause = (TpVoid)dlsym(h, "ggml_threadpool_pause");
    }
    LOGP("TUNE pool_api new=%d free=%d pause=%d", g_tpNew != nullptr, g_tpFree != nullptr, g_tpPause != nullptr);
}

struct Core {
    int cpu;
    long cap;
    long khz;
};

long readLong(const std::string &path, long dflt) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return dflt;
    long v = dflt;
    if (fscanf(f, "%ld", &v) != 1) v = dflt;
    fclose(f);
    return v;
}

// Cores this process may run on, highest cpu_capacity first (then max frequency; ties keep ascending index), as
// RefinerKernelBenchTest's topN.
std::vector<Core> coresByCapacity() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    bool haveMask = sched_getaffinity(0, sizeof allowed, &allowed) == 0;
    int n = (int)sysconf(_SC_NPROCESSORS_CONF);
    std::vector<Core> v;
    for (int i = 0; i < n && i < CPU_SETSIZE; i++) {
        if (haveMask && !CPU_ISSET(i, &allowed)) continue;
        std::string d = "/sys/devices/system/cpu/cpu" + std::to_string(i);
        long khz = readLong(d + "/cpufreq/cpuinfo_max_freq", 0);
        v.push_back({i, readLong(d + "/cpu_capacity", khz / 1000), khz});
    }
    std::stable_sort(v.begin(), v.end(), [](const Core &a, const Core &b) {
        return a.cap != b.cap ? a.cap > b.cap : a.khz > b.khz;
    });
    return v;
}

struct Pool {
    ggml_threadpool_t tp = nullptr;
    int k = 0;
    std::vector<int> cpus;
};

// pin false (bench spec pin=0 only): the same persistent pool, threads left to the scheduler (no cpumask, not strict).
Pool makePool(const std::vector<Core> &cores, int k, int poll, bool pin = true) {
    Pool p;
    if (!g_tpNew || k <= 0 || k > (int)cores.size()) return p;
    ggml_threadpool_params tpp = ggml_threadpool_params_default(k);
    tpp.poll = (uint32_t)poll;
    tpp.strict_cpu = pin;
    tpp.paused = true;  // the first graph resumes it; creating it unpaused would pin the calling thread right away
    for (int i = 0; i < k; i++) {
        int c = cores[i].cpu;
        if (pin && c >= 0 && c < GGML_MAX_N_THREADS) tpp.cpumask[c] = true;
        p.cpus.push_back(pin ? c : -1);
    }
    std::sort(p.cpus.begin(), p.cpus.end());
    p.tp = g_tpNew(&tpp);
    p.k = p.tp ? k : 0;
    return p;
}

void freePool(Pool &p) {
    if (p.tp && g_tpFree) g_tpFree(p.tp);
    p = Pool();
}

std::string join(const std::vector<int> &v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}

// Restores the calling thread's CPU affinity on scope exit (ggml pins the thread that kicks a graph to the pool's
// last core) and pauses the pools, so no worker spins while the keyboard is idle.
struct AffinityGuard {
    cpu_set_t saved;
    bool ok;
    std::vector<ggml_threadpool_t> pause;
    AffinityGuard() { CPU_ZERO(&saved); ok = sched_getaffinity(0, sizeof saved, &saved) == 0; }
    ~AffinityGuard() {
        if (g_tpPause)
            for (auto tp : pause)
                if (tp) g_tpPause(tp);
        if (ok) sched_setaffinity(0, sizeof saved, &saved);
    }
};

void memInfo(long &totalMb, long &availMb) {
    totalMb = availMb = -1;
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        long kb;
        if (sscanf(line, "MemTotal: %ld kB", &kb) == 1) totalMb = kb / 1024;
        else if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) availMb = kb / 1024;
    }
    fclose(f);
}

long long cpuMsNow() {
    timespec ts{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// "key=value,key=value" overrides for the bench (the keyboard passes ""). Keys: k (force K, no probe, no guard), kb
// (prefill pool size), poll (ggml poll level, default 50), probe (0 = ignore the cache file), pprobe (0 = no prefill
// probe), passes (probe passes, default 1), guard (0 = no runtime collapse guard), pin (0 = pools not pinned).
int specInt(const std::string &spec, const char *key, int dflt) {
    std::string k = std::string(key) + "=";
    size_t pos = 0;
    while (pos < spec.size()) {
        size_t end = spec.find(',', pos);
        if (end == std::string::npos) end = spec.size();
        if (spec.compare(pos, k.size(), k) == 0) return atoi(spec.c_str() + pos + k.size());
        pos = end + 1;
    }
    return dflt;
}

// One cached prompt head: its tokens and sequence 0's KV + conv state right after them.
struct Head {
    std::vector<llama_token> tokens;
    std::vector<uint8_t> state;
    uint64_t used = 0;
};

// Heads kept at once: one per writing mode (4) plus two with a dictionary vocabulary line. About 1 MB each.
constexpr size_t kMaxHeads = 6;

struct Refiner {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    const llama_vocab *vocab = nullptr;
    llama_sampler *sampler = nullptr;
    // Cached prompt heads, one per writing mode (and per vocabulary line), least recently used dropped first. Before
    // 2026-11-12 only the last head was kept, so every dictation after a mode switch re-prefilled its head.
    std::vector<Head> heads;
    uint64_t tick = 0;
    Pool pool, poolBatch;           // auto mode only; poolBatch.tp == nullptr -> prefill shares `pool`
    // runtime collapse guard (auto mode): reference ms/token (probe result, then the fastest token seen with this K)
    std::vector<Core> cores;
    int poll = 50;
    double refTokMs = 0;
    std::vector<int> triedK;
    std::string tunePath, tuneKey;
    bool guardOn = false;
    // Pools no longer attached. ggml's CPU backend keeps a pointer to the last pool it ran and pauses it when the next
    // graph brings a different one, so a pool may only be freed after a graph has run with its successor.
    std::vector<Pool> retired;
    bool pin = true;
    // Shortlist output head (ios/llama-b12132-shortlist.patch): on only when the model file carries phonon.sl.k,
    // phonon.sl.order and output.slx (ios/shortlist_gguf.py); llama_phonon_sl_k is 0 for today's tied refiner.gguf,
    // which keeps the stock greedy sampler and the stock output matmul.
    bool shortlist = false;
};

// Rule R1 of the shortlist head, the iPhone's value (ios/App/Refiner.swift slThreshold): when the best in-list logit
// is below this, every other row is computed and the full argmax taken.
constexpr float kSlThreshold = 28.152f;

// Decode tokens went pathological on the pinned pool (2 in a row > 4x the reference and +150 ms): move decode to the
// next K in 4, 2, 6 not yet tried (always < the core count), remember it in the tune file, keep generating. Thread
// count does not change the greedy output.
bool switchDecodePool(Refiner *r, double slowMs) {
    if (std::find(r->triedK.begin(), r->triedK.end(), r->pool.k) == r->triedK.end()) r->triedK.push_back(r->pool.k);
    int next = 0;
    for (int k : {4, 2, 6})
        if (k < (int)r->cores.size() && std::find(r->triedK.begin(), r->triedK.end(), k) == r->triedK.end()) {
            next = k;
            break;
        }
    if (!next) {
        r->guardOn = false;  // nothing left to try
        LOGP("TUNE collapse k=%d slow_ms=%.0f, no other K left", r->pool.k, slowMs);
        return false;
    }
    Pool np = makePool(r->cores, next, r->poll, r->pin);
    if (!np.tp) return false;
    int oldK = r->pool.k;
    llama_detach_threadpool(r->ctx);
    if (g_tpPause) g_tpPause(r->pool.tp);
    r->retired.push_back(r->pool);  // the CPU backend still points at it until the next graph: free it later
    r->pool = np;
    r->triedK.push_back(next);
    ggml_threadpool_t batch = r->poolBatch.tp ? r->poolBatch.tp : r->pool.tp;
    llama_attach_threadpool(r->ctx, r->pool.tp, batch);
    llama_set_n_threads(r->ctx, r->pool.k, r->poolBatch.tp ? r->poolBatch.k : r->pool.k);
    r->refTokMs = 0;
    if (!r->tunePath.empty()) {
        FILE *f = fopen(r->tunePath.c_str(), "w");
        if (f) { fprintf(f, "%s\t%d\t%d\n", r->tuneKey.c_str(), next, r->poolBatch.k); fclose(f); }
    }
    LOGP("TUNE collapse k=%d slow_ms=%.0f -> k=%d cpus=%s", oldK, slowMs, next, join(r->pool.cpus).c_str());
    return true;
}

std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &s) {
    int32_t n = (int32_t)s.size();
    std::vector<llama_token> toks(n + 16);
    int32_t r = llama_tokenize(vocab, s.c_str(), n, toks.data(), (int32_t)toks.size(), false, true);
    if (r < 0) {
        toks.resize(-r);
        r = llama_tokenize(vocab, s.c_str(), n, toks.data(), (int32_t)toks.size(), false, true);
    }
    toks.resize(r > 0 ? r : 0);
    return toks;
}

std::string piece(const llama_vocab *vocab, llama_token t) {
    char buf[64];
    int32_t n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
    return n > 0 ? std::string(buf, n) : std::string();
}

// Swift's split(separator: " ").count: non-empty runs between spaces (spaces only, not newlines).
int wordCount(const std::string &s) {
    int c = 0;
    bool in = false;
    for (char ch : s) {
        if (ch == ' ') { in = false; } else if (!in) { in = true; c++; }
    }
    return c;
}

std::string trim(const std::string &s) {
    const char *ws = " \t\n\r\v\f";
    size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

std::string jsonEscape(const std::string &s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char)c;
        }
    }
    return o;
}

long long msSince(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

namespace {

// Median ms of single-token decodes with pool `p` attached: 1 warm-up + 5 timed tokens from a cleared memory.
// Bails out (returns 1e9) as soon as one token, warm-up included, takes longer than limitMs: pinned pools can
// collapse to seconds per token on some phones (Pixel 5 top2 4.56 s, Galaxy S9 all-8 5.6 s), so a bad K costs one
// token, not six.
double probeDecodeMs(Refiner *r, Pool &p, const std::vector<llama_token> &toks, double limitMs) {
    llama_attach_threadpool(r->ctx, p.tp, p.tp);
    llama_set_n_threads(r->ctx, p.k, p.k);
    llama_memory_clear(llama_get_memory(r->ctx), true);
    std::vector<double> ms;
    bool bail = false;
    for (size_t i = 0; i < 6 && i < toks.size(); i++) {
        llama_token t = toks[i];
        auto t0 = std::chrono::steady_clock::now();
        if (llama_decode(r->ctx, llama_batch_get_one(&t, 1)) != 0) { bail = true; break; }
        double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (d > limitMs) { bail = true; break; }
        if (i >= 1) ms.push_back(d);
    }
    if (g_tpPause) g_tpPause(p.tp);
    llama_memory_clear(llama_get_memory(r->ctx), true);
    if (bail || ms.empty()) return 1e9;
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

// ms of one batched prefill of the first `n` tokens with pool `p` attached (called right after probeDecodeMs with the
// same pool, so the cores are already clocked up).
double probePrefillMs(Refiner *r, Pool &p, std::vector<llama_token> toks, size_t n) {
    toks.resize(std::min(n, toks.size()));
    llama_attach_threadpool(r->ctx, p.tp, p.tp);
    llama_set_n_threads(r->ctx, p.k, p.k);
    llama_memory_clear(llama_get_memory(r->ctx), true);
    auto t0 = std::chrono::steady_clock::now();
    bool ok = llama_decode(r->ctx, llama_batch_get_one(toks.data(), (int32_t)toks.size())) == 0;
    double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (g_tpPause) g_tpPause(p.tp);
    llama_memory_clear(llama_get_memory(r->ctx), true);
    return ok ? d : 1e9;
}

// The prompt head of a writing mode: the system prompt, the mode line and, when the dictionary has terms for this
// dictation, the vocabulary line the refiner was trained with (sidecar/polish_server.py refiner_system_prompt), then
// the user header. Empty `vocab` gives exactly the head used before the vocabulary line existed.
std::string headText(const std::string &mode, const std::string &vocab) {
    std::string system = std::string(kSystemPrompt) + "\n" + modeLine(mode);
    if (!vocab.empty()) system += "\nVocabulary the speaker may use: " + vocab;
    return "<|startoftext|><|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n";
}

// Restores the cached state of `head` into sequence 0, or prefills it from a cleared memory and caches the state
// (dropping the least recently used head when full). Sets `cached` when the state came from the cache.
bool loadHead(Refiner *r, const std::vector<llama_token> &head, bool &cached) {
    llama_memory_t mem = llama_get_memory(r->ctx);
    cached = false;
    for (auto &h : r->heads) {
        if (h.tokens != head) continue;
        llama_memory_clear(mem, true);
        if (llama_state_seq_set_data(r->ctx, h.state.data(), h.state.size(), 0) > 0) {
            h.used = ++r->tick;
            cached = true;
            return true;
        }
        break;  // a state that does not restore: prefill below and overwrite it
    }
    llama_memory_clear(mem, true);
    std::vector<llama_token> toks = head;
    if (llama_decode(r->ctx, llama_batch_get_one(toks.data(), (int32_t)toks.size())) != 0) return false;
    Head fresh;
    fresh.tokens = head;
    fresh.state.resize(llama_state_seq_get_size(r->ctx, 0));
    size_t got = llama_state_seq_get_data(r->ctx, fresh.state.data(), fresh.state.size(), 0);
    if (got == 0) return true;  // decoded fine, just not cacheable
    fresh.state.resize(got);
    fresh.used = ++r->tick;
    auto same = std::find_if(r->heads.begin(), r->heads.end(), [&](const Head &h) { return h.tokens == head; });
    if (same != r->heads.end()) {
        *same = std::move(fresh);
    } else if (r->heads.size() < kMaxHeads) {
        r->heads.push_back(std::move(fresh));
    } else {
        auto lru = std::min_element(r->heads.begin(), r->heads.end(),
                                    [](const Head &a, const Head &b) { return a.used < b.used; });
        *lru = std::move(fresh);
    }
    return true;
}

size_t headBytes(const Refiner *r) {
    size_t n = 0;
    for (auto &h : r->heads) n += h.state.size();
    return n;
}

std::string jstr(JNIEnv *env, jstring s) {
    if (!s) return std::string();
    const char *c = env->GetStringUTFChars(s, nullptr);
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}

std::string readFile(const std::string &path) {
    std::string s;
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return s;
    char buf[512];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

}  // namespace

// threads/repack apply only when autoTune is false (the pre-2026-11-06-evening path: no threadpool, `threads`
// unpinned, repack as given, no RAM gate). autoTune true: RAM gate, pinned persistent pool, K probed once and cached
// in tunePath, spec = bench overrides ("" in the keyboard).
extern "C" JNIEXPORT jlong JNICALL
Java_ai_phonon_keyboard_Refiner_nativeLoad(JNIEnv *env, jclass, jstring jpath, jint threads, jboolean repack,
                                           jboolean autoTune, jstring jtunePath, jstring jspec) {
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    std::string tunePath, spec;
    if (jtunePath) {
        const char *s = env->GetStringUTFChars(jtunePath, nullptr);
        tunePath = s;
        env->ReleaseStringUTFChars(jtunePath, s);
    }
    if (jspec) {
        const char *s = env->GetStringUTFChars(jspec, nullptr);
        spec = s;
        env->ReleaseStringUTFChars(jspec, s);
    }
    static bool backends_loaded = false;
    if (!backends_loaded) {
        // the CPU variants (libggml-cpu-android_armv*.so) sit beside this library in the app's native lib dir
        Dl_info info{};
        if (dladdr(reinterpret_cast<void *>(&Java_ai_phonon_keyboard_Refiner_nativeLoad), &info) && info.dli_fname) {
            std::string dir(info.dli_fname);
            dir = dir.substr(0, dir.find_last_of('/'));
            ggml_backend_load_all_from_path(dir.c_str());
            // which CPU variant ggml picked: the libggml-cpu-*.so now mapped into the process
            std::string variant = "none";
            dl_iterate_phdr(
                [](dl_phdr_info *i, size_t, void *out) {
                    const char *name = i->dlpi_name ? std::strstr(i->dlpi_name, "libggml-cpu") : nullptr;
                    if (name) {
                        *static_cast<std::string *>(out) = name;
                        g_variantPath = i->dlpi_name;
                    }
                    return 0;
                },
                &variant);
            __android_log_print(ANDROID_LOG_INFO, "Phonon", "BACKENDS dir=%s cpu=%s variant=%s", dir.c_str(),
                                ggml_backend_dev_count() > 0 ? ggml_backend_dev_description(ggml_backend_dev_get(0)) : "none",
                                variant.c_str());
        }
        backends_loaded = true;
    }
    llama_backend_init();
    // llama.cpp's warnings and its per-buffer weight sizes (which weights went to the repacked heap copy) to logcat
    llama_log_set(
        [](ggml_log_level level, const char *text, void *) {
            if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR || std::strstr(text, "buffer size"))
                __android_log_print(ANDROID_LOG_INFO, "PhononLlama", "%s", text);
        },
        nullptr);
    AffinityGuard guard;
    bool useRepack = repack;
    std::vector<Core> cores;
    std::vector<int> cands;
    int poll = 50, kb = 0, forced = 0, cachedK = 0, cachedKb = 0, pprobe = 0;
    bool pin = true;
    long memT = -1, memA = -1;
    std::string tuneKey;
    if (autoTune) {
        resolvePool();
        memInfo(memT, memA);
        cores = coresByCapacity();
        if ((memT >= 0 && memT < 4000) || (memA >= 0 && memA < 1500)) {
            LOGP("TUNE gate=skip memtotal_mb=%ld memavail_mb=%ld (refiner off: raw transcript)", memT, memA);
            env->ReleaseStringUTFChars(jpath, path);
            return 0;
        }
        useRepack = memT < 0 || memT >= 5000;
        poll = specInt(spec, "poll", 50);
        kb = specInt(spec, "kb", 0);
        forced = specInt(spec, "k", 0);
        pin = specInt(spec, "pin", 1) != 0;
        // prefill gets its own pool when another K prefills faster (A56: K=2 decode, 4-6 prefill, -9% per dictation)
        pprobe = specInt(spec, "pprobe", 1);
        // never all cores on big phones (Pixel 11 pinned to all 7 ran 937 ms/token, Galaxy S9 all 8 5.6 s/token):
        // K < cores. On 4 or fewer allowed cores K = all is still a probe candidate (x86 emulator: K=2 1692 ms vs
        // 4 threads 1062 ms); the probe's 3x bail drops it if it collapses.
        // Also probed: every big core (capacity >= half the top one, i.e. all clusters but the efficiency cores) and one
        // fewer, when that is 3 or more. Test Lab, same process, token-identical, ms/token over 8 dictations x 4 runs
        // (2026-11-12): Xiaomi 14 K=5 22.7 vs K=4 24.7 and K=6 24.4 (big = 6: prime + 5 A720); Galaxy A56 K=3 57.7
        // vs K=2 62.4 (big = the 3 A720 the process may use); Galaxy A16 K=3 80.3 vs K=2 79.6 (big = 2, nothing added).
        if (!useRepack) cands = {2};
        else {
            cands = {4, 6, 2};
            long maxCap = cores.empty() ? 0 : cores.front().cap;
            int nBig = (int)std::count_if(cores.begin(), cores.end(), [&](const Core &c) { return c.cap * 2 >= maxCap; });
            for (int k : {nBig - 1, nBig})
                if (k >= 3 && std::find(cands.begin(), cands.end(), k) == cands.end()) cands.push_back(k);
        }
        const int nCores = (int)cores.size();
        cands.erase(std::remove_if(cands.begin(), cands.end(),
                                   [&](int k) { return k > nCores || (k == nCores && nCores > 4); }),
                    cands.end());
        if (cands.empty()) cands = {std::max(1, (int)cores.size() - 1)};
        if (forced > 0) cands = {std::min(forced, std::max(1, (int)cores.size()))};
        struct stat st{};
        long long modelBytes = stat(path, &st) == 0 ? (long long)st.st_size : -1;
        // v3: the big-core candidates were added (a v2 file would keep a phone on its old K without probing them)
        tuneKey = "v3 pp=" + std::to_string(pprobe) + " ps=" + std::to_string(specInt(spec, "passes", 1)) + " repack=" + std::to_string(useRepack ? 1 : 0) +
                  " model=" + std::to_string(modelBytes) + " cores=";
        for (auto &c : cores) tuneKey += std::to_string(c.cpu) + ":" + std::to_string(c.cap) + ";";
        if (cands.size() > 1 && !tunePath.empty() && specInt(spec, "probe", 1) != 0) {
            // file: key \t K \t Kb  (Kb 0 = prefill shares the decode pool)
            std::string s = readFile(tunePath);
            size_t tab = s.find('\t');
            if (tab != std::string::npos && s.compare(0, tab, tuneKey) == 0) {
                cachedK = atoi(s.c_str() + tab + 1);
                size_t tab2 = s.find('\t', tab + 1);
                if (tab2 != std::string::npos) cachedKb = atoi(s.c_str() + tab2 + 1);
            }
            if (std::find(cands.begin(), cands.end(), cachedK) == cands.end()) cachedK = cachedKb = 0;
            if (cachedKb > (int)cores.size()) cachedKb = 0;
        }
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    // repack = llama.cpp's CPU extra buffer: a second, kernel-friendly copy of the weights on the heap (~1.36 GB on
    // Test Lab phones on top of the 943 MB mmap). Off keeps only the mmap; the speed cost is what the bench measures.
    mp.use_extra_bufts = useRepack;
    llama_model *m = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!m) return 0;
    int k0 = autoTune ? (cachedK ? cachedK : *std::max_element(cands.begin(), cands.end())) : threads;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = k0;
    cp.n_threads_batch = k0;
    llama_context *c = llama_init_from_model(m, cp);
    if (!c) { llama_model_free(m); return 0; }
    auto *r = new Refiner();
    r->model = m;
    r->ctx = c;
    r->vocab = llama_model_get_vocab(m);
    r->sampler = llama_sampler_init_greedy();
    r->shortlist = llama_phonon_sl_k(c) > 0;
    LOGP("SHORTLIST k=%d", llama_phonon_sl_k(c));
    if (autoTune && g_tpNew) {
        auto t0 = std::chrono::steady_clock::now();
        std::string probeLog;
        int best = cachedK, bestKb = cachedKb;
        if (!best && cands.size() == 1) best = cands[0];
        if (!best) {
            std::vector<llama_token> toks = tokenize(r->vocab, "<|startoftext|><|im_start|>system\n" + std::string(kSystemPrompt));
            // passes > 1: the candidates again in reverse order, keeping each K's best time, so one slow sample (a
            // governor ramp, a background burst) cannot decide K
            int passes = std::max(1, specInt(spec, "passes", 1));
            std::vector<double> decMs(cands.size(), 1e18), preMs(cands.size(), 1e18);
            double bestSoFar = 1e18;
            bool first = true;
            for (int pass = 0; pass < passes; pass++) {
                for (size_t ci = 0; ci < cands.size(); ci++) {
                    size_t idx = pass % 2 ? cands.size() - 1 - ci : ci;
                    int k = cands[idx];
                    if (pass > 0 && decMs[idx] >= 1e9) continue;  // bailed once: never again
                    Pool p = makePool(cores, k, poll, pin);
                    if (!p.tp) continue;
                    // any token slower than 3x the best K so far ends this K's probe (2 s before there is a best)
                    double limit = bestSoFar < 1e17 ? 3 * bestSoFar : 2000;
                    if (first) {  // page in the weights and let the governor ramp before anything is timed
                        probeDecodeMs(r, p, toks, 1e9);
                        first = false;
                    }
                    double ms = probeDecodeMs(r, p, toks, limit);
                    double pre = pprobe && ms < 1e9 ? probePrefillMs(r, p, toks, 32) : 1e9;
                    char b[64];
                    if (ms >= 1e9) snprintf(b, sizeof b, "%s%d:bail>%.0f", probeLog.empty() ? "" : ",", k, limit);
                    else if (pprobe) snprintf(b, sizeof b, "%s%d:%.1f/p%.0f", probeLog.empty() ? "" : ",", k, ms, pre);
                    else snprintf(b, sizeof b, "%s%d:%.1f", probeLog.empty() ? "" : ",", k, ms);
                    probeLog += b;
                    decMs[idx] = std::min(decMs[idx], ms);
                    bestSoFar = std::min(bestSoFar, ms);
                    if (pprobe) preMs[idx] = std::min(preMs[idx], pre);
                    llama_detach_threadpool(r->ctx);
                    r->retired.push_back(p);
                }
            }
            double bestMs = 1e18, bestPre = 1e18;
            for (size_t i = 0; i < cands.size(); i++) {
                if (decMs[i] < bestMs) { bestMs = decMs[i]; best = cands[i]; }
                if (pprobe && preMs[i] < bestPre) { bestPre = preMs[i]; bestKb = cands[i]; }
            }
            if (bestMs >= 1e9) best = cands[0];  // every K bailed: keep the first and let the runtime guard move it
            else r->refTokMs = bestMs;
            if (bestKb == best) bestKb = 0;
            if (best && !tunePath.empty()) {
                FILE *f = fopen(tunePath.c_str(), "w");
                if (f) { fprintf(f, "%s\t%d\t%d\n", tuneKey.c_str(), best, bestKb); fclose(f); }
            }
        }
        if (kb <= 0) kb = bestKb;  // spec kb overrides the probed one
        if (best) r->pool = makePool(cores, best, poll, pin);
        if (r->pool.tp && kb > 0 && kb != best) r->poolBatch = makePool(cores, kb, poll, pin);
        if (r->pool.tp) {
            llama_attach_threadpool(r->ctx, r->pool.tp, r->poolBatch.tp ? r->poolBatch.tp : r->pool.tp);
            llama_set_n_threads(r->ctx, r->pool.k, r->poolBatch.tp ? r->poolBatch.k : r->pool.k);
        } else {
            llama_set_n_threads(r->ctx, std::min(4, std::max(1, (int)cores.size())), std::min(4, std::max(1, (int)cores.size())));
        }
        guard.pause = {r->pool.tp, r->poolBatch.tp};
        r->cores = cores;
        r->poll = poll;
        r->pin = pin;
        r->tunePath = tunePath;
        r->tuneKey = tuneKey;
        r->guardOn = forced <= 0 && specInt(spec, "guard", 1) != 0;
        LOGP("TUNE gate=run memtotal_mb=%ld memavail_mb=%ld repack=%d k=%d cpus=%s kb=%d poll=%d cached=%d probe=%s "
             "tune_ms=%lld pause=%d key=%s",
             memT, memA, useRepack ? 1 : 0, r->pool.k, join(r->pool.cpus).c_str(), r->poolBatch.k, poll,
             cachedK ? 1 : 0, probeLog.empty() ? "-" : probeLog.c_str(), (long long)msSince(t0), g_tpPause != nullptr,
             tuneKey.c_str());
    } else if (autoTune) {
        LOGP("TUNE gate=run no-threadpool-api repack=%d threads=%d", useRepack ? 1 : 0, k0);
    }
    LOGI("loaded %s threads=%d auto=%d system_info=%s", "model", k0, autoTune ? 1 : 0, llama_print_system_info());
    return (jlong)r;
}

extern "C" JNIEXPORT void JNICALL Java_ai_phonon_keyboard_Refiner_nativeFree(JNIEnv *, jclass, jlong h) {
    auto *r = (Refiner *)h;
    if (!r) return;
    llama_sampler_free(r->sampler);
    llama_free(r->ctx);  // before the pools: the context references them
    for (auto &p : r->retired) freePool(p);
    freePool(r->poolBatch);
    freePool(r->pool);
    llama_model_free(r->model);
    delete r;
}

// CPU time of the whole process (all threads) in ms, for the bench's idle-cost check.
extern "C" JNIEXPORT jlong JNICALL Java_ai_phonon_keyboard_Refiner_nativeCpuMs(JNIEnv *, jclass) {
    return (jlong)cpuMsNow();
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_phonon_keyboard_Refiner_nativeRefine(JNIEnv *env, jclass, jlong h, jstring jraw, jstring jmode,
                                             jstring jvocab) {
    auto *r = (Refiner *)h;
    std::string raw = jstr(env, jraw), mode = jstr(env, jmode), vocabLine = jstr(env, jvocab);

    std::vector<llama_token> head = tokenize(r->vocab, headText(mode, vocabLine));
    std::vector<llama_token> tail = tokenize(r->vocab, raw + "<|im_end|>\n<|im_start|>assistant\n<think></think>");
    int promptTokens = (int)(head.size() + tail.size());
    int inputTokens = (int)tokenize(r->vocab, raw).size();
    int cap = (int)(1.5 * inputTokens) + 32;
    llama_sampler_reset(r->sampler);

    AffinityGuard guard;  // pauses the pools and restores this thread's affinity on return
    guard.pause = {r->pool.tp, r->poolBatch.tp};
    long long cpu0 = cpuMsNow();
    std::vector<Pool> retiredBefore;
    retiredBefore.swap(r->retired);
    uint64_t tokHash = 1469598103934665603ULL;  // FNV-1a over the generated token ids
    auto t0 = std::chrono::steady_clock::now();
    // Deviation from Refiner.swift: LFM is a hybrid (conv + attention) model, and llama.cpp's recurrent memory
    // refuses a partial llama_memory_seq_rm once generation has moved past the last ubatch (it returns false and
    // leaves the cache untouched), so the Swift code's seq_rm silently keeps the previous dictation in context.
    // Here the state right after the head is snapshotted and restored instead (one per mode, see Refiner::heads).
    bool headCached = false;
    bool ok = loadHead(r, head, headCached);
    if (ok) ok = llama_decode(r->ctx, llama_batch_get_one(tail.data(), (int32_t)tail.size())) == 0;
    long long prefillMs = msSince(t0);
    // a graph has now run with the current pools, so the backend no longer points at pools retired before this call
    for (auto &p : retiredBefore) {
        if (ok) freePool(p);
        else r->retired.push_back(p);
    }

    auto t1 = std::chrono::steady_clock::now();
    std::string out;
    int gen = 0, slowRun = 0, fallbacks = 0;
    bool switched = false;
    if (r->shortlist) {  // the dictation's own token ids join the shortlist for this dictation (as on the iPhone)
        std::vector<llama_token> ids = tokenize(r->vocab, raw);
        llama_phonon_sl_set_extras(r->ctx, ids.data(), (int32_t)ids.size());
    }
    while (ok && gen < cap) {
        llama_token tok;
        if (r->shortlist) {
            int32_t fb = 0;
            tok = llama_phonon_sl_sample(r->ctx, kSlThreshold, 0, &fb, nullptr, nullptr);
            fallbacks += fb;
        } else {
            tok = llama_sampler_sample(r->sampler, r->ctx, -1);
        }
        if (llama_vocab_is_eog(r->vocab, tok)) break;
        out += piece(r->vocab, tok);
        tokHash = (tokHash ^ (uint32_t)tok) * 1099511628211ULL;
        gen++;
        auto ts = std::chrono::steady_clock::now();
        ok = llama_decode(r->ctx, llama_batch_get_one(&tok, 1)) == 0;
        if (r->guardOn && r->pool.tp) {
            double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
            if (r->refTokMs > 0 && d > 4 * r->refTokMs && d > r->refTokMs + 150) {
                if (++slowRun >= 2) {
                    if (switchDecodePool(r, d)) switched = true;
                    slowRun = 0;
                }
            } else {
                slowRun = 0;
                if (gen > 1 && (r->refTokMs <= 0 || d < r->refTokMs)) r->refTokMs = d;  // fastest token seen
            }
        }
    }
    guard.pause = {r->pool.tp, r->poolBatch.tp};
    long long decodeMs = msSince(t1);
    long long cpuMs = cpuMsNow() - cpu0;

    std::string text = trim(out);
    int rawWords = wordCount(raw);
    bool fired = !ok || text.empty() || (double)wordCount(text) > 1.2 * rawWords + 20;
    if (fired) text = raw;

    char nums[512];
    snprintf(nums, sizeof nums,
             "\"prompt_tokens\":%d,\"prefill_tokens\":%d,\"gen_tokens\":%d,\"prefill_ms\":%lld,\"decode_ms\":%lld,"
             "\"head_cached\":%s,\"guard\":%s,\"k\":%d,\"kb\":%d,\"switched\":%s,\"cpu_ms\":%lld,\"tok_hash\":\"%016llx\","
             "\"heads\":%d,\"head_bytes\":%zu,\"vocab\":%s,\"shortlist\":%s,\"fallbacks\":%d",
             promptTokens, (int)(headCached ? tail.size() : head.size() + tail.size()), gen, prefillMs, decodeMs,
             headCached ? "true" : "false", fired ? "true" : "false", r->pool.k, r->poolBatch.k,
             switched ? "true" : "false", cpuMs, (unsigned long long)tokHash, (int)r->heads.size(), headBytes(r),
             vocabLine.empty() ? "false" : "true", r->shortlist ? "true" : "false", fallbacks);
    std::string json = "{\"text\":\"" + jsonEscape(text) + "\",\"raw\":\"" + jsonEscape(raw) + "\",\"mode\":\"" +
                       jsonEscape(mode) + "\"," + nums + "}";
    // bytes, not NewStringUTF: JNI modified UTF-8 mangles 4-byte characters
    jbyteArray arr = env->NewByteArray((jsize)json.size());
    env->SetByteArrayRegion(arr, 0, (jsize)json.size(), (const jbyte *)json.data());
    return arr;
}

// Prefills and caches the prompt head of `mode` (with `vocab`, usually empty) without generating anything, so the
// first dictation in that mode starts from the cached head. Called by Refiner.warmUp for every writing mode after
// the keyboard loads the model. Returns the ms it took (0 when the head was already cached, -1 on failure).
extern "C" JNIEXPORT jlong JNICALL
Java_ai_phonon_keyboard_Refiner_nativePrime(JNIEnv *env, jclass, jlong h, jstring jmode, jstring jvocab) {
    auto *r = (Refiner *)h;
    std::string mode = jstr(env, jmode), vocabLine = jstr(env, jvocab);
    std::vector<llama_token> head = tokenize(r->vocab, headText(mode, vocabLine));
    AffinityGuard guard;
    guard.pause = {r->pool.tp, r->poolBatch.tp};
    std::vector<Pool> retiredBefore;
    retiredBefore.swap(r->retired);
    auto t0 = std::chrono::steady_clock::now();
    bool cached = false;
    bool ok = loadHead(r, head, cached);
    long long ms = msSince(t0);
    for (auto &p : retiredBefore) {
        if (ok && !cached) freePool(p);  // a graph ran with the current pools
        else r->retired.push_back(p);
    }
    llama_memory_clear(llama_get_memory(r->ctx), true);
    LOGP("PRIME mode=%s tokens=%d cached=%d ok=%d ms=%lld heads=%d head_bytes=%zu", mode.c_str(), (int)head.size(),
         cached ? 1 : 0, ok ? 1 : 0, ms, (int)r->heads.size(), headBytes(r));
    return ok ? (cached ? 0 : (jlong)std::max(1LL, ms)) : -1;
}
