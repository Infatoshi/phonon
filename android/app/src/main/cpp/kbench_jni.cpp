// Kernel benchmark for the refiner on phones (androidTest/RefinerKernelBenchTest.kt only; the keyboard never calls it).
// Built as its own library, libphonon_kbench.so, so the product path in refiner_jni.cpp is untouched.
//  - memBw: streaming-read bandwidth (NEON loads, threads pinned to given CPUs)
//  - peak: int8 SDOT / SMMLA, fp32 FMA and fp16 FMLA throughput on one or more pinned cores
//  - run: prefill and decode timing of the real refiner prompt under a thread configuration: "disp" = no threadpool
//    (what the keyboard does: ggml spawns and joins its worker threads on every graph, i.e. every token), "tp" = a
//    persistent ggml threadpool attached to the context, optionally pinned (cpumask + strict placement)
//  - profile: per-op wall time from llama's eval callback (each graph node is computed on its own, so the numbers
//    carry one threadpool kick + barrier per node; the totals are compared against an unprofiled run)
#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>
#include <link.h>
#include <pthread.h>
#include <sched.h>
#include <sys/auxv.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#include <asm/hwcap.h>
#endif

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "Phonon", __VA_ARGS__)

namespace {

using clk = std::chrono::steady_clock;
double usSince(clk::time_point t) { return std::chrono::duration<double, std::micro>(clk::now() - t).count(); }

void pin(int cpu) {
    if (cpu < 0) return;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof s, &s);
}

std::vector<int> ints(JNIEnv *env, jintArray a) {
    std::vector<int> v;
    if (!a) return v;
    jsize n = env->GetArrayLength(a);
    v.resize(n);
    env->GetIntArrayRegion(a, 0, n, v.data());
    return v;
}

jstring str(JNIEnv *env, const std::string &s) { return env->NewStringUTF(s.c_str()); }

// ---------- memory bandwidth ----------

uint64_t readSlice(const uint8_t *p, size_t n) {
#if defined(__aarch64__)
    uint64x2_t a0 = vdupq_n_u64(0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (size_t i = 0; i + 128 <= n; i += 128) {
        uint64x2x4_t x = vld1q_u64_x4((const uint64_t *)(p + i));
        uint64x2x4_t y = vld1q_u64_x4((const uint64_t *)(p + i + 64));
        a0 = veorq_u64(a0, x.val[0]); a1 = veorq_u64(a1, x.val[1]);
        a2 = veorq_u64(a2, x.val[2]); a3 = veorq_u64(a3, x.val[3]);
        a4 = veorq_u64(a4, y.val[0]); a5 = veorq_u64(a5, y.val[1]);
        a6 = veorq_u64(a6, y.val[2]); a7 = veorq_u64(a7, y.val[3]);
    }
    uint64x2_t r = veorq_u64(veorq_u64(veorq_u64(a0, a1), veorq_u64(a2, a3)),
                             veorq_u64(veorq_u64(a4, a5), veorq_u64(a6, a7)));
    return vgetq_lane_u64(r, 0) ^ vgetq_lane_u64(r, 1);
#else
    uint64_t r = 0;
    for (size_t i = 0; i + 8 <= n; i += 8) r ^= *(const uint64_t *)(p + i);
    return r;
#endif
}

std::atomic<uint64_t> g_sink{0};

// ---------- compute peak ----------
// kAcc independent accumulator chains per thread. 12 (the first version, used for the kb-20261106-1543 tables) can be
// latency-bound on cores with 4 vector pipes (4 pipes x 3-4 cycles latency needs 12-16 chains in flight); 24 is not,
// and still fits the 32 vector registers with the two operands.
constexpr int kAcc = 24;
#if defined(__aarch64__)
__attribute__((target("dotprod"))) double sdotLoop(long iters) {
    int8x16_t a = vdupq_n_s8((int8_t)(g_sink.load() | 1)), b = vdupq_n_s8(3);
    int32x4_t c[kAcc];
    for (auto &x : c) x = vdupq_n_s32(0);
    for (long i = 0; i < iters; i++) {
        for (int k = 0; k < kAcc; k++) c[k] = vdotq_s32(c[k], a, b);
        __asm__ volatile("" : "+w"(a), "+w"(b));
    }
    int32x4_t s = c[0];
    for (int k = 1; k < kAcc; k++) s = vaddq_s32(s, c[k]);
    g_sink += vgetq_lane_s32(s, 0);
    return (double)iters * kAcc * 32;  // 16 int8 MACs per SDOT = 32 ops
}
__attribute__((target("i8mm"))) double smmlaLoop(long iters) {
    int8x16_t a = vdupq_n_s8((int8_t)(g_sink.load() | 1)), b = vdupq_n_s8(3);
    int32x4_t c[kAcc];
    for (auto &x : c) x = vdupq_n_s32(0);
    for (long i = 0; i < iters; i++) {
        for (int k = 0; k < kAcc; k++) c[k] = vmmlaq_s32(c[k], a, b);
        __asm__ volatile("" : "+w"(a), "+w"(b));
    }
    int32x4_t s = c[0];
    for (int k = 1; k < kAcc; k++) s = vaddq_s32(s, c[k]);
    g_sink += vgetq_lane_s32(s, 0);
    return (double)iters * kAcc * 64;  // 32 int8 MACs per SMMLA = 64 ops
}
double fmaLoop(long iters) {
    float32x4_t a = vdupq_n_f32(1.0001f + (float)(g_sink.load() & 1)), b = vdupq_n_f32(0.9999f);
    float32x4_t c[kAcc];
    for (auto &x : c) x = vdupq_n_f32(0);
    for (long i = 0; i < iters; i++) {
        for (int k = 0; k < kAcc; k++) c[k] = vfmaq_f32(c[k], a, b);
        __asm__ volatile("" : "+w"(a), "+w"(b));
    }
    float32x4_t s = c[0];
    for (int k = 1; k < kAcc; k++) s = vaddq_f32(s, c[k]);
    g_sink += (uint64_t)vgetq_lane_f32(s, 0);
    return (double)iters * kAcc * 8;  // 4 lanes x 2 flops
}
__attribute__((target("fullfp16"))) double fp16Loop(long iters) {
    float16x8_t a = vdupq_n_f16((float16_t)(1.0f + (float)(g_sink.load() & 1) * 0.001f)), b = vdupq_n_f16((float16_t)0.5f);
    float16x8_t c[kAcc];
    for (auto &x : c) x = vdupq_n_f16((float16_t)0.0f);
    for (long i = 0; i < iters; i++) {
        for (int k = 0; k < kAcc; k++) c[k] = vfmaq_f16(c[k], a, b);
        __asm__ volatile("" : "+w"(a), "+w"(b));
    }
    float16x8_t s = c[0];
    for (int k = 1; k < kAcc; k++) s = vaddq_f16(s, c[k]);
    g_sink += (uint64_t)(float)vgetq_lane_f16(s, 0);
    return (double)iters * kAcc * 16;  // 8 lanes x 2 flops
}
bool hasFp16() { return getauxval(AT_HWCAP) & HWCAP_ASIMDHP; }
bool hasDot() { return getauxval(AT_HWCAP) & HWCAP_ASIMDDP; }
bool hasI8mm() {
#ifdef HWCAP2_I8MM
    return getauxval(AT_HWCAP2) & HWCAP2_I8MM;
#else
    return false;
#endif
}
#endif

// ---------- backend + model ----------

std::string g_dir;
std::string g_variant = "none";
ggml_backend_reg_t g_cpu_reg = nullptr;

std::string mappedVariant() {
    std::string v = "none";
    dl_iterate_phdr(
        [](dl_phdr_info *i, size_t, void *out) {
            const char *name = i->dlpi_name ? std::strstr(i->dlpi_name, "libggml-cpu") : nullptr;
            if (name) *static_cast<std::string *>(out) = name;
            return 0;
        },
        &v);
    return v;
}

struct Model {
    llama_model *model = nullptr;
    const llama_vocab *vocab = nullptr;
};

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

// ---------- per-op profile ----------

struct OpStat {
    double us = 0;
    long n = 0;
    double bytes = 0;  // weight bytes read (MUL_MAT src0)
    double ops = 0;    // 2 * M * N * K for MUL_MAT
};

struct Prof {
    bool on = false;
    clk::time_point t;
    std::map<std::string, OpStat> stats;
    double total_us = 0;
};

std::string opKey(const ggml_tensor *t) {
    if (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) {
        std::string w = t->src[0] && t->src[0]->name[0] ? t->src[0]->name : "?";
        if (w.rfind("blk.", 0) == 0) w = w.substr(w.find('.', 4) + 1);  // blk.N.ffn_up.weight -> ffn_up.weight
        if (w.size() > 7 && w.compare(w.size() - 7, 7, ".weight") == 0) w.resize(w.size() - 7);
        return std::string("mm:") + w + ":" + ggml_type_name(t->src[0]->type);
    }
    return std::string("op:") + ggml_op_desc(t);
}

bool evalCb(ggml_tensor *t, bool ask, void *ud) {
    auto *p = static_cast<Prof *>(ud);
    if (!p->on) return false;  // compute everything in one go when not profiling
    if (ask) {
        p->t = clk::now();
        return true;
    }
    double us = usSince(p->t);
    OpStat &s = p->stats[opKey(t)];
    s.us += us;
    s.n++;
    p->total_us += us;
    if ((t->op == GGML_OP_MUL_MAT) && t->src[0] && t->src[1]) {
        s.bytes += (double)ggml_nbytes(t->src[0]);
        s.ops += 2.0 * (double)t->src[0]->ne[0] * (double)t->src[0]->ne[1] * (double)t->src[1]->ne[1];
    }
    return true;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

const char *kSystemPrompt =
    "Rewrite the raw dictation transcript into the exact text the speaker intended. Fix recognition errors "
    "and technical terms, keep the speaker's wording and casing style, do not add or drop content.";
const char *kRaw =
    "hi sarah just wanted to let you know the the deploy is gonna slip to thursday because the tests are failing on "
    "the arm runners";

}  // namespace

extern "C" {

// variant "" = ggml's best pick (as the keyboard does); otherwise e.g. "android_armv8.6_1". Switching variants
// unloads the previous CPU backend module first (no llama objects may be alive).
JNIEXPORT jstring JNICALL Java_ai_phonon_keyboard_KernelBench_nativeInit(JNIEnv *env, jclass, jstring jvariant) {
    const char *vc = env->GetStringUTFChars(jvariant, nullptr);
    std::string want(vc);
    env->ReleaseStringUTFChars(jvariant, vc);
    if (g_dir.empty()) {
        Dl_info info{};
        if (dladdr(reinterpret_cast<void *>(&Java_ai_phonon_keyboard_KernelBench_nativeInit), &info) && info.dli_fname) {
            g_dir = info.dli_fname;
            g_dir = g_dir.substr(0, g_dir.find_last_of('/'));
        }
    }
    if (g_cpu_reg) {
        ggml_backend_unload(g_cpu_reg);
        g_cpu_reg = nullptr;
    }
    if (want.empty()) {
        ggml_backend_load_all_from_path(g_dir.c_str());
        g_variant = mappedVariant();
    } else {
        std::string p = g_dir + "/libggml-cpu-" + want + ".so";
        if (ggml_backend_load(p.c_str())) g_variant = "libggml-cpu-" + want + ".so";
        else g_variant = "load-failed:" + want;
    }
    g_cpu_reg = ggml_backend_reg_by_name("CPU");
    llama_backend_init();
    std::string feats;
    if (g_cpu_reg) {
        auto get = (ggml_backend_get_features_t)ggml_backend_reg_get_proc_address(g_cpu_reg, "ggml_backend_get_features");
        if (get)
            for (ggml_backend_feature *f = get(g_cpu_reg); f && f->name; f++) feats += std::string(f->name) + "=" + f->value + " ";
    }
    return str(env, "variant=" + g_variant + " cpu_reg=" + (g_cpu_reg ? "yes" : "no") + " features=" + feats);
}

// Streaming read of `mb` MB split over threads (thread i pinned to cpus[i % n], or unpinned if cpus is empty).
// Returns "best_gbps median_gbps".
JNIEXPORT jstring JNICALL Java_ai_phonon_keyboard_KernelBench_nativeMemBw(JNIEnv *env, jclass, jint mb, jint nthreads,
                                                                           jintArray jcpus, jint reps) {
    std::vector<int> cpus = ints(env, jcpus);
    size_t bytes = (size_t)mb << 20;
    uint8_t *buf = nullptr;
    if (posix_memalign((void **)&buf, 4096, bytes) != 0) return str(env, "alloc-failed");
    for (size_t i = 0; i < bytes; i += 64) buf[i] = (uint8_t)i;  // touch every line so pages are resident
    size_t slice = (bytes / nthreads) & ~(size_t)127;
    // Each pass is timed by the workers themselves (first start to last end), not by this thread: when every core
    // has a pinned worker, a spinning coordinator only gets the CPU on a scheduler tick, which quantized the first
    // version's pass times to multiples of ~4 ms (e.g. 8.39 GB/s = 256 MB in 32.0 ms on several phones). Workers
    // yield while idle and the coordinator sleeps between polls.
    std::atomic<int> gen{0}, done{0};
    std::atomic<bool> quit{false};
    std::vector<double> t0us(nthreads), t1us(nthreads);
    const auto epoch = clk::now();
    std::vector<std::thread> th;
    for (int i = 0; i < nthreads; i++) {
        th.emplace_back([&, i] {
            pin(cpus.empty() ? -1 : cpus[i % cpus.size()]);
            int seen = 0;
            uint64_t acc = 0;
            while (true) {
                while (gen.load(std::memory_order_acquire) == seen && !quit.load()) sched_yield();
                if (quit.load()) break;
                seen++;
                t0us[i] = usSince(epoch);
                acc ^= readSlice(buf + (size_t)i * slice, slice);
                t1us[i] = usSince(epoch);
                done.fetch_add(1, std::memory_order_acq_rel);
            }
            g_sink ^= acc;
        });
    }
    auto pass = [&] {
        done.store(0);
        gen.fetch_add(1, std::memory_order_acq_rel);
        while (done.load(std::memory_order_acquire) < nthreads) std::this_thread::sleep_for(std::chrono::microseconds(50));
        double a = *std::min_element(t0us.begin(), t0us.end()), b = *std::max_element(t1us.begin(), t1us.end());
        return (double)slice * nthreads / ((b - a) * 1e3);
    };
    std::vector<double> gbps;
    // untimed warm-up passes for about 300 ms so the governor has raised the CPU and memory clocks
    for (auto w = clk::now(); usSince(w) < 300000.0;) pass();
    for (int r = 0; r < reps; r++) gbps.push_back(pass());
    quit.store(true);
    for (auto &x : th) x.join();
    free(buf);
    char out[96];
    snprintf(out, sizeof out, "%.2f %.2f", *std::max_element(gbps.begin(), gbps.end()), median(gbps));
    return str(env, out);
}

// Throughput of one instruction kind ("sdot", "smmla", "fma") with one thread pinned to each cpu in cpus, for
// about `ms` ms. Returns "GOPS" summed over threads, or "unsupported".
JNIEXPORT jstring JNICALL Java_ai_phonon_keyboard_KernelBench_nativePeak(JNIEnv *env, jclass, jstring jkind,
                                                                          jintArray jcpus, jint ms) {
#if defined(__aarch64__)
    const char *kc = env->GetStringUTFChars(jkind, nullptr);
    std::string kind(kc);
    env->ReleaseStringUTFChars(jkind, kc);
    if ((kind == "sdot" && !hasDot()) || (kind == "smmla" && !hasI8mm()) || (kind == "fp16" && !hasFp16()))
        return str(env, "unsupported");
    std::vector<int> cpus = ints(env, jcpus);
    std::vector<double> gops(cpus.size());
    std::vector<std::thread> th;
    for (size_t i = 0; i < cpus.size(); i++) {
        th.emplace_back([&, i] {
            pin(cpus[i]);
            auto run = [&](long it) {
                return kind == "sdot" ? sdotLoop(it) : kind == "smmla" ? smmlaLoop(it) : kind == "fp16" ? fp16Loop(it) : fmaLoop(it);
            };
            run(200000);  // warm up and let the governor raise the clock
            long it = 1000000;
            auto t = clk::now();
            double ops = 0;
            while (usSince(t) < ms * 1000.0) ops += run(it);
            gops[i] = ops / (usSince(t) * 1e3);
        });
    }
    for (auto &x : th) x.join();
    double sum = 0;
    for (double g : gops) sum += g;
    char out[64];
    snprintf(out, sizeof out, "%.1f", sum);
    return str(env, out);
#else
    return str(env, "unsupported");
#endif
}

JNIEXPORT jlong JNICALL Java_ai_phonon_keyboard_KernelBench_nativeLoad(JNIEnv *env, jclass, jstring jpath, jboolean repack) {
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = repack;
    llama_model *m = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!m) return 0;
    auto *h = new Model();
    h->model = m;
    h->vocab = llama_model_get_vocab(m);
    return (jlong)h;
}

JNIEXPORT void JNICALL Java_ai_phonon_keyboard_KernelBench_nativeFree(JNIEnv *, jclass, jlong h) {
    auto *m = (Model *)h;
    if (!m) return;
    llama_model_free(m->model);
    delete m;
}

// One thread configuration on the real "formal" prompt. mode "disp" (no threadpool, as the keyboard) or "tp"
// (persistent threadpool; pinned to cpus with strict placement when cpus is non-empty). Measures, after a 1-token
// warm-up: the full prompt as one batch, the user tail after the head (the keyboard's cached path), and `ndecode`
// greedy decode steps (EOG ignored). Repeated `reps` times; reports the median. With profile, one more decode pass of
// `ndecode` tokens and one tail prefill run with the eval callback and per-op times are appended.
JNIEXPORT jstring JNICALL Java_ai_phonon_keyboard_KernelBench_nativeRun(JNIEnv *env, jclass, jlong h, jstring jmode,
                                                                         jint nthreads, jintArray jcpus, jint poll,
                                                                         jint ubatch, jint ndecode, jint reps,
                                                                         jboolean profile) {
    auto *m = (Model *)h;
    const char *mc = env->GetStringUTFChars(jmode, nullptr);
    std::string mode(mc);
    env->ReleaseStringUTFChars(jmode, mc);
    std::vector<int> cpus = ints(env, jcpus);

    Prof prof;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 512;
    cp.n_ubatch = ubatch > 0 ? ubatch : 512;
    cp.n_threads = nthreads;
    cp.n_threads_batch = nthreads;
    if (profile) {
        cp.cb_eval = evalCb;
        cp.cb_eval_user_data = &prof;
    }
    llama_context *ctx = llama_init_from_model(m->model, cp);
    if (!ctx) return str(env, "ctx-failed");

    ggml_threadpool_t tp = nullptr;
    void (*tpFree)(ggml_threadpool_t) = nullptr;
    if (mode == "tp" && g_cpu_reg) {
        auto tpNew = (ggml_threadpool_t(*)(ggml_threadpool_params *))ggml_backend_reg_get_proc_address(g_cpu_reg, "ggml_threadpool_new");
        tpFree = (void (*)(ggml_threadpool_t))ggml_backend_reg_get_proc_address(g_cpu_reg, "ggml_threadpool_free");
        if (tpNew) {
            ggml_threadpool_params p = ggml_threadpool_params_default(nthreads);
            p.poll = (uint32_t)poll;
            if (!cpus.empty()) {
                for (int c : cpus)
                    if (c >= 0 && c < GGML_MAX_N_THREADS) p.cpumask[c] = true;
                p.strict_cpu = true;
            }
            tp = tpNew(&p);
            if (tp) llama_attach_threadpool(ctx, tp, tp);
        }
        if (!tp) {
            llama_free(ctx);
            return str(env, "threadpool-failed");
        }
    }

    std::string system = std::string(kSystemPrompt) + "\nWriting mode: formal (an email or document).";
    std::vector<llama_token> head =
        tokenize(m->vocab, "<|startoftext|><|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n");
    std::vector<llama_token> tail =
        tokenize(m->vocab, std::string(kRaw) + "<|im_end|>\n<|im_start|>assistant\n<think></think>");
    std::vector<llama_token> full = head;
    full.insert(full.end(), tail.begin(), tail.end());
    llama_memory_t mem = llama_get_memory(ctx);
    llama_sampler *smp = llama_sampler_init_greedy();
    bool ok = true;

    auto decodeBatch = [&](std::vector<llama_token> &t) {
        return llama_decode(ctx, llama_batch_get_one(t.data(), (int32_t)t.size())) == 0;
    };
    // warm-up: one token touches every weight once (pages in the mmap when repack is off)
    llama_memory_clear(mem, true);
    {
        std::vector<llama_token> one(head.begin(), head.begin() + 1);
        ok = decodeBatch(one);
    }

    std::vector<double> fullMs, tailMs, tokMs;
    auto decodeSteps = [&](int n, std::vector<double> *per) {
        for (int i = 0; ok && i < n; i++) {
            llama_token tok = llama_sampler_sample(smp, ctx, -1);
            auto t = clk::now();
            ok = llama_decode(ctx, llama_batch_get_one(&tok, 1)) == 0;
            if (per) per->push_back(usSince(t) / 1000.0);
        }
    };
    for (int r = 0; ok && r < reps; r++) {
        llama_memory_clear(mem, true);
        auto t = clk::now();
        ok = decodeBatch(full);
        fullMs.push_back(usSince(t) / 1000.0);
        llama_memory_clear(mem, true);
        ok = ok && decodeBatch(head);
        t = clk::now();
        ok = ok && decodeBatch(tail);
        tailMs.push_back(usSince(t) / 1000.0);
        llama_sampler_reset(smp);
        decodeSteps(ndecode, &tokMs);
    }

    char buf[512];
    double tok = median(tokMs);
    snprintf(buf, sizeof buf,
             "ok=%d head=%zu tail=%zu full_ms=%.1f tail_ms=%.1f tok_ms=%.2f tok_ms_min=%.2f tok_s=%.2f "
             "full_tok_s=%.1f tail_tok_s=%.1f n_tok=%zu",
             ok ? 1 : 0, head.size(), tail.size(), median(fullMs), median(tailMs), tok,
             tokMs.empty() ? 0.0 : *std::min_element(tokMs.begin(), tokMs.end()), tok > 0 ? 1000.0 / tok : 0.0,
             full.size() * 1000.0 / std::max(1e-9, median(fullMs)), tail.size() * 1000.0 / std::max(1e-9, median(tailMs)),
             tokMs.size());
    std::string out = buf;

    if (profile && ok) {
        auto dump = [&](const char *phase, double wallMs, int units) {
            std::vector<std::pair<std::string, OpStat>> v(prof.stats.begin(), prof.stats.end());
            std::sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.second.us > b.second.us; });
            char line[256];
            snprintf(line, sizeof line, " |%s wall_ms_per=%.2f ops_ms_per=%.2f", phase, wallMs / units,
                     prof.total_us / 1000.0 / units);
            out += line;
            for (auto &kv : v) {
                const OpStat &s = kv.second;
                if (s.us / 1000.0 / units < 0.02) continue;
                double gbs = s.bytes > 0 ? s.bytes / (s.us * 1e3) : 0, gops = s.ops > 0 ? s.ops / (s.us * 1e3) : 0;
                snprintf(line, sizeof line, " ;%s ms=%.3f n=%ld gbs=%.1f gops=%.0f", kv.first.c_str(),
                         s.us / 1000.0 / units, s.n / units, gbs, gops);
                out += line;
            }
        };
        // decode
        llama_memory_clear(mem, true);
        ok = decodeBatch(full);
        llama_sampler_reset(smp);
        prof.on = true;
        auto t = clk::now();
        decodeSteps(ndecode, nullptr);
        double wall = usSince(t) / 1000.0;
        prof.on = false;
        dump("PROF_DECODE", wall, ndecode);
        // tail prefill
        prof.stats.clear();
        prof.total_us = 0;
        llama_memory_clear(mem, true);
        ok = ok && decodeBatch(head);
        prof.on = true;
        t = clk::now();
        ok = ok && decodeBatch(tail);
        wall = usSince(t) / 1000.0;
        prof.on = false;
        dump("PROF_TAIL", wall, 1);
    }

    llama_sampler_free(smp);
    if (tp) llama_detach_threadpool(ctx);
    llama_free(ctx);
    if (tp && tpFree) tpFree(tp);
    return str(env, out);
}

}  // extern "C"
