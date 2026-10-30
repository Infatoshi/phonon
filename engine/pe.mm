// pe: Phonon refiner engine. Batch 1, greedy, LFM3-1.7B (Q4_K / Q5_K GGUF). Fixed shapes: the target runs on a Metal
// megakernel (mk.metal) compiled at startup for NB = 1 (decode) and NB = 8 (verify: last token + 7 drafted); prompts
// are prefilled by llama.cpp (512-token chunks, flash attention) and the KV / conv state is handed to the megakernel;
// the refiner's fixed system-prompt prefix can be cached. Context cap 2048 tokens (>= 99.9% of real requests); a
// request that reaches it continues on the llama.cpp path. Drafters: DFlash2 natively (df.metal, dflash2n; 1-4
// layers, any draft vocab) or DFlash2 / DSpark through libllama (dflash2, dspark), draft length 7, fed with the
// target's layer inputs. llama.cpp's prefill mat-muls use kernel_mul_mm_ph (64-token tiles) where the device tune
// says so (M1).
//
//   pe TARGET.gguf serve [--draft ...]         JSON lines on stdin / stdout: one-shot {"prefix"} / {"prompt"} requests
//                                              and per-dictation sessions (open / rollback / forward ...; see serve)
//   pe TARGET.gguf sessbench                   session forward cost at cached offsets, rollback probes
//   pe TARGET.gguf bench [--draft ...]         decode, verify-8, (speculative step,) prefill 175/350
//   pe TARGET.gguf run PROMPTS.jsonl [opts]    {"prompt": str[, "max_tokens": n]} per line (default 256); greedy;
//                                              prints wall p50/p90/mean, acceptance, per-row split; --out OUT.json
//      --draft dflash2n|dflash2|dspark DRAFT.gguf   speculative decoding
//      --prefix FILE                          cache the system-prompt prefix in FILE (exact text the prompts start with)
//      --n N  first N rows;  --warm W  rows excluded from stats (default 5)
//   pe TARGET.gguf reqbench PAIRS.tsv [--every N] [--tpf X]   synthetic requests of given (prompt, output) token counts
//   pe TARGET.gguf gen "PROMPT"                one prompt (already chat-formatted) -> greedy text on stdout
//   pe TARGET.gguf logits PROMPTS.jsonl N      logit probe: megakernel NB=1 / NB=8 vs llama.cpp decode / batch-8
//   pe TARGET.gguf prof                        stack GPU time NB=1 / NB=8 (PE_ONLY=kind: one phase kind x10)
// env: PE_DIR (mk.metal / df.metal; default next to the binary), PE_NSG, PE_TG, PE_TG8, PE_TN, PE_SPLIT, PE_MAX_TPT,
//      PE_MAX_TPT8, PE_MM (override the per-device tuning table), PE_DEFS / PE_DEFINE (extra shader #defines), PE_NOFA,
//      PE_VERBOSE, PE_ROWS (run: per-request rows file), PE_SESSION_LLAMA (sessions: every forward through llama.cpp)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gguf.h"
#include "ggml-metal.h"
#include "llama.h"
#include "llama-ext.h"
#include "common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <arm_neon.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

extern double ggml_metal_debug_gpu_seconds;  // ggml-metal (host patch 02): GPU busy time of llama.cpp's command buffers

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// GPU time of a completed command buffer, seconds. GPUEndTime can still read 0 right after a spin-wait sees the
// completed status (seen as negative GPU times in bench), so this returns -1 (unknown) then; callers skip those.
static double cb_gpu_s(id<MTLCommandBuffer> cb) {
    const double t0 = cb.GPUStartTime, t1 = cb.GPUEndTime;
    return t0 > 0 && t1 >= t0 ? t1 - t0 : -1;
}

// ---- must match mk.metal
enum { OP_NONE = 0, OP_MATVEC = 1, OP_SWIGLU = 2, OP_CONV = 3, OP_QKROPE = 4, OP_ATTN = 5 };
enum { PRO_STAGE = 0, PRO_EMBED = 1, PRO_RESID = 2 };
struct Seg { uint64_t w; uint32_t wtype, n_rows, out, pad; };
struct Phase {
    uint32_t op, pro, k, in;
    uint32_t hres, hout, n_seg, ynorm;
    uint64_t w_post, w_pre;
    Seg seg[4];
    uint64_t aux_w[4];
    uint32_t aux[8];
};
struct Params {
    uint32_t n_phases, n_tg, spin_limit, pos;
    float eps;
    uint32_t ctx, n_valid, ring;  // ring: conv history positions per channel (power of 2)
    uint32_t tokens[32];
};
static_assert(sizeof(Seg) == 24 && sizeof(Phase) == 208 && sizeof(Params) == 160, "layout must match mk.metal");

// model shape (LFM3-1.7B) and the fixed engine shapes
static const int H = 2048, FF = 7680, NL = 24, HD = 128, NQH = 16, NKVH = 4;
static const int CTX = 2048;             // context cap: >= 99.9% of real requests (prompt + output)
// conv input history per conv layer and channel (Params.ring positions, a power of 2): a ring of the last 64 for
// one-shot requests (7 MB), every position once a serve-mode session opens, so it can roll back anywhere
// (13 layers x 2048 channels x 2048 positions f32 = 218 MB; Target::full_ring)
static const int RING_SHORT = 64;
static const int ATT_CH = 32, ATT_MAXCH = CTX / ATT_CH;  // mk.metal ATT_CH
static const int SHAPES[] = { 1, 8 };  // rows per launch: decode, verify (1 + 7 drafted)
static const int MAX_NB = 8;
static const int SP_MAX = 3;             // rows <= SP_MAX: ggml's single-row matvec (bit-identical decode)

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

struct Arena {  // float offsets into the activation buffer, NB rows each
    uint32_t hin[NL + 1], hmid[NL], f, o, yconv, act, qraw, kraw, vraw, gate, q, attn, ynorm, part, amax, logits, size;
};

static Arena make_arena(int nb, int n_vocab) {
    Arena a;
    uint32_t off = 0;
    const uint32_t rows = nb <= SP_MAX ? nb : (nb + 7) / 8 * 8;
    auto take = [&](uint32_t n) { const uint32_t o = off; off += (n * rows + 15) / 16 * 16; return o; };
    for (int i = 0; i <= NL; ++i) a.hin[i] = take(H);
    for (int i = 0; i < NL; ++i) a.hmid[i] = take(H);
    a.f = take(H); a.o = take(H); a.yconv = take(H); a.act = take(FF);
    a.qraw = take(H); a.kraw = take(NKVH * HD); a.vraw = take(NKVH * HD); a.gate = take(NQH);
    a.q = take(H); a.attn = take(H); a.ynorm = take(H);
    a.part = take(NQH * ATT_MAXCH * (HD + 2));
    a.amax = take(1);
    a.logits = take(0);
    off += (uint32_t) (((size_t) n_vocab * rows + 15) / 16 * 16);
    a.size = off;
    return a;
}

// ---------------------------------------------------------------------------------------------
// target: the megakernel
struct Target {
    GGUF g;
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> wbuf, abuf, sync, kc, vc, ring;
    int ring_len = RING_SHORT;    // conv history positions per channel (Params.ring)

    // switch the conv history to every position (sessions); the one-shot path keeps working on it. The current
    // history is not carried over: callers rebuild it (a session open writes its prefix's).
    void full_ring() {
        if (ring_len == CTX) return;
        const int nc = (int) std::count(is_attn.begin(), is_attn.end(), false);
        ring = [dev newBufferWithLength:(size_t) nc * H * CTX * 4 options:MTLResourceStorageModeShared];
        ring_len = CTX;
    }
    NSString * src;
    int nsg = 8, n_tg = 0, n_vocab = 0, tn = 2, split = 8;  // tn, split: q4m verify tiles (mk.metal TN, SPLIT)
    int max_tpt = 0, max_tpt8 = 0, tg_fixed = 0, tg8_fixed = 0, n_tg8 = 0;
    uint32_t prefetch = 4096;
    float eps = 1e-5f;
    std::vector<bool> is_attn, is_swa;
    std::vector<int> attn_idx, conv_idx;
    struct Variant {
        id<MTLComputePipelineState> ps, ps_head, ps_argmax;
        id<MTLBuffer> phases, head;
        uint32_t n_phases;
        Arena ar;
        // multi-dispatch experiment: one mk_phase dispatch per phase
        id<MTLBuffer> pbuf_head;  // the head dispatch's Params (its grid size)
        id<MTLBuffer> phases256;  // the phases at 256-byte strides
        id<MTLComputePipelineState> ps_phase;
        std::vector<uint32_t> grids;  // per-phase grid sizes
    };
    bool multi = false;           // one dispatch per phase (indirect command buffer) instead of the persistent kernel
    bool ph_mm = false;           // llama.cpp prefill mat-muls through kernel_mul_mm_ph (per-device tune, PE_MM)
    int sk_rows = 0;              // session llama.cpp forwards of <= sk_rows tokens: kernel_mul_mm_sk (tune, PE_MM_SK)
    int head_tg = 0, head_tg8 = 0;  // LM head grid caps for NB = 1 / 8 (0: one unit per simdgroup; PE_HEAD_TG / _TG8)
    id<MTLBuffer> pbuf;           // Params of the current step (multi-dispatch mode)
    std::map<int, Variant> variants;
    double last_gpu_ms = 0, compile_s = 0;

    uint64_t w(const std::string & name) const { return g.t(name).offset; }
    uint32_t wt(const std::string & name) const { return g.t(name).type; }

    bool init(const char * path, const char * dir) {
        std::string err;
        if (!g.open(path, &err)) { fprintf(stderr, "%s\n", err.c_str()); return false; }
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        wbuf = [dev newBufferWithBytesNoCopy:g.map length:g.map_size options:MTLResourceStorageModeShared deallocator:nil];
        if (!wbuf) { fprintf(stderr, "cannot wrap the model mapping in a Metal buffer\n"); return false; }
        eps = (float) g.num("lfm3.attention.layer_norm_rms_epsilon");
        const auto kvh = g.arr("lfm3.attention.head_count_kv"), swa = g.arr("lfm3.attention.sliding_window_pattern");
        int na = 0, nc = 0;
        for (int il = 0; il < NL; ++il) {
            is_attn.push_back(kvh[il] != 0);
            is_swa.push_back(swa[il] != 0);
            attn_idx.push_back(is_attn[il] ? na++ : -1);
            conv_idx.push_back(is_attn[il] ? -1 : nc++);
        }
        n_vocab = (int) g.t("token_embd.weight").ne[1];
        // per-device tuning, measured ahead of time (env PE_* overrides): simdgroups per threadgroup, register cap
        // (MAX_TPT: lets 2 threadgroups share an M1 core), persistent grid, q4m tiles and split; ph_mm: llama.cpp
        // prefill mat-muls through kernel_mul_mm_ph (64-token tiles; M1 prefill -7%, M4 Max no gain: 42.9 vs 42.1 ms)
        // sk_rows: session forwards of at most this many tokens use kernel_mul_mm_sk (split-k, for wide GPUs)
        // head_tg: NB = 1 LM head grid cap (threadgroups loop over row units; y staged once per threadgroup)
        struct Tune { const char * name; int nsg, max_tpt, tg, tn, split; bool multi, ph_mm; int sk_rows, head_tg; };
        static const Tune tunes[] = {
            { "Apple M1", 8, 512, 16, 4, 4, false, true, 0, 128 },     // decode 18.7 ms (head cap: 19.0), verify-8 30.8 ms
            { "Apple M4 Max", 8, 0, 64, 2, 8, false, false, 64, 0 },  // session forward 32 tokens 17 vs 20 ms
        };
        Tune tu = { "", 8, 0, 0, 2, 8, false, false, 0, 0 };
        for (const Tune & t : tunes) if (strcmp(dev.name.UTF8String, t.name) == 0) tu = t;
        ph_mm = getenv("PE_MM") ? atoi(getenv("PE_MM")) != 0 : tu.ph_mm;
        sk_rows = getenv("PE_MM_SK") ? atoi(getenv("PE_MM_SK")) : tu.sk_rows;
        head_tg = getenv("PE_HEAD_TG") ? atoi(getenv("PE_HEAD_TG")) : tu.head_tg;
        head_tg8 = getenv("PE_HEAD_TG8") ? atoi(getenv("PE_HEAD_TG8")) : 0;
        nsg = getenv("PE_NSG") ? atoi(getenv("PE_NSG")) : tu.nsg;
        max_tpt = getenv("PE_MAX_TPT") ? atoi(getenv("PE_MAX_TPT")) : tu.max_tpt;
        max_tpt8 = getenv("PE_MAX_TPT8") ? atoi(getenv("PE_MAX_TPT8")) : max_tpt;  // the verify variant's register cap
        tg_fixed = getenv("PE_TG") ? atoi(getenv("PE_TG")) : tu.tg;
        tn = getenv("PE_TN") ? atoi(getenv("PE_TN")) : tu.tn;
        split = getenv("PE_SPLIT") ? atoi(getenv("PE_SPLIT")) : std::min(tu.split, nsg);
        for (auto & kv : g.tensors) {  // the verify path (q4m) reads Q4_K and Q5_K matrices only
            const GGUFTensor & t = kv.second;
            if (t.ne.size() == 2 && t.ne[1] > 16 && t.type != 12 && t.type != 13 && t.type != 0) {
                fprintf(stderr, "%s has type %u: this engine is built for Q4_K / Q5_K weights\n", t.name.c_str(), t.type);
                return false;
            }
        }
        sync = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        memset(sync.contents, 0, 4096);
        const size_t kv_bytes = (size_t) na * CTX * NKVH * HD * 2;
        kc = [dev newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];  // the prefill handoff writes them
        vc = [dev newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
        ring = [dev newBufferWithLength:(size_t) nc * H * ring_len * 4 options:MTLResourceStorageModeShared];
        abuf = [dev newBufferWithLength:(size_t) make_arena(MAX_NB, n_vocab).size * 4 options:MTLResourceStorageModeShared];
        src = [NSString stringWithContentsOfFile:[[NSString stringWithUTF8String:dir] stringByAppendingPathComponent:@"mk.metal"]
                                        encoding:NSUTF8StringEncoding error:nil];
        if (!src) { fprintf(stderr, "cannot read mk.metal in %s (set PE_DIR)\n", dir); return false; }
        multi = getenv("PE_MULTI") ? atoi(getenv("PE_MULTI")) != 0 : tu.multi;
        pbuf = [dev newBufferWithLength:sizeof(Params) options:MTLResourceStorageModeShared];
        const double t0 = now();
        for (int nb : SHAPES) variant(nb);  // every shape compiled up front: no compile on the request path
        compile_s = now() - t0;
        reset();
        n_tg = multi ? 1 : tg_fixed > 0 ? tg_fixed : probe_grid();
        n_tg8 = getenv("PE_TG8") ? atoi(getenv("PE_TG8")) : tg8_fixed;
        return n_tg > 0;
    }

    void reset() { memset(ring.contents, 0, ring.length); }  // KV entries are written before they are read

    std::vector<Phase> build_phases(const Arena & a, bool split_norm) {
        std::vector<Phase> ph;
        auto blank = [] { Phase p; memset(&p, 0, sizeof p); return p; };
        auto L = [](int il, const char * n) { return "blk." + std::to_string(il) + "." + n; };
        for (int il = 0; il < NL; ++il) {
            Phase p = blank();
            p.pro = il == 0 ? PRO_EMBED : PRO_RESID;
            p.k = H;
            p.in = a.f;
            p.hres = il == 0 ? 0 : a.hmid[il - 1];
            p.hout = a.hin[il];
            p.ynorm = a.ynorm;
            p.w_post = il == 0 ? 0 : w(L(il - 1, "post_ffw_norm.weight"));
            p.w_pre = w(L(il, "attn_norm.weight"));
            p.aux_w[3] = w("token_embd.weight");
            p.aux[6] = wt("token_embd.weight");
            if (!is_attn[il]) {
                p.op = OP_CONV;
                p.n_seg = 1;
                p.seg[0] = { w(L(il, "shortconv.in_proj.weight")), wt(L(il, "shortconv.in_proj.weight")), 3 * H, a.yconv, 0 };
                p.aux_w[0] = w(L(il, "shortconv.conv.weight"));
                p.aux[0] = conv_idx[il] * H;  // history rows of this layer (x Params.ring positions)
                ph.push_back(p);
                Phase q = blank();
                q.op = OP_MATVEC; q.pro = PRO_STAGE; q.k = H; q.in = a.yconv; q.n_seg = 1;
                q.seg[0] = { w(L(il, "shortconv.out_proj.weight")), wt(L(il, "shortconv.out_proj.weight")), (uint32_t) H, a.o, 0 };
                ph.push_back(q);
            } else {
                p.op = OP_MATVEC;
                p.n_seg = 4;
                p.seg[0] = { w(L(il, "attn_q.weight")), wt(L(il, "attn_q.weight")), (uint32_t) H, a.qraw, 0 };
                p.seg[1] = { w(L(il, "attn_k.weight")), wt(L(il, "attn_k.weight")), (uint32_t) (NKVH * HD), a.kraw, 0 };
                p.seg[2] = { w(L(il, "attn_v.weight")), wt(L(il, "attn_v.weight")), (uint32_t) (NKVH * HD), a.vraw, 0 };
                p.seg[3] = { w(L(il, "attn_gate.weight")), wt(L(il, "attn_gate.weight")), (uint32_t) NQH, a.gate, 0 };
                ph.push_back(p);
                Phase r = blank();
                r.op = OP_QKROPE; r.pro = PRO_STAGE;
                r.aux[0] = a.qraw; r.aux[1] = a.kraw; r.aux[2] = a.vraw; r.aux[3] = a.q; r.aux[4] = attn_idx[il];
                r.aux[5] = fbits(is_swa[il] ? (float) g.num("lfm3.rope.freq_base_swa") : (float) g.num("lfm3.rope.freq_base"));
                r.aux_w[0] = w(L(il, "attn_q_norm.weight"));
                r.aux_w[1] = w(L(il, "attn_k_norm.weight"));
                r.aux_w[2] = is_swa[il] ? 0 : w("rope_freqs.weight");
                ph.push_back(r);
                Phase t = blank();
                t.op = OP_ATTN; t.pro = PRO_STAGE;
                t.aux[0] = a.q; t.aux[1] = a.gate; t.aux[3] = a.attn; t.aux[4] = attn_idx[il];
                t.aux[5] = is_swa[il] ? (uint32_t) g.num("lfm3.attention.sliding_window") : 0;
                t.aux[6] = fbits(1.0f / sqrtf((float) HD));
                t.aux[7] = a.part;
                ph.push_back(t);
                Phase o = blank();
                o.op = OP_MATVEC; o.pro = PRO_STAGE; o.k = H; o.in = a.attn; o.n_seg = 1;
                o.seg[0] = { w(L(il, "attn_output.weight")), wt(L(il, "attn_output.weight")), (uint32_t) H, a.o, 0 };
                ph.push_back(o);
            }
            Phase f = blank();
            f.op = OP_SWIGLU; f.pro = PRO_RESID; f.k = H; f.in = a.o; f.hres = a.hin[il]; f.hout = a.hmid[il];
            f.ynorm = a.ynorm;
            f.w_post = w(L(il, "post_attention_norm.weight"));
            f.w_pre = w(L(il, "ffn_norm.weight"));
            f.n_seg = 2;
            f.seg[0] = { w(L(il, "ffn_gate.weight")), wt(L(il, "ffn_gate.weight")), (uint32_t) FF, a.act, 0 };
            f.seg[1] = { w(L(il, "ffn_up.weight")), wt(L(il, "ffn_up.weight")), (uint32_t) FF, a.act, 0 };
            ph.push_back(f);
            Phase d = blank();
            d.op = OP_MATVEC; d.pro = PRO_STAGE; d.k = FF; d.in = a.act; d.n_seg = 1;
            d.seg[0] = { w(L(il, "ffn_down.weight")), wt(L(il, "ffn_down.weight")), (uint32_t) H, a.f, 0 };
            ph.push_back(d);
        }
        Phase e = blank();
        e.op = OP_NONE; e.pro = PRO_RESID; e.k = H; e.in = a.f; e.hres = a.hmid[NL - 1]; e.hout = a.hin[NL];
        e.ynorm = a.ynorm;
        e.w_post = w(L(NL - 1, "post_ffw_norm.weight"));
        e.w_pre = w("output_norm.weight");
        ph.push_back(e);
        // verify path (8 rows): the RMS-norm prologues become phases of their own, computed by one threadgroup and
        // written once (fused, every threadgroup wrote the same 64 KB of y: 0.67 of a 5.3 ms stack on the M4 Max)
        if (split_norm && !getenv("PE_FUSED_NORM")) {
            std::vector<Phase> sp;
            for (Phase p : ph) {
                if (p.pro != PRO_STAGE && p.op != OP_NONE) {
                    Phase n = p;
                    n.op = OP_NONE;
                    n.n_seg = 0;
                    sp.push_back(n);
                    p.pro = PRO_STAGE;
                    p.in = p.ynorm;
                }
                sp.push_back(p);
            }
            ph = sp;
        }
        // profiling (results are wrong): only one phase kind, repeated 10x. PE_ONLY=attn|qk|conv|down|oproj|ffn|qkv|norm
        if (const char * only = getenv("PE_ONLY")) {
            const std::string s = only;
            std::vector<Phase> kept;
            for (int rep = 0; rep < 10; ++rep) {
                for (Phase p : ph) {
                    const bool keep = (s == "attn" && p.op == OP_ATTN) || (s == "qk" && p.op == OP_QKROPE) ||
                                      (s == "conv" && p.op == OP_CONV) || (s == "down" && p.k == FF) ||
                                      (s == "oproj" && p.op == OP_MATVEC && p.pro == PRO_STAGE && p.k == H) ||
                                      (s == "ffn" && p.op == OP_SWIGLU) || (s == "qkv" && p.op == OP_MATVEC && p.n_seg == 4) ||
                                      (s == "norm" && p.pro != PRO_STAGE);
                    if (s == "norm") p.op = OP_NONE;
                    if (keep) kept.push_back(p);
                }
            }
            if (s == "empty") {  // grid barriers only
                Phase z = blank();
                kept.assign(100, z);
            }
            ph = kept;
        }
        return ph;
    }

    Variant & variant(int nb) {
        auto it = variants.find(nb);
        if (it != variants.end()) return it->second;
        Variant v;
        NSError * err = nil;
        MTLCompileOptions * opt = [MTLCompileOptions new];
        // PE_DEFINE=NAME=VAL,NAME=VAL: extra #defines without spaces (for env strings that are split on spaces)
        std::string extra = getenv("PE_DEFS") ? getenv("PE_DEFS") : "";
        if (const char * d = getenv("PE_DEFINE")) {
            std::stringstream ss(d);
            std::string item;
            while (std::getline(ss, item, ',')) {
                const size_t eq = item.find('=');
                extra += "\n#define " + (eq == std::string::npos ? item : item.substr(0, eq) + " " + item.substr(eq + 1));
            }
        }
        NSString * s = [NSString stringWithFormat:@"#define NSG %d\n#define NB %d\n#define SP_MAX %d\n#define ATT_MAXCH %d\n"
                                                   "#define TN %d\n#define SPLIT %d\n%s\n%s\n%@",
                                                  nsg, nb, SP_MAX, ATT_MAXCH, tn, split,
                                                  (nb == 1 ? max_tpt : max_tpt8)
                                                      ? [NSString stringWithFormat:@"#define MAX_TPT %d", nb == 1 ? max_tpt : max_tpt8].UTF8String : "",
                                                  extra.c_str(), src];
        id<MTLLibrary> lib = [dev newLibraryWithSource:s options:opt error:&err];
        if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); exit(1); }
        v.ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mk_run"] error:&err];
        v.ps_head = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mk_head"] error:&err];
        v.ps_argmax = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mk_argmax"] error:&err];
        if (!v.ps || !v.ps_head || !v.ps_argmax || (int) v.ps.maxTotalThreadsPerThreadgroup < nsg * 32) {
            fprintf(stderr, "pipeline NB=%d: %s\n", nb, v.ps ? "too many threads per threadgroup" : err.localizedDescription.UTF8String);
            exit(1);
        }
        if (getenv("PE_VERBOSE")) {
            fprintf(stderr, "pipeline NB=%d: max threads/threadgroup %d, static threadgroup memory %d B\n", nb,
                    (int) v.ps.maxTotalThreadsPerThreadgroup, (int) v.ps.staticThreadgroupMemoryLength);
        }
        v.ar = make_arena(nb, n_vocab);
        std::vector<Phase> ph = build_phases(v.ar, nb > SP_MAX || multi);  // multi-dispatch: norms are their own phases
        v.n_phases = (uint32_t) ph.size();
        v.phases = [dev newBufferWithBytes:ph.data() length:ph.size() * sizeof(Phase) options:MTLResourceStorageModeShared];
        Phase hd;
        memset(&hd, 0, sizeof hd);
        hd.op = OP_MATVEC; hd.pro = PRO_STAGE; hd.k = H; hd.in = v.ar.ynorm; hd.n_seg = 1;
        hd.seg[0] = { w("token_embd.weight"), wt("token_embd.weight"), (uint32_t) n_vocab, v.ar.logits, 0 };
        hd.aux[0] = v.ar.amax;
        v.head = [dev newBufferWithBytes:&hd length:sizeof hd options:MTLResourceStorageModeShared];
        if (multi) build_icb(v, lib, ph, nb);
        return variants[nb] = v;
    }

    // grid of one phase in multi-dispatch mode: enough threadgroups for one work unit per simdgroup (single-row path)
    // or per unit group (q4m path); one threadgroup for norm-only phases
    uint32_t phase_grid(const Phase & p, int nb) const {
        const bool sp = nb <= SP_MAX;
        const uint32_t per_tg = sp ? (uint32_t) nsg : (uint32_t) (nsg / split);
        auto tgs = [&](uint32_t units) { return std::max<uint32_t>(1, (units + per_tg - 1) / per_tg); };
        const uint32_t rows_u = sp ? 2 : 8 * tn;
        switch (p.op) {
            case OP_NONE: return 1;
            case OP_SWIGLU: return tgs(p.seg[0].n_rows / rows_u);
            case OP_MATVEC: {
                uint32_t u = 0;
                for (uint32_t s = 0; s < p.n_seg; ++s) u += (p.seg[s].n_rows + rows_u - 1) / rows_u;
                return tgs(u);
            }
            case OP_CONV: return tgs(H / rows_u);
            case OP_QKROPE: return std::max<uint32_t>(1, ((uint32_t) nb * (NQH + 2 * NKVH) + nsg - 1) / nsg);
            case OP_ATTN: return 64;  // units are strided over the grid; enough for 8 rows x 16 heads x ~16 chunks
        }
        return 1;
    }

    // multi-dispatch experiment (PE_MULTI=1): per-phase grids, the phases at 256-byte strides (constant-buffer offset
    // alignment). Measured: M4 Max NB=8 stack 5.54 vs 5.85 ms persistent, NB=1 3.52 vs 3.19; M1 NB=8 33.6 vs ~29, NB=1
    // 22.8 vs ~18.9 (dispatch boundary 2.1-3.5 us vs grid barrier 0.9-2.8 us). Off by default. An indirect command
    // buffer version faulted (GPU address fault on the M1) and was dropped.
    void build_icb(Variant & v, id<MTLLibrary> lib, const std::vector<Phase> & ph, int nb) {
        NSError * err = nil;
        v.ps_phase = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mk_phase"] error:&err];
        if (!v.ps_phase) { fprintf(stderr, "mk_phase: %s\n", err.localizedDescription.UTF8String); exit(1); }
        for (const Phase & p : ph) v.grids.push_back(phase_grid(p, nb));
        v.phases256 = [dev newBufferWithLength:ph.size() * 256 options:MTLResourceStorageModeShared];
        for (size_t i = 0; i < ph.size(); ++i) memcpy((uint8_t *) v.phases256.contents + i * 256, &ph[i], sizeof(Phase));
        Params hp;
        memset(&hp, 0, sizeof hp);
        hp.n_tg = nb <= SP_MAX ? (uint32_t) ((n_vocab / 2 + nsg - 1) / nsg)
                               : (uint32_t) ((n_vocab / (8 * tn) + nsg / split - 1) / (nsg / split));
        v.pbuf_head = [dev newBufferWithBytes:&hp length:sizeof hp options:MTLResourceStorageModeShared];
    }

    void encode(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> ps, const Params & p, id<MTLBuffer> phases, uint32_t n_tg) {
        [e setComputePipelineState:ps];
        [e setBytes:&p length:sizeof p atIndex:0];
        [e setBuffer:phases offset:0 atIndex:1];
        [e setBuffer:wbuf offset:0 atIndex:2];
        [e setBuffer:abuf offset:0 atIndex:3];
        [e setBuffer:sync offset:0 atIndex:4];
        [e setBuffer:kc offset:0 atIndex:5];
        [e setBuffer:vc offset:0 atIndex:6];
        [e setBuffer:ring offset:0 atIndex:7];
        [e setBytes:&prefetch length:4 atIndex:8];
        [e dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(nsg * 32, 1, 1)];
    }

    // one forward over nb token rows (n_valid real, the rest padding) at positions pos..; logits of row b at
    // logits(nb) + b * n_vocab. Spins on the command buffer status instead of blocking (lower wake-up latency).
    bool step(const int * tokens, int nb, int n_valid, int pos, bool with_head = true, bool retry = true,
              const std::function<void(id<MTLCommandBuffer>)> & then = nullptr) {
        Variant & v = variant(nb);
        int & grid = nb == 1 || n_tg8 == 0 ? n_tg : n_tg8;  // the verify variant may have its own grid (register cap)
        Params p;
        memset(&p, 0, sizeof p);
        p.n_phases = v.n_phases;
        p.n_tg = (uint32_t) grid;
        p.spin_limit = 1u << 20;
        p.pos = (uint32_t) pos;
        p.eps = eps;
        p.ctx = CTX;
        p.n_valid = (uint32_t) n_valid;
        p.ring = (uint32_t) ring_len;
        for (int b = 0; b < nb; ++b) p.tokens[b] = (uint32_t) (b < n_valid ? tokens[b] : tokens[n_valid - 1]);
        uint32_t * s = (uint32_t *) sync.contents;
        s[0] = s[1] = 0;
        if (multi) {  // experiment: one dispatch per phase (see mk_phase), encoded per step
            memcpy(pbuf.contents, &p, sizeof p);
            id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            const MTLSize tpt = MTLSizeMake(nsg * 32, 1, 1);
            [e setComputePipelineState:v.ps_phase];
            [e setBuffer:pbuf offset:0 atIndex:0];
            [e setBuffer:wbuf offset:0 atIndex:2];
            [e setBuffer:abuf offset:0 atIndex:3];
            [e setBuffer:sync offset:0 atIndex:4];
            [e setBuffer:kc offset:0 atIndex:5];
            [e setBuffer:vc offset:0 atIndex:6];
            [e setBuffer:ring offset:0 atIndex:7];
            for (uint32_t i = 0; i < v.n_phases; ++i) {
                [e setBuffer:v.phases256 offset:i * 256 atIndex:1];
                [e dispatchThreadgroups:MTLSizeMake(v.grids[i], 1, 1) threadsPerThreadgroup:tpt];
            }
            if (with_head) {
                Params ph = p;
                ph.n_phases = 1;
                ph.n_tg = ((const Params *) v.pbuf_head.contents)->n_tg;
                encode(e, v.ps_head, ph, v.head, ph.n_tg);
                [e setComputePipelineState:v.ps_argmax];
                [e dispatchThreadgroups:MTLSizeMake(n_valid, 1, 1) threadsPerThreadgroup:tpt];
            }
            [e endEncoding];
            if (then) then(cb);
            [cb commit];
            while (cb.status < MTLCommandBufferStatusCompleted) {}
            if (cb.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "direct multi-dispatch step failed: %s\n", cb.error.localizedDescription.UTF8String);
                exit(1);
            }
            last_gpu_ms = cb_gpu_s(cb) * 1e3;
            return true;
        }
        id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        encode(e, v.ps, p, v.phases, (uint32_t) grid);
        if (with_head) {
            Params ph = p;
            ph.n_phases = 1;
            if (nb <= SP_MAX) ph.n_tg = (uint32_t) ((n_vocab / 2 + nsg - 1) / nsg);  // RU = 2 rows per simdgroup
            else ph.n_tg = (uint32_t) ((n_vocab / (8 * tn) + nsg / split - 1) / (nsg / split));  // q4m units
            // fewer threadgroups, each looping over many row units (NB = 1: y is staged once per threadgroup)
            const int cap = nb <= SP_MAX ? head_tg : head_tg8;
            if (cap > 0) ph.n_tg = std::min<uint32_t>(ph.n_tg, (uint32_t) cap);
            encode(e, v.ps_head, ph, v.head, ph.n_tg);
            [e setComputePipelineState:v.ps_argmax];  // buffers 1 (head phase) and 3 (arena) are still bound
            [e dispatchThreadgroups:MTLSizeMake(n_valid, 1, 1) threadsPerThreadgroup:MTLSizeMake(nsg * 32, 1, 1)];
        }
        [e endEncoding];
        if (then) then(cb);  // more work in the same command buffer (the drafter's K/V injection of these rows)
        // (Tried: reading the picks as soon as a counter written after them by mk_argmax said they were done, before
        // the completion notice. The counter became visible to the CPU before the picks did: 95/300 outputs wrong.
        // GPU writes are only guaranteed visible to the CPU at command buffer completion.)
        [cb commit];
        while (cb.status < MTLCommandBufferStatusCompleted) {}
        last_gpu_ms = cb_gpu_s(cb) * 1e3;  // negative: unknown
        if (!s[1]) return true;
        memset(sync.contents, 0, sync.length);
        if (retry && grid > 1) {
            fprintf(stderr, "grid barrier timed out at G=%d (NB=%d); retrying with G=%d\n", grid, nb, grid / 2);
            grid /= 2;
            return step(tokens, nb, n_valid, pos, with_head, true, then);
        }
        return false;
    }

    float * acts() { return (float *) abuf.contents; }
    const float * logits(int nb) { return acts() + variant(nb).ar.logits; }
    // greedy token of row b after a step with the head (GPU argmax)
    int pick(int nb, int b) { return (int) ((const uint32_t *) acts())[variant(nb).ar.amax + b]; }
    // layer-il input (residual stream entering layer il) of row b after a step with nb rows
    const float * layer_in(int nb, int il, int b) { return acts() + variant(nb).ar.hin[il] + (size_t) b * H; }

    int probe_grid() {
        int best = 0, tok = 0;
        double best_ms = 1e9;
        for (int G : { 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128 }) {
            n_tg = G;
            double ms = 1e9;
            bool ok = true;
            for (int r = 0; r < 3 && ok; ++r) {
                ok = step(&tok, 1, 1, 0, false, false);
                if (r > 0 && last_gpu_ms >= 0) ms = std::min(ms, last_gpu_ms);
            }
            if (!ok || ms > 3 * best_ms) break;
            if (ms < best_ms * 0.98) { best_ms = ms; best = G; }
        }
        reset();
        if (getenv("PE_VERBOSE")) fprintf(stderr, "grid: %d threadgroups x %d threads (stack %.2f ms)\n", best, nsg * 32, best_ms);
        return best;
    }
};

// first index of the maximum: a vector max, then the first element equal to it
static int argmax(const float * x, int n) {
    float32x4_t m4 = vdupq_n_f32(x[0]);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        m4 = vmaxq_f32(m4, vmaxq_f32(vmaxq_f32(vld1q_f32(x + i), vld1q_f32(x + i + 4)),
                                     vmaxq_f32(vld1q_f32(x + i + 8), vld1q_f32(x + i + 12))));
    }
    float m = vmaxvq_f32(m4);
    for (; i < n; ++i) m = std::max(m, x[i]);
    for (int j = 0; j < n; ++j) if (x[j] == m) return j;
    return 0;
}

// ---------------------------------------------------------------------------------------------
// drafter: DFlash2 (selector lattice) or DSpark (Markov head, anchor-first block), as llama.cpp's
// common_speculative_impl_draft_dflash, greedy. Needs the target loaded in llama.cpp (the drafter shares its
// token embeddings and output head through ctx_other).
struct OpProfile {
    bool active = false;
    double last = 0;
    std::map<std::string, double> ms;
    std::map<std::string, int> n;
    void print(int calls) {
        std::vector<std::pair<double, std::string>> v;
        for (auto & kv : ms) v.push_back({ kv.second, kv.first });
        std::sort(v.rbegin(), v.rend());
        double tot = 0;
        for (auto & e : v) tot += e.first;
        printf("drafter op profile over %d draft calls (ms per call, synchronized per node), total %.2f:\n", calls, tot / calls);
        for (size_t i = 0; i < v.size() && i < 25; ++i) printf("  %7.3f  x%-3d %s\n", v[i].first / calls, n[v[i].second] / calls, v[i].second.c_str());
    }
};

struct Drafter {
    OpProfile prof;
    int n_prof_calls = 0;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    bool dspark = false, dflash2 = false, from_anchor = true;
    int block = 8, top_k = 0, n_max = 7, n_embd = 0;
    llama_token mask = 0;
    std::vector<int> layers;
    common_batch batch, inject;
    std::vector<float> feat;

    bool init(const char * path, bool is_dspark, llama_context * ctx_tgt) {
        dspark = is_dspark;
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        model = llama_model_load_from_file(path, mp);
        if (!model) return false;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = CTX;
        cp.n_batch = cp.n_ubatch = 512;
        cp.n_seq_max = 1;
        cp.n_rs_seq = 0;
        cp.ctx_other = ctx_tgt;
        cp.flash_attn_type = getenv("PE_DFA") ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        if (getenv("PE_DRAFT_PROFILE")) {  // per-op wall time (each node synchronized: overhead included)
            cp.cb_eval = [](ggml_tensor * t, bool ask, void * ud) -> bool {
                auto * P = (OpProfile *) ud;
                if (ask) return true;
                const double t1 = now();
                if (P->active && P->last > 0) {
                    char key[96];
                    snprintf(key, sizeof key, "%s %s [%lld,%lld,%lld]", ggml_op_desc(t), ggml_type_name(t->type), (long long) t->ne[0],
                             (long long) t->ne[1], (long long) t->ne[2]);
                    P->ms[key] += (t1 - P->last) * 1e3;
                    P->n[key] += 1;
                }
                P->last = now();
                return true;
            };
            cp.cb_eval_user_data = &prof;
        }
        ctx = llama_init_from_model(model, cp);
        if (!ctx) return false;
        const int32_t * ids = llama_model_target_layer_ids(model);
        for (uint32_t k = 0; k < llama_model_target_layer_ids_n(model); ++k) layers.push_back(ids[k]);
        char buf[32] = {};
        if (llama_model_meta_val_str(model, "dflash.block_size", buf, sizeof buf) >= 0) block = atoi(buf);
        if (llama_model_meta_val_str(model, "dflash.sample_from_anchor", buf, sizeof buf) >= 0) from_anchor = strcmp(buf, "true") == 0;
        bool causal = false;
        if (llama_model_meta_val_str(model, "dflash.attention.causal", buf, sizeof buf) >= 0) causal = strcmp(buf, "true") == 0;
        top_k = llama_model_dflash_selector_top_k(model);
        dflash2 = top_k > 0;
        mask = llama_vocab_mask(llama_model_get_vocab(model));
        n_embd = llama_model_n_embd(model);
        n_max = std::min(7, dspark && from_anchor ? block : block - 1);
        batch = common_batch(ctx);
        inject = common_batch(ctx);
        llama_set_embeddings_nextn(ctx, true, /*masked*/ !dflash2);
        llama_set_causal_attn(ctx, causal);
        return true;
    }

    void clear() { llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1); }
    void trim(int n) { llama_memory_seq_rm(llama_get_memory(ctx), 0, n, -1); }  // keep positions [0, n)

    // inject target features (layer inputs, NL x H per row, rows at positions pos0..) into the drafter's cache
    void add(const std::vector<float> & rows_feat, int n_rows, int pos0) {
        const size_t n_enc = layers.size() * H;
        llama_memory_seq_rm(llama_get_memory(ctx), 0, pos0, -1);
        for (int off = 0; off < n_rows; off += 512) {
            const int n = std::min(512, n_rows - off);
            inject.clear();
            for (int i = 0; i < n; ++i) {
                const llama_pos p = pos0 + off + i;
                const llama_pos pa[4] = { p, p, p, 0 };
                inject.add_embd({ rows_feat.data() + (size_t) (off + i) * n_enc, 1, n_enc }, pa, 0, false);
            }
            const double t0 = now();
            llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, inject.get());
            t_proc_add += now() - t0;
        }
    }
    double t_proc_add = 0, t_proc_draft = 0, t_wait_draft = 0, t_gpu_draft = 0;

    // draft up to n_max tokens following id_last at position pos0
    void draft(llama_token id_last, int pos0, std::vector<int> & out) {
        out.clear();
        const int n_block = n_max + (dspark && from_anchor ? 0 : 1);
        batch.clear();
        for (int i = 0; i < n_block; ++i) batch.add(i == 0 ? id_last : mask, pos0 + i, 0, !dflash2);
        const double t0 = now(), g0 = ggml_metal_debug_gpu_seconds;
        prof.active = true;
        prof.last = 0;
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) return;
        llama_synchronize(ctx);
        prof.active = false;
        ++n_prof_calls;
        t_proc_draft += now() - t0;
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        if (dflash2) {
            const float * lattice = llama_get_embeddings_nextn(ctx);
            t_wait_draft += now() - t0;
            t_gpu_draft += ggml_metal_debug_gpu_seconds - g0;
            int pred = 0;
            for (int i = 1; i < n_block; ++i) {
                const float * row = lattice + (size_t) i * n_embd;
                const float * sc = row + top_k + (size_t) pred * top_k;
                pred = (int) (std::max_element(sc, sc + top_k) - sc);
                out.push_back((int) row[pred]);
            }
        } else if (dspark) {
            for (int i = from_anchor ? 0 : 1; i < n_block; ++i) out.push_back(argmax(llama_get_logits_ith(ctx, i), n_vocab));
        } else {
            for (int i = 1; i < n_block; ++i) out.push_back(argmax(llama_get_logits_ith(ctx, i), n_vocab));
        }
        llama_memory_seq_rm(llama_get_memory(ctx), 0, pos0, -1);  // the noise block is not context
    }
};

// ---------------------------------------------------------------------------------------------
// native DFlash2 drafter (df.metal): the same graph as llama.cpp's dflash decoder, as ~30 dispatches in one command
// buffer per draft, K/V injection straight from the megakernel's arena; the selector walk runs on the CPU.
struct DfOp {
    uint32_t x, y, z, out;
    uint64_t w0, w1;
    uint32_t pos0, n_kv, layer, side;
    uint32_t flags, n_rows, pad0, pad1;
};
struct DfMv { uint64_t w, w2; uint32_t n_rows, k, y, out, mode, pad; };
static_assert(sizeof(DfOp) == 64 && sizeof(DfMv) == 40, "layout must match df.metal");

struct NativeDrafter {
    enum { DF_EMBED = 1, DF_RESID = 2, DF_OUT_YQ = 4, MV_ROWS = 0, MV_YQ = 1, MV_SWIGLU = 2 };
    static const int NL_MAX = 4, RANK = 256, TOPK = 16, NQ = 32, NKV = 8, HDD = 64, NFEAT = 5;
    // layers, draft vocab, FFN width (from the GGUF: dflt / s1a 2 x 32k x 6144, d3_v16k 3 x 16k x 6144, the public
    // stock drafter 3 x 16k x 4096)
    int nl_d = 2, nvd = 32768, ffd = 6144;
    GGUF g;
    Target * T = nullptr;
    id<MTLBuffer> wbuf, abuf, kc, vc, stage, tofs;
    id<MTLComputePipelineState> p_mv, p_mv4, p_norm, p_conv, p_qk, p_attn, p_feat;
    int tn = 2, nsg = 8;
    float eps = 1e-5f, rope_base = 1e6f;
    std::vector<int> layers;
    std::vector<int64_t> d2t;
    llama_token mask = 0;
    int n_max = 7;
    // arena offsets (floats)
    uint32_t X, Y, DYN, YC, Q, K, V, QR, ATT, O, OC, HH, Y2, DYN2, Y2C, ACT, F, FCV, GATE, LOGITS, FEAT, G, GY, KI, VI, SIZE;
    static const int IG = 4;              // prompt injection: 4 groups of 8 rows per pass (one weight read per 32 rows)
    static const int STAGE_ROWS = 512;    // prompt rows staged per command buffer
    struct Layer { uint64_t attn_norm, conv_proj, conv_base, q, k, v, q_norm, k_norm, o, ffn_norm, fconv_proj, fconv_base, gate, up, down; };
    Layer L[NL_MAX];
    uint64_t fc, enc_norm, out_norm, head, sel_hidden, sel_prev, sel_next, tok_embd;
    double t_gpu = 0, t_walk = 0;  // drafter GPU time (draft + injections), CPU selector walk

    uint64_t off(const std::string & n, uint32_t type) {
        const GGUFTensor & t = g.t(n);
        if (t.type != type) { fprintf(stderr, "drafter tensor %s: type %u, expected %u\n", n.c_str(), t.type, type); exit(1); }
        return t.offset;
    }

    bool init(const char * path, Target * target, llama_token mask_id, const std::string & dir) {
        T = target;
        mask = mask_id;
        std::string err;
        if (!g.open(path, &err)) { fprintf(stderr, "%s\n", err.c_str()); return false; }
        nl_d = (int) g.num("dflash.block_count");
        nvd = (int) g.t("output.weight").ne[1];
        ffd = (int) g.num("dflash.feed_forward_length");
        if (nl_d < 1 || nl_d > NL_MAX || nvd % 64 != 0 || (int) g.t("d2t").ne[0] != nvd || ffd % 256 != 0 ||
            (int) g.num("dflash.selector_top_k") != TOPK ||
            (int) g.num("dflash.selector_rank") != RANK || (int) g.num("dflash.block_size") != 8 ||
            (int) g.num("dflash.conv_kernel_size") != 2 || (int) g.num("dflash.conv_group_size") != 16 ||
            (int) g.num("dflash.attention.head_count") != NQ ||
            (int) g.num("dflash.attention.head_count_kv") != NKV || (int) g.num("dflash.attention.key_length") != HDD) {
            fprintf(stderr, "this native drafter is built for the DFlash2 shapes (1-4 layers, top-16, rank 256, block 8)\n");
            return false;
        }
        eps = (float) g.num("dflash.attention.layer_norm_rms_epsilon");
        rope_base = (float) g.num("dflash.rope.freq_base");
        for (double v : g.arr("dflash.target_layers")) layers.push_back((int) v);
        if ((int) layers.size() != NFEAT) return false;
        wbuf = [T->dev newBufferWithBytesNoCopy:g.map length:g.map_size options:MTLResourceStorageModeShared deallocator:nil];
        auto Lt = [](int il, const char * n) { return "blk." + std::to_string(il) + "." + n; };
        for (int il = 0; il < nl_d; ++il) {
            Layer & l = L[il];
            l.attn_norm = off(Lt(il, "attn_norm.weight"), 0);
            l.conv_proj = off(Lt(il, "attn_conv_proj.weight"), 8);
            l.conv_base = off(Lt(il, "attn_conv_base"), 0);
            l.q = off(Lt(il, "attn_q.weight"), 8);
            l.k = off(Lt(il, "attn_k.weight"), 8);
            l.v = off(Lt(il, "attn_v.weight"), 8);
            l.q_norm = off(Lt(il, "attn_q_norm.weight"), 0);
            l.k_norm = off(Lt(il, "attn_k_norm.weight"), 0);
            l.o = off(Lt(il, "attn_output.weight"), 8);
            l.ffn_norm = off(Lt(il, "ffn_norm.weight"), 0);
            l.fconv_proj = off(Lt(il, "ffn_conv_proj.weight"), 8);
            l.fconv_base = off(Lt(il, "ffn_conv_base"), 0);
            l.gate = off(Lt(il, "ffn_gate.weight"), 8);
            l.up = off(Lt(il, "ffn_up.weight"), 8);
            l.down = off(Lt(il, "ffn_down.weight"), 8);
        }
        fc = off("fc.weight", 8);
        enc_norm = off("enc.output_norm.weight", 0);
        out_norm = off("output_norm.weight", 0);
        head = off("output.weight", 8);
        sel_hidden = off("selector_hidden.weight", 8);
        sel_prev = off("selector_predecessor.weight", 8);
        sel_next = off("selector_successor.weight", 8);
        tok_embd = T->w("token_embd.weight");
        const GGUFTensor & dt = g.t("d2t");
        d2t.resize(nvd);
        memcpy(d2t.data(), (const uint8_t *) g.map + dt.offset, nvd * sizeof(int64_t));
        // arena
        uint32_t o = 0;
        auto take = [&](uint32_t n) { const uint32_t r = o; o += (n + 15) / 16 * 16; return r; };
        X = take(8 * H); Y = take(8 * H); DYN = take(8 * 512); YC = take(8 * H); Q = take(8 * H); K = take(8 * 512);
        V = take(8 * 512); QR = take(8 * H); ATT = take(8 * H); O = take(8 * H); OC = take(8 * H); HH = take(8 * H);
        Y2 = take(8 * H); DYN2 = take(8 * 512); Y2C = take(8 * H); ACT = take(8 * ffd); F = take(8 * H); FCV = take(8 * H);
        GATE = take(8 * RANK); LOGITS = take(8 * nvd); FEAT = take(IG * 8 * NFEAT * H); G = take(IG * 8 * H); GY = take(IG * 8 * H);
        KI = take(IG * 8 * 512); VI = take(IG * 8 * 512);
        SIZE = o;
        abuf = [T->dev newBufferWithLength:(size_t) SIZE * 4 options:MTLResourceStorageModeShared];
        const size_t kv = (size_t) nl_d * CTX * NKV * HDD * 2;
        kc = [T->dev newBufferWithLength:kv options:MTLResourceStorageModePrivate];
        vc = [T->dev newBufferWithLength:kv options:MTLResourceStorageModePrivate];
        stage = [T->dev newBufferWithLength:(size_t) STAGE_ROWS * NFEAT * H * 4 options:MTLResourceStorageModeShared];
        tofs = [T->dev newBufferWithLength:(STAGE_ROWS / (IG * 8) + 1) * 32 options:MTLResourceStorageModeShared];
        // pipelines
        NSString * src = [NSString stringWithContentsOfFile:[[NSString stringWithUTF8String:dir.c_str()] stringByAppendingPathComponent:@"df.metal"]
                                                   encoding:NSUTF8StringEncoding error:nil];
        if (!src) { fprintf(stderr, "cannot read df.metal\n"); return false; }
        NSError * e = nil;
        id<MTLLibrary> lib = [T->dev newLibraryWithSource:[NSString stringWithFormat:@"#define NSG %d\n#define TN %d\n%@", nsg, tn, src]
                                                   options:[MTLCompileOptions new] error:&e];
        if (!lib) { fprintf(stderr, "%s\n", e.localizedDescription.UTF8String); return false; }
        auto P = [&](const char * n) {
            id<MTLComputePipelineState> p = [T->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:n]] error:nil];
            if (!p) { fprintf(stderr, "df pipeline %s failed\n", n); exit(1); }
            return p;
        };
        p_mv = P("df_mv"); p_mv4 = P("df_mv4"); p_norm = P("df_norm"); p_conv = P("df_conv"); p_qk = P("df_qk"); p_attn = P("df_attn"); p_feat = P("df_feat");
        return true;
    }

    // ---- encoding helpers (one compute encoder per call sequence)
    id<MTLComputeCommandEncoder> enc;
    void bind_common() {
        [enc setBuffer:wbuf offset:0 atIndex:1];
        [enc setBuffer:abuf offset:0 atIndex:2];
    }
    void mv(uint64_t w, uint32_t n_rows, uint32_t k, uint32_t y, uint32_t out, uint32_t mode, uint64_t w2 = 0, bool g4 = false) {
        DfMv a = { w, w2, n_rows, k, y, out, mode, 0 };
        [enc setComputePipelineState:g4 ? p_mv4 : p_mv];
        [enc setBytes:&a length:sizeof a atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((n_rows + 8 * tn - 1) / (8 * tn), 1, 1) threadsPerThreadgroup:MTLSizeMake(nsg * 32, 1, 1)];
    }
    void norm(uint32_t flags, uint32_t x, uint32_t y, uint32_t z, uint32_t out, uint64_t w, const uint32_t * toks = nullptr,
              int rows = 8) {
        DfOp O = {};
        O.x = x; O.y = y; O.z = z; O.out = out; O.w0 = w; O.w1 = tok_embd; O.flags = flags;
        [enc setComputePipelineState:p_norm];
        [enc setBytes:&O length:sizeof O atIndex:0];
        [enc setBuffer:T->wbuf offset:0 atIndex:3];
        uint32_t tb[8] = { 0 };
        if (toks) memcpy(tb, toks, sizeof tb);
        [enc setBytes:tb length:sizeof tb atIndex:4];
        [enc setBytes:&eps length:4 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nsg * 32, 1, 1)];
    }
    void conv(uint32_t in, uint32_t dyn, uint32_t out, uint64_t base, uint32_t side, uint32_t flags) {
        DfOp O = {};
        O.x = in; O.y = dyn; O.out = out; O.w0 = base; O.side = side; O.flags = flags;
        [enc setComputePipelineState:p_conv];
        [enc setBytes:&O length:sizeof O atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(8, H / (nsg * 32), 1) threadsPerThreadgroup:MTLSizeMake(nsg * 32, 1, 1)];
    }
    void qk(int il, uint32_t pos0, bool kv_only, int rows = 8) {
        DfOp O = {};
        O.x = Q; O.y = kv_only ? KI : K; O.z = kv_only ? VI : V; O.out = QR; O.w0 = L[il].q_norm; O.w1 = L[il].k_norm;
        O.pos0 = pos0; O.layer = (uint32_t) il;
        O.flags = kv_only ? 1 : 0;
        memcpy(&O.n_rows, &rope_base, 4);
        const uint32_t ctx = CTX;
        [enc setComputePipelineState:p_qk];
        [enc setBytes:&O length:sizeof O atIndex:0];
        [enc setBuffer:kc offset:0 atIndex:3];
        [enc setBuffer:vc offset:0 atIndex:4];
        [enc setBytes:&ctx length:4 atIndex:5];
        [enc setBytes:&eps length:4 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(rows, NQ + 2 * NKV, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    void attn(int il, uint32_t n_kv) {
        DfOp O = {};
        O.x = QR; O.out = ATT; O.n_kv = n_kv; O.layer = (uint32_t) il;
        const uint32_t ctx = CTX;
        [enc setComputePipelineState:p_attn];
        [enc setBytes:&O length:sizeof O atIndex:0];
        [enc setBuffer:kc offset:0 atIndex:3];
        [enc setBuffer:vc offset:0 atIndex:4];
        [enc setBytes:&ctx length:4 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(8, NQ, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    void begin(id<MTLCommandBuffer> cb) { enc = [cb computeCommandEncoder]; bind_common(); }

    // K/V injection of rows already in FEAT (YQ groups of 8 rows, 5 * H each) at positions pos0.. : 8 rows, or 32
    // (IG groups, one weight read). Rows past the real ones write positions that a later injection or the next draft
    // block overwrites before they are read.
    void encode_inject(uint32_t pos0, bool wide) {
        const int rows = wide ? IG * 8 : 8;
        mv(fc, H, NFEAT * H, FEAT, G, MV_ROWS, 0, wide);
        norm(DF_OUT_YQ, G, 0, 0, GY, enc_norm, nullptr, rows);
        for (int il = 0; il < nl_d; ++il) {
            mv(L[il].k, NKV * HDD, H, GY, KI, MV_ROWS, 0, wide);
            mv(L[il].v, NKV * HDD, H, GY, VI, MV_ROWS, 0, wide);
            qk(il, pos0, true, rows);
        }
    }
    // gather target-layer features of `rows` rows into FEAT: row b of layer k at src[offs[k] + b * stride]
    void feat_from(id<MTLBuffer> src, const uint32_t * offs, uint32_t stride, int rows, int slot) {
        DfOp O = {};
        O.out = FEAT;
        O.n_rows = stride;
        memcpy((uint32_t *) tofs.contents + slot * 8, offs, NFEAT * 4);
        [enc setComputePipelineState:p_feat];
        [enc setBytes:&O length:sizeof O atIndex:0];
        [enc setBuffer:src offset:0 atIndex:3];
        [enc setBuffer:tofs offset:slot * 32 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(rows, NFEAT * H / (nsg * 32), 1) threadsPerThreadgroup:MTLSizeMake(nsg * 32, 1, 1)];
    }
    void run(id<MTLCommandBuffer> cb) {
        [enc endEncoding];
        [cb commit];
        while (cb.status < MTLCommandBufferStatusCompleted) {}
        t_gpu += std::max(0.0, cb_gpu_s(cb));
    }

    // prompt rows: features [row][5][H] (CPU, from llama.cpp's layer-input extraction), rows at positions start..;
    // up to STAGE_ROWS rows per command buffer, 32 rows per pass
    void inject_rows(const std::vector<float> & feat, int n_rows, int start) { finish(inject_rows_async(feat, n_rows, start)); }

    // the same, committed without waiting (the caller overlaps CPU work, then finish()es it); nil if no rows
    id<MTLCommandBuffer> inject_rows_async(const std::vector<float> & feat, int n_rows, int start) {
        id<MTLCommandBuffer> cb = nil;
        for (int s0 = 0; s0 < n_rows; s0 += STAGE_ROWS) {
            if (cb) finish(cb);  // the staging buffer is reused: the previous pass must be done
            const int ns = std::min(STAGE_ROWS, n_rows - s0);
            memcpy(stage.contents, feat.data() + (size_t) s0 * NFEAT * H, (size_t) ns * NFEAT * H * sizeof(float));
            cb = [T->queue commandBufferWithUnretainedReferences];
            begin(cb);
            for (int r0 = 0; r0 < ns; r0 += IG * 8) {
                const bool wide = ns - r0 > 8;
                const int rows = wide ? std::min(IG * 8, ns - r0) : ns - r0;
                uint32_t offs[NFEAT];
                for (int k = 0; k < NFEAT; ++k) offs[k] = (uint32_t) ((size_t) r0 * NFEAT * H + (size_t) k * H);
                feat_from(stage, offs, NFEAT * H, rows, r0 / (IG * 8));
                encode_inject((uint32_t) (start + s0 + r0), wide);
            }
            [enc endEncoding];
            [cb commit];
        }
        return cb;
    }
    void finish(id<MTLCommandBuffer> cb) {
        if (!cb) return;
        while (cb.status < MTLCommandBufferStatusCompleted) {}
        t_gpu += std::max(0.0, cb_gpu_s(cb));
    }

    // the target's last NB=8 step rows (their layer inputs are in the megakernel arena) at positions pos0..pos0 + 7:
    // encoded into cb (the verify step's own command buffer: no extra round trip). Rejected rows' positions are
    // rewritten by the next draft block before anything reads them.
    void encode_inject_target(id<MTLCommandBuffer> cb, int pos0) {
        const Arena & a = T->variant(8).ar;
        uint32_t offs[NFEAT];
        for (int k = 0; k < NFEAT; ++k) offs[k] = a.hin[layers[k]];
        begin(cb);
        feat_from(T->abuf, offs, H, 8, 0);
        encode_inject((uint32_t) pos0, false);
        [enc endEncoding];
    }
    void inject_target(int pos0) {
        id<MTLCommandBuffer> cb = [T->queue commandBufferWithUnretainedReferences];
        encode_inject_target(cb, pos0);
        [cb commit];
        while (cb.status < MTLCommandBufferStatusCompleted) {}
        t_gpu += std::max(0.0, cb_gpu_s(cb));
    }

    // CPU dequantize of a Q8_0 row of RANK values
    void q8_row(uint64_t w, int row, float * out) const {
        const uint8_t * p = (const uint8_t *) g.map + w + (size_t) row * (RANK / 32) * 34;
        for (int b = 0; b < RANK / 32; ++b) {
            __fp16 d;
            memcpy(&d, p + b * 34, 2);
            const int8_t * q = (const int8_t *) (p + b * 34 + 2);
            for (int i = 0; i < 32; ++i) out[b * 32 + i] = (float) d * q[i];
        }
    }

    // draft block [last, mask x 7] at positions pos0..pos0 + 7 -> up to 7 draft tokens
    void draft(int last, int pos0, std::vector<int> & out) {
        out.clear();
        const uint32_t toks[8] = { (uint32_t) last, (uint32_t) mask, (uint32_t) mask, (uint32_t) mask, (uint32_t) mask,
                                   (uint32_t) mask, (uint32_t) mask, (uint32_t) mask };
        id<MTLCommandBuffer> cb = [T->queue commandBufferWithUnretainedReferences];
        begin(cb);
        for (int il = 0; il < nl_d; ++il) {
            const Layer & l = L[il];
            if (il == 0) norm(DF_EMBED | DF_OUT_YQ, 0, 0, X, Y, l.attn_norm, toks);
            else         norm(DF_RESID | DF_OUT_YQ, FCV, HH, X, Y, l.attn_norm);
            mv(l.conv_proj, 512, H, Y, DYN, MV_ROWS);
            conv(Y, DYN, YC, l.conv_base, 0, DF_OUT_YQ | 8);
            mv(l.q, H, H, YC, Q, MV_ROWS);
            mv(l.k, NKV * HDD, H, YC, K, MV_ROWS);
            mv(l.v, NKV * HDD, H, YC, V, MV_ROWS);
            qk(il, (uint32_t) pos0, false);
            attn(il, (uint32_t) pos0 + 8);
            mv(l.o, H, H, ATT, O, MV_ROWS);
            conv(O, DYN, OC, l.conv_base, 1, 0);
            norm(DF_RESID | DF_OUT_YQ, OC, X, HH, Y2, l.ffn_norm);
            mv(l.fconv_proj, 512, H, Y2, DYN2, MV_ROWS);
            conv(Y2, DYN2, Y2C, l.fconv_base, 0, DF_OUT_YQ | 8);
            mv(l.gate, ffd, H, Y2C, ACT, MV_SWIGLU, l.up);
            mv(l.down, H, ffd, ACT, F, MV_ROWS);
            conv(F, DYN2, FCV, l.fconv_base, 1, 0);
        }
        norm(DF_RESID | DF_OUT_YQ, FCV, HH, X, Y, out_norm);
        mv(head, nvd, H, Y, LOGITS, MV_ROWS);
        mv(sel_hidden, RANK, H, Y, GATE, MV_ROWS);
        run(cb);
        const double tw = now();
        struct WalkTimer { double t0; double * acc; ~WalkTimer() { *acc += now() - t0; } } wt_{ tw, &t_walk };
        // selector walk (llama.cpp common_speculative_impl_draft_dflash, greedy): position i picks, among its top-16
        // draft-vocab candidates, the one maximising successor[cand] . (predecessor[prev] * gate_i) + logit
        const float * A = (const float *) abuf.contents;
        float pe[RANK], se[RANK], cond[RANK];
        int prev = last;
        for (int i = 1; i <= n_max; ++i) {
            const float * lg = A + LOGITS + (size_t) i * nvd;
            int idx[TOPK];
            float val[TOPK];
            int n = 0;
            float thr = -INFINITY;
            for (int v = 0; v < nvd; ++v) {  // top-16, kept sorted descending
                const float x = lg[v];
                if (n == TOPK && x <= thr) continue;
                int j = n < TOPK ? n++ : TOPK - 1;
                while (j > 0 && val[j - 1] < x) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; --j; }
                val[j] = x;
                idx[j] = v;
                thr = val[n - 1];
            }
            q8_row(sel_prev, prev, pe);
            const float * gt = A + GATE + (size_t) i * RANK;
            for (int r = 0; r < RANK; ++r) cond[r] = pe[r] * gt[r];
            int best = 0;
            float best_s = -INFINITY;
            for (int j = 0; j < TOPK; ++j) {
                q8_row(sel_next, (int) d2t[idx[j]], se);
                float s = val[j];
                for (int r = 0; r < RANK; ++r) s += se[r] * cond[r];
                if (s > best_s) { best_s = s; best = j; }
            }
            prev = (int) d2t[idx[best]];
            out.push_back(prev);
        }
    }
};

// ---------------------------------------------------------------------------------------------
struct Engine {
    OpProfile pprof;  // PE_PREFILL_PROFILE
    Target T;
    llama_model * lm = nullptr;   // target in llama.cpp: tokenizer, and the drafter's shared embeddings / head
    llama_context * lctx = nullptr;
    const llama_vocab * vocab = nullptr;
    Drafter * D = nullptr;
    // stats
    long n_drafted = 0, n_accepted = 0, n_steps = 0;
    double t_prefill = 0, t_decode = 0, t_draft = 0, t_inject = 0;

    bool init(const char * target, const char * dir, const char * draft_kind, const char * draft_path) {
        llama_log_set([](ggml_log_level level, const char * text, void *) { if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr); }, nullptr);
        llama_backend_init();
        // the target also lives in llama.cpp: prefill (mul_mm), tokenizer, and the drafter's shared embeddings / head
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        lm = llama_model_load_from_file(target, mp);
        if (!lm) return false;
        vocab = llama_model_get_vocab(lm);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 2 * CTX;              // the llama.cpp path also serves the rare requests past the engine's context cap
        cp.n_batch = cp.n_ubatch = 512;  // prefill chunk: one graph for prompts up to 512 tokens (97% of requests)
        cp.n_seq_max = 1;
        cp.swa_full = true;              // SWA layers keep every position: the handoff copies them all
        // flash attention: prefill 61 vs 63-64 ms (175 tokens, M4 Max), V stays untransposed (handoff 0.4 vs 1.6 ms)
        cp.flash_attn_type = getenv("PE_NOFA") ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (getenv("PE_PREFILL_PROFILE")) {  // per-op wall time of llama.cpp's prefill (each node synchronized)
            cp.cb_eval = [](ggml_tensor * t, bool ask, void * ud) -> bool {
                auto * P = (OpProfile *) ud;
                if (ask) return true;
                const double t1 = now();
                if (P->active && P->last > 0) {
                    char key[96];
                    snprintf(key, sizeof key, "%s %s [%lld,%lld]", ggml_op_desc(t), ggml_type_name(t->type), (long long) t->ne[0],
                             (long long) t->ne[1]);
                    P->ms[key] += (t1 - P->last) * 1e3;
                    P->n[key] += 1;
                }
                P->last = now();
                return true;
            };
            cp.cb_eval_user_data = &pprof;
        }
        lctx = llama_init_from_model(lm, cp);
        if (!lctx) return false;
        const bool native = draft_kind && strcmp(draft_kind, "dflash2n") == 0;
        if (draft_path && !native) {
            D = new Drafter;
            if (!D->init(draft_path, strcmp(draft_kind, "dspark") == 0, lctx)) { fprintf(stderr, "cannot load drafter\n"); return false; }
            feat_layers = D->layers;
        }
        if (!T.init(target, dir)) return false;
        // prefill mat-muls through kernel_mul_mm_ph (64-token tiles, bit-identical to llama.cpp's kernel_mul_mm) where
        // the device tune says it pays; set before the first llama.cpp graph runs (pipelines are fetched lazily)
        ggml_backend_metal_phonon_set_mm(T.ph_mm);
        if (native) {
            ND = new NativeDrafter;
            GGUF probe;
            std::string err;
            if (!probe.open(draft_path, &err)) { fprintf(stderr, "%s\n", err.c_str()); return false; }
            const llama_token mask_id = (llama_token) probe.num("tokenizer.ggml.mask_token_id");
            if (!ND->init(draft_path, &T, mask_id, dir)) { fprintf(stderr, "cannot load native drafter\n"); return false; }
            feat_layers = ND->layers;
        }
        for (int il : feat_layers) llama_set_embeddings_layer_inp(lctx, (uint32_t) il, true);
        cbuf.resize((size_t) 6 * H);
        pbatch = common_batch(lctx);
        return true;
    }

    NativeDrafter * ND = nullptr;
    std::vector<int> feat_layers;  // target layers whose inputs feed the drafter

    std::vector<float> cbuf;  // handoff staging: conv state [channel][6]
    common_batch pbatch;
    double t_handoff = 0;

    // llama.cpp objects go before the Metal device is torn down (ggml-metal asserts on live residency sets)
    void shutdown() {
        if (D) { llama_free(D->ctx); llama_model_free(D->model); delete D; D = nullptr; }
        pbatch = common_batch();
        if (lctx) llama_free(lctx);
        if (lm) llama_model_free(lm);
        lctx = nullptr;
        lm = nullptr;
        llama_backend_free();
    }

    // copy llama.cpp's state after a prefill of n positions into the megakernel's KV cache and conv rings
    bool handoff(int n) {
        uint16_t * kc = (uint16_t *) T.kc.contents, * vc = (uint16_t *) T.vc.contents;
        float * ring = (float *) T.ring.contents;
        for (int il = 0; il < NL; ++il) {
            if (T.is_attn[il]) {
                const size_t off = (size_t) T.attn_idx[il] * CTX * NKVH * HD;
                if (!llama_phonon_export_layer(lctx, il, n, kc + off, vc + off, nullptr)) return false;
            } else {
                if (!llama_phonon_export_layer(lctx, il, n, nullptr, nullptr, cbuf.data())) return false;
                const int R = T.ring_len;
                float * rg = ring + (size_t) T.conv_idx[il] * H * R;
                for (int t = 0; t < 6; ++t) {
                    const int p = n - 6 + t;
                    if (p < 0) continue;
                    for (int ch = 0; ch < H; ++ch) rg[(size_t) ch * R + (p & (R - 1))] = cbuf[(size_t) ch * 6 + t];
                }
            }
        }
        return true;
    }

    std::vector<int> tokenize(const std::string & s) {
        std::vector<int> t(s.size() + 16);
        const int n = llama_tokenize(vocab, s.c_str(), (int) s.size(), t.data(), (int) t.size(), true, true);
        t.resize(std::max(n, 0));
        return t;
    }

    std::string detok(const std::vector<int> & t) {
        std::string s;
        char buf[256];
        for (int id : t) {
            const int n = llama_token_to_piece(vocab, id, buf, sizeof buf, 0, false);
            if (n > 0) s.append(buf, n);
        }
        return s;
    }

    void features(int nb, int b, float * dst) {
        for (size_t k = 0; k < D->layers.size(); ++k) memcpy(dst + k * H, T.layer_in(nb, D->layers[k], b), H * sizeof(float));
    }

    // llama.cpp forward of toks[start, n) at positions start.. (512-token chunks); the drafter's target-layer
    // inputs of those rows go to feat ([row - start][layer][H]); with bx (and llama_phonon_set_capture_bx on), the
    // conv inputs of positions start.. go to (*bx)[conv layer][channel][position], n positions per channel
    void llama_forward(const std::vector<int> & toks, int start, std::vector<float> & feat, std::vector<float> * bx = nullptr) {
        const int n = (int) toks.size();
        const size_t n_enc = feat_layers.size() * H;
        if (n_enc) feat.resize((size_t) (n - start) * n_enc);
        if (bx) bx->resize((size_t) n_conv() * H * n);
        for (int off = start; off < n; off += 512) {
            const int m = std::min(512, n - off);
            pbatch.clear();
            for (int i = 0; i < m; ++i) pbatch.add(toks[off + i], off + i, 0, off + i == n - 1);
            if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, pbatch.get()) != 0) { fprintf(stderr, "prefill failed\n"); exit(1); }
            if (bx) {
                for (int il = 0; il < NL; ++il) {
                    if (T.is_attn[il]) continue;
                    if (!llama_phonon_copy_bx(lctx, il, m, bx->data() + (size_t) T.conv_idx[il] * H * n + off, (size_t) n)) {
                        fprintf(stderr, "conv input capture failed\n");
                        exit(1);
                    }
                }
            }
            if (n_enc) {
                for (size_t k = 0; k < feat_layers.size(); ++k) {
                    const float * L = llama_get_embeddings_layer_inp(lctx, (uint32_t) feat_layers[k]);
                    for (int i = 0; i < m; ++i) {
                        memcpy(feat.data() + (size_t) (off - start + i) * n_enc + k * H, L + (size_t) i * H, H * sizeof(float));
                    }
                }
            }
        }
    }

    // prefix cache: the refiner's fixed system-prompt prefix is prefilled once; its llama.cpp state (KV + conv) is
    // restored per request and the drafter keeps its injected prefix rows
    std::vector<int> prefix;
    std::vector<uint8_t> prefix_state;
    std::vector<float> prefix_feat;
    bool dft_has_prefix = false;
    int n_prefix_hits = 0;

    void set_prefix(const std::vector<int> & P) {
        prefix = P;
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_forward(P, 0, prefix_feat, capture_bx ? &prefix_bx : nullptr);
        if (!capture_bx) prefix_bx.clear();
        prefix_state.resize(llama_state_seq_get_size_ext(lctx, 0, 0));
        llama_state_seq_get_data_ext(lctx, prefix_state.data(), prefix_state.size(), 0, 0);
        dft_has_prefix = false;
    }

    int n_conv() const { return (int) std::count(T.is_attn.begin(), T.is_attn.end(), false); }

    // ---- sessions (serve mode): one live per-dictation state that the caller extends, rolls back and queries ----
    // The megakernel's buffers hold the session's state: K/V of every position (the sliding-window layers keep all
    // positions and apply the window as a mask) and every position's conv input (the full-length history, allocated
    // at the first session open: Target::full_ring), so a rollback is a length change. Forwards of up to 8 tokens run on the megakernel (NB = 1 / 8); longer ones on llama.cpp's
    // prefill path, whose state follows lazily: it holds positions [0, lv) (truncated, conv state rebuilt from the
    // megakernel's history, before its next forward if a rollback went below its end); positions the megakernel
    // added after lv are re-run in that forward. Each llama.cpp forward hands its K/V and conv inputs back.
    bool capture_bx = false;                // llama_phonon_set_capture_bx is on (from the first session on)
    std::vector<float> prefix_bx;           // conv inputs of the prefix positions [conv layer][channel][position]
    std::vector<int> s_toks;                // session state: token ids of positions 0..n-1
    int s_id = 0, s_lv = 0;                 // session id (0: none), llama.cpp end
    bool s_trim = false;                    // llama.cpp holds more than s_lv: truncate before its next forward
    std::atomic<bool> s_cancel{false};

    // conv inputs of positions [p0, p0 + n) of every conv layer, from a [layer][channel][stride] buffer, to the ring
    void ring_put(const float * src, size_t stride, int p0, int n) {
        float * ring = (float *) T.ring.contents;
        for (int c = 0; c < n_conv(); ++c)
            for (int ch = 0; ch < H; ++ch)
                memcpy(ring + ((size_t) c * H + ch) * CTX + p0, src + ((size_t) c * H + ch) * stride, (size_t) n * sizeof(float));
    }

    // K/V of positions [p0, p1) from llama.cpp into the megakernel's cache
    bool kv_put(int p0, int p1) {
        uint16_t * kc = (uint16_t *) T.kc.contents, * vc = (uint16_t *) T.vc.contents;
        for (int il = 0; il < NL; ++il) {
            if (!T.is_attn[il]) continue;
            const size_t off = (size_t) T.attn_idx[il] * CTX * NKVH * HD;
            if (!llama_phonon_export_range(lctx, il, p0, p1, kc + off, vc + off, nullptr)) return false;
        }
        return true;
    }

    // open: the state of the prefix tokens (from the prefix cache when they are its tokens). Returns the session id.
    int session_open(const std::vector<int> & P) {
        if (!capture_bx) {  // graph changes (one buffer re-plan): conv input capture, device argmax per output row
            llama_phonon_set_capture_bx(lctx, true);
            llama_phonon_set_argmax(lctx, true);
            capture_bx = true;
        }
        T.full_ring();
        if (P != prefix || prefix_bx.size() != (size_t) n_conv() * H * P.size()) set_prefix(P);
        const int np = (int) P.size();
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_state_seq_set_data_ext(lctx, prefix_state.data(), prefix_state.size(), 0, 0);
        if (getenv("PE_SESSION_DEBUG")) {  // the captured conv inputs of the last 6 positions vs llama.cpp's conv state
            double dmax = 0;
            for (int il = 0; il < NL; ++il) {
                if (T.is_attn[il]) continue;
                llama_phonon_export_layer(lctx, il, np, nullptr, nullptr, cbuf.data());
                const float * bx = prefix_bx.data() + (size_t) T.conv_idx[il] * H * np;
                double dl = 0;
                int worst = 0;
                for (int ch = 0; ch < H; ++ch)
                    for (int t = 0; t < 6; ++t) {
                        const double d = fabsf(cbuf[(size_t) ch * 6 + t] - bx[(size_t) ch * np + np - 6 + t]);
                        if (d > dl) { dl = d; worst = ch; }
                    }
                if (dl > 0) fprintf(stderr, "  layer %d: max |diff| %.3g (channel %d)\n", il, dl, worst);
                dmax = std::max(dmax, dl);
            }
            fprintf(stderr, "session open: captured conv inputs vs llama.cpp conv state, max |diff| %.3g\n", dmax);
        }
        if (!kv_put(0, np)) { fprintf(stderr, "session open: K/V export failed\n"); exit(1); }
        ring_put(prefix_bx.data(), np, 0, np);
        s_toks = P;
        s_lv = np;
        s_trim = false;
        dft_has_prefix = false;  // the one-shot path's drafter prefix bookkeeping no longer holds
        return ++s_id;
    }

    // rollback: keep positions [0, n)
    bool session_rollback(int n) {
        if (n < 1 || n > (int) s_toks.size()) return false;
        s_toks.resize(n);
        if (s_lv > n) { s_lv = n; s_trim = true; }
        return true;
    }

    // forward: append toks; returns the argmax ids of rows from..k-1 (row i: the token after toks[i]). On a cancel
    // between llama.cpp passes, the processed part stays appended and *cancelled is set.
    std::vector<int> session_forward(const std::vector<int> & toks, int from, bool * cancelled, const char ** path) {
        std::vector<int> ids;
        const int n = (int) s_toks.size(), k = (int) toks.size();
        *cancelled = false;
        if ((k == 1 || (k <= MAX_NB && n + MAX_NB <= CTX)) && !getenv("PE_SESSION_LLAMA")) {
            *path = "mk";
            const int nb = k == 1 ? 1 : MAX_NB;
            T.step(toks.data(), nb, k, n);
            for (int b = from; b < k; ++b) ids.push_back(T.pick(nb, b));
            s_toks.insert(s_toks.end(), toks.begin(), toks.end());
            return ids;
        }
        *path = "llama";
        if (s_trim) {  // llama.cpp back to s_lv: K/V truncated, conv state = the conv inputs of the 6 positions before
            const float * ring = (const float *) T.ring.contents;
            std::vector<std::vector<float>> st(NL);
            std::vector<const float *> ptr(NL, nullptr);
            for (int il = 0; il < NL; ++il) {
                if (T.is_attn[il]) continue;
                st[il].assign((size_t) 6 * H, 0.f);
                const float * rg = ring + (size_t) T.conv_idx[il] * H * CTX;
                for (int t = 0; t < 6; ++t) {
                    const int p = s_lv - 6 + t;
                    if (p < 0) continue;
                    for (int ch = 0; ch < H; ++ch) st[il][(size_t) ch * 6 + t] = rg[(size_t) ch * CTX + p];
                }
                ptr[il] = st[il].data();
            }
            if (!llama_phonon_truncate(lctx, s_lv, ptr.data())) { fprintf(stderr, "session: llama.cpp truncate failed\n"); exit(1); }
            s_trim = false;
        }
        // rows s_lv .. n - 1 (megakernel-only positions) and the new ones, 512 per pass
        std::vector<int> seq(s_toks.begin() + s_lv, s_toks.end());
        seq.insert(seq.end(), toks.begin(), toks.end());
        const int base = s_lv, first_out = n + from;  // absolute positions
        std::vector<float> bx((size_t) n_conv() * H * 512);
        int done = 0;
        while (done < (int) seq.size()) {
            if (done > 0 && s_cancel.load()) { *cancelled = true; break; }
            const int m = std::min(512, (int) seq.size() - done), p0 = base + done;
            pbatch.clear();
            for (int i = 0; i < m; ++i) pbatch.add(seq[done + i], p0 + i, 0, p0 + i >= first_out);
            ggml_backend_metal_phonon_set_mm_sk(T.sk_rows);  // sessions only: the one-shot path stays bit-identical
            llama_phonon_set_ids_only(lctx, true);          // the argmax of each output row comes from the GPU
            const int rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, pbatch.get());
            llama_phonon_set_ids_only(lctx, false);
            ggml_backend_metal_phonon_set_mm_sk(0);
            if (rc != 0) { fprintf(stderr, "session forward failed\n"); exit(1); }
            int n_out = 0;
            for (int i = 0; i < m; ++i) n_out += p0 + i >= first_out;
            if (n_out > 0) {
                int32_t n_am = 0;
                const int32_t * am = llama_phonon_argmax_ids(lctx, &n_am);
                if (n_am != n_out) { fprintf(stderr, "session forward: %d argmax ids for %d rows\n", n_am, n_out); exit(1); }
                ids.insert(ids.end(), am, am + n_out);
            }
            for (int il = 0; il < NL; ++il) {
                if (T.is_attn[il]) continue;
                if (!llama_phonon_copy_bx(lctx, il, m, bx.data() + (size_t) T.conv_idx[il] * H * 512, 512)) {
                    fprintf(stderr, "conv input capture failed\n");
                    exit(1);
                }
            }
            ring_put(bx.data(), 512, p0, m);
            if (!kv_put(p0, p0 + m)) { fprintf(stderr, "session: K/V export failed\n"); exit(1); }
            done += m;
        }
        s_lv = base + done;
        if (s_lv > n) s_toks.insert(s_toks.end(), toks.begin(), toks.begin() + (s_lv - n));
        return ids;
    }

    // prefill in llama.cpp, then hand the state to the megakernel; returns the first generated token. With a
    // drafter, the prompt's target-layer inputs go into the drafter cache.
    int prefill(const std::vector<int> & toks) {
        const double t0 = now();
        const int n = (int) toks.size(), np = (int) prefix.size();
        const bool hit = np > 0 && n > np && std::equal(prefix.begin(), prefix.end(), toks.begin());
        llama_memory_clear(llama_get_memory(lctx), true);
        if (hit) {
            llama_state_seq_set_data_ext(lctx, prefix_state.data(), prefix_state.size(), 0, 0);
            ++n_prefix_hits;
        }
        const int start = hit ? np : 0;
        std::vector<float> feat;
        llama_forward(toks, start, feat);
        const int next = argmax(llama_get_logits_ith(lctx, -1), T.n_vocab);
        t_prefill += now() - t0;
        // native drafter: its K/V injection of the prompt rows runs on the GPU while the CPU copies the handoff
        id<MTLCommandBuffer> inj = nil;
        double t_inj0 = 0;
        if (ND) {
            const double t2 = now();
            if (hit && !dft_has_prefix) ND->inject_rows(prefix_feat, np, 0);
            dft_has_prefix = hit;
            inj = ND->inject_rows_async(feat, n - start, start);
            t_inj0 = now() - t2;
        }
        const double t1 = now();
        if (!handoff(n)) { fprintf(stderr, "state handoff failed\n"); exit(1); }
        t_handoff += now() - t1;
        if (ND) {
            const double t3 = now();
            ND->finish(inj);
            t_inject += t_inj0 + (now() - t3);
        }
        if (D) {
            const double t2 = now();
            if (hit) {
                if (!dft_has_prefix) {
                    D->clear();
                    D->add(prefix_feat, np, 0);
                    dft_has_prefix = true;
                }
                D->trim(np);
            } else {
                D->clear();
                dft_has_prefix = false;
            }
            D->add(feat, n - start, start);
            t_inject += now() - t2;
        }
        return next;
    }

    // the llama.cpp path (prefill + one-token greedy decode, no drafter): requests that do not fit the engine's
    // context cap (0.05% of real requests) continue here. Returns at most n tokens, stops after an end-of-generation
    // token (included).
    int n_fallback = 0;
    std::vector<int> llama_generate(const std::vector<int> & seq, int n) {
        ++n_fallback;
        std::vector<int> out, none;
        std::vector<float> feat;
        llama_memory_clear(llama_get_memory(lctx), true);
        const auto saved = feat_layers;
        feat_layers.clear();  // no drafter features on this path
        llama_forward(seq, 0, feat);
        feat_layers = saved;
        int pos = (int) seq.size();
        while ((int) out.size() < n && pos < 2 * CTX) {
            const int t = argmax(llama_get_logits_ith(lctx, -1), T.n_vocab);
            out.push_back(t);
            if (llama_vocab_is_eog(vocab, t)) break;
            pbatch.clear();
            pbatch.add(t, pos++, 0, true);
            if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, pbatch.get()) != 0) break;
        }
        dft_has_prefix = false;  // the llama.cpp state no longer matches the prefix snapshot's assumptions
        return out;
    }

    std::vector<int> generate(const std::vector<int> & prompt, int max_new) {
        std::vector<int> out;
        if ((int) prompt.size() + 16 > CTX) {  // the prompt alone does not fit: llama.cpp path
            out = llama_generate(prompt, max_new);
            if (!out.empty() && llama_vocab_is_eog(vocab, out.back())) out.pop_back();
            return out;
        }
        int last = prefill(prompt);
        int pos = (int) prompt.size();
        out.push_back(last);
        std::vector<int> draft, rows(8);
        std::vector<float> feat;
        while (!llama_vocab_is_eog(vocab, last) && (int) out.size() < max_new && pos + 9 <= CTX) {
            if (!D && !ND) {
                const double t0 = now();
                T.step(&last, 1, 1, pos);
                last = T.pick(1, 0);
                t_decode += now() - t0;
                out.push_back(last);
                ++pos;
                continue;
            }
            double t0 = now();
            if (ND) ND->draft(last, pos, draft);
            else    D->draft(last, pos, draft);
            t_draft += now() - t0;
            t0 = now();
            const int k = 1 + (int) draft.size();
            rows[0] = last;
            for (int i = 0; i < (int) draft.size(); ++i) rows[1 + i] = draft[i];
            const int nb = k == 1 && !ND ? 1 : 8;
            if (ND && !getenv("PE_SEPARATE_INJECT")) {  // verify + the drafter's injection of its rows: one command buffer
                T.step(rows.data(), nb, k, pos, true, true, [&](id<MTLCommandBuffer> cb) { ND->encode_inject_target(cb, pos); });
            } else {
                T.step(rows.data(), nb, k, pos);
            }
            int na = 0, next = 0;
            for (;;) {  // accept while the target's argmax matches the draft
                next = T.pick(nb, na);
                if (na < (int) draft.size() && draft[na] == next) { ++na; continue; }
                break;
            }
            t_decode += now() - t0;
            ++n_steps;
            n_drafted += (long) draft.size();
            n_accepted += na;
            // committed rows 0..na (last, draft[0..na-1]) feed the drafter
            t0 = now();
            if (ND) {
                if (getenv("PE_SEPARATE_INJECT")) ND->inject_target(pos);
            } else {
                const size_t n_enc = D->layers.size() * H;
                feat.resize((size_t) (na + 1) * n_enc);
                for (int b = 0; b <= na; ++b) features(nb, b, feat.data() + (size_t) b * n_enc);
                D->add(feat, na + 1, pos);
            }
            t_inject += now() - t0;
            bool stop = false;
            for (int i = 0; i < na && !stop; ++i) {
                out.push_back(draft[i]);
                stop = llama_vocab_is_eog(vocab, draft[i]) || (int) out.size() >= max_new;
            }
            if (stop) break;
            out.push_back(next);
            last = next;
            pos += na + 1;
        }
        if ((int) out.size() > max_new) out.resize(max_new);
        if (!out.empty() && !llama_vocab_is_eog(vocab, out.back()) && (int) out.size() < max_new && pos + 9 > CTX) {
            // the context cap is reached: continue on the llama.cpp path from the full sequence
            std::vector<int> seq = prompt;
            seq.insert(seq.end(), out.begin(), out.end());
            const std::vector<int> more = llama_generate(seq, max_new - (int) out.size());
            out.insert(out.end(), more.begin(), more.end());
        }
        if (!out.empty() && llama_vocab_is_eog(vocab, out.back())) out.pop_back();
        return out;
    }
};

static std::string json_str(const std::string & s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char) c;
        }
    }
    return o + "\"";
}

// minimal reader for one string field of a JSON line, e.g. {"prompt": "..."} (unescaping incl. \uXXXX + surrogate pairs)
static std::string read_str(const std::string & line, const char * key) {
    size_t i = line.find(key);
    if (i == std::string::npos) return "";
    i = line.find('"', line.find(':', i) + 1) + 1;
    std::string s;
    auto put = [&](uint32_t cp) {
        if (cp < 0x80) s += (char) cp;
        else if (cp < 0x800) { s += (char) (0xC0 | (cp >> 6)); s += (char) (0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { s += (char) (0xE0 | (cp >> 12)); s += (char) (0x80 | ((cp >> 6) & 0x3F)); s += (char) (0x80 | (cp & 0x3F)); }
        else { s += (char) (0xF0 | (cp >> 18)); s += (char) (0x80 | ((cp >> 12) & 0x3F)); s += (char) (0x80 | ((cp >> 6) & 0x3F)); s += (char) (0x80 | (cp & 0x3F)); }
    };
    while (i < line.size() && line[i] != '"') {
        char c = line[i++];
        if (c != '\\') { s += c; continue; }
        c = line[i++];
        switch (c) {
            case 'n': s += '\n'; break;
            case 't': s += '\t'; break;
            case 'r': s += '\r'; break;
            case 'b': s += '\b'; break;
            case 'f': s += '\f'; break;
            case 'u': {
                uint32_t cp = (uint32_t) strtoul(line.substr(i, 4).c_str(), nullptr, 16);
                i += 4;
                if (cp >= 0xD800 && cp < 0xDC00 && line[i] == '\\' && line[i + 1] == 'u') {
                    const uint32_t lo = (uint32_t) strtoul(line.substr(i + 2, 4).c_str(), nullptr, 16);
                    i += 6;
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                }
                put(cp);
                break;
            }
            default: s += c;
        }
    }
    return s;
}

static std::string read_prompt(const std::string & line) { return read_str(line, "\"prompt\""); }

// integer value of a JSON key (def if absent) and an integer array value
static long read_int(const std::string & line, const char * key, long def) {
    const size_t i = line.find(key);
    if (i == std::string::npos) return def;
    return strtol(line.c_str() + line.find(':', i) + 1, nullptr, 10);
}
static std::vector<int> read_ints(const std::string & line, const char * key) {
    std::vector<int> v;
    size_t i = line.find(key);
    if (i == std::string::npos) return v;
    i = line.find('[', i);
    const size_t e = line.find(']', i);
    if (i == std::string::npos || e == std::string::npos) return v;
    const char * p = line.c_str() + i + 1, * end = line.c_str() + e;
    while (p < end) {
        char * q;
        const long x = strtol(p, &q, 10);
        if (q == p) { ++p; continue; }
        v.push_back((int) x);
        p = q;
    }
    return v;
}
static std::string ints_json(const std::vector<int> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}

static double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[(size_t) (p * (v.size() - 1))];
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 3) { fprintf(stderr, "usage: see the header of pe.mm\n"); return 1; }
        const std::string mode = argv[2];
        const char * draft_kind = nullptr, * draft_path = nullptr, * out_path = nullptr, * prefix_path = nullptr;
        int n_rows = 1 << 30, warm = 5;
        for (int i = 3; i < argc; ++i) {
            if (!strcmp(argv[i], "--draft") && i + 2 < argc) { draft_kind = argv[i + 1]; draft_path = argv[i + 2]; i += 2; }
            else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
            else if (!strcmp(argv[i], "--n") && i + 1 < argc) n_rows = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--warm") && i + 1 < argc) warm = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--prefix") && i + 1 < argc) prefix_path = argv[++i];
        }
        std::string dir = getenv("PE_DIR") ? getenv("PE_DIR") : "";
        if (dir.empty()) {
            dir = argv[0];
            dir = dir.find('/') == std::string::npos ? "." : dir.substr(0, dir.rfind('/')) + "/..";
        }
        const double t0 = now();
        Engine E;
        if (!E.init(argv[1], dir.c_str(), draft_kind, draft_path)) return 1;
        fprintf(stderr, "init %.2f s (shader compile %.2f s for NB = 1, 8), grid %d x %d\n", now() - t0, E.T.compile_s,
                E.T.n_tg, E.T.nsg * 32);
        if (prefix_path) {
            std::ifstream pf(prefix_path);
            std::stringstream ss;
            ss << pf.rdbuf();
            const double tp = now();
            E.set_prefix(E.tokenize(ss.str()));
            fprintf(stderr, "prefix cache: %zu tokens (%.1f ms)\n", E.prefix.size(), (now() - tp) * 1e3);
        }

        if (mode == "bench") {
            // decode at 250 tokens of context, verify-8 at 250, prefill 175 and 350 from empty
            std::string text;
            for (int i = 0; i < 40; ++i) text += "so the plan is to move the pie torch loop onto flash attention. ";
            std::vector<int> toks = E.tokenize(text);
            auto run_prefill = [&](int n) {
                std::vector<int> p(toks.begin(), toks.begin() + n);
                double best = 1e9, sum = 0;
                for (int r = 0; r < 4; ++r) {
                    const double a = now();
                    E.prefill(p);
                    const double ms = (now() - a) * 1e3;
                    if (r > 0) { best = std::min(best, ms); sum += ms; }
                }
                printf("prefill %d: min %.2f ms mean %.2f ms\n", n, best, sum / 3);
            };
            E.prefill(std::vector<int>(toks.begin(), toks.begin() + 250));
            int t = toks[250];
            const int N = 64;
            double wall = 0, gpu = 0, best = 1e9;
            int n_gpu = 0;
            for (int i = 0; i < N + 8; ++i) {
                const double a = now();
                E.T.step(&t, 1, 1, 250 + i);
                t = E.T.pick(1, 0);
                const double dt = now() - a;
                if (i >= 8) {
                    wall += dt;
                    best = std::min(best, dt);
                    if (E.T.last_gpu_ms >= 0) { gpu += E.T.last_gpu_ms; ++n_gpu; }
                }
            }
            printf("decode: wall %.3f ms (min %.3f), GPU %.3f ms\n", wall / N * 1e3, best * 1e3, gpu / std::max(1, n_gpu));
            for (int k : { 8 }) {
                double bw = 1e9, sw = 0, bg = 1e9;
                for (int rep = 0; rep < 12; ++rep) {
                    const double a = now();
                    E.T.step(&toks[250], k, k, 250);
                    for (int b = 0; b < k; ++b) (void) E.T.pick(k, b);
                    const double ms = (now() - a) * 1e3;
                    if (rep > 1) { bw = std::min(bw, ms); sw += ms; if (E.T.last_gpu_ms >= 0) bg = std::min(bg, E.T.last_gpu_ms); }
                }
                printf("verify-%d: wall min %.3f ms mean %.3f ms, GPU min %.3f ms\n", k, bw, sw / 10, bg);
            }
            if (E.ND) {  // one speculative step at 250 tokens of context: draft, verify-8, inject the committed rows
                std::vector<int> dr;
                double sd = 0, sv = 0, si = 0;
                const int R = 20;
                for (int rep = 0; rep < R + 2; ++rep) {
                    double a = now();
                    E.ND->draft(toks[250], 250, dr);
                    const double d1 = now() - a;
                    std::vector<int> rows = { toks[250] };
                    rows.insert(rows.end(), dr.begin(), dr.end());
                    a = now();
                    E.T.step(rows.data(), 8, 8, 250);
                    for (int b = 0; b < 8; ++b) (void) E.T.pick(8, b);
                    const double d2 = now() - a;
                    a = now();
                    E.ND->inject_target(250);
                    const double d3 = now() - a;
                    if (rep >= 2) { sd += d1; sv += d2; si += d3; }
                }
                printf("spec step (native drafter): draft %.3f ms, verify-8 %.3f ms, inject %.3f ms, total %.3f ms\n",
                       sd / R * 1e3, sv / R * 1e3, si / R * 1e3, (sd + sv + si) / R * 1e3);
                for (int n : { 123, 175, 350 }) {
                    std::vector<float> feat((size_t) n * NativeDrafter::NFEAT * H, 0.01f);
                    double best = 1e9;
                    for (int rep = 0; rep < 4; ++rep) {
                        const double a = now();
                        E.ND->inject_rows(feat, n, 0);
                        best = std::min(best, now() - a);
                    }
                    printf("prompt inject %d rows: %.3f ms\n", n, best * 1e3);
                }
            }
            run_prefill(175);
            run_prefill(350);
            if (getenv("PE_PREFILL_SWEEP")) for (int n : { 8, 16, 32, 64, 123, 256 }) run_prefill(n);
            if (getenv("PE_PREFILL_PROFILE")) {
                std::vector<int> p(toks.begin(), toks.begin() + 123);
                E.pprof.active = true;
                for (int r = 0; r < 3; ++r) { E.pprof.last = 0; E.prefill(p); }
                E.pprof.active = false;
                printf("llama.cpp prefill of 123 tokens, per-op wall (each node synchronized):\n");
                E.pprof.print(3);
            }
            if (!E.prefix.empty()) {  // as served: the cached system prefix, then the rest of the prompt
                for (int n : { 175, 350 }) {
                    std::vector<int> p(E.prefix.begin(), E.prefix.end());
                    p.insert(p.end(), toks.begin() + 1, toks.begin() + 1 + (n - (int) p.size()));
                    double best = 1e9;
                    for (int r = 0; r < 4; ++r) {
                        const double a = now();
                        E.prefill(p);
                        if (r > 0) best = std::min(best, (now() - a) * 1e3);
                    }
                    printf("prefill %d with the %zu-token prefix cached: min %.2f ms\n", n, E.prefix.size(), best);
                }
            }
        } else if (mode == "layerdiff") {
            // where the megakernel's decode departs from llama.cpp's: after a llama.cpp prefill of each prompt, one
            // decode step of the first generated token by both, then max |diff| of every layer's input (the residual
            // stream entering layer il; il = NL is after the last layer) and of the logits. argv[3] prompts, argv[4] N
            std::ifstream in(argv[3]);
            std::string line;
            std::vector<std::string> prompts;
            while (std::getline(in, line)) if (!line.empty()) prompts.push_back(read_prompt(line));
            const int n = std::min((int) prompts.size(), atoi(argv[4]));
            for (int il = 0; il < NL; ++il) llama_set_embeddings_layer_inp(E.lctx, (uint32_t) il, true);
            std::vector<double> dmax(NL + 1, 0.0), rmax(NL + 1, 0.0);
            double dlog = 0;
            for (int i = 0; i < n; ++i) {
                const std::vector<int> p = E.tokenize(prompts[i]);
                const int np = (int) p.size();
                int t = E.prefill(p);
                E.T.step(&t, 1, 1, np);
                std::vector<float> mk((size_t) (NL + 1) * H);
                for (int il = 0; il <= NL; ++il) memcpy(mk.data() + (size_t) il * H, E.T.layer_in(1, il, 0), H * sizeof(float));
                std::vector<float> mlog(E.T.logits(1), E.T.logits(1) + E.T.n_vocab);
                E.pbatch.clear();
                E.pbatch.add(t, np, 0, true);
                llama_process(E.lctx, LLAMA_PROCESS_TYPE_DECODE, E.pbatch.get());
                const float * ll = llama_get_logits_ith(E.lctx, -1);
                for (int j = 0; j < E.T.n_vocab; ++j) dlog = std::max(dlog, (double) fabsf(ll[j] - mlog[j]));
                for (int il = 0; il < NL; ++il) {
                    const float * ref = llama_get_embeddings_layer_inp(E.lctx, (uint32_t) il);
                    double d = 0, m = 0;
                    for (int j = 0; j < H; ++j) { d = std::max(d, (double) fabsf(ref[j] - mk[(size_t) il * H + j])); m = std::max(m, (double) fabsf(ref[j])); }
                    dmax[il] = std::max(dmax[il], d);
                    rmax[il] = std::max(rmax[il], d / std::max(m, 1e-30));
                }
            }
            printf("layer-input max|diff| megakernel vs llama.cpp decode over %d prompts (relative to max|value|):\n", n);
            for (int il = 0; il < NL; ++il) printf("  layer %2d (%s): %.3g (%.2g)\n", il, E.T.is_attn[il] ? (E.T.is_swa[il] ? "swa " : "attn") : "conv", dmax[il], rmax[il]);
            printf("  logits: %.4g\n", dlog);
        } else if (mode == "reqbench") {
            // request-shape bench: argv[3] = TSV of (prompt tokens, generated tokens) pairs; synthetic prompts of those
            // lengths, the given number of generated tokens forced (EOS ignored). Plain: NB=1 decode steps. With
            // --draft dflash2n: ceil(gen / tpf) speculative steps (draft + verify-8 + inject), tpf from --tpf (a
            // projection of the acceptance measured elsewhere). --every N takes every Nth pair.
            std::ifstream in(argv[3]);
            std::vector<std::pair<int, int>> pairs;
            int a, b;
            while (in >> a >> b) pairs.push_back({ a, b });
            int every = 1;
            double tpf = 4.54;
            for (int i = 4; i < argc; ++i) {
                if (!strcmp(argv[i], "--every") && i + 1 < argc) every = atoi(argv[++i]);
                if (!strcmp(argv[i], "--tpf") && i + 1 < argc) tpf = atof(argv[++i]);
            }
            std::string text;
            for (int i = 0; i < 200; ++i) text += "so the plan is to move the pie torch loop onto flash attention and then profile it. ";
            const std::vector<int> base = E.tokenize(text);
            std::vector<double> walls, pre;
            double t_steps = 0;
            int skipped = 0, n_steps_total = 0;
            for (size_t r = 0; r < pairs.size(); r += every) {
                const int np = pairs[r].first, ng = pairs[r].second;
                if (np + ng + 16 > CTX || np > (int) base.size()) { ++skipped; continue; }
                // with a prefix cache, prompts start with the refiner's fixed system-prompt prefix, as real ones do
                std::vector<int> p(E.prefix.begin(), E.prefix.begin() + std::min<size_t>(E.prefix.size(), np));
                p.insert(p.end(), base.begin() + 1, base.begin() + 1 + (np - (int) p.size()));
                const double t0 = now();
                int last = E.prefill(p);
                const double t1 = now();
                int pos = np;
                if (!E.ND) {
                    for (int g = 1; g < ng; ++g) { E.T.step(&last, 1, 1, pos++); last = E.T.pick(1, 0); }
                } else {
                    const int steps = (int) ceil((ng - 1) / tpf);
                    std::vector<int> dr;
                    for (int s = 0; s < steps; ++s) {
                        E.ND->draft(last, pos, dr);
                        std::vector<int> rows = { last };
                        rows.insert(rows.end(), dr.begin(), dr.end());
                        E.T.step(rows.data(), 8, 8, pos);
                        last = E.T.pick(8, 0);
                        E.ND->inject_target(pos);
                        pos += (int) std::min(tpf, 8.0);  // positions advance like an accepted step (shape only)
                    }
                    n_steps_total += steps;
                }
                const double t2 = now();
                walls.push_back((t2 - t0) * 1e3);
                pre.push_back((t1 - t0) * 1e3);
                t_steps += t2 - t1;
            }
            const double nr = (double) walls.size();
            printf("reqbench %s: %d requests (%d over the context cap skipped) wall ms p50 %.0f p90 %.0f mean %.0f  "
                   "prefill p50 %.0f mean %.0f  decode/spec mean %.0f ms%s\n",
                   E.ND ? "spec (PROJECTION: forced tpf)" : "plain", (int) walls.size(), skipped, pct(walls, .5), pct(walls, .9),
                   std::accumulate(walls.begin(), walls.end(), 0.0) / nr, pct(pre, .5), std::accumulate(pre.begin(), pre.end(), 0.0) / nr,
                   t_steps * 1e3 / nr, E.ND ? "" : "");
            if (E.ND) printf("  tpf %.2f, %d steps, %.2f ms per step\n", tpf, n_steps_total, t_steps * 1e3 / std::max(1, n_steps_total));
        } else if (mode == "prof") {
            // GPU time of the stack (no head) for NB = 1 and 8 at position 250; with PE_ONLY, one phase kind x10
            std::vector<int> toks(8, 1000);
            for (int nb : { 1, 8 }) {
                double best = 1e9;
                for (int rep = 0; rep < 20; ++rep) {
                    E.T.step(toks.data(), nb, nb, 250, false);
                    if (rep > 2 && E.T.last_gpu_ms >= 0) best = std::min(best, E.T.last_gpu_ms);
                }
                printf("prof %s NB=%d: stack GPU %.3f ms (%u phases)\n", getenv("PE_ONLY") ? getenv("PE_ONLY") : "all", nb, best,
                       E.T.variant(nb).n_phases);
            }
        } else if (mode == "sessbench") {
            // session forward cost: n new tokens appended at a cached offset (rolled back to it before each rep), and
            // a correctness probe: greedy continuation by one-token forwards vs the same tokens re-verified in one
            // llama.cpp pass after a rollback
            std::string text;
            for (int i = 0; i < 40; ++i) {
                text += "so the plan is to move the pie torch loop onto flash attention, then run sweep " + std::to_string(i) +
                        " on the cluster before the review with " + (i % 2 ? "Dana" : "Marcus") + ". ";
            }
            std::vector<int> toks = E.tokenize(text);
            while ((int) toks.size() < CTX) toks.insert(toks.end(), toks.begin() + 1, toks.begin() + std::min((int) toks.size(), CTX));
            const std::vector<int> P(toks.begin(), toks.begin() + 52);
            bool cx = false;
            const char * path = "";
            double a = now();
            E.session_open(P);
            printf("sessbench: open (52-token prefix) %.2f ms\n", (now() - a) * 1e3);
            auto extend_to = [&](int off) {
                if ((int) E.s_toks.size() > off) E.session_rollback(off);
                while ((int) E.s_toks.size() < off) {
                    const int n0 = (int) E.s_toks.size(), m = std::min(512, off - n0);
                    E.session_forward(std::vector<int>(toks.begin() + n0, toks.begin() + n0 + m), m - 1, &cx, &path);
                }
            };
            const int offs[] = { 64, 256, 512, 1024, 1536 }, ns[] = { 1, 8, 32, 64, 128, 512 };
            printf("forward ms (min / mean of 3; rollback to the offset before each)\n%6s", "n\\off");
            for (int off : offs) printf("  %13d", off);
            printf("\n");
            for (int n : ns) {
                printf("%6d", n);
                for (int off : offs) {
                    if (off + std::max(n, MAX_NB) > CTX) { printf("  %13s", "-"); continue; }
                    extend_to(off);
                    double best = 1e9, sum = 0;
                    for (int r = 0; r < 4; ++r) {
                        E.session_rollback(off);
                        const std::vector<int> t(toks.begin() + off, toks.begin() + off + n);
                        a = now();
                        E.session_forward(t, n - 1, &cx, &path);
                        const double ms = (now() - a) * 1e3;
                        if (r > 0) { best = std::min(best, ms); sum += ms; }
                    }
                    printf("  %6.1f / %5.1f", best, sum / 3);
                }
                printf("   (%s)\n", n <= MAX_NB ? "megakernel" : "llama.cpp");
            }
            for (int n : { 64, 128, 512 }) {  // verification of a supplied draft: argmax for every row
                extend_to(512);
                double best = 1e9;
                for (int r = 0; r < 3; ++r) {
                    E.session_rollback(512);
                    const std::vector<int> t(toks.begin() + 512, toks.begin() + 512 + n);
                    a = now();
                    E.session_forward(t, 0, &cx, &path);
                    if (r > 0) best = std::min(best, (now() - a) * 1e3);
                }
                printf("forward %d tokens at 512 with argmax for all %d rows: %.1f ms\n", n, n, best);
            }
            // probe: greedy from the 300-token state by single-token forwards (megakernel), then a rollback and the
            // same continuation verified in one llama.cpp pass, then the same at the start of a rolled-back region
            extend_to(300);
            std::vector<int> g;
            int last = E.session_forward({ toks[300] }, 0, &cx, &path)[0];  // position 300 -> g[0]
            g.push_back(last);
            for (int i = 0; i < 47; ++i) {  // g[i] at position 301 + i -> g[i + 1]
                last = E.session_forward({ last }, 0, &cx, &path)[0];
                g.push_back(last);
            }
            E.session_rollback(301);
            std::vector<int> blk(g.begin(), g.end() - 1);  // feed g[0..46] at 301..347 -> argmax after each
            const std::vector<int> v = E.session_forward(blk, 0, &cx, &path);
            int same = 0;
            for (size_t i = 0; i < v.size(); ++i) same += v[i] == g[i + 1];
            printf("probe: 47 greedy tokens (megakernel, one at a time) re-verified in one %s pass after a rollback: %d/47 equal\n",
                   path, same);
            // rollback into the llama.cpp-written region and re-run on the megakernel
            E.session_rollback(301);
            int mk_same = 0;
            for (int i = 1; i < 48; ++i) mk_same += E.session_forward({ g[i - 1] }, 0, &cx, &path)[0] == g[i];
            printf("probe: the same continuation teacher-forced on the megakernel after the rollback: %d/47 equal\n", mk_same);
        } else if (mode == "gen") {
            const std::vector<int> p = E.tokenize(argv[3]);
            const std::vector<int> out = E.generate(p, 256);
            printf("%s\n", E.detok(out).c_str());
        } else if (mode == "serve") {
            // The Phonon refiner sidecar's engine: one JSON line in on stdin, one reply line out on stdout; EOF ends it.
            //   {"prefix": str}                   cache the prefix's state; its last token is left out (it may merge
            //                                     with the text after it), as in the sidecar's MLX prefix cache
            //   {"prompt": str, "max_tokens": N}  greedy, with the drafter if one was given ->
            //                                     {"text", "prompt_tokens", "tokens", "prefix_hit", "drafted", "accepted", "ms"}
            // Sessions (token ids in and out; the caller tokenizes; one live session, a new open replaces it, a one-shot
            // prompt request ends it):
            //   {"op": "open", "prefix": [ids]}                      -> {"ok", "session", "n", "ms"}
            //   {"op": "tokens", "session": s}                       -> {"ok", "n", "tokens": [ids]}
            //   {"op": "rollback", "session": s, "n": n}             -> {"ok", "n"}   keep positions 0..n-1
            //   {"op": "forward", "session": s, "tokens": [ids], "logits_from": j}
            //        -> {"ok", "ids": [argmax after tokens[j..k-1]], "n", "path": "mk" | "llama", "ms"[, "cancelled"]}
            //   {"op": "cancel", "session": s}   (read at once by the input thread; a running forward stops after its
            //                                     current pass) -> {"ok"}
            //   {"op": "close", "session": s}                        -> {"ok"}
            //   {"op": "tokenize", "text": str} -> {"ok", "ids"};  {"op": "detok", "ids": [ids]} -> {"ok", "text"}
            printf("{\"ready\": true, \"init_s\": %.3f}\n", now() - t0);
            fflush(stdout);
            std::mutex mu;
            std::condition_variable cv;
            std::deque<std::string> q;
            bool eof = false;
            std::thread reader([&] {
                std::string l;
                while (std::getline(std::cin, l)) {
                    if (l.find("\"cancel\"") != std::string::npos) E.s_cancel = true;
                    { std::lock_guard<std::mutex> g(mu); q.push_back(l); }
                    cv.notify_one();
                }
                { std::lock_guard<std::mutex> g(mu); eof = true; }
                cv.notify_one();
            });
            auto reply = [](const std::string & s) { fputs(s.c_str(), stdout); fputc('\n', stdout); fflush(stdout); };
            auto err = [&](const char * m) { reply(std::string("{\"ok\": false, \"error\": \"") + m + "\"}"); };
            std::string line;
            for (;;) {
                {
                    std::unique_lock<std::mutex> g(mu);
                    cv.wait(g, [&] { return !q.empty() || eof; });
                    if (q.empty()) break;
                    line = q.front();
                    q.pop_front();
                }
                if (line.empty()) continue;
                const std::string op = read_str(line, "\"op\"");
                if (!op.empty()) {
                    const double a = now();
                    const long sid = read_int(line, "\"session\"", 0);
                    char buf[256];
                    if (op == "cancel") { E.s_cancel = false; reply("{\"ok\": true}"); continue; }
                    if (op == "tokenize") {  // helpers for callers and tests: {"text": str} -> ids, {"ids": [..]} -> text
                        reply("{\"ok\": true, \"ids\": " + ints_json(E.tokenize(read_str(line, "\"text\""))) + "}");
                        continue;
                    }
                    if (op == "detok") {
                        reply("{\"ok\": true, \"text\": " + json_str(E.detok(read_ints(line, "\"ids\""))) + "}");
                        continue;
                    }
                    if (op == "open") {
                        const std::vector<int> p = read_ints(line, "\"prefix\"");
                        if (p.empty() || (int) p.size() + 16 > CTX) { err("prefix empty or too long"); continue; }
                        const int s = E.session_open(p);
                        snprintf(buf, sizeof buf, "{\"ok\": true, \"session\": %d, \"n\": %zu, \"ms\": %.2f}", s, E.s_toks.size(), (now() - a) * 1e3);
                        reply(buf);
                        continue;
                    }
                    if (sid == 0 || sid != E.s_id) { err("no such session"); continue; }
                    if (op == "close") { E.s_id = 0; E.s_toks.clear(); reply("{\"ok\": true}"); continue; }
                    if (op == "tokens") {
                        reply("{\"ok\": true, \"n\": " + std::to_string(E.s_toks.size()) + ", \"tokens\": " + ints_json(E.s_toks) + "}");
                        continue;
                    }
                    if (op == "rollback") {
                        if (!E.session_rollback((int) read_int(line, "\"n\"", -1))) { err("bad rollback length"); continue; }
                        reply("{\"ok\": true, \"n\": " + std::to_string(E.s_toks.size()) + "}");
                        continue;
                    }
                    if (op == "forward") {
                        const std::vector<int> t = read_ints(line, "\"tokens\"");
                        const int from = (int) read_int(line, "\"logits_from\"", (long) t.size() - 1);
                        if (t.empty() || from < 0 || from > (int) t.size()) { err("bad tokens or logits_from"); continue; }
                        // (2..8 tokens within 8 of the cap go through llama.cpp: the verify shape writes 8 rows)
                        if ((int) E.s_toks.size() + (int) t.size() > CTX) { err("context cap reached"); continue; }
                        bool cancelled = false;
                        const char * path = "";
                        const std::vector<int> ids = E.session_forward(t, from, &cancelled, &path);
                        snprintf(buf, sizeof buf, "\"n\": %zu, \"path\": \"%s\", \"ms\": %.2f%s}", E.s_toks.size(), path, (now() - a) * 1e3,
                                 cancelled ? ", \"cancelled\": true" : "");
                        reply("{\"ok\": " + std::string(cancelled ? "false" : "true") + ", \"ids\": " + ints_json(ids) + ", " + buf);
                        continue;
                    }
                    err("unknown op");
                    continue;
                }
                E.s_id = 0;  // a one-shot request below reuses the engine's state
                if (line.rfind("{\"prefix\"", 0) == 0) {
                    std::vector<int> p = E.tokenize(read_str(line, "\"prefix\""));
                    if (!p.empty()) p.pop_back();
                    E.set_prefix(p);
                    printf("{\"ok\": true, \"prefix_tokens\": %zu}\n", p.size());
                    fflush(stdout);
                    continue;
                }
                const size_t k = line.find("\"max_tokens\"");
                const int cap = k == std::string::npos ? 256 : atoi(line.c_str() + line.find(':', k) + 1);
                const long d0 = E.n_drafted, a0 = E.n_accepted;
                const int h0 = E.n_prefix_hits;
                const double a = now();
                const std::vector<int> p = E.tokenize(read_prompt(line));
                const std::vector<int> out = E.generate(p, cap);
                const std::string text = E.detok(out);
                printf("{\"ok\": true, \"text\": %s, \"prompt_tokens\": %zu, \"tokens\": %zu, \"prefix_hit\": %s, "
                       "\"drafted\": %ld, \"accepted\": %ld, \"ms\": %.2f}\n",
                       json_str(text).c_str(), p.size(), out.size(), E.n_prefix_hits > h0 ? "true" : "false",
                       E.n_drafted - d0, E.n_accepted - a0, (now() - a) * 1e3);
                fflush(stdout);
            }
            reader.join();
        } else if (mode == "run" || mode == "logits") {
            std::ifstream in(argv[3]);
            std::string line;
            std::vector<std::string> prompts;
            std::vector<int> caps;  // per-request max_tokens ("max_tokens" in the line, default 256)
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                prompts.push_back(read_prompt(line));
                const size_t k = line.find("\"max_tokens\"");
                caps.push_back(k == std::string::npos ? 256 : atoi(line.c_str() + line.find(':', k) + 1));
            }
            if (mode == "logits") {
                // logit probe: after a llama.cpp prefill of each prompt, the same 8 rows (first generated token +
                // the prompt's first 7 tokens) through (a) llama.cpp one row at a time, (b) llama.cpp as one 8-row
                // batch, (c) the megakernel NB=1 for row 0 and NB=8 for all rows. Reports max |diff| of the
                // logits and argmax agreement; (a) vs (b) is the incumbent's own decode-vs-verify noise (the budget).
                const int n = std::min((int) prompts.size(), atoi(argv[4]));
                const int V = E.T.n_vocab;
                double d_ab = 0, d_ac1 = 0, d_ac8 = 0, d_bc8 = 0;
                int agree_ab = 0, agree_ac1 = 0, agree_ac8 = 0, rows = 0;
                std::vector<float> la((size_t) 8 * V), lb((size_t) 8 * V);
                auto maxdiff = [&](const float * x, const float * y) {
                    double m = 0;
                    for (int j = 0; j < V; ++j) m = std::max(m, (double) fabsf(x[j] - y[j]));
                    return m;
                };
                for (int i = 0; i < n; ++i) {
                    std::vector<int> p = E.tokenize(prompts[i]);
                    const int np = (int) p.size();
                    std::vector<int> r8 = { E.prefill(p) };
                    for (int j = 0; j < 7; ++j) r8.push_back(p[j + 1]);
                    // (c) megakernel
                    E.T.step(r8.data(), 1, 1, np);
                    std::vector<float> c1(E.T.logits(1), E.T.logits(1) + V);
                    E.T.step(r8.data(), 8, 8, np);
                    std::vector<float> c8(E.T.logits(8), E.T.logits(8) + (size_t) 8 * V);
                    // (a) llama.cpp one row at a time
                    for (int b = 0; b < 8; ++b) {
                        E.pbatch.clear();
                        E.pbatch.add(r8[b], np + b, 0, true);
                        llama_process(E.lctx, LLAMA_PROCESS_TYPE_DECODE, E.pbatch.get());
                        memcpy(la.data() + (size_t) b * V, llama_get_logits_ith(E.lctx, -1), V * sizeof(float));
                    }
                    // (b) llama.cpp 8-row batch (the conv state cannot roll back: prefill again)
                    E.prefill(p);
                    E.pbatch.clear();
                    for (int b = 0; b < 8; ++b) E.pbatch.add(r8[b], np + b, 0, true);
                    llama_process(E.lctx, LLAMA_PROCESS_TYPE_DECODE, E.pbatch.get());
                    for (int b = 0; b < 8; ++b) memcpy(lb.data() + (size_t) b * V, llama_get_logits_ith(E.lctx, b), V * sizeof(float));
                    d_ac1 = std::max(d_ac1, maxdiff(la.data(), c1.data()));
                    agree_ac1 += argmax(la.data(), V) == argmax(c1.data(), V);
                    for (int b = 0; b < 8; ++b) {
                        const float * A = la.data() + (size_t) b * V, * B = lb.data() + (size_t) b * V, * C = c8.data() + (size_t) b * V;
                        d_ab = std::max(d_ab, maxdiff(A, B));
                        d_ac8 = std::max(d_ac8, maxdiff(A, C));
                        d_bc8 = std::max(d_bc8, maxdiff(B, C));
                        agree_ab += argmax(A, V) == argmax(B, V);
                        agree_ac8 += argmax(A, V) == argmax(C, V);
                        ++rows;
                    }
                }
                printf("logit probe, %d prompts: max|diff| llama decode vs llama batch-8 %.4f (argmax %d/%d) | megakernel NB=1 vs "
                       "llama decode %.4f (argmax %d/%d) | megakernel NB=8 vs llama decode %.4f (argmax %d/%d), vs llama batch-8 %.4f\n",
                       n, d_ab, agree_ab, rows, d_ac1, agree_ac1, n, d_ac8, agree_ac8, rows, d_bc8);
                E.shutdown();
                return 0;
            }
            std::vector<double> walls;
            std::vector<std::string> texts;
            long ntok = 0, steps0 = 0;
            const int n = std::min((int) prompts.size(), n_rows);
            // PE_ROWS=FILE: per request "index, wall ms, prompt tokens, output tokens, speculative steps, prefill ms"
            FILE * rows_f = getenv("PE_ROWS") ? fopen(getenv("PE_ROWS"), "w") : nullptr;
            for (int i = 0; i < n; ++i) {
                if (i == warm) {
                    E.n_drafted = E.n_accepted = E.n_steps = 0;
                    E.n_prefix_hits = 0;
                    if (E.D) E.D->t_proc_draft = E.D->t_proc_add = E.D->t_wait_draft = E.D->t_gpu_draft = 0;
                    if (E.ND) E.ND->t_gpu = E.ND->t_walk = 0;
                    E.t_prefill = E.t_decode = E.t_draft = E.t_inject = E.t_handoff = 0;
                }
                const double a = now(), tp0 = E.t_prefill;
                const std::vector<int> p = E.tokenize(prompts[i]);
                const std::vector<int> out = E.generate(p, caps[i]);
                texts.push_back(E.detok(out));
                const double ms = (now() - a) * 1e3;
                if (i >= warm) { walls.push_back(ms); ntok += (long) out.size(); }
                if (rows_f) fprintf(rows_f, "%d\t%.2f\t%zu\t%zu\t%ld\t%.2f\n", i, ms, p.size(), out.size(), E.n_steps - steps0, (E.t_prefill - tp0) * 1e3);
                steps0 = E.n_steps;
            }
            const double nr = (double) walls.size();
            printf("rows %d wall ms p50 %.0f p90 %.0f mean %.1f  out tok mean %.1f  drafts accepted %ld/%ld (%.1f tok/step)"
                   "  per row ms: prefill %.1f handoff %.2f decode %.1f draft %.1f inject %.1f  prefix hits %d  llama.cpp fallbacks %d\n",
                   (int) walls.size(), pct(walls, .5), pct(walls, .9), std::accumulate(walls.begin(), walls.end(), 0.0) / nr, ntok / nr,
                   E.n_accepted, E.n_drafted, E.n_steps ? (double) (E.n_accepted + E.n_steps) / E.n_steps : 0.0,
                   E.t_prefill * 1e3 / nr, E.t_handoff * 1e3 / nr, E.t_decode * 1e3 / nr, E.t_draft * 1e3 / nr, E.t_inject * 1e3 / nr,
                   E.n_prefix_hits, E.n_fallback);
            if (E.D) {
                printf("drafter per row ms: llama_process draft %.2f (until lattice %.2f, GPU %.2f), inject %.2f (%.2f steps/row)\n",
                       E.D->t_proc_draft * 1e3 / nr, E.D->t_wait_draft * 1e3 / nr, E.D->t_gpu_draft * 1e3 / nr,
                       E.D->t_proc_add * 1e3 / nr, E.n_steps / nr);
                if (getenv("PE_DRAFT_PROFILE")) E.D->prof.print(E.D->n_prof_calls);
            }
            if (E.ND) {
                printf("native drafter per row: GPU (draft + injections) %.2f ms, CPU selector walk %.2f ms (%.2f steps/row)\n",
                       E.ND->t_gpu * 1e3 / nr, E.ND->t_walk * 1e3 / nr, E.n_steps / nr);
            }
            if (rows_f) fclose(rows_f);
            if (out_path) {
                FILE * f = fopen(out_path, "w");
                fprintf(f, "[");
                for (size_t i = 0; i < texts.size(); ++i) fprintf(f, "%s%s", i ? ", " : "", json_str(texts[i]).c_str());
                fprintf(f, "]\n");
                fclose(f);
            }
        }
        E.shutdown();
    }
    return 0;
}
