// pa: Parakeet TDT 0.6B v2 encoder on Metal (pa.metal): mel [Tm][128] -> encoder output [T][1024], one command buffer.
//   pa WEIGHTS.bin MEL.npy OUT.npy [reps]      (WEIGHTS.idx next to WEIGHTS.bin: name offset dims, from ref.py export)
// env: PA_DIR (where pa.metal is; default the binary's directory)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Lin { uint32_t x, y, r, w, b, rows, n, k, act; float alpha; uint32_t res, ksplit, qoff, qsb, qg; };
struct Norm { uint32_t x, y, w, b, rows, d; float eps; uint32_t yh; };
struct Att { uint32_t q, k, v, p, o, u, vb, T, d, heads; float scale; uint32_t ld, Tv, Tp; };
struct Conv { uint32_t g, y, w, bnw, bnb, bnm, bnv, T, d; float eps; };
struct Sub { uint32_t x, y, w, b, Ti, Fi, To, Fo, C, Tvalid, mode; };
struct MelP { uint32_t wav, n, pow, lmel, mel, fb, tw, frames; };
struct Tdt { uint32_t E, T, S, out, maxout, emb, wx0, wh0, b0, wx1, wh1, b1, pw, pb, jw, jb, max_symbols, dbg, dbg_step, dbgl, xtab,
             trie, spans, maxspans; };
// pa.metal's TdtCtl: the decode's control block at the start of the state (tdt_decide writes it between dispatches)
struct TdtCtl {
    uint32_t lstm_args[3], joint_args[3], decide_args[3];  // the next slot's indirect dispatch sizes
    uint32_t done, step, need, tok, run, iter, nout, new_symbols, plain, boost_live, node, ck_step, ck_nout, nspans, n_flips, n_rollbacks;
    int32_t no_boost_until;
    uint32_t changed, dumped, base, nf, slots, pad;  // slots: tdt_decide runs (the slots the decode used)
    int64_t deficit;
};
static_assert(sizeof(TdtCtl) == 136 && offsetof(TdtCtl, deficit) == 128, "TdtCtl layout must match pa.metal");
// decode geometry, as pa.metal's: LSTM units per threadgroup (HD / TDT_LNU threadgroups), frames per joint batch,
// joint row slices, uints per slice partial; the state after the 64-word control block (see pa.metal ST_*)
static const uint32_t TDT_LNU = 8, TDT_KF = 16, TDT_NSL = 20, TDT_PSTRIDE = 12;

static std::vector<double> json_nums(const std::string & s, const char * key) {  // "key": [1, 2.5, ...]
    std::vector<double> v;
    size_t a = s.find(std::string("\"") + key + "\"");
    if (a == std::string::npos) return v;
    a = s.find('[', a);
    const size_t b = s.find(']', a);
    std::stringstream ss(s.substr(a + 1, b - a - 1));
    std::string tok;
    while (std::getline(ss, tok, ',')) if (tok.find_first_of("0123456789") != std::string::npos) v.push_back(std::stod(tok));
    return v;
}

static std::vector<float> load_npy(const char * path, std::vector<size_t> & shape) {
    std::ifstream f(path, std::ios::binary);
    char magic[6];
    f.read(magic, 6);
    uint8_t ver[2];
    f.read((char *) ver, 2);
    uint32_t hl = 0;
    if (ver[0] == 1) { uint16_t h; f.read((char *) &h, 2); hl = h; } else f.read((char *) &hl, 4);
    std::string hdr(hl, ' ');
    f.read(&hdr[0], hl);
    if (hdr.find("'<f4'") == std::string::npos) { fprintf(stderr, "%s: float32 npy expected\n", path); exit(1); }
    const size_t a = hdr.find('(') + 1, b = hdr.find(')');
    std::stringstream ss(hdr.substr(a, b - a));
    std::string tok;
    shape.clear();
    size_t n = 1;
    while (std::getline(ss, tok, ',')) if (!tok.empty() && tok.find_first_not_of(' ') != std::string::npos) { shape.push_back(std::stoul(tok)); n *= shape.back(); }
    std::vector<float> v(n);
    f.read((char *) v.data(), n * 4);
    return v;
}

static void save_npy(const char * path, const float * data, size_t rows, size_t cols) {
    char hdr[128];
    int n = snprintf(hdr, sizeof hdr, "{'descr': '<f4', 'fortran_order': False, 'shape': (%zu, %zu), }", rows, cols);
    std::string h(hdr, n);
    while ((10 + h.size() + 1) % 64) h += ' ';
    h += '\n';
    FILE * f = fopen(path, "wb");
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    const uint16_t hl = (uint16_t) h.size();
    fwrite(&hl, 2, 1, f);
    fwrite(h.data(), 1, h.size(), f);
    fwrite(data, 4, rows * cols, f);
    fclose(f);
}

static float bf16_round(float x) {  // round to nearest even bf16 (the MLX model dtype), as float
    uint32_t u;
    memcpy(&u, &x, 4);
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u;
    memcpy(&x, &u, 4);
    return x;
}

// model.safetensors -> one shared fp16 buffer: names -> half offsets (256-byte aligned tensors; each layer's
// linear_q, linear_k, linear_v back to back, as the fused q / k / v projection reads them). F32 (rounded to bf16
// first, the app's MLX dtype), BF16 and F16 sources.
static bool load_safetensors(const char * path, id<MTLDevice> dev, __strong id<MTLBuffer> & wbuf, std::map<std::string, uint32_t> & woff,
                             size_t & wsize) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path); return false; }
    const size_t fsize = (size_t) lseek(fd, 0, SEEK_END);
    const uint8_t * map = (const uint8_t *) mmap(nullptr, fsize, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (map == MAP_FAILED || fsize < 8) { fprintf(stderr, "cannot map %s\n", path); return false; }
    uint64_t hlen;
    memcpy(&hlen, map, 8);
    if (hlen + 8 > fsize) { fprintf(stderr, "%s: bad header\n", path); return false; }
    const std::string h((const char *) map + 8, hlen);
    const uint8_t * data = map + 8 + hlen;
    struct T { std::string name; int type; size_t a, b; };  // type 0 F32, 1 BF16, 2 F16
    std::vector<T> ts;
    // the header is {"name": {"dtype": "...", "shape": [...], "data_offsets": [a, b]}, ..., "__metadata__": {...}}
    size_t p = h.find('{') + 1;
    while (true) {
        const size_t q0 = h.find('"', p);
        if (q0 == std::string::npos) break;
        const size_t q1 = h.find('"', q0 + 1);
        const std::string name = h.substr(q0 + 1, q1 - q0 - 1);
        const size_t ob = h.find('{', q1), oe = h.find('}', ob);
        const std::string body = h.substr(ob, oe - ob);
        p = oe + 1;
        if (name == "__metadata__") continue;
        const size_t dt = body.find("\"dtype\""), d0 = body.find('"', dt + 7) + 1, d1 = body.find('"', d0);
        const std::string dtype = body.substr(d0, d1 - d0);
        const size_t of = body.find("\"data_offsets\""), lb = body.find('[', of);
        T t;
        t.name = name;
        t.type = dtype == "F32" ? 0 : dtype == "BF16" ? 1 : dtype == "F16" ? 2 : -1;
        if (t.type < 0) { fprintf(stderr, "%s: tensor %s has dtype %s\n", path, name.c_str(), dtype.c_str()); return false; }
        const size_t lbh = ob + lb;  // the '[' of data_offsets, in the whole header
        t.a = strtoull(h.c_str() + lbh + 1, nullptr, 10);
        t.b = strtoull(h.c_str() + h.find(',', lbh) + 1, nullptr, 10);
        ts.push_back(t);
    }
    std::sort(ts.begin(), ts.end(), [](const T & x, const T & y) { return x.name < y.name; });
    std::map<std::string, size_t> at;
    for (size_t i = 0; i < ts.size(); ++i) at[ts[i].name] = i;
    std::vector<size_t> order;
    std::vector<bool> placed(ts.size(), false);
    for (size_t i = 0; i < ts.size(); ++i) {
        if (placed[i]) continue;
        const std::string & n = ts[i].name;
        const size_t qp = n.find("linear_q.weight");
        if (qp != std::string::npos || n.find("linear_k.weight") != std::string::npos || n.find("linear_v.weight") != std::string::npos) {
            const std::string base = n.substr(0, n.find("linear_"));
            for (const char * s : { "linear_q.weight", "linear_k.weight", "linear_v.weight" }) {
                auto it = at.find(base + s);
                if (it != at.end() && !placed[it->second]) { order.push_back(it->second); placed[it->second] = true; }
            }
            continue;
        }
        order.push_back(i);
        placed[i] = true;
    }
    size_t off = 0;  // in halves
    std::vector<size_t> dst(ts.size());
    for (size_t i : order) {
        dst[i] = off;
        const size_t n = (ts[i].b - ts[i].a) / (ts[i].type == 0 ? 4 : 2);
        off += (n * 2 + 255) / 256 * 128;  // halves, 256-byte aligned (q / k / v sizes are multiples of 256 bytes)
    }
    wsize = off * 2;
    wbuf = [dev newBufferWithLength:wsize options:MTLResourceStorageModeShared];
    if (!wbuf) { fprintf(stderr, "cannot allocate %zu bytes for the weights\n", wsize); return false; }
    _Float16 * out = (_Float16 *) wbuf.contents;
    dispatch_apply(ts.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
        const T & t = ts[i];
        const uint8_t * src = data + t.a;
        _Float16 * o = out + dst[i];
        if (t.type == 0) {
            const size_t n = (t.b - t.a) / 4;
            for (size_t j = 0; j < n; ++j) {
                uint32_t u;
                memcpy(&u, src + 4 * j, 4);
                u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;  // nearest-even bf16
                float f;
                memcpy(&f, &u, 4);
                o[j] = (_Float16) f;
            }
        } else if (t.type == 1) {
            const size_t n = (t.b - t.a) / 2;
            for (size_t j = 0; j < n; ++j) {
                uint16_t b;
                memcpy(&b, src + 2 * j, 2);
                const uint32_t u = (uint32_t) b << 16;
                float f;
                memcpy(&f, &u, 4);
                o[j] = (_Float16) f;
            }
        } else {
            memcpy(o, src, t.b - t.a);
        }
    });
    for (size_t i = 0; i < ts.size(); ++i) woff[ts[i].name] = (uint32_t) dst[i];
    munmap((void *) map, fsize);
    return true;
}

// `pa busy [TGS [SECS]]`: a GPU load in its own process for the sidecar's startup self-test, standing in for the
// refiner's persistent kernel (its threadgroups spin on the GPU during a dictation): TGS (16, pe's grid on a base
// M4) threadgroups of up to 1024 threads that spin on ALU work, dispatched back to back (two in flight) for SECS (10)
// seconds or until killed. Prints "busy: running" once the first dispatch has run. (A second queue in the pa
// process itself is not a load: Metal runs it before or after pa's own work, never beside it.)
static int busy_main(int argc, char ** argv) {
    const int tgs = argc > 2 ? atoi(argv[2]) : 16;
    const double secs = argc > 3 ? atof(argv[3]) : 10.0;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSString * src = @"#include <metal_stdlib>\nusing namespace metal;\n"
                      "kernel void busy(device uint * F [[buffer(0)]], uint tid [[thread_index_in_threadgroup]]) {\n"
                      "    float x = (float) tid;\n"
                      "    for (uint i = 0; i < (1u << 18); ++i) x = fma(x, 0.999f, 1.f);\n"
                      "    if (x == 12345.f) F[0] = 1;\n"
                      "}\n";
    NSError * err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
    id<MTLComputePipelineState> p = lib ? [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"busy"] error:&err] : nil;
    if (!p) { fprintf(stderr, "busy: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLBuffer> f = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    const NSUInteger threads = std::min<NSUInteger>(1024, p.maxTotalThreadsPerThreadgroup);
    const double t0 = now();
    id<MTLCommandBuffer> prev = nil;
    bool said = false;
    while (now() - t0 < secs) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:f offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [e endEncoding];
        [cb commit];
        if (prev) {
            [prev waitUntilCompleted];
            if (!said) { printf("busy: running\n"); fflush(stdout); said = true; }
        }
        prev = cb;
    }
    [prev waitUntilCompleted];
    return 0;
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc >= 2 && !strcmp(argv[1], "busy")) return busy_main(argc, argv);
        if (argc < 4) { fprintf(stderr, "usage: pa WEIGHTS.bin MEL.npy OUT.npy [reps [bucket]] | pa WEIGHTS.bin serve MAXMEL | pa busy [TGS [SECS]]\n"); return 1; }
        const int reps = argc > 4 ? atoi(argv[4]) : 5;
        // PA_BF16=1: mimic an all-bf16 MLX run (bf16 mel) instead of the app's path (float32 mel, so float32
        // activations against bf16 weights: asr_server.transcribe_samples)
        const int bf16_run = getenv("PA_BF16") ? atoi(getenv("PA_BF16")) : 0;
        // weights: WEIGHTS.bin (+ .idx, from ref.py export: fp16, mmapped) or the Hugging Face snapshot's
        // model.safetensors itself (mlx-community/parakeet-tdt-0.6b-v2: 697 float32 tensors, the MLX parameter
        // names), converted at load exactly as the export does: rounded to bf16 (the app's MLX dtype), stored fp16
        std::map<std::string, uint32_t> woff;  // half offsets
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        id<MTLBuffer> wbuf = nil;
        void * wptr = nullptr;
        size_t wsize = 0;
        const size_t page = getpagesize();
        const std::string wpath = argv[1];
        if (wpath.size() > 12 && wpath.substr(wpath.size() - 12) == ".safetensors") {
            const double tl = now();
            if (!load_safetensors(wpath.c_str(), dev, wbuf, woff, wsize)) return 1;
            wptr = wbuf.contents;
            if (getenv("PA_VERBOSE")) fprintf(stderr, "pa: %zu tensors from %s, %.0f MB fp16 in %.2f s\n", woff.size(), wpath.c_str(), wsize / 1e6, now() - tl);
        } else {
            std::string ip = wpath.substr(0, wpath.size() - 4) + ".idx";
            std::ifstream f(ip);
            std::string line;
            while (std::getline(f, line)) {
                std::stringstream ss(line);
                std::string name;
                size_t off;
                ss >> name >> off;
                woff[name] = (uint32_t) (off / 2);
            }
            const int fd = open(argv[1], O_RDONLY);
            if (fd < 0) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
            wsize = (size_t) lseek(fd, 0, SEEK_END);
            const size_t wmap = (wsize + page - 1) / page * page;
            wptr = mmap(nullptr, wmap, PROT_READ, MAP_SHARED, fd, 0);
            wbuf = [dev newBufferWithBytesNoCopy:wptr length:wmap options:MTLResourceStorageModeShared deallocator:nil];
        }
        auto W = [&](const std::string & n) -> uint32_t {
            auto it = woff.find(n);
            if (it == woff.end()) { fprintf(stderr, "missing weight %s\n", n.c_str()); exit(1); }
            return it->second;
        };
        if (getenv("PA_CHECK_WEIGHTS")) {  // the converted weights against an exported .bin / .idx: bitwise, per tensor
            const std::string cb = getenv("PA_CHECK_WEIGHTS");
            std::ifstream f(cb.substr(0, cb.size() - 4) + ".idx");
            const int cfd = open(cb.c_str(), O_RDONLY);
            const size_t csz = (size_t) lseek(cfd, 0, SEEK_END);
            const uint16_t * c = (const uint16_t *) mmap(nullptr, csz, PROT_READ, MAP_SHARED, cfd, 0);
            std::string line;
            size_t n = 0, bad = 0, elems = 0;
            while (std::getline(f, line)) {
                std::stringstream ss(line);
                std::string name;
                size_t off, cnt = 1, d;
                ss >> name >> off;
                while (ss >> d) cnt *= d;
                const uint16_t * a = (const uint16_t *) wptr + W(name), * b = c + off / 2;
                bad += memcmp(a, b, cnt * 2) != 0;
                ++n;
                elems += cnt;
            }
            printf("weights check: %zu tensors (%zu values), %zu differ from %s\n", n, elems, bad, cb.c_str());
            return bad != 0;
        }

        // weight-only quantization of the encoder GEMMs (PA_Q): q8 (int8 per output channel), q8g64, q4g32, q4g64,
        // mix32 / mix64 (int4 g32 / g64, int8 per channel for the subsampling output projection, blocks 0 and 23 and
        // the attention projections). Affine per group: w = s * q + b, (s, b) half2. Kept fp16: linear_pos (cached
        // per T), the 1x1 subsampling convs, joint.enc and the decoder. Built once, cached as data/pk.<variant>.qbin.
        struct QEnt { uint32_t qoff, qsb, bits, g; };
        std::map<uint32_t, QEnt> qmap;  // half offset of the (first) fp16 tensor -> quantized copy
        id<MTLBuffer> qbuf = nil;
        size_t qbytes = 0, qsrc_bytes = 0;  // quantized blob size, fp16 bytes it replaces
        const std::string qv = getenv("PA_Q") ? getenv("PA_Q") : "";
        if (!qv.empty()) {
            struct G { std::string name; uint32_t n, k; int bits, g; };
            // ff32 / ff64: int4 g32 / g64 for the feed-forward linears of blocks 1-22 only, int8 g64 for the rest
            const bool mix = qv.rfind("mix", 0) == 0, ffv = qv.rfind("ff", 0) == 0;
            const int b_main = qv[1] == '8' ? 8 : 4,
                      g_main = mix ? atoi(qv.c_str() + 3) : ffv ? atoi(qv.c_str() + 2) : qv.size() > 2 ? atoi(qv.c_str() + 3) : 0;
            std::vector<G> gl;
            auto add = [&](const std::string & nm, uint32_t n, uint32_t k, bool exc) {
                if (ffv) {
                    const bool ff4 = !exc && nm.find("feed_forward") != std::string::npos;
                    gl.push_back({ nm, n, k, ff4 ? 4 : 8, ff4 ? g_main : 64 });
                } else if (mix && exc) gl.push_back({ nm, n, k, 8, 0 });
                else gl.push_back({ nm, n, k, mix ? 4 : b_main, g_main });
            };
            add("encoder.pre_encode.out.weight", 1024, 4096, true);
            for (int l = 0; l < 24; ++l) {
                const std::string L = "encoder.layers." + std::to_string(l) + ".";
                const bool edge = l == 0 || l == 23;
                add(L + "feed_forward1.linear1.weight", 4096, 1024, edge);
                add(L + "feed_forward1.linear2.weight", 1024, 4096, edge);
                add(L + "self_attn.linear_q.weight", 3072, 1024, true);  // q, k, v contiguous: one [3072, 1024] tensor
                add(L + "self_attn.linear_out.weight", 1024, 1024, true);
                add(L + "conv.pointwise_conv1.weight", 2048, 1024, edge);
                add(L + "conv.pointwise_conv2.weight", 1024, 1024, edge);
                add(L + "feed_forward2.linear1.weight", 4096, 1024, edge);
                add(L + "feed_forward2.linear2.weight", 1024, 4096, edge);
            }
            size_t off = 0;
            for (G & g : gl) {
                if (g.g == 0) g.g = (int) g.k;
                QEnt q;
                q.bits = (uint32_t) g.bits; q.g = (uint32_t) g.g;
                q.qoff = (uint32_t) off; off += ((size_t) g.n * g.k * g.bits / 8 + 255) / 256 * 256;
                q.qsb = (uint32_t) off; off += ((size_t) g.n * (g.k / g.g) * 4 + 255) / 256 * 256;
                qmap[W(g.name)] = q;
                qsrc_bytes += (size_t) g.n * g.k * 2;
            }
            qbytes = off;
            const std::string qp = std::string(argv[1]).substr(0, std::string(argv[1]).size() - 4) + "." + qv + ".qbin";
            const size_t qmap_len = (qbytes + page - 1) / page * page;
            int qfd = open(qp.c_str(), O_RDONLY);
            if (qfd < 0 || (size_t) lseek(qfd, 0, SEEK_END) != qbytes) {
                if (qfd >= 0) close(qfd);
                const double tq = now();
                std::vector<uint8_t> blob(qbytes, 0);
                const _Float16 * src = (const _Float16 *) wptr;
                for (const G & g : gl) {
                    const QEnt q = qmap[W(g.name)];
                    const _Float16 * w = src + W(g.name);
                    uint8_t * qd = blob.data() + q.qoff;
                    _Float16 * sb = (_Float16 *) (blob.data() + q.qsb);
                    const int levels = (1 << g.bits) - 1, ng = (int) (g.k / q.g);
                    dispatch_apply(g.n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t n) {
                        for (int j = 0; j < ng; ++j) {
                            const _Float16 * v = w + n * g.k + (size_t) j * q.g;
                            float mn = 1e30f, mx = -1e30f;
                            for (uint32_t i = 0; i < q.g; ++i) { mn = std::min(mn, (float) v[i]); mx = std::max(mx, (float) v[i]); }
                            const _Float16 hs = (_Float16) (mx > mn ? (mx - mn) / levels : 1.f), hb = (_Float16) mn;
                            sb[2 * (n * ng + j)] = hs;
                            sb[2 * (n * ng + j) + 1] = hb;
                            for (uint32_t i = 0; i < q.g; ++i) {
                                const int c = std::min(levels, std::max(0, (int) lrintf(((float) v[i] - (float) hb) / (float) hs)));
                                const size_t e = n * g.k + (size_t) j * q.g + i;
                                if (g.bits == 8) qd[e] = (uint8_t) c;
                                else qd[e / 2] |= (uint8_t) (c << (4 * (e % 2)));
                            }
                        }
                    });
                }
                FILE * f = fopen(qp.c_str(), "wb");
                fwrite(blob.data(), 1, qbytes, f);
                fclose(f);
                fprintf(stderr, "quantized %s in %.1f s -> %s\n", qv.c_str(), now() - tq, qp.c_str());
                qfd = open(qp.c_str(), O_RDONLY);
            }
            void * qptr = mmap(nullptr, qmap_len, PROT_READ, MAP_SHARED, qfd, 0);
            qbuf = [dev newBufferWithBytesNoCopy:qptr length:qmap_len options:MTLResourceStorageModeShared deallocator:nil];
        }

        // input: one window from MEL.npy (CLI), or `pa serve WEIGHTS.bin MAXMEL` (windows of up to MAXMEL mel frames over
        // stdin / stdout, see serve below). The arena is sized for the capacity; every per-window size is set by
        // set_window, which the encode lambda reads by reference.
        const bool serve = !strcmp(argv[2], "serve");
        std::vector<size_t> shp;
        std::vector<float> mel;
        uint32_t cap_mel;
        const uint32_t F0 = 128;
        // a 1-D npy (CLI) or a samples request (serve) is 16 kHz float32 audio: the log-mel runs on the GPU (melpow /
        // melfb / melnorm) in the same command buffer; a 2-D npy / mel request is get_logmel's output
        bool gpu_mel = false, have_fb = false;  // have_fb: melfb.npy loaded (audio requests need it)
        uint32_t nwav = 0;
        if (serve) cap_mel = (uint32_t) atoi(argv[3]);
        else {
            mel = load_npy(argv[2], shp);
            if (shp.size() == 1) {
                gpu_mel = true;
                nwav = (uint32_t) shp[0];
                cap_mel = (nwav + 272) / 160;
            } else {
                if (shp[1] != F0) { fprintf(stderr, "mel: %zu features, expected 128\n", shp[1]); return 1; }
                cap_mel = (uint32_t) shp[0];
            }
        }
        auto sub_len = [](uint32_t t) { return (t + 2 - 3) / 2 + 1; };
        // valid lengths of the window, and the (optional, argv[5]) padded bucket of T frames: mel padded with zeros to
        // 8 T frames, every cross-frame op bounded by the valid length
        const int barg = !serve && argc > 5 ? atoi(argv[5]) : 0;  // > 0: bucket of barg frames; < 0: Tv rounded up to a multiple of -barg
        uint32_t Tm = 0, V1 = 0, V2 = 0, Tv = 0, Tb = 0, Tmb = 0, T1 = 0, T2 = 0, T3 = 0, T = 0, maxout = 0;
        auto set_window = [&](uint32_t tm) -> bool {
            Tm = tm; V1 = sub_len(Tm); V2 = sub_len(V1); Tv = sub_len(V2);
            Tb = barg > 0 ? (uint32_t) barg : barg < 0 ? (Tv + (uint32_t) -barg - 1) / (uint32_t) -barg * (uint32_t) -barg : Tv;
            if (Tb < Tv) { fprintf(stderr, "bucket %u < valid frames %u\n", Tb, Tv); return false; }
            Tmb = barg ? 8 * Tb : Tm;
            T1 = sub_len(Tmb); T2 = sub_len(T1); T3 = sub_len(T2); T = T3;
            maxout = 10 * Tv + 16;
            return true;
        };
        if (!set_window(cap_mel)) return 1;
        // capacity sizes (the first window in CLI mode; MAXMEL in serve mode)
        const uint32_t cTmb = Tmb, cT1 = T1, cT2 = T2, cT3 = T3, cT = T, cmaxout = maxout;
        const uint32_t F1 = F0 / 2, F2 = F1 / 2, F3 = F2 / 2, C = 256, D = 1024, H = 8, FF = 4096;
        auto pad32 = [](uint32_t n) { return (n + 31) / 32 * 32; };

        // arena (floats)
        uint32_t o = 0;
        auto take = [&](size_t n) { const uint32_t r = o; o += (uint32_t) ((n + 63) / 64 * 64); return r; };
        const uint32_t aMel = take((size_t) cTmb * F0), aS1 = take((size_t) pad32(cT1 * F1) * C), aS2 = take((size_t) pad32(cT2 * F2) * C),
                       aS2p = take((size_t) pad32(cT2 * F2) * C), aS3 = take((size_t) pad32(cT3 * F3) * C),
                       aS3p = take((size_t) pad32(cT3 * F3) * C), aFlat = take((size_t) pad32(cT) * C * F3);
        // rel-pos table for positions Tp-1 .. -(Tp-1), Tp = the capacity T: exact for any window T <= Tp
        const uint32_t Tp = cT, PT = 2 * Tp - 1;
        // aXn, aH, aQ, aK, aV, aP, aAt, aG, aCm hold half data (offset 2x in the half view); aP has one slot per layer
        const uint32_t pslot = pad32(PT) * D / 2;
        const uint32_t aX = take((size_t) pad32(cT) * D), aXn = take((size_t) pad32(cT) * D), aH = take((size_t) pad32(cT) * FF),
                       aQKV = take((size_t) pad32(cT) * 3 * D), aPe = take((size_t) pad32(PT) * D), aP = take((size_t) 24 * pslot),
                       aAt = take((size_t) pad32(cT) * D), aG = take((size_t) pad32(cT) * 2 * D), aCm = take((size_t) pad32(cT) * D);
        // decode: joint.enc projection E [T][640], decoder state (control block, h0 / h1 ping-pong, g, c0, c1, the walk
        // checkpoint, joint partials), output
        const uint32_t HDd = 640;
        const uint32_t aE = take((size_t) pad32(cT) * HDd), nS = 64 + 12 * HDd + TDT_KF * TDT_NSL * TDT_PSTRIDE, aS = take(nS),
                       aOut = take(1 + 4 * cmaxout),
                       aDbg = take(4 * 512), aDbgL = take(1030 + 2 * 640);
        // GPU log-mel: samples, power spectrum [frames][257], log mel [frames][128], filterbank [128][257], cos / sin / window
        const uint32_t aWav = take((size_t) cap_mel * 160 + 1024), aPow = take((size_t) cap_mel * 257), aLmel = take((size_t) cap_mel * F0),
                       aFb = take(128 * 257), aTw = take(2 * 512 + 400);
        // layer-0 input projection table b0 + Wx0 embed[t] (PA_XTAB=0: off): embed copied in as half rows, float table out
        const bool use_xtab = !getenv("PA_XTAB") || atoi(getenv("PA_XTAB"));
        const uint32_t aEmb = take((size_t) pad32(1025) * HDd / 2), aXtab = take((size_t) 1025 * 4 * HDd);
        // dictionary boosting (PA_TRIE=file from bias/make_trie.py: the flat trie the tdt kernel walks, uint32 words);
        // committed spans out: count, {start token, end token, canonical term, boosted} x maxspans, flips, rollbacks
        std::vector<uint32_t> trie;
        if (getenv("PA_TRIE") && *getenv("PA_TRIE")) {
            std::ifstream tf(getenv("PA_TRIE"), std::ios::binary | std::ios::ate);
            if (!tf.good()) { fprintf(stderr, "cannot read PA_TRIE %s\n", getenv("PA_TRIE")); return 1; }
            trie.resize((size_t) tf.tellg() / 4);
            tf.seekg(0);
            tf.read((char *) trie.data(), trie.size() * 4);
        }
        // serve keeps room for a dictionary sent later (request ~0 below): 1M words, about 11k terms with case variants
        const size_t trie_cap = std::max<size_t>(trie.size(), serve ? (size_t) 1 << 20 : 1);
        const uint32_t aTrie = take(trie_cap), aSpans = take(3 + 4 * (size_t) cmaxout);
        const uint32_t dbg_step = getenv("PA_DBG_STEP") ? (uint32_t) atoi(getenv("PA_DBG_STEP")) : ~0u;
        id<MTLBuffer> abuf = [dev newBufferWithLength:(size_t) o * 4 options:MTLResourceStorageModeShared];
        float * A = (float *) abuf.contents;
        memset(A, 0, (size_t) o * 4);
        if (!serve && !gpu_mel) memcpy(A + aMel, mel.data(), mel.size() * 4);
        {   // mel front-end constants: filterbank (ref.py melfb, next to the weights), DFT tables, hann(400)
            // melfb.npy: PA_DIR (the shader directory; the app bundle's Resources/asr), else next to the weights
            std::string fp = std::string(argv[1]);
            fp = fp.substr(0, fp.find_last_of('/') + 1) + "melfb.npy";
            if (getenv("PA_DIR")) {
                const std::string dp = std::string(getenv("PA_DIR")) + "/melfb.npy";
                if (std::ifstream(dp).good()) fp = dp;
            }
            std::ifstream probe(fp);
            if (probe.good()) {
                std::vector<size_t> fs;
                std::vector<float> fb = load_npy(fp.c_str(), fs);
                if (fb.size() == 128 * 257) { memcpy(A + aFb, fb.data(), fb.size() * 4); have_fb = true; }
            }
            if (!have_fb && (gpu_mel || serve)) fprintf(stderr, "no 128 x 257 %s: audio requests are refused\n", fp.c_str());
            if (!have_fb && gpu_mel) return 1;
            for (int i = 0; i < 512; ++i) {
                A[aTw + i] = (float) cos(2.0 * M_PI * i / 512.0);
                A[aTw + 512 + i] = (float) sin(2.0 * M_PI * i / 512.0);
            }
            for (int i = 0; i < 400; ++i) A[aTw + 1024 + i] = (float) (0.5 - 0.5 * cos(2.0 * M_PI * i / 400.0));  // np.hanning(401)[:-1]
        }
        // relative positional encoding: positions Tp-1 .. -(Tp-1), sin / cos interleaved (MLX casts it to the activation
        // dtype: bf16 in an all-bf16 run, float32 in the app)
        for (uint32_t r = 0; r < PT; ++r) {
            const float pos = (float) ((int) Tp - 1 - (int) r);
            for (uint32_t i = 0; i < D / 2; ++i) {
                const float div = expf((float) (2 * i) * -(logf(10000.f) / D)), s = sinf(pos * div), c = cosf(pos * div);
                A[aPe + r * D + 2 * i] = bf16_run ? bf16_round(s) : s;
                A[aPe + r * D + 2 * i + 1] = bf16_run ? bf16_round(c) : c;
            }
        }

        // pipelines
        NSString * dir = getenv("PA_DIR") ? [NSString stringWithUTF8String:getenv("PA_DIR")]
                                          : [[NSString stringWithUTF8String:argv[0]] stringByDeletingLastPathComponent];
        NSString * src = [NSString stringWithContentsOfFile:[dir stringByAppendingPathComponent:@"pa.metal"] encoding:NSUTF8StringEncoding error:nil];
        if (!src) { fprintf(stderr, "cannot read pa.metal (PA_DIR)\n"); return 1; }
        NSError * err = nil;
        const int lin_v = 4, tn2 = getenv("PA_TN2") ? atoi(getenv("PA_TN2")) : 2;
        // PA_UNROLL=n: unroll lin4's k loop n times (measured at T 22: no gain, within the 0.1 ms noise)
        const std::string unroll = getenv("PA_UNROLL") && *getenv("PA_UNROLL") ? std::string("#define LIN4_UNROLL _Pragma(\"clang loop unroll_count(") + getenv("PA_UNROLL") + ")\")\n" : "";
        src = [[NSString stringWithFormat:@"#define TN2 %d\n#define DEC_BF16 %d\n#define TDT_LNU %u\n#define TDT_KF %u\n%s", tn2, bf16_run, TDT_LNU, TDT_KF,
                                          unroll.c_str()] stringByAppendingString:src];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
        if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 1; }
        auto pso = [&](const char * n) {
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:n]] error:&err];
            if (!p) { fprintf(stderr, "%s: %s\n", n, err.localizedDescription.UTF8String); exit(1); }
            return p;
        };
        id<MTLComputePipelineState> pLin = pso("lin"), pLin2 = pso("lin2"), pNorm = pso("layernorm"), pAtt = pso("attn"), pConv = pso("convmid"),
                                    pSub = pso("subconv"), pFlat = pso("subflat");
        id<MTLComputePipelineState> pAtt8 = pso("attn8"), pMelPow = pso("melpow"), pMelFb = pso("melfb"), pMelNorm = pso("melnorm");
        // the TDT decode: prediction net (3 kernels), joint, decisions; all 256-thread threadgroups
        id<MTLComputePipelineState> pTdtL0 = pso("tdt_l0"), pTdtL1 = pso("tdt_l1"), pTdtPred = pso("tdt_pred"), pTdtJoint = pso("tdt_joint"),
                                    pTdtDecide = pso("tdt_decide");
        for (id<MTLComputePipelineState> p : @[ pTdtL0, pTdtL1, pTdtPred, pTdtJoint, pTdtDecide ])
            if (p.maxTotalThreadsPerThreadgroup < 256) { fprintf(stderr, "tdt: max %lu threads per threadgroup\n", (unsigned long) p.maxTotalThreadsPerThreadgroup); return 1; }
        const uint32_t att8_from = getenv("PA_ATT8_FROM") ? (uint32_t) atoi(getenv("PA_ATT8_FROM")) : 48;  // attn8 when Tv > this (measured: slower at T 22, faster at 71+)
        // lin4_<x>_<y>_<row tiles>_<8-wide n tiles>: TN tn2 (PA_TN2, default 2), narrow n <= 1024 projections TN tns (PA_TNS, default 1)
        const int tns = getenv("PA_TNS") ? atoi(getenv("PA_TNS")) : 1;
        const uint32_t ks1 = getenv("PA_KS1") ? atoi(getenv("PA_KS1")) : 1, ks4 = getenv("PA_KS4") ? atoi(getenv("PA_KS4")) : 1;  // k splits
        id<MTLComputePipelineState> pL4[2][2][4][3];  // [x half][y half][row tiles - 1][TN 1, 2, 4]
        auto tni = [](int tn) { return tn == 1 ? 0 : tn == 2 ? 1 : 2; };
        for (int xt = 0; xt < 2; ++xt)
            for (int yt = 0; yt < 2; ++yt)
                for (int rt = 1; rt <= 4; ++rt)
                    for (int tn : { 1, 2, 4 })
                        pL4[xt][yt][rt - 1][tni(tn)] = pso(("lin4_" + std::string(xt ? "half" : "float") + "_" + (yt ? "half" : "float") + "_" +
                                                            std::to_string(rt) + "_" + std::to_string(tn)).c_str());
        id<MTLComputePipelineState> pL4q[2][3][4][2];  // [8 bit, 4 bit][(f,f) (h,h) (h,f)][row tiles - 1][TN - 1]
        if (qbuf)
            for (int b = 0; b < 2; ++b)
                for (int ty = 0; ty < 3; ++ty)
                    for (int rt = 1; rt <= 4; ++rt)
                        for (int tn = 1; tn <= 2; ++tn) {
                            const char * tys[3] = { "float_float", "half_half", "half_float" };
                            pL4q[b][ty][rt - 1][tn - 1] = pso(("lin4q" + std::string(b ? "4" : "8") + "_" + tys[ty] + "_" + std::to_string(rt) + "_" +
                                                               std::to_string(tn)).c_str());
                        }
        if (getenv("PA_COMPILE_ONLY")) { fprintf(stderr, "pa: shaders compiled, pipelines built (no GPU work)\n"); return 0; }
        size_t wbytes_pass = 0;  // GEMM weight bytes one pass reads
        bool cached_pos = false;

        const uint32_t NONE = ~0u;
        const char * only = getenv("PA_ONLY");  // profiling: encode one kernel kind only (lin, norm, attn, conv, sub)
        int n_disp = 0;
        auto encode = [&](id<MTLComputeCommandEncoder> e) {
            [e setBuffer:wbuf offset:0 atIndex:1];
            [e setBuffer:abuf offset:0 atIndex:2];
            if (qbuf) [e setBuffer:qbuf offset:0 atIndex:3];
            n_disp = 0;
            wbytes_pass = 0;
            // x / y types: 0 float, 1 half (offsets then in halves: 2 * the float offset of the region)
            auto lin = [&](uint32_t x, uint32_t y, uint32_t r, uint32_t w, uint32_t b, uint32_t rows, uint32_t n, uint32_t k,
                           uint32_t act, float alpha, uint32_t res, int xt = 0, int yt = 0) {
                if (only && strncmp(only, "lin", 3) != 0) return;
                if (only && only[3] && (std::string(only + 3) != std::to_string(n) + "x" + std::to_string(k))) return;  // e.g. lin1024x4096
                // residual projections into the float stream (y = r, no bias / act): k split over threadgroups, float atomics
                const uint32_t ks = (res && y == r && b == NONE && act == 0 && !yt) ? (k <= 1024 ? ks1 : ks4) : 1;
                Lin L ={ xt ? 2 * x : x, yt ? 2 * y : y, r, w, b, rows, n, k, act, alpha, res, ks };
                [e setBytes:&L length:sizeof L atIndex:0];
                if (lin_v == 1) {
                    [e setComputePipelineState:pLin];
                    [e dispatchThreadgroups:MTLSizeMake((n + 31) / 32, (rows + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else if (lin_v == 2) {
                    [e setComputePipelineState:pLin2];
                    [e dispatchThreadgroups:MTLSizeMake((n + 8 * tn2 - 1) / (8 * tn2), (rows + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                } else {
                    // row blocks of up to 32 rows, balanced: T = 38 runs 2 blocks of 24 rows instead of 2 of 32
                    const uint32_t nblk = (rows + 31) / 32;
                    const int rt = (int) ((rows + 8 * nblk - 1) / (8 * nblk)), tn = n <= 1024 ? tns : tn2;
                    auto qi = qmap.find(w);
                    if (qi != qmap.end()) {  // quantized weights
                        const QEnt & q = qi->second;
                        L.qoff = q.qoff; L.qsb = q.qsb; L.qg = q.g; L.ksplit = 1;
                        [e setBytes:&L length:sizeof L atIndex:0];
                        const int ty = !xt && !yt ? 0 : xt && yt ? 1 : 2;
                        if (xt == 0 && yt == 1) { fprintf(stderr, "lin4q: no float -> half variant\n"); exit(1); }
                        [e setComputePipelineState:pL4q[q.bits == 8 ? 0 : 1][ty][rt - 1][std::min(tn, 2) - 1]];
                        [e dispatchThreadgroups:MTLSizeMake((n + 8 * std::min(tn, 2) - 1) / (8 * std::min(tn, 2)), (rows + 8 * rt - 1) / (8 * rt), 1)
                          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        wbytes_pass += (size_t) n * k * q.bits / 8 + (size_t) n * (k / q.g) * 4;
                        ++n_disp;
                        return;
                    }
                    wbytes_pass += (size_t) n * k * 2;
                    [e setComputePipelineState:pL4[xt][yt][rt - 1][tni(tn)]];
                    [e dispatchThreadgroups:MTLSizeMake((n + 8 * tn - 1) / (8 * tn), (rows + 8 * rt - 1) / (8 * rt), ks)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                ++n_disp;
            };
            auto norm = [&](uint32_t x, uint32_t y, const std::string & p, uint32_t yh = 1) {
                if (only && strcmp(only, "norm") != 0) return;
                Norm P = { x, yh == 1 ? 2 * y : y, W(p + ".weight"), W(p + ".bias"), T, D, 1e-5f, yh };
                [e setComputePipelineState:pNorm];
                [e setBytes:&P length:sizeof P atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_disp;
            };
            auto sub = [&](uint32_t x, uint32_t y, const std::string & p, uint32_t Ti, uint32_t Fi, uint32_t To, uint32_t Fo, uint32_t mode,
                           uint32_t Tvalid) {
                if (only && strcmp(only, "sub") != 0) return;
                Sub P = { x, y, W(p + ".weight"), W(p + ".bias"), Ti, Fi, To, Fo, C, Tvalid, mode };
                [e setComputePipelineState:pSub];
                [e setBytes:&P length:sizeof P atIndex:0];
                [e dispatchThreads:MTLSizeMake(C, Fo, To) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_disp;
            };
            if (gpu_mel && (!only || !strcmp(only, "mel"))) {  // log-mel of the nwav samples at aWav -> aMel [Tm][128]
                const MelP M = { aWav, nwav, aPow, aLmel, aMel, aFb, aTw, Tm };
                [e setBytes:&M length:sizeof M atIndex:0];
                [e setComputePipelineState:pMelPow];
                [e dispatchThreads:MTLSizeMake(257, Tm, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
                [e setComputePipelineState:pMelFb];
                [e dispatchThreads:MTLSizeMake(128, Tm, 1) threadsPerThreadgroup:MTLSizeMake(128, 2, 1)];
                [e setComputePipelineState:pMelNorm];
                [e dispatchThreadgroups:MTLSizeMake(128, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                n_disp += 3;
            }
            const std::string pe = "encoder.pre_encode.";
            sub(aMel, aS1, pe + "conv.0", Tmb, F0, T1, F1, 0, Tm);
            sub(aS1, aS2, pe + "conv.2", T1, F1, T2, F2, 1, V1);
            lin(aS2, aS2p, 0, W(pe + "conv.3.weight"), W(pe + "conv.3.bias"), T2 * F2, C, C, 1, 1.f, 0);
            sub(aS2p, aS3, pe + "conv.5", T2, F2, T3, F3, 1, V2);
            lin(aS3, aS3p, 0, W(pe + "conv.6.weight"), W(pe + "conv.6.bias"), T3 * F3, C, C, 1, 1.f, 0);
            {
                Sub P = { aS3p, aFlat, 0, 0, T3, F3, T3, F3, C, T3, 0 };
                [e setComputePipelineState:pFlat];
                [e setBytes:&P length:sizeof P atIndex:0];
                [e dispatchThreads:MTLSizeMake(C, F3, T3) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++n_disp;
            }
            lin(aFlat, aX, 0, W(pe + "out.weight"), W(pe + "out.bias"), T, D, C * F3, 0, 1.f, 0);
            const int nlayers = getenv("PA_LAYERS") ? atoi(getenv("PA_LAYERS")) : 24;  // debug: stop after n layers
            for (int l = 0; l < nlayers; ++l) {
                const std::string L = "encoder.layers." + std::to_string(l) + ".";
                norm(aX, aXn, L + "norm_feed_forward1");
                lin(aXn, aH, 0, W(L + "feed_forward1.linear1.weight"), NONE, T, FF, D, 2, 1.f, 0, 1, 1);
                lin(aH, aX, aX, W(L + "feed_forward1.linear2.weight"), NONE, T, D, FF, 0, 0.5f, 1, 1, 0);
                norm(aX, aXn, L + "norm_self_att");
                // q, k, v weights are contiguous in the file: one [3 D, D] projection into rows of stride 3 D
                lin(aXn, aQKV, 0, W(L + "self_attn.linear_q.weight"), NONE, T, 3 * D, D, 0, 1.f, 0, 1, 1);
                if (!cached_pos) lin(aPe, aP + l * pslot, 0, W(L + "self_attn.linear_pos.weight"), NONE, PT, D, D, 0, 1.f, 0, 0, 1);
                if (!only || !strcmp(only, "attn")) {
                    Att P = { 2 * aQKV, 2 * aQKV + D, 2 * aQKV + 2 * D, 2 * (aP + l * pslot), 2 * aAt, W(L + "self_attn.pos_bias_u"),
                              W(L + "self_attn.pos_bias_v"), T, D, H, 1.f / sqrtf((float) (D / H)), 3 * D, Tv, Tp };
                    [e setBytes:&P length:sizeof P atIndex:0];
                    if (Tv > att8_from && Tv <= 512) {  // 8 queries per threadgroup
                        [e setComputePipelineState:pAtt8];
                        [e dispatchThreadgroups:MTLSizeMake((T + 7) / 8, H, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    } else {
                        [e setComputePipelineState:pAtt];
                        [e dispatchThreadgroups:MTLSizeMake(T, H, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                    ++n_disp;
                }
                lin(aAt, aX, aX, W(L + "self_attn.linear_out.weight"), NONE, T, D, D, 0, 1.f, 1, 1, 0);
                norm(aX, aXn, L + "norm_conv");
                lin(aXn, aG, 0, W(L + "conv.pointwise_conv1.weight"), NONE, T, 2 * D, D, 0, 1.f, 0, 1, 1);
                if (!only || !strcmp(only, "conv")) {
                    const std::string c = L + "conv.";
                    Conv P = { 2 * aG, 2 * aCm, W(c + "depthwise_conv.weight"), W(c + "batch_norm.weight"), W(c + "batch_norm.bias"),
                               W(c + "batch_norm.running_mean"), W(c + "batch_norm.running_var"), Tv, D, 1e-5f };
                    [e setComputePipelineState:pConv];
                    [e setBytes:&P length:sizeof P atIndex:0];
                    [e dispatchThreads:MTLSizeMake(D, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    ++n_disp;
                }
                lin(aCm, aX, aX, W(L + "conv.pointwise_conv2.weight"), NONE, T, D, D, 0, 1.f, 1, 1, 0);
                norm(aX, aXn, L + "norm_feed_forward2");
                lin(aXn, aH, 0, W(L + "feed_forward2.linear1.weight"), NONE, T, FF, D, 2, 1.f, 0, 1, 1);
                lin(aH, aX, aX, W(L + "feed_forward2.linear2.weight"), NONE, T, D, FF, 0, 0.5f, 1, 1, 0);
                norm(aX, aX, L + "norm_out", l == 23 && bf16_run ? 2 : 0);
            }
            lin(aX, aE, 0, W("joint.enc.weight"), W("joint.enc.bias"), T, HDd, D, 0, 1.f, 0);  // the joint's encoder projection
        };
        const std::string dp = "decoder.prediction.dec_rnn.lstm.";
        Tdt TD = { aE, Tv, aS, aOut, maxout, W("decoder.prediction.embed.weight"), W(dp + "0.Wx"), W(dp + "0.Wh"), W(dp + "0.bias"),
                   W(dp + "1.Wx"), W(dp + "1.Wh"), W(dp + "1.bias"), W("joint.pred.weight"), W("joint.pred.bias"),
                   W("joint.joint_net.2.weight"), W("joint.joint_net.2.bias"), 10, aDbg, dbg_step, aDbgL, use_xtab ? aXtab : ~0u,
                   trie.empty() ? ~0u : aTrie, aSpans, cmaxout };
        if (!trie.empty()) memcpy((uint32_t *) A + aTrie, trie.data(), trie.size() * 4);
        // the rel-pos projections linear_pos(pos_emb) depend on T only: computed once per T (bucket) and cached
        if (getenv("PA_POS_CACHE") == nullptr || atoi(getenv("PA_POS_CACHE"))) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setBuffer:wbuf offset:0 atIndex:1];
            [e setBuffer:abuf offset:0 atIndex:2];
            for (int l = 0; l < 24; ++l) {
                Lin Lp = { aPe, 2 * (aP + l * pslot), 0, W("encoder.layers." + std::to_string(l) + ".self_attn.linear_pos.weight"), NONE, PT, D, D, 0, 1.f, 0 };
                [e setBytes:&Lp length:sizeof Lp atIndex:0];
                const int rt = 4;
                [e setComputePipelineState:pL4[0][1][rt - 1][tni(tn2)]];
                [e dispatchThreadgroups:MTLSizeMake(D / (8 * tn2), (PT + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            if (use_xtab) {  // b0 + Wx0 embed[t], t < 1025: rows 1025, n 2560, k 640, half in, float out
                memcpy((uint16_t *) A + 2 * aEmb, (const uint16_t *) wptr + W("decoder.prediction.embed.weight"), (size_t) 1025 * HDd * 2);
                Lin Lx = { 2 * aEmb, aXtab, 0, W(dp + "0.Wx"), W(dp + "0.bias"), 1025, 4 * HDd, HDd, 0, 1.f, 0 };
                [e setBytes:&Lx length:sizeof Lx atIndex:0];
                [e setComputePipelineState:pL4[1][0][3][tni(tn2)]];
                [e dispatchThreadgroups:MTLSizeMake(4 * HDd / (8 * tn2), (1025 + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            cached_pos = true;
        }

        // one pass over the window in A + aMel (set_window done): command buffer 1 = encoder + joint.enc projection,
        // then the TDT decode: command buffers of decode slots (pa.metal: five indirect dispatches each, sized by the
        // previous slot; ordinary dispatches only, so it cannot stall on a small or shared GPU) until the done flag.
        // Returns GPU ms of the encoder and of the decode, wall ms, CPU encode ms.
        const bool dec_on = !only;
        uint32_t * U = (uint32_t *) A;
        TdtCtl * CT = (TdtCtl *) (U + aS);
        int last_slots = 0, last_chunks = 0;
        auto encode_slots = [&](id<MTLComputeCommandEncoder> e, int n) {
            [e setBytes:&TD length:sizeof TD atIndex:0];
            [e setBuffer:wbuf offset:0 atIndex:1];
            [e setBuffer:abuf offset:0 atIndex:2];
            const NSUInteger ctl = (NSUInteger) aS * 4, la = ctl + offsetof(TdtCtl, lstm_args), ja = ctl + offsetof(TdtCtl, joint_args),
                             da = ctl + offsetof(TdtCtl, decide_args);
            const MTLSize tpg = MTLSizeMake(256, 1, 1);
            for (int i = 0; i < n; ++i) {
                [e setComputePipelineState:pTdtL0];
                [e dispatchThreadgroupsWithIndirectBuffer:abuf indirectBufferOffset:la threadsPerThreadgroup:tpg];
                [e setComputePipelineState:pTdtL1];
                [e dispatchThreadgroupsWithIndirectBuffer:abuf indirectBufferOffset:la threadsPerThreadgroup:tpg];
                [e setComputePipelineState:pTdtPred];
                [e dispatchThreadgroupsWithIndirectBuffer:abuf indirectBufferOffset:la threadsPerThreadgroup:tpg];
                [e setComputePipelineState:pTdtJoint];
                [e dispatchThreadgroupsWithIndirectBuffer:abuf indirectBufferOffset:ja threadsPerThreadgroup:tpg];
                [e setComputePipelineState:pTdtDecide];
                [e dispatchThreadgroupsWithIndirectBuffer:abuf indirectBufferOffset:da threadsPerThreadgroup:tpg];
            }
        };
        auto run_pass = [&](double & g1, double & g2, double & wall, double & cpu) -> bool {
            TD.T = Tv;
            TD.maxout = maxout;
            memset(U + aS, 0, (size_t) nS * 4);
            // the first slot: the prediction net on no token (zero input and state), then the joint from frame 0
            CT->need = 1;
            CT->tok = ~0u;
            CT->boost_live = TD.trie != ~0u;
            CT->no_boost_until = -1;
            CT->nf = std::min(TDT_KF, Tv);
            for (int i = 0; i < 3; ++i) CT->lstm_args[i] = CT->joint_args[i] = CT->decide_args[i] = 1;
            CT->lstm_args[0] = 640 / TDT_LNU;
            CT->joint_args[0] = TDT_NSL;
            CT->joint_args[1] = CT->nf;
            const double t0 = now();
            id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            encode(e);
            [e endEncoding];
            const double t1 = now();
            [cb commit];
            g2 = 0;
            last_slots = last_chunks = 0;
            if (dec_on) {
                // a slot per emitted token, plus one per run of blanks longer than a joint batch, plus rollbacks.
                // Measured on 402 windows (public and owner, with and without the dictionary): at most 0.73 slots
                // per frame on windows under 82 s, so 0.6 T + 12 covers them in one chunk; a slot past the end costs
                // five empty dispatches (~1-3 us), another chunk a round trip (~1 ms). Cap: a runaway.
                int n = (int) (Tv * 3 / 5 + 12);
                const int cap = (int) (4 * (maxout + Tv) + 64);
                for (;;) {
                    id<MTLCommandBuffer> c2 = [queue commandBufferWithUnretainedReferences];
                    id<MTLComputeCommandEncoder> e2 = [c2 computeCommandEncoder];
                    encode_slots(e2, n);
                    [e2 endEncoding];
                    [c2 commit];
                    [c2 waitUntilCompleted];
                    if (c2.status == MTLCommandBufferStatusError) { fprintf(stderr, "GPU error: %s\n", c2.error.localizedDescription.UTF8String); return false; }
                    g2 += (c2.GPUEndTime - c2.GPUStartTime) * 1e3;
                    last_slots += n;
                    ++last_chunks;
                    if (CT->done) break;
                    if (last_slots >= cap) { fprintf(stderr, "tdt: decode not done after %d slots\n", last_slots); return false; }
                    n = std::min(cap - last_slots, std::max(16, (int) ((Tv - CT->step) * 3 / 5 + 12)));
                }
            }
            [cb waitUntilCompleted];
            const double t2 = now();
            if (cb.status == MTLCommandBufferStatusError) { fprintf(stderr, "GPU error: %s\n", cb.error.localizedDescription.UTF8String); return false; }
            g1 = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
            wall = (t2 - t0) * 1e3;
            cpu = (t1 - t0) * 1e3;
            return true;
        };

        if (getenv("PA_NOOP")) {  // the per-dispatch floor: n dependent 1-threadgroup dispatches in one encoder
            id<MTLComputePipelineState> pNoop = pso("noop");
            const int nd = atoi(getenv("PA_NOOP"));
            const int tgs = getenv("PA_NOOP_TG") ? atoi(getenv("PA_NOOP_TG")) : 1;
            double best = 1e9;
            for (int r = 0; r < 8; ++r) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:pNoop];
                [e setBuffer:abuf offset:0 atIndex:2];
                for (int i = 0; i < nd; ++i) [e dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (r) best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
            }
            printf("noop: %d dispatches of %d threadgroups: GPU %.3f ms, %.2f us each\n", nd, tgs, best, best * 1e3 / nd);
            return 0;
        }
        if (serve) {
            // request: uint32 mel frames n (0 or EOF: quit), n x 128 float32 (the sidecar's get_logmel output, float32)
            // response: uint32 count, count x {uint32 token, uint32 duration index, uint32 frame, float confidence},
            //           float encoder GPU ms, decode GPU ms, wall ms (count = ~0: the window is over capacity or invalid)
            fprintf(stderr, "pa serve: ready, capacity %u mel frames (%u encoder frames)\n", cap_mel, cT);
            // a request whose n has the top bit set carries n & 0x7fffffff audio samples (16 kHz float32, >= 1600)
            // instead of mel frames: the log-mel then runs on the GPU in the same command buffer
            // n = ~0: a dictionary for the windows after it: uint32 words m, m x uint32 (a flat trie as PA_TRIE's; m = 0:
            // boosting off) -> uint32 0 (in use), 1 (over capacity) or 2 (sizes do not add up); on 1 or 2 the old one stays
            for (;;) {
                uint32_t n = 0;
                if (fread(&n, 4, 1, stdin) != 1 || n == 0) break;
                if (n == ~0u) {
                    uint32_t m = 0;
                    if (fread(&m, 4, 1, stdin) != 1) break;
                    std::vector<uint32_t> t(m);
                    if (m && fread(t.data(), 4, m, stdin) != m) break;
                    uint32_t st = m > trie_cap ? 1 : 0;
                    if (!st && m && (m < 16 || (uint64_t) t[0] * 4 + (uint64_t) t[1] * 3 + 16 + 1025 + 2 * 1032 != m)) st = 2;
                    if (!st) {
                        if (m) memcpy((uint32_t *) A + aTrie, t.data(), (size_t) m * 4);
                        TD.trie = m ? aTrie : ~0u;
                    }
                    fwrite(&st, 4, 1, stdout);
                    fflush(stdout);
                    continue;
                }
                gpu_mel = (n >> 31) != 0;
                n &= 0x7fffffffu;
                const uint32_t frames = gpu_mel ? (n + 272) / 160 : n, words = gpu_mel ? n : n * F0;
                const bool ok = frames <= cap_mel && (!gpu_mel || (n >= 1600 && have_fb));  // no filterbank: audio refused
                std::vector<float> buf(ok ? 0 : (size_t) words);
                if (fread(ok ? (void *) (A + (gpu_mel ? aWav : aMel)) : (void *) buf.data(), 4, words, stdin) != words) break;
                nwav = gpu_mel ? n : 0;
                double g1 = 0, g2 = 0, wall = 0, cpu = 0;
                uint32_t cnt = ~0u;
                if (ok && set_window(frames) && run_pass(g1, g2, wall, cpu)) cnt = std::min(U[aOut], maxout);
                if (getenv("PA_VERBOSE")) fprintf(stderr, "pa serve: T %u, decode %u of %d slots in %d chunks, %.3f ms\n", Tv, CT->slots, last_slots, last_chunks, g2);
                fwrite(&cnt, 4, 1, stdout);
                if (cnt != ~0u) fwrite(U + aOut + 1, 4, (size_t) 4 * cnt, stdout);
                const float ms[3] = { (float) g1, (float) g2, (float) wall };
                fwrite(ms, 4, 3, stdout);
                // dictionary boosting: uint32 spans n, n x {start token, end token, canonical term, boosted}, flips, rollbacks
                const uint32_t nsp = cnt != ~0u ? U[aSpans] : 0;
                fwrite(&nsp, 4, 1, stdout);
                fwrite(U + aSpans + 1, 4, (size_t) 4 * nsp, stdout);
                const uint32_t fr[2] = { cnt != ~0u ? U[aSpans + 1 + 4 * cmaxout] : 0, cnt != ~0u ? U[aSpans + 2 + 4 * cmaxout] : 0 };
                fwrite(fr, 4, 2, stdout);
                fflush(stdout);
            }
            return 0;
        }

        double best_gpu = 1e9, best_dec = 1e9, best_wall = 1e9, enc_cpu = 0;
        for (int r = 0; r < reps; ++r) {
            memcpy(A + (gpu_mel ? aWav : aMel), mel.data(), mel.size() * 4);
            double g1, g2, wall, cpu;
            if (!run_pass(g1, g2, wall, cpu)) return 1;
            if (r > 0 || reps == 1) {
                best_gpu = std::min(best_gpu, g1);
                if (dec_on) best_dec = std::min(best_dec, g2);
                best_wall = std::min(best_wall, wall);
                enc_cpu = cpu;
            }
        }
        printf("pa%s%s: mel %u -> T %u frames (bucket %u), %d dispatches, GEMM weights %.1f MB/pass (weights resident %.1f MB); encoder GPU min %.3f ms",
               qv.empty() ? "" : " ", qv.c_str(), Tm, Tv, T, n_disp, wbytes_pass / 1e6, (wsize - qsrc_bytes + qbytes) / 1e6, best_gpu);
        if (dec_on) printf(", decode GPU min %.3f ms (%u of %d slots, %d chunks)", best_dec, CT->slots, last_slots, last_chunks);
        printf(", wall min %.3f ms (encode %.3f ms)\n", best_wall, enc_cpu);
        save_npy(argv[3], A + aX, Tv, D);
        if (getenv("PA_DUMP_E")) save_npy(getenv("PA_DUMP_E"), A + aE, Tv, HDd);  // debug: the joint's encoder projection
        if (dbg_step != ~0u) {  // debug: the joint at the first iteration on that step
            const float * lgd = A + aDbgL;
            std::vector<int> ix(1025);
            for (int i = 0; i < 1025; ++i) ix[i] = i;
            std::partial_sort(ix.begin(), ix.begin() + 3, ix.end(), [&](int a, int b) { return lgd[a] > lgd[b]; });
            printf("  joint at step %u: top (%d %.3f) (%d %.3f) (%d %.3f), blank %.3f, dur %.2f %.2f %.2f %.2f %.2f\n", dbg_step, ix[0], lgd[ix[0]],
                   ix[1], lgd[ix[1]], ix[2], lgd[ix[2]], lgd[1024], lgd[1025], lgd[1026], lgd[1027], lgd[1028], lgd[1029]);
            save_npy("out/dbg_joint.npy", lgd, 1, 1030 + 2 * 640);
        }
        if (dec_on) {  // tokens next to OUT: OUT minus .npy + .tok.json
            std::string tp = argv[3];
            if (tp.size() > 4 && tp.substr(tp.size() - 4) == ".npy") tp = tp.substr(0, tp.size() - 4);
            FILE * f = fopen((tp + ".tok.json").c_str(), "w");
            fprintf(f, "{\"tokens\": [");
            for (uint32_t i = 0; i < std::min(U[aOut], maxout); ++i) fprintf(f, "%s%u", i ? ", " : "", U[aOut + 1 + 4 * i]);
            fprintf(f, "], \"durations\": [");
            for (uint32_t i = 0; i < std::min(U[aOut], maxout); ++i) fprintf(f, "%s%u", i ? ", " : "", U[aOut + 2 + 4 * i]);
            fprintf(f, "], \"steps\": [");
            for (uint32_t i = 0; i < std::min(U[aOut], maxout); ++i) fprintf(f, "%s%u", i ? ", " : "", U[aOut + 3 + 4 * i]);
            fprintf(f, "], \"spans\": [");
            for (uint32_t i = 0; i < U[aSpans]; ++i) {
                const uint32_t * s = U + aSpans + 1 + 4 * i;
                fprintf(f, "%s[%u, %u, %u, %u]", i ? ", " : "", s[0], s[1], s[2], s[3]);
            }
            fprintf(f, "], \"flips\": %u, \"rollbacks\": %u}\n", U[aSpans + 1 + 4 * cmaxout], U[aSpans + 2 + 4 * cmaxout]);
            fclose(f);
        }
        if (gpu_mel && getenv("PA_DUMP_MEL")) save_npy(getenv("PA_DUMP_MEL"), A + aMel, Tm, F0);  // debug: the GPU log-mel
        // fold in the comparisons with the cached MLX reference (REF.enc.npy / REF.json next to REF.mel.npy, or
        // PA_REF=<dir>/<tag> for an audio input)
        std::string base = getenv("PA_REF") ? std::string(getenv("PA_REF")) + ".mel.npy" : argv[2];
        if (base.size() > 8 && base.substr(base.size() - 8) == ".mel.npy") {
            base = base.substr(0, base.size() - 8);
            std::ifstream rf(base + ".enc.npy");
            if (rf.good()) {
                std::vector<size_t> rs;
                std::vector<float> ref = load_npy((base + ".enc.npy").c_str(), rs);
                if (rs[0] == Tv) {
                    double dd = 0, nn = 0, mx = 0, mincos = 1;
                    for (uint32_t t = 0; t < Tv; ++t) {
                        double ab = 0, aa = 0, bb = 0;
                        for (uint32_t c = 0; c < D; ++c) {
                            const double a = A[aX + t * D + c], b = ref[t * D + c];
                            dd += (a - b) * (a - b); nn += b * b; mx = std::max(mx, fabs(a - b));
                            ab += a * b; aa += a * a; bb += b * b;
                        }
                        mincos = std::min(mincos, ab / sqrt(aa * bb));
                    }
                    printf("  encoder vs MLX: rel L2 %.5f, max |diff| %.4f, min row cosine %.5f\n", sqrt(dd / nn), mx, mincos);
                }
            }
            std::ifstream jf(base + ".json");
            if (dec_on && jf.good()) {
                std::string js((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
                std::vector<int> rt, rd;  // MLX durations are in seconds (frames x 0.08)
                for (double x : json_nums(js, "tokens")) rt.push_back((int) x);
                for (double x : json_nums(js, "durations")) rd.push_back((int) lround(x / 0.08));
                const uint32_t n = U[aOut];
                std::vector<int> mt, md;
                const int durs[5] = { 0, 1, 2, 3, 4 };
                for (uint32_t i = 0; i < std::min(n, maxout); ++i) { mt.push_back((int) U[aOut + 1 + 4 * i]); md.push_back(durs[U[aOut + 2 + 4 * i]]); }
                int first = -1;
                for (size_t i = 0; i < std::max(mt.size(), rt.size()); ++i)
                    if (i >= mt.size() || i >= rt.size() || mt[i] != rt[i]) { first = (int) i; break; }
                printf("  decode: %u tokens (MLX fast_tdt %zu): tokens %s, durations %s", n, rt.size(), first < 0 ? "IDENTICAL" : "DIFFER",
                       md == rd ? "identical" : "differ");
                if (first >= 0) printf(" (first difference at %d)", first);
                printf("\n");
                if (first >= 0) {
                    printf("    metal (token:frames@step):");
                    for (size_t i = 0; i < mt.size(); ++i) printf(" %d:%d@%u", mt[i], md[i], U[aOut + 3 + 4 * i]);
                    printf("\n    mlx   (token:frames):");
                    for (size_t i = 0; i < rt.size(); ++i) printf(" %d:%d", rt[i], i < rd.size() ? rd[i] : -1);
                    printf("\n    metal iterations near the difference (step argmax dur margin):");
                    const uint32_t s0 = getenv("PA_DBG_FROM") ? (uint32_t) atoi(getenv("PA_DBG_FROM"))
                                        : first < (int) mt.size() ? U[aOut + 3 + 4 * first] : (first > 0 ? U[aOut + 3 + 4 * (first - 1)] : 0);
                    for (int it = 0; it < 512; ++it) {
                        const uint32_t * d = U + aDbg + 4 * it;
                        if (it > 0 && d[0] == 0 && d[1] == 0) break;
                        float mg;
                        memcpy(&mg, d + 3, 4);
                        if (d[0] + 2 >= s0) printf(" [%u %u %u %.3f]", d[0], d[1], d[2], mg);
                    }
                    printf("\n");
                }
            }
        }
    }
    return 0;
}
