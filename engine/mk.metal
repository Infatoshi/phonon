// LFM3-1.7B decode megakernel. One persistent launch runs the whole 24-layer stack as a list of phases;
// the co-resident threadgroups meet at a grid barrier between phases. Every phase is
//   prologue (stage the input vector: plain copy, embedding lookup, or residual add + RMS norms)
//   + op (matvec segments with fused epilogues, q/k norm + RoPE + KV store, attention).
// Single-row matvecs follow ggml's kernel_mul_mv_q4_K_f32 / q6_K_f32 exactly (bit-identical results);
// multi-row launches (verifying drafted tokens) use 8x8 simdgroup matrix multiplies.
// The host prepends: NSG (simdgroups per threadgroup), NB (token rows per launch: 1 for decode,
// k for verifying k drafted tokens).
#include <metal_stdlib>
using namespace metal;

#ifndef NSG
#define NSG 8
#endif
#ifndef NB
#define NB 1
#endif
#define NT (NSG * 32)
#define QK_K 256
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define PRAGMA_(x) _Pragma(#x)
#define UNROLL_N(n) PRAGMA_(clang loop unroll_count(n))

// model shape (LFM3-1.7B)
#define H 2048          // hidden
#define HD 128          // head dim
#define NQH 16          // query heads
#define NKVH 4          // kv heads
#define CONV_L 7        // short conv taps
// conv input history: Params.ring positions per channel (a power of 2 >= NB + CONV_L - 1), layer rows at P.aux[0]

#ifndef ATT_CH
#define ATT_CH 32       // positions per attention work unit (<= 32: one per lane). 32 vs 16 on the M1: attention
#endif                  // 1.21 vs 1.67 ms per verify-8 (fewer partials and atomics), 0.20 vs 0.26 ms per decode
#ifndef ATT_MAXCH
#define ATT_MAXCH 128   // max chunks per (row, head): context cap 2048 / 16
#endif
#ifndef ATT_UNROLL
#define ATT_UNROLL 8    // attention loops: loads in flight vs registers (the kernel's register budget is
#endif                  // the maximum over all ops, and it sets how many threadgroups fit on a core)

// Matvec path: up to SP_MAX rows use ggml's single-row kernel looped over the rows (inputs staged in
// threadgroup memory, results bit-identical to decode); more rows use 8x8 simdgroup matrix multiplies.
#ifndef SP_MAX
#define SP_MAX 2
#endif
#define SP (NB <= SP_MAX)
#if !SP && NB != 8
#error "the MMA path is built for exactly 8 token rows (verify: 1 + 7 drafted)"
#endif
// rows per simdgroup work unit: 2 (as ggml) on the single-row path, 8 (one MMA tile) on the MMA path
#ifndef RU_SP
#define RU_SP 2
#endif
#define RU (SP ? RU_SP : 8)
#define YS_TG SP
#define YS_LEN (YS_TG ? NB * H : 1)
// MMA path (q4m, below): a work unit is 8 * TN weight rows x 8 tokens, computed by SPLIT simdgroups of one
// threadgroup that take interleaved 64-column chunks of k and add up their partials in threadgroup memory
#ifndef TN
#define TN 2
#endif
#ifndef SPLIT
#define SPLIT NSG
#endif
#define UGROUPS (NSG / SPLIT)  // units in flight per threadgroup
#ifndef Q4M_SHARE
#define Q4M_SHARE 0  // SwiGLU gate/up and conv B/C/X tiles in one pass over y: no gain on the M4 Max, spills on the M1
#endif
#define SCR_LEN (SP ? 1 : NSG * TN * 32 * 2 * 2)  // float2 partials x 2 (SwiGLU: gate, up)
#define NBT 1
#define NBP 8
#define WS 64
// Matvec inputs (normalized y, conv output, SwiGLU output, attention output) on the MMA path are stored in the
// q4m fragment order YQ(i, b): each lane's B fragments of a 64-column chunk are 16 contiguous floats. Everything
// else (and everything on the single-row path) is row-major: token row b at b * dim.
inline uint frag_lane(uint r, uint c) { return 16 * (r / 4) + 8 * (c / 4) + 2 * (r % 4) + (c % 4) / 2; }
inline uint YQ(uint k, uint b) {
    const uint kl = k % 64, s4 = kl / 32, w = kl % 32;
    const uint i = 2 * (w / 8) + (w % 4) / 2;            // B row (= k index within the step)
    const uint s = 4 * s4 + 2 * ((w % 8) / 4) + w % 2;   // step
    return (k / 64) * 512 + frag_lane(i, b) * 16 + s * 2 + b % 2;
}
#define YI(i, b, dim) (SP ? (b) * (dim) + (i) : YQ(i, b))

typedef struct {
    half d, dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
} block_q4_K;  // 144 bytes / 256 weights

typedef struct {
    uchar ql[QK_K / 2];
    uchar qh[QK_K / 4];
    char scales[QK_K / 16];
    half d;
} block_q6_K;  // 210 bytes / 256 weights

typedef struct {
    half d, dmin;
    uchar scales[12];
    uchar qh[QK_K / 8];
    uchar qs[QK_K / 2];
} block_q5_K;  // 176 bytes / 256 weights

enum { WT_Q4_K = 12, WT_Q5_K = 13, WT_Q6_K = 14 };  // ggml type ids

enum {
    OP_NONE = 0,     // prologue only (final norm for the LM head)
    OP_MATVEC = 1,   // out = W . y for each segment
    OP_SWIGLU = 2,   // out = silu(W0 . y) * (W1 . y)
    OP_CONV = 3,     // in_proj (B, C, X) + causal depthwise conv over the history ring + C gating
    OP_QKROPE = 4,   // per-head RMS norm of q and k, NEOX RoPE, k/v into the f16 KV cache
    OP_ATTN = 5,     // softmax attention, split over position chunks, sigmoid head gate
};

enum {
    PRO_STAGE = 0,   // y = A[in] (k floats per row)
    PRO_EMBED = 1,   // h = token_embd[token] (Q6_K row); y = rms(h) * w_pre
    PRO_RESID = 2,   // h = A[hres] + rms(A[in]) * w_post; y = rms(h) * w_pre
};

struct Seg {
    ulong w;          // byte offset of the weights
    uint wtype, n_rows, out, pad;
};

struct Phase {
    uint op, pro, k, in;
    uint hres, hout, n_seg, ynorm;  // ynorm: device copy of y when it is not in threadgroup memory
    ulong w_post, w_pre;
    Seg seg[4];
    ulong aux_w[4];
    uint aux[8];
};

struct Params {
    uint n_phases, n_tg, spin_limit, pos;
    float eps;
    uint ctx;         // KV cache positions per layer
    uint n_valid;     // real token rows (the rest of NB is padding)
    uint ring;        // conv history positions per channel (power of 2: 64, or the context for sessions)
    uint tokens[32];
};

// ---------------------------------------------------------------------------------------------
// grid barrier: thread 0 of each threadgroup increments a counter that only grows during a launch,
// then spins until all n_tg have arrived. fetch_add(0) instead of atomic_load: on the M1 a spinning
// load never observes other cores' increments. Bounded: a non-co-resident launch sets the fail flag.
inline void grid_arrive(device atomic_uint * ctr, uint tid) {
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
    }
}

inline bool grid_wait(device atomic_uint * ctr, device atomic_uint * fail, uint target, uint spin_limit, uint tid,
                      threadgroup uint * tg_ok) {
    if (tid == 0) {
        uint spins = 0, ok = 1;
        while (atomic_fetch_add_explicit(ctr, 0u, memory_order_relaxed) < target) {
            if (++spins > spin_limit || atomic_fetch_add_explicit(fail, 0u, memory_order_relaxed)) {
                atomic_store_explicit(fail, 1u, memory_order_relaxed);
                ok = 0;
                break;
            }
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        *tg_ok = ok;
    }
    threadgroup_barrier(mem_flags::mem_device);
    return *tg_ok != 0;
}

// ---------------------------------------------------------------------------------------------
// Single-row Q4_K / Q6_K dot products of NR consecutive weight rows, one simdgroup: ggml's lane
// layout and arithmetic order. YP: threadgroup or device pointer.
template <int NR, typename YP>
inline void dot_q4_K(device const uchar * w, uint row_bytes, uint k, YP y, ushort lane, thread float * out) {
    constexpr ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    const short ix = lane / 8, it = lane % 8, iq = it / 4, ir = it % 4;
    const int nb = k / QK_K;
    device const block_q4_K * x = (device const block_q4_K *) w;
    float yl[16], yh[16];
    float sumf[NR] = { 0.f };
    YP y4 = y + ix * QK_K + 64 * iq + 8 * ir;
    ushort sc16[4];
    thread const uchar * sc8 = (thread const uchar *) sc16;
    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = { 0.f, 0.f, 0.f, 0.f };
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];   sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];  sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }
        device const ushort * sc = (device const ushort *) x[ib].scales + iq;
        device const ushort * q1 = (device const ushort *) x[ib].qs + 16 * iq + 4 * ir;
        device const half * dh = &x[ib].d;
        for (short row = 0; row < NR; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort * q2 = q1 + 32;
            float4 acc1 = { 0.f, 0.f, 0.f, 0.f };
            float4 acc2 = { 0.f, 0.f, 0.f, 0.f };
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }
            sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                  (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            q1 += row_bytes / 2;
            sc += row_bytes / 2;
            dh += row_bytes / 2;
        }
        y4 += 4 * QK_K;
    }
    for (short row = 0; row < NR; ++row) out[row] = simd_sum(sumf[row]);
}

// ggml's kernel_mul_mv_q5_K_f32 lane layout and arithmetic
template <int NR, typename YP>
inline void dot_q5_K(device const uchar * w, uint row_bytes, uint k, YP yy, ushort lane, thread float * out) {
    constexpr ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    const int nb = k / QK_K;
    device const block_q5_K * x = (device const block_q5_K *) w;
    float sumf[NR] = { 0.f };
    float yl[16], yh[16];
    const short tid = lane / 4, ix = lane % 4, iq = tid / 4, ir = tid % 4;
    const short l0 = 8 * ir, q_offset = 32 * iq + l0, y_offset = 64 * iq + l0;
    const uchar hm1 = 1u << (2 * iq), hm2 = hm1 << 1, hm3 = hm1 << 4, hm4 = hm2 << 4;
    ushort sc16[4];
    thread const uchar * sc8 = (thread const uchar *) sc16;
    YP y1 = yy + ix * QK_K + y_offset;
    for (int i = ix; i < nb; i += 4) {
        device const uchar * q1 = x[i].qs + q_offset;
        device const uchar * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const ushort * a = (device const ushort *) x[i].scales + iq;
        YP y2 = y1 + 128;
        float4 sumy = { 0.f, 0.f, 0.f, 0.f };
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l + 0];  sumy[0] += yl[l + 0];
            yl[l + 8] = y1[l + 32]; sumy[1] += yl[l + 8];
            yh[l + 0] = y2[l + 0];  sumy[2] += yh[l + 0];
            yh[l + 8] = y2[l + 32]; sumy[3] += yh[l + 8];
        }
        for (short row = 0; row < NR; ++row) {
            device const uchar * q2 = q1 + 64;
            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
            float4 acc1 = { 0.f }, acc2 = { 0.f };
            FOR_UNROLL (short l = 0; l < 8; ++l) {
                const uchar h = qh[l];
                acc1[0] += yl[l + 0] * (q1[l] & 0x0F);
                acc1[1] += yl[l + 8] * (q1[l] & 0xF0);
                acc1[2] += yh[l + 0] * (q2[l] & 0x0F);
                acc1[3] += yh[l + 8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l + 0] : 0.f;
                acc2[1] += h & hm2 ? yl[l + 8] : 0.f;
                acc2[2] += h & hm3 ? yh[l + 0] : 0.f;
                acc2[3] += h & hm4 ? yh[l + 8] : 0.f;
            }
            sumf[row] += dh[0] * (sc8[0] * (acc1[0] + 16.f * acc2[0]) + sc8[1] * (acc1[1] / 16.f + 16.f * acc2[1]) +
                                  sc8[4] * (acc1[2] + 16.f * acc2[2]) + sc8[5] * (acc1[3] / 16.f + 16.f * acc2[3])) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            q1 += row_bytes;
            qh += row_bytes;
            dh += row_bytes / 2;
            a += row_bytes / 2;
        }
        y1 += 4 * QK_K;
    }
    for (short row = 0; row < NR; ++row) out[row] = simd_sum(sumf[row]);
}

template <int NR, typename YP>
inline void dot_q6_K(device const uchar * w, uint row_bytes, uint k, YP yy, ushort lane, thread float * out) {
    constexpr uchar kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
    const int nb = k / QK_K;
    device const block_q6_K * x = (device const block_q6_K *) w;
    float sumf[NR] = { 0.f };
    float yl[16];
    const short tid = lane / 2, ix = lane % 2, ip = tid / 8, il = tid % 8, l0 = 4 * il, is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0, q_offset_l = 64 * ip + l0, q_offset_h = 32 * ip + l0;
    for (int i = ix; i < nb; i += 2) {
        device const uchar * q1 = x[i].ql + q_offset_l;
        device const uchar * q2 = q1 + 32;
        device const uchar * qh = x[i].qh + q_offset_h;
        device const char * sc = x[i].scales + is;
        device const half * dh = &x[i].d;
        YP y = yy + i * QK_K + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = y[l + 0];
            yl[4 * l + 1] = y[l + 32];
            yl[4 * l + 2] = y[l + 64];
            yl[4 * l + 3] = y[l + 96];
        }
        for (short row = 0; row < NR; ++row) {
            float4 sums = { 0.f, 0.f, 0.f, 0.f };
            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0] * ((char) ((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4 * l + 1] * ((char) ((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4 * l + 2] * ((char) ((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4 * l + 3] * ((char) ((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
            q1 += row_bytes;
            q2 += row_bytes;
            qh += row_bytes;
            sc += row_bytes;
            dh += row_bytes / 2;
        }
    }
    for (short row = 0; row < NR; ++row) out[row] = simd_sum(sumf[row]);
}

// ---------------------------------------------------------------------------------------------
// q4m: 8-token (verify) matvec for Q4_K / Q5_K on 8x8 simdgroup matrix multiplies, with contiguous loads and
// deferred scales. In an 8x8 simdgroup matrix lane L holds row (L/16)*4 + (L%8)/2, columns ((L%16)/8)*4 + (L%2)*2
// + {0, 1} (measured, same on the M1 and the M4 Max). A 64-column chunk of a row is one 32-byte qs group:
// sub-block 2p in the low nibbles, 2p + 1 in the high nibbles. MMA step s (0..7), column i (0..7) maps to chunk column
//   32 * (s / 4) + 8 * (i / 2) + 4 * ((s % 4) / 2) + (s % 4) % 2 + 2 * (i % 2)
// so lane (row r, columns c, c + 1) of the A fragment holds the nibbles of qs bytes 4c .. 4c + 7: one 8-byte load per
// chunk. Steps 0-3 are sub-block 2p, steps 4-7 sub-block 2p + 1: the quants enter the MMA unscaled (exact halves
// 1024 + q - 1024) and each sub-block's scale and min are applied to its 8x8 partial: acc += d*sc * P - dmin*m * sum(y).
// y is stored in the matching order YQ (above). Measured on the 24-layer stack (mvb, 796 MB): M4 Max 3.9 ms vs 8.9 ms
// for the previous dequantize-to-float MMA, M1 22.2 ms vs 54.7 ms (decode, one token: 2.9 / 17.7 ms).
// Tried and dropped (mvb, ledger rows m05, m06, m1-q4m-tiles2): y in half precision (no faster on either host), 1024 + q
// without the subtraction (no faster, 100x the error), half-precision partials (-2% on the M1 at 6000x the error).
inline half2 h1024(uint v) { return as_type<half2>(v | 0x64006400u) - half2(1024.h, 1024.h); }

// scale and min of sub-block j (0..7) from the 12 packed bytes held as 3 words (ggml get_scale_min_k4)
inline float2 sc_min(uint4 hdr, uint j) {
    const uint w0 = hdr.y, w1 = hdr.z, w2 = hdr.w;  // hdr.x = d, dmin
    if (j < 4) return float2((w0 >> (8 * j)) & 63, (w1 >> (8 * j)) & 63);
    const uint k = 8 * (j - 4);
    return float2(((w2 >> k) & 0xF) | (((w0 >> (k + 6)) & 3) << 4), ((w2 >> (k + 4)) & 0xF) | (((w1 >> (k + 6)) & 3) << 4));
}

// 8 * NT1 rows of each of NM matrices (w0, w1, w2: same type and row size; tiles of 8 rows, rb bytes apart) x 8 tokens
// over chunks c0, c0 + cstep, ... < c1 of k, sharing each chunk's y fragments; acc[m * NT1 + t]: the lane's two outputs
// (row r, tokens c, c + 1) of tile t of matrix m. Rows at or past n_rows read row n_rows - 1.
template <int NT1, int NM = 1>
inline void dot_q4m(uint wtype, device const uchar * w0, uint rb, uint n_rows, uint c0, uint c1, uint cstep,
                    device const float * y, ushort lane, thread float2 * acc,
                    device const uchar * w1 = nullptr, device const uchar * w2 = nullptr) {
    constexpr int NTL = NT1 * NM;
    const uint r = (lane / 16) * 4 + (lane % 8) / 2, c = ((lane % 16) / 8) * 4 + (lane % 2) * 2;
    const bool q5 = wtype == WT_Q5_K;
    const uint bsz = q5 ? 176 : 144, qoff = q5 ? 48 : 16;
    device const uchar * rows[NTL];
    FOR_UNROLL (uint t = 0; t < NTL; ++t) {
        device const uchar * w = t < NT1 ? w0 : t < 2 * NT1 ? w1 : w2;
        rows[t] = w + min(8 * (t % NT1) + r, n_rows - 1) * rb;
    }
    for (uint ch = c0; ch < c1; ch += cstep) {
        const uint blk = ch / 4, p = ch % 4;
        // B fragments of this chunk: 16 contiguous floats per lane
        device const float4 * yq = (device const float4 *) (y + ch * 512 + lane * 16);
        const float4 y0 = yq[0], y1 = yq[1], y2 = yq[2], y3 = yq[3];
        simdgroup_float8x8 bm[8];
        bm[0].thread_elements()[0] = y0[0]; bm[0].thread_elements()[1] = y0[1];
        bm[1].thread_elements()[0] = y0[2]; bm[1].thread_elements()[1] = y0[3];
        bm[2].thread_elements()[0] = y1[0]; bm[2].thread_elements()[1] = y1[1];
        bm[3].thread_elements()[0] = y1[2]; bm[3].thread_elements()[1] = y1[3];
        bm[4].thread_elements()[0] = y2[0]; bm[4].thread_elements()[1] = y2[1];
        bm[5].thread_elements()[0] = y2[2]; bm[5].thread_elements()[1] = y2[3];
        bm[6].thread_elements()[0] = y3[0]; bm[6].thread_elements()[1] = y3[1];
        bm[7].thread_elements()[0] = y3[2]; bm[7].thread_elements()[1] = y3[3];
        // per-token sums of y over each sub-block (lanes sharing columns c, c + 1 differ in lane bits 1, 2, 4)
        float2 s0 = float2(y0[0] + y0[2] + y1[0] + y1[2], y0[1] + y0[3] + y1[1] + y1[3]);
        float2 s1 = float2(y2[0] + y2[2] + y3[0] + y3[2], y2[1] + y2[3] + y3[1] + y3[3]);
        s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 16);
        s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 16);
        FOR_UNROLL (uint t = 0; t < NTL; ++t) {
            device const uchar * b = rows[t] + blk * bsz;
            const uint4 hdr = *(device const uint4 *) b;
            const uint2 q = *(device const uint2 *) (b + qoff + 32 * p + 4 * c);
            uint2 qa = q & 0x0F0F0F0Fu, qb = (q >> 4) & 0x0F0F0F0Fu;  // low (sub-block 2p), high (2p + 1) nibbles
            if (q5) {
                const uint2 h = *(device const uint2 *) (b + 16 + 4 * c);  // qh bytes of these 8 positions
                qa |= ((h >> (2 * p)) & 0x01010101u) << 4;
                qb |= ((h >> (2 * p + 1)) & 0x01010101u) << 4;
            }
            // step pairs: bytes (0, 2), (1, 3), (4, 6), (5, 7) of the lane's 8 bytes
            const half2 a0 = h1024(qa.x & 0x00FF00FFu), a1 = h1024((qa.x >> 8) & 0x00FF00FFu);
            const half2 a2 = h1024(qa.y & 0x00FF00FFu), a3 = h1024((qa.y >> 8) & 0x00FF00FFu);
            const half2 a4 = h1024(qb.x & 0x00FF00FFu), a5 = h1024((qb.x >> 8) & 0x00FF00FFu);
            const half2 a6 = h1024(qb.y & 0x00FF00FFu), a7 = h1024((qb.y >> 8) & 0x00FF00FFu);
            simdgroup_float8x8 P0 = make_filled_simdgroup_matrix<float, 8, 8>(0.f), P1 = P0;
            simdgroup_half8x8 am;
#define Q4M_MMA(P, a, i) am.thread_elements()[0] = a[0]; am.thread_elements()[1] = a[1]; \
                         simdgroup_multiply_accumulate(P, am, bm[i], P);
            Q4M_MMA(P0, a0, 0) Q4M_MMA(P0, a1, 1) Q4M_MMA(P0, a2, 2) Q4M_MMA(P0, a3, 3)
            Q4M_MMA(P1, a4, 4) Q4M_MMA(P1, a5, 5) Q4M_MMA(P1, a6, 6) Q4M_MMA(P1, a7, 7)
#undef Q4M_MMA
            const float2 dm = float2(as_type<half2>(hdr.x));
            const float2 m0 = sc_min(hdr, 2 * p), m1 = sc_min(hdr, 2 * p + 1);
            const float2 p0 = float2(P0.thread_elements()[0], P0.thread_elements()[1]);
            const float2 p1 = float2(P1.thread_elements()[0], P1.thread_elements()[1]);
            acc[t] += (dm.x * m0.x) * p0 - (dm.y * m0.y) * s0 + (dm.x * m1.x) * p1 - (dm.y * m1.y) * s1;
        }
    }
}

// a unit's simdgroup `part` of SPLIT takes interleaved chunks. (Tried: contiguous block ranges with the block header
// loaded once and the four chunks unrolled, so sub-block indices are constants: slower on both hosts, M4 Max stack
// 5.09 vs 3.89 ms, M1 39.4 vs 22.5 ms: register pressure.)
#define Q4M_DOT(NT1_, NM_, wtype, w0, rb, n_rows, part, nch, y, lane, acc, ...) \
    dot_q4m<NT1_, NM_>(wtype, w0, rb, n_rows, part, nch, SPLIT, y, lane, acc, ##__VA_ARGS__)

inline uint row_bytes_of(uint wtype, uint k) {
    return k / QK_K * (wtype == WT_Q4_K ? 144 : wtype == WT_Q5_K ? 176 : 210);
}

// single-row path: RU rows x NB token rows; every lane gets all results. One pass per token row (the second
// pass re-reads the unit's weights from the cache).
template <typename YP>
inline void dotR(uint wtype, device const uchar * w, uint rb, uint k, YP y, ushort lane, thread float (*out)[RU]) {
    for (uint b = 0; b < NB; ++b) {
        if (wtype == WT_Q4_K)      dot_q4_K<RU>(w, rb, k, y + b * k, lane, out[b]);
        else if (wtype == WT_Q5_K) dot_q5_K<RU>(w, rb, k, y + b * k, lane, out[b]);
        else                       dot_q6_K<RU>(w, rb, k, y + b * k, lane, out[b]);
    }
}

// element j of a per-lane result row, selected without dynamic register indexing
inline float pick(thread const float * a, uint j) {
    float r = a[0];
    for (uint i = 1; i < RU; ++i) r = i == j ? a[i] : r;
    return r;
}

inline float silu(float x) { return x / (1.0f + exp(-x)); }

// ---------------------------------------------------------------------------------------------
// prologue helpers (whole threadgroup)
// sum over the threadgroup; red (NSG floats) must not be reused before a later threadgroup barrier
inline float tg_sum(float v, threadgroup float * red, ushort sg, ushort lane) {
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.f;
    for (ushort i = 0; i < NSG; ++i) s += red[i];
    return s;
}

// one element of a Q6_K row, CPU dequantize_row_q6_K arithmetic (llama.cpp looks up embeddings on the CPU)
inline float q6_K_elem(device const block_q6_K * row, uint i) {
    device const block_q6_K & x = row[i / QK_K];
    const uint e = i % QK_K, n = e / 128, r = e % 128, grp = r / 32, l = r % 32, is = l / 16;
    const uchar qlb = x.ql[64 * n + l + (grp & 1) * 32];
    const int nib = grp < 2 ? (qlb & 0xF) : (qlb >> 4);
    const int hb = (x.qh[32 * n + l] >> (2 * grp)) & 3;
    const int q = (nib | (hb << 4)) - 32;
    const float d = (float) x.d;
    return d * (float) x.scales[8 * n + is + 2 * grp] * (float) q;
}

// one element of a Q4_K row, CPU dequantize_row_q4_K arithmetic
inline float q4_K_elem(device const block_q4_K * row, uint i) {
    device const block_q4_K & x = row[i / QK_K];
    const uint e = i % QK_K, j = e / 32, l = e % 32;
    uint sc, m;
    if (j < 4) { sc = x.scales[j] & 63; m = x.scales[j + 4] & 63; }
    else { sc = (x.scales[j + 4] & 0xF) | ((x.scales[j - 4] >> 6) << 4); m = (x.scales[j + 4] >> 4) | ((x.scales[j] >> 6) << 4); }
    const uchar q = x.qs[(j / 2) * 32 + l];
    const float d1 = (float) x.d * (float) sc, m1 = (float) x.dmin * (float) m;
    return d1 * (float) ((j & 1) ? (q >> 4) : (q & 0xF)) - m1;
}

// embedding row element: P.aux[6] = token_embd type
inline float embd_elem(constant Phase & P, constant Params & prm, device const uchar * W, uint b, uint i) {
    if (P.aux[6] == WT_Q4_K) return q4_K_elem((device const block_q4_K *) (W + P.aux_w[3]) + prm.tokens[b] * (H / QK_K), i);
    return q6_K_elem((device const block_q6_K *) (W + P.aux_w[3]) + prm.tokens[b] * (H / QK_K), i);
}

// y = rms_norm(h) * w (ggml: (x * scale) * w, scale = 1/sqrt(mean(x^2) + eps)) for each of NB rows.
// PRO_RESID first forms h = hres + rms_norm(f) * w_post. Threadgroup 0 stores h. Each thread keeps its
// PE4 float4s in registers: all loads issue up front, then two reductions; y is written once (with a
// device-memory y every threadgroup writes the same final values, so no transient value is visible).
#define PE4 ((H / 4 + NT - 1) / NT)  // NT need not divide H/4 (e.g. 384 threads on the M1): excess entries are 0
template <typename YP>
inline void prologue_norm(constant Phase & P, constant Params & prm, device const uchar * W, device float * A,
                          YP y, threadgroup float * red, uint tg, uint tid, ushort sg, ushort lane) {
    device const float4 * wpost = (device const float4 *) (W + P.w_post);
    device const float4 * wpre = (device const float4 *) (W + P.w_pre);
    for (uint b = 0; b < NB; ++b) {
        device const float4 * fin = (device const float4 *) (A + P.in + b * H);
        device const float4 * hin = (device const float4 *) (A + P.hres + b * H);
        float4 hv[PE4], fv[PE4], wq[PE4], wp[PE4];
        FOR_UNROLL (uint e = 0; e < PE4; ++e) {
            const uint i = min(tid + e * NT, (uint) H / 4 - 1);  // clamped: read in bounds, zeroed below
            const bool in = tid + e * NT < H / 4;
            wp[e] = wpre[i];
            if (P.pro == PRO_RESID) {
                fv[e] = fin[i];
                hv[e] = hin[i];
                wq[e] = wpost[i];
            } else {
                for (uint j = 0; j < 4; ++j) hv[e][j] = embd_elem(P, prm, W, b, 4 * i + j);
                fv[e] = 0.f;
            }
            if (!in) { hv[e] = 0.f; fv[e] = 0.f; }
        }
        if (P.pro == PRO_RESID) {
            float ss = 0.f;
            FOR_UNROLL (uint e = 0; e < PE4; ++e) ss += dot(fv[e], fv[e]);
            const float mean = tg_sum(ss, red, sg, lane) / H;
            const float scale_f = 1.0f / sqrt(mean + prm.eps);
            FOR_UNROLL (uint e = 0; e < PE4; ++e) hv[e] = hv[e] + (fv[e] * scale_f) * wq[e];
        }
        float ss = 0.f;
        FOR_UNROLL (uint e = 0; e < PE4; ++e) ss += dot(hv[e], hv[e]);
        const float mean = tg_sum(ss, red + NSG, sg, lane) / H;
        const float scale = 1.0f / sqrt(mean + prm.eps);
        FOR_UNROLL (uint e = 0; e < PE4; ++e) {
            const uint i = tid + e * NT;
            if (i >= H / 4) break;
            const float4 v = (hv[e] * scale) * wp[e];
            for (uint j = 0; j < 4; ++j) y[YI(4 * i + j, b, H)] = v[j];
            if (tg == 0) ((device float4 *) (A + P.hout + b * H))[i] = hv[e];
        }
        if (NB > 1) threadgroup_barrier(mem_flags::mem_threadgroup);  // red is reused by the next row
    }
}

// Multi-row prologue: one simdgroup per token row (rows are independent), so the rows' reductions run
// in parallel and need no threadgroup barriers. Same arithmetic as prologue_norm, summed per lane.
inline void prologue_rows(constant Phase & P, constant Params & prm, device const uchar * W, device float * A,
                          device float * y, uint tg, ushort sg, ushort lane) {
    device const float4 * wpost = (device const float4 *) (W + P.w_post);
    device const float4 * wpre = (device const float4 *) (W + P.w_pre);
    constexpr uint E = H / 4 / 32;  // float4s per lane
    for (uint b = sg; b < NB; b += NSG) {
        device const float4 * fin = (device const float4 *) (A + P.in + b * H);
        device const float4 * hin = (device const float4 *) (A + P.hres + b * H);
        float4 hv[E];
        if (P.pro == PRO_RESID) {
            float ss = 0.f;
            FOR_UNROLL (uint e = 0; e < E; ++e) { const float4 f = fin[lane + 32 * e]; ss += dot(f, f); }
            const float scale_f = 1.0f / sqrt(simd_sum(ss) / H + prm.eps);
            FOR_UNROLL (uint e = 0; e < E; ++e) {
                const uint i = lane + 32 * e;
                hv[e] = hin[i] + (fin[i] * scale_f) * wpost[i];
            }
        } else {
            FOR_UNROLL (uint e = 0; e < E; ++e) for (uint j = 0; j < 4; ++j) hv[e][j] = embd_elem(P, prm, W, b, 4 * (lane + 32 * e) + j);
        }
        float ss = 0.f;
        FOR_UNROLL (uint e = 0; e < E; ++e) ss += dot(hv[e], hv[e]);
        const float scale = 1.0f / sqrt(simd_sum(ss) / H + prm.eps);
        FOR_UNROLL (uint e = 0; e < E; ++e) {
            const uint i = lane + 32 * e;
            const float4 v = (hv[e] * scale) * wpre[i];
            for (uint j = 0; j < 4; ++j) y[YI(4 * i + j, b, H)] = v[j];
            if (tg == 0) ((device float4 *) (A + P.hout + b * H))[i] = hv[e];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// ops
template <typename YP>
inline void op_matvec(constant Phase & P, device const uchar * W, device float * A, YP y, uint gsg, uint n_sg,
                      ushort lane) {
    const uint rb = row_bytes_of(P.seg[0].wtype, P.k);
    // units are dealt round robin, so at any moment the simdgroups of the whole GPU stream one
    // contiguous region of the matrix (measured faster than contiguous per-simdgroup ranges)
    if (P.op == OP_SWIGLU) {
        const uint n = P.seg[0].n_rows;
        for (uint r = gsg * RU; r < n; r += n_sg * RU) {
            float g[NB][RU], u[NB][RU];
            dotR(P.seg[0].wtype, W + P.seg[0].w + (ulong) r * rb, rb, P.k, y, lane, g);
            dotR(P.seg[1].wtype, W + P.seg[1].w + (ulong) r * rb, rb, P.k, y, lane, u);
            if (lane < RU) {  // the SwiGLU output feeds the down matvec
                for (uint b = 0; b < NB; ++b) A[P.seg[0].out + YI(r + lane, b, n)] = silu(pick(g[b], lane)) * pick(u[b], lane);
            }
        }
        return;
    }
    // units of RU rows enumerated across the segments
    uint total = 0;
    for (uint s = 0; s < P.n_seg; ++s) total += P.seg[s].n_rows / RU;
    for (uint u = gsg; u < total; u += n_sg) {
        uint s = 0, r = u * RU;
        while (r >= P.seg[s].n_rows) { r -= P.seg[s].n_rows; ++s; }
        constant Seg & S = P.seg[s];
        const uint rbs = row_bytes_of(S.wtype, P.k);
        float o[NB][RU];
        dotR(S.wtype, W + S.w + (ulong) r * rbs, rbs, P.k, y, lane, o);
        if (lane < RU) {
            for (uint b = 0; b < NB; ++b) A[S.out + b * S.n_rows + r + lane] = pick(o[b], lane);
        }
    }
}

// ---- MMA path (8 token rows): units of 8 * TN rows, SPLIT simdgroups per unit, UGROUPS units per threadgroup.
// Every simdgroup of a threadgroup runs the same number of rounds (threadgroup barriers inside).
// scr: NSG * TN * 32 * 3 float2. Partials of simdgroup sg, tile t, matrix m (0..2) at ((m * NSG + sg) * TN + t) * 32 + lane.
#define Q4M_ROWS (8 * TN)
inline uint q4m_lane_row(uint l) { return (l / 16) * 4 + (l % 8) / 2; }
inline uint q4m_lane_col(uint l) { return ((l % 16) / 8) * 4 + (l % 2) * 2; }

// sum the SPLIT partials of entry i (tile t = i / 32, lane l = i % 32) of matrix m for unit group gi
inline float2 q4m_sum(threadgroup const float2 * scr, uint m, uint gi, uint i) {
    float2 s = 0.f;
    for (uint g = 0; g < SPLIT; ++g) s += scr[((m * NSG + gi * SPLIT + g) * TN) * 32 + i];
    return s;
}

// matvec segments (or SwiGLU over seg[0] = gate, seg[1] = up); y in q4m order
inline void op_matvec8(constant Phase & P, device const uchar * W, device float * A, device const float * y,
                       uint tg, uint n_tg, uint tid, ushort sg, ushort lane, threadgroup float2 * scr) {
    const uint gi = sg / SPLIT, part = sg % SPLIT, nch = P.k / 64;
    const bool swiglu = P.op == OP_SWIGLU;
    uint total = 0;
    if (swiglu) total = P.seg[0].n_rows / Q4M_ROWS;
    else for (uint s = 0; s < P.n_seg; ++s) total += (P.seg[s].n_rows + Q4M_ROWS - 1) / Q4M_ROWS;
    for (uint base = tg * UGROUPS; base < total; base += n_tg * UGROUPS) {
        const uint u = base + gi;
        uint s = 0, r0 = 0;
        if (u < total) {
            r0 = u * Q4M_ROWS;
            if (!swiglu) {
                while (r0 >= (P.seg[s].n_rows + Q4M_ROWS - 1) / Q4M_ROWS * Q4M_ROWS) {
                    r0 -= (P.seg[s].n_rows + Q4M_ROWS - 1) / Q4M_ROWS * Q4M_ROWS;
                    ++s;
                }
            }
            constant Seg & S = P.seg[s];
            const uint rb = row_bytes_of(S.wtype, P.k);
            if (Q4M_SHARE && swiglu && P.seg[1].wtype == S.wtype) {  // gate and up tiles share each chunk's y fragments
                float2 acc[2 * TN];
                FOR_UNROLL (uint t = 0; t < 2 * TN; ++t) acc[t] = 0.f;
                Q4M_DOT(TN, 2, S.wtype, W + S.w + (ulong) r0 * rb, rb, S.n_rows - r0, part, nch, y, lane, acc,
                        W + P.seg[1].w + (ulong) r0 * rb);
                FOR_UNROLL (uint t = 0; t < TN; ++t) {
                    scr[(sg * TN + t) * 32 + lane] = acc[t];
                    scr[((NSG + sg) * TN + t) * 32 + lane] = acc[TN + t];
                }
            } else {
                float2 acc[TN];
                FOR_UNROLL (uint t = 0; t < TN; ++t) acc[t] = 0.f;
                Q4M_DOT(TN, 1, S.wtype, W + S.w + (ulong) r0 * rb, rb, S.n_rows - r0, part, nch, y, lane, acc);
                FOR_UNROLL (uint t = 0; t < TN; ++t) scr[(sg * TN + t) * 32 + lane] = acc[t];
                if (swiglu) {
                    constant Seg & S1 = P.seg[1];
                    const uint rb1 = row_bytes_of(S1.wtype, P.k);
                    FOR_UNROLL (uint t = 0; t < TN; ++t) acc[t] = 0.f;
                    Q4M_DOT(TN, 1, S1.wtype, W + S1.w + (ulong) r0 * rb1, rb1, S1.n_rows - r0, part, nch, y, lane, acc);
                    FOR_UNROLL (uint t = 0; t < TN; ++t) scr[((NSG + sg) * TN + t) * 32 + lane] = acc[t];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (u < total) {
            constant Seg & S = P.seg[s];
            for (uint i = part * 32 + lane; i < TN * 32; i += SPLIT * 32) {
                const uint t = i / 32, l = i % 32;
                const uint row = r0 + 8 * t + q4m_lane_row(l), col = q4m_lane_col(l);
                if (row >= S.n_rows) continue;
                const float2 v = q4m_sum(scr, 0, gi, i);
                if (swiglu) {  // feeds the down matvec: q4m order
                    const float2 up = q4m_sum(scr, 1, gi, i);
                    A[S.out + YQ(row, col)] = silu(v[0]) * up[0];
                    A[S.out + YQ(row, col + 1)] = silu(v[1]) * up[1];
                } else {
                    A[S.out + col * S.n_rows + row] = v[0];
                    A[S.out + (col + 1) * S.n_rows + row] = v[1];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// short conv: in_proj rows [0,H) = B, [H,2H) = C, [2H,3H) = X; bx = B*X goes into a per-channel ring
// indexed by position; out = C * sum_j ring[pos-6+j] * w[j]  (ggml: concat(state, bx) -> ssm_conv).
template <typename YP>
inline void op_conv(constant Phase & P, constant Params & prm, device const uchar * W, device float * A,
                    device float * ring, YP y, uint gsg, uint n_sg, ushort lane) {
    constant Seg & S = P.seg[0];
    const uint rb = row_bytes_of(S.wtype, P.k);
    device const float * cw = (device const float *) (W + P.aux_w[0]);
    const uint rl = prm.ring, rm = rl - 1;
    device float * rg = ring + (ulong) P.aux[0] * rl;
    for (uint c = gsg * RU; c < H; c += n_sg * RU) {
        // lanes 0..RU-1 (one channel each) load the history and taps first: the loads overlap the dots
        const uint ch = c + min((uint) lane, (uint) RU - 1);
        device float * hist = rg + ch * rl;
        float x[CONV_L - 1 + NB], wc[CONV_L];
        FOR_UNROLL (int j = 0; j < CONV_L; ++j) wc[j] = cw[ch * CONV_L + j];
        FOR_UNROLL (int j = 0; j < CONV_L - 1; ++j) {
            const int q = (int) prm.pos - (CONV_L - 1) + j;
            x[j] = q >= 0 ? hist[q & rm] : 0.f;
        }
        float vb[NB][RU], vc[NB][RU], vx[NB][RU];
        dotR(S.wtype, W + S.w + (ulong) c * rb, rb, P.k, y, lane, vb);
        dotR(S.wtype, W + S.w + (ulong) (H + c) * rb, rb, P.k, y, lane, vc);
        dotR(S.wtype, W + S.w + (ulong) (2 * H + c) * rb, rb, P.k, y, lane, vx);
        if (lane < RU) {
            for (uint b = 0; b < NB; ++b) {
                const float bx = pick(vb[b], lane) * pick(vx[b], lane);
                x[CONV_L - 1 + b] = bx;
                hist[(prm.pos + b) & rm] = bx;
                float s = 0.f;
                FOR_UNROLL (int j = 0; j < CONV_L; ++j) s += x[b + j] * wc[j];
                A[S.out + YI(ch, b, H)] = pick(vc[b], lane) * s;  // feeds out_proj (matvec input layout)
            }
        }
    }
}

// short conv on the MMA path: units of 8 * TN channels (B, C and X rows of in_proj); after the reduction one thread per
// channel runs the causal conv over the 8 token rows in order. cv: UGROUPS * 2 * 8 * TN * 8 floats (bx, c).
inline void op_conv8(constant Phase & P, constant Params & prm, device const uchar * W, device float * A, device float * ring,
                     device const float * y, uint tg, uint n_tg, uint tid, ushort sg, ushort lane, threadgroup float2 * scr,
                     threadgroup float * cv) {
    constant Seg & S = P.seg[0];
    const uint rb = row_bytes_of(S.wtype, P.k), nch = P.k / 64;
    const uint gi = sg / SPLIT, part = sg % SPLIT;
    device const float * cw = (device const float *) (W + P.aux_w[0]);
    const uint rl = prm.ring, rm = rl - 1;
    device float * rg = ring + (ulong) P.aux[0] * rl;
    const uint total = H / Q4M_ROWS;
    threadgroup float * cbx = cv + gi * 2 * Q4M_ROWS * 8, * ccc = cbx + Q4M_ROWS * 8;  // [channel][token]
    for (uint base = tg * UGROUPS; base < total; base += n_tg * UGROUPS) {
        const uint u = base + gi, c0 = u * Q4M_ROWS;
        float2 acc[3][TN];  // B, C, X partials of this simdgroup (one pass over y: the three share its fragments)
        FOR_UNROLL (uint m = 0; m < 3; ++m) FOR_UNROLL (uint t = 0; t < TN; ++t) acc[m][t] = 0.f;
        if (u < total) {
            if (Q4M_SHARE) {
                Q4M_DOT(TN, 3, S.wtype, W + S.w + (ulong) c0 * rb, rb, Q4M_ROWS, part, nch, y, lane, &acc[0][0],
                        W + S.w + (ulong) (H + c0) * rb, W + S.w + (ulong) (2 * H + c0) * rb);
            } else {
                FOR_UNROLL (uint m = 0; m < 3; ++m) {
                    Q4M_DOT(TN, 1, S.wtype, W + S.w + (ulong) (m * H + c0) * rb, rb, Q4M_ROWS, part, nch, y, lane, acc[m]);
                }
            }
        }
        // reduce one matrix at a time through the same scratch (a third of the threadgroup memory): B, then X (bx =
        // B * X), then C
        FOR_UNROLL (uint r = 0; r < 3; ++r) {
            const uint m = r == 0 ? 0 : r == 1 ? 2 : 1;
            FOR_UNROLL (uint t = 0; t < TN; ++t) scr[(sg * TN + t) * 32 + lane] = acc[m][t];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (u < total) {
                for (uint i = part * 32 + lane; i < TN * 32; i += SPLIT * 32) {
                    const uint t = i / 32, l = i % 32;
                    const uint ch = 8 * t + q4m_lane_row(l), col = q4m_lane_col(l);
                    const float2 v = q4m_sum(scr, 0, gi, i);
                    threadgroup float * dst = r == 2 ? ccc : cbx;
                    if (r == 1) {
                        dst[ch * 8 + col] *= v[0];
                        dst[ch * 8 + col + 1] *= v[1];
                    } else {
                        dst[ch * 8 + col] = v[0];
                        dst[ch * 8 + col + 1] = v[1];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint ti = part * 32 + lane;
        if (u < total && ti < Q4M_ROWS) {
            const uint ch = c0 + ti;
            device float * hist = rg + ch * rl;
            float x[CONV_L - 1 + NB], wc[CONV_L];
            FOR_UNROLL (int j = 0; j < CONV_L; ++j) wc[j] = cw[ch * CONV_L + j];
            FOR_UNROLL (int j = 0; j < CONV_L - 1; ++j) {
                const int q = (int) prm.pos - (CONV_L - 1) + j;
                x[j] = q >= 0 ? hist[q & rm] : 0.f;
            }
            FOR_UNROLL (uint b = 0; b < NB; ++b) {
                const float bx = cbx[ti * 8 + b];
                x[CONV_L - 1 + b] = bx;
                hist[(prm.pos + b) & rm] = bx;
                float s = 0.f;
                FOR_UNROLL (int j = 0; j < CONV_L; ++j) s += x[b + j] * wc[j];
                A[S.out + YQ(ch, b)] = ccc[ti * 8 + b] * s;  // feeds out_proj
            }
        }
        // the next round's reduction overwrites cbx / ccc only after its own first barrier
    }
}

// q/k RMS norm (per head) + NEOX RoPE; k and v stored as f16 at their positions. One simdgroup per
// (row, head); lane l holds dims 4l..4l+3, its rotation partner is lane l^16.
inline void op_qkrope(constant Phase & P, constant Params & prm, device const uchar * W, device float * A,
                      device half * kc, device half * vc, uint gsg, uint n_sg, ushort lane) {
    const uint n_units = NB * (NQH + 2 * NKVH);
    const float base = as_type<float>(P.aux[5]);
    const bool has_ff = P.aux_w[2] != 0;
    device const float * ff = (device const float *) (W + P.aux_w[2]);
    for (uint u = gsg; u < n_units; u += n_sg) {
        const uint b = u / (NQH + 2 * NKVH), hh = u % (NQH + 2 * NKVH);
        const uint p = prm.pos + b;
        const ulong cache_row = ((ulong) P.aux[4] * prm.ctx + p) * NKVH;  // [layer][pos][kv head][dim]
        if (hh >= NQH + NKVH) {  // v: copy to the cache
            const uint g = hh - NQH - NKVH;
            const float4 v = ((device const float4 *) (A + P.aux[2] + b * NKVH * HD + g * HD))[lane];
            ((device half4 *) (vc + (cache_row + g) * HD))[lane] = half4(v);
            continue;
        }
        const bool is_q = hh < NQH;
        const uint head = is_q ? hh : hh - NQH;
        device const float * src = is_q ? A + P.aux[0] + b * H + head * HD : A + P.aux[1] + b * NKVH * HD + head * HD;
        device const float * nw = (device const float *) (W + (is_q ? P.aux_w[0] : P.aux_w[1]));
        float4 x = ((device const float4 *) src)[lane];
        const float mean = simd_sum(dot(x, x)) / HD;
        const float scale = 1.0f / sqrt(mean + prm.eps);
        x = (x * scale) * ((device const float4 *) nw)[lane];
        const float4 partner = simd_shuffle_xor(x, 16);
        const bool lo = lane < 16;
        float4 r;
        for (short e = 0; e < 4; ++e) {
            const uint i = (lane % 16) * 4 + e;   // pair index: dims (i, i + 64)
            const float theta_base = (float) p;
            const float inv_ndims = -1.f / HD;
            const float theta = theta_base * pow(base, inv_ndims * (2 * i));  // as ggml's rope kernel
            const float freq_factor = has_ff ? ff[i] : 1.0f;
            const float th = theta / freq_factor;
            const float cos_theta = cos(th), sin_theta = sin(th);
            const float x0 = lo ? x[e] : partner[e], x1 = lo ? partner[e] : x[e];
            r[e] = lo ? x0 * cos_theta - x1 * sin_theta : x0 * sin_theta + x1 * cos_theta;
        }
        if (is_q) ((device float4 *) (A + P.aux[3] + b * H + head * HD))[lane] = r;
        else      ((device half4 *) (kc + (cache_row + head) * HD))[lane] = half4(r);
    }
}

// attention for one (row, query head, chunk of ATT_CH positions) per simdgroup, one position per lane:
// the lane computes its full q.k, the chunk's softmax statistics come from simd_max/simd_sum, then
// the lanes accumulate V (lane l owns dims 4l..4l+3). Partials (max, sum, acc) go to memory; the
// last chunk of a (row, head) to finish combines them and applies the sigmoid output gate.
inline void op_attn(constant Phase & P, constant Params & prm, device float * A, device const half * kc,
                    device const half * vc, device atomic_uint * cnt, uint gsg, uint n_sg, ushort lane) {
    const uint window = P.aux[5];
    const float scale = as_type<float>(P.aux[6]);
    device float * part = A + P.aux[7];
    const ulong layer = (ulong) P.aux[4] * prm.ctx;
    uint n_units = 0;
    for (uint b = 0; b < NB; ++b) {
        const uint end = prm.pos + b + 1, start = window && end > window ? end - window : 0;
        n_units += NQH * ((end - start + ATT_CH - 1) / ATT_CH);
    }
    for (uint u = gsg; u < n_units; u += n_sg) {
        uint b = 0, rem = u, end = 0, start = 0, nch = 0;
        for (;; ++b) {
            end = prm.pos + b + 1;
            start = window && end > window ? end - window : 0;
            nch = (end - start + ATT_CH - 1) / ATT_CH;
            if (rem < NQH * nch) break;
            rem -= NQH * nch;
        }
        const uint h = rem / nch, c = rem % nch, g = h / (NQH / NKVH);
        const uint p0 = start + c * ATT_CH, n = min(p0 + ATT_CH, end) - p0;
        float s = -INFINITY;
        if (lane < n) {
            device const half4 * kr = (device const half4 *) (kc + ((layer + p0 + lane) * NKVH + g) * HD);
            device const float4 * q4 = (device const float4 *) (A + P.aux[0] + b * H + h * HD);
            float acc = 0.f;
            UNROLL_N(ATT_UNROLL) for (short i = 0; i < HD / 4; ++i) acc += dot(float4(kr[i]), q4[i]);
            s = acc * scale;
        }
        const float m = simd_max(s);
        const float e = lane < n ? exp(s - m) : 0.f;
        const float l = simd_sum(e);
        float4 o = 0.f;
        // unrolled so the V loads issue together; rows past the chunk are clamped (their weight is 0)
        UNROLL_N(ATT_UNROLL) for (uint i = 0; i < ATT_CH; ++i) {
            const float ei = simd_shuffle(e, (ushort) i);
            const uint pi = p0 + min(i, n - 1);
            o += ei * float4(((device const half4 *) (vc + ((layer + pi) * NKVH + g) * HD))[lane]);
        }
        device float * all = part + (b * NQH + h) * ATT_MAXCH * (HD + 2);
        device float * mine = all + c * (HD + 2);
        ((device float4 *) mine)[lane] = o;
        if (lane == 0) { mine[HD] = m; mine[HD + 1] = l; }
        // the last finisher of this (row, head) combines
        simdgroup_barrier(mem_flags::mem_device);
        uint last = 0;
        if (lane == 0) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            last = atomic_fetch_add_explicit(&cnt[b * NQH + h], 1u, memory_order_relaxed) == nch - 1;
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        }
        last = simd_broadcast(last, 0);
        if (!last) continue;
        // combine: chunk statistics one per lane (32 chunks at a time), weighted sum unrolled
        float M = -INFINITY;
        for (uint base = 0; base < nch; base += 32) {
            M = max(M, simd_max(base + lane < nch ? all[(base + lane) * (HD + 2) + HD] : -INFINITY));
        }
        float L = 0.f;
        float4 out = 0.f;
        for (uint base = 0; base < nch; base += 32) {
            const uint cl = base + lane;
            const float wgt = cl < nch ? exp(all[cl * (HD + 2) + HD] - M) : 0.f;
            L += simd_sum(cl < nch ? wgt * all[cl * (HD + 2) + HD + 1] : 0.f);
            UNROLL_N(ATT_UNROLL) for (uint i = 0; i < 32; ++i) {
                const uint cc = min(base + i, nch - 1);
                out += simd_shuffle(wgt, (ushort) i) * ((device const float4 *) (all + cc * (HD + 2)))[lane];
            }
        }
        const float gate = 1.0f / (1.0f + exp(-A[P.aux[1] + b * NQH + h]));
        const float4 res = (out / L) * gate;  // feeds o_proj: interleaved layout
        if (SP) ((device float4 *) (A + P.aux[3] + b * H + h * HD))[lane] = res;
        else for (uint j = 0; j < 4; ++j) A[P.aux[3] + YI(h * HD + 4 * lane + j, b, H)] = res[j];
        if (lane == 0) atomic_store_explicit(&cnt[b * NQH + h], 0u, memory_order_relaxed);
    }
}

// Touch the first weight rows this threadgroup reads in the next phase (weights never depend on
// activations), overlapping their DRAM latency with the barrier wait.
inline uint prefetch_next(constant Phase & P, device const uchar * W, uint tg, uint tid, uint bytes) {
    if (bytes == 0 || P.n_seg == 0 || P.op == OP_ATTN || P.op == OP_QKROPE) return 0;
    const uint rb = row_bytes_of(P.seg[0].wtype, P.k);
    const uint first = SP ? tg * NSG * RU : tg * UGROUPS * Q4M_ROWS;  // this threadgroup's first rows (units dealt round robin)
    if (first >= P.seg[0].n_rows) return 0;
    device const uint4 * p = (device const uint4 *) (W + P.seg[0].w + (ulong) first * rb);
    const uint n = min(bytes, (P.seg[0].n_rows - first) * rb) / 16;
    uint acc = 0;
    for (uint i = tid; i < n; i += NT) acc ^= p[i].x;
    return acc;
}

// ---------------------------------------------------------------------------------------------
// One phase on a grid of n_tg threadgroups: prologue (staging / norms), then the op. Used by the persistent mk_run
// (grid barriers between phases) and by mk_phase (one dispatch per phase).
inline void run_phase(constant Phase & P, constant Params & prm, device const uchar * W, device float * A,
                      device atomic_uint * sync, device half * kc, device half * vc, device float * ring,
                      uint tg, uint n_tg, uint tid, ushort sg, ushort lane, threadgroup float * ys, threadgroup float * scr,
                      threadgroup float * cv, threadgroup float * red) {
    const uint n_sg = n_tg * NSG;
    const uint gsg = tg * NSG + sg;
    const bool norm = P.pro != PRO_STAGE;
    const bool use_tg = YS_TG && P.k == H;  // staged in threadgroup memory
    if (norm) {
        if (use_tg)      prologue_norm(P, prm, W, A, ys, red, tg, tid, sg, lane);
        else if (!SP) {  // a norm-only phase (the host splits them off on this path): one threadgroup writes y once
            if (P.op != OP_NONE || tg == 0) prologue_rows(P, prm, W, A, A + P.ynorm, tg, sg, lane);
        }
        else             prologue_norm(P, prm, W, A, (device float *) (A + P.ynorm), red, tg, tid, sg, lane);
    } else if (use_tg && P.op != OP_ATTN && P.op != OP_QKROPE) {
        for (uint i = tid; i < NB * H; i += NT) ys[i] = A[P.in + i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    device const float * yd = A + (norm ? P.ynorm : P.in);
    if (P.op == OP_MATVEC || P.op == OP_SWIGLU) {
        if (!SP)         op_matvec8(P, W, A, yd, tg, n_tg, tid, sg, lane, (threadgroup float2 *) scr);
        else if (use_tg) op_matvec(P, W, A, (threadgroup const float *) ys, gsg, n_sg, lane);
        else             op_matvec(P, W, A, yd, gsg, n_sg, lane);
    } else if (P.op == OP_CONV) {
        if (!SP)         op_conv8(P, prm, W, A, ring, yd, tg, n_tg, tid, sg, lane, (threadgroup float2 *) scr, cv);
        else if (use_tg) op_conv(P, prm, W, A, ring, (threadgroup const float *) ys, gsg, n_sg, lane);
        else             op_conv(P, prm, W, A, ring, yd, gsg, n_sg, lane);
    } else if (P.op == OP_QKROPE) {
        op_qkrope(P, prm, W, A, kc, vc, gsg, n_sg, lane);
    } else if (P.op == OP_ATTN) {
        op_attn(P, prm, A, kc, vc, sync + 8, gsg, n_sg, lane);
    } else if (P.op == OP_NONE && norm && use_tg && tg == 0) {
        for (uint i = tid; i < NB * H; i += NT) A[P.ynorm + i] = ys[i];  // a norm-only phase's y, for the next phase
    }
}

// MAX_TPT caps the register budget (the compiler must fit MAX_TPT threads per threadgroup), which is
// what lets several threadgroups share a core on the M1.
#ifdef MAX_TPT
[[max_total_threads_per_threadgroup(MAX_TPT)]]
#endif
kernel void mk_run(constant Params & prm        [[buffer(0)]],
                   constant Phase * phases      [[buffer(1)]],
                   device const uchar * W       [[buffer(2)]],
                   device float * A             [[buffer(3)]],
                   device atomic_uint * sync    [[buffer(4)]],  // [0] barrier, [1] fail, [8..] attention counters
                   device half * kc             [[buffer(5)]],
                   device half * vc             [[buffer(6)]],
                   device float * ring          [[buffer(7)]],
                   constant uint & prefetch     [[buffer(8)]],
                   uint tg                      [[threadgroup_position_in_grid]],
                   uint tid                     [[thread_index_in_threadgroup]],
                   ushort sg                    [[simdgroup_index_in_threadgroup]],
                   ushort lane                  [[thread_index_in_simdgroup]]) {
    threadgroup float ys[YS_LEN];
    threadgroup float scr[SCR_LEN];
    threadgroup float cv[SP ? 1 : UGROUPS * 2 * Q4M_ROWS * 8];
    threadgroup float red[2 * NSG];
    threadgroup uint tg_ok;
    uint dummy = 0;

    for (uint ph = 0; ph < prm.n_phases; ++ph) {
        run_phase(phases[ph], prm, W, A, sync, kc, vc, ring, tg, prm.n_tg, tid, sg, lane, ys, scr, cv, red);
        if (ph + 1 < prm.n_phases) {
            grid_arrive(&sync[0], tid);
            dummy ^= prefetch_next(phases[ph + 1], W, tg, tid, prefetch);
            if (!grid_wait(&sync[0], &sync[1], (ph + 1) * prm.n_tg, prm.spin_limit, tid, &tg_ok)) return;
        }
    }
    if (dummy == 0x9E3779B9u) A[0] += 1e-30f;  // keeps the prefetch loads alive
}

// One phase per dispatch (buffer 1 points at the phase): the grid size is the phase's own, chosen by the host; the
// dispatch boundary replaces the grid barrier (0.7 us vs 2.1-2.8 us on the M4 Max) and every phase runs at the
// occupancy of its own register budget. The host records the whole step once in an indirect command buffer.
kernel void mk_phase(constant Params & prm        [[buffer(0)]],
                     constant Phase * phases      [[buffer(1)]],
                     device const uchar * W       [[buffer(2)]],
                     device float * A             [[buffer(3)]],
                     device atomic_uint * sync    [[buffer(4)]],
                     device half * kc             [[buffer(5)]],
                     device half * vc             [[buffer(6)]],
                     device float * ring          [[buffer(7)]],
                     uint tg                      [[threadgroup_position_in_grid]],
                     uint n_tg                    [[threadgroups_per_grid]],
                     uint tid                     [[thread_index_in_threadgroup]],
                     ushort sg                    [[simdgroup_index_in_threadgroup]],
                     ushort lane                  [[thread_index_in_simdgroup]]) {
    threadgroup float ys[YS_LEN];
    threadgroup float scr[SCR_LEN];
    threadgroup float cv[SP ? 1 : UGROUPS * 2 * Q4M_ROWS * 8];
    threadgroup float red[2 * NSG];
    run_phase(phases[0], prm, W, A, sync, kc, vc, ring, tg, n_tg, tid, sg, lane, ys, scr, cv, red);
}

// LM head: a plain (non-persistent) matvec over phases[0]. A separate function so its register budget
// is the matvec's alone (the megakernel's is the maximum over all its ops), for full occupancy.
kernel void mk_head(constant Params & prm        [[buffer(0)]],
                    constant Phase * phases      [[buffer(1)]],
                    device const uchar * W       [[buffer(2)]],
                    device float * A             [[buffer(3)]],
                    uint tg                      [[threadgroup_position_in_grid]],
                    uint tid                     [[thread_index_in_threadgroup]],
                    ushort sg                    [[simdgroup_index_in_threadgroup]],
                    ushort lane                  [[thread_index_in_simdgroup]]) {
    threadgroup float ys[YS_LEN];
    threadgroup float scr[SCR_LEN];
    constant Phase & P = phases[0];
    const uint n_sg = prm.n_tg * NSG, gsg = tg * NSG + sg;
    if (!SP) {
        op_matvec8(P, W, A, (device const float *) (A + P.in), tg, prm.n_tg, tid, sg, lane, (threadgroup float2 *) scr);
    } else if (YS_TG) {
        for (uint i = tid; i < NB * H; i += NT) ys[i] = A[P.in + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op_matvec(P, W, A, (threadgroup const float *) ys, gsg, n_sg, lane);
    } else {
        op_matvec(P, W, A, (device const float *) (A + P.in), gsg, n_sg, lane);
    }
}

// greedy pick after the LM head: one threadgroup per token row, argmax over the row's logits (first index of the
// maximum, as a CPU scan), written as a uint to A[P.aux[0] + row]
kernel void mk_argmax(constant Phase * phases     [[buffer(1)]],
                      device float * A            [[buffer(3)]],
                      uint tg                     [[threadgroup_position_in_grid]],
                      uint tid                    [[thread_index_in_threadgroup]],
                      ushort sg                   [[simdgroup_index_in_threadgroup]],
                      ushort lane                 [[thread_index_in_simdgroup]]) {
    threadgroup float bv[NSG];
    threadgroup uint bi[NSG];
    constant Phase & P = phases[0];
    const uint n = P.seg[0].n_rows;
    device const float4 * x = (device const float4 *) (A + P.seg[0].out + (ulong) tg * n);
    float v = -INFINITY;
    uint idx = 0;
    for (uint i = tid; i < n / 4; i += NT) {
        const float4 q = x[i];
        FOR_UNROLL (uint j = 0; j < 4; ++j) if (q[j] > v) { v = q[j]; idx = 4 * i + j; }
    }
    // reduce (value, index): larger value wins, the smaller index on ties
    for (ushort o = 16; o > 0; o /= 2) {
        const float ov = simd_shuffle_xor(v, o);
        const uint oi = simd_shuffle_xor(idx, o);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    if (lane == 0) { bv[sg] = v; bi[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint s = 1; s < NSG; ++s) if (bv[s] > v || (bv[s] == v && bi[s] < idx)) { v = bv[s]; idx = bi[s]; }
        ((device uint *) A)[P.aux[0] + tg] = idx;
    }
}
