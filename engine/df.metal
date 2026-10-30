// Native DFlash2 drafter (dflt-Q8_0.gguf: 2 layers, hidden 2048, 32 query / 8 kv heads of 64, FFN 6144, dynamic
// grouped convs around attention and FFN, 32k draft-vocab head + selector), for the pe engine. Always 8 token rows:
// a draft block is [last token, 7 masks]; an injection is up to 8 committed rows (prompt rows in chunks of 8).
// Mirrors llama.cpp's src/models/dflash.cpp graph (decoder: noise-block diffusion, embd batch: K/V injection).
// Matvecs: Q8_0 x 8 rows on 8x8 simdgroup MMAs (q8m), the same fragment order (YQ) as mk.metal's q4m.
#include <metal_stdlib>
using namespace metal;

#ifndef NSG
#define NSG 8
#endif
#define NT (NSG * 32)
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define H 2048
#define NB 8
#define HD 64
#define NQH 32
#define NKVH 8
#define CONV_G 16     // channels per conv group
#define CONV_NG 128   // groups
#ifndef TN
#define TN 2
#endif
#define QK_K 256

inline uint frag_lane(uint r, uint c) { return 16 * (r / 4) + 8 * (c / 4) + 2 * (r % 4) + (c % 4) / 2; }
inline uint YQ(uint k, uint b) {
    const uint kl = k % 64, s4 = kl / 32, w = kl % 32;
    const uint i = 2 * (w / 8) + (w % 4) / 2;
    const uint s = 4 * s4 + 2 * ((w % 8) / 4) + w % 2;
    return (k / 64) * 512 + frag_lane(i, b) * 16 + s * 2 + b % 2;
}
inline uint q_lane_row(uint l) { return (l / 16) * 4 + (l % 8) / 2; }
inline uint q_lane_col(uint l) { return ((l % 16) / 8) * 4 + (l % 2) * 2; }

// signed int8 pairs (bytes 0 and 2 of v) as exact halves: (q + 128) + 1024 - 1152
inline half2 h1152(uint v) { return as_type<half2>(v | 0x64006400u) - half2(1152.h, 1152.h); }

// 8*TN rows x 8*G tokens (G groups of 8, each a YQ vector of k floats * 8, gstride floats apart) of a Q8_0 matrix
// (34-byte blocks of 32) over chunks c0, c0 + cstep, ... < c1 (64 columns each). The weight fragments of a chunk are
// decoded once and reused for every token group. acc[t * G + g]: the lane's two outputs of tile t, group g.
template <int G>
inline void dot_q8m(device const uchar * w, uint rb, uint n_rows, uint c0, uint c1, uint cstep, device const float * y,
                    uint gstride, ushort lane, thread float2 * acc) {
    const uint r = q_lane_row(lane), c = q_lane_col(lane), j = c / 2;
    device const uchar * rows[TN];
    FOR_UNROLL (uint t = 0; t < TN; ++t) rows[t] = w + min(8 * t + r, n_rows - 1) * rb;
    for (uint ch = c0; ch < c1; ch += cstep) {
        half2 af[TN][8];
        float2 dd[TN];
        FOR_UNROLL (uint t = 0; t < TN; ++t) {
            device const uchar * b0 = rows[t] + 2 * ch * 34, * b1 = b0 + 34;
            dd[t] = float2((float) *(device const half *) b0, (float) *(device const half *) b1);
            device const ushort * p0 = (device const ushort *) (b0 + 2 + 8 * j);
            device const ushort * p1 = (device const ushort *) (b1 + 2 + 8 * j);
            const uint2 qa = uint2(p0[0] | ((uint) p0[1] << 16), p0[2] | ((uint) p0[3] << 16)) ^ 0x80808080u;
            const uint2 qb = uint2(p1[0] | ((uint) p1[1] << 16), p1[2] | ((uint) p1[3] << 16)) ^ 0x80808080u;
            af[t][0] = h1152(qa.x & 0x00FF00FFu); af[t][1] = h1152((qa.x >> 8) & 0x00FF00FFu);
            af[t][2] = h1152(qa.y & 0x00FF00FFu); af[t][3] = h1152((qa.y >> 8) & 0x00FF00FFu);
            af[t][4] = h1152(qb.x & 0x00FF00FFu); af[t][5] = h1152((qb.x >> 8) & 0x00FF00FFu);
            af[t][6] = h1152(qb.y & 0x00FF00FFu); af[t][7] = h1152((qb.y >> 8) & 0x00FF00FFu);
        }
        FOR_UNROLL (uint g = 0; g < G; ++g) {
            device const float4 * yq = (device const float4 *) (y + g * gstride + ch * 512 + lane * 16);
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
            FOR_UNROLL (uint t = 0; t < TN; ++t) {
                simdgroup_float8x8 P0 = make_filled_simdgroup_matrix<float, 8, 8>(0.f), P1 = P0;
                simdgroup_half8x8 am;
#define Q8M_MMA(P, a, i) am.thread_elements()[0] = a[0]; am.thread_elements()[1] = a[1]; \
                         simdgroup_multiply_accumulate(P, am, bm[i], P);
                Q8M_MMA(P0, af[t][0], 0) Q8M_MMA(P0, af[t][1], 1) Q8M_MMA(P0, af[t][2], 2) Q8M_MMA(P0, af[t][3], 3)
                Q8M_MMA(P1, af[t][4], 4) Q8M_MMA(P1, af[t][5], 5) Q8M_MMA(P1, af[t][6], 6) Q8M_MMA(P1, af[t][7], 7)
#undef Q8M_MMA
                acc[t * G + g] += dd[t][0] * float2(P0.thread_elements()[0], P0.thread_elements()[1]) +
                                  dd[t][1] * float2(P1.thread_elements()[0], P1.thread_elements()[1]);
            }
        }
    }
}

enum { MV_ROWS = 0, MV_YQ = 1, MV_SWIGLU = 2 };
struct DfMv {
    ulong w, w2;          // byte offsets in the drafter weights (w2: up matrix for MV_SWIGLU)
    uint n_rows, k, y, out, mode, pad;  // y, out: float offsets in the drafter arena
};

// out = W . y for 8 * G tokens; one threadgroup per 8*TN rows, its NSG simdgroups split k. Group g of y is at
// y + g * 8 * k (YQ order); MV_ROWS output row b at out + b * n_rows (G > 1: MV_ROWS only).
template <int G>
kernel void df_mv(constant DfMv & a [[buffer(0)]], device const uchar * W [[buffer(1)]], device float * A [[buffer(2)]],
                  uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float2 scr[(G == 1 ? 2 : 1) * NSG * TN * G * 32];
    const uint rb = a.k / 32 * 34, r0 = tg * 8 * TN, nch = a.k / 64;
    float2 acc[TN * G];
    FOR_UNROLL (uint t = 0; t < TN * G; ++t) acc[t] = 0.f;
    dot_q8m<G>(W + a.w + (ulong) r0 * rb, rb, a.n_rows - r0, sg, nch, NSG, A + a.y, 8 * a.k, lane, acc);
    FOR_UNROLL (uint t = 0; t < TN * G; ++t) scr[(sg * TN * G + t) * 32 + lane] = acc[t];
    if (G == 1 && a.mode == MV_SWIGLU) {
        FOR_UNROLL (uint t = 0; t < TN * G; ++t) acc[t] = 0.f;
        dot_q8m<G>(W + a.w2 + (ulong) r0 * rb, rb, a.n_rows - r0, sg, nch, NSG, A + a.y, 8 * a.k, lane, acc);
        FOR_UNROLL (uint t = 0; t < TN * G; ++t) scr[((NSG + sg) * TN * G + t) * 32 + lane] = acc[t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < TN * G * 32; i += NT) {
        float2 v = 0.f, u = 0.f;
        for (uint s = 0; s < NSG; ++s) v += scr[s * TN * G * 32 + i];
        const uint tg2 = i / 32, l = i % 32, t = tg2 / G, grp = tg2 % G;
        const uint row = r0 + 8 * t + q_lane_row(l), col = 8 * grp + q_lane_col(l);
        if (row >= a.n_rows) continue;
        if (a.mode == MV_ROWS) {
            A[a.out + col * a.n_rows + row] = v[0];
            A[a.out + (col + 1) * a.n_rows + row] = v[1];
        } else if (a.mode == MV_YQ) {
            A[a.out + YQ(row, col)] = v[0];
            A[a.out + YQ(row, col + 1)] = v[1];
        } else {
            for (uint s = 0; s < NSG; ++s) u += scr[(NSG + s) * TN * G * 32 + i];
            A[a.out + YQ(row, col)] = v[0] / (1.0f + exp(-v[0])) * u[0];
            A[a.out + YQ(row, col + 1)] = v[1] / (1.0f + exp(-v[1])) * u[1];
        }
    }
}
template [[host_name("df_mv")]] kernel void df_mv<1>(constant DfMv &, device const uchar *, device float *, uint, uint, ushort, ushort);
template [[host_name("df_mv4")]] kernel void df_mv<4>(constant DfMv &, device const uchar *, device float *, uint, uint, ushort, ushort);

// ---------------------------------------------------------------------------------------------
struct DfOp {
    uint x, y, z, out;    // arena float offsets (meaning per kernel)
    ulong w0, w1;         // weight byte offsets (drafter weights unless noted)
    uint pos0, n_kv, layer, side;
    uint flags, n_rows, pad0, pad1;
};

// one element of a Q4_K row (the target's token embeddings), CPU dequantize_row_q4_K arithmetic
inline float q4_K_elem(device const uchar * row, uint i) {
    device const uchar * b = row + (i / QK_K) * 144;
    const uint e = i % QK_K, j = e / 32, l = e % 32;
    device const uchar * sc = b + 4;
    uint s, m;
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
    else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
    const uchar q = b[16 + (j / 2) * 32 + l];
    const float d = (float) *(device const half *) b, dmin = (float) *(device const half *) (b + 2);
    return (d * (float) s) * (float) ((j & 1) ? (q >> 4) : (q & 0xF)) - dmin * (float) m;
}

// row sum over the threadgroup (NSG floats of scratch)
inline float tg_sum(float v, threadgroup float * red, ushort sg, ushort lane) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.f;
    for (ushort i = 0; i < NSG; ++i) s += red[i];
    return s;
}

#define DF_EMBED 1   // x = token embeddings (target Q4_K, w1 = target token_embd offset, tokens in TB)
#define DF_RESID 2   // x = in[x] + in[y] (row-major)
#define DF_OUT_YQ 4  // normalized output in YQ order (a matvec input), else row-major
// one threadgroup per token row: x (row-major, stored at out if DF_RESID/EMBED) then z = rms(x) * w0 -> out of norm
// O.x, O.y: inputs; O.z: where x is stored; O.out: normalized output; O.w0: norm weight (f32)
kernel void df_norm(constant DfOp & O [[buffer(0)]], device const uchar * W [[buffer(1)]], device float * A [[buffer(2)]],
                    device const uchar * TW [[buffer(3)]], constant uint * TB [[buffer(4)]], constant float & eps [[buffer(5)]],
                    uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[NSG];
    constexpr uint E = H / NT;
    float x[E];
    float ss = 0.f;
    FOR_UNROLL (uint e = 0; e < E; ++e) {
        const uint i = tid + e * NT;
        float v;
        if (O.flags & DF_EMBED) v = q4_K_elem(TW + O.w1 + (ulong) TB[b] * (H / QK_K) * 144, i);
        else if (O.flags & DF_RESID) v = A[O.x + b * H + i] + A[O.y + b * H + i];
        else v = A[O.x + b * H + i];
        x[e] = v;
        ss += v * v;
    }
    const float scale = 1.0f / sqrt(tg_sum(ss, red, sg, lane) / H + eps);
    device const float * w = (device const float *) (W + O.w0);
    FOR_UNROLL (uint e = 0; e < E; ++e) {
        const uint i = tid + e * NT;
        if (O.flags & (DF_EMBED | DF_RESID)) A[O.z + b * H + i] = x[e];
        const float v = (x[e] * scale) * w[i];
        A[O.out + ((O.flags & DF_OUT_YQ) ? (b / 8) * 8 * H + YQ(i, b % 8) : b * H + i)] = v;  // rows past 8: next YQ group
    }
}

// dflash2 dynamic grouped conv over the block (kernel 2, causal within the block): out[p][c] =
//   sum_t (dyn[p][g(c) + 128 t + 256 side] + base[side][t][c]) * in[p - t][c]
// O.x: input rows, O.y: dyn rows (512 per token), O.out: output, O.w0: base (f32 [2][2][2048]), O.side,
// O.flags & DF_OUT_YQ: output order. Grid: 8 rows x (H / NT) groups of threads
kernel void df_conv(constant DfOp & O [[buffer(0)]], device const uchar * W [[buffer(1)]], device float * A [[buffer(2)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint p = tg.x, c = tg.y * NT + tid, g = c / CONV_G;
    device const float * base = (device const float *) (W + O.w0);
    float s = 0.f;
    for (uint t = 0; t < 2; ++t) {
        if (p < t) continue;
        const float wgt = A[O.y + p * 512 + g + 128 * t + 256 * O.side] + base[O.side * 2 * H + t * H + c];
        s += wgt * A[O.x + ((O.flags & 8) ? YQ(c, p - t) : (p - t) * H + c)];  // flag 8: input in YQ order
    }
    A[O.out + ((O.flags & DF_OUT_YQ) ? YQ(c, p) : p * H + c)] = s;
}

// q / k per-head RMS norm + NEOX RoPE (64 dims, theta base in O.n_rows as float bits), K and V into the cache.
// O.x: q rows (2048), O.y: k rows (512), O.z: v rows (512), O.out: roped q rows, O.w0: q norm, O.w1: k norm,
// O.pos0: position of row 0, O.layer; O.flags & 1: k/v only (injection: no q). One simdgroup per (row, head):
// grid 8 rows x (32 + 8 + 8) heads; lane holds dims 2 lane, 2 lane + 1.
kernel void df_qk(constant DfOp & O [[buffer(0)]], device const uchar * W [[buffer(1)]], device float * A [[buffer(2)]],
                  device half * kc [[buffer(3)]], device half * vc [[buffer(4)]], constant uint & ctx [[buffer(5)]],
                  constant float & eps [[buffer(6)]],
                  uint2 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint b = tg.x, hh = tg.y, p = O.pos0 + b;
    const ulong row = ((ulong) O.layer * ctx + p) * NKVH;
    if (hh >= NQH + NKVH) {  // v
        const uint g = hh - NQH - NKVH;
        const float2 v = ((device const float2 *) (A + O.z + b * NKVH * HD + g * HD))[lane];
        ((device half2 *) (vc + (row + g) * HD))[lane] = half2(v);
        return;
    }
    const bool is_q = hh < NQH;
    if (is_q && (O.flags & 1)) return;
    const uint head = is_q ? hh : hh - NQH;
    device const float * src = is_q ? A + O.x + b * NQH * HD + head * HD : A + O.y + b * NKVH * HD + head * HD;
    device const float * nw = (device const float *) (W + (is_q ? O.w0 : O.w1));
    float2 x = ((device const float2 *) src)[lane];
    const float scale = 1.0f / sqrt(simd_sum(dot(x, x)) / HD + eps);
    x = (x * scale) * ((device const float2 *) nw)[lane];
    // NEOX: pairs (i, i + 32); lane l holds dims 2l, 2l + 1, its partner holds 2l + 32 (lane l ^ 16)
    const float2 partner = simd_shuffle_xor(x, 16);
    const bool lo = lane < 16;
    const float base = as_type<float>(O.n_rows);
    float2 r;
    for (uint e = 0; e < 2; ++e) {
        const uint i = (lane % 16) * 2 + e;  // pair index
        const float theta = (float) p * pow(base, -2.0f * i / HD);
        const float cs = cos(theta), sn = sin(theta);
        const float x0 = lo ? x[e] : partner[e], x1 = lo ? partner[e] : x[e];
        r[e] = lo ? x0 * cs - x1 * sn : x0 * sn + x1 * cs;
    }
    if (is_q) ((device float2 *) (A + O.out + b * NQH * HD + head * HD))[lane] = r;
    else      ((device half2 *) (kc + (row + head) * HD))[lane] = half2(r);
}

// non-causal attention of the 8 block rows over cache positions [0, n_kv): one simdgroup per (row, query head), 32
// positions per step (one per lane), online softmax; lane owns output dims 2 lane, 2 lane + 1. Output in YQ order.
// O.x: roped q rows, O.out: output, O.n_kv, O.layer.
kernel void df_attn(constant DfOp & O [[buffer(0)]], device float * A [[buffer(2)]],
                    device const half * kc [[buffer(3)]], device const half * vc [[buffer(4)]], constant uint & ctx [[buffer(5)]],
                    uint2 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint b = tg.x, h = tg.y, g = h / (NQH / NKVH);
    const ulong layer = (ulong) O.layer * ctx;
    device const float4 * q4 = (device const float4 *) (A + O.x + b * NQH * HD + h * HD);
    float4 q[HD / 4];
    FOR_UNROLL (uint i = 0; i < HD / 4; ++i) q[i] = q4[i];
    const float scale = 1.0f / 8.0f;  // 1 / sqrt(64)
    float M = -INFINITY, L = 0.f;
    float2 acc = 0.f;
    for (uint p0 = 0; p0 < O.n_kv; p0 += 32) {
        const uint p = p0 + lane;
        float s = -INFINITY;
        if (p < O.n_kv) {
            device const half4 * kr = (device const half4 *) (kc + ((layer + p) * NKVH + g) * HD);
            float d = 0.f;
            FOR_UNROLL (uint i = 0; i < HD / 4; ++i) d += dot(float4(kr[i]), q[i]);
            s = d * scale;
        }
        const float m = max(M, simd_max(s));
        const float e = p < O.n_kv ? exp(s - m) : 0.f;
        const float corr = exp(M - m);
        L = L * corr + simd_sum(e);
        acc *= corr;
        const uint n = min(32u, O.n_kv - p0);
        for (uint i = 0; i < n; ++i) {
            const float ei = simd_shuffle(e, (ushort) i);
            acc += ei * float2(((device const half2 *) (vc + ((layer + p0 + i) * NKVH + g) * HD))[lane]);
        }
        M = m;
    }
    acc /= L;
    const uint d0 = h * HD + 2 * lane;
    A[O.out + YQ(d0, b)] = acc[0];
    A[O.out + YQ(d0 + 1, b)] = acc[1];
}

// target-layer features of up to 8 rows (row-major in the target arena, 5 layers) -> one YQ vector of 5 * 2048:
// TA: target arena; TO: the 5 layer offsets; O.out: drafter arena offset. Grid: 8 rows x (5 * H / NT)
kernel void df_feat(constant DfOp & O [[buffer(0)]], device float * A [[buffer(2)]], device const float * TA [[buffer(3)]],
                    constant uint * TO [[buffer(4)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint b = tg.x, k = tg.y * NT + tid, l = k / H, i = k % H;
    A[O.out + (b / 8) * 8 * 5 * H + YQ(k, b % 8)] = TA[TO[l] + b * O.n_rows + i];  // O.n_rows: row stride of TA
}
