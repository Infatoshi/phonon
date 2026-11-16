// Parakeet TDT 0.6B v2 encoder (FastConformer, 24 blocks, d 1024, 8 heads, rel-pos attention, conv kernel 9) on
// Metal: fp16 weights (the bf16 values the MLX app runs with), float32 activations, one command buffer per window.
// Rows past the valid frame count T are computed but never read across rows: every cross-row op (depthwise conv,
// attention, the subsampling convolutions) bounds-checks T, so a padded bucket gives the exact-length result.
#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL _Pragma("clang loop unroll(full)") for
#ifndef LIN4_UNROLL
#define LIN4_UNROLL
#endif

struct Lin {           // Y[r][n] = alpha * act(sum_k X[r][k] W[n][k] + b[n]) + (res ? R[r][n] : 0), r < rows
    uint x, y, r, w, b;    // float offsets into the arena (x, y, r) and half offsets into the weights (w, b; b = ~0: none)
    uint rows, n, k;
    uint act;              // 0 none, 1 relu, 2 silu
    float alpha;
    uint res;              // 1: add R
    uint ksplit;           // lin4: > 1 splits k over grid z; then Y += alpha * partial by float atomics (Y = R, no bias / act)
    uint qoff, qsb, qg;    // lin4q: byte offsets of the quantized weight codes and of the (scale, bias) half2 pairs, group size
};

inline float bf16r(float x) {  // round to the nearest-even bf16 value (the MLX app's dtype), as float
    uint u = as_type<uint>(x);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    return as_type<float>(u);
}

inline float act_of(float v, uint a) { return a == 1 ? max(v, 0.f) : a == 2 ? v / (1.f + exp(-v)) : v; }

// linear: one simdgroup per 8 output features x 32 rows (4 row tiles of 8), k in steps of 8: the weight fragment is
// loaded once per step (transposed from W[n][k]) and multiplied with the 4 row tiles' activation fragments
kernel void lin(constant Lin & L [[buffer(0)]], device const half * W [[buffer(1)]], device float * A [[buffer(2)]],
                uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                ushort lane [[thread_index_in_simdgroup]]) {
    const uint n0 = (tg.x * 4 + sg) * 8, r0 = tg.y * 32;
    if (n0 >= L.n) return;
    simdgroup_float8x8 acc[4];
    FOR_UNROLL (uint t = 0; t < 4; ++t) acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    device const half * w = W + L.w + (ulong) n0 * L.k;
    device const float * x = A + L.x + (ulong) r0 * L.k;
    for (uint k = 0; k < L.k; k += 8) {
        simdgroup_half8x8 wf;
        simdgroup_load(wf, w + k, L.k, ulong2(0, 0), true);  // wf[kk][nn] = W[n0 + nn][k + kk]
        FOR_UNROLL (uint t = 0; t < 4; ++t) {
            simdgroup_float8x8 xf;
            simdgroup_load(xf, x + (ulong) t * 8 * L.k + k, L.k);  // xf[rr][kk] = X[r0 + 8t + rr][k + kk]
            simdgroup_multiply_accumulate(acc[t], xf, wf, acc[t]);
        }
    }
    threadgroup float tile[4][64];
    const uint ri = lane / 8, ni = lane % 8;
    FOR_UNROLL (uint t = 0; t < 4; ++t) {
        simdgroup_store(acc[t], &tile[sg][0], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (uint e = 0; e < 2; ++e) {
            const uint rr = ri + 4 * e, r = r0 + 8 * t + rr, n = n0 + ni;
            if (r < L.rows) {
                float v = tile[sg][rr * 8 + ni];
                if (L.b != ~0u) v += (float) W[L.b + n];
                v = L.alpha * act_of(v, L.act);
                if (L.res) v += A[L.r + (ulong) r * L.n + n];
                A[L.y + (ulong) r * L.n + n] = v;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// lin2: a threadgroup of 8 simdgroups owns 8 * TN2 output features x 32 rows; its simdgroups split k (interleaved
// 64-wide chunks: each weight row segment is a whole 128-byte line), and the partials are added in simdgroup order
#ifndef TN2
#define TN2 2
#endif
kernel void lin2(constant Lin & L [[buffer(0)]], device const half * W [[buffer(1)]], device float * A [[buffer(2)]],
                 uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    const uint n0 = tg.x * 8 * TN2, r0 = tg.y * 32;
    simdgroup_float8x8 acc[TN2][4];
    FOR_UNROLL (uint a = 0; a < TN2; ++a) FOR_UNROLL (uint t = 0; t < 4; ++t) acc[a][t] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    device const half * w = W + L.w + (ulong) n0 * L.k;
    device const float * x = A + L.x + (ulong) r0 * L.k;
    for (uint c = sg * 64; c < L.k; c += 8 * 64) {
        FOR_UNROLL (uint kk = 0; kk < 64; kk += 8) {
            simdgroup_half8x8 wf[TN2];
            FOR_UNROLL (uint a = 0; a < TN2; ++a) simdgroup_load(wf[a], w + (ulong) a * 8 * L.k + c + kk, L.k, ulong2(0, 0), true);
            FOR_UNROLL (uint t = 0; t < 4; ++t) {
                simdgroup_float8x8 xf;
                simdgroup_load(xf, x + (ulong) t * 8 * L.k + c + kk, L.k);
                FOR_UNROLL (uint a = 0; a < TN2; ++a) simdgroup_multiply_accumulate(acc[a][t], xf, wf[a], acc[a][t]);
            }
        }
    }
    threadgroup float part[8][TN2 * 4 * 64];
    FOR_UNROLL (uint a = 0; a < TN2; ++a) FOR_UNROLL (uint t = 0; t < 4; ++t) simdgroup_store(acc[a][t], &part[sg][(a * 4 + t) * 64], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < TN2 * 4 * 64; e += 256) {
        float v = 0.f;
        FOR_UNROLL (uint s = 0; s < 8; ++s) v += part[s][e];
        const uint a = e / 256, t = (e / 64) % 4, rr = (e % 64) / 8, nn = e % 8;
        const uint r = r0 + 8 * t + rr, n = n0 + 8 * a + nn;
        if (r < L.rows && n < L.n) {
            if (L.b != ~0u) v += (float) W[L.b + n];
            v = L.alpha * act_of(v, L.act);
            if (L.res) v += A[L.r + (ulong) r * L.n + n];
            A[L.y + (ulong) r * L.n + n] = v;
        }
    }
}

// lin3<XT, YT, RT>: lin2 with typed activations (half inputs halve the activation traffic, which is RT times the
// weight traffic per k step) and RT row tiles of 8 (rows per threadgroup 8 * RT: 1..4 cover T <= 32 exactly).
// Offsets x / y / r are in elements of their own type: the arena is addressed as float (A) and half (Ah) views.
template <typename XT, typename YT, int RT>
kernel void lin3(constant Lin & L [[buffer(0)]], device const half * W [[buffer(1)]], device float * A [[buffer(2)]],
                 uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    const uint n0 = tg.x * 8 * TN2, r0 = tg.y * 8 * RT;
    simdgroup_float8x8 acc[TN2][RT];
    FOR_UNROLL (uint a = 0; a < TN2; ++a) FOR_UNROLL (uint t = 0; t < RT; ++t) acc[a][t] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    device const half * w = W + L.w + (ulong) n0 * L.k;
    device const XT * x = (device const XT *) A + L.x + (ulong) r0 * L.k;
    for (uint c = sg * 64; c < L.k; c += 8 * 64) {
        FOR_UNROLL (uint kk = 0; kk < 64; kk += 8) {
            simdgroup_half8x8 wf[TN2];
            FOR_UNROLL (uint a = 0; a < TN2; ++a) simdgroup_load(wf[a], w + (ulong) a * 8 * L.k + c + kk, L.k, ulong2(0, 0), true);
            FOR_UNROLL (uint t = 0; t < RT; ++t) {
                simdgroup_matrix<XT, 8, 8> xf;
                simdgroup_load(xf, x + (ulong) t * 8 * L.k + c + kk, L.k);
                FOR_UNROLL (uint a = 0; a < TN2; ++a) simdgroup_multiply_accumulate(acc[a][t], xf, wf[a], acc[a][t]);
            }
        }
    }
    threadgroup float part[8][TN2 * RT * 64];
    FOR_UNROLL (uint a = 0; a < TN2; ++a) FOR_UNROLL (uint t = 0; t < RT; ++t) simdgroup_store(acc[a][t], &part[sg][(a * RT + t) * 64], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device YT * Y = (device YT *) A;
    device const float * R = A;
    for (uint e = tid; e < TN2 * RT * 64; e += 256) {
        float v = 0.f;
        FOR_UNROLL (uint s = 0; s < 8; ++s) v += part[s][e];
        const uint a = e / (RT * 64), t = (e / 64) % RT, rr = (e % 64) / 8, nn = e % 8;
        const uint r = r0 + 8 * t + rr, n = n0 + 8 * a + nn;
        if (r < L.rows && n < L.n) {
            if (L.b != ~0u) v += (float) W[L.b + n];
            v = L.alpha * act_of(v, L.act);
            if (L.res) v += R[L.r + (ulong) r * L.n + n];
            Y[L.y + (ulong) r * L.n + n] = (YT) v;
        }
    }
}
#define LIN3(XT, YT, RT) template [[host_name("lin3_" #XT "_" #YT "_" #RT)]] kernel void lin3<XT, YT, RT>( \
    constant Lin &, device const half *, device float *, uint2, ushort, ushort, uint);
LIN3(float, float, 1) LIN3(float, float, 2) LIN3(float, float, 3) LIN3(float, float, 4)
LIN3(half, half, 1) LIN3(half, half, 2) LIN3(half, half, 3) LIN3(half, half, 4)
LIN3(half, float, 1) LIN3(half, float, 2) LIN3(half, float, 3) LIN3(half, float, 4)
LIN3(float, half, 1) LIN3(float, half, 2) LIN3(float, half, 3) LIN3(float, half, 4)

// lin4: lin3 computing Y^T = W X^T with vector loads only. In the 8x8 fragment layout (MLX steel: lane l holds row
// fm = (l/4 & 4) + (l/2 % 4), cols fn, fn+1 with fn = (l/4 & 2) * 2 + (l % 2) * 2) the 64-wide k chunk is permuted:
// fragment kk, inner index i <-> k = 8 i + kk. Then a lane's W elements over the 8 fragments are W[n0+fm][8fn .. 8fn+15]
// (32 contiguous bytes) and its X elements are X[r0+fn (+1)][8fm .. 8fm+7] (16 contiguous bytes per row).
// TN: 8-wide output tiles per threadgroup (1 for the narrow n = 1024 projections: twice the threadgroups).
// (A runtime skip of row tiles past the last valid row made the kernel 1.8x slower: the host balances row blocks instead.)
template <typename XT, typename YT, int RT, int TN>
kernel void lin4(constant Lin & L [[buffer(0)]], device const half * W [[buffer(1)]], device float * A [[buffer(2)]],
                 uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    const uint n0 = tg.x * 8 * TN, r0 = tg.y * 8 * RT;
    const uint kper = L.ksplit > 1 ? L.k / L.ksplit : L.k, kb = tg.z * kper;
    const ushort qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
    simdgroup_float8x8 acc[TN][RT];
    FOR_UNROLL (uint a = 0; a < TN; ++a) FOR_UNROLL (uint t = 0; t < RT; ++t) acc[a][t] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    device const half * w = W + L.w + (ulong) (n0 + fm) * L.k + 8 * fn;
    device const XT * x = (device const XT *) A + L.x + (ulong) (r0 + fn) * L.k + 8 * fm;
    LIN4_UNROLL for (uint c = kb + sg * 64; c < kb + kper; c += 8 * 64) {
        half4 wv[TN][4];
        FOR_UNROLL (uint a = 0; a < TN; ++a) {
            device const half4 * p = (device const half4 *) (w + (ulong) a * 8 * L.k + c);
            FOR_UNROLL (uint q = 0; q < 4; ++q) wv[a][q] = p[q];
        }
        vec<XT, 4> xv[RT][2][2];
        FOR_UNROLL (uint t = 0; t < RT; ++t) FOR_UNROLL (uint j = 0; j < 2; ++j) {
            device const vec<XT, 4> * p = (device const vec<XT, 4> *) (x + (ulong) (8 * t + j) * L.k + c);
            xv[t][j][0] = p[0];
            xv[t][j][1] = p[1];
        }
        FOR_UNROLL (uint kk = 0; kk < 8; ++kk) {
            simdgroup_half8x8 wf[TN];
            FOR_UNROLL (uint a = 0; a < TN; ++a) {
                wf[a].thread_elements()[0] = wv[a][kk / 4][kk % 4];
                wf[a].thread_elements()[1] = wv[a][2 + kk / 4][kk % 4];
            }
            FOR_UNROLL (uint t = 0; t < RT; ++t) {
                simdgroup_matrix<XT, 8, 8> xf;
                xf.thread_elements()[0] = xv[t][0][kk / 4][kk % 4];
                xf.thread_elements()[1] = xv[t][1][kk / 4][kk % 4];
                FOR_UNROLL (uint a = 0; a < TN; ++a) simdgroup_multiply_accumulate(acc[a][t], wf[a], xf, acc[a][t]);
            }
        }
    }
    threadgroup float part[8][TN * RT * 64];
    FOR_UNROLL (uint a = 0; a < TN; ++a) FOR_UNROLL (uint t = 0; t < RT; ++t) simdgroup_store(acc[a][t], &part[sg][(a * RT + t) * 64], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device YT * Y = (device YT *) A;
    device const float * R = A;
    for (uint e = tid; e < TN * RT * 64; e += 256) {
        float v = 0.f;
        FOR_UNROLL (uint s = 0; s < 8; ++s) v += part[s][e];
        const uint a = e / (RT * 64), t = (e / 64) % RT, nn = (e % 64) / 8, rr = e % 8;  // tile [n][r]
        const uint r = r0 + 8 * t + rr, n = n0 + 8 * a + nn;
        if (r < L.rows && n < L.n) {
            if (L.ksplit > 1) {  // residual into the float stream, k split over threadgroups
                atomic_fetch_add_explicit((device atomic_float *) (A + L.y + (ulong) r * L.n + n), L.alpha * v, memory_order_relaxed);
                continue;
            }
            if (L.b != ~0u) v += (float) W[L.b + n];
            v = L.alpha * act_of(v, L.act);
            if (L.res) v += R[L.r + (ulong) r * L.n + n];
            Y[L.y + (ulong) r * L.n + n] = (YT) v;
        }
    }
}
#define LIN4T(XT, YT, RT, TN) template [[host_name("lin4_" #XT "_" #YT "_" #RT "_" #TN)]] kernel void lin4<XT, YT, RT, TN>( \
    constant Lin &, device const half *, device float *, uint3, ushort, ushort, uint);
#define LIN4(XT, YT, RT) LIN4T(XT, YT, RT, 1) LIN4T(XT, YT, RT, 2) LIN4T(XT, YT, RT, 4)
LIN4(float, float, 1) LIN4(float, float, 2) LIN4(float, float, 3) LIN4(float, float, 4)
LIN4(half, half, 1) LIN4(half, half, 2) LIN4(half, half, 3) LIN4(half, half, 4)
LIN4(half, float, 1) LIN4(half, float, 2) LIN4(half, float, 3) LIN4(half, float, 4)
LIN4(float, half, 1) LIN4(float, half, 2) LIN4(float, half, 3) LIN4(float, half, 4)

// lin4q: lin4 with weight-only quantized W (BITS 8 or 4, affine per group of qg along k: w = s * q + b, (s, b) half2
// per (row, group); 4-bit codes packed two per byte, low nibble first). A lane's 16 k values of a chunk
// (8 fn .. 8 fn + 15) are 16 (8-bit) or 8 (4-bit) contiguous bytes and lie in one group for qg >= 16.
template <typename XT, typename YT, int RT, int TN, int BITS>
kernel void lin4q(constant Lin & L [[buffer(0)]], device const half * W [[buffer(1)]], device float * A [[buffer(2)]],
                  device const uchar * Q [[buffer(3)]],
                  uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                  ushort lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    const uint n0 = tg.x * 8 * TN, r0 = tg.y * 8 * RT;
    const ushort qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
    simdgroup_float8x8 acc[TN][RT];
    FOR_UNROLL (uint a = 0; a < TN; ++a) FOR_UNROLL (uint t = 0; t < RT; ++t) acc[a][t] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    const uint ng = L.k / L.qg;  // groups per row
    device const half2 * SB = (device const half2 *) (Q + L.qsb);
    device const XT * x = (device const XT *) A + L.x + (ulong) (r0 + fn) * L.k + 8 * fm;
    for (uint c = sg * 64; c < L.k; c += 8 * 64) {
        half wv[TN][16];
        FOR_UNROLL (uint a = 0; a < TN; ++a) {
            const uint n = n0 + 8 * a + fm, k0 = c + 8 * fn;
            const float2 sb = float2(SB[n * ng + k0 / L.qg]);
            if (BITS == 8) {
                const uint4 qv = *(device const uint4 *) (Q + L.qoff + (ulong) n * L.k + k0);
                FOR_UNROLL (uint j = 0; j < 16; ++j) wv[a][j] = (half) (sb.x * (float) ((qv[j / 4] >> (8 * (j % 4))) & 0xFFu) + sb.y);
            } else {
                const uint2 qv = *(device const uint2 *) (Q + L.qoff + ((ulong) n * L.k + k0) / 2);
                FOR_UNROLL (uint j = 0; j < 16; ++j) wv[a][j] = (half) (sb.x * (float) ((qv[j / 8] >> (4 * (j % 8))) & 0xFu) + sb.y);
            }
        }
        vec<XT, 4> xv[RT][2][2];
        FOR_UNROLL (uint t = 0; t < RT; ++t) FOR_UNROLL (uint j = 0; j < 2; ++j) {
            device const vec<XT, 4> * p = (device const vec<XT, 4> *) (x + (ulong) (8 * t + j) * L.k + c);
            xv[t][j][0] = p[0];
            xv[t][j][1] = p[1];
        }
        FOR_UNROLL (uint kk = 0; kk < 8; ++kk) {
            simdgroup_half8x8 wf[TN];
            FOR_UNROLL (uint a = 0; a < TN; ++a) {
                wf[a].thread_elements()[0] = wv[a][kk];
                wf[a].thread_elements()[1] = wv[a][8 + kk];
            }
            FOR_UNROLL (uint t = 0; t < RT; ++t) {
                simdgroup_matrix<XT, 8, 8> xf;
                xf.thread_elements()[0] = xv[t][0][kk / 4][kk % 4];
                xf.thread_elements()[1] = xv[t][1][kk / 4][kk % 4];
                FOR_UNROLL (uint a = 0; a < TN; ++a) simdgroup_multiply_accumulate(acc[a][t], wf[a], xf, acc[a][t]);
            }
        }
    }
    threadgroup float part[8][TN * RT * 64];
    FOR_UNROLL (uint a = 0; a < TN; ++a) FOR_UNROLL (uint t = 0; t < RT; ++t) simdgroup_store(acc[a][t], &part[sg][(a * RT + t) * 64], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device YT * Y = (device YT *) A;
    device const float * R = A;
    for (uint e = tid; e < TN * RT * 64; e += 256) {
        float v = 0.f;
        FOR_UNROLL (uint s = 0; s < 8; ++s) v += part[s][e];
        const uint a = e / (RT * 64), t = (e / 64) % RT, nn = (e % 64) / 8, rr = e % 8;
        const uint r = r0 + 8 * t + rr, n = n0 + 8 * a + nn;
        if (r < L.rows && n < L.n) {
            if (L.b != ~0u) v += (float) W[L.b + n];
            v = L.alpha * act_of(v, L.act);
            if (L.res) v += R[L.r + (ulong) r * L.n + n];
            Y[L.y + (ulong) r * L.n + n] = (YT) v;
        }
    }
}
#define LIN4QT(XT, YT, RT, TN, B) template [[host_name("lin4q" #B "_" #XT "_" #YT "_" #RT "_" #TN)]] kernel void lin4q<XT, YT, RT, TN, B>( \
    constant Lin &, device const half *, device float *, device const uchar *, uint3, ushort, ushort, uint);
#define LIN4Q(XT, YT, RT) LIN4QT(XT, YT, RT, 1, 8) LIN4QT(XT, YT, RT, 2, 8) LIN4QT(XT, YT, RT, 1, 4) LIN4QT(XT, YT, RT, 2, 4)
LIN4Q(float, float, 1) LIN4Q(float, float, 2) LIN4Q(float, float, 3) LIN4Q(float, float, 4)
LIN4Q(half, half, 1) LIN4Q(half, half, 2) LIN4Q(half, half, 3) LIN4Q(half, half, 4)
LIN4Q(half, float, 1) LIN4Q(half, float, 2) LIN4Q(half, float, 3) LIN4Q(half, float, 4)

struct Norm { uint x, y, w, b, rows, d; float eps; uint yh; };

// layer norm over d of each row: one threadgroup (256 threads) per row
kernel void layernorm(constant Norm & P [[buffer(0)]], device const half * W [[buffer(1)]], device float * A [[buffer(2)]],
                      uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[2][8];
    device const float * x = A + P.x + (ulong) row * P.d;
    float s = 0.f, s2 = 0.f;
    for (uint i = tid; i < P.d; i += 256) { const float v = x[i]; s += v; s2 += v * v; }
    s = simd_sum(s); s2 = simd_sum(s2);
    if (lane == 0) { red[0][sg] = s; red[1][sg] = s2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = 0.f; s2 = 0.f;
    for (uint i = 0; i < 8; ++i) { s += red[0][i]; s2 += red[1][i]; }
    const float mean = s / P.d, var = max(s2 / P.d - mean * mean, 0.f), inv = rsqrt(var + P.eps);
    if (P.yh == 1) {  // half output (a matmul input), offset in halves
        device half * y = (device half *) A + P.y + (ulong) row * P.d;
        for (uint i = tid; i < P.d; i += 256) y[i] = (half) ((x[i] - mean) * inv * (float) W[P.w + i] + (float) W[P.b + i]);
    } else if (P.yh == 2) {  // float output rounded to bf16 (the encoder output: MLX hands bf16 to the joint)
        device float * y = A + P.y + (ulong) row * P.d;
        for (uint i = tid; i < P.d; i += 256) y[i] = bf16r((x[i] - mean) * inv * (float) W[P.w + i] + (float) W[P.b + i]);
    } else {
        device float * y = A + P.y + (ulong) row * P.d;
        for (uint i = tid; i < P.d; i += 256) y[i] = (x[i] - mean) * inv * (float) W[P.w + i] + (float) W[P.b + i];
    }
}

struct Att { uint q, k, v, p, o, u, vb, T, d, heads; float scale; uint ld, Tv, Tp; };

// rel-pos attention (Transformer-XL): score(i, j) = scale * ((q_i + u) . k_j + (q_i + v) . p[Tp - 1 - i + j]);
// one threadgroup (128 threads = 4 simdgroups) per (query i, head h). Keys j < Tv only (the valid frames of a padded
// bucket); p is linear_pos of the relative positions Tp-1 .. -(Tp-1) (any Tp >= Tv: row Tp-1-i+j is position i-j).
// q, k, v rows have stride ld (the fused qkv projection), p and o stride d. Padded queries (i >= Tv) write zeros.
kernel void attn(constant Att & P [[buffer(0)]], device const half * W [[buffer(1)]], device float * A [[buffer(2)]],
                 uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                 ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint i = g.x, h = g.y, hd = P.d / P.heads, T = P.Tv;
    device const half * Hh = (device const half *) A;  // q, k, v, p, o: half views (offsets in halves)
    device half * O = (device half *) A + P.o + i * P.d + h * hd;
    if (i >= T) { O[tid] = 0.h; return; }
    threadgroup float sc[2048], red[4];  // Tv <= 2048 (164 s)
    // lane holds dims 4 lane .. 4 lane + 3 of (q_i + u) and (q_i + v)
    const float4 qi = float4(*((device const half4 *) (Hh + P.q + i * P.ld + h * hd) + lane));
    const float4 qu = qi + float4(*((device const half4 *) (W + P.u + h * hd) + lane));
    const float4 qv = qi + float4(*((device const half4 *) (W + P.vb + h * hd) + lane));
    float mx = -INFINITY;
    for (uint j = sg; j < T; j += 4) {
        const float4 kj = float4(*((device const half4 *) (Hh + P.k + j * P.ld + h * hd) + lane));
        const float4 pj = float4(*((device const half4 *) (Hh + P.p + (P.Tp - 1 - i + j) * P.d + h * hd) + lane));
        const float s = P.scale * simd_sum(dot(qu, kj) + dot(qv, pj));
        if (lane == 0) sc[j] = s;
        mx = max(mx, s);
    }
    if (lane == 0) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = max(max(red[0], red[1]), max(red[2], red[3]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.f;
    for (uint j = tid; j < T; j += 128) { const float e = exp(sc[j] - mx); sc[j] = e; sum += e; }
    sum = simd_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = red[0] + red[1] + red[2] + red[3];
    float o = 0.f;
    for (uint j = 0; j < T; ++j) o += sc[j] * (float) Hh[P.v + j * P.ld + h * hd + tid];
    O[tid] = (half) (o / sum);
}

// attn8: attn with 8 queries per threadgroup (one simdgroup each, 256 threads): the 8 simdgroups stream the same
// key / value rows (and overlapping pos rows), so a threadgroup reads them from memory once instead of 8 times.
kernel void attn8(constant Att & P [[buffer(0)]], device const half * W [[buffer(1)]], device float * A [[buffer(2)]],
                  uint2 g [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                  ushort lane [[thread_index_in_simdgroup]]) {
    const uint i = g.x * 8 + sg, h = g.y, hd = P.d / P.heads, T = P.Tv;
    device const half * Hh = (device const half *) A;
    threadgroup float scs[8][512];  // Tv <= 512 (the host uses attn past that)
    threadgroup float * sc = scs[sg];
    device half4 * O = (device half4 *) ((device half *) A + P.o + i * P.d + h * hd) + lane;
    if (i >= T) {
        if (i < P.T) *O = half4(0.h);
        return;  // no threadgroup barriers below: simdgroups finish independently
    }
    const float4 qi = float4(*((device const half4 *) (Hh + P.q + i * P.ld + h * hd) + lane));
    const float4 qu = qi + float4(*((device const half4 *) (W + P.u + h * hd) + lane));
    const float4 qv = qi + float4(*((device const half4 *) (W + P.vb + h * hd) + lane));
    device const half4 * K = (device const half4 *) (Hh + P.k + h * hd) + lane;
    device const half4 * Pp = (device const half4 *) (Hh + P.p + (P.Tp - 1 - i) * P.d + h * hd) + lane;
    const uint ks = P.ld / 4, ps = P.d / 4;  // row strides in half4
    float mx = -INFINITY;
    for (uint j = 0; j < T; ++j) {
        const float s = P.scale * simd_sum(dot(qu, float4(K[j * ks])) + dot(qv, float4(Pp[j * ps])));
        if (lane == 0) sc[j] = s;
        mx = max(mx, s);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.f;
    for (uint j = lane; j < T; j += 32) { const float e = exp(sc[j] - mx); sc[j] = e; sum += e; }
    sum = simd_sum(sum);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device const half4 * V = (device const half4 *) (Hh + P.v + h * hd) + lane;
    float4 o = 0.f;
    for (uint j = 0; j < T; ++j) o += sc[j] * float4(V[j * ks]);
    *O = half4(o / sum);
}

struct Conv { uint g, y, w, bnw, bnb, bnm, bnv, T, d; float eps; };

// conv module middle: GLU of the pointwise-conv1 output g[t][2d], depthwise conv (kernel 9, zero padding outside
// 0..T-1), batch norm (inference), SiLU -> y[t][d]
kernel void convmid(constant Conv & P [[buffer(0)]], device const half * W [[buffer(1)]], device float * A [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, t = gid.y;
    if (c >= P.d || t >= P.T) return;
    float s = 0.f;
    FOR_UNROLL (int kk = 0; kk < 9; ++kk) {
        const int tt = (int) t + kk - 4;
        if (tt < 0 || tt >= (int) P.T) continue;
        device const half * G = (device const half *) A + P.g;  // g, y: half views (offsets in halves)
        const float a = G[tt * 2 * P.d + c], b = G[tt * 2 * P.d + P.d + c];
        s += (float) W[P.w + c * 9 + kk] * (a / (1.f + exp(-b)));
    }
    const float bn = (s - (float) W[P.bnm + c]) * rsqrt((float) W[P.bnv + c] + P.eps) * (float) W[P.bnw + c] + (float) W[P.bnb + c];
    ((device half *) A)[P.y + t * P.d + c] = (half) (bn / (1.f + exp(-bn)));
}

struct Sub { uint x, y, w, b, Ti, Fi, To, Fo, C, Tvalid; uint mode; };  // mode 0: conv0 (1 -> C), 1: depthwise

// subsampling 3x3 stride-2 convolutions (padding 1) over (time, freq), NHWC: x[t][f][c] -> y[t][f][c], ReLU on conv0;
// input rows at or past Tvalid read as zero (the exact-length conv's zero padding)
kernel void subconv(constant Sub & P [[buffer(0)]], device const half * W [[buffer(1)]], device float * A [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, f = gid.y, t = gid.z;
    if (c >= P.C || f >= P.Fo || t >= P.To) return;
    float s = (float) W[P.b + c];
    FOR_UNROLL (int dt = 0; dt < 3; ++dt) FOR_UNROLL (int df = 0; df < 3; ++df) {
        const int ti = 2 * (int) t + dt - 1, fi = 2 * (int) f + df - 1;
        if (ti < 0 || ti >= (int) P.Tvalid || fi < 0 || fi >= (int) P.Fi) continue;
        const float xv = P.mode == 0 ? A[P.x + ti * P.Fi + fi] : A[P.x + ((ulong) ti * P.Fi + fi) * P.C + c];
        s += (float) W[P.w + c * 9 + dt * 3 + df] * xv;
    }
    A[P.y + ((ulong) t * P.Fo + f) * P.C + c] = P.mode == 0 ? max(s, 0.f) : s;
}

// [T][F][C] -> [T][C * F] (the subsampling output flatten: index c * F + f)
// noop: one store (the per-dispatch floor of a serial chain, PA_NOOP=n)
kernel void noop(device float * A [[buffer(2)]], uint g [[thread_position_in_grid]]) {
    if (g == 0) A[0] = A[0] + 1.f;
}

// ---------------------------------------------------------------------------------------------------------------
// log-mel front end, parakeet-mlx get_logmel exactly: preemphasis 0.97, reflect pad 256, frames of 512 every 160
// with hann(400) zero-padded at the end, magnitude |re| + |im| (its complex-as-float view), squared, slaney mel
// filterbank (128 x 257, from librosa via ref.py melfb), log(x + 1e-5), per-feature (population) normalization.
struct Mel { uint wav, n, pow, lmel, mel, fb, tw, frames; };

inline float preemph_at(device const float * x, uint n, int i) {  // preemphasized sample i of the reflect-padded signal
    int j = i - 256;
    if (j < 0) j = -j;                       // prefix: x'[256 - i]
    else if (j >= (int) n) j = 2 * ((int) n - 1) - j;  // suffix: x'[n - 2 - (i - 256 - n)]
    return j == 0 ? x[0] : x[j] - 0.97f * x[j - 1];
}

// power spectrum p[t][f] = (|re| + |im|)^2 by a direct DFT (512-point, only the 400 windowed samples); tw: cos / sin
// tables [2][512] and the window [400] at float offset P.tw
kernel void melpow(constant Mel & P [[buffer(0)]], device float * A [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint f = gid.x, t = gid.y;
    if (f >= 257 || t >= P.frames) return;
    device const float * x = A + P.wav;
    device const float * cs = A + P.tw, * sn = cs + 512, * win = sn + 512;
    float re = 0.f, im = 0.f;
    for (uint j = 0; j < 400; ++j) {
        const float v = preemph_at(x, P.n, (int) (160 * t + j)) * win[j];
        const uint ph = (f * j) & 511;
        re += v * cs[ph];
        im -= v * sn[ph];
    }
    const float mag = fabs(re) + fabs(im);
    A[P.pow + t * 257 + f] = mag * mag;
}

// log mel lm[t][m] = log(sum_f fb[m][f] p[t][f] + 1e-5)
kernel void melfb(constant Mel & P [[buffer(0)]], device float * A [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint m = gid.x, t = gid.y;
    if (m >= 128 || t >= P.frames) return;
    device const float * fb = A + P.fb + m * 257, * p = A + P.pow + t * 257;
    float s = 0.f;
    for (uint f = 0; f < 257; ++f) s += fb[f] * p[f];
    A[P.lmel + t * 128 + m] = log(s + 1e-5f);
}

// per mel bin (one threadgroup of 256): mel[t][m] = (lm - mean) / (std + 1e-5), mean / population std over t
kernel void melnorm(constant Mel & P [[buffer(0)]], device float * A [[buffer(2)]], uint m [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device const float * lm = A + P.lmel + m;
    float s = 0.f;
    for (uint t = tid; t < P.frames; t += 256) s += lm[t * 128];
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0.f;
    for (uint i = 0; i < 8; ++i) mean += red[i];
    mean /= P.frames;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = 0.f;
    for (uint t = tid; t < P.frames; t += 256) { const float d = lm[t * 128] - mean; v += d * d; }
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.f;
    for (uint i = 0; i < 8; ++i) var += red[i];
    const float inv = 1.f / (sqrt(var / P.frames) + 1e-5f);
    for (uint t = tid; t < P.frames; t += 256) A[P.mel + t * 128 + m] = (lm[t * 128] - mean) * inv;
}

kernel void subflat(constant Sub & P [[buffer(0)]], device float * A [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, f = gid.y, t = gid.z;
    if (c >= P.C || f >= P.Fo || t >= P.To) return;
    A[P.y + (ulong) t * P.C * P.Fo + c * P.Fo + f] = A[P.x + ((ulong) t * P.Fo + f) * P.C + c];
}

// ---------------------------------------------------------------------------------------------------------------
// TDT greedy decode (fast_tdt semantics) as ordinary dispatches in one serial compute encoder: no threadgroup ever
// waits on another, so it runs on any Apple GPU under any load (the dispatch boundary is the only synchronization).
// Prediction net: 2-layer LSTM(640) on the last emitted token (zero input and state before the first), run only when
// a token is emitted; joint: z = relu(E[step] + pred(h)), logits = J z (1024 tokens + blank, 5 durations).
// One decode slot is five indirect dispatches whose sizes the previous slot's tdt_decide wrote:
//   tdt_l0, tdt_l1, tdt_pred (HD / LNU threadgroups each, or none when no token was emitted): layer 0, layer 1 and
//     the pred projection g; each threadgroup owns LNU hidden units (their 4 gate rows and cell state);
//   tdt_joint (NSL x nf threadgroups): the joint for the frames base .. base + nf - 1 (a run of blanks visits frames
//     that all share g), NSL row slices per frame, each slice's argmax / softmax / boosting partials;
//   tdt_decide (one threadgroup): reduces the partials frame by frame and takes tdt_decode.decode's decisions (blank:
//     next frame; token: emit; dictionary walks, commits and rollbacks) until it emits a token, rolls back, runs past
//     the batch or ends the window, then sizes the next slot (all zero once done).
// The host encodes slots in chunks until the done flag is set; a slot past the end is five empty dispatches. Every
// value is computed by the same expressions in the same order as the persistent kernel this replaced (one dispatch
// of 20 co-resident threadgroups on a spin barrier, which stalled when the GPU was small or shared), so tokens,
// durations, frames, confidences and spans are identical.
// The MLX app runs in bf16: h, c, the pred / enc projections, z and the logits are rounded to bf16 where MLX rounds.
#define HD 640
#define NOUT 1030
#define NTOK 1025
#define NSL 20                        // joint row slices per frame (fixed: the order of the softmax partial sums)
#define RPT ((NOUT + NSL - 1) / NSL)
#define PSTRIDE 12  // uints per slice partial: max, argmax, Z, Sx, duration max / argmax, runner-up, best child
#ifndef TDT_LNU
#define TDT_LNU 8                     // LSTM units per threadgroup (HD / TDT_LNU threadgroups of 256 threads)
#endif
#ifndef TDT_KF
#define TDT_KF 16                     // max frames per joint batch
#endif
#define NU TDT_LNU

struct Tdt {
    uint E, T, S, out, maxout;            // float offsets: E [T][640] (joint.enc output), state S, output
    uint emb, wx0, wh0, b0, wx1, wh1, b1; // half offsets of the weights
    uint pw, pb, jw, jb, max_symbols;
    uint dbg;                             // ~0 or a float offset for per-iteration debug records
    uint dbg_step, dbgl;                  // debug: at the first iteration on step dbg_step, logits [1030], z [640], g [640]
    uint xtab;                            // ~0 or the float offset of b0 + Wx0 embed[t], [1025][2560]
    uint trie, spans, maxspans;           // dictionary boosting: ~0 or the trie table (uint offset), committed spans out
};

// decoder control at S (written only by tdt_decide and the host; pa.mm's TdtCtl mirrors it)
struct TdtCtl {
    uint lstm_args[3], joint_args[3], decide_args[3];  // the next slot's indirect dispatch sizes
    uint done, step, need, tok, run, iter, nout, new_symbols, plain, boost_live, node, ck_step, ck_nout, nspans, n_flips,
        n_rollbacks;
    int no_boost_until;
    uint changed, dumped, base, nf, slots, pad;  // slots: tdt_decide runs (the slots the decode used)
    long deficit;
};
// state vectors (floats) after the 64-word control block: h0 [2][640], h1 [2][640] (ping-pong on run), g, c0, c1, the
// walk checkpoint of h0, h1, g, c0, c1, then the joint partials [TDT_KF][NSL][PSTRIDE]
#define ST_H0 64
#define ST_H1 (ST_H0 + 2 * HD)
#define ST_G (ST_H1 + 2 * HD)
#define ST_C0 (ST_G + HD)
#define ST_C1 (ST_C0 + HD)
#define ST_K (ST_C1 + HD)  // checkpoint: h0, h1, g, c0, c1 (HD each)
#define ST_PART (ST_K + 5 * HD)

// Dictionary boosting (asr_bias_v0 tdt_decode.py, config cb446: --variants case --s-start 4 --s-cont 4 --budget 6).
// Flat trie, uint words at P.trie: header [0] nodes, [1] edges, [2] s_start, [3] s_cont, [4] budget (float bits),
// [5] blank_start; node n at 16 + 4 n: {edge offset, edge count, terminal, canonical term}; edge e at E0 + 3 e:
// {token, child node, boostable} (a node's edges sorted by token); root[1025] at R0: the child of a first piece,
// | 1 << 31 when boostable, ~0 when none; word_start[1032] at R0 + 1025, boundary[1032] after it.
struct TrieV { device const uint * nd, * ed, * root, * ws, * bd; float s_start, s_cont; long budget_fx; bool blank_start; };
#define FX_ONE 1099511627776.0f  // 2^40: deficits summed exactly as fixed point (Python sums them in float64)
inline TrieV trie_view(device const uint * t) {
    TrieV v;
    const uint nodes = t[0], edges = t[1];
    v.nd = t + 16;
    v.ed = v.nd + 4 * nodes;
    v.root = v.ed + 3 * edges;
    v.ws = v.root + 1025;
    v.bd = v.ws + 1032;
    v.s_start = as_type<float>(t[2]);
    v.s_cont = as_type<float>(t[3]);
    v.budget_fx = (long) (as_type<float>(t[4]) * FX_ONE);
    v.blank_start = t[5] != 0;
    return v;
}
inline uint trie_child(TrieV v, uint node, uint k) {  // child node of `node` on token k, ~0 when none
    if (node == 0) {
        const uint c = k < 1025 ? v.root[k] : ~0u;
        return c == ~0u ? ~0u : (c & 0x7fffffffu);
    }
    const uint off = v.nd[4 * node], cnt = v.nd[4 * node + 1];
    for (uint j = 0; j < cnt; ++j)
        if (v.ed[3 * (off + j)] == k) return v.ed[3 * (off + j) + 1];
    return ~0u;
}

inline float sigm(float x) { return 1.f / (1.f + exp(-x)); }
// fast-math tanh returns NaN past |x| ~ 44 (exp overflow: inf / inf); found on a 28 s window whose LSTM gate hit it.
// Clamping at 10 is exact in float32: tanh(10) = 1 - 4e-9 rounds to 1.
inline float tanh_c(float x) { return tanh(clamp(x, -10.f, 10.f)); }

// dot of a 640-wide half weight row with a threadgroup float vector, by one simdgroup (lane: 5 x 4 elements)
inline float rowdot(device const half * w, threadgroup const float * v, ushort lane) {
    float s = 0.f;
    FOR_UNROLL (uint j = 0; j < 5; ++j) {
        const uint k = 128 * j + 4 * lane;
        const float4 a = float4(*(device const half4 *) (w + k));
        s += a.x * v[k] + a.y * v[k + 1] + a.z * v[k + 2] + a.w * v[k + 3];
    }
    return simd_sum(s);
}

// DEC_BF16 0 (default): the app's numerics. The sidecar hands generate() a float32 mel, so the encoder output, the
// decoder state and the joint are float32 against bf16 weights; the one bf16 op is layer 0's input projection
// addmm(bias, embed[token], Wx^T) (all three operands bf16). DEC_BF16 1: an all-bf16 MLX run (bf16 mel).
#ifndef DEC_BF16
#define DEC_BF16 0
#endif
#define RB(x) (DEC_BF16 ? bf16r(x) : (x))
inline float lstm_gate(float sx, float sh, float b, bool first, bool layer0) {
    if (first) return sx + b + sh;
    return DEC_BF16 ? bf16r(bf16r(sx + b) + sh) : layer0 ? bf16r(sx + b) + sh : sx + b + sh;
}
// gate rows [i | f | g | o] x NU of this threadgroup; updates the cell state c[u], returns h
inline float lstm_cell(threadgroup const float * gate, threadgroup float * c, uint u, bool first) {
    if (first || !DEC_BF16) {
        const float i = sigm(gate[u]), f = sigm(gate[NU + u]), g = tanh_c(gate[2 * NU + u]), o = sigm(gate[3 * NU + u]);
        const float cn = first ? i * g : f * c[u] + i * g;
        c[u] = RB(cn);
        return o * tanh_c(cn);
    }
    const float i = bf16r(sigm(gate[u])), f = bf16r(sigm(gate[NU + u])), g = bf16r(tanh_c(gate[2 * NU + u])), o = bf16r(sigm(gate[3 * NU + u]));
    const float cn = bf16r(bf16r(f * c[u]) + bf16r(i * g));
    c[u] = cn;
    return bf16r(o * bf16r(tanh_c(cn)));
}

#define TDT_ARGS constant Tdt & P [[buffer(0)]], device const half * W [[buffer(1)]], device float * A [[buffer(2)]], \
    uint tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]

// layer 0: x = embed[tok] (zeros before the first token), h = h0. MLX rounding: the first run (zero float32 input, no
// state) is float32 throughout and only its final h / c are cast to bf16; later runs are bf16 op by op: addmm(b, x,
// Wx^T), addmm(., h, Wh^T), the activations, f * c, i * g, their sum, tanh(c), o * tanh(c). States are read back
// rounded (bf16 casts). 256 threads; reads h0[run & 1], writes h0[run & 1 ^ 1].
kernel void tdt_l0(TDT_ARGS, uint tg [[threadgroup_position_in_grid]]) {
    device const TdtCtl & C = *(device const TdtCtl *) (A + P.S);
    device float * S = A + P.S;
    threadgroup float xv[HD], hv[HD], gate[4 * NU], cc[NU];
    const uint p = C.run & 1, q = p ^ 1;
    const bool first = C.run == 0;
    const int tok = (int) C.tok;
    // P.xtab: b0 + Wx0 embed[t] for every token, precomputed (float [1025][2560]): one row read replaces the 3.3 MB
    // input-projection matvec per emitted token
    const bool tab = P.xtab != ~0u && tok >= 0;
    for (uint k = tid; k < HD; k += 256) {
        if (!tab) xv[k] = tok < 0 ? 0.f : (float) W[P.emb + (uint) tok * HD + k];
        hv[k] = RB(S[ST_H0 + p * HD + k]);
    }
    if (tid < NU) cc[tid] = S[ST_C0 + tg * NU + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint lr = sg; lr < 4 * NU; lr += 8) {
        const uint row = (lr / NU) * HD + tg * NU + lr % NU;
        const float sh = rowdot(W + P.wh0 + row * HD, hv, lane);
        const float sx = tab ? A[P.xtab + (uint) tok * 4 * HD + row] : rowdot(W + P.wx0 + row * HD, xv, lane);
        if (lane == 0) gate[lr] = lstm_gate(sx, sh, tab ? 0.f : (float) W[P.b0 + row], first, true);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NU) {
        const float h = lstm_cell(gate, cc, tid, first);  // first run: unrounded h0 feeds layer 1 in float32
        S[ST_C0 + tg * NU + tid] = cc[tid];
        S[ST_H0 + q * HD + tg * NU + tid] = h;
    }
}

// layer 1: x = the new h0, h = h1; writes h1[run & 1 ^ 1] (the decoder output)
kernel void tdt_l1(TDT_ARGS, uint tg [[threadgroup_position_in_grid]]) {
    device const TdtCtl & C = *(device const TdtCtl *) (A + P.S);
    device float * S = A + P.S;
    threadgroup float xv[HD], hv[HD], gate[4 * NU], cc[NU];
    const uint p = C.run & 1, q = p ^ 1;
    const bool first = C.run == 0;
    for (uint k = tid; k < HD; k += 256) {
        xv[k] = S[ST_H0 + q * HD + k];
        hv[k] = S[ST_H1 + p * HD + k];
    }
    if (tid < NU) cc[tid] = S[ST_C1 + tg * NU + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint lr = sg; lr < 4 * NU; lr += 8) {
        const uint row = (lr / NU) * HD + tg * NU + lr % NU;
        const float sx = rowdot(W + P.wx1 + row * HD, xv, lane), sh = rowdot(W + P.wh1 + row * HD, hv, lane);
        if (lane == 0) gate[lr] = lstm_gate(sx, sh, (float) W[P.b1 + row], first, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NU) {
        const float h = RB(lstm_cell(gate, cc, tid, first));  // decoder output and state (bf16 run: bf16)
        S[ST_C1 + tg * NU + tid] = cc[tid];
        S[ST_H1 + q * HD + tg * NU + tid] = h;
    }
}

// pred projection of the decoder output (the new h1): g rows tg * NU .. + NU
kernel void tdt_pred(TDT_ARGS, uint tg [[threadgroup_position_in_grid]]) {
    device const TdtCtl & C = *(device const TdtCtl *) (A + P.S);
    device float * S = A + P.S;
    threadgroup float xv[HD];
    const uint q = (C.run & 1) ^ 1;
    for (uint k = tid; k < HD; k += 256) xv[k] = S[ST_H1 + q * HD + k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = sg; r < NU; r += 8) {
        const uint row = tg * NU + r;
        const float s = rowdot(W + P.pw + row * HD, xv, lane);
        if (lane == 0) S[ST_G + row] = RB(s + (float) W[P.pb + row]);
    }
}

// the joint at frame base + fi: z = relu(E[frame] + g), slice sl's logit rows, its argmax / softmax / boosting partials
kernel void tdt_joint(TDT_ARGS, uint2 tgp [[threadgroup_position_in_grid]]) {
    device const TdtCtl & C = *(device const TdtCtl *) (A + P.S);
    device float * S = A + P.S;
    const uint sl = tgp.x, fi = tgp.y, frame = C.base + fi;
    threadgroup float xv[HD], lg[RPT];
    for (uint k = tid; k < HD; k += 256) xv[k] = max(RB(RB(A[P.E + frame * HD + k]) + S[ST_G + k]), 0.f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r0 = sl * RPT, r1 = min((uint) NOUT, r0 + RPT);
    for (uint r = r0 + sg; r < r1; r += 8) {
        const float s = rowdot(W + P.jw + r * HD, xv, lane);
        if (lane == 0) lg[r - r0] = RB(s + (float) W[P.jb + r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (frame == P.dbg_step && !C.dumped) {  // debug: kept from the batch in which tdt_decide first visits the frame
        for (uint r = r0 + tid; r < r1; r += 256) A[P.dbgl + r] = lg[r - r0];
        if (sl == 0)
            for (uint k = tid; k < HD; k += 256) {
                A[P.dbgl + NOUT + k] = xv[k];
                A[P.dbgl + NOUT + HD + k] = S[ST_G + k];
            }
    }
    if (tid != 0) return;
    device uint * part = (device uint *) S + ST_PART + (fi * NSL + sl) * PSTRIDE;
    float m = -INFINITY, dm = -INFINITY;
    uint mi = 0, di = 0;
    for (uint r = r0; r < r1; ++r) {
        const float v = lg[r - r0];
        if (r < NTOK) { if (v > m) { m = v; mi = r; } }
        else if (v > dm) { dm = v; di = r - NTOK; }
    }
    // boosting: the best boostable child of the current trie node among this slice's tokens (lowest token on ties,
    // as ct[argmax] over the sorted children)
    float bl = -INFINITY;
    uint bc = ~0u;
    if (C.boost_live) {
        const TrieV TR = trie_view((device const uint *) A + P.trie);
        const uint node = C.node, rt = min(r1, (uint) NTOK);
        if (node == 0) {
            for (uint r = r0; r < rt; ++r) {
                const uint c = TR.root[r];
                if (c != ~0u && (c >> 31) && lg[r - r0] > bl) { bl = lg[r - r0]; bc = r; }
            }
        } else {
            const uint off = TR.nd[4 * node], cnt = TR.nd[4 * node + 1];
            for (uint j = 0; j < cnt; ++j) {
                const uint tk = TR.ed[3 * (off + j)];
                if (TR.ed[3 * (off + j) + 2] && tk >= r0 && tk < rt && lg[tk - r0] > bl) { bl = lg[tk - r0]; bc = tk; }
            }
        }
    }
    part[7] = as_type<uint>(bl);
    part[8] = bc;
    float Z = 0.f, Sx = 0.f;
    for (uint r = r0; r < min(r1, (uint) NTOK); ++r) { const float d = lg[r - r0] - m, e = exp(d); Z += e; Sx += e * d; }
    part[0] = as_type<uint>(m);
    part[1] = mi;
    part[2] = as_type<uint>(Z);
    part[3] = as_type<uint>(Sx);
    part[4] = as_type<uint>(dm);
    part[5] = di;
    // second-best token logit of the slice (debug: the margin of the argmax)
    float m2 = -INFINITY;
    for (uint r = r0; r < min(r1, (uint) NTOK); ++r) if (r != mi) m2 = max(m2, lg[r - r0]);
    part[6] = as_type<uint>(m2);
}

// the decisions over the joint batch (one threadgroup of 256: thread 0 decides, all copy the checkpoint state)
// output (uints): [0] count, then per token: id, duration index, step, confidence (float bits)
kernel void tdt_decide(TDT_ARGS) {
    device TdtCtl & C = *(device TdtCtl *) (A + P.S);
    device float * S = A + P.S;
    device uint * out = (device uint *) A + P.out;
    threadgroup uint copy[2];  // [0] checkpoint the state, [1] restore it (after the checkpoint when both)
    if (tid == 0) {
        const bool trie_on = P.trie != ~0u;
        const TrieV TR = trie_view((device const uint *) A + (trie_on ? P.trie : 0));
        uint step = C.step, run = C.run, iter = C.iter, nout = C.nout, new_symbols = C.new_symbols, node = C.node;
        uint ck_step = C.ck_step, ck_nout = C.ck_nout, nspans = C.nspans, n_flips = C.n_flips, n_rollbacks = C.n_rollbacks;
        bool need = C.need != 0, plain = C.plain != 0, boost_live = C.boost_live != 0, changed = C.changed != 0, dumped = C.dumped != 0;
        int tok = (int) C.tok, no_boost_until = C.no_boost_until;
        long deficit = C.deficit;
        const uint base = C.base, nf = C.nf;
        bool do_ck = false, do_rs = false, done = false;
        if (need) { ++run; need = false; }  // this slot ran the prediction net
        const float max_entropy = log((float) NTOK);
        while (step - base < nf) {
            if (step == P.dbg_step && !dumped) dumped = true;
            device const uint * PART = (device const uint *) S + ST_PART + (step - base) * NSL * PSTRIDE;
            float m = -INFINITY, dm = -INFINITY, BL = -INFINITY;
            uint mi = 0, di = 0, BC = ~0u;
            float pm[NSL], pz[NSL], ps[NSL];
            for (uint t = 0; t < NSL; ++t) {
                device const uint * pt = PART + t * PSTRIDE;
                const float b = as_type<float>(pt[7]);
                if (b > BL) { BL = b; BC = pt[8]; }
                pm[t] = as_type<float>(pt[0]);
                pz[t] = as_type<float>(pt[2]);
                ps[t] = as_type<float>(pt[3]);
                if (pm[t] > m) { m = pm[t]; mi = pt[1]; }
                const float d = as_type<float>(pt[4]);
                if (d > dm) { dm = d; di = pt[5]; }
            }
            float Z = 0.f, Sx = 0.f;
            for (uint t = 0; t < NSL; ++t) {
                if (pz[t] == 0.f) continue;  // no token rows in this slice
                const float sc = exp(pm[t] - m);
                Z += pz[t] * sc;
                Sx += sc * (ps[t] + (pm[t] - m) * pz[t]);
            }
            const float H = log(Z) - Sx / Z;
            const uint conf = as_type<uint>(1.f - H / max_entropy);
            if (P.dbg != ~0u && iter < 512) {  // per iteration: step, argmax, duration index, margin to runner-up
                float m2 = -INFINITY;
                for (uint t = 0; t < NSL; ++t) {
                    device const uint * pt = PART + t * PSTRIDE;
                    m2 = max(m2, as_type<float>(pt[6]));
                    if (pt[1] != mi && pm[t] > -INFINITY) m2 = max(m2, pm[t]);
                }
                device uint * D = (device uint *) A + P.dbg + 4 * iter;
                D[0] = step; D[1] = mi; D[2] = di; D[3] = as_type<uint>(m - m2);
            }
            // ---- the step's decision (tdt_decode.decode) ----
            uint k = mi, d = di, act = 0;
            float delta = 0.f;
            bool flip = false;
            if (boost_live && (int) step > no_boost_until && BC != ~0u && trie_child(TR, node, mi) == ~0u) {
                const float s = node == 0 ? TR.s_start : TR.s_cont;
                const bool allowed = node != 0 || (mi != NTOK - 1 && TR.ws[mi]) || (TR.blank_start && mi == NTOK - 1);
                // logit[c] + s > logit[k0], exactly (Python compares in float64): k0 - c is exact here (Sterbenz)
                if (allowed && m - BL < s) { k = BC; flip = true; delta = m - BL; d = min(d, 1u); }
            }
            if (k == NTOK - 1) {  // blank: next frame (a 0 duration counts as 1)
                step += max(d, 1u);
                new_symbols = 0;
            } else {
                bool emit = true;
                if (trie_on && !plain) {
                    if (node != 0) {
                        const uint ch = trie_child(TR, node, k);
                        if (ch != ~0u) {
                            node = ch;
                            changed = changed || flip;
                            deficit += (long) (delta * FX_ONE);
                        } else {
                            const bool ok = TR.nd[4 * node + 2] && TR.bd[k] && deficit <= TR.budget_fx;
                            if (!ok && changed) {  // roll back to the walk's start, decode it unboosted
                                no_boost_until = (int) step;
                                step = ck_step;
                                nout = ck_nout;
                                need = false;
                                new_symbols = 0;
                                act = 2;
                                emit = false;
                                ++n_rollbacks;
                            } else if (ok) {  // a committed term: span [walk start, here) -> canonical spelling
                                if (nspans < P.maxspans) {
                                    device uint * sp = (device uint *) A + P.spans + 1 + 4 * nspans;
                                    sp[0] = ck_nout; sp[1] = nout; sp[2] = TR.nd[4 * node + 3]; sp[3] = changed;
                                }
                                ++nspans;
                            }
                            node = 0; changed = false; deficit = 0;
                        }
                    }
                    if (emit && node == 0) {
                        const uint ch = trie_child(TR, 0, k);
                        if (ch != ~0u) {  // a walk starts: checkpoint the decoder state before this piece
                            ck_step = step; ck_nout = nout;
                            node = ch; changed = flip; deficit = (long) (delta * FX_ONE);
                            act = 1;
                        }
                    }
                }
                if (emit) {
                    if (nout < P.maxout) {
                        out[1 + 4 * nout] = k; out[2 + 4 * nout] = d; out[3 + 4 * nout] = step; out[4 + 4 * nout] = conf;
                    }
                    n_flips += flip;
                    ++nout;
                    tok = (int) k;
                    need = true;
                    step += d;  // durations [0, 1, 2, 3, 4]
                    ++new_symbols;
                    if (d != 0) new_symbols = 0;
                    else if (new_symbols >= P.max_symbols) { step += 1; new_symbols = 0; }
                }
            }
            boost_live = trie_on && !plain;
            do_ck = do_ck || act == 1;
            do_rs = do_rs || act == 2;
            ++iter;
            if (step >= P.T) {
                // end of the window inside a walk (tdt_decode: no boundary check): commit, or roll back and decode
                // the rest plainly (decode_from: no boosting, no trie)
                if (trie_on && !plain && node != 0) {
                    const bool ok = TR.nd[4 * node + 2] && deficit <= TR.budget_fx;
                    if (!ok && changed) {
                        step = ck_step; nout = ck_nout; need = false; new_symbols = 0;
                        plain = true; boost_live = false; ++n_rollbacks;
                        do_rs = true;
                    } else if (ok) {
                        if (nspans < P.maxspans) {
                            device uint * sp = (device uint *) A + P.spans + 1 + 4 * nspans;
                            sp[0] = ck_nout; sp[1] = nout; sp[2] = TR.nd[4 * node + 3]; sp[3] = changed;
                        }
                        ++nspans;
                    }
                    node = 0; changed = false; deficit = 0;
                }
                if (step >= P.T) { done = true; break; }
            }
            if (need || do_rs) break;  // a token (the prediction net runs next) or a rollback (the joint reruns)
        }
        C.step = step; C.need = need; C.tok = (uint) tok; C.run = run; C.iter = iter; C.nout = nout;
        C.new_symbols = new_symbols; C.plain = plain; C.boost_live = boost_live; C.node = node; C.ck_step = ck_step;
        C.ck_nout = ck_nout; C.nspans = nspans; C.n_flips = n_flips; C.n_rollbacks = n_rollbacks;
        C.no_boost_until = no_boost_until; C.changed = changed; C.dumped = dumped; C.deficit = deficit;
        C.done = done;
        C.slots += 1;
        const uint nf2 = done ? 0 : min((uint) TDT_KF, P.T - step);
        C.base = step; C.nf = nf2;
        C.lstm_args[0] = !done && need ? HD / NU : 0; C.lstm_args[1] = 1; C.lstm_args[2] = 1;
        C.joint_args[0] = done ? 0 : NSL; C.joint_args[1] = done ? 0 : nf2; C.joint_args[2] = 1;
        C.decide_args[0] = done ? 0 : 1; C.decide_args[1] = 1; C.decide_args[2] = 1;
        out[0] = nout;
        if (P.spans != ~0u) {
            device uint * sp = (device uint *) A + P.spans;
            sp[0] = min(nspans, P.maxspans);
            sp[1 + 4 * P.maxspans] = n_flips;
            sp[2 + 4 * P.maxspans] = n_rollbacks;
        }
        copy[0] = do_ck; copy[1] = do_rs;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (!copy[0] && !copy[1]) return;
    const uint cur = (C.run & 1) * HD;  // the newest h0 / h1 (run was advanced above)
    if (copy[0])  // checkpoint the decoder state before the walk's first piece
        for (uint u = tid; u < HD; u += 256) {
            S[ST_K + u] = S[ST_H0 + cur + u];
            S[ST_K + HD + u] = S[ST_H1 + cur + u];
            S[ST_K + 2 * HD + u] = S[ST_G + u];
            S[ST_K + 3 * HD + u] = S[ST_C0 + u];
            S[ST_K + 4 * HD + u] = S[ST_C1 + u];
        }
    if (copy[1]) {  // restore it (the same thread copies a unit both ways, so no barrier between)
        for (uint u = tid; u < HD; u += 256) {
            S[ST_H0 + cur + u] = S[ST_K + u];
            S[ST_H1 + cur + u] = S[ST_K + HD + u];
            S[ST_G + u] = S[ST_K + 2 * HD + u];
            S[ST_C0 + u] = S[ST_K + 3 * HD + u];
            S[ST_C1 + u] = S[ST_K + 4 * HD + u];
        }
    }
}
