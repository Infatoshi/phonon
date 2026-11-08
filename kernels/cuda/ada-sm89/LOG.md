# Phonon refiner on Ada (RTX 4090, sm_89): hill-climb log

Target: Infatoshi/phonon-refiner-1.7b @ 2b0ab74, phonon-refiner-1.7b-Q4_K_S-e4.gguf (988,453,696 bytes), LFM3,
batch 1. Metric: `llama-bench -m refiner.gguf -ngl 99 -pg 16,32 -r 10`, ms per dictation = 48 / (pg t/s) * 1000.

Box: vast.ai instance 59474398, RTX 4090 24 GB (power limit 350 W, below the 450 W retail default), driver 595.99.02,
CUDA 12.8.93, AMD EPYC 7452 container with a ~15-CPU cgroup quota. Built for sm_89 only
(`-DCMAKE_CUDA_ARCHITECTURES=89`). Run on 2026-11-06, about 4:00 to 5:00 pm Alberta time.

## Result

Three alternating stock/final runs per setting, each `-r 10`; numbers are the mean of the three llama-bench averages.

| setting      | build | ms/dictation | pg16+32 t/s | pp16 t/s | pp64 t/s | tg32 t/s |
|--------------|-------|-------------:|------------:|---------:|---------:|---------:|
| fa default   | stock |        67.02 |       716.3 |     5395 |    16316 |    493.4 |
| fa default   | final |        55.52 |       864.5 |     5747 |    17133 |    581.4 |
| -fa 1        | stock |        66.63 |       720.4 |     5496 |    16446 |    493.6 |
| -fa 1        | final |        55.98 |       857.6 |     5756 |    17075 |    572.8 |
| -fa 0        | stock |        68.15 |       704.3 |     5494 |    15793 |    480.0 |
| -fa 0        | final |        56.88 |       843.9 |     5681 |    15995 |    565.7 |

- Default (flash attention auto, which picks the vec FA kernel here): 67.0 -> 55.5 ms per dictation, -17.2%.
  Per-run means: stock 67.16 / 66.06 / 67.83, final 55.57 / 55.67 / 55.33. Medians of the 10 samples: stock
  66.85 / 65.38 / 67.68, final 55.01 / 55.15 / 54.98.
- Noise: within a run the pg sd is 7-13 t/s for stock and 23-39 t/s for final; the final spread comes from the first
  of the 10 samples, which still pays the one-time CUDA graph capture (about 70 ms vs about 55 ms). Between runs the
  stock mean moved by 1.8 ms (66.06 to 67.83).
- pp16 sd is about 1200 t/s for both builds in every run (first-rep warmup), so treat pp16 as rough.
- The reference 3090 numbers in the brief (pp16 3640, pp64 10262, tg32 449) are from another box; this 4090 runs
  stock tg32 at only 493 t/s because decode is dominated by fixed per-token costs, not bandwidth (see profile).

Correctness: greedy output (llama-completion, temp 0, top-k 1, 64 tokens) for the four brief prompts is
byte-identical to stock b12132 on this box with fa default, `-fa 1` and `-fa 0`. Every fused kernel reproduces the
stock arithmetic order (same block reduction, same FMA shape, quantization max and integer sums are
order-independent), so no top-1 fallback was needed.

## Profile of stock (nsys 2024.6, per decode token, steady state)

- 526 kernels per token, 1749 us kernel time, 1827 us first-to-last span (per-node trace); 2 CUDA graph chunks.
- Mat-vecs: about 1190 us. FFN up+gate fused 19.7 us x 24 at 897 GB/s, lm head 152 us at 967 GB/s, in_proj 812 GB/s,
  but the K=2048 q/o/out_proj mat-vecs run at 370-620 GB/s and k/v/gate at 13-410 GB/s.
- The other ~396 kernels are 1-2 us each (130 quantize_q8_1, 97 rms_norm, 37 mul, 22 set_rows, 15 get_rows, ...),
  about 550 us. Sm_89 has no programmatic dependent launch, so every kernel boundary is a full drain.
- Host: per token the scheduler uploaded ~14 tiny inputs, each memcpy + cudaStreamSynchronize (~130 us idle GPU);
  logits D2H 512 KB ~27-58 us.
- Every pg16+32 rep ran the 16-token prompt and the first two decode tokens eagerly (~6.5 us CPU per kernel launch on
  this EPYC, ~3.5 ms per eager token): prompt and decode graphs share one CUDA-graph cache key (the first node), so the
  prompt evicted the decode graph and graph warmup restarted every dictation (GGML_CUDA_GRAPH_DEBUG shows
  "warmup reset" once per rep).

Final build per decode token: 334 kernels, 1532 us kernel time, 1552 us span; graph-level trace 755 us vs 870 us per
graph chunk.

## Hill-climb

ms/dictation is the mean of two paired runs (candidate vs its parent, alternating, `-r 10` each). "identical" means
the four-prompt greedy output matched stock byte for byte.

| step | change | parent ms | step ms | delta | correctness | kept |
|------|--------|----------:|--------:|------:|-------------|------|
| s1 | CUDA graph cache key mixes node count and token counts, so prompt and decode graphs coexist | 66.87 | 64.61 | -3.4% | identical | yes |
| s2 | skip per-upload sync for pageable sources | 65.28 | 64.17 | n/a | identical | superseded: the inputs are pinned, the path never fired (profile) |
| s2b | same for pinned sources, compute stream waits on an event | s1 | 63.57 | ~-1.5% (upload phase 130 -> 100 us/token) | identical | superseded by s7 |
| s3 | one-slot q8_1 cache in the mmvq host path (q, k, v, gate quantize their shared input once) | 63.57 | 62.83 | -1.2% | identical | yes |
| s4 | fused [rms_norm, mul, add, rms_norm, mul] at each layer boundary, writing the q8_1 copy for the next mat-vec | 62.11 | 59.37 | -4.4% | identical | yes (first try did not fire: generic fusion memory check rejected exact output/input aliasing; replaced with an exact-alias-safe check) |
| s5 | Ada: take the small-K mmvq path when K exactly fills one block iteration | 59.26 | 60.10 | +1.4% | identical | no, reverted |
| s6 | fused LFM short-conv decode step: b*x, concat with state, state write, 7-tap conv, c*conv, q8_1 for out_proj (6 kernels -> 1; SSM_CONV and MUL run early, marked done) | 59.25 | 56.79 | -4.1% (median -4.6%) | identical | yes |
| s7 | batched graph-input uploads: small set_tensor into compute buffers goes to a host staging list, flushed as one H2D copy + one scatter kernel before any device access | 56.87 | 55.87 | -1.8% | identical | yes |
| s8 | 2 rows per mmvq block for Q4_K/Q6_K (bit-exact by construction) | 55.57 | 56.04 | +0.8% | identical | no, reverted |

Stopped after s7 (1.8% faster) and s8 (0.8% slower): two consecutive ideas under 3%, per the brief.

Also tried with no code change: `GGML_CUDA_GRAPH_OPT=1` (concurrent streams for independent branches): 67.04 vs
66.70 ms, no gain.

## What the patch changes (all in ggml/src/ggml-cuda, marked `[phonon]`)

1. `ggml_cuda_graph_get_key`: key = hash(first node, n_nodes, first/last node ne). Decode keeps its CUDA graph
   across prompts. In llama-bench the identical pp16 also becomes graphed; in Phonon prompt lengths vary, so expect
   the decode part of this gain, not the prompt part.
2. `q8_cache_*` on `ggml_backend_cuda_context` + `ggml_cuda_mul_mat_vec_q`: single-row mat-vec inputs are quantized
   into a persistent 64 KB slot keyed by (tensor, data pointer, ne), reset at every graph evaluation.
3. `rms_norm_chain_f32` (norm.cu) + matcher in `ggml_cuda_try_fuse`.
4. `lfm_shortconv_step_f32` (ssm-conv.cu) + `ggml_cuda_try_fuse_lfm_shortconv`: matches the non-contiguous LFM
   short-conv chain for one token (the next layer's state GET_ROWS is scheduled in the middle of it), checks that the
   nodes in between do not use or overlap the fused tensors, runs everything at the first MUL, and skips SSM_CONV and
   the gating MUL when the loop reaches them.
5. Batched uploads (`ggml_cuda_upload_batch`): set_tensor of <= 64 KB into COMPUTE-usage buffers is staged on the
   host; every other device access (get/set/memset/cpy/clear/free on buffers, backend async ops, graph_compute) flushes
   first. Weights, KV and recurrent caches keep the stock synchronous path.

Prefill (n_tokens > 1) takes the stock kernels except where fix 1 or 5 applies.

## Where the remaining time goes (final, per decode token, about 1.65 ms)

- ~1190 us mat-vecs (about 83% of the 4090's ~1 TB/s for ~985 MB of weights per token).
- ~340 us in ~210 small kernels: attention block (q/k norm-rope, set_rows, FA vec + combine, sigmoid, gate mul,
  quantize), 13 recurrent-state GET_ROWS copies, 24 quantizes of the SwiGLU output for ffn_down.
- ~100 us host between graphs (llama-bench loop, 15 x ~1 us scheduler syncs, upload flush ~17 us, graph start ~20 us)
  plus ~40 us logits D2H (512 KB).
- Not attempted for lack of payoff per effort under the stop rule: sigmoid*mul+q8 fusion in the attention tail
  (~24 us), folding the state GET_ROWS into the short-conv kernel (~19 us), a multi-matrix q/k/v launch (v is Q6_K
  while q/k are Q4_K, so no simple weight concat), a persistent decode megakernel.

## Does it carry to other architectures?

Nothing here uses sm_89-specific instructions; the only Ada-specific tuning tried (s5, s8) lost. The gains come from
removing per-token fixed costs (eager tokens, kernel count, host syncs), so they should carry to Ampere (3090) and
Blackwell (5090). Expect a larger relative gain where kernels finish faster or the host is slower (5090), and a
somewhat smaller one on Hopper/Blackwell where PDL already overlaps kernel boundaries. Projection, not measured.

## How to build

    git clone --depth 1 --branch b12132 https://github.com/ggml-org/llama.cpp
    cd llama.cpp && git apply /path/to/b12132.patch
    cmake -S . -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
    cmake --build build-cuda -j --target llama-bench llama-completion
    build-cuda/bin/llama-bench -m refiner.gguf -ngl 99 -pg 16,32 -r 10

Patch: 7 files, +633/-5 lines against b12132 (commit 2ada830); `git apply --check` passes on a pristine tree.
