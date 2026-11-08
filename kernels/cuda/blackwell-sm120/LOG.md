# Phonon refiner on RTX 5090 (Blackwell consumer, sm_120): hill-climb log

Box: vast.ai instance 59474761 (RTX 5090 32 GB, driver 580.178.04, CUDA 12.8, Ryzen 7 7800X3D), 2026-11-06,
about 4:00 to 4:52 pm Alberta time. Destroyed at the end; cost about US$0.47 (0.95 h at US$0.496/h).

Model: phonon-refiner-1.7b-Q4_K_S-e4.gguf (LFM3, 24 layers: 12 short-conv + 12 attention, n_embd 2048, n_ff 7680,
vocab 128000 tied output, Q4_K with 7 Q5_K tensors). Metric: `llama-bench -m refiner.gguf -ngl 99 -pg 16,32 -r 10`,
ms/dictation = 48 / (pp16+tg32 t/s) * 1000.

## Final numbers (clean b12132 + b12132.patch build, two interleaved rounds, same box)

| build             | flash attn     | pp16+tg32 t/s (r1 / r2)    | ms/dictation  | tg32 t/s      | pp16 t/s        | pp64 t/s         |
|-------------------|----------------|----------------------------|---------------|---------------|-----------------|------------------|
| stock b12132      | default (auto) | 1053.12 / 1054.09 (+-2.4)  | 45.58 / 45.54 | 754.8 / 755.5 | 6440 / 6453     | 18145 / 18142    |
| patched           | default (auto) | 1179.12 / 1178.71 (+-1.6)  | 40.71 / 40.72 | 843.5 / 842.6 | 6601 / 6617     | 18289 / 18278    |
| stock b12132      | -fa 0          | 999.76 / 1001.08           | 48.01 / 47.95 | 712.4 / 712.2 | 6456 / 6531     | 17604 / 17503    |
| patched           | -fa 0          | 1106.50 / 1107.55          | 43.38 / 43.34 | 787.9 / 788.0 | 6575 / 6615     | 17598 / 17549    |
| stock b12132      | -fa 1          | same as default (auto picks FA on this box) |  |               |                 |                  |

Default (auto = FA on): 45.56 -> 40.72 ms per dictation (-10.6%), tg32 +11.6%. pp16 spread over -r 10 is about
+-1200 t/s for both builds (the first repetition is slow), so the pp16 change (+2%) is inside noise; pp64 unchanged.
The tg32 and pp16+tg32 spreads are +-1.2 to +-5 t/s, so the 12% gain is far outside noise.

Correctness: greedy output (temp 0, top_k 1, 64 tokens, --ignore-eos) is byte-identical to stock b12132 for all four
gate prompts at every step of the climb, and for an extra test (~900-token prompt, -b 128 -ub 64 so prefill runs in
several micro-batches, 200 generated tokens). Every change keeps the exact float operation order of the kernels it
replaces, so this is bit-exact by construction, not by luck.

## Profile of stock b12132 (nsys 2024.6, per decoded token, tg)

- 1334 us per token = 1200 us GPU (two CUDA-graph chunks of 566 + 635 us, GGML_CUDA_GRAPH_CHUNK_SIZE=512)
  + 130 us CPU gap between tokens.
- CPU gap: 13 graph inputs each uploaded as sync + cudaMemcpyAsync + cudaStreamSynchronize (~6 us each, 78 us),
  two cudaGraphLaunch at ~14 us, the rest llama-side graph reuse and the logits copy.
- GPU: ~530 kernels per token. 130 quantize_q8_1 (one per mat-vec, even when q/k/v/gate share the same input),
  97 rms_norm, 123 Q4_K + 7 Q5_K mat-vecs, ~150 small elementwise/conv/attention kernels.
- PDL (programmatic dependent launch) is already on by default in b12132 and worth 18% (GGML_CUDA_PDL=0 drops tg32
  from 757 to 617 t/s), so plain launch overhead is mostly hidden already; removing a small kernel saves only ~0.4 to
  0.7 us of GPU time.
- Isolated mat-vec speed (test-backend-ops perf on the model's own shapes, L2 warm): ffn_down 7680x2048 Q4_K 4.7 us
  (1772 GB/s), in_proj 6144x2048 4.4 us, 2048x2048 2.0 us, 512x2048 1.7 us, output 128000x2048 96 us (1434 GB/s).
  Summed per token that is ~640 us of mat-vec; the other ~450 us of GPU time is the chain of small dependent kernels.
- ncu is not usable on vast.ai (ERR_NVGPUCTRPERM), so no counter-level data.

## Hill-climb (each row = cumulative build; pp16+tg32 t/s, default FA; gate = 4 prompts identical)

| step | change                                                                                   | t/s     | ms/dict | gate | kept |
|------|------------------------------------------------------------------------------------------|---------|---------|------|------|
| 0    | stock b12132                                                                             | 1054.05 | 45.54   | ref  |      |
| 1    | mmvq: PDL early trigger + L2 prefetch of the block's own weight rows before grid sync     | 1051.82 | 45.64   | same | yes (pays off in step 7) |
| 2    | q8_1 activation cache per graph + fused rms_norm*w writes the q8_1 copy (drops ~96 quantize kernels) | 1091.99 | 43.96 | same | yes |
| 3    | small (<=64 KB) graph-input uploads enqueued on the compute stream, no per-input sync (pinned src bounced via pageable) | 1098.67 | 43.69 | same | yes |
| 4    | ggml-backend sched: one backend sync per split before user inputs, not one per input      | 1123.63 | 42.72   | same | yes |
| 5    | fused SSM_CONV + MUL(c gate), writes out_proj's q8_1 copy (12 layers)                     | 1135.96 | 42.25   | same | yes |
| 6    | fused short-conv state update: MUL(b,x)+TRANSPOSE+CONCAT+VIEW+CPY into one kernel          | 1145.32 | 41.91   | same | yes |
| 7    | mmvq Q4_K/Q5_K, 1 column: 2 rows per block (per-row sum order unchanged)                  | 1177.09 | 40.78   | same | yes |
| 7b   | 4 rows per block                                                                          | 1070.22 | 44.85   | same | no   |
| 7c   | step 7 without the step-1 own-row prefetch                                                | 1135.24 | 42.28   | same | no (prefetch worth 3.7% at 2 rows) |
| 8    | each mat-vec also L2-prefetches the next mat-vec group's weights (before sync; cap 16/32/64 MB) | 1121.9 / 1118.6 / 1118.5 | 42.8 | same | no |
| 8b   | same, issued after the main loop (cap 8/16 MB)                                            | 1093.2 / 1066.9 | 43.9+ | same | no |
| final| clean b12132 + b12132.patch                                                               | 1179.12 / 1178.71 | 40.71 / 40.72 | same | |

Stop rule: steps 7 (+2.8%) and 8 (negative) were two consecutive ideas under 3%.

Other things measured and not pursued: GGML_CUDA_GRAPH_CHUNK_SIZE=0 (one graph per token) 1051 vs 1053 t/s;
GGML_CUDA_GRAPH_OPT=1 (concurrent streams) 1049; GGML_CUDA_DISABLE_FUSION=1 942 (upstream fusions are worth 12%).

## What the patch changes (b12132.patch, 8 files, all under ggml/)

- ggml-cuda/mmvq.cu: on sm_120 only (`__CUDA_ARCH__ == 1200`), the dense mat-vec kernel triggers PDL launch completion
  at entry and prefetches its own weight rows into L2 before cudaGridDependencySynchronize; Q4_K/Q5_K single-column
  mat-vec uses 2 rows per block (PHONON_MMVQ_KQ_ROWS); single-column activations are quantized once per graph into a
  context-owned q8_1 buffer and reused by every mat-vec reading the same data (q/k/v/gate share one).
- ggml-cuda/norm.cu, norm.cuh: the vectorized rms_norm*w kernel optionally writes the q8_1 copy (same math as
  quantize_q8_1, 8 lanes x float4 per 32-value block).
- ggml-cuda/phonon-fused.cu/.cuh (new): the two short-conv fusions.
- ggml-cuda/ggml-cuda.cu: fusion matching for the above, q8 cache invalidation on any node that writes the cached
  range, cache buffer (1 MB) allocated at the first graph compute, async small input uploads when exactly one CUDA
  backend context exists on the device.
- ggml-cuda/common.cuh: three context fields for the q8 cache.
- ggml-backend.cpp: user-input copies of one split wait for the split backend once instead of once per input.

## How to build

    git clone --depth 1 --branch b12132 https://github.com/ggml-org/llama.cpp && cd llama.cpp
    patch -p1 < b12132.patch
    cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j --target llama-bench llama-completion   # new .cu file: run a fresh cmake configure

## Does it carry to other architectures

- Architecture-neutral, should carry to Ampere/Ada/Hopper: the q8_1 activation cache and norm-side quantize, the
  short-conv fusions, the input upload and scheduler sync changes (these are CPU-side and fixed ~35 us per token).
- PDL-dependent: the early trigger + own-row prefetch is compiled only for sm_120 and needs PDL (sm_90+); it was
  worth ~3.7% here and only after the 2-rows change.
- 2 rows per block is applied in the generic mmvq parameter table, so a build of this patch for sm_86/sm_89 would
  also use it; it is untested there and may need its own measurement.
