# Merged CUDA patch for the Phonon refiner (llama.cpp b12132), all architectures

Date: 2026-11-06, about 5:56 pm to 7:15 pm Alberta time.
Patch: `b12132.patch` (applies with `git apply` to tag b12132, commit 2ada830; 10 files, +1629/-10; md5
cab9042393b861c5baca695d0c375d75). One source tree; each build compiles for one sm (`-DCMAKE_CUDA_ARCHITECTURES=<sm>`).
Workload and metric as in the per-arch tracks: `llama-bench -m refiner.gguf -ngl 99 -pg 16,32 -p 16,64 -n 32 -r 10`,
ms/dictation = 48 / (pg16,32 t/s) * 1000, default flash-attention setting (auto).

## Result (merged = the shipped patch, "v3" below)

Every card: stock, the arch's own patch and merged built in the same b12132 tree on the same box (static builds),
benchmarked alternated stock / own / merged. Numbers are the mean of the llama-bench averages, (min-max) over rounds.

| card (sm), box | stock ms | own patch ms | merged ms | merged vs own | rounds |
|---|---|---|---|---|---|
| RTX 5090 (120), vast 500 W | 44.74 (44.73-44.76) | 40.16 (40.13-40.18) | 37.24 (37.23-37.25) | -7.3% | 2 full + 3 pg-only (-r 20): own 40.18-40.22, merged 37.07-37.15 |
| RTX 4090 (89), vast 450 W | 65.51 (65.41-65.60) | 55.15 (55.11-55.18) | 54.55 (54.54-54.56) | -1.1% | 2 full + 5 pg-only (-r 20): own 54.87-55.45, merged 54.30-55.16 |
| RTX 3090 (86), local, 420 W | 75.58 (75.43-75.74) | 67.26 (67.14-67.33) | 64.28 (64.10-64.46) | -4.4% | 3 full |
| RTX 2080 Ti (75), vast 300 W | 119.69 (118.87-120.51) | 100.69 (99.53-101.84) | 102.24 (99.31-105.17) | tie, see note | 2 full + 8 pg-only (-r 20): own mean 103.63 (100.94-107.20), merged mean 101.62 (99.20-103.57) |
| GTX 1660 Ti (75), vast 120 W | 208.17 (204.75-210.20) | 191.03 (187.18-194.42) | 188.01 (185.72-190.25) | -1.6% | 3 full |
| GTX 1080 Ti (61), vast 130 W | 233.99 (232.58-235.39) | 171.31 (171.28-171.35) | 155.74 (155.43-156.04) | -9.1% | 2 full + 2 pg-only: own 171.37/171.72, merged 156.17/156.18 |

- 2080 Ti note: that host ran at load average ~30-40 from other tenants the whole time, so single llama-bench runs swing
  +-3-7 ms. In the 8 extra alternated pg-only rounds merged won 6 of 8 (mean 101.6 vs 103.6 ms); in the 2 full rounds
  it won one and lost one. Call it at least as fast, within that box's noise; the earlier v2 build gave the same picture
  (12 alternated rounds, means 100.27 vs 100.56).
- tg32 t/s stock / own / merged: 5090 766 / 855 / 912; 4090 507 / 594 / 608; 3090 448 / 509 / 525; 2080 Ti 276 / 331 / 331;
  1660 Ti 176 / 197 / 198; 1080 Ti 145 / 201 / 222.
- Prefill: pp64 unchanged within noise everywhere except the 5090, where merged v3 measured 17342 t/s against stock
  18466 and own 18700 (2 runs each, sd ~900-1100). The v1 build on the 5090 had pp64 18624-18756, so the drop comes
  with the v3 upload staging; cause not found (the boxes were gone by then). pp16, which is Phonon's prefill size, is
  unchanged there (6655-6660 vs stock 6548-6631). Projection: no effect on dictation latency, it is inside pg16,32 above.
- Numbers from rented boxes are not comparable across boxes (power caps, shared CPUs); compare within a row.

## Correctness

Gate on every card, stock vs own vs merged: the four brief prompts, 64 tokens, with `llama-simple` (greedy),
`llama-completion --temp 0 --top-k 1` (fa auto) and the same with `-fa on`: 12 output files per build, md5-identical
to stock b12132 on all six cards for the shipped patch (and for every intermediate version tested). Bit-exact, not
top-1: every fused kernel repeats the stock per-element arithmetic and reduction shapes, and the one piece that
changes summation order (the Ampere Q4_K matvec) is off by default (below).

## Which piece came from where

| piece | source | gate |
|---|---|---|
| q8_1 activation cache, one quantization reused by q/k/v/gate; invalidation scan for writes over the cached row; `done` set for non-adjacent fusions | Turing (cache kept beside the context in mmvq.cuh so common.cuh, included by every .cu, is untouched); buffer allocated before capture as in Blackwell | all; batch-1 only |
| fused post-norm + residual + next pre-norm (and lone pre-norm) writing the q8_1 copy | Turing's `rms_norm_chain_f32` (explicit 8-lane shuffles), made PDL-aware and without `__restrict__` (outputs may alias inputs, as Ada noted) | all; one-row only |
| short-conv decode step as one kernel (b*x, concat, state write, conv, *c, q8_1) | Turing's matcher and kernel (fuses all conv layers, non-contiguous), with a per-graph reader index (built once per graph uid) instead of a full-graph scan per candidate, so eager mode on Pascal does not pay ~5 scans per layer per token | all |
| gated attention output (sigmoid, *gate, q8_1) | Turing | all |
| batched per-token uploads: host staging list, one H2D copy + one scatter kernel per graph compute | Ada (`ggml_cuda_upload_batch`), plus a single wait per split for host inputs in the scheduler (Blackwell's idea, Ampere's code shape) | all; `PHONON_NO_BATCHED_UPLOADS=1` off |
| CUDA graph key mixes node count and token counts, so the prompt graph does not evict the decode graph | Ada | all (CUDA graphs, Volta+) |
| vector flash-attention kernel for batch-1 decode | Turing | cc 7.x only (pre-Ampere tensor-core branch) |
| Q4_K mmvq with 16-byte loads and fused SWIGLU | Ampere | compiled in, off by default (see below); `PHONON_Q4K_MMVQ=1` on, `PHONON_NO_Q4K_MMVQ=1` off |
| mmvq retune (nwarps 2, 2 rows, vdr 4) + persistent q4_K kernel with in-kernel quantization | Pascal | cc 6.x only; on a q8_1 cache hit the kernel stages the cached row instead of quantizing |
| 2 rows per block for Q4_K/Q5_K, PDL early trigger + L2 prefetch of the block's own weight rows | Blackwell | sm_120 builds only (`MMVQ_PARAMETERS_BLACKWELL` table, `__CUDA_ARCH__ == 1200`) |
| `PHONON_PHASE_TIMING=1` host-phase timers (llama-context, scheduler, CUDA graph compute) | Ampere | off unless set |

Pascal got all the fusions it lacked: 1080 Ti 171.3 -> 155.7 ms on top of its own matvec work.

## Left gated off, and why

- Ampere's Q4_K matvec is off by default on every GPU. On the local 420 W 3090 with the merged fusions it measured about 1%
  slower than the stock matvec (merged v1 with it on 65.67 / 65.86 / 65.65 / 65.86 ms, with `PHONON_NO_Q4K_MMVQ=1`
  65.11 / 65.22 ms, alternated), and it is the only piece that is not bit-exact. The Ampere track's +1.7% came from a
  260 W rented 3090 without the other fusions. Build with `-DPHONON_Q4K_MMVQ_DEFAULT_ON` (CMake `-DCMAKE_CUDA_FLAGS=...`)
  to restore the Ampere default, or set `PHONON_Q4K_MMVQ=1`. Worth re-measuring on a power-capped 3090.
- Blackwell's prefetch and 2-rows table: sm_120 only. The prefetch needs PDL (sm_90+) and was only measured on sm_120;
  Ada's own track measured 2 rows per block as a loss on sm_89, Turing's as no gain.
- Not carried: the Ampere patch's q8_1 emission inside the stock `rms_norm_f32` used `warp_reduce_sum<8>(int)`, which on
  sm_80+ is `__reduce_add_sync` over all 32 lanes, so the q8_1 block sums (ds.y) it wrote were wrong. Q4_K/Q5_K/Q6_K
  dot products only read ds.x, so its outputs were still correct for this model. The merged patch uses Turing's kernel
  with explicit 8-lane shuffles instead.

## Versions tried on the way (all gate-identical)

| version | difference | result |
|---|---|---|
| v1 | Ampere-style uploads (async copy per input after one wait, one sync); Q4_K matvec on for cc 8.0-8.7 | 5090 37.61, 1080 Ti 155.9, 3090 65.75 (all beat own); 4090 55.48 / 55.97 vs Ada 54.98 / 54.95, 1660 Ti 191.3 / 193.4 vs Turing 187.5 / 193.0 (lost) |
| v2 | v1 with the Q4_K matvec default off | 3090 65.11-65.22; 4090 still ~0.7% behind Ada over 5 pg-only rounds (55.48-55.75 vs 54.97-55.76); with uploads off 57.0 |
| v3 (shipped) | Ada's staged uploads replace the Ampere-style async copies | 4090 54.30-55.16 vs Ada 54.87-55.45 (5 rounds); beats or ties own on every card |

The 4090 loss was the upload path: Ada's staging (one copy + one scatter kernel, no per-input API call or sync) costs
less host time per token than 13 `cudaMemcpyAsync` + a sync.

## Build (one sm per build)

```
git clone --depth 1 --branch b12132 https://github.com/ggml-org/llama.cpp && cd llama.cpp
git apply /path/to/merged/b12132.patch
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<SM> -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target llama-bench llama-completion llama-simple llama-server
```

`<SM>`: 120 (RTX 50), 89 (RTX 40), 86 (RTX 30), 75 (RTX 20 / GTX 16), 61 (GTX 10). Measured builds used
`-DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF`, CUDA 12.8.1 on the rented boxes (nvidia/cuda:12.8.1-devel-ubuntu24.04)
and CUDA 13.3 on the local 3090 box. CUDA 13 no longer builds sm_61, so Pascal builds need CUDA 12.x. Scripts (build_variant.sh,
gate.sh, bench.sh, run_box.sh, ab.sh) and raw logs were kept locally, not in the repository.

## Roofline rows

Each box also ran a roofline probe (read bandwidth, int8 / fp16 / fp32 peaks, power cap,
profiler check); 12 rows (stock and merged per card) are in `kernels/roofline/nvidia.tsv`. nsys works on the local 3090 box only
(not installed in the vast.ai image); ncu has no counters anywhere (ERR_NVGPUCTRPERM), the local 3090 box included.

## Cost and instances (vast.ai)

Credit 7.37 -> 6.52 (read at 7:14 pm), about US$0.85 including the re-rents for v3 (billing can lag a few minutes).
Instances created and destroyed (`vastai show instances`: 0 at 7:14 pm):
59487777 (5090), 59487778 (4090), 59487779 (2080 Ti), 59487782 (1080 Ti), 59489643 (1660 Ti),
59492114 (5090, v3 re-run), 59492118 (2080 Ti, v3 re-run), 59492120 (1080 Ti, v3 re-run).
The 3090 runs used a local workstation.
