# Phonon refiner on Ampere consumer (RTX 3090, sm_86): kernel hill-climb log

Date: 2026-11-06, 4:00 pm to 5:05 pm Alberta time.
Patch: `b12132.patch` (applies cleanly to llama.cpp tag b12132 with `git apply`; 9 files, +683/-19).

## Hardware

- Hill-climb box: vast.ai instance 59475539 (offer 44375061), RTX 3090 24 GB, driver 580.173.02, power limit 260 W,
  host CPU AMD EPYC 7K62 (2.6 GHz), CUDA 12.8 (nvidia/cuda:12.8.1-devel-ubuntu24.04). Destroyed at 5:05 pm.
  Cost: about 1.04 h x US$0.196/h = about US$0.21.
- Second data point (a local workstation, stock only, from an earlier baseline run; the hill-climb itself ran on
  the rented box): RTX 3090, 420 W limit, Ryzen 9 9950X3D, stock b12132, `-r 5`:
  pp16 3640 t/s, pp64 10262, tg32 449 (fa on: pp16 3871, pp64 10278, tg32 446). pg16,32 was not measured there.
  The vast box is slower than the workstation at stock (tg32 about 390 against 449): slower host CPU and a 260 W cap.

## Final numbers (vast 3090, `llama-bench -m refiner.gguf -ngl 99 -pg 16,32 -p 16,64 -n 32 -r 10`)

Stock and patched were interleaved, two rounds each. ms/dictation = 48 / (pg16,32 t/s) x 1000; the +- is the
llama-bench standard deviation over 10 reps propagated to ms.

| fa      | build   | ms/dict r1     | ms/dict r2     | tg32 t/s r1/r2 | pp16 t/s r1/r2 | pp64 t/s r1/r2 |
|---------|---------|----------------|----------------|----------------|----------------|----------------|
| default | stock   | 86.02 +- 0.51  | 86.29 +- 0.64  | 390.0 / 388.5  | 3814 / 3835    | 9661 / 9627    |
| default | patched | 73.73 +- 0.59  | 73.98 +- 0.46  | 462.1 / 459.8  | 3817 / 3818    | 9641 / 9618    |
| 0       | stock   | 87.63 +- 0.50  | 88.21 +- 0.62  | 381.8 / 378.4  | 3840 / 3833    | 9351 / 9300    |
| 0       | patched | 75.64 +- 0.61  | 75.90 +- 0.55  | 447.5 / 445.0  | 3881 / 3857    | 9416 / 9337    |
| 1       | stock   | 86.18 +- 0.59  | 86.69 +- 0.65  | 387.9 / 386.9  | 3852 / 3799    | 9636 / 9591    |
| 1       | patched | 73.89 +- 0.58  | 74.16 +- 0.58  | 460.4 / 458.7  | 3821 / 3844    | 9630 / 9671    |

Result: ms/dictation 86.2 -> 73.9 (-14.3%) with default/fa 1, 87.9 -> 75.8 (-13.8%) with fa 0. tg32 +18%.
Prefill (pp16, pp64) is unchanged within noise (pp16 noise is about +-17% on its own; every change targets
single-token decode).

## Correctness

- `llama-simple -n 64` (greedy) on the four brief prompts: output byte-identical to stock b12132 after every step
  (4/4 at s1..s9).
- `llama-completion -n 64 --temp 0 --top-k 1`, with `-fa on` and `-fa off`: identical to stock on all four prompts.
  Prompts 3 and 4 hit end-of-generation early (74 and 79 bytes of output), so they check fewer steps than 1 and 2.
- Bit-exactness: the norm/q8, shortconv and norm+add+norm fusions repeat the stock per-element arithmetic in the
  same order (same block_reduce, same float4 mapping, same quantize_q8_1 math), so they are bit-identical by
  construction. The new Q4_K matvec sums the same dp4a partials in a different cross-thread order, so logits are
  not bit-identical to stock; greedy tokens (top-1 at every step) still match on all prompts above. Logits were not
  dumped and diffed.

## Hill-climb (vast 3090, default fa, pg16,32 -r 10; step N includes all earlier kept steps)

| step | change                                                              | ms/dict        | vs prev | tokens |
|------|---------------------------------------------------------------------|----------------|---------|--------|
| s0   | stock b12132 (same-session reference)                               | 85.97 +- 0.57  |         | ref    |
| s1   | q8_1 activation cache: matvecs on the same row quantize once        | 84.69 +- 0.57  | -1.5%   | 4/4    |
| s2   | rms_norm*w kernel also emits the q8_1 row (quantize kernel removed) | 84.15 +- 0.59  | -0.6%   | 4/4    |
| s3   | new Q4_K x q8_1 batch-1 matvec, 16-byte weight loads, fused SWIGLU  | 82.73 +- 0.41  | -1.7%   | 4/4    |
| s4   | fused LFM shortconv step (MUL,CONCAT,CPY,SSM_CONV,MUL + q8_1 out)   | 79.02 +- 0.37  | -4.5%   | 4/4    |
| s5   | fused post-norm + residual add + next pre-norm (+ q8_1 out)         | 74.88 +- 0.45  | -5.2%   | 4/4    |
| s6   | sched: host inputs uploaded async behind one wait, one sync         | 73.83 +- 0.54  | -1.4%   | 4/4    |
| s7a  | matvec: 2 super-block groups of loads in flight per trip            | rejected       | worse   | n/a    |
| s7b  | matvec: 2 rows per warp sharing activation loads                    | rejected       | worse   | n/a    |
| s9   | final = s6 kept code, debug dump removed                            | 73.73 / 73.98  |         | 4/4    |

s7a/s7b were judged by nsys kernel times (no bench): s7a gate/up 23.5 -> 24.5 us, output 183 -> 187 us;
s7b output 183 -> 193 us. Stop rule: s6 (1.4%) and s7 (no gain) were two consecutive ideas under 3%.

## Where the time goes (one decode token)

nsys 2026.5.2, `--cuda-graph-trace=node` for per-kernel times, `=graph` for gaps (tracing inflates API times).

| per decode token                       | stock   | final   |
|----------------------------------------|---------|---------|
| kernels                                | 526     | 342     |
| kernel time, us (node trace)           | 2377    | 1983    |
| graph GPU duration, us (graph trace)   | 2271    | 1956    |
| GPU idle between tokens, us (median)   | 313     | 187     |
| token period, us                       | 2583    | 2145    |
| Q4_K/Q5_K matvec kernel time, us       | 1547    | 1493    |
| everything else on GPU, us             | 830     | 490     |

Final per-token kernel mix: Q4_K matvec 123 calls 1433 us, rms_norm+add+rms_norm 47 calls 123 us, flash attention
11 calls 86 us (+11 stream-k fixups 17 us), quantize_q8_1 37 calls 61 us (down-proj and o-proj inputs), stock
Q5_K matvec 7 calls 61 us, q/k rms_norm+rope 22 calls 51 us, KV set_rows 22 calls 33 us, recurrent-state
get_rows 15 calls 29 us, fused shortconv 11 calls 27 us.

Matvec bandwidth (3090 peak 936 GB/s): stock reaches about 700 GB/s on the big matrices (gate+up 17.7 MB in 25.4 us,
output 147 MB in 207 us). The new kernel: gate+up 23.5 us (752 GB/s), output 183 us (805 GB/s), in_proj 7.1 MB
12.5 -> 10.8 us, 2048x2048 2.36 MB 5.7 -> 5.1 us. Small matrices stay launch/latency bound (k-proj 0.59 MB about
3 us; it got slightly slower than stock's 2.75 us, the 16-row attention gate went 1.76 -> 2.4 us).

## Host (CPU-side) numbers, for the Windows-vs-Linux comparison

Measured with `PHONON_PHASE_TIMING=1` (in the patch; prints at exit), `llama-bench -p 0 -n 128 -r 3`, medians over
385 single-token decodes, on the vast EPYC 7K62 host. These are wall-clock host phases without nsys.

| phase per decode token (us)                                  | stock-equivalent | final |
|--------------------------------------------------------------|------------------|-------|
| decode prep (ubatch split, memory slot)                      | 15               | 15    |
| graph reuse check (graph is reused every token, no rebuild)  | 2                | 2     |
| set_inputs (host writes into pinned input tensors)           | 3                | 3     |
| graph_compute total                                          | 108              | 61    |
|   of which sched input uploads (13 inputs, 2 splits)         | 94               | 48    |
|   of which CUDA backend compute (replay)                     | 14               | 13    |
|     graph property checks                                    | 4.4              | 4.1   |
|     cudaGraphLaunch (2 chunks, 996 ggml nodes per graph)     | 7.5              | 7.2   |
| decode post (logits D2H enqueue)                             | 5                | 5     |
| host work before the GPU can start the next token           | about 133        | about 86 |

- The graph is built once and reused (`llama_graph_reuse` hits every token); CPU-side graph build per token is
  about 2 us, not a factor. CUDA graphs replay every token; the ggml graph is split into two CUDA graphs of up to
  512 nodes (`GGML_CUDA_GRAPH_CHUNK_SIZE`), launched back to back with a 3 us gap.
- Stock uploads each of the 13 host inputs with a blocking event wait + blocking cudaMemcpy + stream sync (about
  7 us each on this CPU). The patch issues them as async copies behind one wait and one sync (48 us). The remaining
  48 us is 13 cudaMemcpyAsync calls plus one sync.
- The time between "last kernel of token N ends" and "first kernel of token N+1 starts" was 313 us (stock) and
  187 us (final) under nsys. The host phases above explain about 86-133 us of it; the rest is not separately
  measured. Likely parts (unmeasured): the 512 KB f32 logits copy (128000 vocab) device-to-host after each token,
  stream-sync wakeup, and nsys API overhead.
- Kernel-to-kernel gaps inside a CUDA graph: about 21 us per token over 341 gaps (median), so launch gaps inside
  the graph are not the problem; per-kernel fixed cost (ramp/drain, about 1.5-2.5 us per small kernel) is.

## What changed (files in the patch)

- `ggml-cuda/common.cuh`, `ggml-cuda.cu`: per-context single-row q8_1 cache (64 KB persistent buffer). Matvecs on a
  row already quantized skip `quantize_q8_1`; producers (norms, shortconv) write the q8_1 row directly. The cache is
  dropped at graph start and whenever any node's output overlaps the cached row. Only for batch-1 rows (ne11 = 1,
  no ids), so prompt processing (MMQ) is untouched.
- `mmvq.cu`: `phonon_mmvq_q4k`, batch-1 Q4_K matvec, one warp per row, 8 lanes per 144-byte super-block with
  16-byte weight loads, optional fused gate (SWIGLU) for ffn gate/up. Falls back to stock for anything else.
- `norm.cu`: rms_norm*w optionally emits q8_1; new `rms_norm_add_rms_norm_f32` fuses
  RMS_NORM,MUL,ADD,RMS_NORM,MUL (post-norm, residual, next pre-norm) for one row in one block.
- `ssm-conv.cu`: `shortconv_step_f32`, fused single-token LFM shortconv step; 11 of 13 conv layers match (layer 0
  and one other are interleaved with other nodes by the graph optimizer and stay unfused, by design: only
  contiguous patterns fuse, so no tensor is written earlier than its allocation slot).
- `ggml-backend.cpp`: batched async upload of host-resident split inputs (one wait, one sync).
- `llama-context.cpp`, `ggml-backend.cpp`, `ggml-cuda.cu`: `PHONON_PHASE_TIMING` instrumentation (off by default).

Kill switches (env): `PHONON_NO_Q4K_MMVQ`, `PHONON_NO_SHORTCONV_FUSION`, `PHONON_NO_NORM2_FUSION`,
`PHONON_NO_BATCHED_UPLOADS`. The q8_1 cache and the norm-emits-q8_1 path have no switch.

## Build

```
git clone --depth 1 --branch b12132 https://github.com/ggml-org/llama.cpp && cd llama.cpp
git apply /path/to/b12132.patch
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-bench llama-simple llama-completion llama-server
```

## Does it carry to other architectures?

- The fusions (norm+q8, norm+add+norm, shortconv step, q8 cache) cut 184 of 526 kernels per token and the upload
  batching cuts host time; none of that is sm_86-specific and should carry to Ada and Blackwell, and the host part
  helps most on slow CPUs. Projection, not measured.
- The Q4_K matvec was tuned only on the 3090 (dp4a, 4 warps per CTA, `unroll 2`). Two caveats for newer parts: it is
  launched without programmatic dependent launch, which stock enables on Hopper/Blackwell, and its block shape was
  not swept there. Re-measure it per architecture; `PHONON_NO_Q4K_MMVQ=1` turns it off.

## Not done (estimates are projections from the kernel profile, not measurements)

- Attention gate: fuse sigmoid + mul + quantize before o-proj (22 kernels per token, about 1.5%).
- Fold the recurrent-state get_rows into the shortconv kernel (11 kernels, about 1%).
- Emit q8_1 from the gate/up kernel for the down-proj input (24 quantize kernels, about 2%; needs 32 consecutive
  rows per CTA, which costs occupancy).
- One launch for q/k/gate matvecs (same input, all Q4_K; about 2%).
- Greedy decode only needs the argmax, but every token copies 512 KB of logits to the host; unmeasured.
- A Q5_K version of the matvec (7 calls, 61 us per token).
