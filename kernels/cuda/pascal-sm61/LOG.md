# Pascal sm_61 (GTX 1080 Ti) refiner kernel track

Patch: `b12132.patch` against llama.cpp tag b12132 (touches only `ggml/src/ggml-cuda/mmvq.cu`).
Box: vast.ai instance 59475488, GTX 1080 Ti 11 GB, driver 580.173.02, CUDA 12.8.1 image, Xeon E5-2630 v4 (20 threads).
The host caps board power at 130 W (stock 1080 Ti is 250 W); measured sustained DRAM read ceiling with a plain
uint4 streaming kernel: 350-364 GB/s (2 MB in-L2: 520 GB/s). Session 4:02 pm to 5:54 pm Alberta time, 2026-11-06.

## Does stock b12132 build and run on sm_61?

Yes. `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61` builds clean in 785 s on this CPU; the only diagnostics are 147
copies of nvcc's "offline compilation for architectures prior to sm_75 will be removed in a future release"
deprecation warning. Output is correct: GPU greedy output vs the CPU backend on the same binary is identical for
prompts 2, 3, 4 and identical for the first 18 words of prompt 1 (then a near-tie flips; normal reduction-order
difference). No fix needed.

## Final numbers (same box, interleaved stock/patched, two rounds, llama-bench -r 10)

| metric (fa auto) | stock b12132 | patched | change |
|---|---|---|---|
| ms/dictation (pg16,32) | 232.85 / 236.54 | 171.64 / 172.78 | -26% (1.37x) |
| pg16,32 t/s | 206.1 / 202.9 | 279.7 / 277.8 | |
| tg32 t/s | 145.5 / 143.0 | 200.4 / 199.5 | +38% |
| pp16 t/s | 1240 / 1243 | 1297 / 1280 | noise (sd 30-95) |
| pp64 t/s | 2491 / 2459 | 2516 / 2502 | noise |

With `-fa 1`: stock 234.25 / 237.76 ms, patched 171.67 / 172.76 ms (flash attention changes nothing here: the
tile kernel is used either way on Pascal and attention is ~3% of a token).
Noise: per-run sd over -r 10 is about 1 t/s on pg (0.4%); stock drifted ~1.5% slower in round 2 (power cap).
Reference RTX 3090 stock tg32 is 449 t/s; this card is ~3x slower at batch 1.

Correctness: all four gate prompts, 64 greedy tokens with llama-simple, md5-identical to stock b12132 on the same
box (bit-exact, not just top-1), for every configuration in the hill-climb below.

## Where the time goes (nvprof, decode token, stock)

nvprof works on Pascal (ncu does not support sm_61; hardware counters are blocked in the container, so metrics are
derived from kernel durations). Stock decode token: 6.9 ms span, 6.5 ms GPU busy, 540 kernels; launch gaps only ~6%.
- q4_K matvec (mul_mat_vec_q, 1 column): 4.9 ms = 75%. Effective bandwidth only 150-230 GB/s of ~360:
  ffn gate+up (fused, 7680x2048 x2) 83 us = 212 GB/s; ffn_down 2048x7680 40 us = 222 GB/s;
  2048x2048 16 us = 147 GB/s; in_proj 6144x2048 43 us = 165 GB/s; output head 128000x2048 840 us = 175 GB/s.
- q5_K ffn_down (3 layers) 0.14 ms, quantize_q8_1 0.24 ms (130 launches), rms_norm 0.35 ms, everything else
  (shortconv chain, attention, rope, copies) ~1.2 ms, each kernel 1.5-7 us.
Diagnosis: the stock matvec is issue/load bound on Pascal, not DRAM bound. Each thread does ~12 narrow loads per
8 bytes of weights (2-byte scale loads, 4-byte quant loads, 36-byte q8_1 blocks that cannot be vector-loaded),
and with K=2048 and 4 warps per row every thread does exactly one block before a shared-memory reduction.

## Hill-climb (ms/dictation from llama-bench -pg 16,32 -r 5 unless noted; all bit-identical to stock)

| # | change | tg32 t/s | ms/dict | gain vs prev best |
|---|---|---|---|---|
| 0 | stock b12132 (-r 10) | 146.6 | 230.9 | - |
| 1 | enable CUDA graphs on Pascal (stock disables < Volta) | 146.3 | 230.4 | 0% (dropped; cudaGraphLaunch costs ~390 us of CPU per chunk on Pascal and GPU time dominates anyway) |
| 2 | Pascal mmvq table, 1 col k-quants: nwarps 2 (stock 4) | 160.5 | 211.5 | +9% |
| 3 | + rows per block 2 | 169.1 | 202.8 | +4% |
| 4 | + q4_K vec_dot with 4 ints/thread (stock 2), 16-byte quant loads | 195.7 | 175.5 | +14% |
| 5 | new kernel `mul_mat_vec_q4_K_pascal` (y staged in smem, 3 x 16-byte loads per lane per 32-byte group) | 189.8 | 181.4 | -3% (per-call parity with #4) |
| 6 | + persistent grid, best NW=16 BPS=4 | 197.5 | 174.7 | +0.5% |
| 7 | + quantize f32 activations inside the matvec (no quantize_q8_1 launch) | 193.6 | 177.3 | -1.5% |
| 8 | + load first weights before staging y, software-pipeline next weights; sweep NW/BPS -> NW=16, 1 block/SM | 199.8 | 171.1 | +2.0% vs #6 |
| final | #8, -r 10, two interleaved rounds | 200.4 / 199.5 | 171.6 / 172.8 | -26% vs stock |

Other configs tried (ms/dict): #2-#4 grid of nwarps {1,2,4} x rows {1,2,4} x ints/thread {2,4,8}: 8 ints/thread
peaks at 188.7 t/s (no better than 4). New kernel NW/R/BPS: R=2 and more blocks per SM always lost
(NW=8 BPS=8: 191.9 ms; NW=8 R=2: 188.2 ms; NW=32 BPS=1: 174.7 ms; NW=16 BPS=2: 172.1; BPS=3: 177.1).

Per-call after the change (nvprof): gate+up 59.7 us (296 GB/s, ~82% of the measured ceiling), and no
quantize_q8_1 launches for q4_K. The tuned stock-structure kernel (#4) reaches the same per-call times (gate+up
59.9 us, output head 422 us = 350 GB/s); the new kernel's net win is the removed quantize launches. The matvecs are
close to the DRAM ceiling, so further matvec work has little left.

Stop reason: ideas 5-8 each gained under 3% over the previous best (stop rule), and the matvec is near the
bandwidth ceiling. What remains is ~1.5 ms/token (30%) of small kernels at 1.5-7 us each.

## What did not work
- CUDA graphs on Pascal: neutral with the final kernels too (graphs off 171.11 / 172.02 ms vs on 171.82 /
  172.71 ms); left disabled as upstream has it.
- 8 ints/thread in the stock-structure kernel, R=2 rows per warp, many small blocks: more registers or more y
  staging per byte of weights, slower.
- Software pipelining alone (#8 at NW=16 BPS=4) was slower (191.8 t/s); it only pays with one 512-thread block per
  SM, where y is quantized once per SM.

## Next ideas not done (projections, unmeasured)
- Fuse the shortconv decode chain (mul b*x, concat with state, state copy, get_rows, ssm_conv, mul c*y: 6 kernels,
  ~25 us per conv layer x 13 layers ~ 0.33 ms/token) into one kernel: projected ~5% of a dictation. Not
  Pascal-specific.
- rms_norm + quantize, q5_K Pascal kernel for the 3 q5_K ffn_down layers (0.17 ms/token): projected 1-3% each.

## Architecture specificity
The vdr=4 / nwarps 2 / rows 2 table change and the new kernel are gated to Pascal (`MMVQ_PARAMETERS_PASCAL`,
cc 600-699); other archs are untouched. The diagnosis (issue-bound narrow loads, y re-read per row) is
Pascal-shaped: no tensor cores, 130 W cap, ~360 GB/s. Maxwell (sm_52, no dp4a) cannot use it. Volta/Turing have
their own tuned table already; the kernel would probably help sm_70/75 too but was not measured. Expect the
same direction, smaller magnitude, on a GTX 1080 / 1070 (sm_61, same ISA, lower bandwidth). P100 (sm_60) is
covered by the cc range but untested (HBM2, much more bandwidth, so the load-issue limit matters even more).

## Build
```
git clone --depth 1 --branch b12132 https://github.com/ggml-org/llama.cpp && cd llama.cpp
git apply /path/to/pascal-sm61/b12132.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-bench llama-simple llama-cli
```
Tunables (compile-time, defaults are the tuned values): MMVQ_PASCAL_KERNEL=1, MMVQ_PASCAL_NW=16, MMVQ_PASCAL_R=1,
MMVQ_PASCAL_BPS=1, MMVQ_PASCAL_FUSE_Q8=1, MMVQ_PASCAL_NWARPS=2, MMVQ_PASCAL_RPB=2, MMVQ_PASCAL_Q4K_VDR=4.

## Cost
One instance (59475488, GTX 1080 Ti, US$0.061/h) from 4:02 pm to 5:54 pm, about US$0.12 of rent (rate x time,
not read from a vast.ai invoice). Destroyed at 5:54 pm.
