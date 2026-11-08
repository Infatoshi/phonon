# Turing (sm_75) refiner kernels, llama.cpp b12132

Target: Phonon refiner (phonon-refiner-1.7b-Q4_K_S-e4.gguf, LFM3 hybrid, 24 layers = 13 short-conv + 11 attention,
n_embd 2048, n_ff 7680, vocab 128000, tied Q4_K output head), batch 1 decode.
Metric: `llama-bench -m refiner.gguf -ngl 99 -pg 16,32 -r 10`, ms/dictation = 48 / (pg16+32 t/s) * 1000.
Hill-climb box: vast.ai RTX 2080 Ti (TU102, 68 SMs, 616 GB/s, driver 570.190, CUDA 12.8), host Xeon E5-2682 v4 shared
with other tenants (load average 40-60 from other containers the whole session), so every number below is a same-session
interleaved stock/candidate pair. The GPU sits at its 250 W power cap during decode, which moves kernel times about 8%
between runs; compare within a row, not across rows.

## Result (2080 Ti, 3 interleaved pairs, -r 10 each, spread = run-to-run range)

| config            | ms/dictation        | pg16+32 t/s | tg32 t/s      | pp16 t/s (noisy) | pp64 t/s (noisy) |
|-------------------|---------------------|-------------|---------------|------------------|------------------|
| stock b12132      | 133.6 (133.1-134.2) | 359         | 244 (240-249) | 1768-2343        | 3546-5589        |
| patched           | 114.5 (114.1-114.8) | 419         | 294 (293-295) | 1955-2369        | 5480-5739        |
| stock, -fa 1      | 134.3 (134.1-134.5) | 357         | 248.5 (245-251) | 2341-2347      | 5528-5584        |
| patched, -fa 1    | 114.7 (114.2-115.1) | 418         | 293 (292-294) | 2028-2259        | 4882-5477        |

Decode is 95% of a dictation; prefill (16 tokens) is untouched by the patch apart from the input-upload change, and pp
numbers on this box swing +-20% between runs (CPU contention), so no pp claim is made.
Within-run spread over -r 10: +-1 to +-8 t/s on pg16+32 for both builds. Two earlier patched -fa 1 runs hit a host
CPU burst (177 and 182 ms, stddev 17-24 t/s); the three -fa 1 pairs in the table were rerun right after and are clean.

Correctness: greedy output (llama-completion, temp 0, top-k 1, 64 tokens) byte-identical to stock b12132 on the
same box for all four brief prompts, with default flags and with `-fa on`, at every step of the climb. Nothing changes
the arithmetic: the fused kernels replay the stock per-element operations and reduction orders (same block sizes,
same q8_1 formula), and the FA vec kernel happened to produce identical tokens too.

## Where the time went (stock, nsys on the 2080 Ti)

Per decode token (tg, nsys node-level trace): 526 kernels, 3.15 ms kernel busy, 3.32 ms first-to-last kernel span;
real token period 3.86 ms (tg32 259 t/s), so roughly 0.5 ms per token is outside the CUDA graph.

| kernel                                   | per token      | note                                   |
|------------------------------------------|----------------|----------------------------------------|
| mmvq Q4_K, 99 launches                   | 1220 us        | incl. 128000x2048 output head 276 us   |
| mmvq Q4_K fused gate/up+swiglu, 24       | 885 us         | 17.7 MB each, ~480 GB/s (78% of peak)  |
| mmvq Q5_K, 7                             | 86 us          |                                        |
| quantize_q8_1, 130                       | 239 us         | one per matvec, 4 per attention layer on the same input |
| rms_norm (+mul, +add), 97                | 271 us         | one 512-thread block each, ~2.8 us     |
| flash attn mma + stream-k fixup, 11+11   | 145 us         |                                        |
| bin_bcast mul 37, set_rows 22, get_rows 15, cpy 13, concat 13, ssm_conv 13, sigmoid 11, rope-norm 22 | ~330 us | 1.5-3 us each, launch-latency bound |

Matvecs are ~2.2 ms of a ~1.6 ms bandwidth floor (988 MB per token at 616 GB/s); the other ~1 ms of kernel time is ~400
tiny kernels. Outside the graph (graph-level trace): per token ~13 user inputs are uploaded, each with an event wait plus a
blocking memcpy + stream sync (~18 us each on this CPU), then the 512 KB logits copy (~40 us) and two graph-chunk launches.

## Hill-climb log (2080 Ti; each row is candidate vs stock measured back to back in the same session)

| step | change | ms/dict candidate | ms/dict stock same session | kernels/token | correct | kept |
|------|--------|-------------------|----------------------------|---------------|---------|------|
| 0 | stock b12132 | | 127.7-134.5 over the session | 526 | ref | |
| 1 | q8_1 activation cache + fused decode norm chain (post-norm+residual+next pre-norm in one kernel that also writes q8_1) | 121.5-122.2 | 129.8-132.1 | 397 | identical | yes |
| 2 | Turing mmvq nwarps 2 -> 4 for Q4_K/Q5_K | 120.3-120.4 | (vs step 1: 121.5-122.2) | 397 | identical | no: mmvq kernel time unchanged (1326 vs 1316 us/token) |
| 3 | flash-attn vec kernel for batch-1 decode on Turing (was mma + stream-k fixup) | 119.2-119.8 | 127.7-132.7 | 397 | identical | yes: FA 14.7 -> 8.7 us/layer |
| 4 | fused short-conv decode (b*x, state concat, state write, conv, *c, q8_1 in one kernel) + fused attention output gate (sigmoid, *gate, q8_1) | 112.4-116.0 | 132.7-133.3 | 320 | identical | yes |
| 5 | Turing mmvq 2 rows per block for Q4_K/Q5_K | 113.4-117.3 | 128.6-129.2 | 320 | identical | no: no kernel-time gain |
| 6 | scheduler: upload user inputs with one wait + async copies + one sync instead of wait + blocking copy per input | 112.5-118.4 | 130.2-132.3 | 320 | identical | yes: idle between graphs 949 -> 857 us/token (-2.6%) |

Stop: steps 5 and 6 each gained under 3% (two in a row), per the brief.

Final per-token decode profile (step 4/6 build): 320 kernels, 2.68 ms kernel busy (nsys node trace), 2.79 ms span;
graph-level trace token period 3.50 ms vs 4.18 ms stock under the same tracer.

## What the patch does (b12132.patch, 9 files, ~650 lines)

- `mmvq.cuh/.cu`: `ggml_cuda_q8_cache`, a single-entry per-context cache of the q8_1 copy of a batch-1 activation.
  `ggml_cuda_mul_mat_vec_q` reuses it when src1 matches (q, k, v and gate all read the same normed input), otherwise
  quantizes into it. The buffer is allocated once outside stream capture and never moved, so captured graphs stay valid.
- `ggml-cuda.cu`: the evaluator enables the cache only for single-stream evaluation, clears it per graph, and drops the
  entry when any later node writes over its source range. New decode fusions: `[RMS_NORM, MUL, ADD, RMS_NORM, MUL]`
  and lone `[RMS_NORM, MUL]` on one row -> `rms_norm_chain_f32` (writes both fp32 outputs plus q8_1); the LFM short
  conv chain -> `lfm_shortconv_decode_f32`; the gated attention output -> `attn_gate_decode_f32`. The last two fuse
  non-adjacent nodes, so fused-away nodes go into a `done` set; before fusing, a hazard check rejects the fusion if
  any node that still runs in between reads or writes the early-written ranges or writes what the fused kernel reads.
- `norm.cu`, `ssm-conv.cu`: the three kernels. They copy the stock per-element arithmetic and reduction shapes so
  outputs are bit-identical (same 512-thread block reduce, same `amax/127`, `roundf`, int sum for q8_1).
- `fattn.cu`: batch-1 f16 KV decode on pre-Ampere NVIDIA uses the vector kernel instead of mma + fixup.
- `ggml-backend.cpp`: batched user-input upload described in step 6 (backend-agnostic; only taken when the split
  backend has `set_tensor_async`).

Everything only triggers for one-row (batch-1) shapes, so prefill and other models keep the stock paths.

## Did not work / not tried

- Turing mmvq launch shape (nwarps 4, 2 rows per block): no kernel-time change; the Q4_K matvec stays at ~78-87% of
  bandwidth. Getting the last ~0.5 ms needs a repacked Q4_K layout with 128-bit loads, which changes the dot-product
  summation order (top-1 agreement instead of bit-exact) and did not fit the time box.
- `GGML_CUDA_GRAPH_CHUNK_SIZE=0` (one graph instead of two chunks): measured while a build was running, slower and
  inconclusive; not retested.
- Not done: removing the 24 remaining down-projection quantizes (needs a q8_1 epilogue across 32 rows of the
  gate/up matvec), fusing k/v set_rows with rope (a V matmul sits between them in the node order), the 40 us logits copy.

## 1660 Ti check (TU116, sm_75, no tensor cores)

vast.ai GTX 1660 Ti (6 GB, 120 W cap, driver 580.82.09), same sm_75-only build of the same patch, same scripts.
Greedy outputs byte-identical to stock on that box for all four prompts, default flags and `-fa on`.

| config (1660 Ti)                    | ms/dictation        | pg16+32 t/s | tg32 t/s | pp16 t/s | pp64 t/s |
|-------------------------------------|---------------------|-------------|----------|----------|----------|
| stock b12132 (3 runs)               | 207.5 (206.1-209.6) | 231         | 178.6    | 590      | 725      |
| patched (3 runs)                    | 188.9 (187.2-189.7) | 254         | 199.7    | 589      | 723      |
| patched minus the FA-vec change (3) | 192.8 (190.7-194.3) | 249         | 194.2    | 590      | 722      |
| stock, -fa 1 (2 runs)               | 207.3 (205.8-208.7) | 232         | 177.9    | 587      | 719      |
| patched, -fa 1 (2 runs)             | 190.4 (189.8-191.0) | 252         | 197.3    | 586      | 720      |

The 2080 Ti choices also help the 1660 Ti: -9.0% ms/dictation (vs -14.3% on the 2080 Ti), and the one arch-flavoured
choice, FA vec for batch-1 decode, is itself a win there (192.8 -> 188.9 ms), so no tensor-core runtime switch is
needed. The smaller relative gain is expected: the 1660 Ti has 288 GB/s, so matvec time dominates a decode token more.
Prefill is the 1660 Ti's other cost: pp16 runs at ~590 t/s (27 ms of the 207 ms dictation, vs ~7 ms on the 2080 Ti)
because MMQ there has no tensor cores; the patch does not touch prefill. That is the next lever on TU116.

## Build

```
git clone --depth 1 --branch b12132 https://github.com/ggml-org/llama.cpp && cd llama.cpp
git apply /path/to/b12132.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target llama-bench llama-completion
build/bin/llama-bench -m refiner.gguf -ngl 99 -pg 16,32 -r 10
```

## Portability

The fusions (norm chain, short-conv, attention gate, q8_1 reuse) and the input-upload change are launch-count and
CPU-overhead wins with no architecture dependence; they should carry to Pascal through Blackwell (Hopper+ PDL already
hides some launch latency, so the gain there is likely smaller). Only the FA-vec rule is gated to pre-Ampere NVIDIA
(stock already picks vec on Ada+). Patch was built only with `-DCMAKE_CUDA_ARCHITECTURES=75`.

## Cost and instances (2026-11-06, Alberta time)

- 59475481 RTX 2080 Ti (offer 58567248): host rejected the ssh key for 10+ min, destroyed.
- 59475920 and 59476864 RTX 2080 Ti: stuck in "loading"/stopped (host could not start them), destroyed.
- 59477635 RTX 2080 Ti (offer 54058112, US$0.083/h): hill-climb box, about 4:23 to 5:22 pm, destroyed.
- 59482817 GTX 1660 Ti (offer 46831471, US$0.081/h): check box, about 5:15 to 5:43 pm, destroyed.

Estimated spend about US$0.15 (rent rates times wall time; the account credit is shared with other agents, so the
balance delta is not attributable).
