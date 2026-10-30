# ml: the Phonon refiner pipeline

Lives in the app repo under `ml/` (code only). Runs on a Linux GPU box: it was built on 4x RTX
PRO 6000 Blackwell (96 GB, sm_120, CUDA 13), a 64-core CPU and 60 GB of RAM. Data, adapters,
audio and run outputs live under `/data` on that box and never enter git. Nothing here trains or
runs heavy inference on a Mac; the Mac only runs the shipped int4 model and the recorder script.

The research scripts hardcode their root as `/home/user/phonon`, which is this `ml/` directory
checked out on the GPU box (the original path had a username in it). Run from a checkout there,
or edit the constant. `/data/...` paths are real and listed under Data below.

## What this is

Phonon's text path is Parakeet TDT 0.6B v2 (ASR), then a small language model that rewrites the
raw transcript into what the speaker meant. This tree builds that second model. The refiner is
LFM2.5-1.2B-Instruct with a LoRA adapter; it sees the raw transcript plus a short vocabulary list
retrieved from the user's own code, and it is trained on acoustic synthetic pairs (generated
sentence, Kokoro TTS, Parakeet) plus the user's real dictation pairs. `SPEC.md` has the design and
the eval roadmap. The full experiment log is a private `DEVLOG.md` kept with the research tree on the
GPU box (ignored here, like the root one); the numbers that decide anything are in these two files.

## State (2026-10-23)

Earlier distillation experiments are kept out of this repository.

## State (2026-09-28)

**Ship recipe.** LFM2.5-1.2B, LoRA r16, trained on the 14.5k-term acoustic corpus with the real
pairs repeated x12 (`restr_real12` recipe); retrieval v2, two-stage, with the adaptive list budget
`min(80, max(30, 3 x words))` over a ranked, capped per-user symbol index; the word-count guard on;
int4 as MLX `custom_attn6emb8` (746 MB) or GGUF IQ4_XS with the mixed imatrix (663 MB). Users whose
pairs cannot enter the corpus get the generic adapter plus a short personal second stage.
`bigrun_xl_r16` (pool 2 continue-train) is the wide-vocabulary variant.

**Reference numbers** from phonon-bench (guard on, retrieved lists, same fixtures):

| model | real_580 fair WER | real damage | real ENTITY/1k | term_new hit | term_pool2 hit | term_pool3 hit |
|---|---:|---:|---:|---:|---:|---:|
| raw Parakeet TDT v2 | 0.1838 | - | 22.6 | 0.413 | - | - |
| bigrun_mid_r16 | 0.1134 | 0.057 | 28.0 | 0.881 | 0.810 | 0.873 |
| restr_real12 | 0.1078 | 0.055 | 20.3 | 0.863 | 0.791 | 0.851 |

Noise floors: fair WER +/- 0.002 on real_580, term hit +/- 0.01. ENTITY is errors per 1000
reference words; raw's 22.6 is the floor a refiner must not exceed.

**Ship queue `ship_v0` is stopped.** It was halted on 2026-09-21 at the owner's request with 4 of
18 jobs done (both new corpora, both reference benches). The three trains (`ship_lfm12`,
`ship_qwen17`, `ship_lfm12_gen`) restart from scratch on resume; everything else depends on them.
State is in `/data/phonon_queue/ship_v0/state.json`; the three trains are recorded as `running`
there, which is stale. Resume from the checkout root:

```bash
uv run --project research/harness phonon-research queue research/harness/configs/ship_v0/jobs_ship_v0.yaml
uv run --project research/harness phonon-research status --run ship_v0
```

The shell jobs call scripts under `/data/phonon_ship_v0/scripts/`; copies are in
`research/ship_v0/scripts/`. Check `nvidia-smi` first: the box is shared.

**Open work**, in order:

1. Resume `ship_v0` and fill the ship and quantization tables (`research/ship_v0/scripts/report.py`).
2. Record the field set: 300 held-out-term sentences read aloud (`research/record_terms/record.py`).
   No clips exist yet. This is the one number the pipeline cannot produce without a person. The
   script reads `read_script.jsonl` beside it, built from the held-out terms and kept out of git.
3. Multiprocess the bench's retrieval stage (140 s of GIL-bound Python on every backend; target
   under 15 s). `retrieval_mp` in the queue is that job.
4. Lift `research/bench_v0` into `tools/` as the release gate.

## Layout

| path | what |
|---|---|
| `research/harness/` | `phonon-research`: corpus / train / eval / queue from yaml configs, adapter and set registries |
| `research/bench_v0/` | `phonon-bench`: one benchmark over frozen fixtures, backends hf / mlx / llama / vllm |
| `research/corrector_v0/` | `train_lora.py` (the only trainer), shared prompt and scoring helpers, date-split builders |
| `research/synth_v0..v3/` | text-synthetic corpora (repo walk, LFM2.5 utterances, corruption); they never beat raw |
| `research/term_eval_v0/` | held-out term clips (Kokoro TTS, Parakeet) and the term-hit scorer |
| `research/vocab_v0`, `vocab_v1/` | vocabulary-in-prompt refiner, retrieval v1 (`retrieve.py`), unseen voices, real audio |
| `research/retrieval_v2/` | retrieval v2 (`retrieve2.py`): n-gram window 8, spoken realisations, phoneme weight, kind prior, two-stage |
| `research/scaling_v0/`, `bigrun_v0/`, `pool2_v0/`, `pool3_v0/` | the acoustic scaling law and the big corpora; `refine_bigrun.py` has the guard |
| `research/index_v0/` | tree-sitter symbol index (definitions plus mentions tiers) and the adaptive budget |
| `research/restraint_v0/`, `sizesweep_v0/`, `nbest_v0/` | restraint levers, 350M to E2B size sweep, n-best alternatives |
| `research/quant_v0/` | adapter merge, MLX and GGUF quant sweeps, LFM-aware mixed-precision predicates |
| `research/perf_v1/`, `kernels_sm120/` | LoRA step profile, fused-CE check, sm_120 GEMM bake-off |
| `research/asr_errors_v0/`, `asr_redux_v0/` | multi-ASR error classifier; parakeet-redux comparison (rejected) |
| `research/latent_v0/`, `fusion_v0/`, `corrector_v1/` | encoder latent-space study, ASR fusion, multi-hypothesis corrector |
| `research/ship_v0/scripts/` | quant, retrieval multiprocess and report scripts the ship queue calls |
| `research/record_terms/record.py` | Mac recorder for the field set (sox, 16 kHz mono) |
| `src/phonon/`, `scripts/option0/` | the older ASR benchmark package and gate scripts some experiments import |
| `src/phonon/vendor.py`, `src/phonon/eval.py` | vendor comparison tools; they call third-party APIs and are not part of the app |

Per-experiment `results.md` tables, generated plots and all jsonl/json outputs stay on the GPU box.

## Running

Environment on the box: `HF_HOME=/data/hf`, `UV_TORCH_BACKEND=cu130`, `TOKENIZERS_PARALLELISM=false`,
venvs under `/data/venvs/` (a torch venv is 6 GB). `research/bench_v0/bench.env` sets these for the
bench. Python is always `uv run`; the two packages pin their own `uv.lock`.

```bash
# harness
uv run --project research/harness phonon-research corpus research/harness/configs/corpus_smoke_e2e.yaml
uv run --project research/harness phonon-research train research/harness/configs/train_smoke_a.yaml
uv run --project research/harness phonon-research eval bigrun_mid_r16 --sets real_580,term_new
uv run --project research/harness phonon-research queue <jobs.yaml>

# bench
set -a; . research/bench_v0/bench.env; set +a
uv run --project research/bench_v0 phonon-bench fixtures verify
uv run --project research/bench_v0 phonon-bench run --backend hf --adapter /data/phonon_corrector_v0/adapters/bigrun_mid_r16
uv run --project research/bench_v0 phonon-bench compare bigrun_mid_r16 restr_real12
```

A job is `{name, kind: corpus|train|eval|shell, gpu: 0-3|any|cpu, needs, config|command, timeout_s}`.
The scheduler runs in tmux `pq-<run>`, logs to `/data/phonon_queue/<run>/<job>.log`, rewrites
`state.json` on every transition, retries once, skips dependants of a failure, and re-invoking the
same yaml reruns only what failed. Harness overhead is about 0.1 s per job. A new experiment is a
copied config plus two lines in a jobs.yaml, not a new script.

ASR (NeMo) runs in the root project env (`pyproject.toml` here), not the harness env; NeMo's pins
fight the training pins. vLLM needs its own venv (`/data/venvs/vllm`) and is the fastest bench
backend: 64 s for all six sets.

## Data on the GPU box

| path | what |
|---|---|
| `/data/phonon_bench_v0/sets/` | frozen bench fixtures with a sha256 manifest; `ref/` cached bf16 numerics; `bench.jsonl` run log |
| `/data/phonon_corrector_v0/` | real pairs, the date split (`datesplit/`, cutoff 2026-09-04), all adapters under `adapters/` |
| `/data/phonon_bigrun_v0/`, `phonon_pool2_v0/`, `phonon_pool3_v0/` | acoustic corpora (about 104 GB of clips across pools) |
| `/data/phonon_term_eval_v0/`, `v1/` | held-out term clips, seen and unseen voices |
| `/data/phonon_quant_v0/` | merged model, MLX and GGUF builds, imatrix |
| `/data/phonon_queue/<run>/` | queue state and logs |
| `/data/phonon_personal/` | the owner's dictation exports. Never copied, never published, never a generation source |

## Rules

- Held-out sets never enter training: `wispr_holdout120`, the date-split holdout, and every held-out
  term set (term_eval v0/v1, heldout_new, pool2, pool3). `wispr_edit25` is contaminated (15 of 25
  rows are in `train.jsonl`); do not report it.
- Personal dictation text never leaves `/data`: no quotes in code, logs, docs or commits.
- Compare runs only at the same `--max-batch`. Batched greedy decode is not bit-reproducible when the
  batch's `max_new_tokens` changes (batch 96 vs 24 agree 0.966 by word).
- The word-count guard is on for every reported number: `max_new_tokens = 1.5 x input + 32`, fall
  back to the raw input when the output is over 2x or under 0.4x the input words (inputs over 8 words).
- A refiner that raises real-audio ENTITY above raw, or damage above 0.057, does not ship regardless
  of term hit.
- Before any run longer than a minute, time its stages and fix the biggest one first.

## Traps

- transformers 5.9 to 5.17 drop `block_ff_dim` when saving a merged LFM2; copy the base
  `config.json` back before `mlx_lm.convert`.
- MLX on CUDA needs `mlx[cuda13]`, and `MLX_CUDA_CONV_CACHE_SIZE`, `MLX_CUDA_SDPA_CACHE_SIZE` and
  `MLX_CUDA_GRAPH_CACHE_SIZE` raised, or per-row token caps abort on cache thrashing.
- mlx_lm's `mixed_*` recipes do not match LFM2 layer names; use the predicates in `research/quant_v0/quant_eval.py`.
- LFM2's conv path falls back to reference PyTorch without `causal_conv1d`; it works, slower.
- Qwen3.5 training without flash-linear-attention and causal-conv1d is silently 7x slower.
- Kokoro's spaCy model installs itself on first use and races TTS workers; it is pinned in the harness.
- Over ssh, `pkill -f` can match your own shell and the tmux server; kill by pid or `tmux kill-session`.
