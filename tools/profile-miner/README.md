# profile-miner

Dev tool for SPEC "User vocabulary onboarding", step 3. Deterministic
mining pipeline plus a dev-only scorer. Runs local only. Nothing is uploaded.

## Run

```sh
cd tools/profile-miner
PHONON_MINER_OUT=/path/to/out UNTIL=2026-08-30T00:00:00 ./run.sh
```

`run.sh` runs the stages below in order. Each stage is also a subcommand:

```sh
uv run --offline --python 3.12 python -m profile_miner extract [--until ISO] [--source claude|codex|grok|repos]
uv run --offline --python 3.12 python -m profile_miner seed [--no-github]
uv run --offline --python 3.12 python -m profile_miner candidates [--min-count 2]
HF_HUB_OFFLINE=1 uv run --offline --python 3.12 --with parakeet-mlx==0.5.2 python -m profile_miner oracle [--top 8000] [--workers 12]
uv run --offline --python 3.12 python -m profile_miner rank
HF_HUB_OFFLINE=1 uv run --offline --python 3.12 --with mlx-lm==0.31.3 python -m profile_miner gemma [--minutes 20] [--top 400]
HF_HUB_OFFLINE=1 uv run --offline --python 3.12 --with parakeet-mlx==0.5.2 python -m profile_miner score
```

Output dir (`PHONON_MINER_OUT`, default `./out`, git-ignored):

| path | stage | content |
|---|---|---|
| `extract/<source>.txt`, `extract/counts.json` | 2 | user-authored lines per source, line counts |
| `seed/seeds.json` | 3a | identity and repo seeds |
| `candidates/candidates_raw.json` | 3b | all rule-based candidates with counts per source |
| `oracle/cache.jsonl` | 3c | TTS to Parakeet output per term, two voices; reruns skip cached terms |
| `mined/candidates.json` | 3 | ranked output: `{term, count, sources, spoken_forms, diff, evidence, classes, seed, score, rank}` |
| `mined/gemma_pass.json` | 3d | optional local Gemma proposals and `kind` labels, time-boxed |
| `score/report.json` | eval | recall, precision proxy, miss list |

## Stages

- extract: Claude `type=user` string content; Codex `payload.role=user`
  `input_text` lines minus harness wrappers, headings, tags, fences and lines
  over 80 words; Grok `prompt`; repo docs (README, AGENTS, DEVLOG, SPEC,
  CLAUDE) plus file tree to depth 2 for git repos under `$HOME` to depth 3,
  `Library` and dot dirs excluded. Chat messages over 300 words are pastes:
  only their first 120 words are kept. `--until` drops records at or after a
  timestamp (held-out cutoff). The `phonon` repo docs are skipped because they
  quote the evaluation set. Files stream line by line.
- seed: `id -F`, `git config --global user.name`, `$USER`, `gh api user`
  login when `gh` is authed, repo basenames.
- candidates: classes `camel`, `caps`, `digit`, `ident` (hyphen, underscore,
  dot), `cap` (Capitalized, not sentence-initial, count >= 2), `lower`
  (absent from `/usr/share/dict/words` after simple inflection stripping),
  `file` (name with extension), `ngram` (2 or 3 adjacent special tokens).
  English words that are only sometimes Capitalized or ALLCAPS must be so in
  30% of their occurrences. URLs, paths, emails, hex, keys and tokens over 40
  chars are dropped. Exact duplicate lines count once. Keeps count >= 2 plus
  all seeds.
- oracle: `say -v Samantha` and `say -v Daniel` of "let's use {term} for
  this", `sox` to 16 kHz mono, Parakeet `mlx-community/parakeet-tdt-0.6b-v2`
  loaded once. `say` runs in a 12-wide pool, transcription overlaps in the
  main thread. Diff classes: `same`, `case`, `format` (equal after lowercasing
  and removing separators), `phonetic`. Budget: top N raw candidates by
  `log1p(count) * (1 + 0.5 * (sources - 1))`.
- rank: keeps terms the oracle changed (plus seeds). Score is the prelim
  score times a diff weight (`phonetic`, `format` 1.0; `case` 0.2) plus 1 for
  seeds.
- gemma: `mlx-community/gemma-4-e2b-it-4bit` (Phonon's polish model).
  Assigns `kind` to the top 400, then proposes names and multiword units from
  sampled short lines; proposals are kept only when they appear verbatim in
  the sampled text. Stops at the time box and writes `finished: false`.
- score: the only code that reads `~/Library/Application Support/Phonon`.
  Gold is used-or-starred dictionary entries, replacement pairs collapsed
  onto their term, filtered to terms the oracle changes or the corpus shows
  mangled (term in final or intended transcript, absent from raw). Recall at
  100/200/400/all with spoken-form variants collapsed; precision proxy is
  the share of the top 100/400 present in the dictionary or corpus text.

## Notes

- The oracle and Gemma both use the GPU. Run them one at a time.
- Do not commit the output dir; extracts are the user's private text.
