# persona-gym

Dev tool for SPEC "User vocabulary onboarding". Manufactures synthetic
"user machines" with planted gold vocabulary, runs a teacher model
agentically over each, and grades the rollouts. Companion to
`tools/profile-miner` (same style: stdlib Python, uv run, local only).

## Run

```sh
cd tools/persona-gym
uv run --offline --python 3.12 python -m persona_gym build --n 20 --out out/personas
uv run --offline --python 3.12 python -m persona_gym rollout \
  --personas out/personas --endpoint http://localhost:8000/v1 \
  --model qwen3-32b --out out/rollouts --n-per 1 --max-turns 24
uv run --offline --python 3.12 python -m persona_gym grade \
  --personas out/personas --rollouts out/rollouts
```

Tests: `uv run --offline --python 3.12 --with pytest pytest tests`

## build

Each persona dir under `--out`:

- `repos/`: 2-4 small public GitHub repos, shallow-cloned from a curated
  ~60-entry list (embedded, gamedev, bio, web, ml, audio), one domain each,
  `.git` and binaries deleted. Dead URLs are skipped at clone time. Clones
  are cached in `out/.cache/repos` and copied per persona.
- `logs/claude.txt`, `logs/codex.txt`: 300-800 synthetic chat lines in
  profile-miner extract format (one user message per line), from templates
  plus randomization, no LLM. Styles: voice-typed lowercase with filler,
  terse, verbose. Lines mix ~25 fabricated terms with real repo vocabulary.
- Fabricated terms come from word-part combinators (project codenames,
  machine names, people, model names like `deneb-rl` / `Qorvex 2B`, tools),
  checked against `/usr/share/dict/words` and unique across personas.
- A sample of 5 terms per persona goes through profile-miner's ASR oracle
  (`say` two voices -> Parakeet, one batch, model loads once) and the
  spoken-form manglings are injected into the logs ("the neb rl" for
  `deneb-rl`). `--no-oracle` skips this; the skip reason lands in gold.json.

Gold lives OUTSIDE the visible tree: `out/meta/<persona>/gold.json` with
planted terms + counts, injected manglings, oracle note, and source repos.

## rollout

Agent loop against an OpenAI-compatible `/v1/chat/completions` endpoint with
native tool calling. One tool: `bash` in the persona dir, 15 s timeout,
output truncated to 4 KB, no network (proxy envs stripped; curl, wget, ssh,
git, scp, sftp blocked by a PATH shim). The system prompt states the mining
task and the exact output schema `{term, count, kind, spoken_forms,
evidence}`, final answer as one JSON array in a ```json fence. One retry on
transport errors; one nudge if the final message lacks the fence. Per
rollout: `trajectory.jsonl` (all messages and tool results), `answer.json`,
`meta.json` (turns, status, token usage when the API returns it).
`OPENAI_API_KEY` is sent as the bearer token if set.

## grade

Per rollout vs gold.json:

- planted-term recall (normalized match; a term's injected manglings count
  for it),
- injected-mangling recall (normalized form),
- precision proxy: share of emitted terms that are planted, real repo vocab
  (term-frequency pass over the persona repos), or verbatim in the persona
  tree,
- turns used, JSON validity.

PASS = planted recall >= 0.7 and precision proxy >= 0.6 and valid JSON.
Writes `report.json` next to the rollouts and a compact table to stdout.

## Notes

- `build` needs network (GitHub clones) and, for manglings, the cached
  Parakeet model plus a free GPU; everything else is offline.
- Do not commit `out/`; it is generated and can be large.
