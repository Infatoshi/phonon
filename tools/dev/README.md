# tools/dev

Developer tooling. These scripts send data to cloud APIs and are not part of
the app; Phonon itself never sends audio or text off the Mac.

- `muse_bench.py`: uploads recordings from your local corpus to Meta's Muse
  speech API and compares its transcripts with Phonon's. Needs `MODEL_API_KEY`.
- `label_opus.py`: sends vocabulary candidates to OpenRouter for keep/drop
  labels (step 3 of `tools/profile-miner/judge/runner.sh`). Needs
  `OPENROUTER_API_KEY`.
