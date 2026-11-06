#!/bin/sh
# Speech fixtures for the pipeline check and the in-app self-test, spoken by macOS `say`.
# expected.json holds what the speaker meant; checks compare ASR and refiner output to it.
set -e
cd "$(dirname "$0")/../Fixtures"
mk() { say -v Samantha -o "$1.aiff" "$2"; afconvert -f WAVE -d LEI16@16000 -c 1 "$1.aiff" "$1.wav"; rm "$1.aiff"; }
mk 01-plain "Hey, are we still on for dinner tomorrow at seven?"
mk 02-restart "Let's ship it on Friday. No wait, let's ship it on Monday instead."
mk 03-filler "Um, so I think, uh, the build is broken because the cache is stale."
mk 04-technical "Can you run the CUDA kernel benchmark with V L L M on the H100 tonight?"
cat > expected.json <<'JSON'
{
 "01-plain": "Hey, are we still on for dinner tomorrow at seven?",
 "02-restart": "Let's ship it on Monday instead.",
 "03-filler": "So I think the build is broken because the cache is stale.",
 "04-technical": "Can you run the CUDA kernel benchmark with vLLM on the H100 tonight?"
}
JSON
