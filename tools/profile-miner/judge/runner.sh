#!/bin/bash
# Runs on the GPU box, detached: oracle the live candidates (Kokoro), label them with Opus 5 via OpenRouter,
# zero-shot 4B judge (two prompts), LoRA per-term judge, gate on held-out Opus labels. Restartable: each
# stage is skipped when its output exists. Writes report.json and done.txt (RC=n).
export PATH="$HOME/.local/bin:$PATH"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
cd ~/gym/judge
rm -f done.txt
AGENT=phonon-judge
log() { echo "[runner] $(date '+%F %T') $*" | tee -a progress.log; }
finish() {
  RC=$1
  tmux kill-session -t jserve 2>/dev/null
  [ -n "${HB_PID:-}" ] && kill "$HB_PID" 2>/dev/null
  overnight-compute release --agent $AGENT --status $([ "$RC" = 0 ] && echo done || echo failed) >/dev/null 2>&1
  log "exit RC=$RC"
  echo "RC=$RC" > done.txt
  exit "$RC"
}

# 0. lease gpu1 (the wait auto-schedules an ephemeral lease)
log "waiting for gpu1 lease"
overnight-compute wait --agent $AGENT --resource gpu1 --ttl 4h --timeout 10h --poll 60s > wait.log 2>&1 || { log "lease wait failed: $(tail -1 wait.log)"; finish 2; }
log "lease acquired"
( while true; do sleep 600; overnight-compute heartbeat --agent $AGENT --ttl 30m >/dev/null 2>&1; done ) &
HB_PID=$!
export CUDA_VISIBLE_DEVICES=1

# 1. oracle the live terms that have no spoken form yet
if [ ! -f oracle.ok ]; then
  python3 -c "import json; t=[c['term'] for c in json.load(open('judge_input.json')) if c.get('diff') is None]; json.dump(t, open('terms_needed.json','w'), ensure_ascii=False); print(len(t))" > terms_needed.count
  log "oracle: $(cat terms_needed.count) terms"
  ( . ~/gym/oracle/.venv-oracle/bin/activate && python ~/gym/oracle/oracle_linux.py terms_needed.json judge_cache.jsonl --batch 64 2> oracle.log > oracle.out ) || { log "oracle failed: $(tail -3 oracle.log)"; finish 3; }
  grep -q ORACLE_DONE oracle.out || [ "$(cat terms_needed.count)" = 0 ] || { log "oracle did not finish"; finish 3; }
  touch oracle.ok
fi

# 2. attach spoken forms, drop 'same', top 2000, held-out split
if [ ! -f judge_final.json ]; then
  python3 prep.py judge_input.json judge_cache.jsonl judge_final.json heldout_ids.json 2000 2>> progress.log || finish 4
fi

# 3. Opus labels through OpenRouter (network, no GPU). label_opus.py lives in tools/dev/
#    because it sends data to a cloud API; copy it next to this script on the box.
if [ ! -f labels_opus.json ]; then
  log "opus labelling"
  python3 label_opus.py judge_final.json labels_opus.jsonl labels_opus.json 2> opus.log || { log "opus failed: $(tail -3 opus.log)"; finish 5; }
  tail -1 opus.log | tee -a progress.log
  python3 -c "import json,sys; l=json.load(open('labels_opus.json')); c=json.load(open('judge_final.json')); sys.exit(0 if len(l) >= 0.98*len(c) else 1)" || { log "opus labelled too few"; finish 5; }
fi

# 4. zero-shot batch judge, base and strict prompts, through vLLM
if [ ! -f zs_strict.json ]; then
  log "vllm serve base 4B"
  tmux kill-session -t jserve 2>/dev/null; rm -f serve.log
  tmux new-session -d -s jserve ". ~/.venv-vllm/bin/activate && CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3.5-4B --served-model-name base --port 8398 --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.5 --reasoning-parser qwen3 2>&1 | tee serve.log"
  for i in $(seq 1 120); do
    grep -q "Application startup complete" serve.log 2>/dev/null && break
    grep -q "EngineCore failed to start" serve.log 2>/dev/null && break
    sleep 5
  done
  if grep -q "Application startup complete" serve.log; then
    python3 judge_zs.py judge_final.json zs_base.json http://localhost:8398/v1 base base 2> zs_base.log
    python3 judge_zs.py judge_final.json zs_strict.json http://localhost:8398/v1 base strict 2> zs_strict.log
    tail -1 zs_base.log zs_strict.log | tee -a progress.log
  else
    log "vllm failed to start; skipping zero-shot batch judge"
    echo '{}' > zs_base.json; echo '{}' > zs_strict.json
  fi
  tmux kill-session -t jserve 2>/dev/null
  sleep 10
fi

# 5. LoRA per-term judge
if [ ! -f out/summary.json ]; then
  log "training LoRA judge"
  mkdir -p out
  ( . ~/ft/.venv/bin/activate && python train_judge.py --input judge_final.json --labels labels_opus.json --heldout heldout_ids.json --gold gold_terms.json --out out 2> train.log > train.out ) || { log "train failed: $(tail -5 train.log)"; finish 6; }
  grep -q TRAIN_DONE train.out || { log "train did not finish"; finish 6; }
fi

# 6. report
python3 - <<'EOF' 2>&1 | tee -a progress.log
import json
c = json.load(open("judge_final.json")); held = set(json.load(open("heldout_ids.json")))
lab = {int(k): v for k, v in json.load(open("labels_opus.json")).items()}
gold = set(json.load(open("gold_terms.json")))
rep = {"candidates": len(c), "heldout": len(held), "opus_keep": sum(lab.values()), "opus_labelled": len(lab),
       "opus_gold_kept": sum(lab.get(x["id"], False) for x in c if x["term"].strip().lower() in gold),
       "gold_in_candidates": sum(x["term"].strip().lower() in gold for x in c)}
try:
    rep["opus_usage"] = json.load(open("labels_opus.json.usage.json"))
except Exception:
    pass
for name in ("zs_base", "zs_strict"):
    try:
        z = {int(k): v for k, v in json.load(open(f"{name}.json")).items()}
    except Exception:
        continue
    for split, ids in (("heldout", held), ("all", set(lab))):
        ids = [i for i in ids if i in z and i in lab]
        if not ids:
            continue
        tp = sum(z[i] and lab[i] for i in ids); fp = sum(z[i] and not lab[i] for i in ids); fn = sum(lab[i] and not z[i] for i in ids)
        rep[f"{name}_{split}"] = {"n": len(ids), "agreement": round(sum(z[i] == lab[i] for i in ids) / len(ids), 3),
                                  "keep_rate": round(sum(z[i] for i in ids) / len(ids), 3),
                                  "keep_precision": round(tp / max(tp + fp, 1), 3), "keep_recall": round(tp / max(tp + fn, 1), 3)}
try:
    rep["lora"] = json.load(open("out/summary.json"))
except Exception as e:
    rep["lora_error"] = str(e)
# keep rate by miner rank band under Opus
bands = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1200), (1200, 2000)]
rep["opus_keep_by_band"] = {f"{a+1}-{b}": round(sum(lab.get(x["id"], False) for x in c[a:b]) / max(len(c[a:b]), 1), 2) for a, b in bands}
json.dump(rep, open("report.json", "w"), indent=1)
print("[report]", json.dumps({k: v for k, v in rep.items() if k != "lora"}, indent=None))
print("[report] lora", json.dumps({k: v for k, v in rep.get("lora", {}).items() if k != "heldout_disagreements"}))
EOF
finish 0
