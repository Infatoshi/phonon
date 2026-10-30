"""Label candidates keep/drop with Opus 5 through OpenRouter. Resumable: appends per-item results to a JSONL
and skips ids already labelled. The key comes from OPENROUTER_API_KEY (or a dotenv file below) and is never printed.

Developer tooling, not part of the app: this script SENDS vocabulary candidates to OpenRouter's cloud API.
It is step 3 of tools/profile-miner/judge/runner.sh; the prompts live in tools/profile-miner/judge/prompts.py.
usage: python label_opus.py judge_final.json labels_opus.jsonl labels_opus.json [--model anthropic/claude-opus-5]
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "profile-miner" / "judge"))
from prompts import BASE, fmt_batch_item

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("jsonl"); ap.add_argument("out")
ap.add_argument("--model", default="anthropic/claude-opus-5")
ap.add_argument("--batch", type=int, default=20)
a = ap.parse_args()


def load_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for p in (os.path.expanduser("~/.omp/agent/.env"), os.path.expanduser("~/.env_vars")):
        if not os.path.exists(p):
            continue
        for line in open(p):
            m = re.match(r"^(?:export\s+)?OPENROUTER_API_KEY\s*=\s*['\"]?([^'\"\s]+)", line)
            if m:
                return m.group(1)
    sys.exit("no OPENROUTER_API_KEY found")


KEY = load_key()
cands = json.load(open(a.inp))
done = {}
if os.path.exists(a.jsonl):
    for line in open(a.jsonl):
        try:
            d = json.loads(line)
            done[int(d["id"])] = bool(d["keep"])
        except Exception:
            pass
todo = [c for c in cands if c["id"] not in done]
print(f"[opus] {len(cands)} candidates, {len(done)} already labelled, {len(todo)} to do", file=sys.stderr)
usage = {"prompt": 0, "completion": 0, "calls": 0}


def call(batch):
    msgs = [{"role": "system", "content": BASE},
            {"role": "user", "content": "Candidates:\n" + "\n".join(fmt_batch_item(c) for c in batch) + "\nReturn only the JSON array."}]
    body = {"model": a.model, "messages": msgs, "temperature": 0, "max_tokens": 4000}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}",
                                          "HTTP-Referer": "https://github.com/Infatoshi/phonon", "X-Title": "phonon judge"})
    for attempt in range(6):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            u = r.get("usage") or {}
            usage["prompt"] += u.get("prompt_tokens", 0); usage["completion"] += u.get("completion_tokens", 0); usage["calls"] += 1
            txt = r["choices"][0]["message"]["content"]
            s, e = txt.find("["), txt.rfind("]")
            res = {int(x["id"]): bool(x["keep"]) for x in json.loads(txt[s:e + 1])}
            want = {c["id"] for c in batch}
            if not want <= set(res):
                print(f"[opus] batch missing ids {sorted(want - set(res))[:5]} (attempt {attempt})", file=sys.stderr)
                if attempt < 2:
                    continue
            return {k: v for k, v in res.items() if k in want}
        except urllib.error.HTTPError as ex:
            msg = ex.read().decode(errors="replace")[:200]
            print(f"[opus] HTTP {ex.code} attempt {attempt}: {msg}", file=sys.stderr)
        except Exception as ex:
            print(f"[opus] attempt {attempt}: {ex}", file=sys.stderr)
        time.sleep(5 * (attempt + 1))
    return {}


t0 = time.time()
with open(a.jsonl, "a") as f:
    for i in range(0, len(todo), a.batch):
        batch = todo[i:i + a.batch]
        res = call(batch)
        for k, v in res.items():
            f.write(json.dumps({"id": k, "keep": v}) + "\n")
            done[k] = v
        f.flush()
        print(f"[opus] {min(i + a.batch, len(todo))}/{len(todo)} {time.time() - t0:.0f}s labelled {len(done)}", file=sys.stderr)
    # second pass in smaller batches for anything still missing
    missing = [c for c in cands if c["id"] not in done]
    for i in range(0, len(missing), 5):
        res = call(missing[i:i + 5])
        for k, v in res.items():
            f.write(json.dumps({"id": k, "keep": v}) + "\n")
            done[k] = v
        f.flush()
json.dump({str(k): v for k, v in sorted(done.items())}, open(a.out, "w"))
cost = usage["prompt"] * 5e-6 + usage["completion"] * 25e-6
print(f"[opus] done: {sum(done.values())} keep of {len(done)} labelled ({len(cands)} in) in {time.time() - t0:.0f}s; "
      f"{usage['calls']} calls, {usage['prompt']} prompt + {usage['completion']} completion tokens, ~${cost:.2f}",
      file=sys.stderr)
json.dump({**usage, "cost_usd": round(cost, 2), "model": a.model}, open(a.out + ".usage.json", "w"))
