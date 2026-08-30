"""Stage 3c: ASR oracle. `say` two voices -> sox 16 kHz mono -> Parakeet.

Run under: HF_HUB_OFFLINE=1 uv run --offline --python 3.12 --with parakeet-mlx==0.5.2
Results are cached per term in <out>/oracle/cache.jsonl, so reruns are cheap.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import norm, out_dir, read_json, write_json

VOICES = ["Samantha", "Daniel"]
CARRIER = "let's use {} for this"
RE_CARRIER = re.compile(r"^\W*let'?s\s+use\s+(.*?)\s*for\s+this\W*$", re.I)
MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v2"


def load_cache(path: Path):
    cache = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    cache[d["term"]] = d["voices"]
                except Exception:
                    pass
    return cache


def synth(term, voice, tmpdir):
    safe = re.sub(r"[^A-Za-z0-9]+", "_", term)[:40]
    aiff = Path(tmpdir) / f"{safe}_{voice}_{os.getpid()}_{time.time_ns()}.aiff"
    wav = aiff.with_suffix(".wav")
    text = CARRIER.format(term)
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True, capture_output=True, timeout=60)
    subprocess.run(["sox", str(aiff), "-r", "16000", "-c", "1", "-b", "16", str(wav)], check=True,
                   capture_output=True, timeout=60)
    aiff.unlink(missing_ok=True)
    return wav


def strip_carrier(text):
    m = RE_CARRIER.match(text.strip())
    if m:
        return m.group(1).strip(), True
    return text.strip().rstrip(".!?,"), False


def diff_class(term, heard):
    """same | case | format | phonetic"""
    t = term.strip()
    h = heard.strip().rstrip(".!?,")
    if h == t:
        return "same"
    if h.lower() == t.lower():
        return "case"
    if norm(h) == norm(t):
        return "format"
    return "phonetic"


def summarize(term, voices):
    """Combine per-voice outputs into (diff, spoken_forms)."""
    order = {"same": 0, "case": 1, "format": 2, "phonetic": 3}
    worst = "same"
    forms = []
    for v in VOICES:
        heard = voices.get(v)
        if heard is None:
            continue
        d = diff_class(term, heard)
        if order[d] > order[worst]:
            worst = d
        if d != "same" and heard not in forms:
            forms.append(heard)
    return worst, forms


def run(terms, cache_path=None, workers=12, model=None):
    od = out_dir()
    cache_path = cache_path or od / "oracle" / "cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    todo = [t for t in dict.fromkeys(terms) if t not in cache]
    print(f"[oracle] {len(terms)} terms, {len(cache)} cached, {len(todo)} to run", file=sys.stderr)
    if todo:
        if model is None:
            from parakeet_mlx import from_pretrained
            model = from_pretrained(MODEL_ID)
        tmpdir = tempfile.mkdtemp(prefix="oracle_", dir=str(od / "oracle"))
        t0 = time.time()
        done = 0
        mismatch = 0
        say_s = 0.0
        tx_s = 0.0
        pending = {t: {} for t in todo}
        with open(cache_path, "a") as cf, ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for t in todo:
                for v in VOICES:
                    futs[ex.submit(synth, t, v, tmpdir)] = (t, v)
            for fut in as_completed(futs):
                t, v = futs[fut]
                try:
                    wav = fut.result()
                except Exception as e:
                    pending[t][v] = None
                    print(f"[oracle] say failed for {t!r}: {e}", file=sys.stderr)
                else:
                    s = time.time()
                    try:
                        text = model.transcribe(str(wav)).text
                    except Exception as e:
                        text = ""
                        print(f"[oracle] transcribe failed for {t!r}: {e}", file=sys.stderr)
                    tx_s += time.time() - s
                    wav.unlink(missing_ok=True)
                    heard, ok = strip_carrier(text)
                    if not ok:
                        mismatch += 1
                    pending[t][v] = heard
                if len(pending[t]) == len(VOICES):
                    voices = pending.pop(t)
                    cache[t] = voices
                    cf.write(json.dumps({"term": t, "voices": voices}, ensure_ascii=False) + "\n")
                    done += 1
                    if done % 200 == 0:
                        el = time.time() - t0
                        cf.flush()
                        print(f"[oracle] {done}/{len(todo)} terms, {el:.0f}s, {done / el:.2f} terms/s "
                              f"(transcribe {tx_s:.0f}s)", file=sys.stderr)
        el = time.time() - t0
        print(f"[oracle] done {done} terms in {el:.0f}s = {done / max(el, 1e-9):.2f} terms/s, "
              f"{2 * done / max(el, 1e-9):.2f} clips/s; carrier mismatches {mismatch}; transcribe {tx_s:.0f}s",
              file=sys.stderr)
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
    return cache


def prelim_score(c):
    import math
    return math.log1p(c["count"]) * (1.0 + 0.5 * max(len(c["sources"]) - 1, 0))


def run_candidates(workers=12, top=6000):
    od = out_dir()
    cands = read_json(od / "candidates" / "candidates_raw.json")
    ranked = sorted(cands, key=lambda c: (-prelim_score(c), c["key"]))
    keep = ranked[:top] + [c for c in ranked[top:] if c["seed"]]
    print(f"[oracle] {len(cands)} candidates, oracle budget top {top} by count and breadth "
          f"(+{len(keep) - min(top, len(ranked))} seeds)", file=sys.stderr)
    cands = keep
    terms = [c["term"] for c in cands]
    cache = run(terms, workers=workers)
    stats = {"same": 0, "case": 0, "format": 0, "phonetic": 0}
    for c in cands:
        diff, forms = summarize(c["term"], cache.get(c["term"], {}))
        stats[diff] += 1
    write_json(od / "oracle" / "stats.json", stats)
    print(f"[oracle] classes {stats}", file=sys.stderr)
    return stats
