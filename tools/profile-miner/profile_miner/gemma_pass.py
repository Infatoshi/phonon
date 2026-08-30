"""Optional stage 3d: local Gemma pass (the model only adds and labels).

Run under: HF_HUB_OFFLINE=1 uv run --offline --python 3.12 --with mlx-lm==0.31.3
Two jobs, time-boxed: (A) propose names and multiword units from sampled short
lines, kept only when grounded in the sampled text; (B) assign `kind` to the
top N ranked candidates. Writes mined/gemma_pass.json with finished=true/false.
"""

import json
import random
import re
import sys
import time

from .common import norm, out_dir, read_json, write_json

MODEL_ID = "mlx-community/gemma-4-e2b-it-4bit"  # POLISH_MODEL_ID in crates/phonon-llm/src/paths.rs
KINDS = ["person", "project", "machine", "company", "product", "model", "tool", "hardware", "command", "file",
         "jargon", "acronym", "place", "noise"]
RE_THOUGHT = re.compile(r"<\|channel>thought.*?(?:<channel\|>|$)|<think>.*?(?:</think>|$)", re.S)
RE_JSON = re.compile(r"\[.*\]|\{.*\}", re.S)

PROMPT_A = (
    "These lines were written by one software developer. List the proper names, product and project names, "
    "machine names, people, company names, and multiword technical units (two or three words that form one name) "
    "that appear in the lines. Only list strings that appear verbatim. No common words, no explanation. "
    "Reply with a JSON array of strings.\n\n{lines}"
)
PROMPT_B = (
    "Classify each term written by a software developer. Kinds: " + ", ".join(KINDS) + ". "
    "Use noise for anything a person would not dictate as a term: code abbreviations (env, tmp, args), "
    "identifier fragments, plain English words, greetings, month names, numbers. "
    "Reply with a JSON object mapping each term to one kind. No explanation.\n\n{terms}"
)


def _gen(model, tok, text, max_tokens):
    from mlx_lm import generate
    msgs = [{"role": "user", "content": text}]
    try:
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
    out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
    out = RE_THOUGHT.sub("", out)
    return out.strip()


def _parse(out, want):
    out = out.replace("```json", "").replace("```", "")
    m = RE_JSON.search(out)
    if not m:
        return None
    try:
        v = json.loads(m.group())
    except Exception:
        return None
    return v if isinstance(v, want) else None


def sample_lines(path, n, rng, lo=5, hi=40):
    pool = []
    with open(path) as f:
        for line in f:
            w = len(line.split())
            if lo <= w <= hi:
                pool.append(line.strip())
    pool = list(dict.fromkeys(pool))
    rng.shuffle(pool)
    return pool[:n]


def run(minutes=20.0, top=400, per_source=150, batch=15, seed=0, model_id=MODEL_ID, out_name="gemma_pass.json"):
    from mlx_lm import load
    od = out_dir()
    deadline = time.time() + minutes * 60
    t0 = time.time()
    model, tok = load(model_id)
    print(f"[gemma] loaded {model_id} in {time.time() - t0:.1f}s", file=sys.stderr)
    rng = random.Random(seed)
    result = {"model": model_id, "finished": False, "proposals": [], "kinds": {}, "seconds": 0, "prompts": 0}
    out_path = od / "mined" / out_name

    def save(final=False):
        result["seconds"] = round(time.time() - t0, 1)
        result["finished"] = final
        write_json(out_path, result)

    # Job B first: kinds for the top N ranked candidates (short, bounded).
    ranked = read_json(od / "mined" / "candidates.json")[:top]
    kinds = {}
    for i in range(0, len(ranked), 40):
        if time.time() > deadline:
            break
        chunk = ranked[i:i + 40]
        terms = "\n".join(f"- {c['term']}" + (f"  (e.g. \"{c['evidence'][0][:90]}\")" if c["evidence"] else "")
                          for c in chunk)
        out = _gen(model, tok, PROMPT_B.format(terms=terms), 600)
        result["prompts"] += 1
        obj = _parse(out, dict)
        if obj:
            for c in chunk:
                k = obj.get(c["term"])
                if isinstance(k, str) and k.lower() in KINDS:
                    kinds[c["term"]] = k.lower()
        print(f"[gemma] kinds {len(kinds)}/{i + len(chunk)} after {time.time() - t0:.0f}s", file=sys.stderr)
        result["kinds"] = kinds
        save()

    # Job A: proposals from sampled short lines, grounded in the batch text.
    counts = {}
    sources = {}
    for src in ("claude", "grok", "codex", "repos"):
        path = od / "extract" / f"{src}.txt"
        if not path.exists():
            continue
        lines = sample_lines(path, per_source, rng)
        for i in range(0, len(lines), batch):
            if time.time() > deadline:
                break
            chunk = lines[i:i + batch]
            text = "\n".join(f"- {ln[:300]}" for ln in chunk)
            out = _gen(model, tok, PROMPT_A.format(lines=text), 400)
            result["prompts"] += 1
            arr = _parse(out, list)
            if not arr:
                continue
            blob = norm(" ".join(chunk))
            for t in arr:
                if not isinstance(t, str) or not (2 <= len(t) <= 40):
                    continue
                if norm(t) and norm(t) in blob:
                    counts[t] = counts.get(t, 0) + 1
                    sources.setdefault(t, set()).add(src)
        print(f"[gemma] proposals {len(counts)} after {src} ({time.time() - t0:.0f}s)", file=sys.stderr)
        result["proposals"] = [{"term": t, "count": n, "sources": sorted(sources[t])}
                               for t, n in sorted(counts.items(), key=lambda kv: -kv[1])]
        save()
    finished = time.time() <= deadline
    save(final=finished)
    print(f"[gemma] {'finished' if finished else 'TIME-BOXED, partial'}: {result['prompts']} prompts, "
          f"{len(kinds)} kinds, {len(counts)} proposals in {time.time() - t0:.0f}s", file=sys.stderr)
    return result
