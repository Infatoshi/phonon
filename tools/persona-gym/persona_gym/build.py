"""Subcommand: build. Manufacture N synthetic user machines with planted gold."""

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from . import chatlog, fabricate, repos, vocab
from .common import load_dict_words, write_json

ORACLE_SAMPLE = 5
ORACLE_TIMEOUT = 900


def run_oracle(sampled: list[str], out_dir: Path) -> tuple[dict, str]:
    """Batch TTS->Parakeet over sampled terms via profile-miner's oracle.

    Returns ({term: {diff, forms}}, note). Empty dict + note on failure.
    """
    if not sampled:
        return {}, "no terms sampled"
    work = out_dir / ".cache" / "oracle"
    work.mkdir(parents=True, exist_ok=True)
    terms_path = work / "terms.json"
    result_path = work / "result.json"
    with open(terms_path, "w") as f:
        json.dump(sampled, f)
    pkg_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    cmd = ["uv", "run", "--offline", "--python", "3.12", "--with", "parakeet-mlx==0.5.2",
           "python", "-m", "persona_gym._oracle_worker",
           str(terms_path), str(result_path), str(work)]
    try:
        r = subprocess.run(cmd, cwd=pkg_root, env=env, capture_output=True, check=False,
                           text=True, timeout=ORACLE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {}, f"oracle skipped: timeout after {ORACLE_TIMEOUT}s (parakeet busy?)"
    except FileNotFoundError:
        return {}, "oracle skipped: uv not found"
    if r.returncode != 0:
        tail = (r.stderr or "")[-500:]
        return {}, f"oracle skipped: worker failed: {tail}"
    with open(result_path) as f:
        return json.load(f), "ok"


def build(n: int, out: Path, seed: int = 0, no_oracle: bool = False,
          lines_min: int = 300, lines_max: int = 800) -> None:
    out = out.resolve()
    meta_root = out / "meta"
    cache = out / ".cache" / "repos"
    out.mkdir(parents=True, exist_ok=True)
    words = load_dict_words()
    taken: set[str] = set()
    master = random.Random(seed)

    pool = list(repos.CURATED)
    master.shuffle(pool)
    failed: set[str] = set()
    personas = []

    for i in range(n):
        rng = random.Random(f"{seed}:{i}")
        name = f"persona-{i + 1:02d}"
        pdir = out / name
        if pdir.exists():
            shutil.rmtree(pdir)
        (pdir / "repos").mkdir(parents=True)

        # 2-4 repos, domain-diverse, verified by cloning; skip dead URLs.
        want = rng.randint(2, 4)
        chosen = []
        domains_used = set()
        offset = rng.randrange(len(pool))
        for j in range(len(pool)):
            slug, domain = pool[(offset + j) % len(pool)]
            if len(chosen) >= want:
                break
            if slug in failed or domain in domains_used:
                continue
            cached = repos.clone_into_cache(slug, cache)
            if cached is None:
                failed.add(slug)
                continue
            rname = slug.split("/")[1]
            shutil.copytree(cached, pdir / "repos" / rname)
            chosen.append({"name": rname, "slug": slug, "url": f"https://github.com/{slug}",
                           "domain": domain})
            domains_used.add(domain)
        if len(chosen) < 2:
            raise RuntimeError(f"{name}: could not clone 2 repos (network?)")

        # Real vocabulary and file names from the cloned repos.
        rv_counts = vocab.repo_vocab(pdir / "repos", words)
        rvocab = [t for t, _ in rv_counts.most_common(60)][:40]
        repo_files = []
        for p in vocab.iter_text_files(pdir / "repos"):
            repo_files.append(str(p.relative_to(pdir)))
        rng.shuffle(repo_files)
        repo_files = repo_files[:30]

        terms = fabricate.gen_terms(rng, words, taken)
        lb = chatlog.LogBuilder(rng, terms, [c["name"] for c in chosen], repo_files, rvocab)
        lines = lb.build(rng.randint(lines_min, lines_max))
        sample = rng.sample([t["term"] for t in terms], min(ORACLE_SAMPLE, len(terms)))
        personas.append({"name": name, "dir": pdir, "terms": terms, "lines": lines,
                         "sample": sample, "repos": chosen, "rng": rng,
                         "rvocab": rvocab[:20]})
        print(f"[build] {name}: {len(chosen)} repos {[c['name'] for c in chosen]}, "
              f"{len(terms)} terms, {len(lines)} lines", file=sys.stderr)

    # One oracle batch for all personas (model loads once).
    oracle_note = "skipped: --no-oracle"
    oracle_out: dict = {}
    if not no_oracle:
        all_sampled = sorted({t for p in personas for t in p["sample"]})
        print(f"[build] oracle over {len(all_sampled)} sampled terms", file=sys.stderr)
        oracle_out, oracle_note = run_oracle(all_sampled, out)
        if oracle_note != "ok":
            print(f"[build] {oracle_note}", file=sys.stderr)

    for p in personas:
        rng = p["rng"]
        lines = p["lines"]
        manglings = []
        for term in p["sample"]:
            res = oracle_out.get(term)
            if not res or res["diff"] == "same":
                continue
            lines, injected = chatlog.inject_manglings(rng, lines, term, res["forms"])
            for form, cnt in injected.items():
                manglings.append({"term": term, "mangled": form,
                                  "diff": res["diff"], "count": cnt})

        # Split lines into claude.txt / codex.txt.
        logs = p["dir"] / "logs"
        logs.mkdir(exist_ok=True)
        cut = int(len(lines) * 0.6)
        (logs / "claude.txt").write_text("\n".join(lines[:cut]) + "\n")
        (logs / "codex.txt").write_text("\n".join(lines[cut:]) + "\n")

        planted = [{"term": t["term"], "kind": t["kind"],
                    "count": chatlog.count_ci(lines, t["term"])} for t in p["terms"]]
        gold = {
            "persona": p["name"],
            "planted": planted,
            "manglings": manglings,
            "oracle_sample": p["sample"],
            "oracle_note": oracle_note,
            "repos": p["repos"],
            "repo_vocab_top": p["rvocab"],
            "seed": seed,
        }
        write_json(meta_root / p["name"] / "gold.json", gold)
        size_mb = repos.tree_bytes(p["dir"]) / 1e6
        print(f"[build] {p['name']}: {size_mb:.1f} MB, {len(manglings)} manglings injected, "
              f"sample terms {p['sample'][:3]}", file=sys.stderr)

    print(f"[build] done: {len(personas)} personas in {out} (gold in {meta_root})",
          file=sys.stderr)
