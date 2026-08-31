"""Subcommand: grade. Score rollouts against planted gold."""

import json
import sys
from pathlib import Path

from . import vocab
from .common import load_dict_words, norm, read_json

PASS_RECALL = 0.7
PASS_PRECISION = 0.6


def emitted_forms(item: dict) -> set[str]:
    forms = {norm(str(item.get("term", "")))}
    for s in item.get("spoken_forms") or []:
        if isinstance(s, str):
            forms.add(norm(s))
    forms.discard("")
    return forms


def valid_answer(answer) -> bool:
    return (isinstance(answer, list) and len(answer) > 0
            and all(isinstance(i, dict) and isinstance(i.get("term"), str) for i in answer))


def persona_text(pdir: Path) -> str:
    """All text in the persona tree, for the verbatim-plausibility check."""
    parts = []
    for p in vocab.iter_text_files(pdir):
        parts.append(vocab.read_text(p))
    return "\n".join(parts)


def score_rollout(answer, gold: dict, rvocab_norms: set[str], tree_text: str) -> dict:
    valid = valid_answer(answer)
    items = answer if valid else []
    all_emitted: set[str] = set()
    per_item = [emitted_forms(i) for i in items]
    for f in per_item:
        all_emitted |= f

    # A planted term is recalled when any emitted term/spoken form matches the
    # term or one of its injected manglings by normalized form.
    mangle_by_term: dict[str, set[str]] = {}
    for m in gold["manglings"]:
        mangle_by_term.setdefault(norm(m["term"]), set()).add(norm(m["mangled"]))
    planted_hits = 0
    missed = []
    for g in gold["planted"]:
        gset = {norm(g["term"])} | mangle_by_term.get(norm(g["term"]), set())
        if gset & all_emitted:
            planted_hits += 1
        else:
            missed.append(g["term"])
    n_planted = len(gold["planted"])
    recall = planted_hits / n_planted if n_planted else 0.0

    mangles = {norm(m["mangled"]) for m in gold["manglings"]} - {""}
    mangle_hits = len(mangles & all_emitted)
    mangle_recall = mangle_hits / len(mangles) if mangles else None

    gold_norms = {norm(g["term"]) for g in gold["planted"]}
    for ms in mangle_by_term.values():
        gold_norms |= ms
    good = 0
    stray = []
    for item, forms in zip(items, per_item):
        term = item.get("term", "")
        if forms & gold_norms or norm(term) in rvocab_norms or term and term in tree_text:
            good += 1
        else:
            stray.append(term)
    precision = good / len(items) if items else 0.0

    passed = bool(valid and recall >= PASS_RECALL and precision >= PASS_PRECISION)
    return {
        "valid_json": valid, "n_emitted": len(items),
        "planted_recall": round(recall, 3), "planted_hits": planted_hits,
        "n_planted": n_planted,
        "mangling_recall": None if mangle_recall is None else round(mangle_recall, 3),
        "n_manglings": len(mangles),
        "precision_proxy": round(precision, 3),
        "missed_planted": missed, "stray_terms": stray[:20],
        "pass": passed,
    }


def run(personas: Path, rollouts: Path) -> None:
    personas = personas.resolve()
    rollouts = rollouts.resolve()
    words = load_dict_words()
    rows = []
    report = {}
    for gold_path in sorted(personas.glob("meta/*/gold.json")):
        gold = read_json(gold_path)
        name = gold["persona"]
        pdir = personas / name
        rdirs = sorted((rollouts / name).glob("r*")) if (rollouts / name).exists() else []
        if not rdirs:
            print(f"[grade] {name}: no rollouts", file=sys.stderr)
            continue
        rvocab_norms = {norm(t) for t in vocab.repo_vocab(pdir / "repos", words)}
        tree_text = persona_text(pdir)
        for rdir in rdirs:
            try:
                answer = read_json(rdir / "answer.json")
            except (OSError, json.JSONDecodeError):
                answer = None
            meta = {}
            try:
                meta = read_json(rdir / "meta.json")
            except (OSError, json.JSONDecodeError):
                pass
            s = score_rollout(answer, gold, rvocab_norms, tree_text)
            s["turns"] = meta.get("turns")
            s["usage"] = meta.get("usage")
            report.setdefault(name, {})[rdir.name] = s
            rows.append((name, rdir.name, s))

    out_path = rollouts / "report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    hdr = f"{'persona':<12} {'roll':<5} {'recall':>7} {'mangle':>7} {'prec':>6} {'terms':>6} {'turns':>6} {'valid':>6} {'pass':>5}"
    print(hdr)
    print("-" * len(hdr))
    for name, rname, s in rows:
        mg = "-" if s["mangling_recall"] is None else f"{s['mangling_recall']:.2f}"
        print(f"{name:<12} {rname:<5} {s['planted_recall']:>7.2f} {mg:>7} "
              f"{s['precision_proxy']:>6.2f} {s['n_emitted']:>6} {s['turns'] or '-'!s:>6} "
              f"{s['valid_json']!s:>6} {'PASS' if s['pass'] else 'fail':>5}")
    n_pass = sum(1 for _, _, s in rows if s["pass"])
    print(f"\n{n_pass}/{len(rows)} rollouts pass "
          f"(recall >= {PASS_RECALL}, precision >= {PASS_PRECISION}, valid JSON). "
          f"Report: {out_path}")
