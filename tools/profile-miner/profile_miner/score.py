"""Dev-only scoring against the held-out dictionary and corpus.

This is the only module that reads ~/Library/Application Support/Phonon.
Gold = used-or-starred dictionary entries that the ASR oracle mangles or that
the corpus shows mangled (raw vs final transcript). Reports recall @100/200/400/
all with spoken-form variants collapsed, the miss list, and a precision proxy.
"""

import glob
import json
import sys
from collections import Counter

from .common import PHONON_SUPPORT, norm, norm_words, out_dir, read_json, write_json
from . import oracle

POC = {"note": "previous POC (Opus miners, 2026-08-30, before re-basing)", "gold": 107,
       "used_recall_at_400": 0.47, "used_recall_full": 0.61}
MAX_N = 5


def grams(text):
    """Set of concatenated 1..MAX_N-grams of normalized words."""
    w = norm_words(text)
    out = set()
    for i in range(len(w)):
        s = ""
        for j in range(i, min(i + MAX_N, len(w))):
            s += w[j]
            out.add(s)
    return out


def forms(term, spoken):
    fs = {norm(term)} | {norm(s) for s in spoken}
    fs.discard("")
    return fs


def run(workers=12):
    od = out_dir()
    dict_path = PHONON_SUPPORT / "dictionary.json"
    entries = read_json(dict_path).get("entries", [])
    gold0 = [e for e in entries if e.get("usage_count", 0) > 0 or e.get("starred")]
    print(f"[score] dictionary {len(entries)} entries, used-or-starred {len(gold0)}", file=sys.stderr)

    # Corpus: raw vs final/intended transcripts.
    corpus = []
    for p in glob.glob(str(PHONON_SUPPORT / "Corpus" / "*" / "metadata.json")):
        try:
            m = read_json(p)
        except Exception:
            continue
        corpus.append(m)
    corpus_text = set()
    raw_sets, fin_sets = [], []
    for m in corpus:
        raw = m.get("raw_transcript") or ""
        fin = (m.get("intended_transcript") or "") + " " + (m.get("final_transcript") or "")
        r, f = grams(raw), grams(fin)
        raw_sets.append(r)
        fin_sets.append(f)
        corpus_text |= r | f
    print(f"[score] corpus {len(corpus)} recordings", file=sys.stderr)

    # Oracle on gold phrases (separate cache, never mixed with the miner's).
    # Collapse spoken-form variants onto their term: an entry with a `replacement`
    # is a mangled phrase for that replacement term.
    groups = {}
    for e in gold0:
        term = e.get("replacement") or e["phrase"]
        g = groups.setdefault(norm(term), {"phrase": term, "forms": set(), "variants": [], "starred": False,
                                             "usage_count": 0})
        g["forms"] |= forms(e["phrase"], e.get("spoken_forms", [])) | {norm(term)}
        g["variants"].append(e["phrase"])
        g["starred"] = g["starred"] or bool(e.get("starred"))
        g["usage_count"] += e.get("usage_count", 0)
    print(f"[score] gold terms after collapsing replacement variants: {len(groups)}", file=sys.stderr)
    phrases = sorted({g["phrase"] for g in groups.values()} | {v for g in groups.values() for v in g["variants"]})
    cache = oracle.run(phrases, cache_path=od / "score" / "oracle_cache.jsonl", workers=workers)
    gold = []
    reasons = Counter()
    for g in groups.values():
        diff, heard = oracle.summarize(g["phrase"], cache.get(g["phrase"], {}))
        gf = g["forms"]
        corpus_mangled = any(gf & f and not (gf & r) for r, f in zip(raw_sets, fin_sets))
        why = []
        if diff in ("phonetic", "format"):
            why.append(f"oracle:{diff}")
        elif diff == "case":
            why.append("oracle:case")
        if corpus_mangled:
            why.append("corpus")
        if why:
            reasons[",".join(why)] += 1
            gold.append({"phrase": g["phrase"], "forms": gf, "why": why, "heard": heard, "variants": g["variants"],
                         "starred": g["starred"], "usage_count": g["usage_count"]})
    gold_strict = [g for g in gold if any(w in ("oracle:phonetic", "oracle:format", "corpus") for w in g["why"])]
    print(f"[score] gold after oracle/corpus filter: {len(gold)} (strict, no case-only: {len(gold_strict)}); "
          f"reasons {dict(reasons)}", file=sys.stderr)

    cands = read_json(od / "mined" / "candidates.json")
    raw = read_json(od / "candidates" / "candidates_raw.json")
    cand_forms = [(c, forms(c["term"], c["spoken_forms"])) for c in cands]

    def recall(gold_list, k):
        hit = 0
        for g in gold_list:
            for c, cf in cand_forms[:k]:
                if cf & g["forms"]:
                    hit += 1
                    break
        return hit

    ks = [100, 200, 400, len(cands)]
    table = {}
    for name, gl in (("gold", gold), ("gold_strict", gold_strict)):
        table[name] = {"size": len(gl)}
        for k in ks:
            h = recall(gl, k)
            table[name][f"@{k}"] = {"hit": h, "recall": round(h / max(len(gl), 1), 3)}

    # Dictionary presence for precision proxy (all entries).
    dict_forms = set()
    for e in entries:
        dict_forms |= forms(e["phrase"], e.get("spoken_forms", []))
    prec = {}
    for k in (100, 400):
        n = 0
        for c, cf in cand_forms[:k]:
            if cf & dict_forms or cf & corpus_text:
                n += 1
        prec[f"@{k}"] = round(n / max(min(k, len(cands)), 1), 3)

    # Miss list with diagnosis.
    misses = []
    raw_by_forms = {}
    for i, c in enumerate(raw):
        raw_by_forms.setdefault(norm(c["term"]), (i + 1, c))
    ranked_pos = {}
    for c in cands:
        for f in forms(c["term"], c["spoken_forms"]):
            ranked_pos.setdefault(f, c["rank"])
    miss_keys = {}
    for g in gold:
        pos = min((ranked_pos[f] for f in g["forms"] if f in ranked_pos), default=None)
        if pos is not None:
            continue
        miss_keys[g["phrase"]] = g
    # Count occurrences in the extracts.
    ext_counts = {p: Counter() for p in miss_keys}
    if miss_keys:
        lookup = {}
        for p, g in miss_keys.items():
            for f in g["forms"]:
                lookup.setdefault(f, []).append(p)
        for src in ("claude", "codex", "grok", "repos"):
            path = od / "extract" / f"{src}.txt"
            if not path.exists():
                continue
            seen = set()
            with open(path) as fh:
                for line in fh:
                    if line in seen:
                        continue
                    seen.add(line)
                    for gsz in grams(line) & lookup.keys():
                        for p in lookup[gsz]:
                            ext_counts[p][src] += 1
    for p, g in miss_keys.items():
        cnt = ext_counts[p]
        rawhit = None
        for f in g["forms"]:
            if f in raw_by_forms:
                rawhit = raw_by_forms[f]
                break
        if sum(cnt.values()) == 0:
            diag = "not in extracts"
        elif rawhit is None:
            diag = "in extracts, no candidate rule fired or below min_count"
        else:
            rank, rc = rawhit
            heard = cache.get(rc["term"]) if rc["term"] in cache else None
            if rc["term"] in oracle.load_cache(od / "oracle" / "cache.jsonl"):
                diag = f"candidate (raw rank {rank}), oracle said same"
            else:
                diag = f"candidate (raw rank {rank}, count {rc['count']}), outside oracle budget"
        misses.append({"phrase": p, "why_gold": g["why"], "heard": g["heard"], "extract_counts": dict(cnt),
                       "diagnosis": diag, "starred": g["starred"], "usage_count": g["usage_count"]})
    misses.sort(key=lambda m: -sum(m["extract_counts"].values()))

    report = {"dictionary_entries": len(entries), "gold_entries_used_or_starred": len(gold0),
              "gold_before_filter": len(groups), "gold_after_filter": len(gold),
              "gold_strict": len(gold_strict), "filter_reasons": dict(reasons), "candidates": len(cands),
              "recall": table, "precision_proxy": prec, "misses": misses, "previous_poc": POC}
    write_json(od / "score" / "report.json", report)

    print("\n| list | gold | @100 | @200 | @400 | @all |\n|---|---|---|---|---|---|")
    for name in ("gold", "gold_strict"):
        t = table[name]
        print(f"| {name} | {t['size']} | " + " | ".join(f"{t[f'@{k}']['recall']:.2f} ({t[f'@{k}']['hit']})" for k in ks) + " |")
    print(f"\ngold: {len(gold0)} used-or-starred entries -> {len(groups)} terms after collapsing variants -> "
          f"{len(gold)} after oracle/corpus filter ({len(gold_strict)} without case-only)")
    print(f"precision proxy: @100 {prec['@100']}, @400 {prec['@400']}; candidates {len(cands)}")
    print(f"previous POC: used-recall {POC['used_recall_at_400']} @400, {POC['used_recall_full']} full on {POC['gold']} gold")
    print(f"\nmisses ({len(misses)}):")
    for m in misses:
        print(f"- {m['phrase']!r} heard={m['heard']} why={m['why_gold']} counts={m['extract_counts']} :: {m['diagnosis']}")
    return report
