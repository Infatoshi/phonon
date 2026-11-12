"""Summarise shortlist_eval output (debug mode). usage: shortlist_eval_report.py OUT.jsonl
Prints: dictations token-identical to the full head, step fallback rate, missed steps, bit-identity checks, extra-row
overflows, and the class of the word holding each dictation's first missed token (classes and order as the offline
part2.py: punctuation, number, code identifier, non-English, name or term, rare word). Public items only."""
import collections
import json
import re
import sys


def wclass(w):
    if not re.search(r"[A-Za-z0-9]", w) and not re.search(r"[^\x00-\x7f]", w):
        return "punctuation"
    if re.search(r"\d", w):
        return "number"
    core = w.strip(".,;:!?\"'()[]")
    if "_" in core or re.search(r"[a-z][A-Z]", core) or re.search(r"\w[./()`]\w", core):
        return "code identifier"
    if re.search(r"[^\x00-\x7f]", w):
        return "non-English"
    if core[:1].isupper():
        return "name or term"
    return "rare word"


rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n = len(rows)
ok = [r for r in rows if r["ok"]]
steps = sum(r["steps"] for r in ok)
fb = sum(r["fallback"] for r in ok)
missed = sum(r["missed"] for r in ok)
diff = [r for r in ok if r["missed"] > 0]
checked = [r for r in ok if "logit_bitdiff" in r]
classes = collections.Counter()
examples = []
for r in diff:
    m = r["misses"][0]
    t, off = r["full_text"], m["off"]
    s = off + (len(m["full"]) - len(m["full"].lstrip()))
    a = t.rfind(" ", 0, s) + 1
    b = t.find(" ", s)
    word = t[a:b if b >= 0 else len(t)] or m["full"]
    classes[wclass(word)] += 1
    examples.append((r["id"], word, m["full"], m["sl"], round(m["best"], 2)))
print(f"dictations {n} (decode ok {len(ok)}); steps {steps}; median steps {sorted(r['steps'] for r in ok)[len(ok)//2]}")
print(f"token-identical to the full head: {len(ok) - len(diff)} of {len(ok)} ({100 * (len(ok) - len(diff)) / len(ok):.2f}%); "
      f"differing {len(diff)} ({100 * len(diff) / len(ok):.2f}%)")
print(f"fallback: {fb} of {steps} steps ({100 * fb / steps:.2f}%); dictations with >=1 fallback "
      f"{sum(r['fallback'] > 0 for r in ok)} ({100 * sum(r['fallback'] > 0 for r in ok) / len(ok):.1f}%)")
print(f"missed steps (no fallback, shortlist choice != full argmax): {missed} ({100 * missed / steps:.4f}% of steps)")
print(f"extra rows: overflow (all steps fall back) {sum(r['extra_groups'] < 0 for r in ok)}; "
      f"groups median {sorted(r['extra_groups'] for r in ok)[len(ok)//2]} max {max(r['extra_groups'] for r in ok)}; "
      f"with vocabulary line {sum(r['has_vocab'] for r in ok)}")
if checked:
    print(f"bit-identity vs the stock tied head: {len(checked)} dictations, {sum(r['steps'] for r in checked)} steps x 128,000 "
          f"logits; differing logits {sum(r['logit_bitdiff'] for r in checked)}; token sequences differing "
          f"{sum(r['seq_vs_tied_first_diff'] >= 0 for r in checked)}")
print("first-missed-token classes:", dict(classes))
for e in examples:
    print("  ", e)
