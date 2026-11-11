"""Build the static shortlist (LISTS.json for shortlist_gguf.py) from PUBLIC training rows, never owner data.

Rows of the static list: forced rows (every special added token, every single-byte token, every token that decodes to
ASCII digits or ASCII punctuation, with an optional leading space), then regular-vocabulary ids ordered by target
count (descending), then input (transcript) count (descending), then BPE rank (id ascending), until exactly K rows.
K counts every static row. This is the rule the shipped 32k list was built with (the target-count order alone filled
only about 28.4k distinct ids, hence the input-count and BPE-rank fill).

usage: uv run --no-project --with numpy --with tokenizers python -I shortlist_lists.py TOKENIZER.json OUT.json \
           ROWS.jsonl [ROWS.jsonl ...] [--k 16000,32000,48000] [--n-vocab 128000]
  ROWS.jsonl: one JSON object per line with "input" (raw transcript) and "target" (corrected text) strings
  OUT.json: {"K": [ids...]} for every K, ids sorted ascending
"""
import argparse
import json
import re

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

ap = argparse.ArgumentParser()
ap.add_argument("tokenizer")
ap.add_argument("out")
ap.add_argument("rows", nargs="+")
ap.add_argument("--k", default="16000,32000,48000")
ap.add_argument("--n-vocab", type=int, default=128000)
a = ap.parse_args()
KS = [int(x) for x in a.k.split(",")]
N_VOCAB = a.n_vocab

targets, inputs = [], []
for path in a.rows:
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            targets.append(r["target"])
            inputs.append(r["input"])

tk = Tokenizer.from_file(a.tokenizer)


def count(texts):
    c = np.zeros(N_VOCAB, dtype=np.int64)
    for i in range(0, len(texts), 4096):
        for e in tk.encode_batch(texts[i:i + 4096], add_special_tokens=False):
            np.add.at(c, np.array(e.ids, dtype=np.int64), 1)
    return c


cnt, icnt = count(targets), count(inputs)

tj = json.load(open(a.tokenizer))
dec = ByteLevel()
forced = {t["id"] for t in tj["added_tokens"] if t["special"]}
digit_re = re.compile(r"^ ?[0-9]+$")
punct_re = re.compile(r"^ ?[!-/:-@\[-`{-~]+$")
for s, i in tj["model"]["vocab"].items():
    txt = dec.decode([s])
    if len(txt.encode("utf-8", "surrogatepass")) == 1 or len(s) == 1 or digit_re.match(txt) or punct_re.match(txt):
        forced.add(i)
n_regular = len(tj["model"]["vocab"])  # ids >= this are added tokens or padding rows
idx = np.arange(N_VOCAB)
order = [int(i) for i in np.lexsort((idx, -icnt, -cnt)) if i < n_regular]

lists = {}
for K in KS:
    s = set(forced)
    for i in order:
        if len(s) >= K:
            break
        s.add(i)
    assert len(s) == K, (K, len(s))
    lists[str(K)] = sorted(s)
json.dump(lists, open(a.out, "w"))
tot = cnt.sum()
k999 = int(np.searchsorted(np.cumsum(np.sort(cnt)[::-1]), 0.999 * tot) + 1)
print(f"rows {len(targets)}, target tokens {int(tot)}, distinct target ids {int((cnt > 0).sum())}, "
      f"forced {len(forced)}, K for 99.9% target coverage {k999}, lists {[len(v) for v in lists.values()]}")
