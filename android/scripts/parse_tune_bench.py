"""Parse RefinerTuneBenchTest logcat (TB lines) into per-arm medians and a token-identity check.

usage: python android/scripts/parse_tune_bench.py LOGCAT [LOGCAT...]
Per arm: median total ms over its REFINE lines (the 8 dictations), median of rep0 and rep1, head_cached count,
and the tok_hash per (mode, rep). Identity: every (mode, rep) hash equal across all arms of a file; the script
exits 1 when any differs. The per-mode head cache (2026-11-12) was checked this way with arms
"auto/heads1,auto,auto/heads1,auto" (heads1 = the old warm-up, minimal head only).
"""
import json
import re
import statistics
import sys

LINE = re.compile(r"TB REFINE arm=(\d+) name=(\S+) mode=(\w+) rep=(\d) total_ms=(\d+) .*? (\{.*\})\s*$")
DEV = re.compile(r"TB DEVICE model=(.*?) device=")
WARM = re.compile(r"TB WARMUP arm=(\d+) total_ms=(\d+)")


def parse(path):
    arms = {}
    device = "?"
    warm = {}
    for line in open(path, errors="replace"):
        m = DEV.search(line)
        if m:
            device = m.group(1)
        m = WARM.search(line)
        if m:
            warm[int(m.group(1))] = int(m.group(2))
        m = LINE.search(line)
        if not m:
            continue
        arm, name, mode, rep, total, js = m.groups()
        try:
            o = json.loads(js)
        except ValueError:
            continue
        a = arms.setdefault(int(arm), {"name": name, "rows": []})
        a["rows"].append(
            {
                "mode": mode,
                "rep": int(rep),
                "total": int(total),
                "pre": o["prefill_ms"],
                "dec": o["decode_ms"],
                "gen": o["gen_tokens"],
                "cached": o["head_cached"],
                "hash": o["tok_hash"],
                "k": o.get("k"),
            }
        )
    return device, arms, warm


def main():
    failed = False
    for path in sys.argv[1:]:
        device, arms, warm = parse(path)
        print(f"== {path}  device={device}")
        hashes = {}
        for i, a in sorted(arms.items()):
            rows = a["rows"]
            tot = [r["total"] for r in rows]
            r0 = [r["total"] for r in rows if r["rep"] == 0]
            r1 = [r["total"] for r in rows if r["rep"] == 1]
            ms_tok = [r["dec"] / r["gen"] for r in rows if r["gen"]]
            print(
                f"arm {i} {a['name']:<14} n={len(rows)} median={statistics.median(tot):.0f} "
                f"rep0={statistics.median(r0) if r0 else float('nan'):.0f} "
                f"rep1={statistics.median(r1) if r1 else float('nan'):.0f} "
                f"cached={sum(r['cached'] for r in rows)}/{len(rows)} "
                f"ms_per_tok={statistics.median(ms_tok):.1f} k={rows[0]['k']} warm_ms={warm.get(i)}"
            )
            for r in rows:
                hashes.setdefault((r["mode"], r["rep"]), set()).add(r["hash"])
        bad = {k: v for k, v in hashes.items() if len(v) > 1}
        print(f"token identity across arms: {'OK' if not bad else 'DIFF ' + str(bad)} ({len(hashes)} (mode,rep) keys)")
        failed |= bool(bad) or not hashes
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
