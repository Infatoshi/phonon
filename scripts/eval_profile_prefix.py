#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Measure the speaker profile block in the correction prompt.

Runs `phonon corpus polish-eval` (the shipped path: dictionary retrieval,
Gemma sidecar, output guard) once per prompt configuration and scores the
results against intended transcripts: word error rate, exact matches, time to
first token and total latency with the prefix cache warm.

Configurations:
  baseline  no profile, whole prompt prefilled every call (the pre-cache path)
  cache     no profile, stable prefix served from the prefix cache
  profile   profile block within its token budget, prefix cache on
  full      profile block unbounded (--full), to see what the budget costs

Cases: every corpus recording with an intended transcript, plus the fixture
files (hand-written raw/intended pairs). `--unlabeled` adds recordings without
an intended transcript; those only show where the configurations disagree.

Example:
  scripts/eval_profile_prefix.py --limit 60 --unlabeled
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = [
    ROOT / "fixtures" / "correction_cases.json",
    ROOT / "fixtures" / "behaviour_cases.json",
]
MODES = {
    "baseline": {"PHONON_PROFILE_PREFIX": "0", "PHONON_POLISH_PREFIX_CACHE": "0"},
    "cache": {"PHONON_PROFILE_PREFIX": "0"},
    "profile": {"PHONON_PROFILE_PREFIX": "1"},
    "full": {"PHONON_PROFILE_PREFIX": "1", "PHONON_PROFILE_TOKEN_BUDGET": "100000"},
}


def comparable_words(text: str) -> list[str]:
    """Same normalization as the Rust `comparable`: alphanumerics, lowercase."""
    kept = "".join(c if c.isalnum() or c.isspace() else "" for c in text)
    return kept.lower().split()


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (word != other),
                )
            )
        previous = current
    return previous[-1]


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def run_mode(phonon: Path, mode: str, args: argparse.Namespace) -> dict:
    command = [
        str(phonon),
        "corpus",
        "polish-eval",
        "--json",
        "--passes",
        str(args.passes),
    ]
    for fixture in args.fixtures:
        command += ["--fixtures", str(fixture)]
    if args.unlabeled:
        command.append("--unlabeled")
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    env = dict(os.environ)
    env.update(MODES[mode])
    env.setdefault("PHONON_ROOT", str(ROOT))
    if args.budget is not None and mode == "profile":
        env["PHONON_PROFILE_TOKEN_BUDGET"] = str(args.budget)
    print(f"== {mode}: {' '.join(command)}", file=sys.stderr)
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"{mode} failed with exit code {result.returncode}")
    return json.loads(result.stdout)


def score(report: dict) -> dict:
    labeled = [case for case in report["cases"] if case.get("intended") is not None]
    ref_words = 0
    edits = 0
    exact = 0
    for case in labeled:
        reference = comparable_words(case["intended"])
        hypothesis = comparable_words(case["output"])
        ref_words += len(reference)
        edits += edit_distance(hypothesis, reference)
        exact += reference == hypothesis
    last = [case["passes"][-1] for case in report["cases"] if case["passes"]]
    return {
        "cases": len(report["cases"]),
        "labeled": len(labeled),
        "exact": exact,
        "wer": edits / ref_words if ref_words else float("nan"),
        "ttft_ms": median([p["ttft_ms"] for p in last]),
        "latency_ms": median([p["latency_ms"] for p in last]),
        "wall_ms": median([p["wall_ms"] for p in last]),
        "prompt_tokens": median([p["prompt_tokens"] for p in last]),
        "cached_tokens": median([p["cached_prompt_tokens"] for p in last]),
        "profile_tokens": report["profile_tokens"],
        "ready": report["ready_message"],
    }


def table(scores: dict[str, dict]) -> str:
    header = (
        "| mode | cases | labeled | exact | WER | ttft ms | total ms | wall ms | "
        "prompt tok | cached tok | profile tok |"
    )
    rows = [header, "|" + "---|" * 11]
    for mode, s in scores.items():
        rows.append(
            f"| {mode} | {s['cases']} | {s['labeled']} | {s['exact']} | {s['wer']:.3f} | "
            f"{s['ttft_ms']:.0f} | {s['latency_ms']:.0f} | {s['wall_ms']:.0f} | "
            f"{s['prompt_tokens']:.0f} | {s['cached_tokens']:.0f} | {s['profile_tokens']} |"
        )
    return "\n".join(rows)


def disagreements(reports: dict[str, dict], reference: str, limit: int) -> str:
    lines = []
    base = {case["id"]: case for case in reports[reference]["cases"]}
    for mode, report in reports.items():
        if mode == reference:
            continue
        changed = [
            (case, base[case["id"]])
            for case in report["cases"]
            if case["id"] in base and case["output"] != base[case["id"]]["output"]
        ]
        lines.append(
            f"\n{mode} vs {reference}: {len(changed)}/{len(report['cases'])} outputs differ"
        )
        for case, other in changed[:limit]:
            lines.append(f"- {case['id']} ({case['source']})")
            lines.append(f"    raw:        {case['raw']}")
            if case.get("intended") is not None:
                lines.append(f"    intended:   {case['intended']}")
            lines.append(f"    {reference:<10}: {other['output']}")
            lines.append(f"    {mode:<10}: {case['output']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--modes", default="baseline,cache,profile")
    parser.add_argument(
        "--full", action="store_true", help="also run the unbounded profile"
    )
    parser.add_argument("--fixtures", type=Path, nargs="*", default=DEFAULT_FIXTURES)
    parser.add_argument("--unlabeled", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument(
        "--budget", type=int, help="profile token budget for the profile mode"
    )
    parser.add_argument(
        "--show", type=int, default=12, help="disagreements to print per mode"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "target" / "eval_profile_prefix.json"
    )
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()

    profile = "release" if args.release else "debug"
    subprocess.run(
        ["cargo", "build", "-p", "phonon-cli"]
        + (["--release"] if args.release else []),
        cwd=ROOT,
        check=True,
    )
    phonon = ROOT / "target" / profile / "phonon"

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if args.full and "full" not in modes:
        modes.append("full")
    reports = {mode: run_mode(phonon, mode, args) for mode in modes}
    scores = {mode: score(report) for mode, report in reports.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"scores": scores, "reports": reports}, indent=2))

    for mode, s in scores.items():
        print(f"{mode}: {s['ready']}")
    print()
    print(table(scores))
    print(disagreements(reports, modes[0], args.show))
    print(f"\nfull results: {args.out}")


if __name__ == "__main__":
    main()
