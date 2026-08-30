import argparse
import sys

from . import candidates, extract, rank, seed


def main(argv=None):
    p = argparse.ArgumentParser(prog="profile_miner", description="Phonon vocabulary miner (dev tool)")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="stage 2: user-authored text per source")
    e.add_argument("--source", action="append", choices=list(extract.SOURCES))
    e.add_argument("--until", help="ISO timestamp; drop records at or after it (held-out cutoff)")
    s = sub.add_parser("seed", help="stage 3a: identity seeds")
    s.add_argument("--no-github", action="store_true")
    c = sub.add_parser("candidates", help="stage 3b: rule-based candidates")
    c.add_argument("--min-count", type=int, default=candidates.MIN_COUNT)
    o = sub.add_parser("oracle", help="stage 3c: TTS -> Parakeet oracle (needs parakeet-mlx)")
    o.add_argument("--workers", type=int, default=12)
    o.add_argument("--top", type=int, default=6000, help="oracle budget: top N by count and breadth")
    sub.add_parser("rank", help="write mined/candidates.json")
    g = sub.add_parser("gemma", help="optional stage 3d: local Gemma pass (needs mlx-lm)")
    g.add_argument("--minutes", type=float, default=20)
    g.add_argument("--top", type=int, default=400)
    sc = sub.add_parser("score", help="dev-only scoring against the held-out dictionary and corpus")
    sc.add_argument("--workers", type=int, default=12)
    a = p.parse_args(argv)
    if a.cmd == "extract":
        extract.run(a.source, a.until)
    elif a.cmd == "seed":
        seed.run(github=not a.no_github)
    elif a.cmd == "candidates":
        candidates.run(min_count=a.min_count)
    elif a.cmd == "oracle":
        from . import oracle
        oracle.run_candidates(workers=a.workers, top=a.top)
    elif a.cmd == "rank":
        rank.run()
    elif a.cmd == "gemma":
        from . import gemma_pass
        gemma_pass.run(minutes=a.minutes, top=a.top)
    elif a.cmd == "score":
        from . import score
        score.run(workers=a.workers)


if __name__ == "__main__":
    sys.exit(main())
