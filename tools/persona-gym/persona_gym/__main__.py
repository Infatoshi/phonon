import argparse
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog="persona_gym",
                                description="Synthetic user machines for the vocabulary miner (dev tool)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="manufacture N personas with planted gold vocab")
    b.add_argument("--n", type=int, default=20)
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--no-oracle", action="store_true",
                   help="skip TTS->Parakeet manglings (no GPU / parakeet busy)")
    b.add_argument("--lines-min", type=int, default=300)
    b.add_argument("--lines-max", type=int, default=800)

    r = sub.add_parser("rollout", help="run a teacher model agentically over each persona")
    r.add_argument("--personas", required=True)
    r.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n-per", type=int, default=1)
    r.add_argument("--max-turns", type=int, default=24)

    g = sub.add_parser("grade", help="score rollouts against gold.json")
    g.add_argument("--personas", required=True)
    g.add_argument("--rollouts", required=True)

    a = p.parse_args(argv)
    if a.cmd == "build":
        from . import build
        build.build(a.n, Path(a.out), seed=a.seed, no_oracle=a.no_oracle,
                    lines_min=a.lines_min, lines_max=a.lines_max)
    elif a.cmd == "rollout":
        from . import rollout
        rollout.run(Path(a.personas), a.endpoint, a.model, Path(a.out),
                    n_per=a.n_per, max_turns=a.max_turns)
    elif a.cmd == "grade":
        from . import grade
        grade.run(Path(a.personas), Path(a.rollouts))


if __name__ == "__main__":
    sys.exit(main())
