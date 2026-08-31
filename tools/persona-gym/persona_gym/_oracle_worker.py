"""Runs under: HF_HUB_OFFLINE=1 uv run --offline --python 3.12 --with parakeet-mlx==0.5.2

Reuses profile-miner's oracle (say two voices -> sox -> Parakeet) unchanged.
argv: terms.json out.json workdir
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "profile-miner"))


def main():
    terms_path, out_path, workdir = sys.argv[1:4]
    workdir = Path(workdir)
    (workdir / "oracle").mkdir(parents=True, exist_ok=True)
    os.environ["PHONON_MINER_OUT"] = str(workdir)  # oracle tmpdir lives here
    from profile_miner import oracle

    with open(terms_path) as f:
        terms = json.load(f)
    cache = oracle.run(terms, cache_path=workdir / "oracle" / "cache.jsonl", workers=6)
    out = {}
    for t in terms:
        diff, forms = oracle.summarize(t, cache.get(t, {}))
        out[t] = {"diff": diff, "forms": forms}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
