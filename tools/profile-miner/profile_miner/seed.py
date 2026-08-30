"""Stage 3a: identity seeds. No model."""

import os
import re
import subprocess
import sys

from .common import out_dir, read_json, write_json


def _cmd(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def run(github=True):
    seeds = {}
    full = _cmd(["id", "-F"])
    if full:
        seeds[full] = "id -F"
        for part in full.split():
            if len(part) > 1:
                seeds.setdefault(part, "id -F")
    gitname = _cmd(["git", "config", "--global", "user.name"])
    if gitname:
        seeds.setdefault(gitname, "git user.name")
        for part in gitname.split():
            if len(part) > 1:
                seeds.setdefault(part, "git user.name")
    user = os.environ.get("USER", "")
    if user:
        seeds.setdefault(user, "$USER")
    if github and _cmd(["gh", "auth", "status"]) != "" or github and _cmd(["gh", "auth", "token"]):
        login = _cmd(["gh", "api", "user", "-q", ".login"])
        if login and re.fullmatch(r"[A-Za-z0-9-]+", login):
            seeds.setdefault(login, "gh login")
    repos_path = out_dir() / "extract" / "repos.json"
    if repos_path.exists():
        for name in read_json(repos_path):
            if 2 <= len(name) <= 40 and not name.startswith("."):
                seeds.setdefault(name, "repo")
    write_json(out_dir() / "seed" / "seeds.json", seeds)
    by_kind = {}
    for k, v in seeds.items():
        by_kind[v] = by_kind.get(v, 0) + 1
    print(f"[seed] {len(seeds)} seeds {by_kind}", file=sys.stderr)
    return seeds
