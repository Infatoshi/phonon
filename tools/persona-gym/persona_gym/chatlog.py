"""Synthetic developer chat logs: templates + randomization, no LLM.

Output matches profile-miner extract format: one plain-text line per
user-authored message, files named claude.txt and codex.txt.
"""

import random
import re

FILLERS = ["uh", "um", "like", "basically", "kinda", "you know", "i mean", "so yeah"]

# Slots: {project} {machine} {person} {model} {tool} {repo} {file} {rv} (repo vocab)
TEMPLATES = {
    "voicetyped": [
        "ok so the {model} run on {machine} is {f} still crashing can you look at {file}",
        "um can we rename {project} to something shorter its {f} a mouthful",
        "hey {f} check what {person} pushed to {repo} yesterday i think it broke {rv}",
        "the {rv} thing in {repo} is {f} eating memory again on {machine}",
        "run the {project} bench on {machine} tonight {f} before {person} wakes up",
        "i want {tool} to just tail the {model} logs {f} instead of polling",
        "so {f} the {model} eval numbers look wrong compare against {rv}",
        "can you {f} grep {repo} for {rv} and tell me where its set",
        "ping {person} about the {project} deadline {f} its slipping again",
        "move the {project} checkpoints off {machine} the disk is {f} full",
        "why does {tool} keep {f} choking on the {file} parser",
        "lets {f} retrain {model} with the new {rv} settings from {repo}",
        "note to self {f} {project} needs the {rv} fix before the demo",
        "the {machine} box is {f} thermal throttling during {model} decode",
    ],
    "terse": [
        "fix {file}",
        "{project}: bench {rv} path again",
        "push {model} eval to {machine}",
        "ask {person} re {repo} {rv} regression",
        "{tool} v2: drop the {rv} shim",
        "rebase {project} onto {repo} changes",
        "{machine} disk 92%, prune {model} ckpts",
        "grep {repo} for {rv}, patch, test",
        "{person} owns {project} rollout now",
        "wire {tool} into the {project} ci",
        "{model} wer up 2pts after {rv} change, revert",
        "clone {repo}, port {rv} into {project}",
    ],
    "verbose": [
        "I went through the {rv} implementation in {repo} and I think {project} should adopt the same approach before we ship the next build to {machine}.",
        "Could you compare the {model} outputs against the baseline? {person} claims the {rv} change in {repo} explains the regression but I am not convinced.",
        "Please schedule the long {project} sweep on {machine} for tonight, and make sure {tool} captures the {rv} counters this time.",
        "After reading {file} I believe the bug {person} reported is in the {rv} path, not in {project} itself.",
        "The plan for this week: finish the {tool} refactor, rerun the {model} evaluation on {machine}, and write up the {project} notes for {person}.",
        "When {model} finishes decoding on {machine}, copy the transcripts into the {project} folder and diff them against the {rv} fixtures from {repo}.",
        "I renamed the staging box to {machine} and moved the {project} artifacts there; {person} has the credentials if you need them.",
        "The {rv} constant in {file} looks hand-tuned; ask {person} whether {project} depends on that exact value before changing it.",
    ],
}

WS = re.compile(r"\s+")


def one_line(s: str) -> str:
    return WS.sub(" ", s).strip()


class LogBuilder:
    def __init__(self, rng: random.Random, terms: list[dict], repo_names: list[str],
                 repo_files: list[str], rvocab: list[str]):
        self.rng = rng
        self.by_kind = {k: [t["term"] for t in terms if t["kind"] == k]
                        for k in ("project", "machine", "person", "model", "tool")}
        self.repo_names = repo_names or ["scratch"]
        self.repo_files = repo_files or ["Makefile"]
        self.rvocab = rvocab or ["main"]
        # Zipf-ish per-term weights so counts vary.
        self.weights = {}
        for terms_k in self.by_kind.values():
            for t in terms_k:
                self.weights[t] = self.rng.choice([1, 1, 1, 2, 2, 3, 4, 6, 9])
        w = [0.25, 0.45, 0.30]
        kinds = ["voicetyped", "terse", "verbose"]
        self.rng.shuffle(w)
        self.style_w = dict(zip(kinds, w))

    def _pick(self, kind: str) -> str:
        pool = self.by_kind.get(kind) or ["it"]
        ws = [self.weights.get(t, 1) for t in pool]
        return self.rng.choices(pool, weights=ws, k=1)[0]

    def line(self, style: str | None = None) -> str:
        rng = self.rng
        style = style or rng.choices(list(self.style_w), weights=list(self.style_w.values()), k=1)[0]
        tpl = rng.choice(TEMPLATES[style])
        s = tpl.format(
            project=self._pick("project"), machine=self._pick("machine"),
            person=self._pick("person"), model=self._pick("model"),
            tool=self._pick("tool"), repo=rng.choice(self.repo_names),
            file=rng.choice(self.repo_files), rv=rng.choice(self.rvocab),
            f=rng.choice(FILLERS),
        )
        if style == "voicetyped":
            s = s.lower()
        return one_line(s)

    def build(self, n_lines: int) -> list[str]:
        lines = [self.line() for _ in range(n_lines)]
        # Every fabricated term appears at least once, verbatim.
        text_low = "\n".join(lines).lower()
        for kind, pool in self.by_kind.items():
            for t in pool:
                if t.lower() not in text_low:
                    lines.append(one_line(
                        f"reminder: {t} is our {kind} name, keep it out of public docs"))
        self.rng.shuffle(lines)
        return lines


def count_ci(lines: list[str], phrase: str) -> int:
    text = "\n".join(lines).lower()
    return text.count(phrase.lower()) if phrase else 0


def inject_manglings(rng: random.Random, lines: list[str], term: str,
                     forms: list[str], max_lines: int = 3) -> tuple[list[str], dict]:
    """Replace the term with a spoken-form mangling in a few lines.

    Returns (lines, {mangled_form: injected_count}).
    """
    injected: dict = {}
    for form in forms:
        f = one_line(form)
        if not f or f.lower() == term.lower():
            continue
        idx = [i for i, ln in enumerate(lines) if term.lower() in ln.lower()]
        rng.shuffle(idx)
        n = 0
        for i in idx[: rng.randint(1, max_lines)]:
            pat = re.compile(re.escape(term), re.IGNORECASE)
            lines[i] = pat.sub(f.lower(), lines[i], count=1)
            n += 1
        if n == 0:  # no line left with the term; add a fresh voice-typed one
            lines.append(one_line(f"uh the {f.lower()} run is done can you check the numbers"))
            n = 1
        injected[f] = injected.get(f, 0) + n
    return lines, injected
