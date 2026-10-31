"""Draft proposers for draft-verified greedy decoding of the refiner, and a pure-Python trace of
the verification loop. No MLX here: the simulator and the tests import this module alone.

A proposer is called as propose(out_so_far, k) and returns up to k draft tokens; the engine forwards
[last accepted token] + draft in one pass and keeps the longest prefix the model agrees with.
"""

from __future__ import annotations


class DraftProposer:
    """Prompt-lookup style proposer over one or more source token sequences.

    Keeps a pointer into the preferred source; when the output diverges, re-aligns by matching
    the last n output tokens against each source (nearest occurrence after the last pointer).
    """

    def __init__(
        self, sources: list[list[int]], ngram: int = 3, substitute: bool = False
    ):
        self.sources = [s for s in sources if s]
        self.ngram = ngram
        self.substitute = substitute
        self.ptr: tuple[int, int] | None = (
            None  # (source index, position of next token)
        )
        self.out_len = 0

    def _align(self, out: list[int]) -> tuple[int, int] | None:
        for n in range(min(self.ngram, len(out)), 0, -1):
            tail = out[-n:]
            for si, src in enumerate(self.sources):
                start_min = 0
                if self.ptr is not None and self.ptr[0] == si:
                    start_min = max(0, self.ptr[1] - 8)
                best = None
                for p in range(start_min, len(src) - n + 1):
                    if src[p : p + n] == tail:
                        best = p + n
                        break
                if best is None and start_min > 0:
                    for p in range(len(src) - n + 1):
                        if src[p : p + n] == tail:
                            best = p + n
                            break
                if best is not None and best < len(src):
                    return (si, best)
            if n == 1 and len(out) > 1:
                break
        return None

    def initial(self, k: int) -> list[int]:
        """Opening block before any output exists: the primary source's first k tokens."""
        if not self.sources:
            return []
        self.ptr = (0, 0)
        self.out_len = 0
        return self.sources[0][:k]

    def __call__(self, out: list[int], k: int) -> list[int]:
        if self.ptr is not None:
            si, p = self.ptr
            # Advance the pointer by what was accepted since the last call if it matches.
            src = self.sources[si]
            moved = len(out) - self.out_len
            cand = p + moved
            if moved > 0 and cand <= len(src) and src[p:cand] == out[self.out_len :]:
                self.ptr = (si, cand)
            else:
                aligned = self._align(out)
                if aligned is None and self.substitute and si == 0 and cand < len(src):
                    # No n-gram anchor yet right after an edit: assume the model replaced as
                    # many draft tokens as it produced and keep following the draft.
                    aligned = (si, cand)
                self.ptr = aligned
        else:
            self.ptr = self._align(out) if out else None
        self.out_len = len(out)
        if self.ptr is None:
            return []
        si, p = self.ptr
        return self.sources[si][p : p + k]


class AdaptiveProposer(DraftProposer):
    """Draft length follows recent acceptance along a ladder of pass sizes that are cheap on the
    quantized kernels (a pass of 4 tokens costs about one of 1; 16 to 32 cost the same).

    After a rejection the next draft is the first rung; each fully accepted draft moves one rung
    up, capped by `cap_primary` while the pointer is in a refiner draft (source 0 when `primary`)
    and by `cap_lookup` while it is in transcript text."""

    LADDER = (3, 31, 63, 127, 255, 511)

    def __init__(
        self,
        sources: list[list[int]],
        primary: bool,
        ngram: int = 3,
        first: int = 63,
        cap_primary: int = 255,
        cap_lookup: int = 31,
        primary_reset_rung: int = 0,
        leap: bool = False,
        leap_rung: int = 5,
        substitute: bool = True,
        ladder: tuple[int, ...] | None = None,
    ):
        super().__init__(sources, ngram, substitute=substitute and leap and primary)
        if ladder:  # per backend: pass sizes that cost about the same as one token
            self.LADDER = tuple(ladder)
        self.primary = primary and bool(self.sources)
        # leap: inside the refiner draft, a block accepted in full right after an edit means
        # the draft is back in step, so the next block jumps to rung `leap_rung`
        self.leap = leap
        self.leap_rung = leap_rung
        self.first = first
        self.cap_primary = cap_primary
        self.cap_lookup = cap_lookup
        # rung to restart from after a rejection when the pointer is back in the refiner draft
        # (a refiner draft usually differs from the final by a local edit, so a long block
        # right after the edit is still a good bet)
        self.primary_reset_rung = primary_reset_rung
        self.rung = 0
        self.last_prop = 0
        self.last_len = 0

    def initial(self, k: int) -> list[int]:
        if not self.primary:
            return []
        d = super().initial(min(k, self.first))
        self.last_prop, self.last_len = len(d), 0
        self.rung = max(0, sum(1 for r in self.LADDER if r <= len(d)) - 1)
        return d

    def __call__(self, out: list[int], k: int) -> list[int]:
        rejected = False
        full = False
        if self.last_prop:
            accepted = len(out) - self.last_len - 1
            if accepted >= self.last_prop:
                self.rung = min(self.rung + 1, len(self.LADDER) - 1)
                full = True
            else:
                self.rung = 0
                rejected = True
        d = super().__call__(out, k)
        in_primary = self.primary and self.ptr is not None and self.ptr[0] == 0
        if rejected and in_primary:
            self.rung = self.primary_reset_rung
        if full and in_primary and self.leap:
            self.rung = max(self.rung, min(self.leap_rung, len(self.LADDER) - 1))
        cap = self.cap_primary if in_primary else self.cap_lookup
        d = d[: min(self.LADDER[self.rung], cap, k)]
        self.last_prop, self.last_len = len(d), len(out)
        return d


def trace_speculative(
    tail_len: int,
    target: list[int],
    propose,
    block: int,
    max_tokens: int,
    fuse: bool,
    eos: int = -1,
) -> dict:
    """Pure-Python replay of Engine.speculative (stream_refiner.py) when the greedy output
    `target` is known: the forward pass sizes (tokens per pass) and acceptance counts, so a
    simulator can price a policy without running the model. Mirrors the engine loop step for step.
    """

    def model_at(i: int) -> int:
        return target[i] if i < len(target) else eos

    info = {"passes": [], "drafted": 0, "accepted": 0}
    first = (
        propose.initial(block)[:max_tokens]
        if fuse and hasattr(propose, "initial")
        else []
    )
    if not first:
        info["passes"].append(max(1, tail_len))
        out: list[int] = []
        tok = model_at(0)
    else:
        info["passes"].append(max(1, tail_len) + len(first))
        info["drafted"] += len(first)
        j = 0
        while j < len(first) and model_at(j) == first[j] and first[j] != eos:
            j += 1
        info["accepted"] += j
        out = list(first[:j])
        tok = model_at(j)
        if len(out) >= max_tokens:
            return info
    while True:
        if tok == eos or len(out) >= max_tokens:
            break
        out.append(tok)
        if len(out) >= max_tokens:
            break
        draft = propose(out, block)[: max(0, max_tokens - len(out))]
        info["passes"].append(1 + len(draft))
        info["drafted"] += len(draft)
        j = 0
        base = len(out)
        while j < len(draft) and model_at(base + j) == draft[j] and draft[j] != eos:
            j += 1
        info["accepted"] += j
        out.extend(draft[:j])
        if len(out) >= max_tokens:
            break
        tok = model_at(len(out))
    assert out == target[: len(out)], "trace diverged from target"
    return info
