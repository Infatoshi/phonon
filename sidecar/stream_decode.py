"""Draft-verified greedy decoding over a rollback-able model state, for any backend.

A backend holds the model state of a token sequence (KV and conv state) that can be cut back to
any length and extended by a forward pass that reports the greedy choice after each input row.
Two backends implement it: stream_refiner.Engine (MLX, in this process) and PeEngine below (the
native refiner engine's per-dictation session, `pe serve`, over its JSON-lines pipe).

The decode loop is the same for both, so the streaming policy (StreamSession) runs unchanged on
either. stream_draft.trace_speculative mirrors the loop step for step; keep them in sync.
No MLX import here: the native backend runs without loading the weights in Python.
"""

from __future__ import annotations

from collections.abc import Callable
from difflib import SequenceMatcher

try:  # imported as sidecar.stream_decode (tests) or as a top-level module (the sidecar)
    from .stream_draft import AdaptiveProposer
except ImportError:
    from stream_draft import AdaptiveProposer


class Aborted(Exception):
    """Raised between forward passes when the caller's should_abort() turns true."""


class EngineError(RuntimeError):
    """The backend refused a request (context cap, lost session, malformed reply)."""


class RollbackDecoder:
    """The decode loop. A backend implements:

    _truncate(n_keep): drop the model state of positions n_keep and later.
    _pass(toks, row0, tie_from, tie_margin) -> (ids, tie): forward toks after the state (the
        state then holds them) and return the greedy choice after each of rows row0..len-1. With
        tie_margin > 0 and tie_from not None, tie is the first row index (counted from row0, at
        or after tie_from) whose top-2 logit gap is under the margin, else None. A backend that
        does not see logits returns None.
    """

    def __init__(self, eos: set[int]):
        self.eos = set(eos)
        self.tokens: list[int] = []  # tokens whose state the backend holds
        self.next_id: int | None = None  # greedy choice after tokens[-1], when known
        # Accounting, read by benchmarks: forward passes and the tokens they covered.
        self.stats = {"forwards": 0, "forward_tokens": 0}

    # -- backend -------------------------------------------------------------
    def _truncate(self, n_keep: int) -> None:
        raise NotImplementedError

    def _pass(
        self, toks: list[int], row0: int, tie_from: int | None, tie_margin: float
    ) -> tuple[list[int], int | None]:
        raise NotImplementedError

    # -- state ---------------------------------------------------------------
    def rollback(self, n_keep: int) -> None:
        if len(self.tokens) - n_keep <= 0:
            return
        self._truncate(n_keep)
        del self.tokens[n_keep:]
        self.next_id = None

    def common_prefix(self, prompt: list[int]) -> int:
        lcp = 0
        n = min(len(prompt), len(self.tokens))
        while lcp < n and prompt[lcp] == self.tokens[lcp]:
            lcp += 1
        return lcp

    def forward_ids(
        self,
        toks: list[int],
        row0: int = 0,
        tie_from: int | None = None,
        tie_margin: float = 0.0,
    ) -> tuple[list[int], int | None]:
        self.next_id = None
        ids, tie = self._pass(toks, row0, tie_from, tie_margin)
        self.tokens.extend(toks)
        self.stats["forwards"] += 1
        self.stats["forward_tokens"] += len(toks)
        return ids, tie

    def extend(self, prompt: list[int]) -> int:
        """Make the state hold exactly `prompt`, with the greedy choice after it in next_id;
        returns how many tokens were forwarded."""
        lcp = self.common_prefix(prompt)
        if lcp == len(prompt) == len(self.tokens) and self.next_id is not None:
            return 0
        if lcp == len(prompt):
            lcp -= 1  # the choice after the last prompt token: forward it again
        self.rollback(lcp)
        new = prompt[lcp:]
        ids, _ = self.forward_ids(new, row0=len(new) - 1)
        self.next_id = ids[-1]
        return len(new)

    # -- decoding ------------------------------------------------------------
    def greedy(self, prompt: list[int], max_tokens: int) -> list[int]:
        """Plain one-token-at-a-time greedy decode (the reference path)."""
        self.extend(prompt)
        out: list[int] = []
        tok = self.next_id
        while True:
            if tok in self.eos or len(out) >= max_tokens:
                break
            out.append(tok)
            if len(out) >= max_tokens:
                break
            tok = self.forward_ids([tok])[0][-1]
        self.next_id = None
        return out

    def speculative(
        self,
        prompt: list[int],
        max_tokens: int,
        propose: Callable[[list[int], int], list[int]],
        block: int = 16,
        min_block: int = 0,
        fuse: bool = False,
        should_abort: Callable[[], bool] | None = None,
        tie_margin: float = 0.0,
    ) -> tuple[list[int], dict]:
        """Greedy decode where `propose(out_so_far, k)` supplies up to k draft tokens.

        Each step forwards [last accepted token] + draft and accepts the longest prefix the
        model agrees with, plus the model's own next token. Output equals greedy decoding.
        With `fuse`, the first forward carries the uncached prompt tail and the proposer's
        opening block together. `should_abort` is checked before every forward pass; when it
        returns true the call raises Aborted with the state holding a consistent prefix.
        With tie_margin, a drafted position whose top-2 gap is under it is re-decided by a
        one-token forward (a batched pass may flip such a near-tie against one-token decoding).
        """

        def check():
            if should_abort is not None and should_abort():
                self.next_id = None
                raise Aborted()

        def redecide(n_keep):
            """One-token forward of the last kept input, after rolling back to n_keep - 1."""
            last = self.tokens[n_keep - 1]
            self.rollback(n_keep - 1)
            info["tie_redecides"] = info.get("tie_redecides", 0) + 1
            return self.forward_ids([last])[0][-1]

        tie_on = tie_margin > 0
        check()
        info = {"steps": 0, "drafted": 0, "accepted": 0, "prefill": 0, "passes": []}
        first = propose.initial(block) if fuse and hasattr(propose, "initial") else []
        first = first[:max_tokens]
        if not first:
            info["prefill"] = self.extend(prompt)
            if info["prefill"]:
                info["passes"].append(info["prefill"])
            out: list[int] = []
            tok = self.next_id
        else:
            # Fused first pass: prompt tail + the draft's opening block in one forward.
            lcp = min(self.common_prefix(prompt), len(prompt) - 1)
            self.rollback(lcp)
            tail = prompt[lcp:]
            info["prefill"] = len(tail)
            # row 0 of the choices is the first output token, which one-token decoding also
            # takes from a batched prompt prefill, so near-ties are re-decided from row 1 on
            choice, tie = self.forward_ids(
                tail + first, len(tail) - 1, 1 if tie_on else None, tie_margin
            )
            info["steps"] += 1
            info["drafted"] += len(first)
            info["passes"].append(len(tail) + len(first))
            j = 0
            while (
                j < len(first)
                and (tie is None or j < tie)
                and choice[j] == first[j]
                and first[j] not in self.eos
            ):
                j += 1
            info["accepted"] += j
            self.rollback(lcp + len(tail) + j)
            out = list(first[:j])
            tok = (
                redecide(lcp + len(tail) + j)
                if tie is not None and j == tie
                else choice[j]
            )
            if len(out) >= max_tokens:
                self.next_id = None
                return out, info
        self.next_id = None
        while True:
            if tok in self.eos or len(out) >= max_tokens:
                break
            out.append(tok)
            if len(out) >= max_tokens:
                break
            draft = propose(out, block)[: max(0, max_tokens - len(out))]
            if len(draft) < min_block:
                draft = []
            check()
            base = len(self.tokens)
            choice, tie = self.forward_ids(
                [tok] + draft, 0, 0 if tie_on and draft else None, tie_margin
            )
            info["passes"].append(1 + len(draft))
            info["steps"] += 1
            info["drafted"] += len(draft)
            j = 0
            while (
                j < len(draft)
                and (tie is None or j < tie)
                and choice[j] == draft[j]
                and draft[j] not in self.eos
            ):
                j += 1
            info["accepted"] += j
            # Keep the state of tok + the accepted draft tokens.
            self.rollback(base + 1 + j)
            out.extend(draft[:j])
            if len(out) >= max_tokens:
                break
            # a pass of one token is already a one-token decode
            tok = redecide(base + 1 + j) if tie is not None and j == tie else choice[j]
        self.next_id = None
        return out, info


# Draft sizes for the native engine: forwards of 2 to 8 tokens all run its megakernel's 8-row
# verify shape (6.2-7.0 ms against 3.6-3.8 for one token, sessbench M4 Max), so the smallest
# draft after a rejection is 7, and 15 bridges to the llama.cpp passes (32 tokens: 17 ms + the
# per-row argmax). Simulated on 240 eval rows: >= 5 s p90 163.6 -> 150.5 ms (to be measured).
PE_LADDER = (7, 15, 31, 63, 127, 255, 511)


class PeEngine(RollbackDecoder):
    """The native refiner engine's session (`pe serve`, one live session per process) as a
    decode backend. `call(request) -> reply` sends one JSON request line and returns the reply
    (polish_server.RefinerEngine.call). The caller tokenizes; ids go over the pipe.

    Session ops used (pe.mm, serve): open {prefix ids} -> session; rollback {n} keeps positions
    0..n-1; forward {tokens, logits_from j} -> argmax ids after tokens[j..]. A one-shot prompt
    request to the same process ends the session, so `invalidate()` must follow every one-shot
    call; the next forward then opens a new session from the prefix. A session cannot be cut
    below its first position: rollback(0) drops the session instead, and every prompt must start
    with the prefix's first token.
    """

    def __init__(self, call: Callable[[dict], dict], prefix: list[int], eos: set[int]):
        super().__init__(eos)
        if not prefix:
            raise ValueError("a session needs a non-empty prefix")
        self.call = call
        self.prefix = list(prefix)
        self.session = 0
        self.stats.update(
            {"opens": 0, "llama_passes": 0, "mk_passes": 0, "engine_ms": 0.0}
        )

    def _ok(self, request: dict) -> dict:
        reply = self.call(request)
        if not reply.get("ok"):
            raise EngineError(f"refiner engine session: {reply}")
        return reply

    def open(self) -> None:
        """A fresh session holding the prefix (its state comes from the engine's prefix cache
        when the engine cached the same tokens)."""
        reply = self._ok({"op": "open", "prefix": self.prefix})
        self.session = int(reply["session"])
        self.tokens = list(self.prefix)
        self.next_id = None
        self.stats["opens"] += 1

    def invalidate(self) -> None:
        """The engine served a one-shot request: its session is gone."""
        self.session = 0
        self.tokens = []
        self.next_id = None

    def close(self) -> None:
        if self.session:
            self.call({"op": "close", "session": self.session})
        self.invalidate()

    def _truncate(self, n_keep: int) -> None:
        if not self.session:
            return  # nothing held; the next forward opens a session
        if n_keep < 1:
            self.session = 0  # start over: the next forward opens a new session
            return
        self._ok({"op": "rollback", "session": self.session, "n": n_keep})

    def _pass(self, toks, row0, tie_from, tie_margin):
        if self.session:
            return self._forward_raw(list(toks), row0), None
        # No session: open one (it holds the prefix), cut it back to where the sequence leaves
        # the prefix or to the first row whose choice is wanted, and forward the rest.
        held = self.tokens + list(toks)
        first_row = (
            len(held) - len(toks) + row0
        )  # absolute position of the first wanted row
        self.open()
        keep = min(self.common_prefix(held), first_row)
        if keep < 1:
            self.invalidate()
            raise EngineError("prompt does not start like the session prefix")
        if keep < len(self.tokens):
            self._ok({"op": "rollback", "session": self.session, "n": keep})
        self.tokens = held[:keep]
        ids = self._forward_raw(held[keep:], first_row - keep)
        self.tokens = held[: len(held) - len(toks)]  # forward_ids appends toks
        return ids, None

    def _forward_raw(self, toks: list[int], row0: int) -> list[int]:
        reply = self.call(
            {
                "op": "forward",
                "session": self.session,
                "tokens": toks,
                "logits_from": row0,
            }
        )
        if not reply.get("ok"):
            if reply.get("cancelled"):
                # the processed part stays appended: resynchronise, then report an abort
                n = int(reply.get("n", len(self.tokens)))
                self.tokens = (self.tokens + toks)[:n]
                raise Aborted()
            raise EngineError(f"refiner engine session forward: {reply}")
        ids = [int(i) for i in reply["ids"]]
        if len(ids) != len(toks) - row0:
            raise EngineError(
                f"engine returned {len(ids)} ids for {len(toks) - row0} rows"
            )
        self.stats["llama_passes" if reply.get("path") == "llama" else "mk_passes"] += 1
        self.stats["engine_ms"] += float(reply.get("ms", 0.0))
        return ids


def _revised(old: list[int], new: list[int]) -> int:
    """Tokens of `old` that `new` no longer contains (in order)."""
    sm = SequenceMatcher(a=old, b=new, autojunk=False)
    return len(old) - sum(m.size for m in sm.get_matching_blocks())


class StreamSession:
    """Refine while the speaker is still talking (lossless policy), on any RollbackDecoder.

    commit(text): the transcript so far grew by a closed segment. Re-refines the whole transcript
    so far, drafting from the previous draft and the transcript, and keeps the prompt prefix up to
    the end of the transcript in the model state.
    final(text): at key release, refines the final transcript with the latest draft. The prompt
    tail after the cached prefix is prefilled in the same forward as the draft's opening block, and
    the rest of the draft is verified in large blocks. The result is the model's greedy output for
    the final prompt whatever the draft says.

    `prompter` supplies prompt_tokens(text), open_prompt_tokens(text), budget(text) and
    text_tokens(text) for one request (vocabulary already bound, or carried in `text`).

    Long dictations: a prompter that also has fits(text), words(text) and with_words(text,
    words) lets the session freeze a head. When the transcript's prompt plus its output budget
    would pass the backend's context, the session refines the transcript's older sentences
    (all but the last KEEP_WORDS words) once, keeps that output, and from then on streams only
    the words after them. The answer is the frozen outputs then the tail's output (`pieces`).
    """

    KEEP_WORDS = 80  # words a freeze leaves streaming (about 30 s of speech)
    # A tail longer than this freezes too, so key up refines at most this many words: on the
    # owner's 6.2 min dictation a tail of about 400 words took 761 ms at key up.
    FREEZE_WORDS = 200

    def __init__(
        self,
        engine: RollbackDecoder,
        prompter,
        first: int = 511,
        cap_primary: int = 511,
        cap_lookup: int = 31,
        redraft_every: int = 1,
        prefill: bool = True,
        commit_cap: int = 127,
        leap: bool = False,
        leap_rung: int = 5,
        reset_rung: int = 0,
        ladder: tuple[int, ...] | None = None,
    ):
        """first/cap_primary: draft tokens in the release pass and in later blocks; commit_cap
        bounds the passes of background jobs, so a release that aborts one waits at most one
        short pass (simulator: p90 >= 5 s 174.8 -> 169.7 ms at 127). ladder: draft sizes after
        a rejection, smallest first (None: AdaptiveProposer.LADDER; the native engine's verify
        shape takes 8 rows for the price of 2, see PE_LADDER)."""
        self.e = engine
        self.p = prompter
        self.first, self.cap_primary, self.cap_lookup = first, cap_primary, cap_lookup
        self.redraft_every, self.prefill = redraft_every, prefill
        self.commit_cap = commit_cap
        self.leap, self.leap_rung, self.reset_rung = leap, leap_rung, reset_rung
        self.ladder = ladder
        self.draft: list[int] | None = None
        self.draft_prompt: list[int] | None = None  # the prompt the draft answers
        self.draft_text: str | None = None  # and its request (the tail of a long one)
        self.commits = 0
        # speculative work thrown away: cached prompt tokens rolled back because the transcript
        # was revised, and draft tokens a later draft no longer contains
        self.waste = {"rolled_back": 0, "draft_revised": 0, "forwarded": 0}
        self.head: list[str] = []  # transcript words frozen so far
        self.head_out: list[list[int]] = []  # their outputs, one per freeze

    def reset(self, prompter=None) -> None:
        if prompter is not None:
            self.p = prompter
        self.draft = None
        self.draft_prompt = None
        self.draft_text = None
        self.commits = 0
        self.waste = {"rolled_back": 0, "draft_revised": 0, "forwarded": 0}
        self.head = []
        self.head_out = []

    def pieces(self, tail: list[int] | None) -> list[list[int]]:
        """The answer in parts: each frozen head's output, then the tail's."""
        return self.head_out + ([tail] if tail is not None else [])

    # -- long dictations -----------------------------------------------------
    def _freezes(self) -> bool:
        return all(hasattr(self.p, m) for m in ("fits", "words", "with_words"))

    @staticmethod
    def _norm(word: str) -> str:
        return "".join(c for c in word.lower() if c.isalnum())

    def _tail(self, text: str) -> str:
        """The request for the words after the frozen head. A transcript whose start no longer
        matches the head (a deep revision) unfreezes it."""
        if not self.head:
            return text
        words = self.p.words(text)
        n = len(self.head)
        if [self._norm(w) for w in words[:n]] != [self._norm(w) for w in self.head]:
            self.head, self.head_out = [], []
            self.draft = self.draft_prompt = self.draft_text = None
            return text
        return self.p.with_words(text, words[n:])

    def _split(self, text: str) -> int:
        """Words of `text` to freeze: up to the last sentence end that leaves KEEP_WORDS
        streaming and whose request fits; 0 when no such split exists."""
        words = self.p.words(text)
        last = len(words) - self.KEEP_WORDS
        ends = [i + 1 for i in range(last) if words[i][-1:] in ".?!"]
        for k in reversed(ends or [last]):
            if k > 0 and self.p.fits(self.p.with_words(text, words[:k])):
                return k
        k = last
        while k > 0 and not self.p.fits(self.p.with_words(text, words[:k])):
            k //= 2
        return max(k, 0)

    def _fit(self, text: str, should_abort=None) -> str:
        """Freeze heads until the request for the rest fits; returns that request."""
        tail = self._tail(text)
        while self._freezes() and (
            not self.p.fits(tail) or len(self.p.words(tail)) > self.FREEZE_WORDS
        ):
            k = self._split(tail)
            if k <= 0:
                break
            words = self.p.words(tail)
            head = self.p.with_words(tail, words[:k])
            # the previous draft drafts it: that draft opens with the head's answer
            out, info = self.e.speculative(
                self.p.prompt_tokens(head),
                self.p.budget(head),
                self._proposer(head, self.draft, background=True),
                block=512,
                fuse=bool(self.draft),
                should_abort=should_abort,
            )
            self.waste["forwarded"] += sum(info.get("passes", []))
            self.head += words[:k]
            self.head_out.append(out)
            self.draft = self.draft_prompt = self.draft_text = None  # it answered more
            tail = self.p.with_words(tail, words[k:])
        return tail

    def _proposer(
        self, text: str, primary: list[int] | None, background: bool = False
    ) -> AdaptiveProposer:
        src = self.p.text_tokens(text)
        sources = ([primary] if primary else []) + [src]
        first, cap = self.first, self.cap_primary
        if background and self.commit_cap:
            first, cap = min(first, self.commit_cap), min(cap, self.commit_cap)
        return AdaptiveProposer(
            sources,
            primary=bool(primary),
            first=first,
            cap_primary=cap,
            cap_lookup=self.cap_lookup,
            primary_reset_rung=self.reset_rung,
            leap=self.leap,
            leap_rung=self.leap_rung,
            ladder=self.ladder,
        )

    def _rolled_back(self, prompt: list[int]) -> int:
        """Cached prompt tokens a job for `prompt` throws away (the transcript was revised)."""
        return len(self.e.tokens) - self.e.common_prefix(prompt)

    def commit(self, text: str, should_abort=None) -> dict:
        self.commits += 1
        text = self._fit(text, should_abort)
        op = self.p.open_prompt_tokens(text)
        if self.redraft_every and self.commits % self.redraft_every == 0:
            fp = self.p.prompt_tokens(text)
            if self.draft is not None and fp == self.draft_prompt:
                return {"passes": [], "unchanged": True}  # the draft already answers it
            self.waste["rolled_back"] += self._rolled_back(fp)
            out, info = self.e.speculative(
                fp,
                self.p.budget(text),
                self._proposer(text, self.draft, background=True),
                block=512,
                fuse=bool(self.draft),
                should_abort=should_abort,
            )
            if self.draft:
                self.waste["draft_revised"] += _revised(self.draft, out)
            self.waste["forwarded"] += sum(info.get("passes", []))
            self.draft = out
            self.draft_prompt = fp
            self.draft_text = text
            keep = 0
            while keep < min(len(op), len(fp)) and op[keep] == fp[keep]:
                keep += 1
            self.e.rollback(keep)
            return info
        if self.prefill:
            if should_abort is not None and should_abort():
                raise Aborted()
            n = self.e.extend(op)
            return {"passes": [n] if n else []}
        return {"passes": []}

    def final(self, text: str) -> tuple[list[int], dict]:
        """The tail's output; the whole answer is pieces(out)."""
        text = self._fit(text)
        fp = self.p.prompt_tokens(text)
        if self.draft is not None and fp == self.draft_prompt:
            # the last background job refined exactly this prompt (the speaker stopped before
            # the last round): its output is the answer, no forward pass needed
            info = {"passes": [], "reused_draft": True}
            info["waste"] = dict(self.waste, final_tokens=len(self.draft))
            return list(self.draft), info
        self.waste["rolled_back"] += self._rolled_back(fp)
        out, info = self.e.speculative(
            fp,
            self.p.budget(text),
            self._proposer(text, self.draft),
            block=512,
            fuse=bool(self.draft),
        )
        if self.draft:
            self.waste["draft_revised"] += _revised(self.draft, out)
        info["waste"] = dict(self.waste, final_tokens=len(out))
        return out, info
