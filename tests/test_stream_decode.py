"""The shared decode loop (stream_decode) on a fake model, through a plain in-process backend and
through PeEngine speaking the native engine's session protocol to a fake `pe serve`."""

import pytest

from sidecar.stream_decode import (
    Aborted,
    EngineError,
    PeEngine,
    RollbackDecoder,
    StreamSession,
)
from sidecar.stream_draft import AdaptiveProposer, trace_speculative

EOS = 99
PREFIX = [1, 11, 12, 13, 14, 15]  # the system prompt's tokens (cached by the engine)
SUFFIX = [2, 3]  # end of the user turn and the assistant header


def model_next(seq: list[int]) -> int:
    """The fake model's greedy choice after seq: echoes the user's words (tokens >= 20) in
    order after the assistant header, with every word w > 60 rewritten to w - 30, then EOS."""
    if SUFFIX[-1] not in seq:
        return (seq[-1] * 7 + len(seq)) % 40 + 20  # anything inside the prompt
    head = len(seq) - 1 - seq[::-1].index(SUFFIX[-1])
    words = [t for t in seq[len(PREFIX) : head] if t >= 20]
    i = len(seq) - head - 1
    if i >= len(words):
        return EOS
    return words[i] - 30 if words[i] > 60 else words[i]


class Local(RollbackDecoder):
    """In-process backend: the state is the token list itself."""

    def __init__(self):
        super().__init__({EOS})
        self.pass_sizes = []

    def _truncate(self, n_keep):
        pass

    def _pass(self, toks, row0, tie_from, tie_margin):
        self.pass_sizes.append(len(toks))
        seq = self.tokens + list(toks)
        base = len(self.tokens)
        return [model_next(seq[: base + r + 1]) for r in range(row0, len(toks))], None


class FakePe:
    """The session part of `pe serve` (pe.mm): one live session, ids in and out."""

    CTX = 400

    def __init__(self):
        self.sid = 0
        self.toks = []
        self.log = []

    def __call__(self, req):
        self.log.append(req)
        op = req.get("op")
        if op is None:  # a one-shot prompt request ends the session
            self.sid = 0
            return {"ok": True, "text": "", "tokens": 0, "prompt_tokens": 0}
        if op == "open":
            self.sid += 1
            self.toks = list(req["prefix"])
            return {"ok": True, "session": self.sid, "n": len(self.toks), "ms": 0.1}
        if req.get("session") != self.sid or self.sid == 0:
            return {"ok": False, "error": "no such session"}
        if op == "close":
            self.sid = 0
            return {"ok": True}
        if op == "rollback":
            n = req["n"]
            if n < 1 or n > len(self.toks):
                return {"ok": False, "error": "bad rollback length"}
            del self.toks[n:]
            return {"ok": True, "n": n}
        if op == "forward":
            t, j = req["tokens"], req["logits_from"]
            if len(self.toks) + len(t) > self.CTX:
                return {"ok": False, "error": "context cap reached"}
            base = len(self.toks)
            self.toks += t
            ids = [model_next(self.toks[: base + r + 1]) for r in range(j, len(t))]
            path = "mk" if len(t) <= 8 else "llama"
            return {
                "ok": True,
                "ids": ids,
                "n": len(self.toks),
                "path": path,
                "ms": 1.0,
            }
        return {"ok": False, "error": "unknown op"}


def prompt(words):
    return PREFIX + list(words) + SUFFIX


def reference(words, budget=200):
    return Local().greedy(prompt(words), budget)


WORDS = [21, 35, 64, 22, 70, 41, 90, 33, 25, 61]


def test_fake_model_echoes_and_rewrites():
    assert reference(WORDS) == [21, 35, 34, 22, 40, 41, 60, 33, 25, 31]


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("draft_kind", ["perfect", "wrong", "none"])
def test_speculative_equals_greedy_and_its_trace(fuse, draft_kind):
    target = reference(WORDS)
    draft = {"perfect": target, "wrong": WORDS, "none": None}[draft_kind]
    sources = ([draft] if draft else []) + [WORDS]

    def proposer():
        return AdaptiveProposer(sources, primary=bool(draft), first=63)

    e = Local()
    e.extend(PREFIX)
    e.pass_sizes.clear()
    out, info = e.speculative(prompt(WORDS), 200, proposer(), block=512, fuse=fuse)
    assert out == target
    tail = len(prompt(WORDS)) - len(PREFIX)
    trace = trace_speculative(tail, target, proposer(), 512, 200, fuse=fuse, eos=EOS)
    assert info["passes"] == trace["passes"] == e.pass_sizes
    assert info["accepted"] == trace["accepted"]


def test_extend_reuses_the_common_prefix():
    e = Local()
    assert e.extend(prompt(WORDS)) == len(prompt(WORDS))
    assert e.extend(prompt(WORDS)) == 0  # held, with the choice after it
    revised = WORDS[:6] + [50, 51]
    assert e.extend(prompt(revised)) == len(revised) - 6 + len(SUFFIX)


def test_pe_engine_decodes_like_the_local_backend():
    pe = FakePe()
    e = PeEngine(pe, PREFIX, {EOS})
    target = reference(WORDS)
    assert e.greedy(prompt(WORDS), 200) == target
    out, _info = e.speculative(
        prompt(WORDS), 200, AdaptiveProposer([WORDS], primary=False), block=512
    )
    assert out == target
    assert pe.toks == e.tokens
    assert e.stats["opens"] == 1
    assert e.stats["llama_passes"] + e.stats["mk_passes"] == e.stats["forwards"]


def test_pe_engine_reopens_after_a_one_shot_request():
    pe = FakePe()
    e = PeEngine(pe, PREFIX, {EOS})
    e.extend(prompt(WORDS[:4]))
    # the sidecar's prewarm, a one-shot request, ends the session
    pe({"prompt": "warmup pass", "max_tokens": 8})
    e.invalidate()
    out, _ = e.speculative(
        prompt(WORDS),
        200,
        AdaptiveProposer([reference(WORDS)], primary=True),
        fuse=True,
    )
    assert out == reference(WORDS)
    # opened lazily by the first forward, and again by the first one after the reset
    assert e.stats["opens"] == 2
    assert pe.toks == e.tokens


def test_pe_engine_restarts_from_the_prefix_after_a_full_rollback():
    pe = FakePe()
    e = PeEngine(pe, PREFIX, {EOS})
    # the prefix alone: the open restores it, the last prefix token is forwarded again for
    # the choice after it
    e.extend(PREFIX)
    assert [r["op"] for r in pe.log] == ["open", "rollback", "forward"]
    assert pe.log[-1]["tokens"] == PREFIX[-1:]
    assert e.next_id == model_next(PREFIX)
    e.extend(prompt(WORDS))
    e.rollback(0)  # a session keeps its first position: this drops it instead
    assert e.session == 0 and e.tokens == []
    assert e.greedy(prompt(WORDS), 200) == reference(WORDS)
    assert pe.toks == e.tokens and e.stats["opens"] == 2


def test_pe_engine_refuses_a_prompt_unlike_the_prefix():
    e = PeEngine(FakePe(), PREFIX, {EOS})
    e.open()
    with pytest.raises(EngineError):
        e.extend([5] + WORDS)
    assert e.session == 0


def test_pe_engine_reports_the_context_cap():
    pe = FakePe()
    pe.CTX = 20
    e = PeEngine(pe, PREFIX, {EOS})
    with pytest.raises(EngineError):
        e.greedy(prompt(WORDS), 200)


class Prompter:
    """Envelope text "21 35 64" -> token ids, as the sidecar's EnvelopePrompter does."""

    @staticmethod
    def words(text):
        return [int(w) for w in text.split()]

    def prompt_tokens(self, text):
        return prompt(self.words(text))

    def open_prompt_tokens(self, text):
        return PREFIX + self.words(text)

    def budget(self, text):
        return 200

    def text_tokens(self, text):
        return self.words(text)


def text(words):
    return " ".join(str(w) for w in words)


@pytest.mark.parametrize("backend", ["local", "pe"])
def test_stream_session_final_is_the_greedy_output_of_the_final_prompt(backend):
    e = Local() if backend == "local" else PeEngine(FakePe(), PREFIX, {EOS})
    s = StreamSession(e, Prompter())
    # the transcript grows, and the ASR revises a word between commits
    s.commit(text(WORDS[:3]))
    s.commit(text(WORDS[:5]))
    s.commit(text(WORDS[:4] + [50, 41]))
    final = WORDS[:4] + [50, 41, 90, 33]
    out, info = s.final(text(final))
    assert out == reference(final)
    assert info["waste"]["draft_revised"] > 0  # the revised word changed the draft
    # a release right after the last commit reuses its draft without a forward pass
    s.reset()
    s.commit(text(final))
    out2, info2 = s.final(text(final))
    assert out2 == reference(final) and info2.get("reused_draft")


def test_stream_session_abort_keeps_a_consistent_state():
    pe = FakePe()
    e = PeEngine(pe, PREFIX, {EOS})
    s = StreamSession(e, Prompter(), commit_cap=3)
    s.commit(text(WORDS[:6]))
    calls = {"n": 0}

    def should_abort():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(Aborted):
        s.commit(text(WORDS), should_abort=should_abort)
    assert pe.toks == e.tokens
    out, _ = s.final(text(WORDS))
    assert out == reference(WORDS)
