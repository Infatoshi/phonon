"""MetalParakeet's helper protocol against a fake `pa serve`: the sticky fallback (a failed window, a helper that
does not answer, a helper that exits) and the self-test's window under `pa busy`."""

import os
import stat
import sys
import time
import types

import numpy as np
import pytest

from sidecar.pa_client import MetalParakeet

# a stand-in for `pa WEIGHTS serve MAXMEL` and `pa busy TGS SECS`: WEIGHTS is the behaviour (ok, fail, hang, exit);
# every request's header is appended to WEIGHTS.log, and both modes append to events.log (busy: start, stop)
FAKE = """#!{python}
import os, signal, struct, sys, time
events = open(os.path.join(os.path.dirname(sys.argv[0]), "events.log"), "a")
if sys.argv[1] == "busy":
    events.write("busy\\n"); events.flush()
    print("busy: running", flush=True)
    time.sleep(float(sys.argv[3]))
    sys.exit(0)
mode = sys.argv[1]
log = open(mode + ".log", "a")
inp, out = sys.stdin.buffer, sys.stdout.buffer
while True:
    h = inp.read(4)
    if len(h) < 4:
        break
    (n,) = struct.unpack("<I", h)
    if n == 0:
        break
    log.write(f"{{n}}\\n"); log.flush()
    if n == 0xFFFFFFFE:  # rescore: jobs -> (heard, cand) per job; WEIGHTS *rescore_exit: exits, *_far: cand far off
        (nj,) = struct.unpack("<I", inp.read(4))
        res = []
        for _ in range(nj):
            f0, f1 = struct.unpack("<2I", inp.read(8))
            lens = []
            for _ in range(3):
                (m,) = struct.unpack("<I", inp.read(4))
                inp.read(4 * m)
                lens.append(m)
            log.write(f"rescore {{f0}} {{f1}} {{lens[0]}} {{lens[1]}} {{lens[2]}}\\n"); log.flush()
            res += [-1.0, -100.0 if mode.endswith("_far") else -2.0]
        if mode.endswith("rescore_exit"):
            sys.exit(1)
        out.write(struct.pack(f"<{{len(res)}}f", *res)); out.flush()
        continue
    if n == 0xFFFFFFFF:  # a dictionary: words m, m words -> status (WEIGHTS "big": over capacity)
        (m,) = struct.unpack("<I", inp.read(4))
        inp.read(4 * m)
        log.write(f"trie {{m}}\\n"); log.flush()
        out.write(struct.pack("<I", 1 if mode.endswith("big") else 0)); out.flush()
        continue
    events.write("request\\n"); events.flush()
    words = n & 0x7FFFFFFF if n >> 31 else n * 128
    inp.read(4 * words)
    kind = mode.rsplit("/", 1)[-1]
    if kind == "hang":
        time.sleep(30)
    if kind == "hang_once":
        marker = mode + ".hung"
        if not os.path.exists(marker):
            open(marker, "w").write("1")
            time.sleep(30)
    if kind == "exit":
        sys.exit(1)
    if kind == "fail":
        out.write(struct.pack("<I", 0xFFFFFFFF))
    else:
        out.write(struct.pack("<I", 1) + struct.pack("<3If", 7, 1, 0, 0.5))
    out.write(struct.pack("<3f", 1.0, 2.0, 3.0) + struct.pack("<I", 0) + struct.pack("<2I", 0, 0))
    out.flush()
"""


@pytest.fixture
def fake_pa(tmp_path):
    path = tmp_path / "pa"
    path.write_text(FAKE.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def make(kind, **kw):
        return MetalParakeet(
            object(), weights=str(tmp_path / kind), pa=str(path), **kw
        ), tmp_path / f"{kind}.log"

    return make


def requests(log):
    if not os.path.exists(log):
        return []
    return [
        int(x)
        for x in log.read_text().split("\n")
        if x.strip() and not x.startswith(("trie", "rescore")) and int(x) != 0xFFFFFFFE
    ]


def rescores(log):
    if not os.path.exists(log):
        return []
    return [
        tuple(int(v) for v in x.split()[1:])
        for x in log.read_text().split("\n")
        if x.startswith("rescore")
    ]


def trie_sizes(log):
    if not os.path.exists(
        log
    ):  # the helper has not opened its log yet: no request reached it
        return []
    return [
        int(x.split()[1]) for x in log.read_text().split("\n") if x.startswith("trie")
    ]


SAMPLES = np.zeros(16000, dtype=np.float32)
FRAMES = (16000 + 272) // 160


def test_answers_while_the_helper_works(fake_pa):
    m, log = fake_pa("ok")
    for _ in range(3):
        assert m.metal_rows(SAMPLES, FRAMES) == [(7, 1, 0, 0.5)]
    assert not m.dead
    assert len(requests(log)) == 3
    m.close()


def test_a_failed_window_turns_the_helper_off_for_the_session(fake_pa, capsys):
    m, log = fake_pa("fail", max_restarts=0)
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert m.dead
    for _ in range(3):  # later windows go straight to MLX: they never reach the helper
        assert m.metal_rows(SAMPLES, FRAMES) is None
    assert len(requests(log)) == 1
    assert capsys.readouterr().err.count("Metal ASR off for this session") == 1
    assert m.p.poll() is not None  # stopped, not left holding GPU memory


def test_a_helper_that_does_not_answer_times_out_once(fake_pa):
    m, log = fake_pa("hang", timeout_s=0.5, max_restarts=0)
    t = time.monotonic()
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert time.monotonic() - t < 5
    assert m.dead
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert len(requests(log)) == 1


def test_a_helper_that_exits_is_off(fake_pa):
    m, _ = fake_pa("exit", max_restarts=0)
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert m.dead


def test_transient_timeout_recovers_on_next_window_and_reapplies_terms(tmp_path, fake_pa):
    path = tmp_path / "pa"
    m = MetalParakeet(
        toy_model(),
        weights=str(tmp_path / "hang_once"),
        pa=str(path),
        timeout_s=0.4,
        max_restarts=2,
        rescore=False,
    )
    m.set_terms(["vLLM", "cuBLAS"])
    # First window stalls -> falls back to MLX without latching Metal ASR off
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert not m.dead
    # Next window respawns pa, re-sends dictionary trie, and succeeds on Metal
    assert m.metal_rows(SAMPLES, FRAMES) == [(7, 1, 0, 0.5)]
    assert not m.dead
    assert len(trie_sizes(tmp_path / "hang_once.log")) == 2
    m.close()


def test_repeated_failures_exhaust_restarts_and_latch_off(fake_pa, capsys):
    m, log = fake_pa("exit", max_restarts=2)
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert not m.dead
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert not m.dead
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert m.dead
    # Fourth call never spawns again
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert len(requests(log)) == 3
    assert capsys.readouterr().err.count("Metal ASR off for this session") == 1


def test_over_capacity_is_mlx_but_keeps_the_helper(fake_pa):
    m, log = fake_pa("ok", max_mel=50)
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert not m.dead
    assert requests(log) == []
    m.close()


def test_self_test_runs_its_last_window_under_load(fake_pa, tmp_path):
    m, log = fake_pa("ok")
    t = time.monotonic()
    ok, ms = m.self_test()
    assert ok and ms == 3.0
    # the busy process is stopped, not waited out (it would run 10 s)
    assert time.monotonic() - t < 5
    # 1 s sample windows: cold, warm, then the load starts and the last one runs under it
    assert requests(log) == [0x80000000 | 16000] * 3
    events = (tmp_path / "events.log").read_text().split()
    assert events == ["request", "request", "busy", "request"]
    m.close()


def test_self_test_budgets(fake_pa):
    m, _ = fake_pa("ok")
    assert m.self_test(budget_ms=2.0)[0] is False  # the warm window (3 ms) over budget
    assert (
        m.self_test(load_budget_ms=2.0)[0] is False
    )  # the window under load over its budget
    assert m.self_test(load=False) == (True, 3.0)
    m.close()


def test_self_test_fails_when_the_helper_cannot_decode(fake_pa):
    m, _ = fake_pa("fail")
    assert m.self_test()[0] is False


def toy_model():
    v = ["<unk>", "▁"] + list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,")
    return types.SimpleNamespace(
        vocabulary=v + [f"<pad{i}>" for i in range(1024 - len(v))]
    )


def test_set_terms_sends_the_trie_and_keeps_the_spellings(fake_pa, tmp_path):
    path = tmp_path / "pa"
    m = MetalParakeet(
        toy_model(), weights=str(tmp_path / "ok"), pa=str(path), rescore=False
    )
    note = m.set_terms(["vLLM", " vLLM ", "cuBLAS", ""])
    assert m.terms == [
        "vLLM",
        "cuBLAS",
    ]  # deduplicated: a span's canonical index is a position in this list
    assert "boosting 2 terms" in note
    assert m.set_terms([]) == "dictionary boosting off" and m.terms is None
    sizes = trie_sizes(tmp_path / "ok.log")
    assert (
        len(sizes) == 2 and sizes[0] > 3089 and sizes[1] == 0
    )  # a trie, then none (off)
    assert m.metal_rows(SAMPLES, FRAMES) == [
        (7, 1, 0, 0.5)
    ]  # windows still answer after a dictionary request
    m.close()


def test_a_refused_dictionary_keeps_the_old_one(tmp_path, fake_pa):
    path = tmp_path / "pa"
    m = MetalParakeet(
        toy_model(), weights=str(tmp_path / "big"), pa=str(path), rescore=False
    )
    m.terms = ["kept"]
    assert "refused (too large)" in m.set_terms(["vLLM"])
    assert m.terms == ["kept"] and not m.dead
    m.close()


def test_set_terms_on_a_dead_helper_does_nothing(fake_pa, tmp_path):
    m, _log = fake_pa("exit", max_restarts=0)
    m.metal_rows(SAMPLES, FRAMES)
    assert m.dead
    assert "Metal ASR is off" in m.set_terms(["vLLM"])
    assert m.terms is None


# ---- acoustic span rescoring (the default): the decode stays unboosted, terms come from rescoring after it


def rows_of(text):
    """greedy rows for text on the toy vocabulary: one piece per frame"""
    from sidecar import asr_terms

    encode, _ = asr_terms.bpe_encoder(toy_model().vocabulary)
    return [(t, 1, i, 0.9) for i, t in enumerate(encode(text))]


def rescorer(tmp_path, kind="ok", **kw):
    return MetalParakeet(
        toy_model(),
        weights=str(tmp_path / kind),
        pa=str(tmp_path / "pa"),
        rescore=True,
        **kw,
    )


def test_rescore_is_the_default_and_env_turns_it_off(fake_pa, monkeypatch):
    monkeypatch.delenv("PHONON_ASR_RESCORE", raising=False)
    m, _ = fake_pa("ok")
    assert m.rescore
    m.close()
    monkeypatch.setenv("PHONON_ASR_RESCORE", "0")
    m, _ = fake_pa("ok")
    assert not m.rescore
    m.close()


def test_rescore_mode_keeps_the_decode_unboosted(fake_pa, tmp_path):
    m = rescorer(tmp_path)
    note = m.set_terms(
        ["FluidAudio", "cuBLAS"],
        entries=[{"term": "cuBLAS", "spoken_forms": ["cue blast"]}],
    )
    assert "rescoring 2 of 2 terms" in note
    assert trie_sizes(tmp_path / "ok.log") == []  # no trie: pa's TD.trie stays off
    assert m.terms == ["FluidAudio", "cuBLAS"]
    assert [(e.term, e.spoken_forms) for e in m.entries] == [
        ("cuBLAS", ("cue blast",)),
        ("FluidAudio", ()),
    ]
    assert m.set_terms(["FluidAudio"]).startswith(
        "dictionary rescoring"
    )  # the old call still works
    assert [e.term for e in m.entries] == ["FluidAudio"]
    assert (
        m.set_terms([]) == "dictionary rescoring off"
        and m.terms is None
        and m.entries == []
    )
    m.close()


def test_rescore_spans_frames_the_request(fake_pa, tmp_path):
    m = rescorer(tmp_path)
    jobs = [(3, 9, [1, 2, 3], [4, 5], [6, 7, 8]), (0, 2, [], [9], [10, 11])]
    assert m.rescore_spans(jobs) == [(-1.0, -2.0), (-1.0, -2.0)]
    assert rescores(tmp_path / "ok.log") == [(3, 9, 3, 2, 3), (0, 2, 0, 1, 2)]
    assert m.rescore_spans([]) == []
    assert m.metal_rows(SAMPLES, FRAMES) == [
        (7, 1, 0, 0.5)
    ]  # the protocol stays in step
    m.close()


def test_rescore_spans_falls_back_when_pa_fails(fake_pa, tmp_path):
    m = rescorer(tmp_path, "rescore_exit", max_restarts=1)
    assert m.rescore_spans([(0, 2, [], [1], [2])]) is None
    assert not m.dead and m.p is None  # the next window respawns the helper
    assert m.metal_rows(SAMPLES, FRAMES) == [(7, 1, 0, 0.5)]
    m._off("test")
    assert m.rescore_spans([(0, 2, [], [1], [2])]) is None  # off: no helper to ask


def test_screen_terms_are_temporary(fake_pa, tmp_path):
    m = rescorer(tmp_path)
    m.set_terms(["cuBLAS"])
    assert (
        m.set_screen_terms(["FluidAudio", "FluidAudio", " "])
        == "1 screen terms for this dictation"
    )
    voc = m.rescore_vocabulary()
    assert [(e.term, e.temporary) for e in voc] == [
        ("FluidAudio", True),
        ("cuBLAS", False),
    ]
    assert m.set_screen_terms([]) == "screen terms cleared"
    assert [e.term for e in m.rescore_vocabulary()] == ["cuBLAS"]
    m.close()


def test_rescore_rows_replaces_when_the_audio_agrees(fake_pa, tmp_path):
    m = rescorer(tmp_path)
    m.set_screen_terms(["FluidAudio"])
    rows = rows_of("I use fluid audio daily")
    # fake scores: heard -1, cand -2; FluidAudio is 10 pieces here, so 2.5 nats a piece carries it
    reps = m.rescore_rows(rows, len(rows) + 4)
    v = toy_model().vocabulary
    assert len(reps) == 1
    a, b, text = reps[0]
    assert (
        text == "FluidAudio" and "".join(v[r[0]] for r in rows[a:b]) == "▁fluid▁audio"
    )
    (job,) = rescores(tmp_path / "ok.log")
    assert (
        job[:2] == (rows[a][2], rows[b][2]) and job[2] == a
    )  # frames, and the prefix is every row before
    assert (
        m.last_rescore[0][:2] == ("fluid audio", "FluidAudio") and m.last_rescore[0][4]
    )
    m.close()


def test_rescore_rows_keeps_the_words_when_the_audio_disagrees(
    fake_pa, tmp_path, monkeypatch
):
    m = rescorer(tmp_path, "ok_far")
    m.set_terms(["FluidAudio"])
    rows = rows_of("I use fluid audio daily")
    assert m.rescore_rows(rows, len(rows) + 4) == []  # cand -100 + 25 < heard -1
    monkeypatch.setenv("PHONON_VOCAB_BOOST", "off")
    m2 = rescorer(tmp_path)
    m2.set_terms(["FluidAudio"])
    assert m2.rescore_rows(rows, len(rows) + 4) == []  # no bonus: -2 < -1
    m.close()
    m2.close()


def test_ordinary_speech_never_reaches_the_rescorer(fake_pa, tmp_path):
    m = rescorer(tmp_path)
    m.set_terms(["FluidAudio", "feed"])
    rows = rows_of("feed all of them")
    assert m.rescore_rows(rows, len(rows) + 4) == []
    assert rescores(tmp_path / "ok.log") == []
    m.close()


def test_spoken_forms_replace_without_asking_pa(fake_pa, tmp_path):
    m = rescorer(tmp_path)
    m.set_terms(["cuBLAS"], entries=[{"term": "cuBLAS", "spoken_forms": ["cue blast"]}])
    rows = rows_of("ran cue blast.")
    reps = m.rescore_rows(rows, len(rows) + 4)
    v = toy_model().vocabulary
    (a, b, text) = reps[0]
    assert text == "cuBLAS" and "".join(v[r[0]] for r in rows[a:b]) == "▁cue▁blast"
    assert v[rows[b][0]] == "."  # the trailing period stays outside
    assert rescores(tmp_path / "ok.log") == []
    m.close()


def test_rewrite_merges_each_span_into_one_token(monkeypatch):
    import dataclasses

    from sidecar import pa_client

    @dataclasses.dataclass
    class Tok:
        id: int
        start: float
        duration: float
        confidence: float
        text: str

    monkeypatch.setitem(sys.modules, "parakeet_mlx", types.ModuleType("parakeet_mlx"))
    monkeypatch.setitem(
        sys.modules, "parakeet_mlx.alignment", types.SimpleNamespace(AlignedToken=Tok)
    )
    v = toy_model().vocabulary
    rows = rows_of("so fluid audio.")
    hyp = [
        Tok(t, 0.08 * i, 0.08, 0.9 - 0.01 * i, v[t].replace("▁", " "))
        for i, (t, _d, _s, _c) in enumerate(rows)
    ]
    # rows: ▁ s o ▁ f l u i d ▁ a u d i o . ; a later span overlapping an earlier one is dropped
    out = pa_client._rewrite(hyp, [(10, 16, "bad overlap"), (3, 15, "FluidAudio")], v)
    assert "".join(t.text for t in out) == " so FluidAudio."
    merged = out[3]
    assert (
        merged.start == hyp[3].start
        and abs(merged.duration - (hyp[14].start + 0.08 - hyp[3].start)) < 1e-9
    )
    assert merged.confidence == min(h.confidence for h in hyp[3:15])
    assert (
        pa_client._encoder_frames(102) == 13
    )  # 1 s of audio: 102 mel frames -> 13 encoder frames
