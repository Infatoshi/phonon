"""MetalParakeet's helper protocol against a fake `pa serve`: the sticky fallback (a failed window, a helper that
does not answer, a helper that exits) and the self-test's window under `pa busy`."""

import os
import stat
import sys
import time

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
    events.write("request\\n"); events.flush()
    words = n & 0x7FFFFFFF if n >> 31 else n * 128
    inp.read(4 * words)
    kind = mode.rsplit("/", 1)[-1]
    if kind == "hang":
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
    return [int(x) for x in log.read_text().split()] if os.path.exists(log) else []


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
    m, log = fake_pa("fail")
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert m.dead
    for _ in range(3):  # later windows go straight to MLX: they never reach the helper
        assert m.metal_rows(SAMPLES, FRAMES) is None
    assert len(requests(log)) == 1
    assert capsys.readouterr().err.count("Metal ASR off for this session") == 1
    assert m.p.poll() is not None  # stopped, not left holding GPU memory


def test_a_helper_that_does_not_answer_times_out_once(fake_pa):
    m, log = fake_pa("hang", timeout_s=0.5)
    t = time.monotonic()
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert time.monotonic() - t < 5
    assert m.dead
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert len(requests(log)) == 1


def test_a_helper_that_exits_is_off(fake_pa):
    m, _ = fake_pa("exit")
    assert m.metal_rows(SAMPLES, FRAMES) is None
    assert m.dead


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
