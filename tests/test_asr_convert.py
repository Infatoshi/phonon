"""The Mac's MLX-layout speech model built from the published int8 ONNX files: the numeric pieces of
the conversion, the safetensors writer, and converted_model's build-once, hash-checked folder."""

import hashlib
import json
import struct

import numpy as np
import pytest

from sidecar import asr_convert
from sidecar.asr_convert import (
    converted_model,
    reorder_lstm_gates,
    to_bf16,
    write_safetensors,
)

REPO = "Org/model-onnx"
REV = "c" * 40
LAYOUT = ["config.json", "vocab.txt"]
CONTENT = b"converted weights"


def test_bf16_rounds_to_nearest_even():
    values = np.array([1.0, -2.5, 1.0 + 2**-8, 1.0 + 3 * 2**-8, 3.0e-3], dtype=np.float32)
    bits = to_bf16(values)
    back = (bits.astype(np.uint32) << 16).view(np.float32)
    assert back[0] == 1.0 and back[1] == -2.5
    assert back[2] == 1.0  # halfway: down to the even mantissa
    assert back[3] == 1.0 + 4 * 2**-8  # halfway: up to the even mantissa
    assert abs(back[4] - 3.0e-3) / 3.0e-3 < 2**-8


def test_lstm_gates_go_from_onnx_iofc_to_mlx_ifgo():
    hidden = 2
    blocks = {gate: np.full((hidden, 3), k, dtype=np.float32) for k, gate in enumerate("iofc")}
    onnx_order = np.concatenate([blocks[g] for g in "iofc"])
    mlx = reorder_lstm_gates(onnx_order, hidden)
    assert [int(mlx[k * hidden, 0]) for k in range(4)] == [0, 2, 3, 1]  # i, f, c(g), o


def test_safetensors_writer_round_trips(tmp_path):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "m.safetensors"
    write_safetensors(str(path), {"b.weight": (a.shape, to_bf16(a)), "a.bias": ((1,), to_bf16(np.ones(1)))})
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    assert n % 8 == 0
    header = json.loads(raw[8 : 8 + n])
    assert list(header) == ["a.bias", "b.weight"]
    start, end = header["b.weight"]["data_offsets"]
    data = np.frombuffer(raw[8 + n + start : 8 + n + end], dtype=np.uint16)
    assert header["b.weight"]["dtype"] == "BF16" and header["b.weight"]["shape"] == [2, 3]
    assert np.array_equal((data.astype(np.uint32) << 16).view(np.float32).reshape(2, 3), a)


def _setup(tmp_path, content=CONTENT):
    onnx_dir = tmp_path / "onnx"
    layout = tmp_path / "layout"
    onnx_dir.mkdir()
    layout.mkdir()
    for name in LAYOUT:
        (layout / name).write_text(name)
    entry = {
        "mlx_layout": {"repo": "Org/mlx", "files": LAYOUT},
        "converted": {"model.safetensors": {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}},
    }
    calls = []

    def fake_convert(src, out):
        calls.append(src)
        with open(out, "wb") as handle:
            handle.write(CONTENT)

    return onnx_dir, layout, entry, calls, fake_convert


def test_the_model_is_built_once_and_then_reused(tmp_path):
    onnx_dir, layout, entry, calls, fake = _setup(tmp_path)
    kwargs = {"output_root": str(tmp_path / "out"), "marker_dir": str(tmp_path / "marks"), "convert_fn": fake}
    folder = converted_model(REPO, REV, str(onnx_dir), str(layout), entry, **kwargs)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [f"Org--model-onnx@{REV}"]
    assert (tmp_path / "out" / f"Org--model-onnx@{REV}" / "model.safetensors").read_bytes() == CONTENT
    for name in LAYOUT:
        assert (tmp_path / "out" / f"Org--model-onnx@{REV}" / name).read_text() == name
    assert converted_model(REPO, REV, str(onnx_dir), str(layout), entry, **kwargs) == folder
    assert calls == [str(onnx_dir)]


def test_a_build_that_does_not_match_its_pin_is_refused_and_left_nowhere(tmp_path):
    onnx_dir, layout, entry, _, fake = _setup(tmp_path, content=b"something else")
    with pytest.raises(ValueError, match="pinned sha256"):
        converted_model(
            REPO, REV, str(onnx_dir), str(layout), entry,
            output_root=str(tmp_path / "out"), marker_dir=None, convert_fn=fake,
        )
    assert list((tmp_path / "out").iterdir()) == []


def test_a_damaged_build_is_rebuilt(tmp_path):
    onnx_dir, layout, entry, calls, fake = _setup(tmp_path)
    kwargs = {"output_root": str(tmp_path / "out"), "marker_dir": None, "convert_fn": fake}
    folder = converted_model(REPO, REV, str(onnx_dir), str(layout), entry, **kwargs)
    weights = tmp_path / "out" / f"Org--model-onnx@{REV}" / "model.safetensors"
    weights.write_bytes(b"x" * len(CONTENT))  # same size, other bytes
    assert converted_model(REPO, REV, str(onnx_dir), str(layout), entry, **kwargs) == folder
    assert len(calls) == 2
    assert weights.read_bytes() == CONTENT


def test_the_output_lives_in_phonons_data_folder(monkeypatch, tmp_path):
    monkeypatch.setattr("sidecar.cache_gc.phonon_data_dir", lambda: str(tmp_path))
    assert asr_convert.default_output_root() == str(tmp_path / asr_convert.CONVERTED_DIR_NAME)
