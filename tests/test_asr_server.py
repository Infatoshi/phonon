import base64
import struct

import pytest

from sidecar.asr_server import decode_pcm16


def test_decode_pcm16_normalizes_little_endian_samples():
    encoded = base64.b64encode(struct.pack("<hhh", -32768, 0, 32767)).decode()

    decoded = decode_pcm16(encoded)

    assert decoded.tolist() == pytest.approx([-1.0, 0.0, 32767 / 32768])


def test_decode_pcm16_rejects_invalid_base64():
    with pytest.raises(ValueError):
        decode_pcm16("not base64")


def test_warmup_stream_uses_transcribe_samples_for_metal_and_stream_finish_ignores_corrupt_wav(
    monkeypatch, tmp_path, capsys
):
    import json
    import sys
    import types
    import wave
    import numpy as np
    from sidecar import asr_server

    wav_path = tmp_path / "warm.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<" + "h" * 1600, *([0] * 1600)))

    corrupt_wav = tmp_path / "corrupt.wav"
    corrupt_wav.write_bytes(b"not a valid wav file")

    class FakeMetalModel:
        def __init__(self):
            self.preprocessor_config = types.SimpleNamespace(sample_rate=16000)
            self.samples_calls = 0
            self.stream_calls = 0

        def parameters(self):
            return {}

        def transcribe_samples(self, samples):
            self.samples_calls += 1
            return types.SimpleNamespace(
                text="warmed on metal",
                sentences=[
                    types.SimpleNamespace(
                        tokens=[types.SimpleNamespace(text=" hello", start=0.1)]
                    )
                ],
            )

        def transcribe_stream(self, **kw):
            self.stream_calls += 1
            raise AssertionError("warmup_stream should not call transcribe_stream on MetalParakeet")

    fake_model = FakeMetalModel()
    monkeypatch.setattr(asr_server, "metal_asr_config", lambda: None)
    monkeypatch.setattr(asr_server, "stream_refine_enabled", lambda: False)

    fake_mlx = types.ModuleType("mlx")
    fake_mlx.__path__ = []
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.set_cache_limit = lambda limit: None
    fake_mx.eval = lambda params: None
    fake_mlx.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    fake_parakeet_mlx = types.ModuleType("parakeet_mlx")
    fake_parakeet_mlx.from_pretrained = lambda source: fake_model
    monkeypatch.setitem(sys.modules, "parakeet_mlx", fake_parakeet_mlx)

    cmds = [
        {"cmd": "warmup_stream", "id": "w1", "path": str(wav_path)},
        {"cmd": "stream_start", "id": "s1", "segments": True},
        {
            "cmd": "stream_chunk",
            "id": "s1",
            "pcm16": base64.b64encode(np.zeros(3200, dtype="<i2").tobytes()).decode(),
        },
        {"cmd": "stream_finish", "id": "s1", "path": str(corrupt_wav)},
        {"cmd": "shutdown"},
    ]

    class FakeReader:
        def __init__(self, _fd):
            self.items = list(cmds)
            self.eof = False

        def poll(self, _timeout):
            if self.items:
                return [self.items.pop(0)]
            self.eof = True
            return []

    import sidecar.linereader as lr
    import sidecar.stream_asr as sasr
    import sidecar.stream_vad as svad

    monkeypatch.setitem(sys.modules, "linereader", lr)
    monkeypatch.setitem(sys.modules, "stream_asr", sasr)
    monkeypatch.setitem(sys.modules, "stream_vad", svad)
    monkeypatch.setattr(lr, "LineReader", FakeReader)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(fileno=lambda: 0))
    monkeypatch.setattr(sys, "argv", ["asr_server.py"])

    asr_server.main()
    out_lines = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    warm_res = next(r for r in out_lines if r.get("id") == "w1")
    assert warm_res["type"] == "result" and warm_res["text"] == "warmed on metal"
    assert fake_model.samples_calls >= 1
    assert fake_model.stream_calls == 0

    fin_res = next(
        r for r in out_lines if r.get("id") == "s1" and r.get("type") == "result" and not r.get("partial")
    )
    assert fin_res["text"] == "hello"
