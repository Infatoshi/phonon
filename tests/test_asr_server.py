import base64
import struct
import sys
import types

import pytest

from sidecar.asr_server import decode_pcm16, model_source


def test_decode_pcm16_normalizes_little_endian_samples():
    encoded = base64.b64encode(struct.pack("<hhh", -32768, 0, 32767)).decode()

    decoded = decode_pcm16(encoded)

    assert decoded.tolist() == pytest.approx([-1.0, 0.0, 32767 / 32768])


def test_decode_pcm16_rejects_invalid_base64():
    with pytest.raises(ValueError):
        decode_pcm16("not base64")


def _fake_hub(monkeypatch, snapshot_download):
    # a stand-in module, so the test needs neither huggingface_hub nor the network
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))


def _no_download(*args, **kwargs):
    raise AssertionError("a local model folder must not reach the hub")


def test_model_source_uses_an_existing_folder_as_it_is(tmp_path, monkeypatch):
    _fake_hub(monkeypatch, _no_download)

    for revision in ("8ae155301e23d820d82aa60d24817c900e69e487", None):
        for need_folder in (True, False):
            assert model_source(str(tmp_path), revision, need_folder) == str(tmp_path)


def test_model_source_resolves_a_pinned_hub_id(monkeypatch):
    calls = []

    def fake(model_id, revision=None):
        calls.append((model_id, revision))
        return "/snapshots/" + (revision or "main")

    _fake_hub(monkeypatch, fake)

    hub = "mlx-community/parakeet-tdt-0.6b-v2"
    assert model_source(hub, "abc", False) == "/snapshots/abc"
    assert model_source(hub, None, True) == "/snapshots/main"
    assert model_source(hub, None, False) == hub  # parakeet_mlx resolves an unpinned id itself
    assert calls == [(hub, "abc"), (hub, None)]
