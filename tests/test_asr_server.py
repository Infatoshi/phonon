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
        assert model_source(str(tmp_path), revision) == str(tmp_path)


def test_model_source_resolves_a_pinned_hub_id_and_verifies_it(monkeypatch):
    calls = []

    def fake(model_id, revision=None):
        calls.append((model_id, revision))
        return "/snapshots/" + revision

    _fake_hub(monkeypatch, fake)
    verified = []

    def fake_verified(model_id, revision, fetch, **kwargs):
        verified.append((model_id, revision))
        return fetch()

    import sidecar.model_pins

    monkeypatch.setattr(sidecar.model_pins, "verified_snapshot", fake_verified)
    hub = "mlx-community/parakeet-tdt-0.6b-v2"
    pin = "8ae155301e23d820d82aa60d24817c900e69e487"
    assert model_source(hub, pin) == "/snapshots/" + pin
    assert calls == verified == [(hub, pin)]


def test_model_source_builds_the_fine_tuned_model_from_its_onnx_files(monkeypatch):
    downloads = []

    def fake(model_id, revision=None, allow_patterns=None):
        downloads.append((model_id, revision, sorted(allow_patterns)))
        return f"/snapshots/{model_id}"

    _fake_hub(monkeypatch, fake)
    import sidecar.asr_convert
    import sidecar.model_pins

    verified = []

    def fake_verified(model_id, revision, fetch, **kwargs):
        verified.append((model_id, kwargs.get("only")))
        return fetch()

    built = []

    def fake_converted(model_id, revision, onnx_dir, layout_dir, entry):
        built.append((model_id, revision, onnx_dir, layout_dir))
        return "/converted"

    monkeypatch.setattr(sidecar.model_pins, "verified_snapshot", fake_verified)
    monkeypatch.setattr(sidecar.asr_convert, "converted_model", fake_converted)
    hub = "Infatoshi/phonon-asr-parakeet-tdt-0.6b-v2-onnx"
    pin = "76fbda42bfcf7be26cca2488f8fe38ea4328f755"
    layout = ["config.json", "tokenizer.model", "tokenizer.vocab", "vocab.txt"]
    assert model_source(hub, pin) == "/converted"
    onnx = ["decoder.int8.onnx", "encoder.int8.onnx", "encoder.int8.onnx.data", "joiner.int8.onnx"]
    stock = "mlx-community/parakeet-tdt-0.6b-v2"
    stock_pin = "8ae155301e23d820d82aa60d24817c900e69e487"
    assert downloads == [(hub, pin, onnx), (stock, stock_pin, layout)]
    assert verified == [(hub, None), (stock, layout)]
    assert built == [(hub, pin, f"/snapshots/{hub}", f"/snapshots/{stock}")]


def test_model_source_never_fetches_a_branch(monkeypatch):
    _fake_hub(monkeypatch, _no_download)

    hub = "mlx-community/parakeet-tdt-0.6b-v2"
    for revision in (None, "main", "a" * 40):  # no revision, a branch, a commit with no hashes
        with pytest.raises(ValueError):
            model_source(hub, revision)
