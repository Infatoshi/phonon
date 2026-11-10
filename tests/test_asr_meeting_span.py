import types
import wave

import numpy as np
import pytest

from sidecar.asr_server import (
    QUIET_RMS,
    is_quiet,
    model_label,
    read_span,
    resample_linear,
    sentences_of,
)


def _chunk(path, values, rate=48_000):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.asarray(values, dtype="<i2").tobytes())
    return str(path)


def test_read_span_slices_across_chunks_on_the_track_clock(tmp_path):
    # two 10-sample chunks: values are the track sample index
    a = _chunk(tmp_path / "0000.wav", range(10))
    b = _chunk(tmp_path / "0001.wav", range(10, 20))
    chunks = [{"path": b, "start_sample": 10}, {"path": a, "start_sample": 0}]

    samples, rate = read_span(chunks, 6, 14)

    assert rate == 48_000
    assert (samples * 32768).round().tolist() == [6, 7, 8, 9, 10, 11, 12, 13]


def test_read_span_fills_a_gap_and_a_short_last_chunk_with_silence(tmp_path):
    a = _chunk(tmp_path / "0000.wav", [100] * 10)
    # the next chunk starts at 15: samples 10-14 are missing, and it holds only 3 samples
    b = _chunk(tmp_path / "0001.wav", [200] * 3)
    chunks = [{"path": a, "start_sample": 0}, {"path": b, "start_sample": 15}]

    samples, _ = read_span(chunks, 8, 20)

    assert (samples * 32768).round().tolist() == [
        100,
        100,
        0,
        0,
        0,
        0,
        0,
        200,
        200,
        200,
        0,
        0,
    ]


def test_read_span_without_a_covering_chunk_fails():
    with pytest.raises(ValueError):
        read_span([], 0, 10)


def test_resample_linear_takes_48k_to_16k_and_keeps_16k():
    tone = np.sin(np.arange(48_000) / 48_000 * 2 * np.pi * 200).astype(np.float32)

    down = resample_linear(tone, 48_000, 16_000)

    assert down.size == 16_000
    assert down.dtype == np.float32
    assert np.allclose(down, tone[::3], atol=1e-6)
    assert resample_linear(down, 16_000, 16_000) is down


def test_is_quiet_skips_digital_silence_and_a_noise_floor_but_not_speech():
    rate = 48_000
    assert is_quiet(np.zeros(rate * 2, dtype=np.float32), rate)
    rng = np.random.default_rng(0)
    floor = (rng.standard_normal(rate * 2) * QUIET_RMS * 0.3).astype(np.float32)
    assert is_quiet(floor, rate)
    # one 100 ms word at about -30 dBFS in two seconds of floor
    word = floor.copy()
    word[rate : rate + rate // 10] += 0.03 * np.sin(np.arange(rate // 10) * 0.05)
    assert not is_quiet(word, rate)
    assert is_quiet(np.zeros(0, dtype=np.float32), rate)


def test_sentences_of_keeps_times_and_drops_empty_sentences():
    result = types.SimpleNamespace(
        sentences=[
            types.SimpleNamespace(text=" Hello there. ", start=0.48, end=1.2345),
            types.SimpleNamespace(text="  ", start=1.3, end=1.4),
            types.SimpleNamespace(text="Next one", start=2.0, end=1.9),
        ]
    )

    assert sentences_of(result) == [
        {"start": 0.48, "end": 1.234, "text": "Hello there."},
        {"start": 2.0, "end": 2.0, "text": "Next one"},
    ]
    assert sentences_of(types.SimpleNamespace(text="no sentences")) == []


def test_model_label_never_carries_a_local_path(tmp_path):
    hub = "mlx-community/parakeet-tdt-0.6b-v2"
    assert model_label(hub) == hub
    folder = tmp_path / "merged-model"
    folder.mkdir()
    assert model_label(str(folder) + "/") == "local:merged-model"
