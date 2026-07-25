#!/usr/bin/env python3
"""JSONL ASR sidecar: load Parakeet MLX, transcribe wav paths."""

from __future__ import annotations

import json
import base64
import sys
import time
import traceback
import wave
from pathlib import Path


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def decode_pcm16(encoded: str):
    """Decode little-endian mono PCM16 into normalized float samples."""
    import numpy as np

    pcm = base64.b64decode(encoded, validate=True)
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def read_wav(path: str, sample_rate: int):
    """Decode a 16-bit PCM WAV to mono float samples at `sample_rate`.

    parakeet-mlx decodes audio by shelling out to ffmpeg, which a Mac without
    developer tooling does not have. Phonon records its own WAVs, so decode them
    here with the standard library and leave ffmpeg for formats we do not write.
    Returns None when the file is not 16-bit PCM, so the caller can fall back.
    """
    import numpy as np

    with wave.open(path, "rb") as wav:
        if wav.getsampwidth() != 2:
            return None
        channels = wav.getnchannels()
        frames = wav.readframes(wav.getnframes())
        rate = wav.getframerate()
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != sample_rate and samples.size:
        # Linear resample. Phonon records at the model's rate, so this only runs
        # for imported audio, where exactness matters less than working at all.
        duration = samples.size / rate
        target = np.linspace(
            0.0, duration, num=int(duration * sample_rate), endpoint=False
        )
        source = np.arange(samples.size, dtype=np.float32) / rate
        samples = np.interp(target, source, samples).astype(np.float32)
    return samples


def transcribe_file(model, path: str):
    """Batch transcription without an ffmpeg dependency where possible."""
    import mlx.core as mx

    rate = model.preprocessor_config.sample_rate
    samples = None
    if path.lower().endswith(".wav"):
        try:
            samples = read_wav(path, rate)
        except (wave.Error, EOFError):
            samples = None
    if samples is None:
        # Not something we can decode ourselves; parakeet-mlx will use ffmpeg,
        # and its own error says so plainly when ffmpeg is absent.
        return model.transcribe(path)
    from parakeet_mlx.audio import get_logmel

    mel = get_logmel(mx.array(samples, dtype=mx.bfloat16), model.preprocessor_config)
    return model.generate(mel)[0]


def main() -> None:
    model_id = "mlx-community/parakeet-tdt-0.6b-v2"
    revision = None
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--model" and i + 1 < len(args):
            model_id = args[i + 1]
        elif a == "--revision" and i + 1 < len(args):
            revision = args[i + 1]

    emit(
        {
            "type": "status",
            "phase": "loading",
            "pct": 0.05,
            "msg": f"importing ({model_id})",
        }
    )
    try:
        from parakeet_mlx import from_pretrained
    except Exception as e:
        emit({"type": "error", "msg": f"parakeet_mlx import failed: {e}"})
        return

    emit(
        {
            "type": "status",
            "phase": "loading",
            "pct": 0.15,
            "msg": "loading weights (MLX)",
        }
    )
    t0 = time.perf_counter()
    try:
        # parakeet_mlx has no revision argument, so pin by resolving the exact
        # snapshot first and loading it from disk.
        source = model_id
        if revision:
            from huggingface_hub import snapshot_download

            source = snapshot_download(model_id, revision=revision)
        model = from_pretrained(source)
    except Exception as e:
        emit({"type": "error", "msg": f"load failed: {e}"})
        traceback.print_exc(file=sys.stderr)
        return
    emit(
        {
            "type": "status",
            "phase": "loading",
            "pct": 0.55,
            "msg": "weights mapped; making tensors resident",
        }
    )
    try:
        import mlx.core as mx

        mx.eval(model.parameters())
    except Exception as e:
        emit({"type": "error", "msg": f"weight materialization failed: {e}"})
        traceback.print_exc(file=sys.stderr)
        return
    emit(
        {
            "type": "status",
            "phase": "loading",
            "pct": 0.72,
            "msg": f"weights resident in {time.perf_counter() - t0:.1f}s",
        }
    )
    emit({"type": "ready", "model": model_id})

    streamer = None
    stream_id = None

    def close_stream() -> None:
        nonlocal streamer, stream_id
        if streamer is not None:
            streamer.__exit__(None, None, None)
        streamer = None
        stream_id = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            emit({"type": "error", "msg": f"bad json: {e}"})
            continue
        cmd = req.get("cmd")
        if cmd == "shutdown":
            close_stream()
            emit({"type": "status", "phase": "shutdown", "pct": 1.0, "msg": "bye"})
            break
        if cmd == "stream_start":
            close_stream()
            stream_id = req.get("id")
            try:
                streamer = model.transcribe_stream(context_size=(128, 64))
                streamer.__enter__()
                emit({"type": "stream_started", "id": stream_id})
            except Exception as e:
                close_stream()
                emit(
                    {
                        "type": "error",
                        "id": stream_id,
                        "msg": f"stream start failed: {e}",
                    }
                )
                traceback.print_exc(file=sys.stderr)
            continue
        if cmd == "stream_chunk":
            rid = req.get("id")
            if streamer is None or rid != stream_id:
                emit({"type": "error", "id": rid, "msg": "no matching ASR stream"})
                continue
            try:
                import mlx.core as mx

                audio = decode_pcm16(req.get("pcm16") or "")
                if audio.size:
                    t1 = time.perf_counter()
                    streamer.add_audio(mx.array(audio, dtype=mx.bfloat16))
                    text = (getattr(streamer.result, "text", "") or "").strip()
                    emit(
                        {
                            "type": "result",
                            "id": rid,
                            "text": text,
                            "partial": True,
                            "seconds": time.perf_counter() - t1,
                        }
                    )
            except Exception as e:
                emit({"type": "error", "id": rid, "msg": f"stream chunk failed: {e}"})
                traceback.print_exc(file=sys.stderr)
            continue
        if cmd == "stream_stop":
            rid = req.get("id")
            if stream_id is None or rid == stream_id:
                close_stream()
            continue
        if cmd == "warmup_stream":
            close_stream()
            rid = req.get("id")
            path = req.get("path", "")
            started = time.perf_counter()
            try:
                with wave.open(path, "rb") as wav:
                    if (
                        wav.getnchannels() != 1
                        or wav.getsampwidth() != 2
                        or wav.getframerate() != 16_000
                    ):
                        raise ValueError(
                            "startup WAV must be mono 16-bit PCM at 16 kHz"
                        )
                    audio = decode_pcm16(
                        base64.b64encode(wav.readframes(wav.getnframes())).decode()
                    )
                with model.transcribe_stream(context_size=(128, 64)) as demo_stream:
                    for offset in range(0, len(audio), 6_400):
                        demo_stream.add_audio(
                            mx.array(audio[offset : offset + 6_400], dtype=mx.bfloat16)
                        )
                    text = (getattr(demo_stream.result, "text", "") or "").strip()
                emit(
                    {
                        "type": "result",
                        "id": rid,
                        "text": text,
                        "seconds": time.perf_counter() - started,
                        "partial": False,
                    }
                )
            except Exception as e:
                emit({"type": "error", "id": rid, "msg": f"stream warmup failed: {e}"})
            continue
        if cmd == "transcribe":
            path = req.get("path") or ""
            rid = req.get("id")
            if not path or not Path(path).is_file():
                emit({"type": "error", "id": rid, "msg": f"missing file: {path}"})
                continue
            emit(
                {
                    "type": "status",
                    "phase": "transcribing",
                    "pct": 0.0,
                    "msg": Path(path).name,
                }
            )
            t1 = time.perf_counter()
            try:
                result = transcribe_file(model, path)
                text = getattr(result, "text", None)
                if text is None:
                    text = str(result)
                text = (text or "").strip()
                emit(
                    {
                        "type": "result",
                        "id": rid,
                        "path": path,
                        "text": text,
                        "seconds": time.perf_counter() - t1,
                    }
                )
            except Exception as e:
                emit({"type": "error", "id": rid, "msg": f"transcribe failed: {e}"})
                traceback.print_exc(file=sys.stderr)
            continue
        if cmd == "ping":
            emit({"type": "pong"})
            continue
        emit({"type": "error", "msg": f"unknown cmd: {cmd}"})


if __name__ == "__main__":
    main()
