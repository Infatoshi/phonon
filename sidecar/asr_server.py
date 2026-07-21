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


def main() -> None:
    model_id = "mlx-community/parakeet-tdt-0.6b-v2"
    for i, a in enumerate(sys.argv[1:]):
        if a == "--model" and i + 2 <= len(sys.argv[1:]):
            model_id = sys.argv[i + 2]

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
        model = from_pretrained(model_id)
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
                        raise ValueError("startup WAV must be mono 16-bit PCM at 16 kHz")
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
                result = model.transcribe(path)
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
