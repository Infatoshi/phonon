#!/usr/bin/env python3
"""JSONL ASR sidecar: load Parakeet MLX, transcribe wav paths.

Segment streaming ({"cmd":"stream_start","id":..,"segments":true}): stream_chunk audio goes
through a causal energy VAD (stream_vad.py) instead of parakeet-mlx's chunked streaming. Each
time a pause closes a speech segment, that segment is transcribed as a batch and emitted as
{"type":"segment","id":..,"index":i,"text":..} plus a partial result with the text so far.
stream_stop keeps the state; {"cmd":"stream_finish","id":..,"path":wav} appends any audio the
wav holds beyond what was streamed, transcribes the open segment, and emits the final
{"type":"result","partial":false,"text": all segments joined}. At key release only the open
segment is transcribed instead of the whole recording.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import traceback
import wave
from pathlib import Path

# The sidecar's own modules (linereader, stream_asr, ...) live inside the signed app bundle;
# a __pycache__ written next to them breaks the bundle's code signature.
sys.dont_write_bytecode = True


def stream_refine_enabled() -> bool:
    """Streaming refine's ASR side: the engine sets PHONON_STREAM_REFINE (1/true/on) from
    settings.json `stream_refine` or its own environment."""
    value = os.environ.get("PHONON_STREAM_REFINE", "")
    return value.strip().lower() in ("1", "true", "on")


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
    return resample_linear(samples, rate, sample_rate)


def resample_linear(samples, rate: int, sample_rate: int):
    """Linear resample of mono float samples. Dictation records at the model's rate, so this
    runs for imported audio and for meeting tracks (48 kHz), where exactness matters less than
    working at all."""
    import numpy as np

    if rate == sample_rate or not samples.size:
        return samples
    duration = samples.size / rate
    target = np.linspace(0.0, duration, num=int(duration * sample_rate), endpoint=False)
    source = np.arange(samples.size, dtype=np.float64) / rate
    return np.interp(target, source, samples).astype(np.float32)


# A span whose loudest 20 ms frame stays under this RMS (-50 dBFS) is treated as silence and
# never reaches the model: digital silence, or a muted or idle track's noise floor.
QUIET_RMS = 10 ** (-50 / 20)


def is_quiet(samples, rate: int) -> bool:
    """True when no 20 ms frame of `samples` rises above QUIET_RMS."""
    import numpy as np

    if not samples.size:
        return True
    frame = max(1, rate // 50)
    usable = samples.size // frame * frame
    if usable == 0:
        return float(np.sqrt(np.mean(samples**2))) < QUIET_RMS
    frames = samples[:usable].reshape(-1, frame)
    loudest = float(np.sqrt(np.max(np.mean(frames.astype(np.float64) ** 2, axis=1))))
    return loudest < QUIET_RMS


def read_span(chunks: list, start_sample: int, end_sample: int):
    """Samples [start_sample, end_sample) of one meeting track, as (mono float samples, rate).

    `chunks`: the track's chunk WAVs that cover the span, each {"path", "start_sample"} with
    start_sample the chunk's first sample on the track. A part no chunk covers (a missing or
    short chunk) is silence, so the span keeps its length and the times stay on the track's clock.
    Only these chunks are read: a long meeting is never held in memory."""
    import numpy as np

    rate = None
    pieces = []
    for chunk in sorted(chunks, key=lambda c: int(c["start_sample"])):
        with wave.open(chunk["path"], "rb") as wav:
            if wav.getsampwidth() != 2:
                raise ValueError("meeting chunks must be 16-bit PCM")
            channels = wav.getnchannels()
            chunk_rate = wav.getframerate()
            first = int(chunk["start_sample"])
            frames = wav.getnframes()
            lo = max(start_sample, first)
            hi = min(end_sample, first + frames)
            if hi <= lo:
                rate = rate or chunk_rate
                continue
            wav.setpos(lo - first)
            data = wav.readframes(hi - lo)
        if rate is not None and chunk_rate != rate:
            raise ValueError("meeting chunks disagree on the sample rate")
        rate = chunk_rate
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        pieces.append((lo, samples))
    if rate is None:
        raise ValueError("no chunk covers the span")
    out = np.zeros(max(0, end_sample - start_sample), dtype=np.float32)
    for lo, samples in pieces:
        at = lo - start_sample
        out[at : at + samples.size] = samples[: out.size - at]
    return out, rate


def sentences_of(result) -> list:
    """The sentences of an AlignedResult as [{"start","end","text"}] in seconds from the
    start of the audio it was given; empty sentences are dropped."""
    out = []
    for sentence in getattr(result, "sentences", None) or []:
        text = (getattr(sentence, "text", "") or "").strip()
        if not text:
            continue
        start = float(getattr(sentence, "start", 0.0))
        end = float(getattr(sentence, "end", start))
        out.append(
            {"start": round(start, 3), "end": round(max(start, end), 3), "text": text}
        )
    return out


def model_label(model_id: str) -> str:
    """The model as a meeting transcript names it: the hub id, or a local folder's name only
    (never its full path)."""
    if os.path.isdir(model_id):
        return "local:" + os.path.basename(os.path.normpath(model_id))
    return model_id


def metal_asr_config() -> dict | None:
    """Parakeet on Metal (pa: the GPU log-mel, encoder and TDT decode) instead of MLX, or None.

    Default: the helper bundled with the app, Contents/Helpers/pa with Contents/Resources/asr/{pa.metal,
    melfb.npy} (package-bar.sh), when present. PHONON_METAL_ASR=0 (or off / false / no) keeps MLX.
    PHONON_METAL_ASR=<dir>: a development layout (<dir>/bin/pa, <dir>/pa.metal, <dir>/melfb.npy,
    <dir>/pa_client.py). pa reads the pinned snapshot's model.safetensors itself, so MLX's Parakeet weights are
    never loaded (pa_client.LightModel reads config.json); MLX loads only if a window needs the fallback (over
    82 s, or the helper failing a window, not answering or exiting: then MLX for the rest of the session)."""
    value = os.environ.get("PHONON_METAL_ASR", "").strip()
    if value.lower() in ("0", "off", "false", "no"):
        return None
    if value:
        return {
            "pa": os.path.join(value, "bin", "pa"),
            "pa_dir": value,
            "client": value,
        }
    here = Path(__file__).resolve().parent  # Contents/Resources/sidecar
    pa = here.parent.parent / "Helpers" / "pa"
    shaders = here.parent / "asr"
    if (
        pa.is_file()
        and (shaders / "pa.metal").is_file()
        and (shaders / "melfb.npy").is_file()
    ):
        return {"pa": str(pa), "pa_dir": str(shaders), "client": str(here)}
    return None


def load_metal_asr(cfg: dict, source: str):
    """MetalParakeet over the bundled (or development) helper, or None when it does not pass its startup self-test
    on this Mac (the caller then loads the MLX model, as before)."""
    weights = os.path.join(source, "model.safetensors")
    if not os.path.isfile(weights):
        return None, "no model.safetensors in the Parakeet snapshot"
    if cfg["client"] not in sys.path:
        sys.path.insert(0, cfg["client"])
    from pa_client import LightModel, MetalParakeet

    model = MetalParakeet(
        LightModel(source),
        weights=weights,
        pa=cfg["pa"],
        pa_dir=cfg["pa_dir"],
        log=os.environ.get("PHONON_METAL_ASR_LOG"),
    )
    # no window under `pa busy` here: the refiner probes its grid at the same moment, and a spinning load would make
    # it settle on a smaller grid for the session. The decode has no barrier left to stall; a stuck window at run
    # time still hits pa_client's timeout and the sticky MLX fallback.
    ok, ms = model.self_test(load=False)
    if not ok:
        try:
            model.close()
        except Exception:  # noqa: BLE001  (a helper that failed its self-test may already be gone)
            traceback.print_exc(file=sys.stderr)
        return None, f"Metal ASR self-test failed ({ms:.0f} ms)"
    return model, f"Metal ASR ready (self-test {ms:.1f} ms)"


def transcribe_samples(model, samples):
    """Batch transcription of mono float samples at the model's rate."""
    # MetalParakeet: the same AlignedResult as generate()
    if hasattr(model, "transcribe_samples"):
        return model.transcribe_samples(samples)
    import mlx.core as mx
    import numpy as np
    from parakeet_mlx.audio import get_logmel

    if len(samples) < 1600:  # the STFT needs a few frames; pad a tiny tail with silence
        samples = np.concatenate(
            [samples, np.zeros(1600 - len(samples), dtype=np.float32)]
        )
    # float32, matching what parakeet-mlx's own loader hands the preprocessor.
    # bfloat16 changes the FFT layout and the mel matmul fails on shape.
    mel = get_logmel(mx.array(samples, dtype=mx.float32), model.preprocessor_config)
    return model.generate(mel)[0]


def transcribe_file(model, path: str):
    """Batch transcription without an ffmpeg dependency where possible."""
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
    return transcribe_samples(model, samples)


def stream_transcriber(model, req: dict):
    """StreamTranscriber over this model with stream_asr.SCHEME; stream_start may override its
    settings."""
    from stream_asr import SCHEME, StreamTranscriber, words_of

    def transcribe_words(samples, offset_s):
        return words_of(transcribe_samples(model, samples), offset_s)

    settings = dict(SCHEME)
    for key in (
        "pause_s",
        "before_s",
        "after_s",
        "release_before_s",
        "release_overlap_s",
        "soft_max_s",
        "soft_overlap_s",
    ):
        if req.get(key) is not None:
            settings[key] = float(req[key])
    if req.get("round_every") is not None:
        settings["round_every"] = int(req["round_every"])
    return StreamTranscriber(transcribe_words, **settings)


def model_source(model_id: str, revision: str | None, need_folder: bool) -> str:
    """Where the Parakeet weights load from. An existing directory (the engine's PHONON_ASR_MODEL_DIR: a folder in
    the pinned snapshot's layout, config.json + model.safetensors + tokenizer files) is used as it is, revision
    ignored, for both MLX and pa (pa_client reads <source>/model.safetensors). Otherwise a hub id: parakeet_mlx has
    no revision argument, so a revision (or pa, which needs the files on disk) resolves the exact snapshot first."""
    if os.path.isdir(model_id):
        return model_id
    if revision or need_folder:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, revision=revision)
    return model_id


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
    except Exception as e:  # noqa: BLE001
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
    metal_cfg = metal_asr_config()
    metal = False
    try:
        source = model_source(model_id, revision, bool(metal_cfg))
        if revision and not os.path.isdir(model_id):
            # The pinned version is on disk: note it as Phonon's, and drop older versions of
            # this model that Phonon fetched (cache_gc's manifest) from the cache.
            try:
                from cache_gc import prune_in_background
            except ImportError:
                from sidecar.cache_gc import prune_in_background
            prune_in_background(model_id, revision, "speech model")
        model = None
        if metal_cfg:
            try:
                model, note = load_metal_asr(metal_cfg, source)
            except Exception as e:  # noqa: BLE001  (the helper did not start): MLX, as before
                model, note = None, f"Metal ASR unavailable: {e}"
                traceback.print_exc(file=sys.stderr)
            metal = model is not None
            emit({"type": "status", "phase": "loading", "pct": 0.3, "msg": note})
        if model is None:
            model = from_pretrained(source)
    except Exception as e:  # noqa: BLE001
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

        # Every utterance has its own length, so freed buffers rarely fit the next request,
        # and MLX's default cache limit is its memory limit (tens of GB on a 36 GB Mac): a
        # process that transcribed a few hundred clips of different lengths grew past 28 GB.
        # 1 GB keeps short clips at 39.7 ms median (256 MB: 43.0 ms; clearing every call:
        # 51.6 ms; M4 Max, 52 clips under 4 s).
        mx.set_cache_limit(1 << 30)
        if not metal:  # Metal: the self-test already ran windows through pa serve
            mx.eval(model.parameters())
        if stream_refine_enabled():  # MLX path (or Metal's fallback model)
            import fast_tdt

            fast_tdt.install()  # same tokens, fewer GPU round trips per decode step
    except Exception as e:  # noqa: BLE001
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
    segments = None  # StreamTranscriber while a segment stream is open
    segment_seq = 0
    stable_commits = False

    def close_stream() -> None:
        nonlocal streamer, stream_id, segments
        if streamer is not None:
            streamer.__exit__(None, None, None)
        streamer = None
        stream_id = None
        segments = None

    from linereader import LineReader

    reader = LineReader(sys.stdin.fileno())
    pending: list = []
    while True:
        if not pending:
            pending = reader.poll(None)
            if not pending:
                if reader.eof:
                    break
                continue
        req = pending.pop(0)
        if isinstance(req, str):
            emit({"type": "error", "msg": f"bad json: {req[:80]}"})
            continue
        cmd = req.get("cmd")
        # the user's dictionary for in-decode boosting (Metal only), from the engine
        if cmd == "terms":
            try:
                if hasattr(model, "set_terms"):
                    note = model.set_terms(req.get("terms") or [])
                else:
                    note = "dictionary not applied: speech recognition runs on MLX"
            except Exception as e:  # noqa: BLE001  (a bad list must not take the sidecar down)
                note = f"dictionary not applied: {e}"
                traceback.print_exc(file=sys.stderr)
            print(f"asr_server: {note}", file=sys.stderr, flush=True)
            continue
        if cmd == "shutdown":
            close_stream()
            emit({"type": "status", "phase": "shutdown", "pct": 1.0, "msg": "bye"})
            break
        if cmd == "stream_start" and req.get("segments"):
            close_stream()
            stream_id = req.get("id")
            segment_seq = 0
            stable_commits = bool(req.get("stable_commits", False))
            try:
                segments = stream_transcriber(model, req)
                emit({"type": "stream_started", "id": stream_id, "segments": True})
            except Exception as e:  # noqa: BLE001
                close_stream()
                emit(
                    {
                        "type": "error",
                        "id": stream_id,
                        "msg": f"stream start failed: {e}",
                    }
                )
            continue
        if cmd == "stream_chunk" and segments is not None:
            rid = req.get("id")
            if rid != stream_id:
                emit({"type": "error", "id": rid, "msg": "no matching ASR stream"})
                continue
            try:
                t1 = time.perf_counter()
                # Fold every chunk already waiting for this stream into one pass: the audio and
                # the VAD take all of them, the transcription work runs once, and not at all when
                # a stream_finish is already queued (the release window covers that audio).
                pending += reader.poll(0.0)
                chunks = [req]
                while (
                    pending
                    and isinstance(pending[0], dict)
                    and pending[0].get("cmd") == "stream_chunk"
                    and pending[0].get("id") == rid
                ):
                    chunks.append(pending.pop(0))
                finishing = any(
                    isinstance(p, dict)
                    and p.get("cmd") == "stream_finish"
                    and p.get("id") == rid
                    for p in pending
                )
                texts = []
                for i, chunk in enumerate(chunks):
                    work = i == len(chunks) - 1 and not finishing
                    texts = (
                        segments.feed(decode_pcm16(chunk.get("pcm16") or ""), work=work)
                        or texts
                    )
                for text in texts:
                    # the transcript so far for the correction stage to refine early ("text"
                    # leaves out the last words the release may still revise; "preview" has all)
                    emit(
                        {
                            "type": "segment",
                            "id": rid,
                            "index": segment_seq,
                            "text": segments.stable if stable_commits else text,
                            "preview": text,
                            "seconds": time.perf_counter() - t1,
                        }
                    )
                    emit(
                        {
                            "type": "result",
                            "id": rid,
                            "text": text,
                            "partial": True,
                            "seconds": time.perf_counter() - t1,
                        }
                    )
                    segment_seq += 1
            except Exception as e:  # noqa: BLE001
                emit({"type": "error", "id": rid, "msg": f"stream chunk failed: {e}"})
                traceback.print_exc(file=sys.stderr)
            continue
        if cmd == "stream_stop" and segments is not None:
            continue  # the stream stays until stream_finish
        if cmd == "stream_finish":
            rid = req.get("id")
            if segments is None or rid != stream_id:
                emit({"type": "error", "id": rid, "msg": "no matching segment stream"})
                continue
            t1 = time.perf_counter()
            try:
                tail = None
                path = req.get("path") or ""
                if path and Path(path).is_file():
                    # the recorder sends audio in 0.4 s chunks, so the last part of the
                    # recording is only in the wav
                    full = read_wav(path, model.preprocessor_config.sample_rate)
                    if full is not None and len(full) > segments.n:
                        tail = full[segments.n :]
                text = segments.finish(tail)
                emit(
                    {
                        "type": "result",
                        "id": rid,
                        "text": text,
                        "partial": False,
                        "segments": len(segments.pieces) + 1,
                        "seconds": time.perf_counter() - t1,
                    }
                )
            except Exception as e:  # noqa: BLE001
                emit({"type": "error", "id": rid, "msg": f"stream finish failed: {e}"})
                traceback.print_exc(file=sys.stderr)
            close_stream()
            continue
        if cmd == "stream_start":
            close_stream()
            stream_id = req.get("id")
            try:
                streamer = model.transcribe_stream(context_size=(128, 64))
                streamer.__enter__()
                emit({"type": "stream_started", "id": stream_id})
            except Exception as e:  # noqa: BLE001
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
                    if audio.size < 1600:
                        # the recorder's last partial chunk can be tiny; parakeet-mlx's STFT
                        # fails under a few frames, so pad it with silence
                        import numpy as np

                        audio = np.concatenate(
                            [audio, np.zeros(1600 - audio.size, dtype=audio.dtype)]
                        )
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
            except Exception as e:  # noqa: BLE001
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
            except Exception as e:  # noqa: BLE001
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
            except Exception as e:  # noqa: BLE001
                emit({"type": "error", "id": rid, "msg": f"transcribe failed: {e}"})
                traceback.print_exc(file=sys.stderr)
            continue
        if cmd == "transcribe_span":
            # One bounded piece of one meeting track (the engine sends one at a time, only while
            # no dictation is under way). Answers {"type":"span"} with sentence times in seconds
            # from start_sample. Nothing of the audio or its text goes to stderr.
            rid = req.get("id")
            t1 = time.perf_counter()
            try:
                start = int(req.get("start_sample") or 0)
                end = int(req.get("end_sample") or 0)
                if end <= start:
                    raise ValueError("empty span")
                samples, rate = read_span(req.get("chunks") or [], start, end)
                quiet = is_quiet(samples, rate)
                sentences = []
                if not quiet:
                    audio = resample_linear(
                        samples, rate, model.preprocessor_config.sample_rate
                    )
                    sentences = sentences_of(transcribe_samples(model, audio))
                emit(
                    {
                        "type": "span",
                        "id": rid,
                        "sentences": sentences,
                        "quiet": quiet,
                        "audio_seconds": round((end - start) / rate, 3),
                        "seconds": time.perf_counter() - t1,
                        "model": model_label(model_id),
                    }
                )
            except Exception as e:  # noqa: BLE001
                emit(
                    {"type": "error", "id": rid, "msg": f"transcribe span failed: {e}"}
                )
            continue
        if cmd == "ping":
            emit({"type": "pong"})
            continue
        emit({"type": "error", "msg": f"unknown cmd: {cmd}"})


if __name__ == "__main__":
    main()
