"""Metal Parakeet TDT for the ASR sidecar: a drop-in for asr_server.transcribe_samples(model, samples).

MetalParakeet(model) keeps one `pa WEIGHTS serve MAXMEL` process (get_logmel's log-mel, the encoder and the greedy
TDT decode, all on the GPU) and returns the same AlignedResult that parakeet-mlx's generate() builds, so
stream_asr.words_of and everything after it are unchanged. WEIGHTS is the pinned snapshot's model.safetensors (pa
rounds it to bf16 as the MLX app does and keeps it as fp16). Windows over capacity, a timed-out decode and a dead
helper fall back to the MLX model, which LightModel loads only then.
"""

from __future__ import annotations

import os
import struct
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2/snapshots/"
    "8ae155301e23d820d82aa60d24817c900e69e487"
)


class LightModel:
    """What MetalParakeet needs from the model, from config.json alone (no MLX weights in memory): vocabulary,
    durations, time_ratio. The MLX model (preprocessor config, generate) loads lazily, only for the fallback."""

    def __init__(self, snap=SNAP):
        import dataclasses
        import json

        from parakeet_mlx.audio import PreprocessArgs

        with open(f"{snap}/config.json") as f:
            c = json.load(f)
        self.snap = snap
        self.vocabulary = c["joint"]["vocabulary"]
        self.durations = c["decoding"]["durations"]
        p = c["preprocessor"]
        self.time_ratio = (
            c["encoder"]["subsampling_factor"]
            / p["sample_rate"]
            * int(p["window_stride"] * p["sample_rate"])
        )
        # the preprocessor config without the model (sample_rate for the wav reader, get_logmel for the fallback)
        names = {f.name for f in dataclasses.fields(PreprocessArgs)}
        self.preprocessor_config = PreprocessArgs(
            **{k: v for k, v in p.items() if k in names}
        )
        self._m = None

    def _full(self):
        if self._m is None:
            from parakeet_mlx import from_pretrained

            self._m = from_pretrained(self.snap)
        return self._m

    def generate(self, mel):
        return self._full().generate(mel)

    def __getattr__(self, name):
        # anything else (transcribe, transcribe_stream, ...): the MLX model, loaded lazily
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._full(), name)


class MetalParakeet:
    def __init__(
        self,
        model,
        weights=f"{HERE}/data/pk.bin",
        pa=f"{HERE}/bin/pa",
        max_mel=8192,
        env=None,
        gpu_mel=True,
        log=None,
        terms=None,
        rewrite="boosted",
        pa_dir=None,
    ):
        """weights: the Parakeet snapshot's model.safetensors (pa converts it at startup) or an exported .bin;
        pa_dir: pa.metal and melfb.npy (default: the dev layout, the directory above bin/pa); terms: the dictionary
        (one term per line) for in-kernel boosting (asr_bias_v0 cb446: case variants, s_start 4, s_cont 4,
        budget 6; needs bias/ next to this file); rewrite: boosted (default, the ASR lane's cb446: only committed
        walks a bonus pushed take the dictionary spelling, tdt_decode.detok --rewrite boosted), all, none.
        Any failure of the helper (it exits, a decode barrier times out, a window over capacity) answers that
        window on the MLX model, loaded on first use; a helper that died stays off for the session."""
        import sys

        self.model = model  # LightModel (or a parakeet-mlx model): vocabulary, durations, time_ratio, fallback
        self.max_mel = max_mel  # 8192 mel frames = 81.9 s (encoder 1024 frames)
        self.gpu_mel = gpu_mel  # send the samples: pa computes get_logmel's log-mel on the GPU (melfb.npy)
        # per call JSON lines: samples, frames, path, GPU / wall ms; open for the session
        self.log = open(log, "a") if log else None  # noqa: SIM115
        self.dead = False
        e = dict(
            os.environ,
            PA_DIR=pa_dir or os.path.dirname(os.path.abspath(pa)) + "/..",
            **(env or {}),
        )
        self.terms, self.rewrite = None, rewrite
        if terms:
            try:
                import tempfile

                sys.path.insert(0, f"{HERE}/bias")
                import make_trie  # tdt_decode's trie, flattened
                import sentencepiece as spm
                import tdt_decode as td

                with open(terms) as f:
                    self.terms = [t for t in (line.strip() for line in f) if t]
                sp = spm.SentencePieceProcessor(
                    model_file=f"{HERE}/bias/tokenizer_v2.model"
                )
                flat = make_trie.flatten(
                    td.build_trie(sp, self.terms, "case"),
                    td.Pieces(sp),
                    4.0,
                    4.0,
                    6.0,
                    False,
                )
                fd, self.trie_path = tempfile.mkstemp(suffix=".trie")
                os.close(fd)
                flat.tofile(self.trie_path)
                e["PA_TRIE"] = self.trie_path
            except Exception as ex:  # noqa: BLE001
                self.terms = None
                print(f"pa_client: dictionary boosting off ({ex})", file=sys.stderr)
        self.p = subprocess.Popen(
            [pa, weights, "serve", str(max_mel)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=e,
        )
        self.last_ms = (0.0, 0.0, 0.0)
        self.last_spans, self.last_boost = [], (0, 0)

    def __getattr__(self, name):
        # preprocessor_config, transcribe, transcribe_stream, ...: the model's
        if name in (
            "model",
            "p",
            "log",
            "last_ms",
            "max_mel",
            "gpu_mel",
            "terms",
            "rewrite",
            "last_spans",
            "last_boost",
            "trie_path",
            "dead",
        ):
            raise AttributeError(name)
        return getattr(self.model, name)

    def warmup(self):
        """One short window: the Metal pipelines and the weight pages ready before the first dictation."""
        self.transcribe_samples(np.zeros(16000, dtype=np.float32))

    def self_test(self, budget_ms=250.0):
        """Startup gate: two 1 s windows through the helper itself (not the fallback). Passes when both answer and
        the second takes under budget_ms of helper wall time; on a GPU where the persistent decode's 20
        threadgroups cannot be co-resident the barrier times out and this fails, and the caller stays on MLX."""
        x = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
        try:
            for _ in range(2):
                if self.tokens(x, samples=True) is None:
                    return False, self.last_ms[2]
            return self.last_ms[2] < budget_ms, self.last_ms[2]
        except Exception:  # noqa: BLE001  (the helper exited)
            self.dead = True
            return False, -1.0

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.write(struct.pack("<I", 0))
            self.p.stdin.flush()
            self.p.wait(timeout=10)

    def tokens(self, mel: np.ndarray, samples: bool = False):
        """mel [n, 128] float32 (or samples [n] float32 with samples=True) -> [(token, duration index, frame,
        confidence)], or None over capacity"""
        n = mel.shape[0]
        head = struct.pack("<I", (n | 0x80000000) if samples else n)
        self.p.stdin.write(head + np.ascontiguousarray(mel, dtype=np.float32).tobytes())
        self.p.stdin.flush()
        (cnt,) = struct.unpack("<I", self.p.stdout.read(4))
        rows = []
        if cnt != 0xFFFFFFFF:
            raw = np.frombuffer(self.p.stdout.read(16 * cnt), dtype="<u4").reshape(
                cnt, 4
            )
            conf = raw[:, 3].copy().view("<f4")
            rows = [
                (int(t), int(d), int(s), float(c)) for (t, d, s, _), c in zip(raw, conf)
            ]
        self.last_ms = struct.unpack("<3f", self.p.stdout.read(12))
        # committed dictionary spans, then flips / rollbacks
        (nsp,) = struct.unpack("<I", self.p.stdout.read(4))
        sp = (
            np.frombuffer(self.p.stdout.read(16 * nsp), dtype="<u4").reshape(nsp, 4)
            if nsp
            else np.zeros((0, 4), np.uint32)
        )
        self.last_spans = [tuple(int(x) for x in s) for s in sp]
        self.last_boost = struct.unpack("<2I", self.p.stdout.read(8))
        return None if cnt == 0xFFFFFFFF else rows

    def transcribe_samples(self, samples):
        import mlx.core as mx
        from parakeet_mlx import tokenizer
        from parakeet_mlx.alignment import (
            AlignedToken,
            sentences_to_result,
            tokens_to_sentences,
        )
        from parakeet_mlx.audio import get_logmel
        from parakeet_mlx.parakeet import DecodingConfig

        m = self.model
        if len(samples) < 1600:  # as asr_server.transcribe_samples
            samples = np.concatenate(
                [samples, np.zeros(1600 - len(samples), dtype=np.float32)]
            )
        samples = np.asarray(samples, dtype=np.float32)
        frames = (len(samples) + 272) // 160  # get_logmel's frame count
        rows = None
        if frames <= self.max_mel and not self.dead:
            try:
                if self.gpu_mel:
                    rows = self.tokens(samples, samples=True)
                else:
                    mel = get_logmel(
                        mx.array(samples, dtype=mx.float32), m.preprocessor_config
                    )
                    rows = self.tokens(
                        np.array(mel[0] if mel.ndim == 3 else mel, dtype=np.float32)
                    )
            except Exception as ex:  # noqa: BLE001  (the helper exited or its pipe broke): MLX from here on
                import sys

                self.dead = True
                rows = None
                print(
                    f"pa_client: Metal helper failed ({ex!r}); transcribing on MLX",
                    file=sys.stderr,
                )
        if self.log:
            import json

            self.log.write(
                json.dumps(
                    {
                        "samples": len(samples),
                        "frames": int(frames),
                        "path": "metal" if rows is not None else "mlx",
                        "enc_ms": round(self.last_ms[0], 3),
                        "dec_ms": round(self.last_ms[1], 3),
                        "pa_wall_ms": round(self.last_ms[2], 3),
                    }
                )
                + "\n"
            )
            self.log.flush()
        if rows is None:  # over capacity (or a timed-out decode barrier): the MLX path
            return m.generate(
                get_logmel(mx.array(samples, dtype=mx.float32), m.preprocessor_config)
            )[0]
        tr = m.time_ratio
        hyp = [
            AlignedToken(
                t,
                start=s * tr,
                duration=m.durations[d] * tr,
                confidence=c,
                text=tokenizer.decode([t], m.vocabulary),
            )
            for t, d, s, c in rows
        ]
        # committed terms take the dictionary spelling (tdt_decode detok)
        if self.terms and self.rewrite != "none":
            keep = sorted(
                (a, b, ci)
                for a, b, ci, boosted in self.last_spans
                if ci != 0xFFFFFFFF and b > a and (self.rewrite == "all" or boosted)
            )
            out, prev = [], 0
            for a, b, ci in keep:
                out += hyp[prev:a]
                first = hyp[a]
                lead = " " if m.vocabulary[first.id].startswith("▁") else ""
                out.append(
                    AlignedToken(
                        first.id,
                        start=first.start,
                        duration=hyp[b - 1].start + hyp[b - 1].duration - first.start,
                        confidence=min(h.confidence for h in hyp[a:b]),
                        text=lead + self.terms[ci],
                    )
                )
                prev = b
            hyp = out + hyp[prev:]
        return sentences_to_result(tokens_to_sentences(hyp, DecodingConfig().sentence))
