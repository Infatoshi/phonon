"""Metal Parakeet TDT for the ASR sidecar: a drop-in for asr_server.transcribe_samples(model, samples).

MetalParakeet(model) keeps one `pa WEIGHTS serve MAXMEL` process (get_logmel's log-mel, the encoder and the greedy
TDT decode, all on the GPU) and returns the same AlignedResult that parakeet-mlx's generate() builds, so
stream_asr.words_of and everything after it are unchanged. WEIGHTS is the pinned snapshot's model.safetensors (pa
rounds it to bf16 as the MLX app does and keeps it as fp16). Windows over capacity fall back to the MLX model, which
LightModel loads only then; a failed window, a helper that does not answer in time or a dead helper turn the helper
off for the rest of the session (one log line), so a failing GPU path never costs every window its failure time.
"""

from __future__ import annotations

import os
import select
import struct
import subprocess
import sys

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
        rewrite="boosted",
        pa_dir=None,
        timeout_s=10.0,
        max_restarts=2,
        rescore=None,
    ):
        """weights: the Parakeet snapshot's model.safetensors (pa converts it at startup) or an exported .bin;
        pa_dir: pa.metal and melfb.npy (default: the dev layout, the directory above bin/pa); rewrite: boosted
        (default, the ASR lane's cb446: only committed walks a bonus pushed take the dictionary spelling,
        tdt_decode.detok --rewrite boosted), all, none. Dictionary boosting starts off; set_terms turns it on.
        A window over capacity is answered on the MLX model (loaded on first use). Transient failures (timeout
        under GPU contention, helper exit) answer that window on MLX and respawn the helper up to max_restarts
        consecutive times (re-sending any active dictionary); exceeding max_restarts latches Metal ASR off for
        the rest of the session. rescore (default: PHONON_ASR_RESCORE, on unless "0"): the greedy decode runs
        unboosted and dictionary terms come from acoustic span rescoring after it (asr_terms); off: the in-decode
        trie boosting."""
        if rescore is None:
            rescore = os.environ.get("PHONON_ASR_RESCORE", "1").strip() not in (
                "0",
                "false",
                "off",
                "",
            )
        self.rescore = bool(rescore)
        self.entries, self.screen_terms = [], []
        self.last_rescore = []  # per window: (heard text, term, score heard, score cand, replaced)
        self._encode = None
        self.model = model  # LightModel (or a parakeet-mlx model): vocabulary, durations, time_ratio, fallback
        self.max_mel = max_mel  # 8192 mel frames = 81.9 s (encoder 1024 frames)
        self.gpu_mel = gpu_mel  # send the samples: pa computes get_logmel's log-mel on the GPU (melfb.npy)
        # per window: an answer later than this means a stuck GPU path
        self.timeout_s = timeout_s
        self.pa_bin = pa
        self._weights = weights
        self._max_restarts = max(0, int(max_restarts))
        self._consecutive_failures = 0
        # per call JSON lines: samples, frames, path, GPU / wall ms; open for the session
        self.log = open(log, "a") if log else None  # noqa: SIM115
        self.dead = False
        self._pa_env = dict(
            os.environ,
            PA_DIR=pa_dir or os.path.dirname(os.path.abspath(pa)) + "/..",
            **(env or {}),
        )
        self.terms, self.rewrite = None, rewrite
        self.p = self._spawn()
        self.last_ms = (0.0, 0.0, 0.0)
        self.last_spans, self.last_boost = [], (0, 0)

    def _spawn(self):
        return subprocess.Popen(
            [self.pa_bin, self._weights, "serve", str(self.max_mel)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=self._pa_env,
        )

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
            "dead",
            "timeout_s",
            "pa_bin",
            "_weights",
            "_pa_env",
            "_max_restarts",
            "_consecutive_failures",
            "rescore",
            "entries",
            "screen_terms",
            "last_rescore",
            "_encode",
        ):
            raise AttributeError(name)
        return getattr(self.model, name)

    def set_terms(self, terms, entries=None):
        """Use these dictionary terms (canonical spellings) from the next window on; an empty list turns the
        dictionary off. entries: the same terms with their spoken forms ({"term", "spoken_forms"} or VocabEntry);
        terms missing from entries count as entries without spoken forms. Rescoring mode keeps them for the
        after-decode rescoring (the helper's decode stays unboosted); otherwise the helper boosts them in-decode.
        Returns a line for the log. The MLX fallback does not boost."""
        asr_terms = _asr_terms()
        terms = [t for t in dict.fromkeys(str(t).strip() for t in terms or ()) if t]
        ents = [asr_terms._entry(e) for e in entries or ()]
        ents = [e for e in ents if e.term]
        known = {e.term.lower() for e in ents}
        ents += [asr_terms.VocabEntry(t) for t in terms if t.lower() not in known]
        if self.dead:
            return "dictionary not applied: Metal ASR is off"
        self.entries = ents
        if self.rescore:
            self.terms = terms or None
            n = len(asr_terms.merge_vocabulary(ents, max_terms=None))
            if not ents:
                return "dictionary rescoring off"
            return f"dictionary rescoring {n} of {len(ents)} terms (acoustic span rescoring)"
        if self.p is None:
            self.p = self._spawn()
        flat = asr_terms.flat_trie(self.model.vocabulary, terms) if terms else None
        words = flat if flat is not None else np.zeros(0, dtype=np.uint32)
        try:
            self.p.stdin.write(
                struct.pack("<2I", 0xFFFFFFFF, len(words))
                + words.astype("<u4").tobytes()
            )
            self.p.stdin.flush()
            (st,) = struct.unpack("<I", self._read(4, first=True))
        except Exception as ex:  # noqa: BLE001  (the helper exited or did not answer)
            self._off(f"dictionary: {ex!r}", stuck=isinstance(ex, TimeoutError))
            return "dictionary not applied: Metal ASR is off"
        if st != 0:
            return f"dictionary of {len(terms)} terms refused ({'too large' if st == 1 else 'malformed'})"
        self.terms = terms if flat is not None else None
        if not self.terms:
            return "dictionary boosting off"
        return f"dictionary boosting {len(terms)} terms ({len(words) * 4 // 1024} KB)"

    def set_screen_terms(self, screen_terms):
        """Terms read off the screen for the dictation under way: rescoring candidates beside the dictionary
        until replaced or cleared ([]). Returns a line for the log."""
        self.screen_terms = [
            t for t in dict.fromkeys(str(t).strip() for t in screen_terms or ()) if t
        ]
        if not self.screen_terms:
            return "screen terms cleared"
        if not self.rescore:
            return f"{len(self.screen_terms)} screen terms kept (rescoring is off: not used)"
        return f"{len(self.screen_terms)} screen terms for this dictation"

    def rescore_vocabulary(self):
        """The rescoring vocabulary: all deduplicated dictionary entries and this dictation's screen terms.
        Post-greedy candidate matching is pure Python and caps acoustic rescoring at max_candidates per window,
        so no alphabetical dictionary truncation is needed."""
        return _asr_terms().merge_vocabulary(
            self.entries, self.screen_terms, max_terms=None
        )

    def rescore_spans(self, jobs, timeout_s=None):
        """jobs: (f_start, f_end, prefix tokens, heard tokens, candidate tokens) over the last decoded window ->
        [(log P(heard), log P(cand))] from pa (request 0xFFFFFFFE), or None when the helper is off or fails (a
        failure counts as a failed window: the helper restarts or turns off)."""
        if self.dead or self.p is None:
            return None
        jobs = list(jobs)
        if not jobs:
            return []
        parts = [struct.pack("<2I", 0xFFFFFFFE, len(jobs))]
        for f0, f1, prefix, heard, cand in jobs:
            parts.append(struct.pack("<2I", int(f0), int(f1)))
            for seq in (prefix, heard, cand):
                seq = np.asarray(seq, dtype="<u4")
                parts.append(struct.pack("<I", len(seq)) + seq.tobytes())
        try:
            self.p.stdin.write(b"".join(parts))
            self.p.stdin.flush()
            raw = self._read(4, first=True, timeout_s=timeout_s) + self._read(
                8 * len(jobs) - 4
            )
        except Exception as ex:  # noqa: BLE001  (the helper exited, its pipe broke or it did not answer)
            self._recover_or_off(f"rescore: {ex!r}", stuck=isinstance(ex, TimeoutError))
            return None
        sc = np.frombuffer(raw, dtype="<f4").reshape(len(jobs), 2)
        return [(float(a), float(b)) for a, b in sc]

    def rescore_rows(self, rows, n_frames):
        """Dictionary terms into the greedy decode's rows: exact spoken forms first, then acoustic rescoring of the
        spans that look like a term. -> [(row start, row end, text)], sorted, non-overlapping (the rows'
        trailing punctuation stays outside); self.last_rescore keeps each rescored span's scores."""
        self.last_rescore = []
        vocab = self.rescore_vocabulary()
        if not vocab or not rows:
            return []
        at = _asr_terms()
        voc = self.model.vocabulary
        words = at.align_words(
            rows,
            voc,
            n_frames,
            tuple(getattr(self.model, "durations", (0, 1, 2, 3, 4))),
        )
        reps = []
        locked = set()
        for r in at.apply_spoken_forms(words, vocab):
            a, b = words[r.w_start], words[r.w_end - 1]
            if b.t_end > a.t_start:
                reps.append((a.t_start, b.t_end, r.text))
                locked.update(range(r.w_start, r.w_end))
        if self._encode is None:
            enc, unk = at.bpe_encoder(voc)

            def encode(text):  # no candidate for a spelling the vocabulary cannot write
                ids = enc(text)
                return [] if unk in ids else ids

            self._encode = encode
        cands = [
            c
            for c in at.find_rescore_candidates(
                words, vocab, encode=self._encode, skip=locked
            )
            if c.cand and c.heard and c.f_end > c.f_start
        ]
        if cands:
            toks = [r[0] for r in rows]
            scores = self.rescore_spans(
                [
                    (c.f_start, c.f_end, toks[: c.t_start], c.heard, c.cand)
                    for c in cands
                ]
            )
            boost = at.boost_nats()
            for c, sc in zip(cands, scores or ()):
                ok = at.should_replace(sc[0], sc[1], len(c.cand), boost)
                heard = " ".join(w.text for w in words[c.w_start : c.w_end])
                self.last_rescore.append((heard, c.text, sc[0], sc[1], ok))
                if ok:
                    reps.append((c.t_start, c.t_end, c.text))
        return sorted(reps)

    def warmup(self):
        """One short window: the Metal pipelines and the weight pages ready before the first dictation."""
        self.transcribe_samples(np.zeros(16000, dtype=np.float32))

    def self_test(self, budget_ms=250.0, load=True, load_budget_ms=1000.0):
        """Startup gate: 1 s windows through the helper itself (not the fallback): a cold one, a warm one that
        must take under budget_ms of helper wall time, then one while `pa busy` (another process) keeps 16
        threadgroups spinning on the GPU, as the refiner's persistent kernel does during a dictation, which must
        answer within load_budget_ms (a stall guard: a shared GPU only slows the decode, and MLX would share it
        too). Returns (passed, warm window ms); on a failure the caller stays on MLX, so a Mac that cannot run
        the decode under load finds out at startup, not mid-dictation."""
        x = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
        busy = None
        try:
            for _ in range(2):
                if self.tokens(x, samples=True) is None:
                    return False, self.last_ms[2]
            warm = self.last_ms[2]
            if not load:
                return warm < budget_ms, warm
            busy = self._busy()
            if self.tokens(x, samples=True) is None:
                return False, warm
            loaded = self.last_ms[2]
            print(
                f"pa_client: self-test {warm:.1f} ms, {loaded:.1f} ms under load",
                file=sys.stderr,
            )
            return warm < budget_ms and loaded < load_budget_ms, warm
        except Exception as ex:  # noqa: BLE001  (the helper exited or did not answer)
            self._off(f"self-test: {ex!r}", stuck=isinstance(ex, TimeoutError))
            return False, -1.0
        finally:
            if busy is not None:
                busy.kill()
                busy.wait()

    def _busy(self, wait_s=5.0):
        """`pa busy 16 10`: the self-test's concurrent GPU load, started and running (or None if it fails to)."""
        try:
            b = subprocess.Popen(
                [self.pa_bin, "busy", "16", "10"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return None
        if select.select([b.stdout], [], [], wait_s)[0]:
            b.stdout.readline()  # "busy: running"
        return b

    def close(self):
        if self.p is not None and self.p.poll() is None:
            try:
                self.p.stdin.write(struct.pack("<I", 0))
                self.p.stdin.flush()
                self.p.wait(timeout=10)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self.p.kill()
                self.p.wait()

    def _stop_proc(self, stuck=False):
        if self.p is None:
            return
        try:
            if stuck:
                self.p.kill()
                self.p.wait()
            else:
                self.close()
        except Exception as ex:  # noqa: BLE001  (already gone)
            print(f"pa_client: stopping the helper: {ex!r}", file=sys.stderr)

    def _off(self, why, stuck=False):
        """Sticky fallback: the helper stays off for the rest of the session (logged once) and is stopped (killed
        when stuck), so a failing GPU path costs one window, not every window."""
        if self.dead:
            return
        self.dead = True
        print(
            f"pa_client: Metal ASR off for this session ({why}); transcribing on MLX",
            file=sys.stderr,
        )
        self._stop_proc(stuck=stuck)

    def _recover_or_off(self, why, stuck=False):
        """Fall back to MLX for this window and either schedule a helper respawn or latch off once
        consecutive failures exceed max_restarts."""
        self._consecutive_failures += 1
        if self._consecutive_failures > self._max_restarts:
            self._off(why, stuck=stuck)
            return
        self._stop_proc(stuck=stuck)
        self.p = None
        print(
            f"pa_client: Metal ASR window fell back to MLX ({why}); will restart helper "
            f"({self._consecutive_failures}/{self._max_restarts})",
            file=sys.stderr,
        )

    def _timeout_for(self, frames):
        """Scale timeout for long meeting spans (>20 s) so GPU contention does not falsely kill pa."""
        if frames > 2000:
            return max(self.timeout_s, 30.0)
        return self.timeout_s

    def _read(self, n, first=False, timeout_s=None):
        """n bytes of the helper's answer. first (an answer's first bytes; the pipe's buffer is empty then):
        raises TimeoutError when nothing arrives within timeout_s (a stuck GPU path)."""
        wait_s = self.timeout_s if timeout_s is None else timeout_s
        if first and not select.select([self.p.stdout], [], [], wait_s)[0]:
            raise TimeoutError(f"no answer from pa in {wait_s:g} s")
        b = self.p.stdout.read(n)
        if len(b) != n:
            raise EOFError("pa exited")
        return b

    def tokens(self, mel: np.ndarray, samples: bool = False, timeout_s=None):
        """mel [n, 128] float32 (or samples [n] float32 with samples=True) -> [(token, duration index, frame,
        confidence)], or None when pa could not answer the window (over its capacity, or a failed decode)"""
        n = mel.shape[0]
        head = struct.pack("<I", (n | 0x80000000) if samples else n)
        self.p.stdin.write(head + np.ascontiguousarray(mel, dtype=np.float32).tobytes())
        self.p.stdin.flush()
        (cnt,) = struct.unpack("<I", self._read(4, first=True, timeout_s=timeout_s))
        rows = []
        if cnt != 0xFFFFFFFF:
            raw = np.frombuffer(self._read(16 * cnt), dtype="<u4").reshape(cnt, 4)
            conf = raw[:, 3].copy().view("<f4")
            rows = [
                (int(t), int(d), int(s), float(c)) for (t, d, s, _), c in zip(raw, conf)
            ]
        self.last_ms = struct.unpack("<3f", self._read(12))
        # committed dictionary spans, then flips / rollbacks
        (nsp,) = struct.unpack("<I", self._read(4))
        sp = (
            np.frombuffer(self._read(16 * nsp), dtype="<u4").reshape(nsp, 4)
            if nsp
            else np.zeros((0, 4), np.uint32)
        )
        self.last_spans = [tuple(int(x) for x in s) for s in sp]
        self.last_boost = struct.unpack("<2I", self._read(8))
        return None if cnt == 0xFFFFFFFF else rows

    def metal_rows(self, samples, frames, mel_rows=None):
        """The helper's rows for one window, or None for the MLX path: over capacity, or the helper is off.
        Transient timeouts or helper exits respawn pa up to max_restarts consecutive times (re-sending any
        active dictionary); repeated failures latch Metal ASR off for the session."""
        if frames > self.max_mel or self.dead:
            return None
        try:
            if self.p is None:
                self.p = self._spawn()
                if self.terms and not self.rescore:
                    saved_terms = list(self.terms)
                    self.set_terms(saved_terms, entries=list(self.entries))
                    if self.dead:
                        return None
            wait_s = self._timeout_for(frames)
            rows = (
                self.tokens(samples, samples=True, timeout_s=wait_s)
                if self.gpu_mel
                else self.tokens(mel_rows(), timeout_s=wait_s)
            )
        except Exception as ex:  # noqa: BLE001  (the helper exited, its pipe broke or it did not answer)
            self._recover_or_off(
                f"helper failed: {ex!r}", stuck=isinstance(ex, TimeoutError)
            )
            return None
        if rows is None:  # a window within capacity that pa could not decode
            self._recover_or_off("pa could not decode a window")
            return None
        self._consecutive_failures = 0
        return rows

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

        def mel_rows():
            mel = get_logmel(mx.array(samples, dtype=mx.float32), m.preprocessor_config)
            return np.array(mel[0] if mel.ndim == 3 else mel, dtype=np.float32)

        rows = self.metal_rows(samples, frames, mel_rows)
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
        if rows is None:  # over capacity, or the helper is off: the MLX path
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
                text=(
                    ""
                    if 0 <= t < len(m.vocabulary) and m.vocabulary[t] == "<unk>"
                    else tokenizer.decode([t], m.vocabulary)
                ),
            )
            for t, d, s, c in rows
        ]
        if self.rescore:
            try:
                reps = self.rescore_rows(rows, _encoder_frames(frames))
            except Exception as ex:  # noqa: BLE001  (rescoring must never cost the window its transcript)
                print(f"pa_client: rescoring skipped: {ex!r}", file=sys.stderr)
                reps = []
            hyp = _rewrite(hyp, reps, m.vocabulary)
        # committed terms take the dictionary spelling (tdt_decode detok)
        elif self.terms and self.rewrite != "none":
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


def _asr_terms():
    try:  # the sidecar dir on sys.path, or imported as a package
        import asr_terms
    except ImportError:
        from . import asr_terms
    return asr_terms


def _encoder_frames(mel_frames):
    """The encoder frames of a window of mel_frames (three stride-2 subsamplings, as pa's sub_len)."""
    t = mel_frames
    for _ in range(3):
        t = (t - 1) // 2 + 1
    return t


def _rewrite(hyp, reps, vocabulary):
    """AlignedTokens with each (row start, row end, text) span as one token carrying the text (after a space when
    its first piece starts a word), spanning the span's time, at its lowest confidence."""
    from parakeet_mlx.alignment import AlignedToken

    out, prev = [], 0
    for a, b, text in sorted(reps):
        if a < prev or b <= a or b > len(hyp):
            continue
        out += hyp[prev:a]
        first = hyp[a]
        lead = " " if vocabulary[first.id].startswith("▁") else ""
        out.append(
            AlignedToken(
                first.id,
                start=first.start,
                duration=hyp[b - 1].start + hyp[b - 1].duration - first.start,
                confidence=min(h.confidence for h in hyp[a:b]),
                text=lead + text,
            )
        )
        prev = b
    return out + hyp[prev:]
