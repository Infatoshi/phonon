"""Streaming refiner engine: a rollback-able KV cache plus draft-verified greedy decoding.

The refiner prompt is system (fixed instructions + per-request vocabulary line) + transcript.
During dictation the engine is fed the transcript of closed speech segments; it keeps the KV
state of the longest prompt prefix it has seen, so at key release only the tokens after the
longest common prefix are prefilled. Drafts (the refiner's own output for the transcript so far)
are verified in blocks against the model's greedy choice, so the final text is the greedy output
of the full prompt whatever the draft says: lossless by construction, up to the floating point
difference between a batched and a one-token forward.

LFM3 has three layer kinds. Full attention uses a plain KVCache (trimmable). Sliding attention
also uses a plain KVCache with a windowed mask instead of mlx-lm's RotatingKVCache, because a
rotated cache cannot be trimmed back. Short-conv layers keep their whole input history so the
6-sample conv state can be restored at any position.

The decode loop (extend, greedy, speculative) lives in stream_decode.RollbackDecoder and is
shared with the native engine's session backend (stream_decode.PeEngine); StreamSession runs on
either.
"""

from __future__ import annotations

import mlx.core as mx

try:  # imported as sidecar.stream_refiner (tests) or as a top-level module (the sidecar)
    from .stream_decode import (
        Aborted,
        EngineError,
        PeEngine,
        RollbackDecoder,
        StreamSession,
    )
    from .stream_draft import AdaptiveProposer, DraftProposer, trace_speculative
except ImportError:
    from stream_decode import (  # noqa: F401
        Aborted,
        EngineError,
        PeEngine,
        RollbackDecoder,
        StreamSession,
    )
    from stream_draft import AdaptiveProposer, DraftProposer, trace_speculative  # noqa: F401


class ConvHistoryCache:
    """Conv state with rollback: keeps every Bx row so the state at any offset is recoverable."""

    def __init__(self, n_keep: int):
        self.n_keep = n_keep
        self.hist: mx.array | None = None  # (B, cap, D); rows [0, n_keep) are zeros
        self.offset = 0

    def _ensure(self, bx: mx.array, extra: int) -> None:
        need = self.n_keep + self.offset + extra
        if self.hist is None:
            cap = max(1024, need + 512)
            self.hist = mx.zeros((bx.shape[0], cap, bx.shape[2]), dtype=bx.dtype)
        elif need > self.hist.shape[1]:
            grow = mx.zeros(
                (
                    bx.shape[0],
                    max(need, 2 * self.hist.shape[1]) - self.hist.shape[1],
                    bx.shape[2],
                ),
                dtype=bx.dtype,
            )
            self.hist = mx.concatenate([self.hist, grow], axis=1)

    def window(self, bx: mx.array) -> mx.array:
        """State rows followed by bx, and record bx into the history."""
        t = bx.shape[1]
        self._ensure(bx, t)
        start = self.offset  # hist index of the first state row
        state = self.hist[:, start : start + self.n_keep, :]
        self.hist[:, start + self.n_keep : start + self.n_keep + t, :] = bx
        self.offset += t
        return mx.concatenate([state, bx], axis=1)

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def state(self):
        return [] if self.hist is None else [self.hist]

    def make_mask(self, n: int):
        return None


def _patch_conv(model) -> None:
    """Route ShortConv through ConvHistoryCache when it is the cache in use."""
    from mlx_lm.models.lfm2 import ShortConv

    if getattr(ShortConv, "_phonon_patched", False):
        return
    original = ShortConv.__call__

    def call(self, x, mask=None, cache=None):
        if not isinstance(cache, ConvHistoryCache):
            return original(self, x, mask=mask, cache=cache)
        bcx = self.in_proj(x)
        b, c, xx = mx.split(bcx, 3, axis=-1)
        bx = b * xx
        bx = cache.window(bx)
        return self.out_proj(c * self.conv(bx))

    ShortConv.__call__ = call
    ShortConv._phonon_patched = True


class Engine(RollbackDecoder):
    """The MLX backend of the decode loop: the refiner's weights in this process, with a
    rollback-able cache (plain KVCache per attention layer, ConvHistoryCache per conv layer)."""

    def __init__(self, model, tokenizer, prefill_step: int = 512):
        from mlx_lm.models.cache import KVCache

        _patch_conv(model)
        eos = getattr(tokenizer, "eos_token_ids", None) or {tokenizer.eos_token_id}
        super().__init__(set(eos))
        self.model = model
        self.tokenizer = tokenizer
        self.prefill_step = prefill_step
        self.caches = []
        for layer in model.layers:
            if layer.layer_type == "conv":
                self.caches.append(ConvHistoryCache(model.args.conv_L_cache - 1))
            else:
                self.caches.append(KVCache())

    # -- backend -------------------------------------------------------------
    def _truncate(self, n_keep: int) -> None:
        drop = len(self.tokens) - n_keep
        for c in self.caches:
            c.trim(drop)

    def _logits(self, toks: list[int]) -> mx.array:
        """Forward toks after the cache (prefill_step tokens per model call); logits (len, V)."""
        out = []
        for i in range(0, len(toks), self.prefill_step):
            part = toks[i : i + self.prefill_step]
            out.append(self.model(mx.array(part)[None], cache=self.caches)[0])
        return out[0] if len(out) == 1 else mx.concatenate(out, axis=0)

    def _pass(self, toks, row0, tie_from, tie_margin):
        logits = self._logits(toks)[row0:]
        ids = mx.argmax(logits, axis=-1).tolist()
        if tie_from is None or tie_margin <= 0:
            return ids, None
        top2 = mx.topk(logits.astype(mx.float32), 2, axis=-1)
        ties = (mx.abs(top2[:, 0] - top2[:, 1]) < tie_margin).tolist()
        return ids, next((i for i, t in enumerate(ties) if t and i >= tie_from), None)

    def _forward(self, toks: list[int]) -> mx.array:
        """Benchmarks: forward toks after the cache, keep them, return the logits (len, V)."""
        logits = self._logits(toks)
        self.tokens.extend(toks)
        self.next_id = None
        self.stats["forwards"] += 1
        self.stats["forward_tokens"] += len(toks)
        return logits
