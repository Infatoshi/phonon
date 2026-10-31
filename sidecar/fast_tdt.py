"""Faster greedy TDT decoding for parakeet-mlx with identical tokens.

parakeet-mlx's ParakeetTDT.decode_greedy reruns the prediction network on every frame, even
after a blank (when its input and state have not changed), and syncs with the GPU three times
per step (token argmax, confidence, duration argmax). This version reuses the decoder output
until a token is emitted and reads token, duration and confidence in one sync. Same arithmetic
in the same order, so the hypothesis is the same token for token.
"""

from __future__ import annotations

import mlx.core as mx


def decode_greedy_fast(
    self,
    features: mx.array,
    lengths: mx.array | None = None,
    last_token=None,
    hidden_state=None,
    *,
    config=None,
):
    from parakeet_mlx import tokenizer
    from parakeet_mlx.alignment import AlignedToken

    B, S, *_ = features.shape
    if hidden_state is None:
        hidden_state = list([None] * B)
    if lengths is None:
        lengths = mx.array([S] * B)
    if last_token is None:
        last_token = list([None] * B)
    nv = len(self.vocabulary)
    vocab_size = nv + 1
    max_entropy = mx.log(mx.array(vocab_size, dtype=features.dtype))
    results = []
    for batch in range(B):
        hypothesis = []
        feature = features[batch : batch + 1]
        length = int(lengths[batch])
        step = 0
        new_symbols = 0
        dec_cache = (
            None  # (decoder_out, decoder_hidden) for the current (last_token, state)
        )
        while step < length:
            if dec_cache is None:
                decoder_out, (hidden, cell) = self.decoder(
                    mx.array([[last_token[batch]]])
                    if last_token[batch] is not None
                    else None,
                    hidden_state[batch],
                )
                decoder_out = decoder_out.astype(feature.dtype)
                decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )
                dec_cache = (decoder_out, decoder_hidden)
            decoder_out, decoder_hidden = dec_cache
            joint_out = self.joint(feature[:, step : step + 1], decoder_out)
            token_logits = joint_out[0, 0, :, :vocab_size]
            pred = mx.argmax(token_logits)
            probs = mx.softmax(token_logits, axis=-1)
            entropy = -mx.sum(probs * mx.log(probs + 1e-10), axis=-1)
            conf = 1.0 - (entropy / max_entropy)
            dec = mx.argmax(joint_out[0, 0, :, vocab_size:])
            conf = conf.reshape(-1)[0]
            # Speculatively run the prediction network on this step's token in the same
            # evaluation; it is used only if the token is not blank.
            nxt_out, (nh, nc) = self.decoder(
                mx.minimum(pred, nv - 1).reshape(1, 1).astype(mx.int32), decoder_hidden
            )
            nxt = (
                nxt_out.astype(feature.dtype),
                (nh.astype(feature.dtype), nc.astype(feature.dtype)),
            )
            mx.eval(pred, dec, conf, nxt[0], nxt[1][0], nxt[1][1])
            pred_token, decision, confidence = pred.item(), dec.item(), conf.item()
            pred_token = int(pred_token)
            decision = int(decision)
            if pred_token != nv:
                hypothesis.append(
                    AlignedToken(
                        pred_token,
                        start=step * self.time_ratio,
                        duration=self.durations[decision] * self.time_ratio,
                        confidence=float(confidence),
                        text=tokenizer.decode([pred_token], self.vocabulary),
                    )
                )
                last_token[batch] = pred_token
                hidden_state[batch] = decoder_hidden
                dec_cache = nxt
            step += self.durations[decision]
            new_symbols += 1
            if self.durations[decision] != 0:
                new_symbols = 0
            else:
                if self.max_symbols is not None and self.max_symbols <= new_symbols:
                    step += 1
                    new_symbols = 0
        results.append(hypothesis)
    return results, hidden_state


def install() -> None:
    from parakeet_mlx.parakeet import ParakeetTDT

    ParakeetTDT.decode_greedy = decode_greedy_fast
