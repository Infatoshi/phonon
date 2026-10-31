"""Transcribe a dictation while it is spoken, with whole-recording quality at release.

Audio streams in (0.4 s chunks from the app). A causal VAD (stream_vad.py) closes a segment at
each pause; once `after_s` of audio past the segment has arrived, the segment is transcribed inside
a window with `before_s` of audio before it and `after_s` after it, keeping only the words whose
first token starts inside the segment. Every `round_every` chunks a speculative round transcribes
the audio after the last transcribed piece (with `before_s` of left context), so the transcript
so far reaches the current moment. At release only [t_cut - release_before_s, end] is transcribed,
where t_cut = (end of the last round) - release_overlap_s: words before t_cut come from the round,
which heard them with at least release_overlap_s of right context. With soft_max_s > 0 a round
that finds more than soft_max_s of pending audio also commits its words up to
(end - soft_overlap_s) as a piece, so rounds stay short in long stretches without pauses.

The app uses SCHEME: no pause cuts, so every round transcribes the whole recording so far, and
the release re-transcribes from 8 s before the last round's cut with 2 s of overlap. Past 60 s of
pending audio a round commits its words and later rounds keep 20 s of left context, so no window
exceeds 80.4 s (Metal Parakeet's capacity is 82 s). On pubeval_v0 (1,351 public clips of 3-60 s
with human transcripts), windowed minus whole-file verbatim WER, paired bootstrap 95% CI:
SCHEME +0.0009 [-0.0006, +0.0023], every length band's CI includes 0; the same rounds with a 1 s
release window +0.0047 [+0.0029, +0.0064]; pause-cut segments with 4 s of left context and a 1 s
release window +0.0119 [+0.0093, +0.0147]; commits every 30 s with 20 s of left context
+0.0044 [+0.0015, +0.0079] (the 30-60 s band +0.010). Parakeet needs the long left context.

When the speaker stopped before letting go, the release needs no transcription: if every frame
from quiet_reuse_s before the last round's end to the end of the audio is silence, the release is
that round's transcript (it heard every word, with the whole recording as left context, and the
refiner already has it), and rounds pause while that holds, so the refiner can finish its draft.
pubeval_v0 with SCHEME: +0.0005 [-0.0010, +0.0019]. On 40 of the owner's recordings with their real
key-up silence it applied to about a third of releases (the median pause before letting go is
0.24 s), and those reached refined text 12 ms after key up instead of 30 ms (2 runs).

`transcribe_words(samples, offset_s)` returns [(word, absolute start s), ...]; MLX lives in the
caller, so this module is testable without it.
"""

from __future__ import annotations

import numpy as np

try:
    from .stream_vad import FRAME, StreamVAD
except ImportError:
    from stream_vad import FRAME, StreamVAD

FRAME_S = 0.02
# the app's settings (asr_server.stream_transcriber); see the module docstring for the measurements
SCHEME = {
    "pause_s": 1e9,  # no pause cuts: rounds transcribe the whole recording so far
    "before_s": 20.0,  # left context of rounds after a commit
    "soft_max_s": 60.0,  # commit past 60 s of pending audio: windows stay under 82 s
    "soft_overlap_s": 2.0,
    "release_before_s": 8.0,
    "release_overlap_s": 2.0,
    "quiet_reuse_s": 0.1,  # release = the last round when only silence followed it
    "quiet_skip": True,  # and no rounds over that silence
}


def words_of(result, offset_s: float) -> list[tuple[str, float]]:
    """Words with absolute start times from a parakeet-mlx AlignedResult."""
    out: list[list] = []
    for sentence in result.sentences:
        for token in sentence.tokens:
            if not out or token.text.startswith(" "):
                out.append([token.text, token.start + offset_s])
            else:
                out[-1][0] += token.text
    return [(w.strip(), s) for w, s in out if w.strip()]


def select(words, start_f: int, end_f: int) -> str:
    a, b = start_f * FRAME_S, end_f * FRAME_S
    return " ".join(w for w, s in words if a <= s < b).strip()


def join(texts) -> str:
    return " ".join(t.strip() for t in texts if t and t.strip()).strip()


class StreamTranscriber:
    def __init__(
        self,
        transcribe_words,
        pause_s: float = 0.5,
        before_s: float = 4.0,
        after_s: float = 2.0,
        release_before_s: float = 1.0,
        release_overlap_s: float = 0.5,
        round_every: int = 1,
        min_seg_s: float = 1.0,
        soft_max_s: float = 0.0,
        soft_overlap_s: float = 1.0,
        quiet_reuse_s: float = 0.0,
        quiet_skip: bool = False,
    ):
        self.tw = transcribe_words
        self.vad = StreamVAD(pause_s, min_seg_s)
        self.before_f = round(before_s / FRAME_S)
        self.after_f = round(after_s / FRAME_S)
        self.rel_before_f = round(release_before_s / FRAME_S)
        self.overlap_f = round(release_overlap_s / FRAME_S)
        self.soft_f = round(soft_max_s / FRAME_S)
        self.soft_overlap_f = round(soft_overlap_s / FRAME_S)
        self.round_every = round_every
        self.audio = np.zeros(16000 * 30, dtype=np.float32)
        self.n = 0
        self.cuts: list[int] = []  # VAD pause cuts, in frames
        self.next_cut = 0  # first cut whose segment is not transcribed yet
        self.pieces: list[
            dict
        ] = []  # committed {"s", "e", "text"}, contiguous from frame 0
        self.pend = 0  # end of the last piece: everything after it is pending
        self.chunks = 0
        self.round_chunk = 0  # chunk count at the last round
        self.last_round = None  # (pend, f1, pieces before the round, words)
        self.stable = ""  # transcript so far without the words the release may revise
        self.asr_calls = 0
        self.soft_cuts = 0
        # quiet_reuse_s > 0: when nothing but silence follows (the last round's end - quiet_reuse_s),
        # the release is that round's transcript; quiet_skip: no round while that holds either
        self.quiet_f = round(quiet_reuse_s / FRAME_S) if quiet_reuse_s > 0 else -1
        self.quiet_skip = quiet_skip and self.quiet_f >= 0
        self.reused = False
        self.skipped_rounds = 0

    # -- audio ----------------------------------------------------------------------------
    def _append(self, samples) -> list[tuple[int, int]]:
        need = self.n + len(samples)
        if need > len(self.audio):
            grown = np.zeros(max(need, 2 * len(self.audio)), dtype=np.float32)
            grown[: self.n] = self.audio[: self.n]
            self.audio = grown
        self.audio[self.n : need] = samples
        self.n = need
        return self.vad.feed(samples)

    def frames(self) -> int:
        return (self.n + FRAME - 1) // FRAME

    def _words(self, start_f: int, end_f: int):
        self.asr_calls += 1
        x = self.audio[start_f * FRAME : min(end_f * FRAME, self.n)]
        return self.tw(x, start_f * FRAME_S)

    def _speech(self, start_f: int, end_f: int) -> bool:
        return self.vad.has_speech(start_f, end_f)

    def _quiet_after_round(self) -> bool:
        """Every frame since quiet_f before the last round's end is silence to the VAD and far
        below speech: under the recording's 5th percentile frame RMS plus a tenth of the way to
        its 95th. Both must agree: the VAD's moving floor rises to speech level in long stretches
        without pauses, and fixed levels can miss soft speech next to loud speech."""
        if self.quiet_f < 0 or self.last_round is None:
            return False
        h = self.vad.hist
        f = max(0, self.last_round[1] - self.quiet_f)
        if f >= len(h) or self._speech(f, len(h)):
            return False
        lv = 10 ** (np.asarray(h, dtype=np.float64) / 20)
        lo, hi = np.percentile(lv, 5), np.percentile(lv, 95)
        return bool(lv[f:].max() < lo + 0.1 * (hi - lo))

    def committed(self, k: int | None = None) -> str:
        return join(
            p["text"] for p in self.pieces[: len(self.pieces) if k is None else k]
        )

    # -- streaming --------------------------------------------------------------------------
    def feed(self, samples, work: bool = True) -> list[str]:
        """Add a chunk. Returns the transcript so far when it changed (at most one string).
        With work=False only the audio and the VAD advance (the caller is behind real time and
        folds several chunks into one pass, or the release is already queued).
        self.stable then holds the same transcript without the round's last
        release_overlap_s, the words the release window may still revise."""
        for cut, _detected in self._append(np.asarray(samples, dtype=np.float32)):
            self.cuts.append(cut)
        if not work:
            self.chunks += 1
            return []
        have = self.frames()
        changed = False
        while (
            self.next_cut < len(self.cuts)
            and have >= self.cuts[self.next_cut] + self.after_f
        ):
            e = self.cuts[self.next_cut]
            self.next_cut += 1
            if e <= self.pend:
                continue  # a soft cut already committed this stretch
            s = self.pend
            # The VAD only decides where to cut, never what to transcribe: a segment it
            # scores as silence is still transcribed. On recordings with little silence its
            # noise floor sits at speech level, and skipping such segments dropped words
            # (public pubeval_v0: +0.023 WER against whole-file ASR, empty transcripts).
            text = select(
                self._words(max(0, s - self.before_f), e + self.after_f), s, e
            )
            self.pieces.append({"s": s, "e": e, "text": text})
            self.pend = e
            changed = True
        out = None
        if changed:
            out = self.stable = self.committed()
        self.chunks += 1
        if self.round_every and self.chunks - self.round_chunk >= self.round_every:
            self.round_chunk = self.chunks
            pend = self.pend
            # Rounds start with the first speech of the stream and then run whatever the VAD says: in
            # long stretches without pauses its noise floor rises to speech level, and rounds gated on
            # the pending audio stopped, leaving the release window everything since the last round.
            if self.quiet_skip and self._quiet_after_round():
                # only silence since the last round: it heard every word
                self.skipped_rounds += 1
            elif have - pend >= 25 and self._speech(0, have):
                words = self._words(max(0, pend - self.before_f), have)
                k = len(self.pieces)
                self.last_round = (pend, have, k, words)
                out = join([self.committed(k), select(words, pend, have)])
                self.stable = join(
                    [self.committed(k), select(words, pend, have - self.overlap_f)]
                )
                c = have - self.soft_overlap_f
                if self.soft_f and have - pend > self.soft_f and c - pend >= 25:
                    self.pieces.append(
                        {"s": pend, "e": c, "text": select(words, pend, c)}
                    )
                    self.pend = c
                    self.soft_cuts += 1
        return [out] if out else []

    def finish(self, tail=None) -> str:
        """Key release: the rest of the audio (`tail`, if the stream missed it) then the final
        transcript."""
        if tail is not None and len(tail):
            self._append(np.asarray(tail, dtype=np.float32))
        end = self.frames()
        if self.last_round is not None:
            pend, f1, k, words = self.last_round
            t_cut = f1 - self.overlap_f
            if t_cut - pend >= 25 and self._quiet_after_round():
                # Only silence since just before the round's end: the round heard every word, with
                # the whole recording as left context, and its transcript went to the refiner.
                self.reused = True
                return join([self.committed(k), select(words, pend, end)])
            if t_cut - pend >= 25:
                # Always transcribe the release window, even when the VAD hears nothing after
                # the cut: it under-detects speech when its noise floor is at speech level.
                rel = self._words(max(0, t_cut - self.rel_before_f), end)
                return join(
                    [
                        self.committed(k),
                        select(words, pend, t_cut),
                        select(rel, t_cut, end),
                    ]
                )
        pend = self.pend
        if end > pend:  # no usable round: the open stretch, whatever the VAD says
            words = self._words(max(0, pend - self.before_f), end)
            return join([self.committed(), select(words, pend, end)])
        return self.committed()
