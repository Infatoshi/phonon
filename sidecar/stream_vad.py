"""Causal energy VAD that cuts streamed audio into speech segments at pauses.

Frames of 20 ms at 16 kHz; a frame is speech when its RMS level is above max(noise floor + 12 dB,
-55 dBFS), with three frames of hangover. The noise floor is the 10th percentile of the last 30 s
of frame levels, refreshed every 10 frames. A cut is placed in the middle of a pause once the
pause has lasted `pause_s` and the segment since the previous cut is at least `min_seg_s` long.
Feeding audio in any chunking gives the same cuts (checked against the offline form in the
stream lane's t_vad.py)."""

from __future__ import annotations

import numpy as np

FRAME = 320  # 20 ms at 16 kHz


def frame_db(audio: np.ndarray) -> np.ndarray:
    n = len(audio) // FRAME
    if n == 0:
        return np.zeros(0)
    x = audio[: n * FRAME].reshape(n, FRAME).astype(np.float64)
    rms = np.sqrt((x * x).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


class StreamVAD:
    def __init__(
        self,
        pause_s: float = 0.5,
        min_seg_s: float = 1.0,
        margin: float = 12.0,
        abs_floor: float = -55.0,
        hang: int = 3,
    ):
        self.need = round(pause_s / 0.02)
        self.min_seg = min_seg_s
        self.margin, self.abs_floor, self.hang = margin, abs_floor, hang
        self.buf = np.zeros(0, dtype=np.float32)
        self.hist: list[float] = []
        self.floor: float | None = None
        self.cnt = 0
        self.i = 0  # frames processed
        self.mask: list[bool] = []
        self.seg_start = 0
        self.sil = 0
        self.had_speech = False

    def feed(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Process a chunk; returns the cuts it completed as (cut_frame, detect_frame)."""
        self.buf = np.concatenate([self.buf, audio.astype(np.float32)])
        n = len(self.buf) // FRAME
        if n == 0:
            return []
        db = frame_db(self.buf[: n * FRAME])
        self.buf = self.buf[n * FRAME :]
        out = []
        for v in db:
            i = self.i
            self.hist.append(float(v))
            if i % 10 == 0 or self.floor is None:
                self.floor = float(np.percentile(self.hist[-1500:], 10))
            thr = max(self.floor + self.margin, self.abs_floor)
            if v > thr:
                speech = True
                self.cnt = self.hang
            elif self.cnt > 0:
                speech = True
                self.cnt -= 1
            else:
                speech = False
            self.mask.append(speech)
            if speech:
                self.had_speech = True
                self.sil = 0
            else:
                self.sil += 1
                if (
                    self.had_speech
                    and self.sil == self.need
                    and (i - self.seg_start) * 0.02 >= self.min_seg
                ):
                    cut = i - self.need // 2
                    out.append((cut, i))
                    self.seg_start = cut
                    self.had_speech = False
            self.i += 1
        return out

    def has_speech(self, start: int, end: int) -> bool:
        return any(self.mask[start:end])
