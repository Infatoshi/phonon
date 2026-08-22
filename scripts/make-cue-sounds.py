#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Synthesize the two recording cues shipped in assets/.

Each cue is three layers: a swept sub sine for weight, band-passed noise for
movement, and a dark convolution tail for space. Rising cue opens a recording,
falling cue closes it. Nothing is sampled or downloaded, so the assets stay
reproducible from this file alone.
"""

import struct
import sys
from pathlib import Path

import numpy as np

RATE = 48_000
PEAK_DBFS = -15.0


def band_sweep(
    duration: float, f0: float, f1: float, q: float, stages: int, seed: int
) -> np.ndarray:
    """Band-passed white noise whose centre frequency glides from f0 to f1.

    One state variable stage at a usable Q is far too wide to colour noise into
    a whoosh, so the same sweep runs in series. Each stage narrows the band.
    """
    n = int(RATE * duration)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    # Glide in log frequency so the sweep sounds linear to the ear.
    fc = np.geomspace(f0, f1, n)
    f = 2.0 * np.sin(np.pi * fc / RATE)
    damp = 1.0 / q
    for _ in range(stages):
        out = np.empty(n)
        low = band = 0.0
        for i in range(n):
            low += f[i] * band
            high = x[i] - low - damp * band
            band += f[i] * high
            out[i] = band
        x = out
    return x


def sine_sweep(duration: float, f0: float, f1: float) -> np.ndarray:
    """The weight under the cue. Phase is integrated so the glide has no step."""
    n = int(RATE * duration)
    fc = np.geomspace(f0, f1, n)
    return np.sin(np.cumsum(2.0 * np.pi * fc / RATE))


def envelope(n: int, attack: float, curve: float) -> np.ndarray:
    """Soft attack, smooth power-law decay, guaranteed to reach zero."""
    a = min(max(1, int(RATE * attack)), n - 1)
    env = np.empty(n)
    env[:a] = np.sin(np.linspace(0.0, np.pi / 2, a)) ** 2
    tail = np.linspace(0.0, 1.0, n - a)
    env[a:] = (1.0 - tail) ** curve
    return env


def one_pole_lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    a = np.exp(-2.0 * np.pi * cutoff / RATE)
    out = np.empty_like(x)
    y = 0.0
    for i, v in enumerate(x):
        y = (1.0 - a) * v + a * y
        out[i] = y
    return out


def reverb(x: np.ndarray, decay: float, cutoff: float, wet: float, seed: int) -> np.ndarray:
    """Dark exponential-noise tail. Cheap, and it is the whole ambience budget."""
    n = int(RATE * decay)
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0.0, 7.0, n))
    ir = one_pole_lowpass(ir, cutoff)
    ir[0] = 0.0  # the dry signal carries the transient; the tail must not double it
    ir /= np.abs(ir).max()
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    tail = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x) + n]
    tail /= np.abs(tail).max()
    dry = np.pad(x / np.abs(x).max(), (0, n))
    return (1.0 - wet) * dry + wet * tail


def normalize(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x))
    if peak == 0:
        return x
    return x / peak * (10.0 ** (PEAK_DBFS / 20.0))


def fade_out(x: np.ndarray, seconds: float) -> np.ndarray:
    """The reverb tail never truly ends, so force the file to close on silence."""
    n = min(int(RATE * seconds), len(x))
    x = x.copy()
    x[-n:] *= np.cos(np.linspace(0.0, np.pi / 2, n)) ** 2
    return x


def write_wav(path: Path, x: np.ndarray) -> None:
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2").tobytes()
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, RATE, RATE * 2, 2, 16)
    body = b"WAVE" + fmt + struct.pack("<4sI", b"data", len(pcm)) + pcm
    path.write_bytes(struct.pack("<4sI", b"RIFF", len(body)) + body)


def cue(
    duration: float,
    noise: tuple[float, float],
    sub: tuple[float, float],
    attack: float,
    curve: float,
    wet: float,
    seed: int,
) -> np.ndarray:
    n = int(RATE * duration)
    env = envelope(n, attack, curve)
    air = band_sweep(duration, noise[0], noise[1], 2.2, stages=3, seed=seed)
    air /= np.abs(air).max()
    low = sine_sweep(duration, sub[0], sub[1])
    # The sub carries a slower envelope so the bass outlasts the movement.
    body = 0.55 * air * env + 0.45 * low * envelope(n, attack * 2.0, curve * 0.6)
    return normalize(fade_out(reverb(body, 0.55, 1_600.0, wet, seed + 100), 0.15))


def build() -> dict[str, np.ndarray]:
    # Opening cue: rises out of the bass. Reads as "listening".
    start = cue(0.34, (260.0, 1_100.0), (92.0, 186.0), 0.030, 1.4, 0.50, seed=11)
    # Closing cue: falls back into it. Reads as "sent".
    stop = cue(0.40, (1_000.0, 250.0), (190.0, 84.0), 0.022, 1.1, 0.55, seed=23)
    return {"record_start.wav": start, "record_stop.wav": stop}


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "assets"
    for name, samples in build().items():
        path = out_dir / name
        write_wav(path, samples)
        print(f"{path.name}  {len(samples) / RATE * 1000:.0f}ms  {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
