"""Sound effects, synthesised here (no downloaded samples, so nothing to license).

Each effect is a short numpy waveform; `sfx_track` lays the post's effects on one
stereo track at their times, which the final ffmpeg pass mixes under the voice.

- `sparkle`: a rising bell arpeggio with shimmer and a soft low thump (big hits)
- `chime`: a lighter three-note bell (the pack's ordinary rare)
- `coin`: the two-note "ding" of a price tag popping up
- `rocket`: a rising whoosh-tone as the progress rocket flies forward
- `whoosh`: filtered noise swelling and fading (into each pack, into the recap)
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 48_000


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SR)) / SR


def _bell(freq: float, dur: float, decay: float) -> np.ndarray:
    """A soft bell: a few harmonics, a 5 ms attack, exponential decay."""
    t = _t(dur)
    wave_ = (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * 2 * freq * t)
             + 0.12 * np.sin(2 * np.pi * 3.01 * freq * t))
    return wave_ * np.minimum(t / 0.005, 1.0) * np.exp(-t / decay)


def _place(out: np.ndarray, sound: np.ndarray, at: float) -> None:
    i = int(at * SR)
    if i >= len(out):
        return
    n = min(len(sound), len(out) - i)
    out[i:i + n] += sound[:n]


def _normal(x: np.ndarray, peak: float = 0.8) -> np.ndarray:
    m = np.abs(x).max()
    return x * (peak / m) if m > 0 else x


def _noise(dur: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(len(_t(dur)))


def _lowpass(x: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    """One-pole low-pass with a per-sample cutoff (Hz), for sweeping noise."""
    a = np.exp(-2 * np.pi * np.asarray(cutoff) / SR)
    a = np.broadcast_to(a, x.shape)
    y = np.empty_like(x)
    acc = 0.0
    for i, (v, k) in enumerate(zip(x, a, strict=True)):
        acc = k * acc + (1 - k) * v
        y[i] = acc
    return y


def chime() -> np.ndarray:
    out = np.zeros(int(0.9 * SR))
    for k, f in enumerate((1318.5, 1661.2, 1975.5)):  # E6 G#6 B6
        _place(out, 0.6 * _bell(f, 0.7, 0.22), k * 0.07)
    return _normal(out, 0.6)


def sparkle() -> np.ndarray:
    out = np.zeros(int(1.6 * SR))
    for k, f in enumerate((1046.5, 1318.5, 1568.0, 2093.0, 2637.0)):  # C6 E6 G6 C7 E7
        _place(out, 0.55 * _bell(f, 1.0, 0.35), k * 0.06)
    rng = np.random.default_rng(7)
    for _ in range(14):  # shimmer: tiny high bells scattered over the tail
        _place(out, 0.12 * _bell(rng.uniform(2500, 5200), 0.4, 0.09), rng.uniform(0.25, 1.1))
    t = _t(0.35)
    _place(out, 0.5 * np.sin(2 * np.pi * 70 * t) * np.exp(-t / 0.09), 0.0)  # soft thump
    return _normal(out)


def coin() -> np.ndarray:
    out = np.zeros(int(0.5 * SR))
    _place(out, 0.7 * _bell(987.8, 0.09, 0.05), 0.0)  # B5
    _place(out, 0.8 * _bell(1318.5, 0.45, 0.16), 0.08)  # E6
    return _normal(out, 0.6)


def whoosh(dur: float = 0.55) -> np.ndarray:
    t = _t(dur)
    cutoff = 400 + 3600 * np.sin(np.pi * t / dur)  # opens then closes
    x = _lowpass(_noise(dur, 3), cutoff)
    env = np.sin(np.pi * np.minimum(t / dur, 1.0)) ** 2
    return _normal(x * env, 0.45)


def rocket() -> np.ndarray:
    dur = 0.8
    t = _t(dur)
    freq = 220 * (4 ** (t / dur))  # 220 → 880 Hz
    phase = 2 * np.pi * np.cumsum(freq) / SR
    tone = np.sin(phase + 0.3 * np.sin(2 * np.pi * 14 * t)) * np.minimum(t / 0.05, 1)
    rumble = _lowpass(_noise(dur, 5), 300 + 2500 * t / dur)
    env = np.minimum(t / 0.1, 1.0) * np.exp(-np.maximum(t - 0.55, 0) / 0.12)
    out = np.zeros(int(1.1 * SR))
    _place(out, (0.35 * tone + 0.9 * rumble) * env, 0.0)
    _place(out, 0.4 * _bell(1568.0, 0.4, 0.12), 0.72)  # a little "ding" on arrival
    return _normal(out, 0.55)


EFFECTS = {"sparkle": sparkle, "chime": chime, "coin": coin, "whoosh": whoosh,
           "rocket": rocket}


def sfx_track(events: list[tuple[float, str]], duration: float) -> np.ndarray:
    """(N, 2) float track with each named effect at its time."""
    out = np.zeros(int(duration * SR) + 1)
    cache: dict[str, np.ndarray] = {}
    for at, name in events:
        if name not in cache:
            cache[name] = EFFECTS[name]()
        _place(out, cache[name], at)
    out = np.clip(out, -1.0, 1.0)
    return np.stack([out, out], axis=1)


def write_wav(path: Path, track: np.ndarray) -> Path:
    pcm = (np.clip(track, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(track.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path
