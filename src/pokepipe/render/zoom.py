"""Zoom paths: where the vertical crop sits at each moment of a zoom shot.

Pure geometry, no I/O. Input is the card's position over the shot (from
`render.track`, or just the detection box when there's no tracker); output is a list
of keyframes the renderer animates the crop through. The rules, in priority order:

1. **The whole card is always inside the crop.** After smoothing, any keyframe whose
   crop would clip the card is widened until it fits, up to the normal framing.
2. **Lost card → normal framing.** If the tracker can't find the card (blur, a hand
   over it), that stretch eases back to the normal framing, which shows everything.
3. **No jumps.** The zoom eases in from where the camera was (the normal framing, or
   the previous card's zoom when two hits follow each other), eases back out at the
   end, and follows the card through a moving average rather than frame by frame.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pokepipe.render.timeline import OUT_H, OUT_W, Segment

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 as 0-1 of the source frame
Crop = tuple[int, int, int, int]  # w, h, x, y in source pixels
Track = list[tuple[float, Box | None]]  # (seconds into the shot's source, card box)

ASPECT = OUT_W / OUT_H  # 9:16


@dataclass(frozen=True)
class ZoomParams:
    margin: float = 1.5  # crop width = card width × this
    max_zoom: float = 1.8  # vs the normal framing
    pad: float = 0.05  # clear space kept around the card, as a fraction of its size
    ease_in: float = 0.6
    ease_out: float = 0.5
    smooth: int = 3  # tracker samples in the moving average
    max_gap: int = 2  # lost samples bridged by interpolation; longer → normal framing
    step: float = 0.15  # keyframe spacing (the renderer interpolates between them)


@dataclass(frozen=True)
class _View:
    """A crop as centre + width (height follows from 9:16), which blends linearly."""

    cx: float
    cy: float
    w: float

    def blend(self, other: _View, a: float) -> _View:
        return _View(
            self.cx + (other.cx - self.cx) * a,
            self.cy + (other.cy - self.cy) * a,
            self.w + (other.w - self.w) * a,
        )


def _view(crop: Crop) -> _View:
    w, h, x, y = crop
    return _View(x + w / 2, y + h / 2, float(w))


def _to_crop(v: _View, window: Crop) -> Crop:
    """Even-sized 9:16 crop, no bigger than the window and shifted to stay inside it."""
    ww, wh, wx, wy = window
    w = min(v.w, ww, wh * ASPECT)
    h = w / ASPECT
    w2, h2 = int(w) // 2 * 2, int(h) // 2 * 2
    x = min(max(v.cx - w2 / 2, wx), wx + ww - w2)
    y = min(max(v.cy - h2 / 2, wy), wy + wh - h2)
    return w2, h2, int(round(x)), int(round(y))


def _px(box: Box, src_w: int, src_h: int) -> tuple[float, float, float, float]:
    return box[0] * src_w, box[1] * src_h, box[2] * src_w, box[3] * src_h


def fit_view(box: Box, window: Crop, src_w: int, src_h: int, p: ZoomParams) -> _View:
    """The zoom for a card at `box`: card width × margin, capped by max_zoom, and never
    narrower or shorter than the card plus padding."""
    x1, y1, x2, y2 = _px(box, src_w, src_h)
    cw, ch = x2 - x1, y2 - y1
    need = max(cw * (1 + 2 * p.pad), ch * (1 + 2 * p.pad) * ASPECT)
    w = max(cw * p.margin, window[0] / p.max_zoom, need)
    return _View((x1 + x2) / 2, (y1 + y2) / 2, min(w, window[0]))


def contain(v: _View, box: Box, window: Crop, src_w: int, src_h: int, p: ZoomParams) -> _View:
    """Widen `v` (keeping its centre where possible) until the padded card fits."""
    x1, y1, x2, y2 = _px(box, src_w, src_h)
    px, py = (x2 - x1) * p.pad, (y2 - y1) * p.pad
    x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py
    w = v.w
    for _ in range(12):  # each pass widens or recentres; converges in a few
        cw, ch, cx, cy = _to_crop(_View(v.cx, v.cy, w), window)
        if cx <= x1 + 1 and cy <= y1 + 1 and cx + cw >= x2 - 1 and cy + ch >= y2 - 1:
            break
        if cw >= window[0] - 2:
            break  # already the normal framing: nothing wider to give
        need_w = max(2 * max(v.cx - x1, x2 - v.cx), 2 * max(v.cy - y1, y2 - v.cy) * ASPECT)
        w = min(window[0], max(w * 1.05, need_w))
    return _View(v.cx, v.cy, w)


def _fill_gaps(track: Track, max_gap: int) -> Track:
    """Interpolate short runs of lost samples between two found ones."""
    out = list(track)
    i = 0
    while i < len(out):
        if out[i][1] is not None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j][1] is None:
            j += 1
        if i > 0 and j < len(out) and j - i <= max_gap:
            (t0, a), (t1, b) = out[i - 1], out[j]
            for k in range(i, j):
                f = (out[k][0] - t0) / (t1 - t0)
                out[k] = (out[k][0], tuple(u + (v - u) * f for u, v in zip(a, b, strict=True)))
        i = j
    return out


def _smoothstep(a: float) -> float:
    a = min(max(a, 0.0), 1.0)
    return a * a * (3 - 2 * a)


def zoom_path(
    seg: Segment,
    track: Track | None,
    src_w: int,
    src_h: int,
    start: Crop,
    ease_out: bool,
    p: ZoomParams | None = None,
) -> list[tuple[float, Crop]]:
    """Keyframes for one zoom shot. `start` is the crop the shot opens on (the normal
    framing, or the previous zoom's last crop for a chained hit). Without a track the
    zoom aims at the detection box (`seg.crop`), still easing in and out."""
    p = p or ZoomParams()
    window = seg.window or seg.crop
    length = seg.src_len
    wide = _view(window)

    # 1. Where the card is → target view per sample (lost → normal framing).
    samples: list[tuple[float, _View, Box | None]] = []
    if track:
        for t, box in _fill_gaps(track, p.max_gap):
            v = fit_view(box, window, src_w, src_h, p) if box else wide
            samples.append((t, v, box))
    else:
        samples = [(0.0, _view(seg.crop), None)]

    # 2. Moving average, then make sure each sample's card still fits.
    if len(samples) > 1 and p.smooth > 1:
        half = p.smooth // 2
        smoothed = []
        for i, (t, _, box) in enumerate(samples):
            near = [v for _, v, _ in samples[max(0, i - half) : i + half + 1]]
            n = len(near)
            avg = _View(sum(v.cx for v in near) / n, sum(v.cy for v in near) / n,
                        sum(v.w for v in near) / n)
            smoothed.append((t, avg, box))
        samples = smoothed
    samples = [
        (t, contain(v, box, window, src_w, src_h, p) if box else v, box)
        for t, v, box in samples
    ]

    def target(t: float) -> _View:
        if t <= samples[0][0]:
            return samples[0][1]
        for (t0, a, _), (t1, b, _) in zip(samples, samples[1:], strict=False):
            if t <= t1:
                return a.blend(b, (t - t0) / (t1 - t0) if t1 > t0 else 1.0)
        return samples[-1][1]

    # 3. Ease in from `start`, out to the normal framing; keyframes every `step`.
    first = _view(start)
    n = max(1, round(length / p.step))
    keys: list[tuple[float, Crop]] = []
    for k in range(n + 1):
        t = length * k / n
        v = first.blend(target(t), _smoothstep(t / p.ease_in) if p.ease_in > 0 else 1.0)
        if ease_out and p.ease_out > 0:
            v = v.blend(wide, _smoothstep((t - (length - p.ease_out)) / p.ease_out))
        keys.append((round(t, 3), _to_crop(v, window)))
    return keys


def add_zoom_paths(
    segments: list[Segment],
    src_w: int,
    src_h: int,
    track_fn: Callable[[Segment], Track | None] | None = None,
    p: ZoomParams | None = None,
) -> int:
    """Give every zoom shot a path. A chained zoom opens on the previous zoom's last
    crop, and that previous zoom then doesn't ease out. Returns how many were tracked."""
    tracked = 0
    for i, seg in enumerate(segments):
        if seg.kind != "zoom":
            continue
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        chained_next = bool(nxt and nxt.kind == "zoom" and nxt.chained)
        prev = segments[i - 1] if i else None
        start = (
            prev.path[-1][1]
            if seg.chained and prev and prev.path
            else (seg.window or seg.crop)
        )
        track = track_fn(seg) if track_fn else None
        tracked += bool(track)
        seg.path = zoom_path(seg, track, src_w, src_h, start,
                             ease_out=not chained_next and seg.hold == 0, p=p)
    return tracked
