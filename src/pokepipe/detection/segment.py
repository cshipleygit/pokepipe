"""Turn noisy per-frame matches into discrete card reveals.

A reveal is a stretch of frames where the same card is the confident best match.
Pure functions over `FrameResult`s — no GPU or video needed, so fully unit-tested.

Default thresholds were calibrated on real 1080p footage (see CLAUDE.md, stage 2):
genuine reveals peak around 0.75-0.87 cosine with a clear lead over the runner-up,
while sealed product, hands and card backs top out around 0.62-0.66.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from pokepipe.detection.matcher import FrameResult


@dataclass(frozen=True)
class SegmentParams:
    frame_min_score: float = 0.68  # a frame "votes" for its top card above this...
    frame_min_margin: float = 0.03  # ...if it also leads the runner-up by this much
    max_gap: float = 1.0  # seconds: merge votes for the same card across short dropouts
    min_frames: int = 3  # discard flickers
    min_peak: float = 0.72  # the segment's best frame must clear this


@dataclass
class Segment:
    card_id: str
    frames: list[FrameResult] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.frames[0].time

    @property
    def end(self) -> float:
        return self.frames[-1].time

    @property
    def peak(self) -> FrameResult:
        return max(self.frames, key=lambda f: f.best.score)

    @property
    def mean_margin(self) -> float:
        return sum(f.margin for f in self.frames) / len(self.frames)

    def alternatives(self, k: int = 5) -> list[tuple[str, float]]:
        """Cards ranked by best score across the segment — the review UI's dropdown."""
        best: dict[str, float] = defaultdict(float)
        for f in self.frames:
            for m in f.top:
                best[m.card_id] = max(best[m.card_id], m.score)
        return sorted(best.items(), key=lambda kv: -kv[1])[:k]


def _votes(frame: FrameResult, p: SegmentParams) -> bool:
    return frame.best.score >= p.frame_min_score and frame.margin >= p.frame_min_margin


def segment(frames: list[FrameResult], params: SegmentParams | None = None) -> list[Segment]:
    p = params or SegmentParams()
    segments: list[Segment] = []
    latest: dict[str, Segment] = {}  # most recent segment per card
    for frame in frames:
        if not _votes(frame, p):
            continue
        card = frame.best.card_id
        seg = latest.get(card)
        # Extend across short dropouts *and* brief flickers to another card mid-reveal.
        if seg and frame.time - seg.end <= p.max_gap:
            seg.frames.append(frame)
            continue
        latest[card] = seg = Segment(card, [frame])
        segments.append(seg)
    return [
        s for s in segments if len(s.frames) >= p.min_frames and s.peak.best.score >= p.min_peak
    ]


def _ramp(x: float, lo: float, hi: float) -> float:
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


def confidence(seg: Segment) -> float:
    """Heuristic 0-1 confidence from three signals:
    how well the best frame matches, how clearly it beats the runner-up, and how long
    the card stayed recognisable. Weights are hand-set; once the review UI produces
    confirmed/corrected labels this is the obvious place for a calibrated model.
    """
    return round(
        0.5 * _ramp(seg.peak.best.score, 0.68, 0.85)
        + 0.3 * _ramp(seg.mean_margin, 0.0, 0.12)
        + 0.2 * _ramp(len(seg.frames), 2, 10),
        3,
    )
