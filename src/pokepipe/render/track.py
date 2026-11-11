"""Find a known card's outline over a zoom shot, so the zoom can follow it.

Detection only needs to know *which* card is showing, so its boxes come from a coarse
grid: good enough to recognise a card, too rough to frame one (the crop could sit
~10% of the frame off the card). Here the card is already known, so the search
optimises the box instead: score candidate boxes against that one card's reference
embedding and keep the best, coarse-to-fine, on the GPU in batched `roi_align` calls
like `detection.matcher`. Because the reference scans are embedded as the whole card
(squashed to a square), the best-scoring box is the one that matches the card's
outline.

Frames are sampled at a few per second; `render.zoom` smooths between samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import roi_align

from pokepipe.cards.index import CardIndex
from pokepipe.detection.frames import sample_frames
from pokepipe.detection.matcher import PixelEmbedder

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackParams:
    fps: float = 5.0
    max_side: int = 960
    lost_below: float = 0.55  # best score under this: the card isn't visible
    # First-frame search around the detection box (fractions of frame width / height).
    span_x: float = 0.18
    span_y: float = 0.12
    scales: tuple[float, ...] = (0.8, 0.9, 1.0, 1.12, 1.25)
    refine_steps: int = 3
    # CLIP scores the artwork alone about as high as the whole card, so the plain
    # argmax drifts to the top two-thirds. Among boxes within `prefer_larger` of the best
    # score, take the largest: the whole card.
    prefer_larger: float = 0.02
    # Box height stays within this range of the detection box's: full-art cards score
    # flatly, and without a cap "largest near-best" grows into the background.
    size_range: tuple[float, float] = (0.7, 1.35)


class CardTracker:
    def __init__(self, embedder: PixelEmbedder, index: CardIndex,
                 params: TrackParams | None = None) -> None:
        self.embedder = embedder
        self.index = index
        self.p = params or TrackParams()
        self.device = embedder.device
        self.aspect = embedder.crop_aspect
        self.last_scores: list[float] = []  # per sample of the last track(), for tuning
        self._h_range = (0.0, float("inf"))  # box height limits for the current track()

    def _refs(self, card_id: str) -> torch.Tensor:
        rows = [i for i, c in enumerate(self.index.card_ids) if c == card_id]
        if not rows:
            raise KeyError(card_id)
        return self.index._vectors[rows]  # (R, D), L2-normalised

    def _score(self, image: torch.Tensor, boxes: np.ndarray, refs: torch.Tensor) -> np.ndarray:
        rois = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
        rois = torch.cat([torch.zeros(len(rois), 1, device=self.device), rois], dim=1)
        size = self.embedder.image_size
        crops = roi_align(image, rois, output_size=(size, size), aligned=True)
        emb = self.embedder.embed_pixels(crops)
        return (emb @ refs.T).max(dim=1).values.float().cpu().numpy()

    def _boxes(self, cands: list[tuple[float, float, float]], w: int, h: int) -> np.ndarray:
        """(cx, cy, box height) in pixels → clamped card-aspect boxes."""
        out = []
        lo, hi = self._h_range
        for cx, cy, bh in cands:
            bh = min(max(bh, lo), hi, h, w / self.aspect)
            bw = bh * self.aspect
            x1 = float(np.clip(cx - bw / 2, 0, w - bw))
            y1 = float(np.clip(cy - bh / 2, 0, h - bh))
            out.append((x1, y1, x1 + bw, y1 + bh))
        return np.asarray(out, dtype=np.float32)

    def _candidates(self, image, refs, w, h, centre, bh, span_x, span_y, n, scales):
        cands = [
            (centre[0] + dx * w, centre[1] + dy * h, bh * sc)
            for dx in np.linspace(-span_x, span_x, n)
            for dy in np.linspace(-span_y, span_y, n)
            for sc in scales
        ]
        boxes = self._boxes(cands, w, h)
        return boxes, self._score(image, boxes, refs)

    def _pick(self, boxes: np.ndarray, scores: np.ndarray):
        """Largest box scoring within `prefer_larger` of the best (see TrackParams)."""
        near = np.flatnonzero(scores >= scores.max() - self.p.prefer_larger)
        heights = boxes[near, 3] - boxes[near, 1]
        best = int(near[heights.argmax()])
        return boxes[best], float(scores[best])

    def _locate(self, image, refs, w, h, box, wide: bool):
        """Best box near `box`: a wide search, then finer passes around the current pick.
        Every scored box stays in the pool, so the pick is always judged against the
        best score seen (the size preference can't compound over passes)."""
        x1, y1, x2, y2 = box
        centre, bh = ((x1 + x2) / 2, (y1 + y2) / 2), y2 - y1
        if wide:
            pool = [self._candidates(image, refs, w, h, centre, bh, self.p.span_x,
                                     self.p.span_y, 7, self.p.scales)]
        else:
            pool = [self._candidates(image, refs, w, h, centre, bh, 0.06, 0.04, 5,
                                     (0.92, 1.0, 1.08))]
        best, _ = self._pick(*pool[0])
        step_x, step_y, scale = 0.025, 0.018, 1.05
        for _ in range(self.p.refine_steps):
            x1, y1, x2, y2 = best
            pool.append(self._candidates(image, refs, w, h, ((x1 + x2) / 2, (y1 + y2) / 2),
                                         y2 - y1, step_x, step_y, 3,
                                         (1 / scale, 1.0, scale)))
            best, _ = self._pick(np.concatenate([b for b, _ in pool]),
                                 np.concatenate([sc for _, sc in pool]))
            step_x, step_y, scale = step_x / 2, step_y / 2, 1 + (scale - 1) / 2
        boxes = np.concatenate([b for b, _ in pool])
        scores = np.concatenate([sc for _, sc in pool])
        return self._pick(boxes, scores)

    def track(self, video: Path, card_id: str, start: float, end: float,
              hint: Box) -> list[tuple[float, Box | None]]:
        """Card box (0-1 of the frame) at each sample in [start, end), keyed by seconds
        after `start`; None where the card can't be seen."""
        refs = self._refs(card_id)
        hint_h = hint[3] - hint[1]  # fraction of frame height
        out: list[tuple[float, Box | None]] = []
        self.last_scores = []
        last: np.ndarray | None = None
        for frame in sample_frames(video, self.p.fps, self.p.max_side, start, end - start):
            h, w = frame.rgb.shape[:2]
            self._h_range = (hint_h * h * self.p.size_range[0], hint_h * h * self.p.size_range[1])
            image = (torch.from_numpy(frame.rgb.copy()).to(self.device)
                     .permute(2, 0, 1)[None].float().div_(255))
            seed = last if last is not None else np.asarray(
                [hint[0] * w, hint[1] * h, hint[2] * w, hint[3] * h], dtype=np.float32)
            box, score = self._locate(image, refs, w, h, seed, wide=last is None)
            self.last_scores.append(round(score, 3))
            if score < self.p.lost_below:
                out.append((frame.time - start, None))
                last = None  # search wide again around the hint next time
                continue
            last = box
            out.append((frame.time - start,
                        (box[0] / w, box[1] / h, box[2] / w, box[3] / h)))
        return out


def load_tracker(set_id: str) -> CardTracker:
    """The CLIP model and card index detection used, loaded for render."""
    from pokepipe.cards import service as cards
    from pokepipe.cards.embeddings import OpenClipEmbedder
    from pokepipe.config import get_settings

    s = get_settings()
    emb = OpenClipEmbedder(s.clip_model, s.clip_pretrained, fit=s.clip_fit)
    return CardTracker(emb, cards.load_index(s.sets_dir, set_id, emb.model_tag))
