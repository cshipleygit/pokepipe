"""Per-frame card matching on the GPU.

For each sampled frame we don't know where the card is, so we score a fixed grid of
square crops (several sizes and positions) and keep the best-matching crop per card.
All cropping happens in one batched `roi_align` call, so a frame never leaves the GPU.

Why square crops: CLIP's preprocessing centre-crops the reference scans to the middle
square of the card (art + name area), so a square crop around a held card is the
like-for-like comparison.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torchvision.ops import roi_align

from pokepipe.cards.index import CardIndex, Match
from pokepipe.detection.frames import Frame


class PixelEmbedder(Protocol):
    image_size: int
    device: str

    def embed_pixels(self, pixels: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class CropGrid:
    """Square crops, as fractions of frame height (sides) and width/height (centres)."""

    sides: tuple[float, ...] = (0.45, 0.6, 0.75)
    centers_x: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    centers_y: tuple[float, ...] = (0.45, 0.55)

    def boxes(self, width: int, height: int) -> np.ndarray:
        """(K, 4) x1, y1, x2, y2 pixel boxes, shifted inward to stay inside the frame."""
        out = []
        for side in self.sides:
            s = min(side * height, width, height)
            for cx in self.centers_x:
                for cy in self.centers_y:
                    x1 = float(np.clip(cx * width - s / 2, 0, width - s))
                    y1 = float(np.clip(cy * height - s / 2, 0, height - s))
                    out.append((x1, y1, x1 + s, y1 + s))
        return np.asarray(out, dtype=np.float32)


@dataclass(frozen=True)
class FrameResult:
    index: int
    time: float
    top: list[Match]  # best cards for this frame, highest first
    box: tuple[float, float, float, float]  # crop that produced the top match

    @property
    def best(self) -> Match:
        return self.top[0]

    @property
    def margin(self) -> float:
        """Gap to the runner-up card — low margin means the frame is ambiguous."""
        return self.top[0].score - self.top[1].score if len(self.top) > 1 else self.top[0].score


class FrameMatcher:
    def __init__(
        self,
        embedder: PixelEmbedder,
        index: CardIndex,
        grid: CropGrid | None = None,
        top_k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.grid = grid or CropGrid()
        self.top_k = top_k
        self.device = embedder.device

    def match_batch(self, frames: list[Frame]) -> list[FrameResult]:
        if not frames:
            return []
        h, w = frames[0].rgb.shape[:2]
        boxes = torch.as_tensor(self.grid.boxes(w, h), device=self.device)
        k = len(boxes)

        rgb = torch.as_tensor(np.stack([f.rgb for f in frames]), device=self.device)
        images = rgb.permute(0, 3, 1, 2).float().div_(255)  # (B, 3, H, W)
        batch_idx = torch.arange(len(frames), device=self.device).repeat_interleave(k)
        rois = torch.cat([batch_idx[:, None].float(), boxes.repeat(len(frames), 1)], dim=1)
        size = self.embedder.image_size
        crops = roi_align(images, rois, output_size=(size, size), aligned=True)

        per_card = self.index.card_scores(self.embedder.embed_pixels(crops))  # (B*K, C)
        per_card = per_card.view(len(frames), k, -1)
        best_per_card, best_crop = per_card.max(dim=1)  # (B, C)
        tops = self.index.top_k(best_per_card, self.top_k)
        top_col = best_per_card.argmax(dim=1)
        crop_of_top = best_crop.gather(1, top_col[:, None]).squeeze(1).tolist()
        box_list = boxes.tolist()
        return [
            FrameResult(f.index, f.time, top, tuple(box_list[c]))
            for f, top, c in zip(frames, tops, crop_of_top, strict=True)
        ]

    def match_stream(self, frames: Iterable[Frame], batch_size: int = 16) -> Iterator[FrameResult]:
        batch: list[Frame] = []
        for frame in frames:
            batch.append(frame)
            if len(batch) == batch_size:
                yield from self.match_batch(batch)
                batch = []
        yield from self.match_batch(batch)
