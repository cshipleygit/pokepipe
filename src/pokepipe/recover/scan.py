"""The real `Scanner`: decode one stretch of the video and match it on the GPU."""

from __future__ import annotations

from pathlib import Path

from pokepipe.cards.index import CardIndex
from pokepipe.detection.frames import sample_frames
from pokepipe.detection.matcher import FrameMatcher, FrameResult, PixelEmbedder


class VideoScanner:
    def __init__(self, video: Path, embedder: PixelEmbedder, index: CardIndex) -> None:
        self.video = video
        self.matcher = FrameMatcher(embedder, index)

    def frames(self, start: float, end: float, fps: float) -> list[FrameResult]:
        stream = sample_frames(self.video, fps=fps, max_side=960, start=start,
                               duration=end - start)
        return list(self.matcher.match_stream(stream))
