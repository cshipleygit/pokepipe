"""On-disk output of stage 2 (`pulls.json`), consumed by review (stage 3)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Alternative(BaseModel):
    card_id: str
    score: float


class Pull(BaseModel):
    card_id: str
    name: str
    number: str
    rarity: str | None = None
    start: float = Field(description="seconds — first frame the card was confidently seen")
    end: float = Field(description="seconds — last confident frame")
    peak_time: float = Field(description="seconds — best-matching frame (use for thumbnails/zoom)")
    peak_score: float = Field(description="best cosine similarity in the segment")
    mean_margin: float = Field(description="mean lead over the runner-up card")
    frames: int
    confidence: float = Field(ge=0, le=1, description="heuristic 0-1, see segment.confidence")
    box: tuple[float, float, float, float] = Field(description="peak crop box, 0-1 of frame")
    alternatives: list[Alternative] = Field(default_factory=list)
    thumbnail: str | None = None
    pack: int | None = Field(None, description="1-based pack this pull belongs to")


class Pack(BaseModel):
    index: int
    start: float = Field(description="seconds — first reveal")
    end: float = Field(description="seconds — last reveal")
    clip_start: float = Field(description="seconds — suggested cut-in (includes the tear)")
    clip_end: float = Field(description="seconds — suggested cut-out")
    cards: int
    expected_cards: int

    @property
    def complete(self) -> bool:
        """False flags the pack for attention in review (missed or extra detections)."""
        return self.cards == self.expected_cards


class DetectionParams(BaseModel):
    fps: float
    frame_min_score: float
    frame_min_margin: float
    max_gap: float
    min_frames: int
    min_peak: float


class PullsFile(BaseModel):
    video: str
    set_id: str
    model_tag: str
    duration: float
    params: DetectionParams
    packs: list[Pack] = Field(default_factory=list)
    pulls: list[Pull]
