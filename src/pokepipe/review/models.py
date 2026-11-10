"""On-disk review state (`review.json`), the hand-off from stage 3 to stages 4-7.

Detection output (`pulls.json`) is never modified; every human decision lives here,
so detection can be re-run or re-tuned without losing review work, and the pair of
files doubles as labelled data for calibrating detection later.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Status(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"  # not a real pull (false positive, re-shown card, ...)


class ReviewedPull(BaseModel):
    id: int = Field(description="stable id within this review")
    detected_index: int | None = Field(
        None, description="index into pulls.json; None for a pull the reviewer added"
    )
    detected_card_id: str | None = None
    card_id: str = Field(description="final card, after any correction")
    pack: int
    time: float = Field(description="seconds — where the card is shown")
    status: Status = Status.PENDING
    reverse_holo: bool = False

    @property
    def corrected(self) -> bool:
        return self.detected_card_id is not None and self.card_id != self.detected_card_id


class ReviewState(BaseModel):
    video_slug: str
    set_id: str
    pulls: list[ReviewedPull]
    completed: bool = False
    updated_at: str = Field(default_factory=lambda: _now())

    def kept(self) -> list[ReviewedPull]:
        """The pulls that count: confirmed ones, in video order."""
        return sorted(
            (p for p in self.pulls if p.status == Status.CONFIRMED), key=lambda p: p.time
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- API request bodies ----------------------------------------------------------------


class PullUpdate(BaseModel):
    card_id: str | None = None
    status: Status | None = None
    reverse_holo: bool | None = None
    pack: int | None = None


class PostPlanBody(BaseModel):
    spec: str = Field(description="packs per post, e.g. '1-3, 9, 10+12'")
    seconds: float = Field(45.0, description="target length of each post")


class NewPull(BaseModel):
    card_id: str
    pack: int
    time: float = 0.0
    reverse_holo: bool = False
