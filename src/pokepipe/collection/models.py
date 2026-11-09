"""Collection state and the per-video report handed to narration and rendering.

Ownership is never stored as a bare counter. It is the sum of *contributions*: one
per applied video, plus manual adjustments. Re-applying a video after its review is
edited replaces that video's contribution instead of double-counting, and removing a
video undoes it exactly.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from pydantic import BaseModel, Field


def slot(card_id: str, variant: str) -> str:
    """A master-set slot: one printing of one card, e.g. 'sv3pt5-25/reverse'."""
    return f"{card_id}/{variant}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Contribution(BaseModel):
    source: str = Field(description="video slug, or 'manual'")
    applied_at: str = Field(default_factory=_now)
    slots: list[str] = Field(default_factory=list, description="one entry per copy")


class Collection(BaseModel):
    name: str
    set_id: str
    contributions: dict[str, Contribution] = Field(default_factory=dict)

    def counts(self, exclude: str | None = None) -> Counter[str]:
        c: Counter[str] = Counter()
        for source, contrib in self.contributions.items():
            if source != exclude:
                c.update(contrib.slots)
        return c


# --- progress ----------------------------------------------------------------------------


class Progress(BaseModel):
    """Owned/total at three levels collectors care about."""

    master_owned: int
    master_total: int  # every printing (normal, holo, reverse, ...)
    full_owned: int
    full_total: int  # every card number incl. secret rares, any printing
    base_owned: int
    base_total: int  # numbers up to the printed set total

    @property
    def master_pct(self) -> float:
        return 100 * self.master_owned / self.master_total if self.master_total else 0.0


class PullResult(BaseModel):
    card_id: str
    name: str
    number: str
    rarity: str | None
    variant: str
    pack: int
    time: float
    new: bool = Field(description="first copy of this printing in the collection")
    new_card: bool = Field(description="first copy of this card number in any printing")
    copy_number: int = Field(description="1 = first copy owned, 2 = first duplicate, ...")


class PackResult(BaseModel):
    index: int
    new_slots: int
    progress_after: Progress


class VideoReport(BaseModel):
    """Written to data/videos/<slug>/collection.json for stages 5-6."""

    video_slug: str
    collection: str
    set_id: str
    set_name: str
    before: Progress
    after: Progress
    packs: list[PackResult]
    pulls: list[PullResult]
    warnings: list[str] = Field(default_factory=list)

    @property
    def new_count(self) -> int:
        return sum(p.new for p in self.pulls)
