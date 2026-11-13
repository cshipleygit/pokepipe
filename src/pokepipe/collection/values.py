"""What a video's pulls are worth: the input for profit/loss tracking.

`video_values` prices every kept pull with the exact printing pulled (pure), and
`pokepipe collection value <slug>` writes the result to data/videos/<slug>/values.json.
Unpriced pulls are listed as such, never guessed, so totals say how complete they are.
Cost basis (what the product cost) is left for the P/L step that builds on this.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from pokepipe.cards.prices import PriceBook
from pokepipe.collection.models import VideoReport


class PricedPull(BaseModel):
    pull_id: str
    pack: int
    card_id: str
    name: str
    rarity: str | None
    variant: str
    usd: float | None = None
    source: str | None = None
    as_of: str | None = None


class VideoValues(BaseModel):
    video_slug: str
    set_id: str
    total_usd: float
    priced: int
    unpriced: int
    per_pack_usd: dict[int, float]
    pulls: list[PricedPull]
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def unpriced_hits(values: VideoValues) -> list[PricedPull]:
    """Hits (the rare slot and better, not reverse holos) with no price: worth adding to
    data/prices/<set>.csv before posting, since their price tag would be missing."""
    from pokepipe.cards.rarity import is_special

    def hit(x: PricedPull) -> bool:
        return x.variant != "reverse" and (x.rarity in ("Rare", "Rare Holo")
                                           or is_special(x.rarity))

    return [x for x in values.pulls if x.usd is None and hit(x)]


def write_values(video_dir: Path, report: VideoReport, book: PriceBook) -> VideoValues:
    values = video_values(report, book)
    (video_dir / "values.json").write_text(values.model_dump_json(indent=2), encoding="utf-8")
    return values


def video_values(report: VideoReport, book: PriceBook) -> VideoValues:
    from pokepipe.narration.script import pull_ids  # same p<pack>-<n> ids as everywhere

    pulls = []
    for pid, p in pull_ids(report).items():
        price = book.get(p.card_id, p.variant)
        pulls.append(PricedPull(
            pull_id=pid, pack=p.pack, card_id=p.card_id, name=p.name, rarity=p.rarity,
            variant=p.variant, usd=price.usd if price else None,
            source=price.source if price else None, as_of=price.as_of if price else None,
        ))
    per_pack: dict[int, float] = {}
    for x in pulls:
        if x.usd is not None:
            per_pack[x.pack] = round(per_pack.get(x.pack, 0.0) + x.usd, 2)
    priced = [x for x in pulls if x.usd is not None]
    return VideoValues(
        video_slug=report.video_slug, set_id=report.set_id,
        total_usd=round(sum(x.usd for x in priced), 2), priced=len(priced),
        unpriced=len(pulls) - len(priced), per_pack_usd=per_pack, pulls=pulls,
    )
