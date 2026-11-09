"""Typed records for card sets and cards, as cached on disk.

We keep only the fields the pipeline needs; the API returns far more.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class CardSet(BaseModel):
    id: str
    name: str
    series: str = ""
    printed_total: int = Field(0, alias="printedTotal")
    total: int = 0
    release_date: str = Field("", alias="releaseDate")
    logo_url: str | None = None
    symbol_url: str | None = None

    model_config = {"populate_by_name": True}

    @classmethod
    def from_api(cls, raw: dict) -> CardSet:
        images = raw.get("images") or {}
        return cls.model_validate(
            {**raw, "logo_url": images.get("logo"), "symbol_url": images.get("symbol")}
        )


class Card(BaseModel):
    id: str
    name: str
    number: str
    set_id: str
    rarity: str | None = None
    supertype: str | None = None
    subtypes: list[str] = Field(default_factory=list)
    image_small: str | None = None
    image_large: str | None = None

    @classmethod
    def from_api(cls, raw: dict) -> Card:
        images = raw.get("images") or {}
        return cls(
            id=raw["id"],
            name=raw["name"],
            number=str(raw["number"]),
            set_id=(raw.get("set") or {}).get("id", raw["id"].rsplit("-", 1)[0]),
            rarity=raw.get("rarity"),
            supertype=raw.get("supertype"),
            subtypes=raw.get("subtypes") or [],
            image_small=images.get("small"),
            image_large=images.get("large"),
        )

    @property
    def image_url(self) -> str | None:
        """Prefer the hi-res scan; it gives CLIP more detail to work with."""
        return self.image_large or self.image_small

    def sort_key(self) -> tuple[int, str, int, str]:
        """Order by collector number: plain numbers first ('2' < '10'), then prefixed
        numbers grouped by prefix ('GG01' < 'GG10' < 'TG05')."""
        match = re.search(r"\d+", self.number)
        numeric = int(match.group()) if match else 10**9
        prefix = self.number[: match.start()] if match else self.number
        return (1 if prefix else 0, prefix, numeric, self.number)
