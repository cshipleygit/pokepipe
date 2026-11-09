"""Typed records for card sets and cards, as cached on disk.

We keep only the fields the pipeline needs; the API returns far more.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

NORMAL, HOLO, REVERSE = "normal", "holo", "reverse"
FIRST_EDITION, FIRST_EDITION_HOLO = "1st-edition", "1st-edition-holo"

# pokemontcg.io exposes which printings exist only via TCGplayer price keys.
VARIANT_KEYS = {
    "normal": NORMAL,
    "holofoil": HOLO,
    "unlimitedHolofoil": HOLO,
    "reverseHolofoil": REVERSE,
    "1stEditionNormal": FIRST_EDITION,
    "1stEditionHolofoil": FIRST_EDITION_HOLO,
}
_VARIANT_ORDER = [NORMAL, HOLO, FIRST_EDITION, FIRST_EDITION_HOLO, REVERSE]


def variants_from_price_keys(prices: dict) -> list[str]:
    found = {VARIANT_KEYS[k] for k in prices if k in VARIANT_KEYS}
    return [v for v in _VARIANT_ORDER if v in found]


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
    # Printings this card exists in (see VARIANT_KEYS). Empty = unknown → rarity rule.
    variants: list[str] = Field(default_factory=list)

    @classmethod
    def from_api(cls, raw: dict) -> Card:
        images = raw.get("images") or {}
        prices = (raw.get("tcgplayer") or {}).get("prices") or {}
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
            variants=variants_from_price_keys(prices),
        )

    def variant_list(self) -> list[str]:
        """Printings for master-set tracking, base printing first. Uses the API's price
        data when present, else the modern-era rule: C/U → normal + reverse, Rare →
        holo + reverse, anything rarer → holo only."""
        if self.variants:
            return self.variants
        if self.rarity in ("Common", "Uncommon"):
            return [NORMAL, REVERSE]
        if self.rarity == "Rare":
            return [HOLO, REVERSE]
        return [HOLO]

    @property
    def base_variant(self) -> str:
        return next((v for v in self.variant_list() if v != REVERSE), self.variant_list()[0])

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
