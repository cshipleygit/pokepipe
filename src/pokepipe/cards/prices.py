"""Card prices: what a pull is worth, for the on-screen price tags and for P/L later.

Two sources, most specific first:
1. A price sheet, `data/prices/<set_id>.csv` (card_id, variant, usd, source, as_of),
   for prices you set yourself: a brand-new set the API hasn't priced yet, or a price
   you'd rather use. Every row says where the number came from and when.
2. pokemontcg.io's TCGplayer market price, stored on each Card by `cards fetch`
   (`--refresh` to update).

A price is only ever looked up for the exact printing pulled (a reverse holo is not
the holo); a pull with no price stays unpriced rather than guessed.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel

from pokepipe.cards.cache import SetCache

SHEET_FIELDS = ["card_id", "variant", "usd", "source", "as_of"]


class Price(BaseModel):
    card_id: str
    variant: str
    usd: float
    source: str
    as_of: str | None = None


class PriceBook:
    def __init__(self, prices: dict[tuple[str, str], Price]) -> None:
        self._prices = prices

    def __len__(self) -> int:
        return len(self._prices)

    def get(self, card_id: str, variant: str) -> Price | None:
        return self._prices.get((card_id, variant))

    @classmethod
    def load(cls, sets_root: Path, set_id: str, sheets_dir: Path) -> PriceBook:
        prices: dict[tuple[str, str], Price] = {}
        cache = SetCache(sets_root, set_id)
        if cache.exists():
            for card in cache.load_cards():
                for variant, usd in card.prices.items():
                    prices[(card.id, variant)] = Price(
                        card_id=card.id, variant=variant, usd=usd,
                        source="tcgplayer market via pokemontcg.io", as_of=card.prices_updated,
                    )
        for row in read_sheet(sheets_dir / f"{set_id}.csv"):
            prices[(row.card_id, row.variant)] = row  # the sheet wins
        return cls(prices)


def read_sheet(path: Path) -> list[Price]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [Price(**{**row, "usd": float(row["usd"])}) for row in csv.DictReader(f)]


def write_sheet(path: Path, rows: list[Price]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r.model_dump())
