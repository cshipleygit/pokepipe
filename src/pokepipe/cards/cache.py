"""On-disk cache for one card set: metadata, checklist, images and indexes.

Layout (see CLAUDE.md):
    <root>/<set_id>/set.json, cards.json, images/<card_id>.png, index/<model_tag>.npz
"""

from __future__ import annotations

import json
from pathlib import Path

from pokepipe.cards.models import Card, CardSet


def _safe(name: str) -> str:
    """Card ids are filesystem-safe today, but guard against odd characters anyway."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


class SetCache:
    def __init__(self, root: Path, set_id: str) -> None:
        self.set_id = set_id
        self.dir = root / _safe(set_id)

    @property
    def set_path(self) -> Path:
        return self.dir / "set.json"

    @property
    def cards_path(self) -> Path:
        return self.dir / "cards.json"

    @property
    def images_dir(self) -> Path:
        return self.dir / "images"

    @property
    def index_dir(self) -> Path:
        return self.dir / "index"

    def image_path(self, card_id: str) -> Path:
        return self.images_dir / f"{_safe(card_id)}.png"

    def index_path(self, model_tag: str) -> Path:
        return self.index_dir / f"{_safe(model_tag)}.npz"

    def exists(self) -> bool:
        return self.set_path.exists() and self.cards_path.exists()

    def save(self, card_set: CardSet, cards: list[Card]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.set_path.write_text(card_set.model_dump_json(indent=2), encoding="utf-8")
        payload = [c.model_dump() for c in cards]
        self.cards_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_set(self) -> CardSet:
        return CardSet.model_validate_json(self.set_path.read_text(encoding="utf-8"))

    def load_cards(self) -> list[Card]:
        raw = json.loads(self.cards_path.read_text(encoding="utf-8"))
        return [Card.model_validate(c) for c in raw]

    def missing_images(self, cards: list[Card]) -> list[Card]:
        return [c for c in cards if not self.image_path(c.id).exists()]
