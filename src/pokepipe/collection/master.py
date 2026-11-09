"""Master-set definition for one set, derived from the cached checklist."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pokepipe.cards.models import REVERSE, Card, CardSet
from pokepipe.collection.models import Progress, slot


@dataclass(frozen=True)
class MasterSet:
    card_set: CardSet
    cards: list[Card]  # in collector-number order

    @property
    def by_id(self) -> dict[str, Card]:
        return {c.id: c for c in self.cards}

    def slots(self) -> list[str]:
        return [slot(c.id, v) for c in self.cards for v in c.variant_list()]

    def is_base(self, card: Card) -> bool:
        """Main set (not a secret rare): numbered within the printed total."""
        printed = self.card_set.printed_total
        return not printed or (card.number.isdigit() and int(card.number) <= printed)

    def resolve_variant(self, card: Card, reverse_holo: bool) -> tuple[str, str | None]:
        """Printing a reviewed pull counts as, plus a warning if the review asked for a
        printing this card doesn't exist in."""
        if reverse_holo:
            if REVERSE in card.variant_list():
                return REVERSE, None
            return card.base_variant, (
                f"{card.name} #{card.number} has no reverse holo; counted as {card.base_variant}"
            )
        return card.base_variant, None

    def progress(self, counts: Counter[str]) -> Progress:
        owned = {s for s, n in counts.items() if n > 0}
        owned_cards = {s.split("/", 1)[0] for s in owned}
        base = [c for c in self.cards if self.is_base(c)]
        slots = self.slots()
        return Progress(
            master_owned=sum(s in owned for s in slots),
            master_total=len(slots),
            full_owned=sum(c.id in owned_cards for c in self.cards),
            full_total=len(self.cards),
            base_owned=sum(c.id in owned_cards for c in base),
            base_total=len(base),
        )

    def missing(self, counts: Counter[str], variant: str | None = None) -> list[tuple[Card, str]]:
        return [
            (c, v) for c in self.cards for v in c.variant_list()
            if counts[slot(c.id, v)] == 0 and (variant is None or v == variant)
        ]
