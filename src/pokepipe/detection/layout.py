"""Pack layouts: the fixed slot order cards come out of a booster pack in.

Openers who don't shuffle reveal cards in print order, so a card's position tells us
what slot it came from. The main payoff: a Common/Uncommon/Rare in a reverse-holo slot
*is* a reverse holo, which CLIP can't see (same card id, same art) and which matters
for master sets.

The modern layout was verified on 12 packs of 151 (Scarlet & Violet, 120 cards) and
the 7 readable packs of a Delta Reign Elite Trainer Box (Mega Evolution): slots 1-4
always Common, 5-7 Uncommon, 8-9 the two reverse slots (C/U/R; slot 9 also produced
an IR in both sets and an SIR in 151), and 10 the rare slot (R, DR, UR). Basic Energy
and the code card aren't pulls, so they're not slots here.

Inference is deliberately conservative: it only applies when the observed rarities fit
the layout, so a shuffled pack yields no guesses rather than wrong ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from pokepipe.cards.rarity import is_special
from pokepipe.detection.models import Pull

C, U, R = "Common", "Uncommon", "Rare"
IR = "Illustration Rare"
BASE = frozenset({C, U, R})


@dataclass(frozen=True)
class Slot:
    kind: str
    allowed: frozenset[str]
    reverse_if: frozenset[str] = frozenset()  # these rarities here are reverse holos
    allow_special: bool = False  # also any special rarity (DR and up, incl. new ones)

    def fits(self, rarity: str | None) -> bool:
        return rarity in self.allowed or (self.allow_special and is_special(rarity))


@dataclass(frozen=True)
class PackLayout:
    name: str
    slots: tuple[Slot, ...]
    tail: int  # trailing slots that identify themselves (reverse + rare slots)

    def __len__(self) -> int:
        return len(self.slots)


_common = Slot("common", frozenset({C}))
_uncommon = Slot("uncommon", frozenset({U}))
MODERN = PackLayout(
    name="modern",
    slots=(
        _common, _common, _common, _common,
        _uncommon, _uncommon, _uncommon,
        Slot("reverse", BASE | {IR}, reverse_if=BASE),  # IR here only in "god packs"
        Slot("reverse-or-special", BASE, reverse_if=BASE, allow_special=True),
        Slot("rare", frozenset({R}), allow_special=True),
    ),
    tail=3,
)
SCARLET_VIOLET = MODERN  # the name it was first verified under

LAYOUTS = {MODERN.name: MODERN}
VERIFIED_SERIES = {"Scarlet & Violet", "Mega Evolution"}


def layout_for(series: str | None) -> PackLayout | None:
    """Layout for a set's series. Only series verified on real packs get one; others
    get no inference rather than a guess."""
    return MODERN if series in VERIFIED_SERIES else None


@dataclass
class LayoutResult:
    mode: str | None  # "full", "tail" or None (no inference)
    note: str = ""
    reverse_ids: list[int] = field(default_factory=list)  # indexes into the pack's pulls


def _fits(pulls: list[Pull], slots: tuple[Slot, ...]) -> int | None:
    """Index of the first pull whose rarity doesn't fit its slot, or None if all fit."""
    for i, (pull, slot) in enumerate(zip(pulls, slots, strict=True)):
        if not slot.fits(pull.rarity):
            return i
    return None


def apply_layout(pulls: list[Pull], layout: PackLayout) -> LayoutResult:
    """Assign `slot` and `reverse_guess` on one pack's pulls (in reveal order).

    - Full pack that fits → every pull gets its slot.
    - Short pack (missed detections) → try every placement of the pulls into the slots
      that keeps their order and fits, and keep only what *all* placements agree on.
      E.g. with one card missing, "C C C C U U U IR R" could be missing slot 8 (the 3rd U
      is slot 7, not a reverse holo) or an earlier slot (that U is slot 8, a reverse
      holo): they disagree, so the U gets no guess. Ambiguity goes to review.
    - Anything else (extra cards, shuffled order) → no inference, with a note why.
    """
    pulls = sorted(pulls, key=lambda p: p.start)
    n, size = len(pulls), len(layout)
    if n == size:
        bad = _fits(pulls, layout.slots)
        if bad is not None:
            return LayoutResult(None, f"card {bad + 1} ({pulls[bad].rarity}) doesn't fit the "
                                      f"{layout.slots[bad].kind} slot — shuffled or misread?")
        placements = [tuple(range(size))]
        mode = "full"
    elif layout.tail <= n < size:
        placements = [
            combo for combo in combinations(range(size), n)
            if all(layout.slots[s].fits(p.rarity) for p, s in zip(pulls, combo, strict=True))
        ]
        if not placements:
            return LayoutResult(None, f"{size - n} card(s) missing and the rest don't fit the "
                                      "pack order")
        mode = "partial"
    else:
        return LayoutResult(None, f"{n} cards; expected {size}")

    result = LayoutResult(mode, "" if mode == "full" else
                          f"{size - n} card(s) missing; only unambiguous slots inferred")
    for i, pull in enumerate(pulls):
        slots = {combo[i] for combo in placements}
        reverse = {pull.rarity in layout.slots[s].reverse_if for s in slots}
        if len(slots) == 1:
            pull.slot = next(iter(slots)) + 1
        if reverse == {True}:  # a reverse holo in every possible placement
            pull.reverse_guess = True
            result.reverse_ids.append(i)
    return result
