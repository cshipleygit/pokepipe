"""Pack layouts: the fixed slot order cards come out of a booster pack in.

Openers who don't shuffle reveal cards in print order, so a card's position tells us
what slot it came from. The main payoff: a Common/Uncommon/Rare in a reverse-holo slot
*is* a reverse holo, which CLIP can't see (same card id, same art) and which matters
for master sets.

The Scarlet & Violet layout was verified on 12 packs of 151 (120 cards): slots 1-4 were
always Common, 5-7 Uncommon, 8-9 the two reverse slots (C/U/R; slot 9 also produced an
IR and an SIR), and 10 the rare slot (R, or a UR). Basic Energy and the code card are
excluded from pulls, so they're not slots here.

Inference is deliberately conservative: it only applies when the observed rarities fit
the layout, so a shuffled pack yields no guesses rather than wrong ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pokepipe.detection.models import Pull

C, U, R = "Common", "Uncommon", "Rare"
DR, IR, SIR = "Double Rare", "Illustration Rare", "Special Illustration Rare"
UR, HR = "Ultra Rare", "Hyper Rare"
BASE = frozenset({C, U, R})


@dataclass(frozen=True)
class Slot:
    kind: str
    allowed: frozenset[str]
    reverse_if: frozenset[str] = frozenset()  # these rarities here are reverse holos


@dataclass(frozen=True)
class PackLayout:
    name: str
    slots: tuple[Slot, ...]
    tail: int  # trailing slots that identify themselves (reverse + rare slots)

    def __len__(self) -> int:
        return len(self.slots)


_common = Slot("common", frozenset({C}))
_uncommon = Slot("uncommon", frozenset({U}))
SCARLET_VIOLET = PackLayout(
    name="scarlet-violet",
    slots=(
        _common, _common, _common, _common,
        _uncommon, _uncommon, _uncommon,
        Slot("reverse", BASE | {IR}, reverse_if=BASE),  # IR only in "god packs"
        Slot("reverse-or-special", BASE | {IR, SIR, HR}, reverse_if=BASE),
        Slot("rare", frozenset({R, DR, UR, SIR, HR, IR})),
    ),
    tail=3,
)

LAYOUTS = {SCARLET_VIOLET.name: SCARLET_VIOLET}


def layout_for(series: str | None) -> PackLayout | None:
    """Default layout for a set's series. Only Scarlet & Violet is verified so far;
    other eras get no inference rather than a guess."""
    return SCARLET_VIOLET if series == "Scarlet & Violet" else None


@dataclass
class LayoutResult:
    mode: str | None  # "full", "tail" or None (no inference)
    note: str = ""
    reverse_ids: list[int] = field(default_factory=list)  # indexes into the pack's pulls


def _fits(pulls: list[Pull], slots: tuple[Slot, ...]) -> int | None:
    """Index of the first pull whose rarity doesn't fit its slot, or None if all fit."""
    for i, (pull, slot) in enumerate(zip(pulls, slots, strict=True)):
        if pull.rarity not in slot.allowed:
            return i
    return None


def apply_layout(pulls: list[Pull], layout: PackLayout) -> LayoutResult:
    """Assign `slot` and `reverse_guess` on one pack's pulls (in reveal order).

    - Full pack that fits → every pull gets its slot.
    - Short pack (a missed detection) → align from the end: the rare is always last and
      the reverse slots just before it, so the tail can still be trusted if it fits.
    - Anything else (extra cards, shuffled order) → no inference, with a note why.
    """
    pulls = sorted(pulls, key=lambda p: p.start)
    n, size = len(pulls), len(layout)
    if n == size:
        bad = _fits(pulls, layout.slots)
        if bad is not None:
            return LayoutResult(None, f"card {bad + 1} ({pulls[bad].rarity}) doesn't fit the "
                                      f"{layout.slots[bad].kind} slot — shuffled or misread?")
        mode, offset = "full", 0
    elif layout.tail <= n < size:
        bad = _fits(pulls[-layout.tail:], layout.slots[-layout.tail:])
        if bad is not None:
            return LayoutResult(None, f"{size - n} card(s) missing and the last cards don't "
                                      f"fit the reverse/rare slots")
        mode, offset = "tail", size - n
    else:
        return LayoutResult(None, f"{n} cards; expected {size}")

    result = LayoutResult(mode, "" if mode == "full" else f"{size - n} card(s) missing; "
                                                         "slots inferred from the end")
    for i, pull in enumerate(pulls):
        slot_index = i + offset
        if mode == "tail" and i < n - layout.tail:
            continue  # early slots are ambiguous when a card is missing
        slot = layout.slots[slot_index]
        pull.slot = slot_index + 1
        pull.reverse_guess = pull.rarity in slot.reverse_if
        if pull.reverse_guess:
            result.reverse_ids.append(i)
    return result
