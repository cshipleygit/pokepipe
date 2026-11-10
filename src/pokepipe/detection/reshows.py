"""Merge repeat showings of the same physical card into one pull.

Some creators show cards more than once on purpose: the hit held up again after the
pack, a glimpse before the pack starts, an end-of-video recap. Counting each showing
as a pull inflates pack counts and the collection; throwing them away loses footage
the edit wants. So a repeat is merged into the card's main pull as a `Showing`.

Two cases, decided at pack level because pack structure is what separates a re-show
from a genuine second copy of a card:

1. **Within a pack**, the same card again is a re-show (a real duplicate inside one
   pack is rare, and the reviewer can add it back).
2. **A satellite group** (a "pack" of only 1-2 cards) whose cards all appear in a real
   pack is a re-show of those cards: before the next pack (a preview), after the
   previous one, or any earlier pack (a recap). The same card in two *full* packs is
   two copies and is never merged.

"""

from __future__ import annotations

from dataclasses import dataclass

from pokepipe.detection.models import Pack, Pull, Showing
from pokepipe.detection.packs import PackParams, group_packs


@dataclass(frozen=True)
class ReshowParams:
    satellite_max_cards: int = 2  # a group this small is a re-show/preview, not a pack


def _as_showing(p: Pull) -> Showing:
    return Showing(start=p.start, end=p.end, peak_time=p.peak_time, peak_score=p.peak_score,
                   box=p.box, frames=p.frames)


def _absorb(main: Pull, repeat: Pull) -> None:
    main.showings.append(_as_showing(repeat))
    main.showings.extend(repeat.showings)
    main.showings.sort(key=lambda s: s.start)


def merge_reshows(
    pulls: list[Pull], duration: float, pack_params: PackParams | None = None,
    params: ReshowParams | None = None,
) -> tuple[list[Pull], list[Pack]]:
    """Returns (pulls with repeats merged, packs regrouped without satellite groups)."""
    params = params or ReshowParams()
    group_packs(pulls, duration, pack_params)  # assigns pull.pack for the analysis below
    by_pack: dict[int, list[Pull]] = {}
    for p in sorted(pulls, key=lambda x: x.start):
        by_pack.setdefault(p.pack, []).append(p)

    # 1. Repeats inside one pack: keep the first showing as the pull.
    for pack, members in by_pack.items():
        first: dict[str, Pull] = {}
        kept = []
        for p in members:
            if p.card_id in first:
                _absorb(first[p.card_id], p)
            else:
                first[p.card_id] = p
                kept.append(p)
        by_pack[pack] = kept

    # 2. Satellite groups whose cards all belong to a real pack.
    order = sorted(by_pack)
    real = {k for k in order if len(by_pack[k]) > params.satellite_max_cards}
    for k in order:
        if k in real:
            continue
        prev = [r for r in order if r in real and r < k]
        nxt = [r for r in order if r in real and r > k]
        # Preference: the previous pack (re-show), the next pack (preview), any earlier
        # pack (recap). Nearest first.
        candidates = prev[-1:] + nxt[:1] + list(reversed(prev[:-1]))
        targets = {}
        for p in by_pack[k]:
            host = next((r for r in candidates
                         if any(q.card_id == p.card_id for q in by_pack[r])), None)
            if host is None:
                break
            targets[id(p)] = next(q for q in by_pack[host] if q.card_id == p.card_id)
        else:  # every card found a home: the whole group was a re-show
            for p in by_pack[k]:
                _absorb(targets[id(p)], p)
            by_pack[k] = []

    merged = sorted((p for ps in by_pack.values() for p in ps), key=lambda x: x.start)
    packs = group_packs(merged, duration, pack_params)
    return merged, packs
