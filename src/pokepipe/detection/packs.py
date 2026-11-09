"""Group pulls into booster packs, the unit we split the video into for posts.

Within a pack, reveals come every 1-3 s; between packs the opener talks, tears the
next wrapper or sorts cards, which leaves a gap well over 8 s (shortest seen on real
footage: ~12 s). Gap-based grouping is therefore simple and robust.

Each pack also gets a clip window for rendering: it starts a little before the first
reveal (to include the tear) and never overlaps its neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass

from pokepipe.detection.models import Pack, Pull


@dataclass(frozen=True)
class PackParams:
    min_gap: float = 8.0  # seconds between reveals that starts a new pack
    lead_in: float = 15.0  # max seconds of pre-reveal footage (pack tear) per clip
    tail: float = 2.0  # seconds kept after the last reveal
    cards_per_pack: int = 10  # 151 / SV-era: 4 C, 3 U, 2 reverse slots, 1 rare+


def group_packs(pulls: list[Pull], duration: float, params: PackParams | None = None) -> list[Pack]:
    """Assign `pull.pack` (1-based) in place and return the packs in order."""
    p = params or PackParams()
    groups: list[list[Pull]] = []
    for pull in sorted(pulls, key=lambda x: x.start):
        if not groups or pull.start - groups[-1][-1].end > p.min_gap:
            groups.append([])
        groups[-1].append(pull)

    packs = []
    for i, group in enumerate(groups):
        prev_end = groups[i - 1][-1].end + p.tail if i else 0.0
        next_start = groups[i + 1][0].start if i + 1 < len(groups) else duration
        first, last = group[0].start, group[-1].end
        for pull in group:
            pull.pack = i + 1
        packs.append(
            Pack(
                index=i + 1,
                start=first,
                end=last,
                clip_start=round(max(prev_end, first - p.lead_in, 0.0), 3),
                clip_end=round(min(last + p.tail, next_start, duration), 3),
                cards=len(group),
                expected_cards=p.cards_per_pack,
            )
        )
    return packs
