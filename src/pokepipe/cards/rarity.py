"""One place that knows how rarities rank and which count as "hits".

Every era invents new chase rarities (Mega Evolution added "Mega Hyper Rare"), so the
rules are written to cope with names they've never seen: an unknown rarity that
contains "Rare" and isn't a base rarity is treated as special, ranked with the
ultra rares, rather than silently treated as a common.
"""

from __future__ import annotations

# Lowest to highest. Display/ranking only; collection tracking uses card variants.
ORDER = [
    "Common",
    "Uncommon",
    "Rare",
    "Rare Holo",
    "Double Rare",
    "ACE SPEC Rare",
    "Illustration Rare",
    "Ultra Rare",
    "Shiny Rare",
    "Shiny Ultra Rare",
    "Hyper Rare",
    "Special Illustration Rare",
    "Mega Hyper Rare",
]
BASE = {"Common", "Uncommon", "Rare", "Rare Holo"}
_UNKNOWN_SPECIAL_RANK = ORDER.index("Ultra Rare")


def is_special(rarity: str | None) -> bool:
    """Double Rare and above, including rarities this list doesn't know yet."""
    return bool(rarity) and rarity not in BASE and "Rare" in rarity


def rank(rarity: str | None) -> int:
    """Sort key: higher is rarer. Unknown special rarities rank with Ultra Rare;
    anything else unknown (or missing) ranks lowest."""
    if rarity in ORDER:
        return ORDER.index(rarity)
    return _UNKNOWN_SPECIAL_RANK if is_special(rarity) else -1
