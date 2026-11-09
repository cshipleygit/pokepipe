"""Grouping packs into posts. The user decides; we only suggest a default.

Spec syntax: comma-separated groups, each a pack number or an inclusive range, joined
with '+' to combine non-adjacent packs into one post:  "1-3, 9, 10+12"
"""

from __future__ import annotations

import re

from pokepipe.narration.models import Post, PostPlan

# Rarities strong enough to carry a post on their own.
HIT_RARITIES = {
    "Illustration Rare",
    "Special Illustration Rare",
    "Ultra Rare",
    "Hyper Rare",
    "Double Rare",
}

_PART = re.compile(r"^(\d+)(?:-(\d+))?$")


class PostSpecError(ValueError):
    pass


def parse_spec(spec: str, available: set[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    seen: set[int] = set()
    for raw_group in spec.split(","):
        if not raw_group.strip():
            continue
        group: list[int] = []
        for part in raw_group.split("+"):
            part = part.strip()
            m = _PART.match(part)
            if not m:
                raise PostSpecError(f"can't read {part!r} — use e.g. '1-3, 9, 10+12'")
            lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
            if hi < lo:
                raise PostSpecError(f"range {part!r} goes backwards")
            group.extend(range(lo, hi + 1))
        if not group:
            continue
        missing = [p for p in group if p not in available]
        if missing:
            raise PostSpecError(
                f"no pack(s) {missing}; this video has packs {min(available)}-{max(available)}"
            )
        dupes = [p for p in group if p in seen]
        if dupes:
            raise PostSpecError(f"pack(s) {dupes} are in more than one post")
        seen.update(group)
        groups.append(sorted(group))
    if not groups:
        raise PostSpecError("no posts in spec")
    return groups


def suggest(
    packs_with_rarities: dict[int, list[str | None]], max_per_post: int = 3
) -> list[list[int]]:
    """Default grouping: a pack with a big hit gets its own post (the hit is the hook);
    the packs between hits are grouped in runs of up to `max_per_post`."""
    groups: list[list[int]] = []
    run: list[int] = []
    for pack in sorted(packs_with_rarities):
        if HIT_RARITIES & set(packs_with_rarities[pack]):
            if run:
                groups.append(run)
                run = []
            groups.append([pack])
            continue
        run.append(pack)
        if len(run) == max_per_post:
            groups.append(run)
            run = []
    if run:
        groups.append(run)
    return groups


def to_spec(groups: list[list[int]]) -> str:
    def fmt(g: list[int]) -> str:
        if len(g) > 1 and g == list(range(g[0], g[-1] + 1)):
            return f"{g[0]}-{g[-1]}"
        return "+".join(map(str, g))

    return ", ".join(fmt(g) for g in groups)


def make_plan(slug: str, groups: list[list[int]]) -> PostPlan:
    return PostPlan(
        video_slug=slug, posts=[Post(index=i, packs=g) for i, g in enumerate(groups, 1)]
    )
