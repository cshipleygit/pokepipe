"""Stage 4: apply a reviewed video to a collection and report what changed."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from pokepipe.cards.cache import SetCache
from pokepipe.collection.master import MasterSet
from pokepipe.collection.models import (
    Collection,
    Contribution,
    PackResult,
    PullResult,
    VideoReport,
    slot,
)
from pokepipe.review.store import ReviewStore

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
MANUAL = "manual"


class CollectionError(ValueError):
    pass


class CollectionStore:
    """One JSON file per (collection name, set): <root>/<name>/<set_id>.json."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, name: str, set_id: str) -> Path:
        if not NAME_RE.match(name):
            raise CollectionError(f"collection name must be lowercase letters/digits/-/_: {name!r}")
        return self.root / name / f"{set_id}.json"

    def load(self, name: str, set_id: str) -> Collection:
        path = self.path(name, set_id)
        if not path.exists():
            return Collection(name=name, set_id=set_id)
        return Collection.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, coll: Collection) -> None:
        path = self.path(coll.name, coll.set_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(coll.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)


def load_master(sets_root: Path, set_id: str) -> MasterSet:
    cache = SetCache(sets_root, set_id)
    if not cache.exists():
        raise CollectionError(f"set {set_id!r} is not cached — run `pokepipe cards fetch {set_id}`")
    return MasterSet(cache.load_set(), cache.load_cards())


def apply_video(
    store: CollectionStore,
    reviews: ReviewStore,
    sets_root: Path,
    slug: str,
    name: str = "main",
) -> VideoReport:
    """Add a finished review's kept pulls to collection `name`. Idempotent: re-applying
    the same video replaces its earlier contribution."""
    review = reviews.load(slug)
    if not review.completed:
        raise CollectionError(f"review for {slug!r} isn't finished — finish it in the review page")
    master = load_master(sets_root, review.set_id)
    cards = master.by_id
    coll = store.load(name, review.set_id)

    before = coll.counts(exclude=slug)
    running: Counter[str] = Counter(before)
    results: list[PullResult] = []
    warnings: list[str] = []
    for pull in review.kept():
        card = cards.get(pull.card_id)
        if card is None:
            warnings.append(f"{pull.card_id} is not in {review.set_id}; skipped")
            continue
        variant, warning = master.resolve_variant(card, pull.reverse_holo)
        if warning:
            warnings.append(warning)
        s = slot(card.id, variant)
        new_card = all(running[slot(card.id, v)] == 0 for v in card.variant_list())
        new = running[s] == 0
        running[s] += 1
        results.append(PullResult(
            card_id=card.id, name=card.name, number=card.number, rarity=card.rarity,
            variant=variant, pack=pull.pack, time=pull.time,
            new=new, new_card=new_card, copy_number=running[s],
        ))

    packs = _pack_progress(master, before, results)
    coll.contributions[slug] = Contribution(
        source=slug, slots=[slot(r.card_id, r.variant) for r in results]
    )
    store.save(coll)

    report = VideoReport(
        video_slug=slug, collection=name, set_id=review.set_id,
        set_name=master.card_set.name,
        before=master.progress(before), after=master.progress(running),
        packs=packs, pulls=results, warnings=warnings,
    )
    out = reviews.video_dir(slug) / "collection.json"
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def _pack_progress(
    master: MasterSet, before: Counter[str], pulls: list[PullResult]
) -> list[PackResult]:
    """Progress after each pack, in pack order — each pack becomes its own post, and its
    overlay should show the count as of the end of that pack."""
    running = Counter(before)
    out = []
    for idx in sorted({p.pack for p in pulls}):
        in_pack = [p for p in pulls if p.pack == idx]
        running.update(slot(p.card_id, p.variant) for p in in_pack)
        out.append(PackResult(
            index=idx, new_slots=sum(p.new for p in in_pack),
            progress_after=master.progress(running),
        ))
    return out


def remove_video(store: CollectionStore, set_id: str, slug: str, name: str = "main") -> bool:
    coll = store.load(name, set_id)
    removed = coll.contributions.pop(slug, None) is not None
    if removed:
        store.save(coll)
    return removed


def adjust_manual(
    store: CollectionStore,
    sets_root: Path,
    set_id: str,
    card_id: str,
    variant: str | None = None,
    count: int = 1,
    name: str = "main",
) -> str:
    """Record cards owned outside any video (e.g. your existing binder). Negative
    `count` removes manual copies (never ones that came from a video)."""
    master = load_master(sets_root, set_id)
    card = master.by_id.get(card_id)
    if card is None:
        raise CollectionError(f"{card_id!r} is not in {set_id}")
    variant = variant or card.base_variant
    if variant not in card.variant_list():
        raise CollectionError(f"{card.name} #{card.number} has no {variant!r} printing "
                              f"(has: {', '.join(card.variant_list())})")
    s = slot(card_id, variant)
    coll = store.load(name, set_id)
    manual = coll.contributions.setdefault(MANUAL, Contribution(source=MANUAL))
    if count >= 0:
        manual.slots.extend([s] * count)
    else:
        for _ in range(-count):
            if s not in manual.slots:
                raise CollectionError(f"no manual copies of {s} left to remove")
            manual.slots.remove(s)
    store.save(coll)
    return s
