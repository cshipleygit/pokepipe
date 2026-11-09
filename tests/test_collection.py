import json
from collections import Counter

import pytest

from pokepipe.cards.cache import SetCache
from pokepipe.cards.models import Card, CardSet, variants_from_price_keys
from pokepipe.collection import service
from pokepipe.collection.master import MasterSet
from pokepipe.detection.models import DetectionParams, PullsFile
from pokepipe.review.models import ReviewedPull, ReviewState, Status
from pokepipe.review.store import ReviewStore

# Set "s": printed total 3, plus one secret rare.
CARDS = [
    Card(id="s-1", name="Pichu", number="1", set_id="s", rarity="Common",
         variants=["normal", "reverse"]),
    Card(id="s-2", name="Raichu", number="2", set_id="s", rarity="Rare",
         variants=["holo", "reverse"]),
    Card(id="s-3", name="Plusle", number="3", set_id="s", rarity="Uncommon"),  # no price data
    Card(id="s-4", name="Pikachu ex", number="4", set_id="s",
         rarity="Special Illustration Rare", variants=["holo"]),
]
CARD_SET = CardSet(id="s", name="Sparks", printedTotal=3, total=4)


def master() -> MasterSet:
    return MasterSet(CARD_SET, CARDS)


# --- card variants -------------------------------------------------------------------


def test_variants_from_api_price_keys():
    assert variants_from_price_keys({"reverseHolofoil": {}, "normal": {}}) == ["normal", "reverse"]
    assert variants_from_price_keys({"holofoil": {}, "unknownKey": {}}) == ["holo"]
    raw = {"id": "s-9", "name": "X", "number": "9", "tcgplayer": {"prices": {"holofoil": {}}}}
    assert Card.from_api(raw).variants == ["holo"]


def test_variant_fallback_by_rarity_and_base_variant():
    assert CARDS[2].variant_list() == ["normal", "reverse"]  # rule, no price data
    assert Card(id="x", name="x", number="1", set_id="s", rarity="Rare").variant_list() == [
        "holo", "reverse"]
    assert CARDS[1].base_variant == "holo"
    assert CARDS[3].base_variant == "holo"


# --- progress ------------------------------------------------------------------------


def test_progress_levels():
    m = master()
    assert len(m.slots()) == 7
    p = m.progress(Counter({"s-1/reverse": 1, "s-4/holo": 2}))
    assert (p.master_owned, p.master_total) == (2, 7)
    assert (p.full_owned, p.full_total) == (2, 4)  # Pichu (any printing) + Pikachu ex
    assert (p.base_owned, p.base_total) == (1, 3)  # secret rare doesn't count for base
    assert m.progress(Counter()).master_pct == 0


def test_missing_filters_by_variant():
    m = master()
    counts = Counter({"s-1/normal": 1})
    assert len(m.missing(counts)) == 6
    assert [(c.id, v) for c, v in m.missing(counts, "reverse")] == [
        ("s-1", "reverse"), ("s-2", "reverse"), ("s-3", "reverse")]


def test_reverse_request_on_card_without_reverse_warns():
    variant, warning = master().resolve_variant(CARDS[3], reverse_holo=True)
    assert variant == "holo" and "no reverse holo" in warning


# --- applying videos -----------------------------------------------------------------


@pytest.fixture
def env(tmp_path):
    sets_root, videos_root = tmp_path / "sets", tmp_path / "videos"
    SetCache(sets_root, "s").save(CARD_SET, CARDS)
    reviews = ReviewStore(videos_root, sets_root)
    store = service.CollectionStore(tmp_path / "collection")
    return store, reviews, sets_root, videos_root


def write_review(videos_root, slug, pulls, completed=True):
    vdir = videos_root / slug
    vdir.mkdir(parents=True)
    det = PullsFile(video="v.mp4", set_id="s", model_tag="m", duration=100, pulls=[],
                    params=DetectionParams(fps=4, frame_min_score=0, frame_min_margin=0,
                                           max_gap=1, min_frames=1, min_peak=0))
    (vdir / "pulls.json").write_text(det.model_dump_json())
    state = ReviewState(video_slug=slug, set_id="s", completed=completed, pulls=[
        ReviewedPull(id=i, card_id=c, pack=pack, time=t, status=status, reverse_holo=rev)
        for i, (c, pack, t, rev, status) in enumerate(pulls)
    ])
    (vdir / "review.json").write_text(state.model_dump_json())


C, R = Status.CONFIRMED, Status.REJECTED


def test_apply_marks_new_duplicates_and_new_cards(env):
    store, reviews, sets_root, videos_root = env
    write_review(videos_root, "v1", [
        ("s-1", 1, 1.0, False, C),   # new card, new slot
        ("s-1", 1, 2.0, True, C),    # same card, new printing (reverse) → new slot, not new card
        ("s-2", 2, 30.0, False, C),  # new
        ("s-1", 2, 31.0, False, C),  # duplicate
        ("s-3", 2, 32.0, False, R),  # rejected in review → ignored
    ])
    report = service.apply_video(store, reviews, sets_root, "v1", "test")
    got = [(p.card_id, p.variant, p.new, p.new_card, p.copy_number) for p in report.pulls]
    assert got == [
        ("s-1", "normal", True, True, 1),
        ("s-1", "reverse", True, False, 1),
        ("s-2", "holo", True, True, 1),
        ("s-1", "normal", False, False, 2),
    ]
    assert report.new_count == 3
    assert (report.before.master_owned, report.after.master_owned) == (0, 3)
    assert [(p.index, p.new_slots, p.progress_after.master_owned) for p in report.packs] == [
        (1, 2, 2), (2, 1, 3)]
    saved = json.loads((videos_root / "v1" / "collection.json").read_text())
    assert saved["after"]["master_owned"] == 3


def test_reapply_is_idempotent_and_later_videos_see_earlier_ones(env):
    store, reviews, sets_root, videos_root = env
    write_review(videos_root, "v1", [("s-1", 1, 1.0, False, C)])
    write_review(videos_root, "v2", [("s-1", 1, 1.0, False, C), ("s-4", 1, 2.0, False, C)])
    service.apply_video(store, reviews, sets_root, "v1", "test")
    service.apply_video(store, reviews, sets_root, "v1", "test")  # again: no double count
    report = service.apply_video(store, reviews, sets_root, "v2", "test")
    assert [p.new for p in report.pulls] == [False, True]  # Pichu already owned via v1
    assert store.load("test", "s").counts()["s-1/normal"] == 2

    assert service.remove_video(store, "s", "v1", "test")
    assert store.load("test", "s").counts()["s-1/normal"] == 1
    assert not service.remove_video(store, "s", "v1", "test")


def test_apply_requires_finished_review(env):
    store, reviews, sets_root, videos_root = env
    write_review(videos_root, "v1", [("s-1", 1, 1.0, False, C)], completed=False)
    with pytest.raises(service.CollectionError, match="isn't finished"):
        service.apply_video(store, reviews, sets_root, "v1", "test")


def test_collections_are_isolated_by_name(env):
    store, reviews, sets_root, videos_root = env
    write_review(videos_root, "v1", [("s-1", 1, 1.0, False, C)])
    service.apply_video(store, reviews, sets_root, "v1", "test")
    assert store.load("main", "s").counts() == Counter()
    with pytest.raises(service.CollectionError, match="name"):
        store.path("../evil", "s")


def test_manual_adjustments(env):
    store, _, sets_root, _ = env
    assert service.adjust_manual(store, sets_root, "s", "s-2", count=2, name="t") == "s-2/holo"
    service.adjust_manual(store, sets_root, "s", "s-2", "reverse", name="t")
    service.adjust_manual(store, sets_root, "s", "s-2", count=-1, name="t")
    assert store.load("t", "s").counts() == Counter({"s-2/holo": 1, "s-2/reverse": 1})
    with pytest.raises(service.CollectionError, match="no 'reverse' printing"):
        service.adjust_manual(store, sets_root, "s", "s-4", "reverse", name="t")
    with pytest.raises(service.CollectionError, match="no manual copies"):
        service.adjust_manual(store, sets_root, "s", "s-1", count=-1, name="t")
    with pytest.raises(service.CollectionError, match="not in"):
        service.adjust_manual(store, sets_root, "s", "zz-1", name="t")
