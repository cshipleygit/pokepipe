"""Repeat showings, modelled on a real video where the opener deliberately re-shows cards."""

from pokepipe.detection.models import Pull
from pokepipe.detection.packs import PackParams
from pokepipe.detection.reshows import merge_reshows


def pull(card: str, t: float, rarity: str = "Common") -> Pull:
    return Pull(card_id=card, name=card, number="1", rarity=rarity, start=t, end=t + 1.0,
                peak_time=t + 0.5, peak_score=0.8, mean_margin=0.1, frames=4, confidence=0.9,
                box=(0.3, 0.3, 0.6, 0.7))


def pack_of(names: list[str], t0: float, step: float = 3.0) -> list[Pull]:
    return [pull(n, t0 + i * step) for i, n in enumerate(names)]


PACK_A = [f"a{i}" for i in range(9)] + ["hitA"]  # 10 cards, hit last
PACK_B = [f"b{i}" for i in range(9)] + ["hitB"]
PARAMS = PackParams(cards_per_pack=10)


def ids(pulls):
    return [p.card_id for p in pulls]


def test_repeat_inside_a_pack_counts_once_and_keeps_the_footage():
    pulls = pack_of(PACK_A, 100)
    pulls.append(pull("hitA", 131))  # held up again 4 s later, same pack
    pulls.append(pull("a3", 133))
    merged, packs = merge_reshows(pulls, 600, PARAMS)
    assert ids(merged) == PACK_A and [p.cards for p in packs] == [10]
    hit = next(p for p in merged if p.card_id == "hitA")
    assert [s.start for s in hit.showings] == [131]
    assert hit.start == 127  # the first showing stays the reveal


def test_satellite_reshow_preview_and_recap_merge_into_their_packs():
    pulls = (
        [pull("b0", 185)]               # preview: glimpse of pack B's first card
        + pack_of(PACK_A, 100)          # pack A: 100-127
        + [pull("hitA", 145)]           # re-show of pack A's hit after the pack
        + pack_of(PACK_B, 200)          # pack B: 200-227
        + [pull("hitA", 400)]           # end-of-video recap of pack A's hit
    )
    merged, packs = merge_reshows(pulls, 600, PARAMS)
    assert [p.cards for p in packs] == [10, 10]  # just the two real packs
    assert len(merged) == 20
    hit_a = next(p for p in merged if p.card_id == "hitA")
    assert [s.start for s in hit_a.showings] == [145, 400]
    b0 = next(p for p in merged if p.card_id == "b0")
    assert b0.start == 200 and [s.start for s in b0.showings] == [185]  # reveal stays in-pack


def test_merged_repeats_do_not_split_a_pack():
    # The opener holds one card up for a while (seen 4x); then a 12 s pause before the
    # next card. Merged, that pause must not look like a gap between packs.
    names = PACK_A[:6]
    pulls = pack_of(names, 100) + [pull("a5", 120), pull("a5", 124)]  # a5 at 115, 120, 124
    pulls += [pull(n, 130 + i * 3) for i, n in enumerate(PACK_A[6:])]  # resumes at 130
    merged, packs = merge_reshows(pulls, 600, PARAMS)
    assert [p.cards for p in packs] == [10]


def test_same_card_in_two_full_packs_is_two_copies():
    names_b = ["a0"] + PACK_B[1:]  # pack B also contains an "a0"
    pulls = pack_of(PACK_A, 100) + pack_of(names_b, 200)
    merged, packs = merge_reshows(pulls, 600, PARAMS)
    assert ids(merged).count("a0") == 2 and [p.cards for p in packs] == [10, 10]
    assert all(not p.showings for p in merged)


def test_unmatched_satellite_stays_for_review():
    # A card shown before any pack (e.g. a box promo) isn't in any pack: keep it,
    # as its own short group the reviewer will see flagged.
    promo = [pull("promo", 30, "Ultra Rare"), pull("promo", 34, "Ultra Rare")]
    pulls = promo + pack_of(PACK_A, 100)
    merged, packs = merge_reshows(pulls, 600, PARAMS)
    assert [p.cards for p in packs] == [1, 10] and not packs[0].complete
    promo = merged[0]
    assert promo.card_id == "promo" and len(promo.showings) == 1
