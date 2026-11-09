import pytest

from pokepipe.detection.layout import SCARLET_VIOLET, apply_layout, layout_for
from pokepipe.detection.models import Pull

C, U, R, IR, SIR, UR = (
    "Common", "Uncommon", "Rare", "Illustration Rare", "Special Illustration Rare", "Ultra Rare")


def pack(*rarities: str) -> list[Pull]:
    return [
        Pull(card_id=f"s-{i}", name=f"c{i}", number=str(i), rarity=r, start=float(i),
             end=i + 0.5, peak_time=float(i), peak_score=0.8, mean_margin=0.1, frames=4,
             confidence=0.9, box=(0, 0, 1, 1), pack=1)
        for i, r in enumerate(rarities)
    ]


def summary(pulls):
    return [(p.slot, p.reverse_guess) for p in pulls]


def test_full_pack_marks_reverse_slots():
    # pack 1 of the test video: two Rares, the first in a reverse slot
    pulls = pack(C, C, C, C, U, U, U, R, U, R)
    result = apply_layout(pulls, SCARLET_VIOLET)
    assert result.mode == "full" and result.reverse_ids == [7, 8]
    assert [p.slot for p in pulls] == list(range(1, 11))
    assert [p.reverse_guess for p in pulls] == [False] * 7 + [True, True, False]


def test_special_rare_in_slot_9_is_not_a_reverse_holo():
    # pack 9: Charizard ex SIR replaced the second reverse holo
    pulls = pack(C, C, C, C, U, U, U, C, SIR, R)
    apply_layout(pulls, SCARLET_VIOLET)
    assert [p.reverse_guess for p in pulls[7:]] == [True, False, False]


def test_ultra_rare_in_rare_slot():
    pulls = pack(C, C, C, C, U, U, U, C, C, UR)
    assert apply_layout(pulls, SCARLET_VIOLET).mode == "full"
    assert pulls[9].reverse_guess is False


def test_shuffled_pack_yields_no_guesses():
    pulls = pack(R, C, C, C, U, U, U, C, U, C)  # rare moved to the front
    result = apply_layout(pulls, SCARLET_VIOLET)
    assert result.mode is None and "doesn't fit" in result.note
    assert summary(pulls) == [(None, False)] * 10


def test_missing_card_aligns_from_the_end():
    pulls = pack(C, C, C, U, U, U, R, U, R)  # one common missed
    result = apply_layout(pulls, SCARLET_VIOLET)
    assert result.mode == "tail" and "missing" in result.note
    assert summary(pulls)[:6] == [(None, False)] * 6  # early slots stay unknown
    assert summary(pulls)[6:] == [(8, True), (9, True), (10, False)]


def test_tail_that_does_not_fit_is_rejected():
    pulls = pack(C, C, C, U, U, U, R, C, C)  # last card isn't a rare-slot rarity
    assert apply_layout(pulls, SCARLET_VIOLET).mode is None


def test_extra_cards_yield_no_guesses():
    pulls = pack(C, C, C, C, C, U, U, U, C, U, R)
    result = apply_layout(pulls, SCARLET_VIOLET)
    assert result.mode is None and "expected 10" in result.note


def test_input_order_does_not_matter():
    pulls = pack(C, C, C, C, U, U, U, R, U, R)
    apply_layout(list(reversed(pulls)), SCARLET_VIOLET)  # sorted by start internally
    assert pulls[7].slot == 8 and pulls[7].reverse_guess


@pytest.mark.parametrize("series,expected", [
    ("Scarlet & Violet", "scarlet-violet"), ("Sword & Shield", None), (None, None)])
def test_layout_for_series(series, expected):
    layout = layout_for(series)
    assert (layout.name if layout else None) == expected
