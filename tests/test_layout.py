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


def test_missing_card_infers_only_what_every_placement_agrees_on():
    pulls = pack(C, C, C, U, U, U, R, U, R)  # one common missed
    result = apply_layout(pulls, SCARLET_VIOLET)
    assert result.mode == "partial" and "missing" in result.note
    assert summary(pulls)[:3] == [(None, False)] * 3  # which common is missing: unknown
    assert summary(pulls)[3:6] == [(5, False), (6, False), (7, False)]
    assert summary(pulls)[6:] == [(8, True), (9, True), (10, False)]


def test_ambiguous_missing_slot_gives_no_reverse_guess():
    # Real Delta Reign pack with one card missed: the 3rd U is either slot 7 (if slot 8
    # was missed) or slot 8, a reverse holo (if an earlier card was missed). Don't guess.
    pulls = pack(C, C, C, C, U, U, U, IR, R)
    result = apply_layout(pulls, SCARLET_VIOLET)
    assert result.mode == "partial" and result.reverse_ids == []
    assert pulls[6].slot is None and not pulls[6].reverse_guess
    # The Rare could be the rare slot (10) or a reverse-holo Rare in slot 9 if the
    # rare-slot card was the one missed: also ambiguous, so also no guess.
    assert (pulls[8].slot, pulls[8].reverse_guess) == (None, False)
    assert [p.slot for p in pulls[:4]] == [1, 2, 3, 4]  # the commons are unambiguous


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
    ("Scarlet & Violet", "modern"), ("Mega Evolution", "modern"),
    ("Sword & Shield", None), (None, None)])
def test_layout_for_series(series, expected):
    layout = layout_for(series)
    assert (layout.name if layout else None) == expected


def test_delta_reign_packs_fit_including_new_rarities():
    # Real Delta Reign packs: Raikou ex (DR) in the rare slot, Kecleon IR in slot 9,
    # and a Mega Hyper Rare (new in this era) must also fit the rare slot.
    raikou = pack(C, C, C, C, U, U, U, C, R, "Double Rare")
    assert apply_layout(raikou, SCARLET_VIOLET).mode == "full"
    assert [p.reverse_guess for p in raikou[7:]] == [True, True, False]  # Kommo-o R = reverse
    kecleon = pack(C, C, C, C, U, U, U, C, IR, R)
    apply_layout(kecleon, SCARLET_VIOLET)
    assert [p.reverse_guess for p in kecleon[7:]] == [True, False, False]
    mega = pack(C, C, C, C, U, U, U, C, U, "Mega Hyper Rare")
    assert apply_layout(mega, SCARLET_VIOLET).mode == "full"
