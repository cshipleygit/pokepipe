import pytest

from pokepipe.cards.rarity import is_special, rank
from pokepipe.render.timeline import RenderPull


@pytest.mark.parametrize("rarity,special", [
    ("Common", False), ("Uncommon", False), ("Rare", False), ("Rare Holo", False), (None, False),
    ("Double Rare", True), ("Special Illustration Rare", True),
    ("Mega Hyper Rare", True),         # new in the Mega Evolution era
    ("Galaxy Sparkle Rare", True),     # an invented future rarity still counts
    ("Promo", False),
])
def test_is_special(rarity, special):
    assert is_special(rarity) is special


def test_rank_orders_known_and_places_unknown_specials_with_ultra_rares():
    assert rank("Common") < rank("Rare") < rank("Double Rare") < rank("Special Illustration Rare")
    assert rank("Mega Hyper Rare") > rank("Special Illustration Rare")
    assert rank("Galaxy Sparkle Rare") == rank("Ultra Rare")
    assert rank(None) == rank("Promo") == -1


def pull(rarity, variant="holo"):
    return RenderPull(id="p1-1", pack=1, name="X", rarity=rarity, variant=variant, new=True,
                      src_start=0, src_end=1, box=None)


def test_render_hits_and_labels_follow_rarity_rules():
    assert pull("Mega Hyper Rare").hit and pull("Mega Hyper Rare").label == "MEGA HYPER RARE"
    assert pull("Rare").hit and pull("Rare").label is None  # the pack's rare: zoom, no label
    assert not pull("Rare", variant="reverse").hit  # reverse-holo slot rare
    assert not pull("Common").hit
