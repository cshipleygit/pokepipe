from pokepipe.cards.models import Card, CardSet
from tests.conftest import card_payload


def test_card_from_api_maps_fields():
    card = Card.from_api(card_payload("sv3pt5", "6", "Charizard ex", "Double Rare"))
    assert card.id == "sv3pt5-6"
    assert card.set_id == "sv3pt5"
    assert card.rarity == "Double Rare"
    assert card.image_url.endswith("6_hires.png")  # prefers hi-res


def test_card_falls_back_to_small_image():
    raw = card_payload("s", "1", "X")
    del raw["images"]["large"]
    assert Card.from_api(raw).image_url.endswith("/1.png")


def test_collector_number_ordering():
    numbers = ["TG05", "10", "2", "GG10", "GG01", "100", "SWSH001"]
    cards = [Card(id=f"s-{n}", name=n, number=n, set_id="s") for n in numbers]
    ordered = [c.number for c in sorted(cards, key=Card.sort_key)]
    assert ordered == ["2", "10", "100", "GG01", "GG10", "SWSH001", "TG05"]


def test_cardset_from_api_reads_aliases_and_images():
    s = CardSet.from_api(
        {
            "id": "sv3pt5", "name": "151", "series": "Scarlet & Violet",
            "printedTotal": 165, "total": 207, "releaseDate": "2023/09/22",
            "images": {"logo": "L", "symbol": "S"},
        }
    )
    assert (s.printed_total, s.total, s.logo_url, s.symbol_url) == (165, 207, "L", "S")
