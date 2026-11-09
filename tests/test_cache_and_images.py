import requests
import responses

from pokepipe.cards.cache import SetCache
from pokepipe.cards.images import download_images
from pokepipe.cards.models import Card, CardSet
from tests.conftest import card_payload, png_bytes


def make_cards(n=3):
    return [Card.from_api(card_payload("t", str(i), f"Mon{i}")) for i in range(1, n + 1)]


def test_cache_roundtrip(tmp_path):
    cache = SetCache(tmp_path, "t")
    assert not cache.exists()
    cards = make_cards()
    cache.save(CardSet(id="t", name="Test"), cards)
    assert cache.exists()
    assert cache.load_set().name == "Test"
    assert cache.load_cards() == cards


@responses.activate
def test_download_is_idempotent_and_reports_failures(tmp_path):
    cards = make_cards(3)
    cache = SetCache(tmp_path, "t")
    responses.get(cards[0].image_url, body=png_bytes((255, 0, 0)))
    responses.get(cards[1].image_url, body=png_bytes((0, 255, 0)))
    responses.get(cards[2].image_url, status=500)

    report = download_images(cards, cache, requests.Session(), workers=2)
    assert sorted(report.downloaded) == ["t-1", "t-2"]
    assert list(report.failed) == ["t-3"]
    assert not list(cache.images_dir.glob("*.part"))  # no partial files left behind

    report2 = download_images(cards[:2], cache, requests.Session())
    assert sorted(report2.skipped) == ["t-1", "t-2"] and not report2.downloaded
    assert cache.missing_images(cards) == [cards[2]]
