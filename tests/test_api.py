import pytest
import requests
import responses
from responses import matchers

from pokepipe.cards.api import PAGE_SIZE, PokemonTCGClient, PokemonTCGError
from tests.conftest import card_payload

BASE = "https://api.test/v2"


def client() -> PokemonTCGClient:
    # Plain session (no retries) keeps tests fast and deterministic.
    return PokemonTCGClient(BASE, session=requests.Session())


@responses.activate
def test_get_cards_paginates_and_sorts():
    page1 = [card_payload("big", str(n), f"C{n}") for n in range(PAGE_SIZE, 0, -1)]
    page2 = [card_payload("big", str(n), f"C{n}") for n in (PAGE_SIZE + 2, PAGE_SIZE + 1)]
    total = PAGE_SIZE + 2
    for page, data in ((1, page1), (2, page2)):
        responses.get(
            f"{BASE}/cards",
            json={"data": data, "totalCount": total},
            match=[matchers.query_param_matcher(
                {"q": "set.id:big", "page": str(page), "pageSize": str(PAGE_SIZE)}
            )],
        )
    cards = client().get_cards("big")
    assert len(cards) == total
    assert [c.number for c in cards[:3]] == ["1", "2", "3"]
    assert cards[-1].number == str(total)


@responses.activate
def test_get_set_404_raises():
    responses.get(f"{BASE}/sets/nope", status=404)
    with pytest.raises(PokemonTCGError, match="Not found"):
        client().get_set("nope")


@responses.activate
def test_search_sets_filters_by_name_or_id():
    sets = [
        {"id": "sv3pt5", "name": "151"},
        {"id": "sv8", "name": "Surging Sparks"},
        {"id": "base1", "name": "Base"},
    ]
    responses.get(f"{BASE}/sets", json={"data": sets, "totalCount": 3})
    assert [s.id for s in client().search_sets("spark")] == ["sv8"]
    assert [s.id for s in client().search_sets("SV3")] == ["sv3pt5"]


@responses.activate
def test_retry_session_recovers_from_502():
    from pokepipe.cards.api import build_session

    responses.get(f"{BASE}/sets/sv1", status=502)
    responses.get(f"{BASE}/sets/sv1", json={"data": {"id": "sv1", "name": "SV"}})
    session = build_session(retries=2)
    session.adapters["https://"].max_retries.backoff_factor = 0
    c = PokemonTCGClient(BASE, session=session)
    assert c.get_set("sv1").name == "SV"


def test_api_key_header():
    from pokepipe.cards.api import build_session

    assert build_session("abc").headers["X-Api-Key"] == "abc"
    assert "X-Api-Key" not in build_session(None).headers
