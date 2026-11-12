"""Prices (sheet over API, exact printing only), video valuation, and the SFX track."""

import numpy as np
import pytest

from pokepipe.cards.cache import SetCache
from pokepipe.cards.models import Card, CardSet
from pokepipe.cards.prices import Price, PriceBook, read_sheet, write_sheet
from pokepipe.collection.models import PackResult, Progress, PullResult, VideoReport
from pokepipe.collection.values import video_values
from pokepipe.render import sfx


def card(cid, prices):
    return Card(id=cid, name=cid, number="1", set_id="s1", prices=prices,
                prices_updated="2026/11/01")


def test_api_prices_are_parsed_per_printing():
    raw = {"id": "s1-1", "name": "Mon", "number": "1", "set": {"id": "s1"},
           "tcgplayer": {"updatedAt": "2026/11/01", "prices": {
               "holofoil": {"market": 3.5}, "reverseHolofoil": {"market": None}}}}
    c = Card.from_api(raw)
    assert c.prices == {"holo": 3.5} and c.variants == ["holo", "reverse"]


def test_sheet_overrides_api_and_lookup_is_exact(tmp_path):
    cache = SetCache(tmp_path / "sets", "s1")
    cache.save(CardSet(id="s1", name="S"),
               [card("s1-1", {"holo": 2.0, "reverse": 1.0}), card("s1-2", {})])
    write_sheet(tmp_path / "prices" / "s1.csv",
                [Price(card_id="s1-1", variant="holo", usd=9.99, source="manual", as_of="d")])
    assert read_sheet(tmp_path / "prices" / "s1.csv")[0].usd == 9.99
    book = PriceBook.load(tmp_path / "sets", "s1", tmp_path / "prices")
    assert book.get("s1-1", "holo").usd == 9.99  # the sheet wins
    assert book.get("s1-1", "reverse").source.startswith("tcgplayer")
    assert book.get("s1-1", "normal") is None and book.get("s1-2", "holo") is None


def test_video_values_price_each_pull_and_count_the_gaps():
    prog = Progress(master_owned=1, master_total=9, full_owned=1, full_total=9,
                    base_owned=1, base_total=9)
    pulls = [PullResult(card_id=c, name=c, number="1", rarity="Rare", variant=v, pack=1,
                        time=t, new=True, new_card=True, copy_number=1)
             for c, v, t in [("a", "holo", 1), ("b", "reverse", 2), ("c", "normal", 3)]]
    report = VideoReport(video_slug="v", collection="t", set_id="s1", set_name="S",
                         before=prog, after=prog, packs=[PackResult(index=1, new_slots=3,
                         progress_after=prog)], pulls=pulls)
    book = PriceBook({("a", "holo"): Price(card_id="a", variant="holo", usd=5.0, source="x"),
                      ("b", "reverse"): Price(card_id="b", variant="reverse", usd=0.25,
                                              source="x")})
    v = video_values(report, book)
    assert v.total_usd == 5.25 and v.priced == 2 and v.unpriced == 1
    assert v.per_pack_usd == {1: 5.25}


def test_sfx_track_places_effects_at_their_times():
    track = sfx.sfx_track([(0.5, "coin"), (1.5, "chime")], 3.0)
    assert track.shape == (int(3.0 * sfx.SR) + 1, 2)
    loud = np.flatnonzero(np.abs(track[:, 0]) > 0.05) / sfx.SR
    assert loud[0] == pytest.approx(0.5, abs=0.01)  # silent before the first effect
    assert np.abs(track).max() <= 1.0
    assert set(sfx.EFFECTS) == {"sparkle", "chime", "coin", "whoosh", "rocket"}
