import numpy as np
import pytest
import requests
import responses
from PIL import Image

from pokepipe.cards import service
from pokepipe.cards.api import PokemonTCGClient
from pokepipe.cards.embeddings import l2_normalize, to_rgb
from pokepipe.cards.index import CardIndex
from tests.conftest import card_payload, png_bytes

BASE = "https://api.test/v2"


def unit(*rows):
    return l2_normalize(np.asarray(rows, dtype=np.float32))


def test_search_ranks_by_cosine():
    index = CardIndex(["a", "b", "c"], unit([1, 0, 0], [0, 1, 0], [0.7, 0.7, 0]), "m", "cpu")
    [hits] = index.search(unit([1, 0.1, 0]), top_k=2)
    assert [h.card_id for h in hits] == ["a", "c"]
    assert hits[0].score == pytest.approx(0.995, abs=1e-3)


def test_multi_view_card_scores_by_best_view():
    # Card "a" has two views; the second matches the query exactly.
    index = CardIndex(["a", "b", "a"], unit([0, 0, 1], [0.6, 0.8, 0], [1, 0, 0]), "m", "cpu")
    assert len(index) == 2
    [hits] = index.search(unit([1, 0, 0]), top_k=5)
    assert [(h.card_id, round(h.score, 3)) for h in hits] == [("a", 1.0), ("b", 0.6)]


def test_dim_mismatch_raises():
    index = CardIndex(["a"], unit([1, 0, 0]), "m", "cpu")
    with pytest.raises(ValueError, match="dim"):
        index.search(unit([1, 0]))


def test_save_load_roundtrip(tmp_path):
    vecs = unit([1, 2, 3], [3, 2, 1])
    path = tmp_path / "idx.npz"
    CardIndex(["x-1", "x-2"], vecs, "model__tag", "cpu").save(path)
    loaded = CardIndex.load(path, device="cpu")
    assert loaded.model_tag == "model__tag" and loaded.card_ids == ["x-1", "x-2"]
    [hits] = loaded.search(vecs[1])
    assert hits[0].card_id == "x-2" and hits[0].score == pytest.approx(1.0, abs=1e-3)


def test_to_rgb_flattens_transparency_onto_white():
    im = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert to_rgb(im).getpixel((0, 0)) == (255, 255, 255)


@responses.activate
def test_fetch_then_embed_end_to_end(tmp_path, fake_embedder):
    colors = {"1": (200, 30, 30), "2": (30, 200, 30), "3": (30, 30, 200)}
    cards = [card_payload("e2e", n, f"Mon{n}") for n in colors]
    responses.get(f"{BASE}/sets/e2e", json={"data": {"id": "e2e", "name": "E2E", "total": 3}})
    responses.get(f"{BASE}/cards", json={"data": cards, "totalCount": 3})
    for n, rgb in colors.items():
        responses.get(f"https://images.example/e2e/{n}_hires.png", body=png_bytes(rgb, mode="RGBA"))

    client = PokemonTCGClient(BASE, session=requests.Session())
    result = service.fetch_set(client, tmp_path, "e2e")
    assert len(result.cards) == 3 and len(result.images.downloaded) == 3

    # Second fetch hits the cache: no new HTTP calls needed.
    calls_before = len(responses.calls)
    assert len(service.fetch_set(client, tmp_path, "e2e").images.skipped) == 3
    assert len(responses.calls) == calls_before

    service.build_index(tmp_path, "e2e", fake_embedder, batch_size=2)
    index = service.load_index(tmp_path, "e2e", fake_embedder.model_tag)

    query = fake_embedder.embed_images([Image.new("RGB", (10, 10), colors["2"])])
    assert index.search(query, top_k=1)[0][0].card_id == "e2e-2"


def test_build_index_adds_prefixed_decoys(tmp_path, fake_embedder):
    from pokepipe.cards.cache import SetCache
    from pokepipe.cards.models import Card, CardSet

    def cached(set_id, colours):
        cache = SetCache(tmp_path, set_id)
        cards = [Card(id=f"{set_id}-{i}", name=f"C{i}", number=str(i), set_id=set_id)
                 for i in range(len(colours))]
        cache.save(CardSet(id=set_id, name=set_id), cards)
        cache.images_dir.mkdir(parents=True)
        for card, rgb in zip(cards, colours, strict=True):
            Image.new("RGB", (40, 56), rgb).save(cache.image_path(card.id))
        return cache

    cached("main", [(200, 0, 0), (0, 200, 0)])
    decoys = cached("sve", [(0, 0, 200)])
    index = service.build_index(tmp_path, "main", fake_embedder, decoys=decoys)
    assert index.unique_ids == ["main-0", "main-1", "decoy:sve-0"]
    assert service.index_has_decoys(index)
    assert not service.index_has_decoys(service.build_index(tmp_path, "main", fake_embedder))
    # a set is never its own decoy
    assert service.build_index(tmp_path, "sve", fake_embedder, decoys=decoys).unique_ids == [
        "sve-0"]


def test_build_index_requires_cache(tmp_path, fake_embedder):
    with pytest.raises(FileNotFoundError, match="fetch"):
        service.build_index(tmp_path, "missing", fake_embedder)
