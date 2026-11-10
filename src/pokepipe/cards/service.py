"""High-level stage-1 operations, shared by the CLI and later pipeline stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from pokepipe.cards.api import PokemonTCGClient
from pokepipe.cards.cache import SetCache
from pokepipe.cards.embeddings import Embedder
from pokepipe.cards.images import DownloadReport, download_images
from pokepipe.cards.index import CardIndex
from pokepipe.cards.models import Card, CardSet


@dataclass
class FetchResult:
    card_set: CardSet
    cards: list[Card]
    images: DownloadReport


def fetch_set(
    client: PokemonTCGClient,
    sets_root: Path,
    set_id: str,
    refresh: bool = False,
    workers: int = 8,
    on_progress: Callable[[Card], None] | None = None,
) -> FetchResult:
    """Ensure a set's checklist and images are cached. Re-running only fills gaps."""
    cache = SetCache(sets_root, set_id)
    if refresh or not cache.exists():
        card_set = client.get_set(set_id)
        cards = client.get_cards(set_id)
        cache.save(card_set, cards)
    else:
        card_set, cards = cache.load_set(), cache.load_cards()
    report = download_images(
        cards, cache, client.session, workers=workers, on_progress=on_progress
    )
    return FetchResult(card_set, cards, report)


# Every pack also contains an ordinary basic Energy. It isn't a pull, but it looks like
# Energy cards that *are* pulls (a gold Basic Energy hyper rare, special Energy ultra
# rares). So the basic Energies are indexed as decoys: the pack's Energy matches a decoy
# and is dropped, while a real Energy pull matches its own card and is kept.
DECOY_SET = "sve"  # pokemontcg.io "Scarlet & Violet Energies"
DECOY_PREFIX = "decoy:"


def is_decoy(card_id: str) -> bool:
    return card_id.startswith(DECOY_PREFIX)


def ensure_decoys(client: PokemonTCGClient, sets_root: Path, workers: int = 8) -> SetCache:
    """Cache the decoy set (fetching it once) and return its cache."""
    cache = SetCache(sets_root, DECOY_SET)
    if not cache.exists() or cache.missing_images(cache.load_cards()):
        fetch_set(client, sets_root, DECOY_SET, workers=workers)
    return cache


def index_has_decoys(index: CardIndex) -> bool:
    return any(is_decoy(c) for c in index.card_ids)


def _embed(cache: SetCache, cards: list, embedder: Embedder, batch_size: int, prefix: str = ""):
    ids, chunks = [], []
    for start in range(0, len(cards), batch_size):
        batch = cards[start : start + batch_size]
        images = []
        for card in batch:
            with Image.open(cache.image_path(card.id)) as im:
                im.load()
                images.append(im)
        chunks.append(embedder.embed_images(images))
        ids.extend(prefix + c.id for c in batch)
    return ids, chunks


def build_index(
    sets_root: Path,
    set_id: str,
    embedder: Embedder,
    batch_size: int = 64,
    decoys: SetCache | None = None,
) -> CardIndex:
    """Embed every cached card image (plus decoys, if given) and save the index next
    to the set cache."""
    cache = SetCache(sets_root, set_id)
    if not cache.exists():
        raise FileNotFoundError(f"Set {set_id!r} is not cached — run `pokepipe cards fetch` first")
    cards = [c for c in cache.load_cards() if cache.image_path(c.id).exists()]
    if not cards:
        raise FileNotFoundError(f"No images cached for set {set_id!r}")

    ids, chunks = _embed(cache, cards, embedder, batch_size)
    if decoys is not None and decoys.exists() and decoys.set_id != set_id:
        decoy_cards = [c for c in decoys.load_cards() if decoys.image_path(c.id).exists()]
        d_ids, d_chunks = _embed(decoys, decoy_cards, embedder, batch_size, DECOY_PREFIX)
        ids, chunks = ids + d_ids, chunks + d_chunks

    index = CardIndex(ids, np.concatenate(chunks), embedder.model_tag)
    index.save(cache.index_path(embedder.model_tag))
    return index


def load_index(sets_root: Path, set_id: str, model_tag: str) -> CardIndex:
    path = SetCache(sets_root, set_id).index_path(model_tag)
    if not path.exists():
        raise FileNotFoundError(f"No {model_tag} index for {set_id!r} — run `pokepipe cards embed`")
    return CardIndex.load(path)
