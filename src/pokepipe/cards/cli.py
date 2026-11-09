"""`pokepipe cards ...` — stage 1 commands."""

from __future__ import annotations

import time
from pathlib import Path

import typer
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from pokepipe.cards import service
from pokepipe.cards.api import PokemonTCGClient
from pokepipe.cards.cache import SetCache
from pokepipe.config import get_settings

app = typer.Typer(help="Stage 1: card checklists, images and embedding indexes.")
console = Console()


def _client() -> PokemonTCGClient:
    s = get_settings()
    return PokemonTCGClient(s.pokemontcg_base_url, api_key=s.pokemontcg_api_key)


def _embedder():
    from pokepipe.cards.embeddings import OpenClipEmbedder

    s = get_settings()
    return OpenClipEmbedder(s.clip_model, s.clip_pretrained, fit=s.clip_fit)


@app.command()
def sets(search: str = typer.Option(None, "--search", "-s", help="Filter by name or id")):
    """List available sets (newest first)."""
    table = Table("id", "name", "series", "released", "cards")
    for st in _client().search_sets(search):
        table.add_row(st.id, st.name, st.series, st.release_date, str(st.total))
    console.print(table)


@app.command()
def fetch(
    set_id: str,
    refresh: bool = typer.Option(False, help="Re-download the checklist even if cached"),
):
    """Cache a set's checklist and card images under data/sets/<set_id>."""
    s = get_settings()
    with Progress(console=console) as progress:
        task = progress.add_task(f"Fetching {set_id}", total=None)

        def tick(_card):
            progress.advance(task)

        result = service.fetch_set(
            _client(), s.sets_dir, set_id, refresh=refresh,
            workers=s.download_workers, on_progress=tick,
        )
    r = result.images
    console.print(
        f"[bold]{result.card_set.name}[/] ({result.card_set.id}): {len(result.cards)} cards — "
        f"{len(r.downloaded)} downloaded, {len(r.skipped)} already cached, {len(r.failed)} failed"
    )
    for cid, err in r.failed.items():
        console.print(f"  [red]failed[/] {cid}: {err}")
    if r.failed:
        raise typer.Exit(1)


@app.command()
def embed(set_id: str):
    """Build the CLIP embedding index for a cached set (GPU if available)."""
    s = get_settings()
    t0 = time.perf_counter()
    embedder = _embedder()
    console.print(f"Model {embedder.model_tag} on [bold]{embedder.device}[/]")
    index = service.build_index(s.sets_dir, set_id, embedder)
    console.print(
        f"Indexed {len(index)} cards (dim {index.dim}) in {time.perf_counter() - t0:.1f}s → "
        f"{SetCache(s.sets_dir, set_id).index_path(embedder.model_tag)}"
    )


@app.command()
def match(set_id: str, image: Path, top_k: int = 5):
    """Sanity check: show the top-k cards in a set matching an image."""
    s = get_settings()
    embedder = _embedder()
    index = service.load_index(s.sets_dir, set_id, embedder.model_tag)
    cards = {c.id: c for c in SetCache(s.sets_dir, set_id).load_cards()}
    with Image.open(image) as im:
        query = embedder.embed_images([im])
    table = Table("rank", "card", "number", "rarity", "score")
    for rank, m in enumerate(index.search(query, top_k)[0], 1):
        c = cards[m.card_id]
        table.add_row(str(rank), c.name, c.number, c.rarity or "", f"{m.score:.3f}")
    console.print(table)
