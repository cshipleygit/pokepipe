"""`pokepipe collection ...` — stage 4 commands."""

from __future__ import annotations

from collections import Counter

import typer
from rich.console import Console
from rich.table import Table

from pokepipe.collection import service
from pokepipe.collection.models import Progress
from pokepipe.config import get_settings
from pokepipe.review.store import ReviewStore

app = typer.Typer(help="Stage 4: master-set progress, updated from reviewed videos.")
console = Console()

NameOpt = typer.Option("main", "--name", "-n", help="Collection name (use e.g. 'test' for trials)")


def _stores():
    s = get_settings()
    return (
        service.CollectionStore(s.data_dir / "collection"),
        ReviewStore(s.data_dir / "videos", s.sets_dir),
        s.sets_dir,
    )


def _fmt(p: Progress) -> str:
    return (f"master {p.master_owned}/{p.master_total} ({p.master_pct:.1f}%) · "
            f"cards {p.full_owned}/{p.full_total} · base {p.base_owned}/{p.base_total}")


def _run(fn):
    try:
        return fn()
    except service.CollectionError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc


@app.command()
def apply(slug: str, name: str = NameOpt):
    """Add a finished review's pulls to a collection (re-running replaces them)."""
    store, reviews, sets_root = _stores()
    report = _run(lambda: service.apply_video(store, reviews, sets_root, slug, name))

    table = Table("pack", "new", "master after")
    for p in report.packs:
        a = p.progress_after
        table.add_row(str(p.index), f"+{p.new_slots}", f"{a.master_owned}/{a.master_total}")
    console.print(table)
    hits = [p for p in report.pulls if p.new and p.rarity not in ("Common", "Uncommon")]
    if hits:
        console.print("New rares & up: " + ", ".join(
            f"{p.name} ({p.rarity}{', reverse' if p.variant == 'reverse' else ''})" for p in hits))
    for w in report.warnings:
        console.print(f"[yellow]warning:[/] {w}")
    console.print(f"[bold]{report.set_name}[/] ({name}): +{report.new_count} new")
    console.print(f"  before: {_fmt(report.before)}")
    console.print(f"  after:  {_fmt(report.after)}")


@app.command()
def status(set_id: str, name: str = NameOpt):
    """Master-set progress for a set, broken down by rarity."""
    store, _, sets_root = _stores()
    master = _run(lambda: service.load_master(sets_root, set_id))
    coll = store.load(name, set_id)
    counts = coll.counts()
    console.print(f"[bold]{master.card_set.name}[/] ({name}): {_fmt(master.progress(counts))}")

    owned: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for card in master.cards:
        for v in card.variant_list():
            key = f"{card.rarity or '?'}{' (reverse)' if v == 'reverse' else ''}"
            total[key] += 1
            owned[key] += counts[f"{card.id}/{v}"] > 0
    table = Table("rarity / printing", "owned", "total")
    for key in total:
        table.add_row(key, str(owned[key]), str(total[key]))
    console.print(table)
    sources = ", ".join(f"{k} ({len(c.slots)})" for k, c in coll.contributions.items()) or "none"
    console.print(f"Sources: {sources}")


@app.command()
def missing(
    set_id: str,
    name: str = NameOpt,
    variant: str = typer.Option(None, help="Only this printing: normal, holo, reverse"),
):
    """List master-set slots you don't own yet."""
    store, _, sets_root = _stores()
    master = _run(lambda: service.load_master(sets_root, set_id))
    rows = master.missing(store.load(name, set_id).counts(), variant)
    table = Table("no.", "card", "rarity", "printing")
    for card, v in rows:
        table.add_row(card.number, card.name, card.rarity or "", v)
    console.print(table)
    console.print(f"{len(rows)} missing")


@app.command()
def add(
    set_id: str,
    card_id: str,
    variant: str = typer.Option(None, help="normal, holo or reverse (default: base printing)"),
    count: int = typer.Option(1, help="Copies to add; negative removes manual copies"),
    name: str = NameOpt,
):
    """Record cards you own from outside the videos (e.g. your existing binder)."""
    store, _, sets_root = _stores()
    s = _run(lambda: service.adjust_manual(store, sets_root, set_id, card_id, variant, count, name))
    console.print(f"{'Added' if count >= 0 else 'Removed'} {abs(count)} × {s}")


@app.command()
def remove(slug: str, name: str = NameOpt):
    """Undo a video's contribution to a collection."""
    store, reviews, _ = _stores()
    set_id = reviews.load(slug).set_id
    if service.remove_video(store, set_id, slug, name):
        console.print(f"Removed {slug} from {name}/{set_id}")
    else:
        console.print(f"{slug} was not applied to {name}/{set_id}")


@app.command()
def value(slug: str):
    """Price every pull of a video (exact printing) → data/videos/<slug>/values.json.
    Prices come from data/prices/<set>.csv, then pokemontcg.io's TCGplayer market."""
    from pokepipe.cards.prices import PriceBook
    from pokepipe.collection.values import video_values
    from pokepipe.narration.service import NarrationError, VideoFiles

    s = get_settings()
    files = VideoFiles(s.videos_dir / slug)
    try:
        report = files.load_report()
    except NarrationError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    book = PriceBook.load(s.sets_dir, report.set_id, s.prices_dir)
    values = video_values(report, book)
    (files.root / "values.json").write_text(values.model_dump_json(indent=2), encoding="utf-8")
    table = Table("pull", "card", "printing", "price", "source")
    for x in sorted((x for x in values.pulls if x.usd is not None), key=lambda x: -x.usd):
        table.add_row(x.pull_id, x.name, x.variant, f"${x.usd:,.2f}", f"{x.source} {x.as_of or ''}")
    console.print(table)
    console.print(f"Total ${values.total_usd:,.2f} from {values.priced} priced pulls "
                  f"({values.unpriced} unpriced) → {files.root / 'values.json'}")
