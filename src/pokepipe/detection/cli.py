"""`pokepipe detect ...` — stage 2 commands."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from pokepipe.cards import rarity
from pokepipe.cards import service as cards_service
from pokepipe.config import get_settings
from pokepipe.detection import service
from pokepipe.detection.frames import probe
from pokepipe.detection.models import PullsFile
from pokepipe.detection.packs import PackParams
from pokepipe.detection.segment import SegmentParams

app = typer.Typer(help="Stage 2: find card reveals in a video and match them.")
console = Console()


def _ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


def print_summary(result: PullsFile, verbose: bool = False) -> None:
    by_pack: dict[int, list] = {}
    for p in result.pulls:
        by_pack.setdefault(p.pack, []).append(p)

    table = Table("pack", "clip", "cards", "reverse", "best pull", "lowest conf")
    for pack in result.packs:
        pulls = by_pack.get(pack.index, [])
        low = min(pulls, key=lambda p: p.confidence)
        count = f"{pack.cards}/{pack.expected_cards}"
        if pack.layout:
            rev = ", ".join(p.name for p in pulls if p.reverse_guess) or "-"
        else:
            rev = f"[yellow]? {pack.layout_note}[/]" if pack.layout_note else ""
        table.add_row(
            str(pack.index), f"{_ts(pack.clip_start)}–{_ts(pack.clip_end)}",
            count if pack.complete else f"[bold red]{count}[/]",
            rev, _top_rarity(pulls), f"{low.name} {low.confidence:.2f}",
        )
    console.print(table)

    if verbose:
        detail = Table("pack", "slot", "time", "card", "no.", "rarity", "conf", "score")
        for p in result.pulls:
            slot = f"{p.slot}{' rev' if p.reverse_guess else ''}" if p.slot else ""
            detail.add_row(str(p.pack), slot, _ts(p.peak_time), p.name, p.number, p.rarity or "",
                           f"{p.confidence:.2f}", f"{p.peak_score:.3f}")
        console.print(detail)


def _reprocess(slug: str, regroup: bool, verbose: bool) -> None:
    """Re-run post-processing on an existing pulls.json without re-scanning the video."""
    from pokepipe.cards.cache import SetCache
    from pokepipe.detection.layout import layout_for
    from pokepipe.detection.reshows import merge_reshows

    s = get_settings()
    vdir = s.data_dir / "videos" / slug
    path = vdir / "pulls.json"
    if not path.exists():
        console.print(f"[red]no pulls.json for {slug!r}[/]")
        raise typer.Exit(1)
    if regroup and (vdir / "review.json").exists():
        # review.json refers to pulls by position; regrouping would scramble it.
        console.print("[red]this video has a review in progress — regrouping would invalidate "
                      "it. Delete review.json first if you want to start the review over.[/]")
        raise typer.Exit(1)
    result = PullsFile.model_validate_json(path.read_text(encoding="utf-8"))
    before = len(result.pulls), len(result.packs)
    for p in result.pulls:
        p.slot, p.reverse_guess = None, False
    if regroup:
        result.pulls, result.packs = merge_reshows(result.pulls, result.duration)
    series = SetCache(s.sets_dir, result.set_id).load_set().series
    chosen = service.infer_slots(result.pulls, result.packs, layout_for(series))
    result.pack_layout = chosen.name if chosen else None
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print_summary(result, verbose)
    if regroup:
        shows = sum(len(p.showings) for p in result.pulls)
        console.print(f"Merged repeats: {before[0]} detections in {before[1]} groups → "
                      f"{len(result.pulls)} pulls in {len(result.packs)} packs "
                      f"({shows} repeat showings kept as footage)")
    n = sum(p.reverse_guess for p in result.pulls)
    console.print(f"Layout {result.pack_layout or 'none'}: {n} reverse holos inferred → {path}")


@app.command()
def layout(
    slug: str,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="List every pull"),
):
    """Re-apply pack-layout inference (slots, reverse holos) to an existing pulls.json,
    without re-scanning the video."""
    _reprocess(slug, regroup=False, verbose=verbose)


@app.command()
def regroup(
    slug: str,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="List every pull"),
):
    """Merge repeat showings into single pulls (keeping their footage), regroup packs and
    re-infer reverse holos — on an existing pulls.json, without re-scanning."""
    _reprocess(slug, regroup=True, verbose=verbose)


def _top_rarity(pulls: list) -> str:
    best = max(pulls, key=lambda p: rarity.rank(p.rarity))
    return f"{best.name} ({best.rarity or '?'})"


@app.command()
def run(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    set_id: str = typer.Option(..., "--set", help="Set id the video opens, e.g. sv3pt5"),
    fps: float = typer.Option(4.0, help="Frames sampled per second"),
    out: Path = typer.Option(None, help="Output dir (default data/videos/<video-name>)"),
    min_peak: float = typer.Option(SegmentParams.min_peak, help="Min best-frame score for a pull"),
    cards_per_pack: int = typer.Option(PackParams.cards_per_pack, help="Expected cards per pack"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="List every pull"),
    dump_frames: bool = typer.Option(False, help="Also write per-frame matches (calibration)"),
):
    """Detect pulls in VIDEO, group them into packs, write pulls.json + thumbnails."""
    from pokepipe.cards.embeddings import OpenClipEmbedder

    s = get_settings()
    out = out or s.data_dir / "videos" / service.video_slug(video)
    embedder = OpenClipEmbedder(s.clip_model, s.clip_pretrained, fit=s.clip_fit)
    index = cards_service.load_index(s.sets_dir, set_id, embedder.model_tag)
    total = int(probe(video).duration * fps)
    frames = []

    t0 = time.perf_counter()
    columns = (
        TextColumn("{task.description}"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"), TimeRemainingColumn(),
    )
    with Progress(*columns, console=console) as progress:
        task = progress.add_task(f"Scanning on {embedder.device}", total=total)

        def tick(r):
            frames.append(r)
            progress.advance(task)

        result = service.detect_pulls(
            video, set_id, s.sets_dir, index, embedder, out, fps=fps,
            params=SegmentParams(min_peak=min_peak),
            pack_params=PackParams(cards_per_pack=cards_per_pack), on_frame=tick,
        )
    if dump_frames:
        rows = [{"t": r.time, "top": [(m.card_id, m.score) for m in r.top]} for r in frames]
        (out / "frames.json").write_text(json.dumps(rows), encoding="utf-8")

    print_summary(result, verbose)
    incomplete = sum(not p.complete for p in result.packs)
    console.print(
        f"{len(result.pulls)} pulls in {len(result.packs)} packs "
        f"({incomplete} need attention) in {time.perf_counter() - t0:.0f}s → {out / 'pulls.json'}"
    )
