"""`pokepipe detect ...` — stage 2 commands."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from pokepipe.cards import service as cards_service
from pokepipe.config import get_settings
from pokepipe.detection import service
from pokepipe.detection.frames import probe
from pokepipe.detection.segment import SegmentParams

app = typer.Typer(help="Stage 2: find card reveals in a video and match them.")
console = Console()


@app.command()
def run(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    set_id: str = typer.Option(..., "--set", help="Set id the video opens, e.g. sv3pt5"),
    fps: float = typer.Option(4.0, help="Frames sampled per second"),
    out: Path = typer.Option(None, help="Output dir (default data/videos/<video-name>)"),
    min_peak: float = typer.Option(SegmentParams.min_peak, help="Min best-frame score for a pull"),
    dump_frames: bool = typer.Option(False, help="Also write per-frame matches (calibration)"),
):
    """Detect pulls in VIDEO and write pulls.json + thumbnails for review."""
    from pokepipe.cards.embeddings import OpenClipEmbedder

    s = get_settings()
    out = out or s.data_dir / "videos" / service.video_slug(video)
    embedder = OpenClipEmbedder(s.clip_model, s.clip_pretrained)
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
            params=SegmentParams(min_peak=min_peak), on_frame=tick,
        )
    if dump_frames:
        rows = [{"t": r.time, "top": [(m.card_id, m.score) for m in r.top]} for r in frames]
        (out / "frames.json").write_text(json.dumps(rows), encoding="utf-8")

    table = Table("#", "time", "card", "no.", "rarity", "conf", "score")
    for i, p in enumerate(result.pulls, 1):
        m, sec = divmod(p.peak_time, 60)
        table.add_row(str(i), f"{int(m)}:{sec:04.1f}", p.name, p.number, p.rarity or "",
                      f"{p.confidence:.2f}", f"{p.peak_score:.3f}")
    console.print(table)
    console.print(
        f"{len(result.pulls)} pulls in {time.perf_counter() - t0:.0f}s → {out / 'pulls.json'}"
    )
