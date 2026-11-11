"""`pokepipe render ...` — stage 6 commands."""

from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.progress import Progress

from pokepipe.config import PROJECT_ROOT, get_settings
from pokepipe.narration.service import NarrationError, VideoFiles
from pokepipe.render.ffmpeg import RenderError
from pokepipe.render.service import RenderInputError, render_post

app = typer.Typer(help="Stage 6: render each post as a vertical 1080x1920 video.")
console = Console()

INTRO = PROJECT_ROOT / "config" / "intro.mp4"


@app.command()
def post(
    slug: str,
    post: list[int] = typer.Option(None, "--post", "-p", help="Post number(s); default all"),
    keep_temp: bool = typer.Option(False, help="Keep the per-segment clips"),
    narration: bool = typer.Option(
        None,
        "--narration/--no-narration",
        help="Voiceover + captions, or the edit alone with source audio (default: NARRATION)",
    ),
    track: bool = typer.Option(
        None, "--track/--no-track", help="Follow each card through its zoom (GPU)"
    ),
):
    """Render posts to data/videos/<slug>/posts/<n>/post.mp4."""
    settings = get_settings()
    if narration is None:
        narration = settings.narration
    if track is None:
        track = settings.render_track
    files = VideoFiles(settings.data_dir / "videos" / slug)
    try:
        indexes = post or [p.index for p in files.load_plan().posts]
    except NarrationError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    intro = INTRO if INTRO.exists() else None
    for index in indexes:
        t0 = time.perf_counter()
        with Progress(console=console) as bar:
            task = bar.add_task(f"Post {index}", total=None)
            try:
                result = render_post(
                    files,
                    index,
                    intro=intro,
                    keep_temp=keep_temp,
                    narration=narration,
                    track=track,
                    on_progress=lambda done, total, t=task: bar.update(
                        t, completed=done, total=total
                    ),
                )
            except (RenderInputError, RenderError, NarrationError, ValueError) as exc:
                console.print(f"[red]Post {index}: {exc}[/]")
                raise typer.Exit(1) from exc
        e = result.edit
        console.print(
            f"Post {index}: {e.duration:.1f}s, {len(e.segments)} shots, reveal speed "
            f"{e.reveal_speed:.2f}x, rendered in {time.perf_counter() - t0:.0f}s → {result.path}"
        )
        for note in e.notes:
            console.print(f"  [yellow]note:[/] {note}")
