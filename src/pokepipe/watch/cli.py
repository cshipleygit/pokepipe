"""`pokepipe watch ...` — stage 7 commands."""

from __future__ import annotations

import threading
import webbrowser
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table

from pokepipe.config import get_settings
from pokepipe.watch.intake import Inbox, JobStore
from pokepipe.watch.models import Status
from pokepipe.watch.runner import Runner

app = typer.Typer(help="Stage 7: drop videos in the inbox; everything runs except review.")
console = Console()


def _log(message: str) -> None:
    console.print(f"[dim]{datetime.now():%H:%M:%S}[/] {message}")


def _jobs() -> JobStore:
    return JobStore(get_settings().videos_dir)


def _start_review_server(port: int) -> str:
    import uvicorn

    from pokepipe.review.app import create_app
    from pokepipe.review.store import ReviewStore

    s = get_settings()
    app_ = create_app(ReviewStore(s.videos_dir, s.sets_dir), s.default_post_seconds)
    # Loopback only: the API can stream source videos and write review state.
    server = uvicorn.Server(uvicorn.Config(app_, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="review-server").start()
    return f"http://127.0.0.1:{port}/"


def _runner(engine: str, ops, jobs: JobStore):
    def on_event(job, message):
        _log(f"{job.slug}: {message}")

    if engine == "runner":
        return Runner(ops, jobs, on_event=on_event)
    if engine == "graph":
        from pokepipe.graph.pipeline import GraphRunner, open_checkpointer

        saver = open_checkpointer(get_settings().graph_checkpoints)
        return GraphRunner(ops, jobs, saver, on_event=on_event)
    console.print(f"[red]unknown engine {engine!r}; use 'runner' or 'graph'[/]")
    raise typer.Exit(2)


@app.command()
def run(
    once: bool = typer.Option(False, help="One pass, then exit (no loop, no review server)"),
    review: bool = typer.Option(True, "--review/--no-review", help="Serve the review page too"),
    port: int = typer.Option(8765, help="Review page port (127.0.0.1)"),
    open_browser: bool = typer.Option(False, "--open", help="Open the review page"),
    engine: str = typer.Option(
        "runner", help="'runner' (default state machine) or 'graph' (LangGraph, checkpointed, "
                       "with the missed-card recovery agent)"),
):
    """Watch the inbox and run every video through the pipeline."""
    from pokepipe.watch.pipeline import PipelineOps
    from pokepipe.watch.service import Watcher

    s = get_settings()
    s.inbox_dir.mkdir(parents=True, exist_ok=True)
    jobs = _jobs()
    inbox = Inbox(s.inbox_dir, jobs, s.watch_default_set, s.watch_settle_seconds)
    runner = _runner(engine, PipelineOps(s, log=_log), jobs)
    if engine == "graph":
        _log(f"engine: LangGraph (checkpoints in {s.graph_checkpoints})")
    from pokepipe.watch.notify import notify

    watcher = Watcher(inbox, runner, jobs, log=_log,
                      notify=notify if s.notify else (lambda title, message: None))

    if once:
        watcher.tick()
        return
    if review:
        url = _start_review_server(port)
        _log(f"review page: [bold]{url}[/]")
        if open_browser:
            webbrowser.open(url)
    where = f"{s.inbox_dir}" + (
        f" (default set {s.watch_default_set})" if s.watch_default_set else ""
    )
    _log(f"watching [bold]{where}[/] — drop videos in <inbox>/<set_id>/  (Ctrl+C to stop)")
    try:
        watcher.loop(s.watch_poll_seconds)
    except KeyboardInterrupt:
        _log("stopped; unfinished jobs resume next time")


@app.command()
def status():
    """Where every video is in the pipeline."""
    table = Table("video", "set", "status", "since", "note")
    for job in _jobs().all():
        last = job.history[-1] if job.history else None
        colour = {Status.DONE: "green", Status.FAILED: "red", Status.AWAITING_REVIEW: "yellow"}.get(
            job.status, "white"
        )
        note = job.error or (last.message if last else "")
        table.add_row(
            job.slug,
            job.set_id,
            f"[{colour}]{job.status.value}[/]",
            last.at if last else "",
            note[:80],
        )
    console.print(table)


@app.command()
def retry(
    slug: str,
    force_budget: bool = typer.Option(False, help="Allow this video to exceed the budget"),
):
    """Retry a failed video from the stage that failed (either engine)."""
    jobs = _jobs()
    try:
        job = jobs.load(slug)
        # Clearing the failure is the same for both engines: the runner resumes from
        # job.status, the graph from its checkpoint.
        Runner(None, jobs).retry(job, force_budget)  # type: ignore[arg-type]
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    _log(f"{slug} will resume at {job.status.value} on the next pass of `pokepipe watch run`")


@app.command()
def redo(
    slug: str,
    engine: str = typer.Option("runner", help="the engine the video ran on: runner | graph"),
    all_posts: bool = typer.Option(False, "--all-posts", help="re-make every post"),
):
    """After fixing a finished video's review: re-apply the collection and re-make only
    the posts the fix changed. Finish the review first, then run this; `watch run`
    picks it up."""
    from pokepipe.narration.service import VideoFiles
    from pokepipe.review.store import ReviewStore
    from pokepipe.watch import redo as redo_mod

    s = get_settings()
    jobs = _jobs()
    job = jobs.load(slug)
    if not job.status.terminal:
        console.print(f"[red]{slug} is {job.status.value}; redo is for done or failed videos[/]")
        raise typer.Exit(1)
    state = ReviewStore(s.videos_dir, s.sets_dir).load(slug)
    if not state.completed:
        console.print("[red]finish the review first (the Finish step on the review page)[/]")
        raise typer.Exit(1)
    files = VideoFiles(s.videos_dir / slug)
    plan = files.load_plan() if files.plan.exists() else None
    if all_posts or plan is None:
        posts_dir = files.root / "posts"
        stale = sorted(int(d.name) for d in posts_dir.glob("*")
                       if d.is_dir() and d.name.isdigit())
    else:
        stale = redo_mod.stale_posts(files, plan, state)
    redo_mod.clear(files, stale)
    if engine == "graph":
        from pokepipe.graph.pipeline import GraphRunner, open_checkpointer
        from pokepipe.watch.pipeline import PipelineOps

        runner = GraphRunner(PipelineOps(s, log=_log), jobs, open_checkpointer(s.graph_checkpoints))
        # pretend the review node just finished: the graph resumes at collect
        runner.graph.update_state({"configurable": {"thread_id": slug}}, {}, as_node="review")
    job.error, job.failed_at = None, None
    job.move(Status.COLLECTING, f"redo: posts {stale or 'none'} cleared")
    jobs.save(job)
    _log(f"{slug}: cleared post(s) {stale or 'none'}; collection and those posts will be "
         "redone on the next pass of `pokepipe watch run`")
