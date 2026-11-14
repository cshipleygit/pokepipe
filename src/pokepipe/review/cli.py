"""`pokepipe review ...` — stage 3 commands."""

from __future__ import annotations

import threading
import webbrowser

import typer
from rich.console import Console
from rich.table import Table

from pokepipe.config import get_settings
from pokepipe.review.store import ReviewStore

app = typer.Typer(help="Stage 3: confirm or fix detected pulls in a local web page.")
console = Console()


def _store() -> ReviewStore:
    s = get_settings()
    return ReviewStore(s.data_dir / "videos", s.sets_dir)


@app.command()
def serve(
    port: int = typer.Option(8765, help="Port on 127.0.0.1"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the page"),
):
    """Start the review page at http://127.0.0.1:PORT (local machine only)."""
    import uvicorn

    from pokepipe.review.app import create_app

    url = f"http://127.0.0.1:{port}/"
    console.print(f"Review page: [bold]{url}[/]  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    # Bound to loopback on purpose: the API can read the source video and write review
    # state, so it must not be reachable from the network.
    app_ = create_app(_store(), get_settings().default_post_seconds)
    uvicorn.run(app_, host="127.0.0.1", port=port, log_level="warning")


@app.command()
def prefill(slug: str):
    """Apply detection's reverse-holo guesses to pulls you haven't reviewed yet."""
    n = _store().prefill_reverse(slug)
    console.print(f"Updated reverse holo on {n} pending pulls")


@app.command()
def status():
    """Show review progress for every processed video."""
    store = _store()
    table = Table("video", "set", "pulls", "pending", "kept", "status", "auto")
    for slug in store.slugs():
        state = store.load(slug)
        pending = sum(p.status == "pending" for p in state.pulls)
        a = state.auto
        auto_col = "" if a is None else (
            f"{a.mode}: {a.confirmed} confirmed"
            + ("" if not a.exceptions else f", {len(a.exceptions)} exception(s)")
            + (f", {unchecked} unchecked" if (unchecked := sum(
                "UNCHECKED" in p.auto_note for p in state.pulls)) else ""))
        table.add_row(slug, state.set_id, str(len(state.pulls)), str(pending),
                      str(len(state.kept())), "finished" if state.completed else "in review",
                      auto_col)
    console.print(table)


@app.command("auto")
def auto_cmd(
    slug: str,
    mode: str = typer.Option(None, help="shadow | exceptions | off (default: REVIEW_MODE)"),
):
    """Auto-review a video that has no review yet (the watcher does this itself)."""
    from pokepipe.review import auto

    s = get_settings()
    mode = mode or s.review_mode
    res = auto.run(_store(), slug, mode, s.auto_policy())
    if res is None:
        console.print("Nothing done: mode is 'always', or a review already exists")
        return
    console.print(f"[bold]{mode}[/]: confirmed {res.confirmed} pulls, "
                  f"{len(res.exceptions)} exception(s)")
    for e in res.exceptions:
        console.print(f"  • {e}")


@app.command()
def replay(
    slug: str = typer.Argument(None, help="a video with a finished human review"),
    all_: bool = typer.Option(False, "--all", help="every finished human review"),
    min_confidence: float = typer.Option(None, help="try another confidence bar"),
    min_lead: float = typer.Option(None, help="try another near-tie lead"),
    no_calibration: bool = typer.Option(False, help="ignore review_calibration.json"),
):
    """Score the auto-reviewer against finished human reviews: what it would have
    confirmed, what it would have asked you, and every silent error."""
    from dataclasses import replace

    from pokepipe.review.replay import replay as run_replay

    s = get_settings()
    policy = s.auto_policy()
    if min_confidence is not None:
        policy = replace(policy, min_confidence=min_confidence)
    if min_lead is not None:
        policy = replace(policy, min_lead=min_lead)
    if no_calibration:
        policy = replace(policy, calibration=None)
    store = _store()
    if not all_ and not slug:
        raise typer.BadParameter("give a video slug or --all")
    slugs = store.slugs() if all_ else [slug]
    table = Table("video", "auto-confirmed", "asked you", "asked but already right",
                  "clean packs", "silent errors")
    errors: list[str] = []
    skipped = []
    for sl in slugs:
        try:
            r = run_replay(store, sl, policy)
        except ValueError as exc:
            skipped.append(str(exc))
            continue
        table.add_row(sl, str(r.auto_confirmed), str(r.escalated), str(r.escalated_but_right),
                      f"{r.clean_packs}/{r.packs}",
                      f"[red]{len(r.silent_errors)}[/]" if r.silent_errors else "[green]0[/]")
        errors += [f"{sl}: {e}" for e in r.silent_errors]
    console.print(table)
    for e in errors:
        console.print(f"[red]silent error[/] {e}")
    for msg in skipped:
        console.print(f"[dim]skipped {msg}[/]")
    if errors:
        raise typer.Exit(1)


@app.command()
def fit(
    save: bool = typer.Option(False, "--save", help="write review_calibration.json"),
):
    """Learn detection confidence from finished human reviews; show the trade-off."""
    from pokepipe.review import calibrate

    s = get_settings()
    store = _store()
    data = []
    for sl in store.slugs():
        data += calibrate.examples(store.detection(sl), store.load(sl))
    wrong = [e for e in data if not e.right]
    console.print(f"{len(data)} labelled pulls from finished reviews, {len(wrong)} wrong")
    for e in sorted(wrong, key=lambda e: -e.confidence):
        console.print(f"  [red]wrong[/] {e.slug}: {e.card} (confidence {e.confidence:.2f}) "
                      f"— {e.why}")
    table = Table("confidence ≥", "wrong pulls let through", "right pulls sent to you")
    for cut, through, asked in calibrate.threshold_table(
            data, [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]):
        table.add_row(f"{cut:.2f}", str(through), str(asked))
    console.print(table)
    try:
        cal = calibrate.fit(data)
    except ValueError as exc:
        console.print(f"[yellow]No calibrated model:[/] {exc}")
        return
    asked = 0
    for sl in store.slugs():
        det = store.detection(sl)
        state = store.load(sl)
        if not state.completed:
            continue
        asked += sum(cal.probability(det.pulls[p.detected_index]) < cal.threshold
                     for p in state.pulls if p.detected_index is not None
                     and p.decided_by != "auto" and p.status == "confirmed" and not p.corrected)
    console.print(f"Calibrated: threshold {cal.threshold:.3f} lets 0 known wrong pulls "
                  f"through and sends {asked} right ones to you")
    if save:
        cal.save(s.calibration_path)
        console.print(f"Saved → {s.calibration_path} (used by auto-review from now on)")


@app.command("export-testset")
def export_testset(
    slug: str = typer.Argument(None, help="a video with a finished human review"),
    all_: bool = typer.Option(False, "--all", help="every finished human review"),
    dest: str = typer.Option("tests/fixtures", help="where the test set lives"),
):
    """Freeze finished human reviews (JSON only, no footage) into the repo's test set,
    so every test run replays the auto-reviewer against your real decisions."""
    from pathlib import Path

    from pokepipe.config import PROJECT_ROOT
    from pokepipe.review.auto import AutoPolicy
    from pokepipe.review.testset import export

    if not all_ and not slug:
        raise typer.BadParameter("give a video slug or --all")
    store = _store()
    root = Path(dest) if Path(dest).is_absolute() else PROJECT_ROOT / dest
    for sl in store.slugs() if all_ else [slug]:
        try:
            n = export(store, sl, root, AutoPolicy())
        except ValueError as exc:
            console.print(f"[dim]skipped {exc}[/]")
            continue
        console.print(f"{sl} → {root / 'reviews' / sl} (baseline: {n} silent errors)")
