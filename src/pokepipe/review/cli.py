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
    uvicorn.run(create_app(_store()), host="127.0.0.1", port=port, log_level="warning")


@app.command()
def status():
    """Show review progress for every processed video."""
    store = _store()
    table = Table("video", "set", "pulls", "pending", "kept", "status")
    for slug in store.slugs():
        state = store.load(slug)
        pending = sum(p.status == "pending" for p in state.pulls)
        table.add_row(slug, state.set_id, str(len(state.pulls)), str(pending),
                      str(len(state.kept())), "finished" if state.completed else "in review")
    console.print(table)
