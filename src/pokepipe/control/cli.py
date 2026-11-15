"""`pokepipe ui ...` — the control room (upload from a phone, follow every video)."""

from __future__ import annotations

import threading

import typer
from rich.console import Console

from pokepipe.config import Settings, get_settings

app = typer.Typer(help="Control room: send videos from a phone and follow the pipeline.")
console = Console()


def _access(s: Settings):
    from pokepipe.control.access import Access

    return Access(s.data_dir / "control" / "access.json")


def build_config(s: Settings, port: int, lan: bool, live=None, review_url: str | None = None):
    from pokepipe.cards.service import DECOY_SET
    from pokepipe.control.app import ControlConfig
    from pokepipe.control.dashboard import set_names
    from pokepipe.control.uploads import UploadStore, ffprobe_check
    from pokepipe.watch.intake import JobStore

    uploads = UploadStore(
        s.data_dir / "uploads", s.inbox_dir,
        valid_sets=lambda: set(set_names(s.sets_dir)) - {DECOY_SET},
        probe=ffprobe_check,
    )
    return ControlConfig(
        jobs=JobStore(s.videos_dir), uploads=uploads, access=_access(s), inbox=s.inbox_dir,
        sets_root=s.sets_dir, decoy_set=DECOY_SET, default_set=s.watch_default_set,
        review_url=review_url, port=port, lan=lan, live=live,
    )


def _server(cfg, host: str, port: int):
    import uvicorn

    from pokepipe.control.app import create_app

    return uvicorn.Server(uvicorn.Config(create_app(cfg), host=host, port=port,
                                         log_level="warning"))


def describe(cfg, port: int) -> list[str]:
    from pokepipe.control.app import lan_addresses

    lines = [f"control room: [bold]http://127.0.0.1:{port}/[/]"]
    if cfg.lan:
        for addr in lan_addresses():
            lines.append(f"  phones on this Wi-Fi: [bold]http://{addr}:{port}/[/]  "
                         f"code [bold]{cfg.access.code}[/]")
        lines.append("  [dim]first time: allow Python on private networks when Windows "
                     "Firewall asks[/]")
    return lines


def start_in_thread(cfg, port: int) -> None:
    """Run the control room next to the watcher (used by `watch run`)."""
    server = _server(cfg, "0.0.0.0" if cfg.lan else "127.0.0.1", port)
    threading.Thread(target=server.run, daemon=True, name="control-room").start()


@app.command()
def serve(
    port: int = typer.Option(8800, help="Port for the control room"),
    lan: bool = typer.Option(False, "--lan", help="Let phones on the home network in "
                                                  "(with the access code)"),
):
    """Serve the control room on its own (the watcher can run in another window)."""
    s = get_settings()
    cfg = build_config(s, port, lan)
    for line in describe(cfg, port):
        console.print(line)
    console.print("[dim]watcher status shows only when the control room runs inside "
                  "`pokepipe watch run`[/]")
    _server(cfg, "0.0.0.0" if lan else "127.0.0.1", port).run()


@app.command()
def code(new: bool = typer.Option(False, "--new", help="Change the code and sign every "
                                                       "phone out")):
    """Show the access code phones sign in with."""
    access = _access(get_settings())
    console.print(f"access code: [bold]{access.reset() if new else access.code}[/]")
