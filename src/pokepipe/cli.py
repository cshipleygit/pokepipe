"""Root CLI. Each pipeline stage registers its own sub-command group."""

import typer

from pokepipe.cards.cli import app as cards_app
from pokepipe.collection.cli import app as collection_app
from pokepipe.detection.cli import app as detect_app
from pokepipe.narration.cli import app as narrate_app
from pokepipe.recover.cli import app as recover_app
from pokepipe.render.cli import app as render_app
from pokepipe.review.cli import app as review_app
from pokepipe.watch.cli import app as watch_app

app = typer.Typer(help="Pokémon pack-opening → TikTok shorts pipeline.", no_args_is_help=True)
app.add_typer(cards_app, name="cards")
app.add_typer(detect_app, name="detect")
app.add_typer(recover_app, name="recover")
app.add_typer(review_app, name="review")
app.add_typer(collection_app, name="collection")
app.add_typer(narrate_app, name="narrate")
app.add_typer(render_app, name="render")
app.add_typer(watch_app, name="watch")



@app.command()
def preflight(
    set_id: str = typer.Option(None, "--set", help="the set you're about to open"),
    videos: int = typer.Option(3, help="how many videos you plan to record"),
    online: bool = typer.Option(True, "--online/--offline", help="check the API keys too"),
):
    """Check the GPU, ffmpeg, API keys, voice credits, card data and disk before a
    recording day."""
    from rich.console import Console
    from rich.table import Table

    from pokepipe.config import get_settings
    from pokepipe.preflight import Level, run_all

    colour = {Level.OK: "green", Level.WARN: "yellow", Level.FAIL: "red"}
    checks = run_all(get_settings(), set_id, videos, online)
    table = Table("check", "", "detail")
    for c in checks:
        table.add_row(c.name, f"[{colour[c.level]}]{c.level.value}[/]", c.detail)
    Console().print(table)
    if any(c.level == Level.FAIL for c in checks):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
