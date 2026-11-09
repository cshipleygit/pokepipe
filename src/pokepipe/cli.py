"""Root CLI. Each pipeline stage registers its own sub-command group."""

import typer

from pokepipe.cards.cli import app as cards_app
from pokepipe.detection.cli import app as detect_app
from pokepipe.review.cli import app as review_app

app = typer.Typer(help="Pokémon pack-opening → TikTok shorts pipeline.", no_args_is_help=True)
app.add_typer(cards_app, name="cards")
app.add_typer(detect_app, name="detect")
app.add_typer(review_app, name="review")

if __name__ == "__main__":
    app()
