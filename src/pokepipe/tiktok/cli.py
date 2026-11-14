"""`pokepipe tiktok ...`: sign in once, then send rendered posts to TikTok as drafts."""

from __future__ import annotations

import typer
from rich.console import Console

from pokepipe.config import get_settings

app = typer.Typer(help="Send finished posts to your TikTok inbox as drafts.",
                  no_args_is_help=True)
console = Console()


def _client():
    from pokepipe.tiktok.client import TikTokClient, TokenStore

    s = get_settings()
    return TikTokClient(s.tiktok_client_key, s.tiktok_client_secret,
                        TokenStore(s.tiktok_token_path))


def _fail(exc: Exception):
    console.print(f"[red]{exc}[/]")
    raise typer.Exit(1)


@app.command()
def login():
    """Sign in to TikTok in the browser (once; Pokepipe stays signed in for a year)."""
    from pokepipe.tiktok.client import TikTokError

    s = get_settings()
    console.print(f"Opening TikTok sign-in (redirect {s.tiktok_redirect_uri}) ...")
    try:
        token = _client().login(s.tiktok_redirect_uri)
    except TikTokError as exc:
        _fail(exc)
    console.print(f"[green]Signed in[/] (scopes: {token.scope}). Saved to {s.tiktok_token_path}")


@app.command()
def draft(slug: str, post: list[int] = typer.Option(None, "--post", "-p",
                                                    help="only these posts (default: all)")):
    """Send a video's rendered posts to the TikTok inbox as drafts (each post once)."""
    from pokepipe.tiktok.client import TikTokError
    from pokepipe.tiktok.service import send_drafts

    s = get_settings()
    try:
        sent = send_drafts(s.videos_dir / slug, _client(), post or None, log=console.print)
    except TikTokError as exc:
        _fail(exc)
    if sent:
        console.print("In TikTok: open the inbox notification, paste the post's caption.txt, "
                      "and post.")


@app.command()
def status(slug: str):
    """What TikTok says about each post sent as a draft."""
    from pokepipe.narration.models import PostPlan
    from pokepipe.tiktok.client import TikTokError
    from pokepipe.tiktok.service import sent

    s = get_settings()
    root = s.videos_dir / slug
    plan = PostPlan.model_validate_json((root / "posts.json").read_text(encoding="utf-8"))
    client = _client()
    for p in plan.posts:
        record = sent(root / "posts" / str(p.index))
        if not record:
            console.print(f"post {p.index}: not sent")
            continue
        try:
            info = client.status(record["publish_id"])
        except TikTokError as exc:
            info = {"error": str(exc)}
        console.print(f"post {p.index}: sent {record['sent_at']} → {info}")
