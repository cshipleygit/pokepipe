"""`pokepipe narrate ...` — stage 5 commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from pokepipe.config import get_settings
from pokepipe.narration import posts as posts_mod
from pokepipe.narration import service
from pokepipe.narration.script import ScriptError, ScriptSettings, anthropic_client
from pokepipe.narration.voice import ElevenLabsClient, VoiceCache, VoiceError, VoiceSettings

app = typer.Typer(
    help="Stage 5: split into posts, write scripts (Claude), voice them (ElevenLabs)."
)
console = Console()

PostsOpt = typer.Option(None, "--post", "-p", help="Only these post numbers (repeatable)")
ForceOpt = typer.Option(False, "--force", help="Ignore the per-video budget")


def _files(slug: str) -> service.VideoFiles:
    root = get_settings().data_dir / "videos" / slug
    if not (root / "pulls.json").exists():
        console.print(f"[red]no video {slug!r} in data/videos[/]")
        raise typer.Exit(1)
    return service.VideoFiles(root)


def _run(fn):
    try:
        return fn()
    except (service.NarrationError, posts_mod.PostSpecError, ScriptError, VoiceError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc


def _budget(force: bool) -> service.Budget:
    return service.Budget(usd=get_settings().budget_usd_per_video, force=force)


@app.command()
def plan(
    slug: str,
    posts: str = typer.Option(
        None, "--posts", help="Packs per post, e.g. '1-3, 9, 10+12'. Omit for a suggested grouping."
    ),
    seconds: float = typer.Option(
        None, "--seconds", help="Target length of each post (default: 12 s per pack, 15-60 s)"
    ),
):
    """Choose which packs go into each post (saved to posts.json)."""
    files = _files(slug)
    result = _run(lambda: service.plan_posts(files, slug, posts, seconds))
    report = files.load_report()
    table = Table("post", "packs", "cards", "best pull")
    for post in result.posts:
        pulls = [p for p in report.pulls if p.pack in post.packs]
        best = max(pulls, key=lambda p: _rank(p.rarity), default=None)
        table.add_row(
            str(post.index),
            posts_mod.to_spec([post.packs]),
            str(len(pulls)),
            f"{best.name} ({best.rarity})" if best else "-",
        )
    console.print(table)
    spec = posts_mod.to_spec([p.packs for p in result.posts])
    console.print(
        f"Saved {len(result.posts)} posts. Change with: "
        f'pokepipe narrate plan {slug} --posts "{spec}"'
    )


_ORDER = [
    "Common",
    "Uncommon",
    "Rare",
    "Double Rare",
    "Illustration Rare",
    "Ultra Rare",
    "Hyper Rare",
    "Special Illustration Rare",
]


def _rank(rarity: str | None) -> int:
    return _ORDER.index(rarity) if rarity in _ORDER else -1


@app.command()
def script(slug: str, post: list[int] = PostsOpt, force: bool = ForceOpt):
    """Write the voiceover script for each post with Claude."""
    import anthropic

    s = get_settings()
    style = s.style_guide.read_text(encoding="utf-8") if s.style_guide.exists() else ""
    client = anthropic_client(s.anthropic_api_key, s.anthropic_workspace_id)
    settings = ScriptSettings(model=s.script_model, effort=s.script_effort)
    try:
        runs = _run(
            lambda: service.write_scripts(
                _files(slug), client, style, settings, _budget(force), post or None
            )
        )
    except anthropic.BadRequestError as exc:
        if "workspace" in str(exc):
            console.print(
                "[red]This Anthropic key isn't tied to a workspace — set ANTHROPIC_WORKSPACE_ID "
                "in .env, or create a key inside a workspace.[/]"
            )
            raise typer.Exit(1) from exc
        raise
    except anthropic.AuthenticationError as exc:
        console.print(
            "[red]Anthropic API key missing or invalid — set ANTHROPIC_API_KEY in .env[/]"
        )
        raise typer.Exit(1) from exc
    for run in runs:
        s_ = run.script
        console.rule(f"Post {s_.post} — {s_.words} words — ${run.cost:.4f}")
        console.print(f"[bold]{s_.hook_caption}[/]  {' '.join(s_.hashtags)}")
        for beat in s_.beats:
            console.print(f"  [dim]{beat.anchor:>6}[/]  {beat.text}")
        for w in run.warnings:
            console.print(f"  [yellow]warning:[/] {w}")
    console.print(f"Total so far for this video: ${_files(slug).load_ledger().total:.4f}")


def _voice_settings() -> VoiceSettings:
    s = get_settings()
    if not s.elevenlabs_voice_id:
        console.print(
            "[red]Set ELEVENLABS_VOICE_ID in .env — list voices with `pokepipe narrate voices`[/]"
        )
        raise typer.Exit(1)
    return VoiceSettings(
        voice_id=s.elevenlabs_voice_id,
        model_id=s.elevenlabs_model_id,
        usd_per_1k_chars=s.elevenlabs_usd_per_1k_chars,
    )


@app.command()
def voice(slug: str, post: list[int] = PostsOpt, force: bool = ForceOpt):
    """Synthesise each post's script with ElevenLabs (cached; budget-checked)."""
    s = get_settings()
    settings = _voice_settings()
    client = _run(lambda: ElevenLabsClient(s.elevenlabs_api_key or ""))
    cache = VoiceCache(s.data_dir / "tts_cache")
    timings = _run(
        lambda: service.voice_posts(
            _files(slug), client, cache, settings, _budget(force), post or None
        )
    )
    table = Table("post", "duration", "chars", "lines", "source")
    for t in timings:
        table.add_row(
            str(t.post),
            f"{t.duration:.1f}s",
            str(t.characters),
            str(len(t.beats)),
            "cache" if t.cached else "ElevenLabs",
        )
    console.print(table)
    console.print(f"Total so far for this video: ${_files(slug).load_ledger().total:.4f}")


@app.command()
def voices():
    """List the voices on your ElevenLabs account (to pick ELEVENLABS_VOICE_ID)."""
    client = _run(lambda: ElevenLabsClient(get_settings().elevenlabs_api_key or ""))
    table = Table("voice id", "name", "category")
    for v in _run(client.voices):
        table.add_row(v.get("voice_id", ""), v.get("name", ""), v.get("category", ""))
    console.print(table)


@app.command()
def cost(slug: str):
    """Show what this video has cost so far in API calls."""
    ledger = _files(slug).load_ledger()
    table = Table("post", "kind", "usd", "detail", "when")
    for e in ledger.entries:
        table.add_row(str(e.post), e.kind, f"${e.usd:.4f}", e.detail, e.at)
    console.print(table)
    console.print(f"Total ${ledger.total:.4f} of ${get_settings().budget_usd_per_video:.2f} budget")
