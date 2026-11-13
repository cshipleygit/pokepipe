"""`pokepipe recover ...` — run the missed-card agent on one video, and score it."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pokepipe.config import get_settings

app = typer.Typer(help="Missed-card recovery agent (LangGraph + Claude, local GPU scans).")
console = Console()


@app.command()
def run(
    slug: str,
    force: bool = typer.Option(False, help="Run again even if recovered.json exists"),
    force_budget: bool = typer.Option(False, help="Allow this video to exceed the budget"),
):
    """Look for cards detection missed in short packs → recovered.json."""
    from pokepipe.cards import service as cards
    from pokepipe.cards.embeddings import OpenClipEmbedder
    from pokepipe.detection.models import PullsFile
    from pokepipe.narration.script import anthropic_client
    from pokepipe.narration.service import Budget
    from pokepipe.recover.agent import AgentSettings
    from pokepipe.recover.scan import VideoScanner
    from pokepipe.recover.service import recover

    s = get_settings()
    root = s.videos_dir / slug
    if (root / "recovered.json").exists() and not force:
        console.print("recovered.json exists; pass --force to run again")
        raise typer.Exit(1)
    det = PullsFile.model_validate_json((root / "pulls.json").read_text(encoding="utf-8"))
    emb = OpenClipEmbedder(s.clip_model, s.clip_pretrained, fit=s.clip_fit)
    scanner = VideoScanner(Path(det.video), emb,
                           cards.load_index(s.sets_dir, det.set_id, emb.model_tag))
    out = recover(
        root, s.sets_dir, anthropic_client(s.anthropic_api_key, s.anthropic_workspace_id),
        scanner,
        AgentSettings(model=s.recover_model, effort=s.recover_effort,
                      max_turns=s.recover_max_turns),
        Budget(usd=s.budget_usd_per_video, force=force_budget),
    )
    table = Table("pack", "card", "time", "rev", "conf", "evidence")
    for x in out.suggestions:
        table.add_row(str(x.pack), x.card_id, f"{x.time:.1f}", "R" if x.reverse_holo else "",
                      x.confidence, x.evidence[:90])
    console.print(table)
    console.print(f"{out.turns} turns, {len(out.tool_calls)} tool calls, ${out.usd:.3f}")
    if out.notes:
        console.print(f"[dim]{out.notes}[/]")
    if out.error:
        console.print(f"[yellow]{out.error}[/]")


@app.command("eval")
def eval_(slug: str):
    """Score recovered.json against the cards a reviewer added by hand."""
    from pokepipe.recover.models import load_recovered
    from pokepipe.recover.service import evaluate
    from pokepipe.review.models import ReviewState

    s = get_settings()
    root = s.videos_dir / slug
    recovered = load_recovered(root)
    review_path = root / "review.json"
    if recovered is None or not review_path.exists():
        console.print("[red]needs both recovered.json and a review.json[/]")
        raise typer.Exit(1)
    review = ReviewState.model_validate_json(review_path.read_text(encoding="utf-8"))
    if not review.completed:
        console.print("[yellow]review not finished: ground truth may be incomplete[/]")
    r = evaluate(recovered, review)
    console.print(f"missed by detection (added by hand): {len(r.truth)}")
    console.print(f"recovered: {len(r.found)}  {r.found}")
    console.print(f"wrong suggestions: {len(r.wrong)}  {r.wrong}")
    console.print(f"still missed: {len(r.missed)}  {r.missed}")
    console.print(f"recall {r.recall:.0%}, precision {r.precision:.0%}, "
                  f"cost ${recovered.usd:.3f}")
