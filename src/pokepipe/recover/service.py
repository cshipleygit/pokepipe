"""Run recovery for one video, and score it against a finished review."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pokepipe.cards.cache import SetCache
from pokepipe.detection.layout import layout_for
from pokepipe.detection.models import PullsFile
from pokepipe.narration.models import CostEntry
from pokepipe.narration.service import Budget, VideoFiles
from pokepipe.recover.agent import AgentSettings, build_agent, kickoff
from pokepipe.recover.evidence import Evidence, Scanner
from pokepipe.recover.models import FILENAME, RecoveredFile, Suggestion
from pokepipe.review.models import ReviewState
from pokepipe.review.models import Status as ReviewStatus


def recover(
    video_dir: Path,
    sets_root: Path,
    client,
    scanner: Scanner | None,
    settings: AgentSettings | None = None,
    budget: Budget | None = None,
) -> RecoveredFile:
    """Run the agent on one video and write `recovered.json`.

    Never raises for an agent problem (API error, refusal, budget): recovery is help
    for the reviewer, so a failure is written down and the video carries on to review.
    """
    settings = settings or AgentSettings()
    budget = budget or Budget()
    det = PullsFile.model_validate_json((video_dir / "pulls.json").read_text(encoding="utf-8"))
    cache = SetCache(sets_root, det.set_id)
    cards = {c.id: c for c in cache.load_cards()}
    card_set = cache.load_set()
    evidence = Evidence(det, cards, layout_for(card_set.series), scanner)
    files = VideoFiles(video_dir)
    slug = video_dir.name

    out = RecoveredFile(video_slug=slug, model=settings.model)
    if not evidence.short_packs()["short_packs"]:
        out.notes = "every pack is complete; nothing to recover"
        files.save(video_dir / FILENAME, out)
        return out

    def check(estimate: float) -> None:
        budget.check(files.load_ledger(), estimate, "missed-card recovery")

    graph = build_agent(client, evidence, settings, check)
    try:
        state = graph.invoke({"messages": [kickoff(slug, card_set.name)]},
                             {"recursion_limit": 4 * settings.max_turns + 4})
    except Exception as exc:  # noqa: BLE001 — see docstring
        out.error = f"{type(exc).__name__}: {exc}"
        files.save(video_dir / FILENAME, out)
        return out

    out.suggestions = _valid(state.get("suggestions", []), det, cards)
    out.notes = state.get("notes", "")
    out.turns = state.get("turns", 0)
    out.usd = round(state.get("usd", 0.0), 4)
    out.tool_calls = state.get("calls", [])
    out.error = state.get("error") or (None if state.get("done") else "turn limit reached")
    if out.usd:
        ledger = files.load_ledger()
        ledger.entries.append(CostEntry(kind="recover", post=0, usd=out.usd,
                                        detail=f"{settings.model}, {out.turns} turns"))
        files.save(files.ledger, ledger)
    files.save(video_dir / FILENAME, out)
    return out


def _valid(suggestions: list[Suggestion], det: PullsFile, cards: dict) -> list[Suggestion]:
    """Drop anything the model invented: unknown cards or packs, or a time outside
    the video. (Review re-validates card ids too.)"""
    packs = {p.index for p in det.packs}
    return [s for s in suggestions
            if s.card_id in cards and s.pack in packs and 0 <= s.time <= det.duration]


# --- evaluation --------------------------------------------------------------------


@dataclass
class EvalResult:
    truth: list[tuple[int, str]]  # (pack, card_id) the reviewer had to add by hand
    found: list[tuple[int, str]] = field(default_factory=list)
    wrong: list[tuple[int, str]] = field(default_factory=list)
    missed: list[tuple[int, str]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return len(self.found) / len(self.truth) if self.truth else 1.0

    @property
    def precision(self) -> float:
        n = len(self.found) + len(self.wrong)
        return len(self.found) / n if n else 1.0


def evaluate(recovered: RecoveredFile, review: ReviewState) -> EvalResult:
    """Compare suggestions with the cards a person added by hand in a finished review.

    Hand-added pulls are exactly the ones detection missed, so they are the ground
    truth. Agent suggestions already in the review are excluded from it, so a review
    that started from suggestions doesn't grade itself.
    """
    truth = [(p.pack, p.card_id) for p in review.pulls
             if p.detected_index is None and p.status == ReviewStatus.CONFIRMED
             and not p.suggested]
    remaining = list(truth)
    result = EvalResult(truth=truth)
    for s in recovered.suggestions:
        key = (s.pack, s.card_id)
        if key in remaining:
            remaining.remove(key)
            result.found.append(key)
        else:
            result.wrong.append(key)
    result.missed = remaining
    return result
