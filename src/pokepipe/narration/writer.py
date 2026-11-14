"""Script writer ⇄ critic, as a LangGraph loop (evaluator-optimizer).

    START → write → critique ──pass──────────────► END
                       │ problems, rewrites left,
                       │ budget ok
                       └──────► write (with the problems as feedback) ...

`write` is one Claude call. `critique` runs the deterministic checks in `critic.py`
and, if configured, the style judge. A draft with problems goes back with every
problem listed and the previous draft attached, up to `max_rewrites` times. The budget
is checked before each rewrite. The loop keeps every attempt and returns the one with
the fewest problems (the latest on a tie), so a rewrite that makes things worse is
never what ships.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from pokepipe.narration.critic import Problem
from pokepipe.narration.models import ScriptDraft


@dataclass
class Attempt:
    draft: ScriptDraft
    response: object
    problems: list[Problem]
    notes: list[str] = field(default_factory=list)  # minor style remarks, not blocking
    cost: float = 0.0


class WriterState(TypedDict, total=False):
    attempts: Annotated[list[Attempt], operator.add]
    feedback: str
    cost: float
    stop: str


Write = Callable[[str], tuple[ScriptDraft, object, float]]  # feedback → draft, response, $
Critique = Callable[[ScriptDraft], tuple[list[Problem], list[str], float]]


def feedback_for(draft: ScriptDraft, problems: list[Problem]) -> str:
    rules = sorted({p.kind for p in problems})
    named = ", ".join("no-spoiler rule" if k == "spoiler" else f"{k} rules" for k in rules)
    lines = "\n".join(f"- {p}" for p in problems)
    return (
        f"\n\nYour previous draft broke the {named}:\n{lines}\n\n"
        f"<previous_draft>\n{draft.model_dump_json(indent=1)}\n</previous_draft>\n"
        "Rewrite the whole script fixing every problem above. Keep what already works."
    )


def build_writer(
    write: Write,
    critique: Critique,
    max_rewrites: int,
    budget_check: Callable[[float], None] = lambda spent: None,
):
    """`budget_check(spent_so_far)` raises to stop before a rewrite."""

    def write_node(state: WriterState) -> dict:
        draft, response, cost = write(state.get("feedback", ""))
        return {"attempts": [Attempt(draft, response, [], cost=cost)],
                "cost": state.get("cost", 0.0) + cost}

    def critique_node(state: WriterState) -> dict:
        last = state["attempts"][-1]
        problems, notes, cost = critique(last.draft)
        last.problems, last.notes, last.cost = problems, notes, last.cost + cost
        update: dict = {"cost": state.get("cost", 0.0) + cost}
        if problems and len(state["attempts"]) <= max_rewrites:
            try:
                budget_check(update["cost"])
            except Exception as exc:  # noqa: BLE001 — out of budget: ship the best so far
                update["stop"] = f"no rewrite: {exc}"
                return update
            update["feedback"] = feedback_for(last.draft, problems)
        return update

    def route(state: WriterState) -> str:
        last = state["attempts"][-1]
        if not last.problems or state.get("stop") or len(state["attempts"]) > max_rewrites:
            return END
        return "write"

    g = StateGraph(WriterState)
    g.add_node("write", write_node)
    g.add_node("critique", critique_node)
    g.add_edge(START, "write")
    g.add_edge("write", "critique")
    g.add_conditional_edges("critique", route, ["write", END])
    # Never checkpoint this loop. Run inside the checkpointed pipeline graph it would
    # inherit the parent's checkpointer; a run stopped mid-script then resumed from
    # SQLite with the drafts and SDK responses turned into dicts and crashed (found on
    # the first real end-to-end run). A post's script is simply rewritten instead.
    return g.compile(checkpointer=False)


def best(attempts: list[Attempt]) -> Attempt:
    """Fewest problems; the later attempt on a tie."""
    return min(reversed(attempts), key=lambda a: len(a.problems))


# --- on-disk record (posts/<n>/critique.json) ----------------------------------------


class AttemptRecord(BaseModel):
    hook_caption: str = ""
    lines: list[str] = Field(default_factory=list, description="'[anchor] text', as drafted")
    problems: list[str]
    kinds: list[str]
    style_notes: list[str] = Field(default_factory=list)
    words: int
    usd: float


class CritiqueRecord(BaseModel):
    post: int
    attempts: list[AttemptRecord]
    chosen: int = Field(description="0-based index of the attempt that was kept")
    stop: str | None = None

    @property
    def passed(self) -> bool:
        return not self.attempts[self.chosen].problems


def record(post: int, attempts: list[Attempt], chosen: Attempt,
           stop: str | None) -> CritiqueRecord:
    return CritiqueRecord(
        post=post,
        attempts=[AttemptRecord(
            hook_caption=a.draft.hook_caption,
            lines=[f"[{b.anchor}] {b.text}" for b in a.draft.beats],
            problems=[str(p) for p in a.problems], kinds=sorted({p.kind for p in a.problems}),
            style_notes=a.notes, words=sum(len(b.text.split()) for b in a.draft.beats),
            usd=round(a.cost, 5)) for a in attempts],
        chosen=attempts.index(chosen),
        stop=stop,
    )
