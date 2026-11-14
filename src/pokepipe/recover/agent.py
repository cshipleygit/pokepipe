"""The recovery agent: a LangGraph tool loop over the evidence tools.

    START → agent ──tool_use──► tools ──► agent ... ──submit_findings──► END

`agent` makes one Claude call. `tools` runs every tool the model asked for and returns
all the results in a single message. The loop ends when the model calls
`submit_findings`, ends its turn without one, refuses, or hits the turn limit, which
is a hard edge in the graph rather than a request in the prompt. The budget is checked
before every call.

Claude is called through the official `anthropic` SDK, as in narration. The model only
ever sees scores, card names and times.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from pokepipe.recover.evidence import Evidence, ToolError
from pokepipe.recover.models import Suggestion, ToolCall

FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = """\
You recover Pokémon cards that automatic detection missed in a pack-opening video.

How detection works: each frame is matched against the set's card scans with CLIP.
A card counts when its frames lead the runner-up by 0.03+ and peak at 0.72+. Cards are
dropped when:
- two prints share the art (an Uncommon and its Illustration Rare), so the lead is ~0;
- frames are borderline (blur, glare, a thumb over the card);
- a second copy of a card already in the pack was merged as a "repeat showing" of
  the first (a reverse-holo slot can hold a copy of a card already in that pack).

Pack layout (unshuffled modern packs, in reveal order): slots 1-4 Common, 5-7
Uncommon, 8-9 reverse-holo slots (Common/Uncommon/Rare; slot 9 may instead be a special
rarity such as an Illustration Rare), 10 the rare slot. A Common/Uncommon/Rare in slot
8 or 9 is a reverse holo. The basic Energy and code card are not pulls.

Work pack by pack. Ignore groups marked probably_not_a_pack. For each short pack, read
its overview, rescan the stretch where the missing slot should be (look at the gaps),
and test candidates with check_fit before suggesting them (all of a pack's additions
in one call). Use the layout to settle near-ties. One stretch can hold two cards held
up together, so read past the top candidate. Give each suggested card the time it was
first the top match (first_top): that keeps reveal order, which sets the slot (8 vs 9)
and the order the edit and script follow. Never give two cards the same time.
A stretch already_detected_as a card is that card, unless it is a repeat inside the
pack and the layout needs a second copy.

Only suggest a card when the scan shows it and it fits the pack. Leave a pack short
rather than guess: a person reviews every suggestion, and a wrong one costs them more
than a gap. When you are done, call submit_findings exactly once.
"""


def _schema(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props,
            "required": list(props) if required is None else required,
            "additionalProperties": False}


TOOLS: list[dict] = [
    {"name": "short_packs", "strict": True,
     "description": "Packs with fewer cards than expected. Start here.",
     "input_schema": _schema({})},
    {"name": "pack_overview", "strict": True,
     "description": "One pack's detected cards in reveal order, with slots, near-tie "
                    "leads, in-pack repeat showings, gaps between reveals, the slots that "
                    "could be missing, and a default scan window.",
     "input_schema": _schema({"pack": {"type": "integer"}})},
    {"name": "scan_window", "strict": True,
     "description": "Re-match a stretch of the video with loose thresholds (GPU, a few "
                    "seconds). Returns card-like stretches with their top candidates, "
                    "scores and leads. null start/end = the pack's default window. "
                    "Limited number of scans per video.",
     "input_schema": _schema({
         "pack": {"type": "integer"},
         "start": {"type": ["number", "null"]},
         "end": {"type": ["number", "null"]},
         "fps": {"type": "number", "description": "2-12; 8 is a good default"},
     })},
    {"name": "check_fit", "strict": True,
     "description": "Would the pack fit its slot order with these cards added together? "
                    "Test every card you plan to add to a pack in one call. Returns each "
                    "card's slot, whether it would be a reverse holo, and duplicates.",
     "input_schema": _schema({
         "pack": {"type": "integer"},
         "add": {"type": "array", "items": _schema({"card_id": {"type": "string"},
                                                    "time": {"type": "number"}})},
     })},
    {"name": "find_cards", "strict": True,
     "description": "Look up cards in the set by name, number or id.",
     "input_schema": _schema({"query": {"type": "string"}})},
    {"name": "submit_findings", "strict": True,
     "description": "Finish: the cards to suggest for review (may be empty) and notes on "
                    "packs left short.",
     "input_schema": _schema({
         "suggestions": {"type": "array", "items": _schema({
             "pack": {"type": "integer"},
             "card_id": {"type": "string"},
             "time": {"type": "number"},
             "reverse_holo": {"type": "boolean"},
             "confidence": {"type": "string", "enum": ["high", "medium"]},
             "evidence": {"type": "string"},
         })},
         "notes": {"type": "string"},
     })},
]


@dataclass(frozen=True)
class AgentSettings:
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_turns: int = 14
    usd_per_mtok_in: float = 4.0
    usd_per_mtok_out: float = 20.0
    estimate_per_turn: float = 0.04  # for the pre-call budget check


class RecoverState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    turns: int
    usd: float
    done: bool
    suggestions: list[Suggestion]
    notes: str
    error: str | None
    calls: Annotated[list[ToolCall], operator.add]


def _cost(usage, s: AgentSettings) -> float:
    cached_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cached_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    tokens_in = usage.input_tokens + 1.25 * cached_write + 0.1 * cached_read
    return (tokens_in * s.usd_per_mtok_in + usage.output_tokens * s.usd_per_mtok_out) / 1e6


def _run_tool(ev: Evidence, name: str, args: dict) -> dict:
    if name == "short_packs":
        return ev.short_packs()
    if name == "pack_overview":
        return ev.pack_overview(int(args["pack"]))
    if name == "scan_window":
        return ev.scan_window(int(args["pack"]), args.get("start"), args.get("end"),
                              float(args.get("fps") or 8))
    if name == "check_fit":
        return ev.check_fit(int(args["pack"]),
                            [(a["card_id"], float(a["time"])) for a in args["add"]])
    if name == "find_cards":
        return ev.find_cards(args["query"])
    raise ToolError(f"unknown tool {name!r}")


def build_agent(
    client,
    evidence: Evidence,
    settings: AgentSettings,
    budget_check: Callable[[float], None] = lambda estimate: None,
):
    """Compile the agent graph. `budget_check(estimate)` raises to stop before a call."""

    def agent(state: RecoverState) -> dict:
        try:
            budget_check(settings.estimate_per_turn)
        except Exception as exc:  # noqa: BLE001 — a budget stop ends recovery, not the video
            return {"done": True, "error": f"stopped before a model call: {exc}"}
        response = client.beta.messages.create(
            model=settings.model,
            max_tokens=16000,
            betas=[FALLBACK_BETA],
            fallbacks="default",  # server-side fallback if a safety classifier declines
            output_config={"effort": settings.effort},
            cache_control={"type": "ephemeral"},  # the growing history is re-read each turn
            system=SYSTEM,
            tools=TOOLS,
            messages=state["messages"],
        )
        update: dict[str, Any] = {
            # Pass the content back unchanged: thinking blocks must round-trip as-is.
            "messages": [{"role": "assistant", "content": response.content}],
            "turns": state.get("turns", 0) + 1,
            "usd": state.get("usd", 0.0) + _cost(response.usage, settings),
        }
        if response.stop_reason == "refusal":
            update.update(done=True, error="the model declined (refusal)")
        elif response.stop_reason == "max_tokens":
            update.update(done=True, error="the response was cut off (max_tokens)")
        elif not any(b.type == "tool_use" for b in response.content):
            update.update(done=True, error="ended without calling submit_findings")
        return update

    def tools(state: RecoverState) -> dict:
        last = state["messages"][-1]["content"]
        results, calls, update = [], [], {}
        for block in last:
            if block.type != "tool_use":
                continue
            args = block.input if isinstance(block.input, dict) else json.loads(block.input)
            ok, payload = True, None
            try:
                if block.name == "submit_findings":
                    update["suggestions"] = [Suggestion.model_validate(s)
                                             for s in args.get("suggestions", [])]
                    update["notes"] = args.get("notes", "")
                    update["done"] = True
                    payload = {"received": len(update["suggestions"])}
                else:
                    payload = _run_tool(evidence, block.name, args)
            except (ToolError, KeyError, TypeError, ValueError) as exc:
                ok, payload = False, {"error": str(exc)}
            calls.append(ToolCall(name=block.name, input=args, ok=ok))
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(payload), "is_error": not ok})
        content: list[dict] = results
        if not update.get("done") and state.get("turns", 0) >= settings.max_turns - 1:
            content = [*results, {"type": "text",
                                  "text": "Turn limit reached: call submit_findings now."}]
        return {"messages": [{"role": "user", "content": content}], "calls": calls, **update}

    def after_agent(state: RecoverState) -> str:
        return END if state.get("done") else "tools"

    def after_tools(state: RecoverState) -> str:
        if state.get("done"):
            return END
        if state.get("turns", 0) >= settings.max_turns:
            return END
        return "agent"

    g = StateGraph(RecoverState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", END])
    g.add_conditional_edges("tools", after_tools, ["agent", END])
    # Never checkpoint this loop, even inside the checkpointed pipeline graph: nested
    # graphs inherit the parent's checkpointer, and its state (SDK message objects)
    # comes back from SQLite as dicts. Rerunning recovery from the start is cheap.
    return g.compile(checkpointer=False)


def kickoff(slug: str, set_name: str) -> dict:
    return {"role": "user", "content": (
        f"Video {slug}, set {set_name}. Find cards detection missed in short packs and "
        "submit your findings.")}
