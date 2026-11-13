"""On-disk output of missed-card recovery (`data/videos/<slug>/recovered.json`).

Recovery never edits `pulls.json`. Its suggestions enter review as *pending* pulls, so
a person still confirms or rejects every one, and `pulls.json` + `recovered.json` +
`review.json` together say what detection found, what the agent added, and what was
true. That is what `pokepipe recover eval` scores.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Suggestion(BaseModel):
    pack: int
    card_id: str
    time: float = Field(description="seconds — where the card is visible")
    reverse_holo: bool = False
    confidence: Literal["high", "medium"]
    evidence: str = Field(description="why the agent thinks this card was there")


class ToolCall(BaseModel):
    name: str
    input: dict
    ok: bool = True


class RecoveredFile(BaseModel):
    video_slug: str
    model: str
    suggestions: list[Suggestion] = Field(default_factory=list)
    notes: str = ""
    turns: int = 0
    usd: float = 0.0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    error: str | None = Field(None, description="why recovery stopped early, if it did")
    created_at: str = Field(default_factory=_now)


FILENAME = "recovered.json"


def load_recovered(video_dir: Path) -> RecoveredFile | None:
    path = video_dir / FILENAME
    if not path.exists():
        return None
    return RecoveredFile.model_validate_json(path.read_text(encoding="utf-8"))
