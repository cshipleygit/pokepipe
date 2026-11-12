"""Stage 5 data: which packs go in which post, the script per post, and its voiceover.

Files per video (data/videos/<slug>/):
    posts.json                 PostPlan — the user's grouping of packs into posts
    posts/<n>/script.json      PostScript — Claude's script for post n
    posts/<n>/voice.mp3        ElevenLabs audio
    posts/<n>/voice.json       VoiceTiming — beat and word timings for render/captions
    costs.json                 CostLedger — every paid call, for the $/video budget
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- posts --------------------------------------------------------------------------------


class Post(BaseModel):
    index: int
    packs: list[int]
    seconds: float | None = Field(None, description="target length; default from pack count")


class PostPlan(BaseModel):
    video_slug: str
    posts: list[Post]


# --- script (this is also the structured-output schema Claude fills in) -------------------


class Beat(BaseModel):
    """One spoken line, tied to a moment so stage 6 can time it to the footage."""

    anchor: str = Field(
        description="'intro', 'outro', or a pull id like 'p9-9' from the list given"
    )
    text: str = Field(description="The spoken line. Plain words only, under 20 words.")


class ScriptDraft(BaseModel):
    hook_caption: str = Field(
        description="On-screen caption for the first seconds, max 8 words: a question or "
        "challenge to the viewer that names no card or rarity"
    )
    beats: list[Beat]
    hashtags: list[str]


class PostScript(ScriptDraft):
    post: int
    model: str
    words: int
    created_at: str = Field(default_factory=_now)

    @property
    def spoken_text(self) -> str:
        return " ".join(b.text.strip() for b in self.beats)


# --- voice ----------------------------------------------------------------------------------


class TimedBeat(BaseModel):
    anchor: str
    text: str
    start: float  # seconds into voice.mp3
    end: float


class TimedWord(BaseModel):
    text: str
    start: float
    end: float


class VoiceTiming(BaseModel):
    post: int
    voice_id: str
    model_id: str
    duration: float
    characters: int
    beats: list[TimedBeat]
    words: list[TimedWord]  # for word-by-word captions in stage 6
    cached: bool = False


# --- cost tracking --------------------------------------------------------------------------


class CostEntry(BaseModel):
    kind: str  # "script" | "voice"
    post: int
    usd: float
    detail: str
    at: str = Field(default_factory=_now)


class CostLedger(BaseModel):
    entries: list[CostEntry] = Field(default_factory=list)

    @property
    def total(self) -> float:
        return round(sum(e.usd for e in self.entries), 4)
