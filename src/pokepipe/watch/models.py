"""Per-video job state for the watch folder (`data/videos/<slug>/job.json`).

The job file is the only state the watcher keeps, and every stage's real output lives
in its own files. So a job can be stopped, inspected, edited or retried at any point,
and restarting the watcher simply carries on from `status`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Status(StrEnum):
    QUEUED = "queued"  # just taken in from the inbox
    DETECTING = "detecting"
    AWAITING_REVIEW = "awaiting_review"  # the one human step
    COLLECTING = "collecting"
    NARRATING = "narrating"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (Status.DONE, Status.FAILED)


# The order work happens in; a failed job is retried from the stage that failed.
ORDER = [
    Status.QUEUED,
    Status.DETECTING,
    Status.AWAITING_REVIEW,
    Status.COLLECTING,
    Status.NARRATING,
    Status.RENDERING,
    Status.DONE,
]


class Event(BaseModel):
    at: str = Field(default_factory=_now)
    status: Status
    message: str = ""


class Job(BaseModel):
    slug: str
    source: str = Field(description="the video, moved here from the inbox")
    set_id: str
    status: Status = Status.QUEUED
    failed_at: Status | None = Field(None, description="stage to retry from after a failure")
    error: str | None = None
    force_budget: bool = False
    history: list[Event] = Field(default_factory=list)

    def move(self, status: Status, message: str = "") -> None:
        self.status = status
        self.history.append(Event(status=status, message=message))

    def fail(self, stage: Status, error: str) -> None:
        self.failed_at, self.error = stage, error
        self.move(Status.FAILED, error)
