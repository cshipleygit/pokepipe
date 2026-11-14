"""The job state machine: move a video through the stages, pausing only for review.

Each status names the work still to do. `step()` does that work and advances, and
`run()` repeats until the job finishes, fails, or has to wait for the human review.
Every step is idempotent (the real ops skip work whose outputs already exist), so a
crash or restart mid-step is safe: the step simply runs again.

The heavy lifting is behind the `Ops` protocol, so the state machine is tested with
fakes instead of a GPU, API keys and ffmpeg.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pokepipe.watch.intake import JobStore
from pokepipe.watch.models import Job, Status


class Ops(Protocol):
    def ensure_set(self, job: Job) -> None: ...  # card data + index for the set
    def detect(self, job: Job) -> None: ...
    def auto_review(self, job: Job) -> None: ...  # decide what's clear (REVIEW_MODE)
    def review_done(self, job: Job) -> bool: ...
    def apply_collection(self, job: Job) -> None: ...
    def narrate(self, job: Job) -> None: ...  # posts (if not chosen in review), scripts, voice
    def render(self, job: Job) -> None: ...


class Runner:
    def __init__(
        self, ops: Ops, jobs: JobStore, on_event: Callable[[Job, str], None] | None = None
    ) -> None:
        self.ops = ops
        self.jobs = jobs
        self.on_event = on_event or (lambda job, msg: None)

    def _advance(self, job: Job, to: Status, message: str = "") -> None:
        job.move(to, message)
        self.jobs.save(job)
        self.on_event(job, message or to.value)

    def step(self, job: Job) -> bool:
        """Do the work for the current status. Returns False when the job is waiting
        (review not finished) or terminal."""
        status = job.status
        try:
            if status == Status.QUEUED:
                self.ops.ensure_set(job)
                self._advance(job, Status.DETECTING)
            elif status == Status.DETECTING:
                self.ops.detect(job)
                self.ops.auto_review(job)
                self._advance(job, Status.AWAITING_REVIEW, "detected — waiting for your review")
            elif status == Status.AWAITING_REVIEW:
                if not self.ops.review_done(job):
                    return False
                self._advance(job, Status.COLLECTING, "review finished")
            elif status == Status.COLLECTING:
                self.ops.apply_collection(job)
                self._advance(job, Status.NARRATING)
            elif status == Status.NARRATING:
                self.ops.narrate(job)
                self._advance(job, Status.RENDERING)
            elif status == Status.RENDERING:
                self.ops.render(job)
                self._advance(job, Status.DONE, "posts rendered")
            else:
                return False
        except Exception as exc:  # noqa: BLE001 — any stage failure is recorded, not fatal
            job.fail(status, f"{type(exc).__name__}: {exc}")
            self.jobs.save(job)
            self.on_event(job, f"failed during {status.value}: {exc}")
            return False
        return True

    def run(self, job: Job) -> Job:
        while not job.status.terminal and self.step(job):
            pass
        return job

    def retry(self, job: Job, force_budget: bool = False) -> Job:
        """Send a failed job back to the stage that failed."""
        if job.status != Status.FAILED or job.failed_at is None:
            raise ValueError(f"{job.slug} is {job.status.value}, not failed")
        job.error = None
        job.force_budget = job.force_budget or force_budget
        self._advance(job, job.failed_at, "retry")
        job.failed_at = None
        self.jobs.save(job)
        return job
