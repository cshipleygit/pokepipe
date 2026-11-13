"""The watch-folder pipeline as a LangGraph state graph.

    ensure_set → detect → recover → review ⏸ → collect → narrate → render

The same stages as `watch/runner.py`, with three differences:

- **Checkpointed.** A SQLite checkpointer (`data/graph/checkpoints.sqlite`, one thread
  per video slug) records each finished node. After a crash, a restart or a failed
  stage, `invoke(None)` carries on at the node that didn't finish.
- **Review is a real interrupt.** The `review` node calls `interrupt()` until the review
  is complete. The graph is paused, not polled through a state machine, and is resumed
  with `Command(resume=True)` once `review.json` says `completed`.
- **Recovery.** The `recover` node runs the missed-card agent (`pokepipe.recover`),
  whose suggestions are waiting in review when the reviewer opens it.

`job.json` stays as a read-only view of where each video is, so `watch status` and
`watch retry` work with either engine. The checkpoint is what decides what runs next.

Single-threaded like the old runner: detection, recovery and render all want the GPU.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from pokepipe.watch.intake import JobStore
from pokepipe.watch.models import Job, Status
from pokepipe.watch.runner import Ops


class GraphOps(Ops, Protocol):
    def recover(self, job: Job) -> None: ...  # missed-card agent → recovered.json


class PipelineState(TypedDict):
    slug: str


# node → the job status it reports while it runs (and the stage a failure is filed under)
STAGES: list[tuple[str, Status, str]] = [
    ("ensure_set", Status.QUEUED, ""),
    ("detect", Status.DETECTING, ""),
    ("recover", Status.DETECTING, "looking for missed cards"),
    ("review", Status.AWAITING_REVIEW, "detected — waiting for your review"),
    ("collect", Status.COLLECTING, "review finished"),
    ("narrate", Status.NARRATING, ""),
    ("render", Status.RENDERING, ""),
]


def open_checkpointer(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The review server runs in another thread but never touches the graph; this
    # connection is only used from the watcher thread.
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


class GraphRunner:
    """Drop-in for `watch.runner.Runner` (same `run` / `retry`), backed by the graph."""

    def __init__(
        self,
        ops: GraphOps,
        jobs: JobStore,
        checkpointer,
        on_event: Callable[[Job, str], None] | None = None,
    ) -> None:
        self.ops = ops
        self.jobs = jobs
        self.on_event = on_event or (lambda job, msg: None)
        self._job: Job | None = None  # the job being run (single-threaded)
        self.graph = self._build().compile(checkpointer=checkpointer)

    # --- graph -------------------------------------------------------------------------

    def _enter(self, status: Status, message: str) -> Job:
        job = self._job
        assert job is not None, "nodes only run inside GraphRunner.run"
        if job.status != status:
            job.move(status, message)
            self.jobs.save(job)
            self.on_event(job, message or status.value)
        return job

    def _build(self) -> StateGraph:
        g = StateGraph(PipelineState)
        work = {
            "ensure_set": self.ops.ensure_set,
            "detect": self.ops.detect,
            "recover": self.ops.recover,
            "collect": self.ops.apply_collection,
            "narrate": self.ops.narrate,
            "render": self.ops.render,
        }

        def node(name: str, status: Status, message: str):
            def run(state: PipelineState) -> dict:
                job = self._enter(status, message)
                if name == "review":
                    # Paused here until the review is complete. On resume the node runs
                    # again from the top, so a review re-opened meanwhile pauses again.
                    while not self.ops.review_done(job):
                        interrupt({"slug": job.slug, "waiting_for": "review"})
                else:
                    work[name](job)
                return {}
            run.__name__ = name
            return run

        prev = START
        for name, status, message in STAGES:
            g.add_node(name, node(name, status, message))
            g.add_edge(prev, name)
            prev = name
        g.add_edge(prev, END)
        return g

    # --- runner interface --------------------------------------------------------------

    def _config(self, slug: str) -> dict:
        return {"configurable": {"thread_id": slug}}

    def run(self, job: Job) -> Job:
        """Advance the job as far as it can go: until it finishes, fails, or waits for
        review. Mutates and returns `job`, like `Runner.run`."""
        if job.status.terminal:
            return job
        self._job = job
        cfg = self._config(job.slug)
        try:
            snap = self.graph.get_state(cfg)
            if not snap.values:  # first run of this video (or one the old runner started)
                self.graph.invoke({"slug": job.slug}, cfg)
            elif snap.interrupts:  # paused at review
                if not self.ops.review_done(job):
                    return job
                self.graph.invoke(Command(resume=True), cfg)
            elif snap.next:  # stopped mid-way: a crash or a failed stage
                self.graph.invoke(None, cfg)
            snap = self.graph.get_state(cfg)
            if not snap.next and not snap.interrupts:
                job.move(Status.DONE, "posts rendered")
                self.jobs.save(job)
                self.on_event(job, "posts rendered")
        except Exception as exc:  # noqa: BLE001 — any stage failure is recorded, not fatal
            stage = job.status
            job.fail(stage, f"{type(exc).__name__}: {exc}")
            self.jobs.save(job)
            self.on_event(job, f"failed during {stage.value}: {exc}")
        finally:
            self._job = None
        return job

    def retry(self, job: Job, force_budget: bool = False) -> Job:
        """Clear a failure. The checkpoint already knows which node to re-run."""
        if job.status != Status.FAILED or job.failed_at is None:
            raise ValueError(f"{job.slug} is {job.status.value}, not failed")
        job.error = None
        job.force_budget = job.force_budget or force_budget
        job.move(job.failed_at, "retry")
        job.failed_at = None
        self.jobs.save(job)
        return job
