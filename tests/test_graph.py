"""The LangGraph watch engine: same contract as watch.runner, with checkpoints."""

from __future__ import annotations

import pytest

from pokepipe.graph.pipeline import GraphRunner, open_checkpointer
from pokepipe.watch.intake import JobStore
from pokepipe.watch.models import Job, Status


class FakeOps:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.reviewed = False
        self.fail_on: str | None = None

    def _do(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_on == name:
            raise RuntimeError(f"{name} broke")

    def ensure_set(self, job):
        self._do("ensure_set")

    def detect(self, job):
        self._do("detect")

    def recover(self, job):
        self._do("recover")

    def review_done(self, job):
        return self.reviewed

    def apply_collection(self, job):
        self._do("collection")

    def narrate(self, job):
        self._do("narrate")

    def render(self, job):
        self._do("render")


ALL = ["ensure_set", "detect", "recover", "collection", "narrate", "render"]


def setup(tmp_path, ops=None):
    jobs = JobStore(tmp_path / "videos")
    job = Job(slug="v", source="x.mp4", set_id="s")
    jobs.save(job)
    ops = ops or FakeOps()
    runner = GraphRunner(ops, jobs, open_checkpointer(tmp_path / "ck.sqlite"))
    return jobs, ops, runner, job


def test_pauses_at_review_interrupt_then_resumes(tmp_path):
    jobs, ops, runner, job = setup(tmp_path)
    runner.run(job)
    assert job.status == Status.AWAITING_REVIEW
    assert ops.calls == ["ensure_set", "detect", "recover"]
    snap = runner.graph.get_state({"configurable": {"thread_id": "v"}})
    assert snap.interrupts and snap.next == ("review",)

    runner.run(job)  # still waiting: nothing re-runs
    assert ops.calls == ["ensure_set", "detect", "recover"]

    ops.reviewed = True
    runner.run(job)
    assert job.status == Status.DONE
    assert ops.calls == ALL
    assert jobs.load("v").status == Status.DONE  # job.json is kept as the status view


def test_restart_resumes_from_the_checkpoint(tmp_path):
    jobs, ops, runner, job = setup(tmp_path)
    runner.run(job)
    # A new process: fresh runner, fresh connection, job read back from disk.
    ops.reviewed = True
    again = GraphRunner(ops, jobs, open_checkpointer(tmp_path / "ck.sqlite"))
    job = again.run(jobs.load("v"))
    assert job.status == Status.DONE
    assert ops.calls == ALL  # detection etc. were not redone


def test_failure_is_recorded_and_retry_resumes_at_the_failed_node(tmp_path):
    jobs, ops, runner, job = setup(tmp_path)
    ops.reviewed, ops.fail_on = True, "narrate"
    runner.run(job)
    assert job.status == Status.FAILED and job.failed_at == Status.NARRATING
    assert "narrate broke" in job.error
    assert jobs.load("v").status == Status.FAILED

    ops.fail_on = None
    runner.retry(job, force_budget=True)
    assert job.status == Status.NARRATING and job.force_budget and job.error is None
    runner.run(job)
    assert job.status == Status.DONE
    assert ops.calls.count("detect") == 1 and ops.calls.count("narrate") == 2


def test_reopened_review_pauses_again(tmp_path):
    """Resuming re-runs the review node, which checks again before moving on."""
    jobs, ops, runner, job = setup(tmp_path)
    runner.run(job)

    class Flaky(FakeOps):
        checks = 0

        def review_done(self, job):  # done when the watcher polls, re-opened by resume
            Flaky.checks += 1
            return Flaky.checks == 1

    flaky = Flaky()
    flaky.calls = ops.calls
    runner = GraphRunner(flaky, jobs, open_checkpointer(tmp_path / "ck.sqlite"))
    runner.run(job)
    assert job.status == Status.AWAITING_REVIEW
    assert "collection" not in flaky.calls


def test_inner_loops_restart_cleanly_after_a_crash_inside_the_pipeline(tmp_path):
    """Regression (first real end-to-end run): the writer loop nested in a checkpointed
    graph inherited its checkpointer; resumed after a crash, its drafts and SDK objects
    came back from SQLite as dicts. It must restart from scratch instead."""
    from types import SimpleNamespace
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    from pokepipe.narration.models import Beat, ScriptDraft
    from pokepipe.narration.writer import build_writer

    draft = ScriptDraft(hook_caption="Guess", hashtags=[], beats=[Beat(anchor="intro",
                                                                       text="Hi")])
    calls = {"write": 0, "crash": True}

    def write(feedback):
        calls["write"] += 1
        return draft, SimpleNamespace(model="m"), 0.01

    def critique(d):
        if calls["crash"]:
            calls["crash"] = False
            raise RuntimeError("stopped mid-script")
        return [], [], 0.0

    class S(TypedDict):
        model: str

    def narrate(state: S) -> dict:
        out = build_writer(write, critique, max_rewrites=2).invoke({})
        return {"model": out["attempts"][-1].response.model}  # crashed here on dicts

    g = StateGraph(S)
    g.add_node("narrate", narrate)
    g.add_edge(START, "narrate")
    g.add_edge("narrate", END)
    app = g.compile(checkpointer=open_checkpointer(tmp_path / "ck.sqlite"))
    cfg = {"configurable": {"thread_id": "v"}}
    with pytest.raises(RuntimeError, match="mid-script"):
        app.invoke({"model": ""}, cfg)
    assert app.invoke(None, cfg)["model"] == "m"
    assert calls["write"] == 2  # the loop started over: no half-finished state reused


def test_jobs_already_done_are_left_alone(tmp_path):
    jobs, ops, runner, job = setup(tmp_path)
    job.move(Status.DONE)
    runner.run(job)
    assert ops.calls == []
