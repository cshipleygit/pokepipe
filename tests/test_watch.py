from pathlib import Path

import pytest

from pokepipe.watch.intake import Inbox, JobStore
from pokepipe.watch.models import Job, Status
from pokepipe.watch.runner import Runner
from pokepipe.watch.service import Watcher


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def env(tmp_path):
    inbox_root, videos = tmp_path / "inbox", tmp_path / "videos"
    (inbox_root / "sv3pt5").mkdir(parents=True)
    jobs = JobStore(videos)
    clock = Clock()
    inbox = Inbox(inbox_root, jobs, default_set=None, settle_seconds=10, clock=clock)
    return inbox_root, jobs, inbox, clock


# --- intake ----------------------------------------------------------------------------------


def test_video_is_taken_only_after_it_stops_changing(env):
    root, jobs, inbox, clock = env
    video = root / "sv3pt5" / "My Opening!.MP4"
    video.write_bytes(b"x" * 100)
    assert inbox.ready() == []  # first sighting

    clock.t = 5
    video.write_bytes(b"x" * 200)  # still copying
    assert inbox.ready() == []
    clock.t = 14  # 9 s since the last change
    assert inbox.ready() == []
    clock.t = 16
    [candidate] = inbox.ready()
    assert candidate.set_id == "sv3pt5"

    job = inbox.take(candidate)
    assert job.slug == "my-opening" and job.status == Status.QUEUED
    assert not video.exists()  # moved, not copied
    source = Path(jobs.load("my-opening").source)
    assert (source.parent.name, source.name) == ("my-opening", "source.mp4")
    assert source.read_bytes() == b"x" * 200


def test_partial_downloads_and_non_videos_are_ignored(env):
    root, _, inbox, clock = env
    (root / "sv3pt5" / "a.mp4.part").write_bytes(b"x")
    (root / "sv3pt5" / "notes.txt").write_bytes(b"x")
    (root / "sv3pt5" / "~lock.mp4").write_bytes(b"x")
    inbox.ready()
    clock.t = 60
    assert inbox.ready() == []


def test_root_drop_uses_default_set_or_refuses(env):
    root, jobs, _, clock = env
    (root / "loose.mp4").write_bytes(b"x")
    no_default = Inbox(root, jobs, default_set=None, settle_seconds=0, clock=clock)
    no_default.ready()
    [c] = no_default.ready()
    assert c.set_id is None
    with pytest.raises(ValueError, match="no set"):
        no_default.take(c)
    assert (root / "loose.mp4").exists()  # left in place

    with_default = Inbox(root, jobs, default_set="sv8", settle_seconds=0, clock=clock)
    with_default.ready()
    [c] = with_default.ready()
    assert with_default.take(c).set_id == "sv8"


def test_slug_collisions_get_a_suffix(env):
    root, jobs, inbox, _ = env
    (jobs.root / "opening").mkdir(parents=True)  # e.g. processed manually before
    (root / "sv3pt5" / "opening.mp4").write_bytes(b"x")
    inbox.settle = 0
    inbox.ready()
    [c] = inbox.ready()
    assert inbox.take(c).slug == "opening-2"


# --- runner ----------------------------------------------------------------------------------


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

    def review_done(self, job):
        return self.reviewed

    def apply_collection(self, job):
        self._do("collection")

    def narrate(self, job):
        self._do("narrate")

    def render(self, job):
        self._do("render")


def new_job(jobs: JobStore, slug="v") -> Job:
    job = Job(slug=slug, source="x.mp4", set_id="s")
    jobs.save(job)
    return job


def test_runner_pauses_at_review_then_finishes(tmp_path):
    jobs, ops = JobStore(tmp_path), FakeOps()
    runner = Runner(ops, jobs)
    job = runner.run(new_job(jobs))
    assert job.status == Status.AWAITING_REVIEW
    assert ops.calls == ["ensure_set", "detect"]
    assert runner.run(job).status == Status.AWAITING_REVIEW  # still waiting, nothing re-run
    assert ops.calls == ["ensure_set", "detect"]

    ops.reviewed = True
    # A fresh runner reading the job from disk carries on: restart-safe.
    job = Runner(ops, jobs).run(jobs.load("v"))
    assert job.status == Status.DONE
    assert ops.calls == ["ensure_set", "detect", "collection", "narrate", "render"]
    assert [e.status for e in job.history][-1] == Status.DONE


def test_failure_is_recorded_and_retry_resumes_at_that_stage(tmp_path):
    jobs, ops = JobStore(tmp_path), FakeOps()
    ops.reviewed, ops.fail_on = True, "narrate"
    runner = Runner(ops, jobs)
    job = runner.run(new_job(jobs))
    assert job.status == Status.FAILED and job.failed_at == Status.NARRATING
    assert "narrate broke" in job.error
    assert jobs.load("v").status == Status.FAILED  # persisted

    ops.fail_on = None
    job = runner.retry(job, force_budget=True)
    assert job.status == Status.NARRATING and job.force_budget and job.error is None
    assert runner.run(job).status == Status.DONE
    assert ops.calls.count("detect") == 1  # earlier stages weren't redone


def test_retry_rejects_jobs_that_did_not_fail(tmp_path):
    jobs = JobStore(tmp_path)
    with pytest.raises(ValueError, match="not failed"):
        Runner(FakeOps(), jobs).retry(new_job(jobs))


def test_watcher_tick_takes_in_and_runs_to_review(env):
    root, jobs, inbox, clock = env
    inbox.settle = 0
    (root / "sv3pt5" / "pack day.mp4").write_bytes(b"x")
    messages: list[str] = []
    watcher = Watcher(inbox, Runner(FakeOps(), jobs), jobs, log=messages.append)
    watcher.tick()  # first sighting
    watcher.tick()  # taken in and run
    watcher.tick()  # still waiting: no repeat announcement
    [job] = jobs.all()
    assert job.status == Status.AWAITING_REVIEW
    assert messages == ["new video: pack-day (set sv3pt5)", "pack-day is ready for review"]
