"""The watch loop: take in new videos, then push every unfinished job forward.

Single-threaded on purpose: detection and rendering both want the whole GPU, so jobs
run one step at a time in arrival order. The review page runs alongside in its own
thread (see cli.py) and only writes review.json/posts.json, which the loop polls.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from pokepipe.watch.intake import Inbox, JobStore
from pokepipe.watch.models import Job, Status
from pokepipe.watch.runner import Runner


class Watcher:
    def __init__(
        self, inbox: Inbox, runner: Runner, jobs: JobStore, log: Callable[[str], None] = print,
        notify: Callable[[str, str], object] = lambda title, message: None,
    ) -> None:
        self.inbox = inbox
        self.runner = runner
        self.jobs = jobs
        self.log = log
        self.notify = notify
        self._warned: set[str] = set()
        self._waiting_announced: set[str] = set()
        self._finished_announced: set[str] = set()

    def intake(self) -> list[Job]:
        taken = []
        for candidate in self.inbox.ready():
            try:
                job = self.inbox.take(candidate)
            except (ValueError, OSError) as exc:
                key = str(candidate.path)
                if key not in self._warned:  # say it once, not every poll
                    self._warned.add(key)
                    self.log(f"skipping {candidate.path.name}: {exc}")
                continue
            self.log(f"new video: {job.slug} (set {job.set_id})")
            taken.append(job)
        return taken

    def tick(self) -> list[Job]:
        """One pass: intake, then advance every unfinished job as far as it can go."""
        self.intake()
        active = [j for j in self.jobs.all() if not j.status.terminal]
        for job in active:
            self.runner.run(job)
            if job.status == Status.AWAITING_REVIEW and job.slug not in self._waiting_announced:
                self._waiting_announced.add(job.slug)
                self.log(f"{job.slug} is ready for review")
                self.notify("Pokepipe: review waiting", f"{job.slug} needs a look")
            if job.status.terminal and job.slug not in self._finished_announced:
                self._finished_announced.add(job.slug)
                if job.status == Status.DONE:
                    self.notify("Pokepipe: posts ready", f"{job.slug} is rendered")
                else:
                    self.notify("Pokepipe: video failed",
                                f"{job.slug}: {(job.error or '')[:120]}")
        return active

    def loop(self, poll_seconds: float, stop: Callable[[], bool] = lambda: False) -> None:
        while not stop():
            self.tick()
            time.sleep(poll_seconds)
