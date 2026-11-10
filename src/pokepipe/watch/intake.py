"""Inbox intake: notice new videos, wait until they've finished copying, take them in.

Layout: `<inbox>/<set_id>/video.mp4` names the set by folder; videos dropped straight
into `<inbox>/` use the configured default set. A file is only taken once its size and
mtime have been unchanged for `settle_seconds` *and* it can be opened for writing (on
Windows, a file still being copied is locked). Then it is moved (not copied) to
`data/videos/<slug>/source.<ext>`, so its path never changes again.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pokepipe.detection.service import video_slug
from pokepipe.watch.models import Job

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}
PARTIAL_SUFFIXES = (".part", ".crdownload", ".tmp", ".download")


@dataclass
class _Seen:
    size: int
    mtime: float
    since: float


@dataclass(frozen=True)
class Candidate:
    path: Path
    set_id: str | None  # None: dropped in the inbox root with no default set


class JobStore:
    """job.json files under data/videos/<slug>/."""

    def __init__(self, videos_root: Path) -> None:
        self.root = videos_root

    def path(self, slug: str) -> Path:
        return self.root / slug / "job.json"

    def save(self, job: Job) -> None:
        path = self.path(job.slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    def load(self, slug: str) -> Job:
        return Job.model_validate_json(self.path(slug).read_text(encoding="utf-8"))

    def all(self) -> list[Job]:
        if not self.root.exists():
            return []
        jobs = [
            Job.model_validate_json(p.read_text(encoding="utf-8"))
            for p in self.root.glob("*/job.json")
        ]
        return sorted(jobs, key=lambda j: j.history[0].at if j.history else "")


class Inbox:
    def __init__(
        self,
        root: Path,
        jobs: JobStore,
        default_set: str | None = None,
        settle_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root = root
        self.jobs = jobs
        self.default_set = default_set
        self.settle = settle_seconds
        self.clock = clock
        self._seen: dict[Path, _Seen] = {}

    def _videos(self) -> list[Path]:
        if not self.root.exists():
            return []
        found = []
        for path in [*self.root.iterdir(), *self.root.glob("*/*")]:
            if (
                path.is_file()
                and path.suffix.lower() in VIDEO_EXTS
                and not path.name.startswith((".", "~"))
                and not path.name.lower().endswith(PARTIAL_SUFFIXES)
            ):
                found.append(path)
        return sorted(found)

    def _set_for(self, path: Path) -> str | None:
        return path.parent.name if path.parent != self.root else self.default_set

    @staticmethod
    def _unlocked(path: Path) -> bool:
        try:
            with path.open("rb+"):
                return True
        except OSError:  # still being written by another program
            return False

    def ready(self) -> list[Candidate]:
        """Videos that have stopped changing for `settle_seconds`."""
        now = self.clock()
        out = []
        present = set()
        for path in self._videos():
            present.add(path)
            stat = path.stat()
            seen = self._seen.get(path)
            if seen is None or (seen.size, seen.mtime) != (stat.st_size, stat.st_mtime):
                self._seen[path] = _Seen(stat.st_size, stat.st_mtime, now)
                continue
            if now - seen.since >= self.settle and stat.st_size > 0 and self._unlocked(path):
                out.append(Candidate(path, self._set_for(path)))
        for gone in set(self._seen) - present:
            del self._seen[gone]
        return out

    def _unique_slug(self, path: Path) -> str:
        base = video_slug(path)
        slug, n = base, 2
        while (self.jobs.root / slug).exists():
            slug, n = f"{base}-{n}", n + 1
        return slug

    def take(self, candidate: Candidate) -> Job:
        if not candidate.set_id:
            raise ValueError(
                f"{candidate.path.name}: no set — drop it in <inbox>/<set_id>/ "
                "or set WATCH_DEFAULT_SET"
            )
        slug = self._unique_slug(candidate.path)
        dest = self.jobs.root / slug / f"source{candidate.path.suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(candidate.path), dest)
        self._seen.pop(candidate.path, None)
        job = Job(slug=slug, source=str(dest), set_id=candidate.set_id)
        job.move(job.status, f"taken in from {candidate.path.name}")
        self.jobs.save(job)
        return job
