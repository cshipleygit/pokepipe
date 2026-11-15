"""What the control room shows: every video's place in the pipeline, read from disk.

Pure reads of the files each stage already writes (job.json, pulls.json, review.json,
costs.json, posts/<n>/post.mp4), so the page works whether or not the watcher runs in
the same process, and showing it never changes anything.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pokepipe.watch.intake import PARTIAL_SUFFIXES, VIDEO_EXTS, JobStore
from pokepipe.watch.models import ORDER, Job, Status

# What each step is called on the page, in pipeline order.
STEPS: list[tuple[Status, str]] = [
    (Status.QUEUED, "Received"),
    (Status.DETECTING, "Finding cards"),
    (Status.AWAITING_REVIEW, "Review"),
    (Status.COLLECTING, "Collection"),
    (Status.NARRATING, "Script & voice"),
    (Status.RENDERING, "Rendering"),
    (Status.DONE, "Posts ready"),
]


@dataclass
class LiveStatus:
    """Filled in by `watch run` when the control room runs inside it."""

    engine: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_tick: datetime | None = None
    working_on: str | None = None
    last_message: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def tick(self) -> None:
        with self._lock:
            self.last_tick = datetime.now(UTC)

    def event(self, slug: str, message: str) -> None:
        with self._lock:
            self.working_on, self.last_message = slug, message

    def view(self) -> dict:
        with self._lock:
            return {
                "running": True,
                "engine": self.engine,
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "last_tick": self.last_tick.isoformat(timespec="seconds")
                if self.last_tick else None,
                "working_on": self.working_on,
                "last_message": self.last_message,
            }


def _json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def step_index(job: Job) -> int:
    """Index into STEPS of the step the job is on (for a failure: the step that failed)."""
    status = job.failed_at if job.status == Status.FAILED and job.failed_at else job.status
    return ORDER.index(status) if status in ORDER else 0


def job_view(job: Job, video_dir: Path, set_names: dict[str, str]) -> dict:
    pulls = _json(video_dir / "pulls.json") or {}
    review = _json(video_dir / "review.json") or {}
    costs = _json(video_dir / "costs.json") or {}
    posts = sorted(
        (int(d.name) for d in (video_dir / "posts").glob("*")
         if d.is_dir() and d.name.isdigit() and (d / "post.mp4").exists()),
    )
    review_pulls = review.get("pulls", []) if isinstance(review, dict) else []
    first = job.history[0].at if job.history else None
    last = job.history[-1] if job.history else None
    return {
        "slug": job.slug,
        "set_id": job.set_id,
        "set_name": set_names.get(job.set_id, job.set_id),
        "status": job.status.value,
        "step": step_index(job),
        "failed": job.status == Status.FAILED,
        "error": job.error,
        "started_at": first,
        "updated_at": last.at if last else None,
        "message": last.message if last else "",
        "packs": len(pulls.get("packs", [])) if isinstance(pulls, dict) else None,
        "pulls": len(pulls.get("pulls", [])) if isinstance(pulls, dict) else None,
        "review_pending": sum(p.get("status") == "pending" for p in review_pulls),
        "review_completed": bool(review.get("completed")) if isinstance(review, dict) else False,
        "cost_usd": round(sum(e.get("usd", 0) for e in costs.get("entries", [])), 2)
        if isinstance(costs, dict) else 0.0,
        "posts": posts,
        "history": [h.model_dump(mode="json") for h in job.history[-8:]],
    }


def inbox_waiting(inbox: Path) -> list[dict]:
    """Videos sitting in the inbox that the watcher hasn't taken yet."""
    if not inbox.exists():
        return []
    out = []
    for path in sorted([*inbox.iterdir(), *inbox.glob("*/*")]):
        if (path.is_file() and path.suffix.lower() in VIDEO_EXTS
                and not path.name.lower().endswith(PARTIAL_SUFFIXES)):
            out.append({
                "name": path.name,
                "set_id": path.parent.name if path.parent != inbox else None,
                "size": path.stat().st_size,
            })
    return out


def set_names(sets_root: Path) -> dict[str, str]:
    names = {}
    for meta in sets_root.glob("*/set.json"):
        data = _json(meta)
        if isinstance(data, dict) and data.get("id"):
            names[data["id"]] = data.get("name", data["id"])
    return names


def build(jobs: JobStore, inbox: Path, sets_root: Path, live: LiveStatus | None) -> dict:
    names = set_names(sets_root)
    videos = [job_view(j, jobs.root / j.slug, names) for j in jobs.all()]
    videos.reverse()  # newest first
    return {
        "watcher": live.view() if live else {"running": False},
        "steps": [label for _, label in STEPS],
        "inbox": inbox_waiting(inbox),
        "videos": videos,
    }
