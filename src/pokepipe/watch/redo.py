"""Redo a finished video after its review was fixed (`pokepipe watch redo <slug>`).

A wrong card spotted in a rendered post is fixed in review, which re-opens it. Before,
the posts already rendered stayed as they were. Now each post records a key of the
review it was made from (`posts/<n>/review_key.txt`): every kept pull in its packs and
the packs before them (the master-set progress and "new card" flags depend on those).
`redo` deletes only the posts whose key changed and sends the job back to collection,
so the collection is re-applied and just those posts are re-scripted and re-rendered.
Voice lines that didn't change come from the TTS cache, free.
"""

from __future__ import annotations

import hashlib
import shutil

from pokepipe.narration.models import PostPlan
from pokepipe.narration.service import VideoFiles
from pokepipe.review.models import ReviewState

KEY_FILE = "review_key.txt"


def post_key(state: ReviewState, packs: list[int]) -> str:
    last = max(packs)
    kept = sorted((p.pack, p.time, p.card_id, p.reverse_holo)
                  for p in state.kept() if p.pack <= last)
    raw = repr((sorted(packs), [(pk, c, r) for pk, _, c, r in kept]))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def stamp(files: VideoFiles, plan: PostPlan, state: ReviewState) -> None:
    """Record what each post is made from (only where not recorded yet)."""
    for post in plan.posts:
        path = files.post_dir(post.index) / KEY_FILE
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(post_key(state, post.packs), encoding="utf-8")


def stale_posts(files: VideoFiles, plan: PostPlan, state: ReviewState) -> list[int]:
    """Posts on disk that no longer match the review or the plan."""
    root = files.root / "posts"
    if not root.exists():
        return []
    planned = {p.index: p for p in plan.posts}
    stale = []
    for d in sorted(root.iterdir(), key=lambda d: (len(d.name), d.name)):
        if not d.is_dir() or not d.name.isdigit():
            continue
        post = planned.get(int(d.name))
        key = d / KEY_FILE
        if post is None or not key.exists() or \
                key.read_text(encoding="utf-8").strip() != post_key(state, post.packs):
            stale.append(int(d.name))
    return stale


def clear(files: VideoFiles, indexes: list[int]) -> None:
    for n in indexes:
        shutil.rmtree(files.post_dir(n), ignore_errors=True)
