"""Which posts have gone to TikTok: posts/<n>/tiktok.json, written once per post so a
post is never sent twice (TikTok allows only 5 pending drafts a day)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pokepipe.narration.models import PostPlan
from pokepipe.tiktok.client import TikTokClient

RECORD = "tiktok.json"


def sent(post_dir: Path) -> dict | None:
    path = post_dir / RECORD
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def send_drafts(video_dir: Path, client: TikTokClient, only: list[int] | None = None,
                log=print) -> list[int]:
    """Send every rendered, not-yet-sent post as a draft; returns the post numbers sent."""
    plan = PostPlan.model_validate_json((video_dir / "posts.json").read_text(encoding="utf-8"))
    done = []
    for post in plan.posts:
        if only and post.index not in only:
            continue
        pdir = video_dir / "posts" / str(post.index)
        video = pdir / "post.mp4"
        if not video.exists():
            log(f"post {post.index}: not rendered yet, skipped")
            continue
        if sent(pdir):
            log(f"post {post.index}: already sent to TikTok, skipped")
            continue
        publish_id = client.send_draft(video)
        (pdir / RECORD).write_text(json.dumps({
            "publish_id": publish_id, "sent_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "bytes": video.stat().st_size}, indent=2), encoding="utf-8")
        log(f"post {post.index}: sent to the TikTok inbox (open the notification in the app, "
            "paste caption.txt, post)")
        done.append(post.index)
    return done
