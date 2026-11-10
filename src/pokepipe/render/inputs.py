"""Inputs shared by the script writer (stage 5) and the renderer (stage 6).

Kept separate from render.service so narration can ask "how long is each card on
screen?" before writing, without importing the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pokepipe.detection.models import PullsFile
from pokepipe.narration.models import Post
from pokepipe.narration.script import pull_ids
from pokepipe.render.timeline import EditParams, PackWindow, RenderPull, plan_edit
from pokepipe.review.models import ReviewState


def render_pulls(det: PullsFile, review: ReviewState, report) -> list[RenderPull]:
    """Kept pulls with footage timing (from detection) and card facts (from collection).
    Ids match the script's anchors because both order each pack's kept pulls by time."""
    facts = pull_ids(report)
    kept = review.kept()
    out = []
    for pack in sorted({p.pack for p in kept}):
        in_pack = sorted((p for p in kept if p.pack == pack), key=lambda p: p.time)
        for n, rp in enumerate(in_pack, 1):
            pid = f"p{pack}-{n}"
            fact = facts.get(pid)
            showings: tuple = ()
            if rp.detected_index is not None:
                d = det.pulls[rp.detected_index]
                start, end, box = d.start, d.end, d.box
                showings = tuple((s.start, s.end, s.box) for s in d.showings)
            else:  # added by hand in review: assume a short reveal around the given time
                start, end, box = max(0.0, rp.time - 0.5), rp.time + 1.5, None
            out.append(
                RenderPull(
                    id=pid,
                    pack=pack,
                    name=fact.name if fact else rp.card_id,
                    rarity=fact.rarity if fact else None,
                    variant=fact.variant if fact else "normal",
                    new=bool(fact and fact.new),
                    src_start=start,
                    src_end=end,
                    box=box,
                    showings=showings,
                )
            )
    return out


def pack_windows(det: PullsFile, post: Post) -> list[PackWindow]:
    return [
        PackWindow(p.index, p.clip_start, p.clip_end) for p in det.packs if p.index in post.packs
    ]


@dataclass(frozen=True)
class Airtime:
    """Seconds of footage around each card in a draft edit (before narration)."""

    before_first: float
    per_card: dict[str, float]  # pull id -> seconds until the next card appears
    after_last: float
    duration: float
    shown_again: dict[str, int] = field(default_factory=dict)  # pull id -> showcase shots


def airtime(det: PullsFile, review: ReviewState, report, post: Post, target: float) -> Airtime:
    pulls = [x for x in render_pulls(det, review, report) if x.pack in post.packs]
    edit = plan_edit(pack_windows(det, post), pulls, [], (1920, 1080), EditParams(target=target))
    order = sorted(edit.reveal_at.items(), key=lambda kv: kv[1])
    per_card = {}
    for (pid, at), nxt in zip(order, order[1:] + [(None, edit.duration)], strict=True):
        per_card[pid] = round(nxt[1] - at, 1)
    first = order[0][1] if order else edit.duration
    shown_again: dict[str, int] = {}
    for s in edit.segments:
        if s.showcase and s.kind != "zoom":  # a zoomed hit is one shot split in two
            shown_again[s.pull_id] = shown_again.get(s.pull_id, 0) + 1
    return Airtime(
        before_first=round(first, 1),
        per_card=per_card,
        after_last=per_card.get(order[-1][0], 0.0) if order else 0.0,
        duration=edit.duration,
        shown_again=shown_again,
    )
