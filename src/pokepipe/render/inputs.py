"""Inputs shared by the script writer (stage 5) and the renderer (stage 6).

Kept separate from render.service so narration can ask "how long is each card on
screen?" before writing, without importing the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from pokepipe.detection.models import PullsFile
from pokepipe.detection.moments import MomentsFile
from pokepipe.narration.models import Post
from pokepipe.narration.script import pull_ids
from pokepipe.render.timeline import Clip, EditParams, PackWindow, RenderPull, plan_edit
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
                    card_id=rp.card_id,
                )
            )
    return out


def pack_windows(det: PullsFile, post: Post) -> list[PackWindow]:
    return [
        PackWindow(p.index, p.clip_start, p.clip_end) for p in det.packs if p.index in post.packs
    ]


def load_moments(video_dir: Path) -> MomentsFile | None:
    path = video_dir / "moments.json"
    if not path.exists():
        return None
    return MomentsFile.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass
class EditInputs:
    packs: list[PackWindow]
    pulls: list[RenderPull]
    lead: list[Clip] = field(default_factory=list)
    recap: list[Clip] = field(default_factory=list)


ENERGY_LEAD = 0.2  # start a pack this long before its Energy shows
RECAP_BIG = 2.0  # seconds per Double Rare-or-better hit in the end recap
RECAP_HIT = 1.2  # seconds per other hit (the pack's rare)


def edit_inputs(
    det: PullsFile, review: ReviewState, report, post: Post, moments: MomentsFile | None
) -> EditInputs:
    """What the edit is made of. Without moments.json: each pack's whole clip window,
    repeat showings in place (the original edit). With it, the series edit:

    - **lead clips**: the box opening (only in the post with the first opened pack), then
      every pack's tear and cards coming out, back to back;
    - **packs** start at their Energy, cutting the straightening, code card and peeks;
      repeat showings are dropped from the packs;
    - **recap**: each hit again at the end, from footage of it being sleeved or shown
      off after its pack when there is some, under the outro.
    """
    pulls = [x for x in render_pulls(det, review, report) if x.pack in post.packs]
    windows = pack_windows(det, post)
    if moments is None:
        return EditInputs(windows, pulls)

    lead: list[Clip] = []
    if moments.packs and moments.packs[0].index in post.packs:
        for name, span in moments.intro:
            if span:
                lead.append(Clip(span.start, span.end, f"box_{name}"))
    for w in windows:
        pm = moments.pack(w.index)
        for name in ("rip", "cards_out"):
            span = getattr(pm, name) if pm else None
            if span:
                lead.append(Clip(span.start, span.end, name))

    packs = []
    for w in windows:
        pm = moments.pack(w.index)
        if pm and pm.energy_at is not None:
            first = min((x.src_start for x in pulls if x.pack == w.index), default=w.clip_end)
            w = PackWindow(w.index, min(pm.energy_at - ENERGY_LEAD, first), w.clip_end)
        packs.append(w)

    recap: list[Clip] = []
    last_card = {w.index: max((x.src_end for x in pulls if x.pack == w.index), default=0.0)
                 for w in windows}
    for x in pulls:
        if not x.hit:
            continue
        length = RECAP_BIG if x.label else RECAP_HIT
        after = [sh for sh in x.showings if sh[0] > last_card[x.pack]]
        if after:  # sleeved / shown off after the pack: fresh footage of the hit
            start, end, box = after[0]
            mid = (start + end) / 2
            begin = max(start, mid - length / 2)
            recap.append(Clip(begin, begin + length, "recap", x.id, box or x.box,
                              max_end=max(end, begin + length)))
        else:
            start = x.src_start + 0.8  # past the reveal, the card held up
            recap.append(Clip(start, start + length, "recap", x.id, x.box,
                              max_end=max(x.src_end, start + length)))
    pulls = [replace(x, showings=()) for x in pulls]
    return EditInputs(packs, pulls, lead, recap)


@dataclass(frozen=True)
class Airtime:
    """Seconds of footage around each card in a draft edit (before narration)."""

    before_first: float
    per_card: dict[str, float]  # pull id -> seconds until the next card appears
    after_last: float
    duration: float
    shown_again: dict[str, int] = field(default_factory=dict)  # pull id -> showcase shots
    opening: list[str] = field(default_factory=list)  # lead clip labels, in order
    recap: list[str] = field(default_factory=list)  # pull ids replayed at the end


def airtime(
    det: PullsFile, review: ReviewState, report, post: Post, target: float,
    moments: MomentsFile | None = None,
) -> Airtime:
    inputs = edit_inputs(det, review, report, post, moments)
    edit = plan_edit(inputs.packs, inputs.pulls, [], (1920, 1080), EditParams(target=target),
                     lead=inputs.lead, recap=inputs.recap)
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
        opening=[c.label for c in inputs.lead],
        recap=[c.pull_id for c in inputs.recap],
    )
