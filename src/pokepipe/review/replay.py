"""Score the auto-reviewer against a finished human review (`pokepipe review replay`).

Re-runs auto-review on a fresh review (detection + recovery, nothing decided) with
the *current* policy, and compares each decision with what the person did:

- **silent errors**: the auto-reviewer confirmed something the person changed
  (rejected it, picked another card, flipped reverse holo, moved pack), or completed a
  pack the person had to add a card to. These would reach the collection and the
  posts unchecked. The goal is zero.
- **escalations**: pulls left for a person. Fewer is better, but these are safe.
- **escalated but already right**: escalations the person simply confirmed, i.e. what
  a looser policy could still automate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pokepipe.review.auto import AutoPolicy, auto_review
from pokepipe.review.models import ReviewedPull, ReviewState, Status
from pokepipe.review.store import ReviewStore


@dataclass
class ReplayResult:
    slug: str
    auto_confirmed: int = 0
    escalated: int = 0
    escalated_but_right: int = 0
    silent_errors: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    clean_packs: int = 0
    packs: int = 0

    @property
    def would_pause(self) -> bool:
        return bool(self.exceptions)


def _human_view(p: ReviewedPull) -> tuple:
    return (p.status, p.card_id, p.reverse_holo, p.pack)


def _diff(auto: ReviewedPull, human: ReviewedPull | None, name: str, namer) -> str | None:
    if human is None:
        return f"{name} (pack {auto.pack}): auto accepted a card the person didn't keep"
    if human.status == Status.REJECTED:
        return f"{name} (pack {auto.pack}): auto confirmed, person rejected"
    if human.card_id != auto.card_id:
        return (f"{name} (pack {auto.pack}): auto kept {name}, person chose "
                f"{namer(human.card_id)}")
    if human.reverse_holo != auto.reverse_holo:
        return (f"{name} (pack {auto.pack}): auto reverse holo={auto.reverse_holo}, "
                f"person {human.reverse_holo}")
    if human.pack != auto.pack:
        return f"{name}: auto pack {auto.pack}, person moved it to pack {human.pack}"
    return None


def replay(store: ReviewStore, slug: str, policy: AutoPolicy) -> ReplayResult:
    human = store.load(slug)
    if not human.completed:
        raise ValueError(f"{slug}: review isn't finished, nothing to compare against")
    if any(p.decided_by == "auto" for p in human.pulls):
        # decisions the auto-reviewer applied and nobody touched aren't ground truth
        raise ValueError(f"{slug}: review contains untouched auto decisions; replay needs a "
                         "review done by a person (shadow or always mode)")
    det = store.detection(slug)
    cards = store.cards(det.set_id)
    fresh: ReviewState = store.initial_state(slug)
    out = auto_review(det, fresh, cards, policy, mode="exceptions")
    res = ReplayResult(slug, exceptions=out.exceptions)

    by_detected = {p.detected_index: p for p in human.pulls if p.detected_index is not None}
    # agent suggestions, or the same card added by hand (reviews started before recovery)
    added_human = {(p.pack, p.card_id): p for p in human.pulls if p.detected_index is None}
    matched: set[tuple[int, str]] = set()

    def name(card_id: str) -> str:
        c = cards.get(card_id)
        return c.name if c else card_id

    auto_packs: set[int] = set()
    all_packs = {p.pack for p in fresh.pulls}
    for p in fresh.pulls:
        h = (by_detected.get(p.detected_index) if p.detected_index is not None
             else added_human.get((p.pack, p.card_id)))
        if p.detected_index is None and h is not None:
            matched.add((p.pack, p.card_id))
        if p.status == Status.CONFIRMED:
            res.auto_confirmed += 1
            auto_packs.add(p.pack)
            err = _diff(p, h, name(p.card_id), name)
            if err:
                res.silent_errors.append(err)
        else:
            res.escalated += 1
            if h is not None and _human_view(h) == (Status.CONFIRMED, p.card_id,
                                                    p.reverse_holo, p.pack):
                res.escalated_but_right += 1
    # a card the person had to add by hand to a pack the auto-reviewer called complete
    for p in human.pulls:
        if p.detected_index is None and p.status == Status.CONFIRMED \
                and p.pack in auto_packs and (p.pack, p.card_id) not in matched:
            res.silent_errors.append(f"{name(p.card_id)} (pack {p.pack}): person added it, "
                                     "auto had called the pack complete")
    res.packs = len(all_packs)
    res.clean_packs = len(all_packs - {p.pack for p in fresh.pulls
                                       if p.status == Status.PENDING})
    return res
