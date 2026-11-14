"""The auto-reviewer: confirm what the evidence settles, escalate only what it doesn't.

Review used to be the one human stop for every video. This decides each pack the way a
reviewer would when the evidence is clear, and leaves the rest pending with a reason:

- A **pack** is auto-confirmed only if, with the recovery agent's high-confidence
  suggestions, it has exactly the expected number of cards, its rarities fit the slot
  layout (so reverse holos are known), and every pull in it is trusted.
- A **pull** is trusted if its score clears the bar (`min_confidence`, or the fitted
  calibration from `pokepipe review fit`) and it beats the best *other* card by at
  least `min_lead` (near-identical art like Enamorus U vs IR is a person's call).
- Otherwise the whole pack stays pending, so the reviewer sees it in context.

Every auto decision carries `decided_by="auto"` and a note; a human edit clears the
flag. `review replay` measures this against finished human reviews, and
`review fit` only learns from human decisions, so the reviewer never grades itself.

Modes (`REVIEW_MODE`):
- `always`: no auto-review (the old behaviour).
- `shadow`: decisions go to `auto_review.json` only; you review as normal, blind to
  them, and `review replay --all` scores how often it would have been right.
- `exceptions`: decisions are applied; the run pauses only if something is left. A
  video with no exceptions goes straight on to collection, script and render.
- `off`: never pause. Unresolved detected pulls keep detection's best guess, and
  unresolved agent suggestions are dropped. Each is noted, for checking later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from pokepipe.cards.models import Card
from pokepipe.detection.layout import LAYOUTS, PackLayout, apply_layout
from pokepipe.detection.models import Pull, PullsFile
from pokepipe.recover.evidence import synthetic_pull
from pokepipe.review.calibrate import Calibration
from pokepipe.review.models import AutoReviewInfo, ReviewedPull, ReviewState, Status
from pokepipe.review.store import ReviewStore

MODES = ("always", "shadow", "exceptions", "off")
SHADOW_FILE = "auto_review.json"


@dataclass(frozen=True)
class AutoPolicy:
    min_confidence: float = 0.6  # on detection's heuristic `confidence` (0-1)
    min_lead: float = 0.02  # peak score over the best different card
    accept_suggestions: bool = True  # high-confidence recovery suggestions
    require_layout: bool = True  # no slot layout → reverse holos unknown → escalate
    calibration: Calibration | None = None  # if fitted, replaces min_confidence

    def describe(self) -> dict:
        d = asdict(self)
        d["calibration"] = self.calibration.summary() if self.calibration else None
        return d


@dataclass
class AutoResult:
    state: ReviewState
    exceptions: list[str] = field(default_factory=list)
    confirmed: int = 0

    @property
    def clean(self) -> bool:
        return not self.exceptions


def lead(pull: Pull) -> float | None:
    """How far the matched card beats the best different card (decoys ignored)."""
    others = [a.score for a in pull.alternatives
              if a.card_id != pull.card_id and not a.card_id.startswith("decoy:")]
    return round(pull.peak_score - max(others), 4) if others else None


def trust(pull: Pull, policy: AutoPolicy) -> tuple[bool, str]:
    """Whether a detected pull is clear enough to confirm, and why not if it isn't."""
    if policy.calibration is not None:
        p = policy.calibration.probability(pull)
        if p < policy.calibration.threshold:
            return False, f"{pull.name}: calibrated {p:.2f} < {policy.calibration.threshold:.2f}"
    elif pull.confidence < policy.min_confidence:
        return False, f"{pull.name}: confidence {pull.confidence:.2f} < {policy.min_confidence}"
    gap = lead(pull)
    if gap is not None and gap < policy.min_lead:
        runner = next(a.card_id for a in pull.alternatives if a.card_id != pull.card_id
                      and not a.card_id.startswith("decoy:"))
        return False, f"{pull.name}: only {gap:.3f} ahead of {runner} (near-identical art?)"
    return True, ""


def _layout_reverse(members: list[ReviewedPull], cards: dict[str, Card], pack: int,
                    layout: PackLayout | None) -> dict[int, bool] | None:
    """Re-infer reverse holos for a whole pack (incl. accepted suggestions), by review
    pull id. None if there is no layout or the pack doesn't fit it."""
    if layout is None:
        return None
    ordered = sorted(members, key=lambda p: p.time)
    fakes = [synthetic_pull(cards[p.card_id], pack, p.time) for p in ordered]
    result = apply_layout(fakes, layout)
    if result.mode != "full":
        return None
    return {p.id: f.reverse_guess for p, f in zip(ordered, fakes, strict=True)}


def auto_review(
    det: PullsFile, state: ReviewState, cards: dict[str, Card], policy: AutoPolicy,
    mode: str = "exceptions",
) -> AutoResult:
    """Decide what can be decided on a fresh review. `state` is changed in place."""
    if mode not in MODES or mode == "always":
        raise ValueError(f"auto-review mode must be one of {MODES[1:]}, not {mode!r}")
    expected = {pk.index: pk.expected_cards for pk in det.packs}
    layouts = {pk.index: pk.layout for pk in det.packs}
    set_layout = LAYOUTS.get(det.pack_layout or "")
    result = AutoResult(state)
    by_pack: dict[int, list[ReviewedPull]] = {}
    for p in state.pulls:
        by_pack.setdefault(p.pack, []).append(p)

    for pack, pulls in sorted(by_pack.items()):
        problems: list[str] = []
        detected = [p for p in pulls if p.detected_index is not None]
        suggested = [p for p in pulls if p.detected_index is None and p.suggested]
        for p in detected:
            ok, why = trust(det.pulls[p.detected_index], policy)
            if not ok:
                problems.append(why)
        accepted = [s for s in suggested
                    if policy.accept_suggestions and s.suggested.startswith("high")]
        for s in suggested:
            if s not in accepted:
                problems.append(f"agent suggestion {cards[s.card_id].name} is "
                                f"{s.suggested.split(':')[0]}")
        members = detected + accepted
        want = expected.get(pack)
        if want is None:
            problems.append("not a pack detection grouped (a card shown on its own?)")
        elif len(members) != want:
            problems.append(f"{len(members)} cards, expected {want}")

        reverse: dict[int, bool] | None = None
        if not problems:
            if accepted:  # pack completed by the agent: re-infer slots for all 10
                reverse = _layout_reverse(members, cards, pack, set_layout)
                if reverse is None and policy.require_layout:
                    problems.append("with the agent's cards the pack doesn't fit the "
                                    "slot layout")
            elif layouts.get(pack) is None and policy.require_layout:
                problems.append("no slot layout, so reverse holos are unknown")

        if problems:
            result.exceptions.extend(f"pack {pack}: {why}" for why in problems)
            for p in pulls:
                p.auto_note = "; ".join(problems)
            continue
        for p in members:
            p.status = Status.CONFIRMED
            p.decided_by = "auto"
            if reverse is not None:
                p.reverse_holo = reverse[p.id]
            p.auto_note = "pack complete, layout fits, every card clear"
            result.confirmed += 1

    if mode == "off":
        _settle_rest(state)
    return result


def _settle_rest(state: ReviewState) -> None:
    """`off` mode: no human is coming, so decide what's left the cautious way."""
    for p in state.pulls:
        if p.status != Status.PENDING:
            continue
        p.decided_by = "auto"
        if p.detected_index is not None:
            p.status = Status.CONFIRMED
            p.auto_note = f"UNCHECKED, kept detection's best guess ({p.auto_note})"
        else:
            p.status = Status.REJECTED
            p.auto_note = f"UNCHECKED, agent suggestion dropped ({p.auto_note})"


def info(result: AutoResult, mode: str, policy: AutoPolicy, completed: bool) -> AutoReviewInfo:
    return AutoReviewInfo(mode=mode, policy=policy.describe(), confirmed=result.confirmed,
                          exceptions=result.exceptions, completed_by_auto=completed)


def write_shadow(video_dir: Path, result: AutoResult, policy: AutoPolicy) -> Path:
    path = video_dir / SHADOW_FILE
    result.state.auto = info(result, "shadow", policy, completed=False)
    path.write_text(result.state.model_dump_json(indent=2), encoding="utf-8")
    return path


def run(store: ReviewStore, slug: str, mode: str, policy: AutoPolicy) -> AutoResult | None:
    """The pipeline step: auto-review a video that has no review yet.

    Never touches a review a person has started (or one already auto-reviewed), so it
    is safe to repeat. Returns None when there was nothing to do.
    """
    if mode == "always":
        return None
    root = store.video_dir(slug)
    if (root / "review.json").exists() or (mode == "shadow" and (root / SHADOW_FILE).exists()):
        return None
    det = store.detection(slug)
    state = store.initial_state(slug)
    result = auto_review(det, state, store.cards(det.set_id), policy, mode)
    if mode == "shadow":
        write_shadow(root, result, policy)
        return result
    done = not any(p.status == Status.PENDING for p in state.pulls)
    state.completed = done
    state.auto = info(result, mode, policy, completed=done)
    store.save(state)
    return result
