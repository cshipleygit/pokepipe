"""Review state persistence and edit rules, independent of the web layer."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

from pokepipe.cards.cache import SetCache
from pokepipe.cards.models import Card
from pokepipe.detection.models import PullsFile
from pokepipe.review.models import (
    NewPull,
    PullUpdate,
    ReviewedPull,
    ReviewState,
    Status,
    _now,
)

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")


class ReviewError(ValueError):
    """A request the review rules reject (shown to the user as a 4xx)."""


class NotFound(ReviewError):
    pass


@dataclass(frozen=True)
class PackSummary:
    index: int
    cards: int  # confirmed + still-pending pulls (rejected ones don't count)
    pending: int
    expected: int

    @property
    def complete(self) -> bool:
        return self.cards == self.expected


class ReviewStore:
    def __init__(self, videos_root: Path, sets_root: Path) -> None:
        self.videos_root = videos_root
        self.sets_root = sets_root
        self._lock = threading.Lock()  # FastAPI runs sync endpoints in a thread pool
        self._cards: dict[str, dict[str, Card]] = {}

    # --- paths & loading -----------------------------------------------------------

    def video_dir(self, slug: str) -> Path:
        if not SLUG_RE.match(slug):
            raise NotFound(f"bad video id {slug!r}")
        path = self.videos_root / slug
        if not (path / "pulls.json").exists():
            raise NotFound(f"no detection results for {slug!r}")
        return path

    def slugs(self) -> list[str]:
        if not self.videos_root.exists():
            return []
        return sorted(
            p.name for p in self.videos_root.iterdir()
            if (p / "pulls.json").exists() and SLUG_RE.match(p.name)
        )

    def detection(self, slug: str) -> PullsFile:
        text = (self.video_dir(slug) / "pulls.json").read_text(encoding="utf-8")
        return PullsFile.model_validate_json(text)

    def cards(self, set_id: str) -> dict[str, Card]:
        if set_id not in self._cards:
            cache = SetCache(self.sets_root, set_id)
            if not cache.exists():
                raise NotFound(f"set {set_id!r} is not cached")
            self._cards[set_id] = {c.id: c for c in cache.load_cards()}
        return self._cards[set_id]

    def load(self, slug: str) -> ReviewState:
        path = self.video_dir(slug) / "review.json"
        if path.exists():
            return ReviewState.model_validate_json(path.read_text(encoding="utf-8"))
        return self._initial_state(slug)

    def _initial_state(self, slug: str) -> ReviewState:
        det = self.detection(slug)
        pulls = [
            ReviewedPull(
                id=i, detected_index=i, detected_card_id=p.card_id, card_id=p.card_id,
                pack=p.pack or 1, time=p.peak_time,
            )
            for i, p in enumerate(det.pulls)
        ]
        return ReviewState(video_slug=slug, set_id=det.set_id, pulls=pulls)

    def save(self, state: ReviewState) -> None:
        state.updated_at = _now()
        path = self.video_dir(state.video_slug) / "review.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic: a crash never leaves a half-written review

    # --- queries -------------------------------------------------------------------

    def pack_summaries(self, state: ReviewState, expected: int) -> list[PackSummary]:
        packs = sorted({p.pack for p in state.pulls})
        out = []
        for idx in packs:
            members = [p for p in state.pulls if p.pack == idx and p.status != Status.REJECTED]
            pending = sum(p.status == Status.PENDING for p in members)
            out.append(PackSummary(idx, len(members), pending, expected))
        return out

    # --- edits (each loads, applies, saves under the lock) ------------------------------

    def _edit(self, slug: str, fn):
        with self._lock:
            state = self.load(slug)
            result = fn(state)
            state.completed = False  # any edit reopens a finished review
            self.save(state)
            return result

    def _check_card(self, state: ReviewState, card_id: str) -> None:
        if card_id not in self.cards(state.set_id):
            raise ReviewError(f"{card_id!r} is not a card in set {state.set_id}")

    def update(self, slug: str, pull_id: int, change: PullUpdate) -> ReviewedPull:
        def apply(state: ReviewState) -> ReviewedPull:
            pull = next((p for p in state.pulls if p.id == pull_id), None)
            if pull is None:
                raise NotFound(f"no pull {pull_id}")
            if change.card_id is not None:
                self._check_card(state, change.card_id)
                pull.card_id = change.card_id
                # picking a card is a decision: treat it as confirming unless told otherwise
                if change.status is None and pull.status == Status.PENDING:
                    pull.status = Status.CONFIRMED
            if change.status is not None:
                pull.status = change.status
            if change.reverse_holo is not None:
                pull.reverse_holo = change.reverse_holo
            if change.pack is not None:
                if change.pack < 1:
                    raise ReviewError("pack must be >= 1")
                pull.pack = change.pack
            return pull

        return self._edit(slug, apply)

    def add(self, slug: str, new: NewPull) -> ReviewedPull:
        """A card detection missed. Added pulls are confirmed by definition."""
        def apply(state: ReviewState) -> ReviewedPull:
            self._check_card(state, new.card_id)
            if new.pack < 1:
                raise ReviewError("pack must be >= 1")
            pull = ReviewedPull(
                id=max((p.id for p in state.pulls), default=-1) + 1,
                card_id=new.card_id, pack=new.pack, time=new.time,
                status=Status.CONFIRMED, reverse_holo=new.reverse_holo,
            )
            state.pulls.append(pull)
            return pull

        return self._edit(slug, apply)

    def remove(self, slug: str, pull_id: int) -> None:
        """Only reviewer-added pulls can be deleted; detected ones are rejected instead,
        which keeps the detection-vs-review record intact."""
        def apply(state: ReviewState) -> None:
            pull = next((p for p in state.pulls if p.id == pull_id), None)
            if pull is None:
                raise NotFound(f"no pull {pull_id}")
            if pull.detected_index is not None:
                raise ReviewError("detected pulls can't be deleted — reject them instead")
            state.pulls.remove(pull)

        self._edit(slug, apply)

    def confirm_pack(self, slug: str, pack: int) -> int:
        def apply(state: ReviewState) -> int:
            todo = [p for p in state.pulls if p.pack == pack and p.status == Status.PENDING]
            for p in todo:
                p.status = Status.CONFIRMED
            return len(todo)

        return self._edit(slug, apply)

    def complete(self, slug: str) -> ReviewState:
        with self._lock:
            state = self.load(slug)
            pending = sum(p.status == Status.PENDING for p in state.pulls)
            if pending:
                raise ReviewError(f"{pending} pulls still pending")
            state.completed = True
            self.save(state)
            return state
