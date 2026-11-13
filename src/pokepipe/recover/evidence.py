"""The recovery agent's tools: deterministic evidence about short packs.

Detection drops a card when its frames never clear the vote thresholds. That happens
on near-identical prints (Enamorus Uncommon vs its Illustration Rare: lead 0.007
< 0.03), on borderline frames, and when a second copy of a card in a reverse slot is
merged into the first as a "repeat showing". Every miss on the Delta Reign test video
was one of these, and each left a pack short of 10.

None of this code calls a model. The tools measure and the agent decides: which window
to rescan, whether a near-tie fits the pack layout, whether a repeat is really a second
copy. Everything returned is plain JSON, and only scores, names and times leave the
machine, never frames.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from pokepipe.cards.models import Card
from pokepipe.cards.service import is_decoy
from pokepipe.detection.layout import PackLayout, apply_layout, placements
from pokepipe.detection.matcher import FrameResult
from pokepipe.detection.models import Pack, Pull, PullsFile
from pokepipe.detection.segment import SegmentParams, segment

# Looser than detection's defaults (0.68 / 0.03 / 0.72 / 3 frames): a near-tie never
# leads by 0.03, so we keep every frame and let the agent weigh the tie.
NEAR_MISS = SegmentParams(
    frame_min_score=0.62, frame_min_margin=0.0, max_gap=0.75, min_frames=2, min_peak=0.66
)
MAX_SCAN_SECONDS = 90.0


class Scanner(Protocol):
    """Re-matches a stretch of the video (GPU). Faked in tests."""

    def frames(self, start: float, end: float, fps: float) -> list[FrameResult]: ...


# --- pure helpers ----------------------------------------------------------------------


@dataclass
class NearMiss:
    start: float
    end: float
    frames: list[FrameResult] = field(default_factory=list)

    def candidates(self) -> list[tuple[str, float, float]]:
        """(card_id, best score, share of frames where it was the top match)."""
        best: dict[str, float] = defaultdict(float)
        wins: dict[str, int] = defaultdict(int)
        for f in self.frames:
            wins[f.best.card_id] += 1
            for m in f.top:
                best[m.card_id] = max(best[m.card_id], m.score)
        n = max(len(self.frames), 1)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        return [(c, s, wins[c] / n) for c, s in ranked]


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def near_misses(frames: list[FrameResult], params: SegmentParams = NEAR_MISS) -> list[NearMiss]:
    """Card-like stretches of frames, with near-ties kept together.

    Segmenting by top card splits a near-tie into two interleaved segments (U, IR, U,
    IR...). Segments that overlap by at least 30% of the shorter one are one physical
    card, so they are merged and their candidates compared side by side.
    """
    segs = sorted(segment(frames, params), key=lambda s: s.start)
    clusters: list[NearMiss] = []
    for s in segs:
        span = (s.start, s.end)
        last = clusters[-1] if clusters else None
        shorter = max(min(s.end - s.start, (last.end - last.start) if last else 0), 0.25)
        if last and _overlap((last.start, last.end), span) >= 0.3 * shorter:
            last.end = max(last.end, s.end)
        else:
            clusters.append(NearMiss(s.start, s.end))
    for c in clusters:
        c.frames = [f for f in frames if c.start <= f.time <= c.end]
    return clusters


def synthetic_pull(card: Card, pack: int, time: float) -> Pull:
    """A placeholder pull for 'what if this card were here' layout checks."""
    return Pull(card_id=card.id, name=card.name, number=card.number, rarity=card.rarity,
                start=time, end=time, peak_time=time, peak_score=0.0, mean_margin=0.0,
                frames=0, confidence=0.0, box=(0, 0, 0, 0), pack=pack)


# --- the tools -------------------------------------------------------------------------


class ToolError(ValueError):
    """Bad tool input. Reported back to the agent as an error result, not raised."""


@dataclass
class Evidence:
    det: PullsFile
    cards: dict[str, Card]
    layout: PackLayout | None
    scanner: Scanner | None = None
    scans_left: int = 8

    # helpers

    def _pack(self, index: int) -> Pack:
        pack = next((p for p in self.det.packs if p.index == index), None)
        if pack is None:
            raise ToolError(f"no pack {index}; packs are 1-{len(self.det.packs)}")
        return pack

    def _pulls(self, index: int) -> list[Pull]:
        return sorted((p for p in self.det.pulls if p.pack == index), key=lambda p: p.start)

    def _card(self, card_id: str) -> Card:
        card = self.cards.get(card_id)
        if card is None:
            raise ToolError(f"{card_id!r} is not a card in {self.det.set_id}; use find_cards")
        return card

    def _describe(self, card_id: str) -> dict:
        if is_decoy(card_id):
            return {"card_id": card_id, "name": "basic Energy (not a pull)"}
        c = self.cards.get(card_id)
        return {"card_id": card_id, "name": c.name if c else card_id,
                "number": c.number if c else None, "rarity": c.rarity if c else None}

    def default_window(self, index: int) -> tuple[float, float]:
        """The pack's reveals plus a margin, never running into the next pack."""
        pack = self._pack(index)
        later = [p.start for p in self.det.packs if p.start > pack.start]
        end = min([pack.end + 6.0] + [s - 0.5 for s in later])
        return max(0.0, pack.start - 3.0), end

    # tools exposed to the agent

    def short_packs(self) -> dict:
        out = []
        for p in self.det.packs:
            if p.cards >= p.expected_cards:
                continue
            out.append({
                "pack": p.index, "found": p.cards, "expected": p.expected_cards,
                "start": round(p.start, 1), "end": round(p.end, 1),
                "layout_note": p.layout_note,
                # A handful of cards shown before the first pack (a preview) is not a pack.
                "probably_not_a_pack": p.cards < p.expected_cards // 2,
            })
        return {"layout": self.layout.name if self.layout else None, "short_packs": out}

    def pack_overview(self, pack: int) -> dict:
        info, pulls = self._pack(pack), self._pulls(pack)
        rows = []
        for p in pulls:
            lead = p.peak_score - p.alternatives[0].score if p.alternatives else None
            rows.append({
                "time": f"{p.start:.1f}-{p.end:.1f}", **self._describe(p.card_id),
                "slot": p.slot, "reverse_guess": p.reverse_guess,
                "peak_score": p.peak_score,
                "lead_over_runner_up": None if lead is None else round(lead, 3),
                "runner_ups": [a.card_id for a in p.alternatives[:3]],
                # A repeat inside this pack's window may really be a second copy.
                "shown_again_in_pack": [
                    round(s.peak_time, 1) for s in p.showings
                    if info.start - 1 <= s.peak_time <= info.end + 1
                ],
            })
        gaps = [
            {"between_cards": [i + 1, i + 2], "from": round(a.end, 1), "to": round(b.start, 1),
             "seconds": round(b.start - a.end, 1)}
            for i, (a, b) in enumerate(zip(pulls, pulls[1:], strict=False))
        ]
        start, end = self.default_window(pack)
        return {
            "pack": pack, "found": len(pulls), "expected": info.expected_cards,
            "cards_in_reveal_order": rows, "gaps": gaps,
            "missing_slots": self._missing_slots(pulls),
            "default_scan_window": [round(start, 1), round(end, 1)],
        }

    def _missing_slots(self, pulls: list[Pull]) -> list[dict] | str:
        if self.layout is None:
            return "no verified pack layout for this set"
        placed = placements(pulls, self.layout)
        if not placed:
            return "the detected cards don't fit the pack order (shuffled or misread)"
        used = [set(c) for c in placed]
        out = []
        for s, slot in enumerate(self.layout.slots):
            empty_in = sum(s not in u for u in used)
            if empty_in:
                allowed = sorted(slot.allowed) + (["any special rarity"] if slot.allow_special
                                                  else [])
                out.append({"slot": s + 1, "kind": slot.kind, "allowed_rarities": allowed,
                            "certain": empty_in == len(used)})
        return out

    def scan_window(self, pack: int, start: float | None, end: float | None,
                    fps: float) -> dict:
        if self.scanner is None:
            raise ToolError("video rescans are not available in this run")
        if self.scans_left <= 0:
            raise ToolError("scan limit reached; decide with the evidence you have")
        lo, hi = self.default_window(pack)
        start = lo if start is None else max(0.0, start)
        end = hi if end is None else end
        if end <= start:
            raise ToolError("end must be after start")
        end = min(end, start + MAX_SCAN_SECONDS)
        fps = min(max(fps, 2.0), 12.0)
        self.scans_left -= 1

        frames = self.scanner.frames(start, end, fps)
        detected = [p for p in self.det.pulls if p.end >= start and p.start <= end]
        stretches = []
        for nm in near_misses(frames):
            cands = nm.candidates()[:4]
            top = cands[0][0]
            same = next((p for p in detected if p.card_id == top and
                         _overlap((p.start - 0.5, p.end + 0.5), (nm.start, nm.end)) > 0), None)
            stretches.append({
                "time": f"{nm.start:.2f}-{nm.end:.2f}", "frames": len(nm.frames),
                "candidates": [{**self._describe(c), "best_score": round(s, 3),
                                "top_in_share_of_frames": round(w, 2)} for c, s, w in cands],
                "lead": round(cands[0][1] - cands[1][1], 3) if len(cands) > 1 else None,
                "already_detected_as": same.card_id if same else None,
                "is_basic_energy": is_decoy(top),
            })
        return {"window": [round(start, 2), round(end, 2)], "fps": fps,
                "frames_scanned": len(frames), "scans_left": self.scans_left,
                "card_like_stretches": stretches}

    def check_fit(self, pack: int, additions: list[tuple[str, float]]) -> dict:
        """Would the pack fit its slot order with these (card_id, time) added together?
        Several at once, because two missed cards can share one stretch (both
        reverse slots held up side by side)."""
        self._pack(pack)
        if not additions:
            raise ToolError("give at least one card to add")
        if self.layout is None:
            return {"fits": None, "note": "no verified pack layout for this set"}
        trial = [p.model_copy() for p in self._pulls(pack)]
        new = [synthetic_pull(self._card(cid), pack, t) for cid, t in additions]
        result = apply_layout([*trial, *new], self.layout)
        return {
            "fits": result.mode is not None, "mode": result.mode, "note": result.note,
            "pack_would_have": len(trial) + len(new),
            "added": [{**self._describe(n.card_id), "time": n.start, "slot": n.slot,
                       "reverse_holo": n.reverse_guess,
                       "already_in_pack_at": [round(p.start, 1) for p in trial
                                              if p.card_id == n.card_id]}
                      for n in new],
        }

    def find_cards(self, query: str) -> dict:
        q = query.strip().lower()
        hits = [c for c in self.cards.values()
                if q in c.name.lower() or q == c.number.lower() or q == c.id.lower()]
        return {"cards": [self._describe(c.id) for c in hits[:12]],
                "more": max(0, len(hits) - 12)}
