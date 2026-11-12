"""Editing moments the creator films on purpose, found automatically.

Besides the card reveals, an opening video has footage worth cutting to: the sealed
Elite Trainer Box, ripping its seal, lifting the lid, the packs fanned out, and for
every pack the tear and the cards sliding out of the wrapper. There's also footage
worth cutting *out*: between the tear and the first card the opener straightens the
stack, flicks past the code card and sometimes peeks at a card early. Each pack is
trimmed to start where the basic Energy is lined up, just before the first card.

Signals, chosen because they held up on real footage (measured on Allen's Delta Reign
ETB, see CLAUDE.md):
- **Sound for tearing.** The room is otherwise quiet; tearing foil and peeling
  shrink-wrap are loud, broadband crinkles. The last sustained crinkle before a pack's
  Energy is its tear followed by the cards sliding out.
- **Card matching for the Energy.** Basic Energy is already in every index as a decoy,
  so "Energy on screen" is an exact, reliable signal.
- **CLIP zero-shot only where it was reliable:** "an open box with packs inside"
  (0.94-0.98 on the lid-off shot) and "booster packs fanned out". Generic prompts such
  as "a sealed pack" fired on almost everything (the ETB display wall sits behind
  every shot), so they aren't used.

The pure functions here turn those signals into spans; `service.find_moments` gathers
the signals. Everything lands in `moments.json`, which can be checked or hand-edited.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class Span(BaseModel):
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


class PackMoments(BaseModel):
    index: int
    rip: Span | None = None
    cards_out: Span | None = None
    energy_at: float | None = None  # pack footage starts here (Energy lined up)


class IntroMoments(BaseModel):
    hero: Span | None = None  # the sealed box held up
    seal: Span | None = None  # ripping the shrink-wrap off
    open: Span | None = None  # lid off, packs inside
    packs: Span | None = None  # packs fanned out


class SleeveMoment(BaseModel):
    """A card sliding into its sleeve, filmed right after its pack."""

    pack: int
    card_id: str
    clip: Span


class MomentsFile(BaseModel):
    video: str
    intro: IntroMoments = Field(default_factory=IntroMoments)
    packs: list[PackMoments] = Field(default_factory=list)
    sleeves: list[SleeveMoment] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def pack(self, index: int) -> PackMoments | None:
        return next((p for p in self.packs if p.index == index), None)


# --- sound ----------------------------------------------------------------------------------

LOUD_DB = -45.0  # high-passed RMS level of tearing foil / shrink-wrap (quiet room ≈ -75)


def loud_runs(
    db: np.ndarray, hop: float, t0: float = 0.0, threshold: float = LOUD_DB,
    max_gap: float = 0.6, min_len: float = 1.5,
) -> list[Span]:
    """Stretches where the level stays above `threshold`, bridging dips up to `max_gap`."""
    loud = np.flatnonzero(db > threshold)
    runs: list[Span] = []
    if not len(loud):
        return runs
    start = prev = loud[0]
    for i in loud[1:]:
        if (i - prev) * hop > max_gap:
            runs.append(Span(start=t0 + start * hop, end=t0 + (prev + 1) * hop))
            start = i
        prev = i
    runs.append(Span(start=t0 + start * hop, end=t0 + (prev + 1) * hop))
    return [r for r in runs if r.length >= min_len]


def rip_and_cards_out(
    runs: list[Span], before: float, clip: float = 0.9, out_lead: float = 0.8
) -> tuple[Span, Span] | None:
    """The pack's tear (start of the last crinkle before the Energy) and the cards
    coming out (near its end), as two clips of about `clip` seconds. The crinkle tails
    off after the cards are out (the wrapper being put down), so the cards-out clip
    ends `out_lead` before the crinkle does."""
    candidates = [r for r in runs if r.end <= before + 0.5]
    if not candidates:
        return None
    run = candidates[-1]
    if run.length < 2 * clip:  # short crinkle: split it in half
        mid = (run.start + run.end) / 2
        return Span(start=run.start, end=mid), Span(start=mid, end=run.end)
    out_end = max(run.start + 2 * clip, run.end - out_lead)
    return (Span(start=run.start, end=run.start + clip),
            Span(start=out_end - clip, end=out_end))


# --- energy ---------------------------------------------------------------------------------


def energy_start(
    samples: list[tuple[float, bool]], first_card: float, max_gap: float = 0.6
) -> float | None:
    """`samples`: (time, Energy is the best match) before the first card. Returns when
    the Energy shown just before the first card first appeared."""
    hits = [t for t, is_energy in samples if is_energy and t < first_card]
    if not hits:
        return None
    start = hits[-1]
    for t in reversed(hits[:-1]):
        if start - t > max_gap:
            break
        start = t
    return start


# --- sleeving ------------------------------------------------------------------------------

SLEEVE_P = 0.5  # zero-shot "sliding a card into a sleeve" probability


def sleeve_slide(times: np.ndarray, p_sleeve: np.ndarray, clip: float = 2.0) -> Span | None:
    """The card going into its sleeve, in the seconds before the sleeved card is held up.

    Sleeving makes almost no sound (-60 to -80 dB), so this is visual: CLIP's "sliding a
    card into a sleeve" score stays high while the card goes in and drops once it's in.
    The slide ends at the last confident sample that has a confident neighbour (lone
    spikes are glare); the clip is the 2 s up to just after that."""
    ok = p_sleeve >= SLEEVE_P
    for i in range(len(times) - 1, -1, -1):
        near = (i > 0 and p_sleeve[i - 1] >= 0.4) or (i + 1 < len(times) and p_sleeve[i + 1] >= 0.4)
        if ok[i] and near:
            end = float(times[i]) + 0.4  # let the card settle fully into the sleeve
            return Span(start=end - clip, end=end)
    return None


# --- the box --------------------------------------------------------------------------------


def intro_moments(
    times: np.ndarray,
    p_open: np.ndarray,
    p_fan: np.ndarray,
    p_table: np.ndarray,
    runs: list[Span],
    end: float,
) -> IntroMoments:
    """ETB shots before the first pack (`end`). Missing ones stay None and are skipped.

    `p_open`/`p_fan`/`p_table`: per-frame zero-shot probabilities of "an open box with
    packs inside", "booster packs fanned out" and "an empty table"."""
    out = IntroMoments()
    inside = times < end
    open_idx = np.flatnonzero(inside & (p_open >= 0.8))
    open_at = float(times[open_idx[0]]) if len(open_idx) else None
    if open_at is not None:
        out.open = Span(start=max(0.0, open_at - 1.0), end=open_at)
        window = inside & (times > open_at) & (times < open_at + 40)
        if window.any() and p_fan[window].max() >= 0.5:
            t = float(times[window][p_fan[window].argmax()])
            out.packs = Span(start=t - 0.4, end=t + 0.8)
    box_from = next((float(t) for t, p in zip(times, p_table, strict=True) if p < 0.3), None)
    seal_runs = [r for r in runs if r.end <= (open_at or end)]
    if seal_runs:  # the longest crinkle before the lid comes off: peeling the wrap
        run = max(seal_runs, key=lambda r: r.length)
        out.seal = Span(start=run.start + 0.5, end=run.start + 1.5)
    if box_from is not None and (out.seal is None or box_from + 1.5 <= out.seal.start):
        out.hero = Span(start=box_from + 0.5, end=box_from + 1.5)
    return out
