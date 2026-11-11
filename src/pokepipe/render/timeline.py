"""Edit decisions for one post, as pure data: which source ranges, at what speed, framed
how, and where each voice line lands. ffmpeg only executes what this module decides,
so the whole edit is unit-testable without rendering a frame.

Editing rules:
- Hits (the rare slot and anything rarer) play at 1× and get a punch-in zoom on the card.
- Everything else (tear, commons, gaps) shares one speed factor, solved so the post
  lands on its target length; gaps run faster than reveals.
- Each voice line is cut from the voiceover and starts when its card appears on screen.
  Lines only slip later if the previous line hasn't finished.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pokepipe.cards.rarity import is_special

OUT_W, OUT_H = 1080, 1920


@dataclass(frozen=True)
class RenderPull:
    id: str  # same ids as the script anchors: p<pack>-<n>
    pack: int
    name: str
    rarity: str | None
    variant: str
    new: bool
    src_start: float
    src_end: float
    box: tuple[float, float, float, float] | None  # 0-1 of source frame, None if unknown
    # Repeat appearances of this same card (start, end, box): showcase footage.
    showings: tuple[tuple[float, float, tuple[float, float, float, float] | None], ...] = ()
    card_id: str = ""  # final card id (after review), used to track the card in zooms

    @property
    def hit(self) -> bool:
        """The pack's rare and anything rarer get real-time playback and a zoom.
        A Rare in a reverse-holo slot is a reverse holo, not the pack's rare."""
        if self.variant == "reverse":
            return False
        return self.rarity in ("Rare", "Rare Holo") or is_special(self.rarity)

    @property
    def label(self) -> str | None:
        """Big enough to get an on-screen rarity label."""
        return self.rarity.upper() if is_special(self.rarity) else None


@dataclass(frozen=True)
class PackWindow:
    index: int
    clip_start: float
    clip_end: float


@dataclass(frozen=True)
class Clip:
    """A quick 1× shot with its sound up: the box opening, a pack's tear, the cards
    coming out, or a hit in the end recap (`pull_id`, framed on `box`)."""

    start: float
    end: float
    label: str
    pull_id: str | None = None
    box: tuple[float, float, float, float] | None = None


@dataclass
class Segment:
    src_start: float
    src_end: float
    kind: str  # "gap" | "reveal" | "hit" | "zoom" | "clip" (quick 1× shot, see Clip)
    pack: int
    pull_id: str | None = None
    speed: float = 1.0
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)  # w, h, x, y in source pixels
    hold: float = 0.0  # seconds to freeze the last frame (only the final segment)
    out_start: float = 0.0
    showcase: bool = False  # a repeat showing of an already-revealed card
    box: tuple[float, float, float, float] | None = None  # card box, if not the pull's
    # Zoom shots only. `crop` is the static zoom; `window` is the normal framing the zoom
    # happens inside; `path` (set by render.zoom) animates the crop over the shot as
    # [(seconds into the shot's source, (w, h, x, y))]. `chained`: this zoom follows
    # straight on from the previous card's zoom, so it pans across instead of zooming
    # out and back in.
    window: tuple[int, int, int, int] | None = None
    chained: bool = False
    label: str = ""  # clips: what the shot shows ("rip", "cards_out", "recap", ...)
    path: list[tuple[float, tuple[int, int, int, int]]] | None = None

    @property
    def src_len(self) -> float:
        return self.src_end - self.src_start

    @property
    def out_len(self) -> float:
        return self.src_len / self.speed + self.hold

    @property
    def out_end(self) -> float:
        return self.out_start + self.out_len


@dataclass(frozen=True)
class EditParams:
    target: float = 45.0
    reveal_lead: float = 0.3  # show a little before the card is recognised
    reveal_tail: float = 0.4
    min_hit: float = 2.5  # hold hits at least this long (source seconds at 1×)
    max_hit: float = 4.0  # at most this much of a hit plays at 1× (the rest speeds up)
    min_gap: float = 0.5  # shorter gaps between reveals are merged into the previous shot
    settle: float = 1.2  # counter ticks up this long after a pack's last card appears
    gap_factor: float = 2.5  # gaps run this much faster than reveals
    max_speed: float = 4.0  # reveal speed cap (cards must stay readable)
    max_gap_speed: float = 12.0
    gap_cap: float = 1.5  # if still too long, cut each gap to this many output seconds
    punch_in: float = 0.35  # seconds at normal framing before a hit zooms in
    zoom_margin: float = 1.5  # zoomed crop width = card width × this
    max_zoom: float = 1.8  # never punch in further than this vs the normal framing
    beat_gap: float = 0.12  # silence between voice lines
    tail: float = 0.6  # silence after the last line
    max_showings: int = 2  # repeat showings used per card (showcase shots)


FIXED = ("hit", "zoom", "clip")  # always play at 1×

# --- building ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Event:
    """A card on screen: its first reveal, or a repeat showing (showcase)."""

    start: float
    end: float
    pull: RenderPull
    showcase: bool
    box: tuple[float, float, float, float] | None


def _pack_events(pack: PackWindow, pulls: list[RenderPull], p: EditParams):
    """(events inside the pack's clip window, showcase events from elsewhere)."""
    inside, outside = [], []
    for pull in (x for x in pulls if x.pack == pack.index):
        inside.append(_Event(pull.src_start, pull.src_end, pull, False, pull.box))
        for start, end, box in pull.showings[: p.max_showings]:
            event = _Event(start, end, pull, True, box)
            in_window = pack.clip_start <= start < pack.clip_end
            (inside if in_window else outside).append(event)
    key = lambda e: e.start  # noqa: E731
    return sorted(inside, key=key), sorted(outside, key=key)


def _shot(event: _Event, start: float, end: float, p: EditParams) -> tuple[float, float]:
    if event.pull.hit:  # real-time moment on the card, but not the whole time it's held up
        end = min(max(end, start + p.min_hit), start + p.max_hit)
    return start, end


def build_segments(
    packs: list[PackWindow], pulls: list[RenderPull], p: EditParams
) -> list[Segment]:
    segments: list[Segment] = []
    for pack in sorted(packs, key=lambda w: w.index):
        events, elsewhere = _pack_events(pack, pulls, p)
        cursor = pack.clip_start
        for i, ev in enumerate(events):
            next_start = events[i + 1].start if i + 1 < len(events) else pack.clip_end
            start, end = _shot(ev, max(cursor, ev.start - p.reveal_lead),
                               ev.end + p.reveal_tail, p)
            end = min(end, max(next_start - p.reveal_lead, start + 0.5), pack.clip_end)
            last = segments[-1] if segments and segments[-1].pack == pack.index else None
            if last and last.kind != "gap" and start - cursor < p.min_gap:
                last.src_end = start  # absorb a sliver of a gap instead of a tiny clip
            elif start > cursor + 0.05:
                segments.append(Segment(cursor, start, "gap", pack.index))
            if end > start:
                segments.append(Segment(
                    start, end, "hit" if ev.pull.hit else "reveal", pack.index, ev.pull.id,
                    showcase=ev.showcase, box=ev.box,
                ))
                cursor = end
        if pack.clip_end > cursor + 0.05:
            segments.append(Segment(cursor, pack.clip_end, "gap", pack.index))
        # Repeat showings from elsewhere in the video (e.g. an end recap) are spliced in
        # right after their pack.
        for ev in elsewhere:
            start, end = _shot(ev, max(0.0, ev.start - p.reveal_lead), ev.end + p.reveal_tail, p)
            segments.append(Segment(
                start, end, "hit" if ev.pull.hit else "reveal", pack.index, ev.pull.id,
                showcase=True, box=ev.box,
            ))
    return segments


def solve_speeds(segments: list[Segment], p: EditParams) -> float:
    """Pick one reveal speed (gaps run `gap_factor` faster) so the cut hits `p.target`.
    Returns the reveal speed used. Hits always play at 1×."""
    fixed = sum(s.src_len for s in segments if s.kind in FIXED)
    reveal = sum(s.src_len for s in segments if s.kind == "reveal")
    gap = sum(s.src_len for s in segments if s.kind == "gap")
    flexible_out = max(p.target - fixed, 1.0)
    speed = (reveal + gap / p.gap_factor) / flexible_out if (reveal or gap) else 1.0
    speed = min(max(speed, 1.0), p.max_speed)
    for s in segments:
        if s.kind in FIXED:
            s.speed = 1.0
        elif s.kind == "reveal":
            s.speed = speed
        else:
            s.speed = min(speed * p.gap_factor, p.max_gap_speed)
    # Still too long at max speed: trim the start of each gap (keep the lead-in to the card).
    if sum(s.out_len for s in segments) > p.target + 0.5:
        for s in segments:
            if s.kind == "gap" and s.out_len > p.gap_cap:
                s.src_start = s.src_end - p.gap_cap * s.speed
    return speed


def frame_segments(
    segments: list[Segment], pulls: dict[str, RenderPull], src_w: int, src_h: int, p: EditParams
) -> list[Segment]:
    """Vertical 9:16 crops. A card's first hit shot is split into a short normal shot and
    a zoom; a hit straight after another card's zoom skips the normal shot (no zoom out
    and back in). Repeat showings of a card that already zoomed stay at normal framing:
    one zoom per card, and a card held up next to another isn't cut in half."""
    base_h = src_h
    base_w = min(src_w, round(base_h * OUT_W / OUT_H))

    def box_of(s: Segment):
        return s.box or (pulls[s.pull_id].box if s.pull_id else None)

    def centre(seg_index: int) -> tuple[float, float]:
        # Frame on this segment's card, else the next card (so gaps lead into it).
        # A clip of the box or a pack has no card: keep it centred.
        if segments[seg_index].kind == "clip" and not box_of(segments[seg_index]):
            return src_w / 2, src_h / 2
        for s in segments[seg_index:]:
            box = box_of(s)
            if box:
                x1, y1, x2, y2 = box
                return (x1 + x2) / 2 * src_w, (y1 + y2) / 2 * src_h
        return src_w / 2, src_h / 2

    def rect(cx: float, cy: float, w: int, h: int) -> tuple[int, int, int, int]:
        w, h = _even(min(w, src_w)), _even(min(h, src_h))
        x = int(min(max(cx - w / 2, 0), src_w - w))
        y = int(min(max(cy - h / 2, 0), src_h - h))
        return w, h, x, y

    out: list[Segment] = []
    zoomed: set[str] = set()
    for i, seg in enumerate(segments):
        cx, cy = centre(i)
        seg.crop = rect(cx, src_h / 2, base_w, base_h)
        box = box_of(seg)
        if seg.kind != "hit" or not box or seg.pull_id in zoomed:
            out.append(seg)
            continue
        zoomed.add(seg.pull_id)
        # Size the zoom by the card's *width*: a 9:16 crop is narrower than a card's
        # 63:88, so width is what gets cut. The margin also absorbs the detection box
        # being coarse (a grid cell, not the card's outline) and the card drifting in
        # the hand during a 2.5–4 s hold. A card that already fills the frame gets
        # little or no zoom rather than a cropped one.
        x1, y1, x2, y2 = box
        card_w = (x2 - x1) * src_w
        zw = min(base_w, max(card_w * p.zoom_margin, base_w / p.max_zoom))
        zh = zw * OUT_H / OUT_W
        common = dict(pack=seg.pack, pull_id=seg.pull_id, speed=1.0,
                      showcase=seg.showcase, box=seg.box, window=seg.crop)
        zoom_crop = rect(cx, cy, round(zw), round(zh))
        prev = out[-1] if out else None
        if prev and prev.kind == "zoom" and abs(prev.src_end - seg.src_start) < 1e-6:
            out.append(Segment(seg.src_start, seg.src_end, "zoom", crop=zoom_crop,
                               chained=True, **common))
            continue
        split = seg.src_start + min(p.punch_in, seg.src_len / 2)
        out.append(Segment(seg.src_start, split, "hit", crop=seg.crop,
                           **{k: v for k, v in common.items() if k != "window"}))
        out.append(Segment(split, seg.src_end, "zoom", crop=zoom_crop, **common))
    return out


def _even(v: float) -> int:
    return int(v) // 2 * 2


def place_out_times(segments: list[Segment], offset: float = 0.0) -> float:
    t = offset
    for s in segments:
        s.out_start = round(t, 3)
        t += s.out_len
    return round(t, 3)


def reveal_times(segments: list[Segment], pulls: dict[str, RenderPull]) -> dict[str, float]:
    """Output time at which each card is first recognisable on screen (its reveal, not a
    showcase, even if a preview showing comes earlier)."""
    times: dict[str, float] = {}
    for s in segments:
        if s.pull_id and not s.showcase and s.pull_id not in times:
            pull = pulls[s.pull_id]
            at = max(pull.src_start, s.src_start)
            times[s.pull_id] = round(s.out_start + (at - s.src_start) / s.speed, 3)
    return times


def pack_done_times(
    segments: list[Segment], reveal_at: dict[str, float], settle: float
) -> dict[int, float]:
    """When each pack is 'done' for the collection counter: shortly after its last card
    appears (so the new count shows while the rare is still on screen), never after the
    pack's footage ends."""
    ends: dict[int, float] = {}
    last_reveal: dict[int, float] = {}
    for s in segments:
        ends[s.pack] = s.out_end
        if s.pull_id in reveal_at:
            last_reveal[s.pack] = max(last_reveal.get(s.pack, 0.0), reveal_at[s.pull_id])
    return {
        pack: round(min(end, last_reveal.get(pack, end) + settle), 3) for pack, end in ends.items()
    }


# --- voice placement --------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiceLine:
    anchor: str
    text: str
    src_start: float  # in voice.mp3
    src_end: float


@dataclass
class PlacedLine:
    line: VoiceLine
    out_start: float
    drift: float  # how late it starts vs. when its card appears (0 for intro/outro)

    @property
    def out_end(self) -> float:
        return self.out_start + (self.line.src_end - self.line.src_start)

    def shift(self, t: float) -> float:
        """Map a time in voice.mp3 (within this line) to output time."""
        return self.out_start + (t - self.line.src_start)


def place_voice(
    lines: list[VoiceLine],
    reveal_at: dict[str, float],
    video_end: float,
    start: float,
    p: EditParams,
    anchors: set[str] | None = None,
) -> list[PlacedLine]:
    """Lay lines on the timeline. A line about a card (in `anchors`, default all) starts
    when that card appears; other lines follow the previous one."""
    honour = set(reveal_at) if anchors is None else anchors
    placed: list[PlacedLine] = []
    prev_end = start
    used: set[str] = set()
    for line in lines:
        dur = line.src_end - line.src_start
        earliest = prev_end + (p.beat_gap if placed else 0.0)
        synced = line.anchor in honour and line.anchor in reveal_at and line.anchor not in used
        if synced:
            want = reveal_at[line.anchor]
            used.add(line.anchor)
        else:  # intro, outro, or another line about the same card: follow on
            want = earliest
        at = max(want, earliest)
        drift = at - reveal_at[line.anchor] if synced else 0.0
        placed.append(PlacedLine(line, round(at, 3), round(max(drift, 0.0), 3)))
        prev_end = at + dur
    return placed


@dataclass
class Edit:
    segments: list[Segment]
    lines: list[PlacedLine]
    reveal_at: dict[str, float]
    pack_done_at: dict[int, float]
    duration: float
    reveal_speed: float
    notes: list[str] = field(default_factory=list)


# --- narration-driven pacing ------------------------------------------------------------------
#
# The voice decides the pace. The post is cut into regions at each card a line is about;
# region r must last at least as long as the narration spoken in it (R_r). Quiet regions
# run fast, talky ones slow down (never below 1×), and leftover time up to the target is
# shared out by how much each region can still slow down. If a region can't get long
# enough even at 1×, the shot of its hit freezes (an intentional hold on the card), and
# the final region may run on into real footage past the pack's clip window.


@dataclass
class Region:
    segments: list[Segment]
    need: float  # seconds of narration that must fit in this region

    def out(self, speed: float, p: EditParams) -> float:
        total = 0.0
        for s in self.segments:
            if s.kind in FIXED:
                total += s.src_len
            elif s.kind == "reveal":
                total += s.src_len / speed
            else:
                total += s.src_len / min(speed * p.gap_factor, p.max_gap_speed)
        return total

    def apply(self, speed: float, p: EditParams) -> None:
        for s in self.segments:
            s.speed = (
                1.0
                if s.kind in FIXED
                else speed
                if s.kind == "reveal"
                else min(speed * p.gap_factor, p.max_gap_speed)
            )

    def speed_for(self, want: float, p: EditParams) -> float:
        """Highest speed in [1, max_speed] whose output is at least `want`."""
        if self.out(p.max_speed, p) >= want:
            return p.max_speed
        if self.out(1.0, p) <= want:
            return 1.0
        lo, hi = 1.0, p.max_speed  # out() decreases with speed
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if self.out(mid, p) >= want else (lo, mid)
        return lo


def anchor_order(lines: list[VoiceLine], first_seg: dict[str, int]) -> list[str]:
    """Cards the narration syncs to: first mention of each, kept only while they move
    forward through the footage (a line looking back at an earlier card just follows on)."""
    anchors: list[str] = []
    for line in lines:
        idx = first_seg.get(line.anchor)
        if idx is None or line.anchor in anchors:
            continue
        if not anchors or idx > first_seg[anchors[-1]]:
            anchors.append(line.anchor)
    return anchors


def build_regions(
    segments: list[Segment], lines: list[VoiceLine], p: EditParams
) -> tuple[list[Region], list[str]]:
    first_seg: dict[str, int] = {}
    for i, s in enumerate(segments):
        if s.pull_id and not s.showcase and s.pull_id not in first_seg:
            first_seg[s.pull_id] = i
    anchors = anchor_order(lines, first_seg)
    cuts = [0] + [first_seg[a] for a in anchors] + [len(segments)]
    regions = [Region(segments[a:b], 0.0) for a, b in zip(cuts, cuts[1:], strict=False)]

    region = 0
    for line in lines:
        if region < len(anchors) and line.anchor == anchors[region]:
            region += 1
        regions[region].need += (line.src_end - line.src_start) + p.beat_gap
    regions[-1].need += p.tail
    return regions, anchors


def pace_regions(regions: list[Region], p: EditParams, tail_room: float) -> list[str]:
    notes: list[str] = []
    floor = [max(r.need, r.out(p.max_speed, p)) for r in regions]
    ceiling = [r.out(1.0, p) for r in regions]
    slack = p.target - sum(floor)
    if slack > 0:  # share spare time by how much each region can still slow down
        room = [max(0.0, c - f) for c, f in zip(ceiling, floor, strict=True)]
        total_room = sum(room)
        want = [
            f + (slack * r / total_room if total_room else 0.0)
            for f, r in zip(floor, room, strict=True)
        ]
    else:
        want = floor
    for r, w, c in zip(regions, want, ceiling, strict=True):
        r.apply(r.speed_for(w, p), p)
        short = r.need - r.out(1.0, p) if w > c else 0.0
        if short <= 0.05:
            continue
        last = r.segments[-1]
        if r is regions[-1] and tail_room > 0:  # run on into real footage after the pack
            extra = min(short, tail_room)
            last.speed, last.src_end = 1.0, last.src_end + extra
            short -= extra
            notes.append(f"ran the final shot {extra:.1f}s longer for the outro")
        if short > 0.05:
            hold_on = next((s for s in reversed(r.segments) if s.kind == "zoom"), last)
            hold_on.hold = round(hold_on.hold + short, 3)
            notes.append(f"held a frame {short:.1f}s so a line could finish")
    return notes


def plan_edit(
    packs: list[PackWindow],
    pulls: list[RenderPull],
    lines: list[VoiceLine],
    src_size: tuple[int, int],
    p: EditParams | None = None,
    intro: float = 0.0,
    tail_room: float = 0.0,
    lead: list[Clip] | tuple = (),
    recap: list[Clip] | tuple = (),
) -> Edit:
    """`tail_room`: source seconds available after the last pack's clip window, which the
    final shot may run into if the outro needs it. `lead` clips open the post (the box,
    each pack's tear and cards coming out); `recap` clips close it (the hits again)."""
    p = p or EditParams()
    by_id = {x.id: x for x in pulls}
    segments = build_segments(packs, pulls, p)
    if not segments:
        raise ValueError("nothing to render: no footage in the selected packs")
    first, last = segments[0].pack, segments[-1].pack
    segments = (
        [Segment(c.start, c.end, "clip", first, c.pull_id, box=c.box, label=c.label,
                 showcase=c.pull_id is not None) for c in lead]
        + segments
        + [Segment(c.start, c.end, "clip", last, c.pull_id, box=c.box, label=c.label,
                   showcase=True) for c in recap]
    )
    if recap:  # the outro plays over the recap: no running on into later footage
        tail_room = 0.0
    speed = solve_speeds(segments, p)  # global first pass; also trims gaps if far too long
    segments = frame_segments(segments, by_id, *src_size, p)

    notes: list[str] = []
    anchors: list[str] = []
    if lines:
        regions, anchors = build_regions(segments, lines, p)
        notes += pace_regions(regions, p, tail_room)
        reveals = [s.speed for s in segments if s.kind == "reveal"]
        speed = round(sum(reveals) / len(reveals), 2) if reveals else 1.0

    video_end = place_out_times(segments, intro)
    reveal_at = reveal_times(segments, by_id)
    placed = place_voice(lines, reveal_at, video_end, intro, p, set(anchors))
    audio_end = max((x.out_end for x in placed), default=0.0) + p.tail
    if audio_end > video_end + 0.05:  # safety net; pacing should already prevent this
        segments[-1].hold = round(segments[-1].hold + audio_end - video_end, 3)
        notes.append(f"held the last frame {audio_end - video_end:.1f}s for the narration")
        video_end = place_out_times(segments, intro)
    late = [x for x in placed if x.drift > 0.5]
    if late:
        notes.append(
            f"{len(late)} line(s) start >0.5 s after their card (max "
            f"{max(x.drift for x in late):.1f}s)"
        )
    done = pack_done_times(segments, reveal_at, p.settle)
    return Edit(segments, placed, reveal_at, done, video_end, speed, notes)
