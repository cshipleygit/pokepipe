"""On-screen text as an ASS subtitle script, burned in by ffmpeg/libass.

ASS gives styled, positioned, timed text without compositing images ourselves:
- word-by-word captions (the spoken word highlighted) from ElevenLabs word timings
- the hook caption for the first seconds
- a master-set progress bar (a rocket's flame trail) that fills at the end of each pack,
  with a "+N NEW" badge; the rocket itself is a PNG overlay (`render.theme`) that
  `rocket_keys` moves along with the fill
- a "PACK 3 OF 9" title over each pack's tear
- a rarity label while a big hit is zoomed in

Look: the rocket-launch theme in `render.theme` (chunky font, navy outlines, flame
orange, mint and sunny yellow). Layout respects TikTok's UI: nothing in the right-hand
button column, the bottom ~20% (the post's caption, username and sound), or the top
~9% (the feed tabs). Spoken captions sit at about 79% of the height: under the card,
just above TikTok's text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from pokepipe.narration.models import TimedWord
from pokepipe.render import theme
from pokepipe.render.timeline import OUT_H, OUT_W, PlacedLine

WHITE = theme.ass(theme.WHITE)
HIGHLIGHT = theme.ass(theme.ORANGE)  # the word being spoken
NAVY = theme.ass(theme.NAVY)
SHADE = theme.ass(theme.NAVY, alpha=0x50)
FONT = theme.FONT

STYLES = [
    # name, size, primary, outline colour, back colour, border style, outline, shadow,
    # alignment (numpad), marginL, marginR, marginV
    ("Caption", 92, WHITE, NAVY, SHADE, 1, 8, 3, 2, 90, 170, 400),
    ("Hook", 66, WHITE, SHADE, SHADE, 3, 20, 0, 8, 90, 170, 430),
    ("Counter", 46, WHITE, NAVY, SHADE, 1, 5, 2, 7, 60, 60, 270),
    ("Badge", 48, theme.ass(theme.BADGE_COLOUR), NAVY, SHADE, 1, 5, 2, 7, 60, 60, 330),
    ("Label", 60, theme.ass(theme.LABEL_COLOUR), NAVY, SHADE, 1, 6, 3, 8, 90, 170, 430),
    ("Title", 80, WHITE, NAVY, SHADE, 1, 8, 3, 8, 90, 170, 430),
    ("Price", 84, theme.ass(theme.PRICE_COLOUR), NAVY, SHADE, 1, 8, 3, 8, 90, 170, 505),
    ("Bar", 10, WHITE, NAVY, NAVY, 1, 0, 0, 7, 0, 0, 0),
]

# Master-set progress bar (PlayRes pixels): below the feed tabs, clear of the buttons.
BAR_X, BAR_Y, BAR_W, BAR_H = 100, 232, 580, 36
BAR_TRACK = theme.ass(theme.NAVY, alpha=theme.BAR_TRACK_ALPHA)  # translucent navy pill
BAR_FILL = theme.ass(theme.ORANGE)  # the flame trail behind the rocket
BAR_TICK = 0.7  # seconds for the fill (and the rocket) to move when a pack is done

# "+N NEW" starts here (after the theme's badge icon, if it has one).
BADGE_X = BAR_X + BAR_W + 34
# Price tag: (icon +) price, centred where the Price style's margins centre text.
PRICE_CX = (90 + OUT_W - 170) / 2
PRICE_CY = 505 + 46

def ts(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def esc(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


@dataclass(frozen=True)
class Overlay:
    hook: str
    hook_seconds: float
    master_total: int
    master_before: int
    master_after_pack: dict[int, int]  # pack -> master owned after it
    new_per_pack: dict[int, int]
    pack_done_at: dict[int, float]  # when each pack's new count should show
    labels: list[tuple[float, float, str]]  # (start, end, text) for big hits
    titles: tuple[tuple[float, float, str], ...] = ()  # (start, end, "PACK 3 OF 9")
    prices: tuple[tuple[float, float, str], ...] = ()  # (start, end, "$31.11") on hits


def caption_chunks(words: list[TimedWord], max_words: int = 3) -> list[list[TimedWord]]:
    """Short chunks, breaking after punctuation so captions follow the phrasing."""
    chunks: list[list[TimedWord]] = []
    cur: list[TimedWord] = []
    for w in words:
        cur.append(w)
        if len(cur) == max_words or w.text[-1:] in ".,!?;:":
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def words_for_line(words: list[TimedWord], placed: PlacedLine) -> list[TimedWord]:
    """This line's words, re-timed onto the output timeline."""
    lo, hi = placed.line.src_start - 0.05, placed.line.src_end + 0.05
    return [
        TimedWord(
            text=w.text, start=round(placed.shift(w.start), 3), end=round(placed.shift(w.end), 3)
        )
        for w in words
        if lo <= w.start <= hi
    ]


def _fill_x(owned: int, total: int) -> int:
    return BAR_X + round(BAR_W * min(owned, total) / total)


def _pill(w: int, h: int) -> str:
    """A rounded bar as ASS drawing commands (each end a half circle of 2 béziers)."""
    r, k = h / 2, 0.5523 * h / 2
    return (f"m {r:.1f} 0 l {w - r:.1f} 0 "
            f"b {w - r + k:.1f} 0 {w:.1f} {r - k:.1f} {w:.1f} {r:.1f} "
            f"b {w:.1f} {r + k:.1f} {w - r + k:.1f} {h:.1f} {w - r:.1f} {h:.1f} "
            f"l {r:.1f} {h:.1f} "
            f"b {r - k:.1f} {h:.1f} 0 {r + k:.1f} 0 {r:.1f} "
            f"b 0 {r - k:.1f} {r - k:.1f} 0 {r:.1f} 0")


def progress_track() -> str:
    """The empty bar: a translucent navy pill drawn with ASS vector commands."""
    return (rf"{{\an7\pos({BAR_X},{BAR_Y})\bord{theme.BAR_RIM_WIDTH}\shad0"
            rf"\1c&H{BAR_TRACK[4:]}&\1a&H{BAR_TRACK[2:4]}&"
            rf"\3c{theme.ass(theme.BAR_RIM)}\p1}}{_pill(BAR_W, BAR_H)}{{\p0}}")


def progress_fill(owned_from: int, owned_to: int, total: int) -> str:
    """The flame trail: a full-width pill clipped to the progress. When the count
    changes, the clip animates from `owned_from` to `owned_to` (libass \t)."""
    y2 = BAR_Y + BAR_H
    grow = ""
    if owned_to != owned_from:
        to = rf"\clip({BAR_X},{BAR_Y},{_fill_x(owned_to, total)},{y2})"
        grow = rf"\t(0,{int(BAR_TICK * 1000)},{to})"
    clip = rf"\clip({BAR_X},{BAR_Y},{_fill_x(owned_from, total)},{y2})"
    return (rf"{{\an7\pos({BAR_X},{BAR_Y})\bord0\shad0\1c{BAR_FILL}{clip}{grow}\p1}}"
            rf"{_pill(BAR_W, BAR_H)}{{\p0}}")


def progress_label(owned: int, total: int) -> str:
    pct = round(100 * owned / total)
    dx, dy = theme.COUNTER_OFFSET
    return (rf"{{\an1\pos({BAR_X + dx},{BAR_Y + dy})}}MASTER SET  {{\c{HIGHLIGHT}}}{owned}"
            rf"{{\c{WHITE}}}/{total}  {pct}%")


def badge(new: int, advance: float = 0) -> str:
    return rf"{{\an4\pos({BADGE_X + advance:g},{BAR_Y + BAR_H // 2})}}+{new} NEW"


def shape(points: list[tuple[float, float]], colour: str, border: float = 0,
          extra: str = "") -> str:
    """A filled polygon as an ASS drawing at absolute PlayRes coordinates (for theme
    icons). Written relative to its own top-left corner and placed there with an7/pos,
    so it lands in the same spot whether libass aligns a drawing by its origin or by its
    bounding box."""
    x0 = min(x for x, _ in points)
    y0 = min(y for _, y in points)
    rel = [f"{x - x0:.1f} {y - y0:.1f}" for x, y in points]
    path = f"m {rel[0]} l {' '.join(rel[1:])}"
    return (rf"{{\an7\pos({x0:.1f},{y0:.1f})\bord{border:.1f}\shad0\1c{colour}\3c{NAVY}"
            rf"{extra}\p1}}{path}{{\p0}}")


def circle(cx: float, cy: float, r: float, n: int = 48) -> list[tuple[float, float]]:
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
            for k in range(n)]


def rocket_keys(
    overlay: Overlay, duration: float, offset: float = 0.0
) -> list[tuple[float, float]]:
    """(time, x) for the rocket's left edge: parked at the fill's tip, flying forward
    with the fill each time a pack is done."""
    total = max(overlay.master_total, 1)
    nose = theme.ROCKET_W * 0.6  # the rocket sits mostly ahead of the trail's end

    def x(owned: int) -> float:
        return _fill_x(owned, total) - nose

    value = overlay.master_before
    keys = [(offset, x(value))]
    for pack in sorted(overlay.pack_done_at):
        end = overlay.pack_done_at[pack]
        after = overlay.master_after_pack.get(pack, value)
        if after != value:
            keys += [(end, x(value)), (min(end + BAR_TICK, duration), x(after))]
        value = after
    keys.append((max(duration, keys[-1][0]), x(value)))
    return keys


ROCKET_Y = BAR_Y + BAR_H / 2 - theme.ROCKET_H / 2


def build_ass(
    lines: list[PlacedLine],
    words: list[TimedWord],
    overlay: Overlay,
    duration: float,
    offset: float = 0.0,
    fonts_dir: Path | None = None,
) -> str:
    """The whole ASS script. `fonts_dir` (the theme's font files) only sharpens the
    price-tag layout when the theme has a price icon: it needs the price's width."""
    events: list[str] = []

    def add(start: float, end: float, style: str, text: str, layer: int = 0) -> None:
        if end - start > 0.01:
            events.append(f"Dialogue: {layer},{ts(start)},{ts(end)},{style},,0,0,0,,{text}")

    # Word-by-word captions: each word gets an event showing its chunk, itself highlighted.
    # A chunk's last word lingers 0.15 s, but never past the next caption's start: with
    # lines back to back, two captions would otherwise stack on screen.
    captions: list[tuple[float, float, str]] = []
    for placed in lines:
        for chunk in caption_chunks(words_for_line(words, placed)):
            for i, w in enumerate(chunk):
                end = chunk[i + 1].start if i + 1 < len(chunk) else w.end + 0.15
                text = " ".join(
                    (
                        f"{{\\c{HIGHLIGHT}}}{esc(x.text.upper())}{{\\c{WHITE}}}"
                        if j == i
                        else esc(x.text.upper())
                    )
                    for j, x in enumerate(chunk)
                )
                captions.append((w.start, end, text))
    captions.sort()
    for k, (start, end, text) in enumerate(captions):
        if k + 1 < len(captions):
            end = min(end, captions[k + 1][0])
        add(start, end, "Caption", text)

    # Hook at the top for the opening seconds.
    if overlay.hook:  # no hook without a script (narration-free render)
        add(offset, offset + overlay.hook_seconds, "Hook", esc(overlay.hook.upper()))

    # Master-set progress bar: fills at each pack's end, with a "+N NEW" badge.
    total = max(overlay.master_total, 1)
    add(offset, duration, "Bar", progress_track(), layer=0)
    t, value = offset, overlay.master_before
    for pack in sorted(overlay.pack_done_at):
        end = overlay.pack_done_at[pack]
        nxt = overlay.master_after_pack.get(pack, value)
        add(t, end, "Bar", progress_fill(value, value, total), layer=1)
        add(t, end, "Counter", progress_label(value, total))
        new = overlay.new_per_pack.get(pack, 0)
        if new:
            gone = min(end + 1.6, duration)
            advance = 0.0
            if theme.badge_icon is not None:
                drawings, advance = theme.badge_icon(BADGE_X, BAR_Y + BAR_H / 2)
                for k, drawing in enumerate(drawings):
                    add(end, gone, "Badge", drawing, layer=2 + k)
            add(end, gone, "Badge", badge(new, advance), layer=2 if advance else 0)
        value_from, value, t = value, nxt, end
        grow = min(end + BAR_TICK, duration)
        add(end, grow, "Bar", progress_fill(value_from, value, total), layer=1)
        add(end, grow, "Counter", progress_label(value, total))
        t = grow
    add(t, duration, "Bar", progress_fill(value, value, total), layer=1)
    add(t, duration, "Counter", progress_label(value, total))

    # Pack titles over each pack's tear.
    for start, end, text in overlay.titles:
        add(start, end, "Title", esc(text))

    # Rarity label during big-hit zooms (after the hook has gone).
    hook_end = offset + overlay.hook_seconds if overlay.hook else offset
    for start, end, text in overlay.labels:
        add(max(start, hook_end), end, "Label", esc(text))

    # Price tags pop in (a little overshoot) under the rarity label while a hit is zoomed.
    pop = r"{\fscx40\fscy40\t(0,160,\fscx115\fscy115)\t(160,280,\fscx100\fscy100)}"
    for start, end, text in overlay.prices:
        start = max(start, hook_end)
        if theme.price_icon is None:
            add(start, end, "Price", pop + esc(text))
            continue
        # icon then price, the pair centred on the tag's column
        icon_w, gap = theme.PRICE_ICON_W, theme.PRICE_ICON_GAP
        left = PRICE_CX - (icon_w + gap + theme.text_width(text, 84, fonts_dir)) / 2
        for k, drawing in enumerate(theme.price_icon(left, PRICE_CY, r"\fad(120,0)")):
            add(start, end, "Price", drawing, layer=2 + k)
        at = rf"{{\an4\pos({left + icon_w + gap:.0f},{PRICE_CY:.0f})}}"
        add(start, end, "Price", at + pop + esc(text), layer=2)

    style_lines = [
        f"Style: {n},{FONT},{size},{prim},{prim},{outl},{back},-1,0,0,0,100,100,0,0,"
        f"{border},{outline},{shadow},{align},{ml},{mr},{mv},1"
        for n, size, prim, outl, back, border, outline, shadow, align, ml, mr, mv in STYLES
    ]
    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {OUT_W}",
            f"PlayResY: {OUT_H}",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            *style_lines,
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            *events,
            "",
        ]
    )
