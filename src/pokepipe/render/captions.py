"""On-screen text as an ASS subtitle script, burned in by ffmpeg/libass.

ASS gives styled, positioned, timed text without compositing images ourselves:
- word-by-word captions (the spoken word highlighted) from ElevenLabs word timings
- the hook caption for the first seconds
- a master-set counter that ticks up at the end of each pack, with a "+N NEW" badge
- a rarity label while a big hit is zoomed in

Layout respects TikTok's UI: nothing in the right-hand button column, the bottom
caption area, or the very top.
"""

from __future__ import annotations

from dataclasses import dataclass

from pokepipe.narration.models import TimedWord
from pokepipe.render.timeline import OUT_H, OUT_W, PlacedLine

WHITE, YELLOW, GREEN, BLACK = "&H00FFFFFF", "&H0000E5FF", "&H0066FF66", "&H00000000"
SHADE = "&H64000000"
FONT = "Arial Black"

STYLES = [
    # name, size, primary, outline colour, back colour, border style, outline, shadow,
    # alignment (numpad), marginL, marginR, marginV
    ("Caption", 88, WHITE, BLACK, SHADE, 1, 7, 2, 2, 90, 170, 620),
    ("Hook", 64, WHITE, BLACK, SHADE, 3, 18, 0, 8, 90, 170, 430),
    ("Counter", 44, WHITE, BLACK, SHADE, 1, 4, 1, 7, 60, 60, 270),
    ("Badge", 46, GREEN, BLACK, SHADE, 1, 4, 1, 7, 60, 60, 330),
    ("Label", 58, YELLOW, BLACK, SHADE, 1, 6, 2, 8, 90, 170, 430),
]


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


def build_ass(
    lines: list[PlacedLine],
    words: list[TimedWord],
    overlay: Overlay,
    duration: float,
    offset: float = 0.0,
) -> str:
    events: list[str] = []

    def add(start: float, end: float, style: str, text: str) -> None:
        if end - start > 0.01:
            events.append(f"Dialogue: 0,{ts(start)},{ts(end)},{style},,0,0,0,,{text}")

    # Word-by-word captions: each word gets an event showing its chunk, itself highlighted.
    for placed in lines:
        for chunk in caption_chunks(words_for_line(words, placed)):
            for i, w in enumerate(chunk):
                end = chunk[i + 1].start if i + 1 < len(chunk) else w.end + 0.15
                text = " ".join(
                    (
                        f"{{\\c{YELLOW}}}{esc(x.text.upper())}{{\\c{WHITE}}}"
                        if j == i
                        else esc(x.text.upper())
                    )
                    for j, x in enumerate(chunk)
                )
                add(w.start, end, "Caption", text)

    # Hook at the top for the opening seconds.
    if overlay.hook:  # no hook without a script (narration-free render)
        add(offset, offset + overlay.hook_seconds, "Hook", esc(overlay.hook.upper()))

    # Master-set counter, stepping up at each pack's end, plus a "+N NEW" badge.
    t, value = offset, overlay.master_before
    for pack in sorted(overlay.pack_done_at):
        end = overlay.pack_done_at[pack]
        add(t, end, "Counter", f"MASTER SET {value}/{overlay.master_total}")
        value = overlay.master_after_pack.get(pack, value)
        new = overlay.new_per_pack.get(pack, 0)
        if new:
            add(end, min(end + 1.6, duration), "Badge", f"+{new} NEW")
        t = end
    add(t, duration, "Counter", f"MASTER SET {value}/{overlay.master_total}")

    # Rarity label during big-hit zooms (after the hook has gone).
    hook_end = offset + overlay.hook_seconds if overlay.hook else offset
    for start, end, text in overlay.labels:
        add(max(start, hook_end), end, "Label", esc(text))

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
