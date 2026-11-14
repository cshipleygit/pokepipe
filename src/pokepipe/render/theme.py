"""The look: a cute rocket-launch theme for the master-set run.

Colours are kept here as RGB and converted to ASS's &HAABBGGRR order in one place.
The rocket is a small PNG drawn with Pillow (4x supersampled for smooth edges) and
overlaid by ffmpeg, so it can fly along the progress bar and bob gently; libass can
draw shapes but not multi-colour artwork that moves as one piece.

**Private brands.** A channel's own look stays out of the repo: if
`data/brand/theme.py` exists (gitignored, like all of `data/`), it is run in this
module's namespace when this module is first imported, so it can replace any name
below: colours, `FONT`/`FONT_FILE`, `draw_rocket` and `ROCKET_W/H`, the bar's look,
and the optional `badge_icon` / `price_icon` drawings. Font files go in
`data/brand/fonts/` and are handed to libass at render time. Set `POKEPIPE_NO_BRAND=1`
to ignore the brand (the test suite does, so it checks this default look).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "Gill Sans Ultra Bold Condensed"  # ships with Windows/Office: chunky and playful
FONT_FILE: str | None = None  # a file in data/brand/fonts/, for measuring text

NAVY = (27, 32, 74)  # outlines, the bar's track, the hook's box
WHITE = (255, 255, 255)
ORANGE = (255, 168, 64)  # rocket flame: the spoken word, the bar's fill
YELLOW = (255, 216, 74)  # rarity labels, the flame's core
MINT = (110, 240, 194)  # "+N NEW"
CORAL = (255, 98, 98)  # rocket nose and fins
SKY = (96, 190, 255)  # rocket window


# Which colour each overlay element uses (a brand can re-point these).
BADGE_COLOUR = MINT  # "+N NEW"
LABEL_COLOUR = YELLOW  # rarity labels
PRICE_COLOUR = MINT  # price tags
# The progress bar's empty track: navy pill, its alpha, rim colour and width.
BAR_TRACK_ALPHA = 0x40
BAR_RIM = NAVY
BAR_RIM_WIDTH = 3
COUNTER_OFFSET = (0, -10)  # "MASTER SET 80/234" relative to the bar's top-left

# Optional drawings (ASS vector shapes), None for text only:
#   badge_icon(x, cy) -> (drawings, advance): an icon before "+N NEW", whose left edge
#     is x and middle cy; the text starts `advance` px right of x.
#   price_icon(x, cy, extra) -> drawings: an icon before the price, PRICE_ICON_W wide,
#     PRICE_ICON_GAP before the text; `extra` is ASS tags to add (e.g. a fade).
badge_icon: Callable[[float, float], tuple[list[str], float]] | None = None
price_icon: Callable[[float, float, str], list[str]] | None = None
PRICE_ICON_W = 0.0
PRICE_ICON_GAP = 14.0


def ass(rgb: tuple[int, int, int], alpha: int = 0) -> str:
    """RGB → ASS colour (&HAABBGGRR; alpha 0 = opaque)."""
    r, g, b = rgb
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


ROCKET_W, ROCKET_H = 160, 102  # output pixels, after the tilt


def draw_rocket(path: Path, scale: int = 4) -> Path:
    """A cartoon rocket pointing right, tilted up 18° (taking off), with a flame."""
    w, h = 120 * scale, 64 * scale
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = scale

    def box(x1, y1, x2, y2):
        return [x1 * s, y1 * s, x2 * s, y2 * s]

    def poly(*pts):
        return [(x * s, y * s) for x, y in pts]

    # flame (behind the body): a teardrop, orange outside, yellow core
    d.ellipse(box(6, 23, 40, 41), fill=ORANGE)
    d.polygon(poly((14, 25), (0, 32), (14, 39)), fill=ORANGE)
    d.ellipse(box(16, 26, 40, 38), fill=YELLOW)
    # fins: chunky, rounded
    d.rounded_rectangle(box(30, 6, 54, 26), radius=7 * s, fill=CORAL, outline=NAVY,
                        width=3 * s)
    d.rounded_rectangle(box(30, 38, 54, 58), radius=7 * s, fill=CORAL, outline=NAVY,
                        width=3 * s)
    # body: a chubby capsule, coral nose first, white body over it (the curved join)
    d.ellipse(box(26, 14, 106, 50), fill=CORAL)
    d.ellipse(box(26, 14, 84, 50), fill=WHITE)
    d.arc(box(26, 14, 84, 50), start=-62, end=62, fill=NAVY, width=3 * s)
    d.ellipse(box(26, 14, 106, 50), outline=NAVY, width=3 * s)
    # window with a highlight
    d.ellipse(box(54, 22, 74, 42), fill=SKY, outline=NAVY, width=3 * s)
    d.ellipse(box(58, 25, 64, 31), fill=WHITE)
    rocket = im.rotate(18, resample=Image.Resampling.BICUBIC, expand=True)
    rocket = rocket.resize((ROCKET_W, ROCKET_H), Image.Resampling.LANCZOS)
    rocket.save(path)
    return path


def text_width(text: str, size: float, fonts_dir: Path | None) -> float:
    """Rough on-screen width of `text` at ASS font size `size`.

    libass scales a font so ascent + descent equals the ASS size; Pillow sizes by em.
    Without the font file, assume an average glyph of 0.6 × size."""
    path = fonts_dir / FONT_FILE if fonts_dir and FONT_FILE else None
    if path is None or not path.exists():
        return 0.6 * size * len(text)
    probe = ImageFont.truetype(str(path), 100)
    ascent, descent = probe.getmetrics()
    em = size * 100 / (ascent + descent)
    return ImageFont.truetype(str(path), max(1, round(em))).getlength(text)


def brand_file() -> Path | None:
    if os.environ.get("POKEPIPE_NO_BRAND"):
        return None
    from pokepipe.config import get_settings  # config imports nothing from render

    path = get_settings().data_dir / "brand" / "theme.py"
    return path if path.exists() else None


BRAND: Path | None = brand_file()
if BRAND is not None:
    exec(compile(BRAND.read_text(encoding="utf-8"), str(BRAND), "exec"), globals())
