"""The look: a cute rocket-launch theme for the master-set run.

Colours are kept here as RGB and converted to ASS's &HAABBGGRR order in one place.
The rocket is a small PNG drawn with Pillow (4x supersampled for smooth edges) and
overlaid by ffmpeg, so it can fly along the progress bar and bob gently; libass can
draw shapes but not multi-colour artwork that moves as one piece.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

FONT = "Gill Sans Ultra Bold Condensed"  # ships with Windows/Office: chunky and playful

NAVY = (27, 32, 74)  # outlines, the bar's track, the hook's box
WHITE = (255, 255, 255)
ORANGE = (255, 168, 64)  # rocket flame: the spoken word, the bar's fill
YELLOW = (255, 216, 74)  # rarity labels, the flame's core
MINT = (110, 240, 194)  # "+N NEW"
CORAL = (255, 98, 98)  # rocket nose and fins
SKY = (96, 190, 255)  # rocket window


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
