"""Render the pokepipe logo set into ``docs/brand/``.

    python scripts/make_logo.py            # writes docs/brand/*.png

Outputs: ``logo-dark.png`` / ``logo-light.png`` (README lockup, transparent, light or
dark wordmark), ``icon.png`` (512x512 tile) and ``social-preview.png`` (1280x640).

The mark is a rounded tile with a holo-foil edge holding a lowercase "p" whose full
stop is a small tilted card. Geometry follows the CSS of the design artboard, so every
size is a fraction of the tile or of the font size. Fonts (all SIL OFL) are downloaded
from the google/fonts GitHub repo into the gitignored ``data/cache/fonts/`` on first
run, never committed. Everything is drawn at ``SS``x and downsampled, which gives the
rounded corners and the rotated card clean anti-aliased edges.
"""

from __future__ import annotations

import argparse
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "data" / "cache" / "fonts"
OUT_DIR = ROOT / "docs" / "brand"

_GF = "https://raw.githubusercontent.com/google/fonts/main/ofl"
FONTS = {
    "unbounded": f"{_GF}/unbounded/Unbounded%5Bwght%5D.ttf",
    "inter": f"{_GF}/inter/Inter%5Bopsz,wght%5D.ttf",
    "mono": f"{_GF}/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
}

INK = "#0B0B0F"  # near-black: tile inside, social background, wordmark on light
PAPER = "#F4F4F6"  # glyphs on dark
LIME = "#D4FF3A"  # the card dot, accents
GREY = "#A3A3B1"  # secondary text
HOLO = ["#ff8fd0", "#8fd3ff", "#b6ffa8", "#fff38f", "#ff8fd0"]
HOLO_ANGLE = 115  # CSS linear-gradient angle, degrees

SS = 4  # supersampling factor

TAGLINE = (
    "Turn a raw Pokémon TCG pack-opening video into narrated, edited vertical shorts, "
    "locally, for under $1 per video."
)
REPO = "github.com/cshipleygit/pokepipe"
STACK = "CLIP · LangGraph · Claude · ffmpeg"


# --------------------------------------------------------------------------- fonts


def font_file(name: str) -> Path:
    """Path to a cached font, downloading it once."""
    path = FONT_DIR / f"{name}.ttf"
    if not path.exists():
        FONT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} font -> {path}")
        with urllib.request.urlopen(FONTS[name], timeout=60) as r:  # noqa: S310 (fixed URL)
            path.write_bytes(r.read())
    return path


def load_font(name: str, px: float, weight: int) -> ImageFont.FreeTypeFont:
    """A variable font at ``px`` pixels with its weight axis set (other axes default,
    except optical size, which follows the pixel size like CSS ``font-optical-sizing``)."""
    font = ImageFont.truetype(str(font_file(name)), size=max(1, round(px)))
    values = []
    for axis in font.get_variation_axes():
        tag = axis["name"].decode() if isinstance(axis["name"], bytes) else axis["name"]
        lo, hi = axis["minimum"], axis["maximum"]
        if tag.lower() == "weight":
            values.append(min(max(weight, lo), hi))
        elif tag.lower() == "optical size":
            values.append(min(max(px / SS, lo), hi))
        else:
            values.append(axis["default"])
    font.set_variation_by_axes(values)
    return font


def baseline_in_line(font: ImageFont.FreeTypeFont) -> float:
    """Baseline offset from the top of a ``line-height: 1`` box, as browsers place it
    (half-leading split evenly above and below the ascent + descent)."""
    ascent, descent = font.getmetrics()
    return (font.size - (ascent + descent)) / 2 + ascent


def tracked_width(text: str, font: ImageFont.FreeTypeFont, tracking: float) -> float:
    """Advance width with CSS letter-spacing, which browsers also add after the last
    glyph (that is how the design's flex layout measures it)."""
    return font.getlength(text) + tracking * len(text)


def draw_tracked(
    draw: ImageDraw.ImageDraw,
    x: float,
    baseline: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    tracking: float,
    fill: str,
) -> None:
    """Draw ``text`` with letter-spacing, keeping the font's own pair kerning."""
    for i, ch in enumerate(text):
        pen = x + font.getlength(text[:i]) + tracking * i
        draw.text((pen, baseline), ch, font=font, fill=fill, anchor="ls")


# --------------------------------------------------------------------------- mark


def holo_gradient(w: int, h: int) -> Image.Image:
    """CSS ``linear-gradient(115deg, ...)`` with evenly spaced stops, as RGBA."""
    a = math.radians(HOLO_ANGLE)
    dx, dy = math.sin(a), -math.cos(a)  # CSS angles: 0deg = up, clockwise
    length = abs(w * dx) + abs(h * dy)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    t = ((xs - w / 2) * dx + (ys - h / 2) * dy) / length + 0.5
    stops = np.array([[int(c[i : i + 2], 16) for i in (1, 3, 5)] for c in HOLO], np.float32)
    pos = np.linspace(0, 1, len(HOLO))
    rgb = np.stack([np.interp(t, pos, stops[:, k]) for k in range(3)], axis=-1)
    alpha = np.full((h, w, 1), 255, np.float32)
    return Image.fromarray(np.concatenate([rgb, alpha], -1).round().astype(np.uint8), "RGBA")


def draw_p_with_card(img: Image.Image, cx_box: float, top: float, font_px: float) -> None:
    """The "p" plus its card-shaped full stop, as an inline-flex box whose left edge
    is chosen so the box is centred on ``cx_box``, with its ``line-height: 1`` top at
    ``top``. Mirrors the artboard: letter-spacing -0.05em, card 0.2em x 0.28em,
    radius 0.04em, 0.06em after the p, bottom on the baseline, rotated -10deg about
    its bottom-left corner."""
    font = load_font("unbounded", font_px, 800)
    em = font.size
    tracking = -0.05 * em
    p_w = font.getlength("p") + tracking
    card_w, card_h, card_r, gap = 0.2 * em, 0.28 * em, 0.04 * em, 0.06 * em
    left = cx_box - (p_w + gap + card_w) / 2
    baseline = top + baseline_in_line(font)

    ImageDraw.Draw(img).text((left, baseline), "p", font=font, fill=PAPER, anchor="ls")

    # Card on its own layer so it can be rotated about its bottom-left corner.
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    x0 = left + p_w + gap
    ImageDraw.Draw(layer).rounded_rectangle(
        (x0, baseline - card_h, x0 + card_w, baseline), radius=card_r, fill=LIME
    )
    # CSS rotate(-10deg) turns counter-clockwise on screen; so does PIL's positive angle.
    layer = layer.rotate(10, resample=Image.BICUBIC, center=(x0, baseline))
    img.alpha_composite(layer)


def tile(size: int) -> Image.Image:
    """The square mark at ``size`` px (already supersampled): holo edge 3% wide,
    corner radius 24%, near-black inside, "p" + card centred like the artboard."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    radius = 0.24 * size
    edge = 0.03 * size

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    img.paste(holo_gradient(size, size), (0, 0), mask)
    ImageDraw.Draw(img).rounded_rectangle(
        (edge, edge, size - 1 - edge, size - 1 - edge), radius=radius - edge, fill=INK
    )

    # Artboard: 200px tile, 136px glyph, line box nudged up by 26px inside a flex
    # centre, so the box top sits at (200 - (136 - 26)) / 2 - 26 = 19px.
    font_px = 0.68 * size
    top = (size - (font_px - 0.13 * size)) / 2 - 0.13 * size
    draw_p_with_card(img, size / 2, top, font_px)
    return img


@dataclass
class Lockup:
    image: Image.Image  # supersampled, transparent, tight to the tile and wordmark
    tile_px: int


def lockup(tile_px: int, word_fill: str) -> Lockup:
    """Tile, a gap of 20% of the tile, then "pokepipe" at 56% of the tile (the
    artboard's 200px tile / 40px gap / 112px type), the type's line box centred on
    the tile."""
    mark = tile(tile_px)
    font = load_font("unbounded", 0.56 * tile_px, 800)
    tracking = -0.035 * font.size
    gap = 0.2 * tile_px
    word_w = tracked_width("pokepipe", font, tracking) - tracking  # drop trailing space
    # Leave room for the glyphs' side bearings / overshoot on the right.
    width = math.ceil(tile_px + gap + word_w + 0.02 * tile_px)
    img = Image.new("RGBA", (width, tile_px), (0, 0, 0, 0))
    img.alpha_composite(mark, (0, 0))
    top = (tile_px - font.size) / 2
    draw_tracked(
        ImageDraw.Draw(img),
        tile_px + gap,
        top + baseline_in_line(font),
        "pokepipe",
        font,
        tracking,
        word_fill,
    )
    return Lockup(img, tile_px)


def downsample(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((round(w / SS), round(h / SS)), Image.LANCZOS)


# --------------------------------------------------------------------------- outputs


def readme_logo(word_fill: str, width: int = 1600) -> Image.Image:
    """Lockup about ``width`` px wide on transparent, sharp at 2x in the README."""
    probe = lockup(100 * SS, word_fill).image.width / (100 * SS)  # width per tile px
    tile_px = round(width / probe) * SS
    return downsample(lockup(tile_px, word_fill).image)


def icon(size: int = 512) -> Image.Image:
    return downsample(tile(size * SS))


def wrap(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        if lines and font.getlength(f"{lines[-1]} {word}") <= max_w:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return lines


def social_preview(w: int = 1280, h: int = 640) -> Image.Image:
    """GitHub social card, laid out like the 1200x630 artboard scaled to 1280 wide:
    lockup and tagline centred as one column, repo and stack in the bottom corners."""
    k = w / 1200 * SS
    W, H = w * SS, h * SS
    img = Image.new("RGBA", (W, H), INK)
    draw = ImageDraw.Draw(img)

    mark = lockup(round(200 * k), PAPER).image
    tag_font = load_font("inter", 24 * k, 400)
    line_h = 24 * k * 1.45
    lines = wrap(TAGLINE, tag_font, 780 * k)
    col_gap = 40 * k
    col_h = mark.height + col_gap + line_h * len(lines)
    y = (H - col_h) / 2

    img.alpha_composite(mark, (round((W - mark.width) / 2), round(y)))
    y += mark.height + col_gap
    base = baseline_in_line(tag_font) + (line_h - tag_font.size) / 2
    for line in lines:
        draw.text((W / 2, y + base), line, font=tag_font, fill=GREY, anchor="ms")
        y += line_h

    mono = load_font("mono", 16 * k, 500)
    foot_baseline = H - 40 * k - mono.size + baseline_in_line(mono)
    draw.text((48 * k, foot_baseline), REPO, font=mono, fill=GREY, anchor="ls")
    draw.text((W - 48 * k, foot_baseline), STACK, font=mono, fill=LIME, anchor="rs")
    return downsample(img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    outputs = {
        "logo-dark.png": readme_logo(PAPER),  # light text, for GitHub's dark theme
        "logo-light.png": readme_logo(INK),  # dark text, for GitHub's light theme
        "icon.png": icon(),
        "social-preview.png": social_preview().convert("RGB"),
    }
    for name, image in outputs.items():
        path = args.out / name
        image.save(path, optimize=True)
        print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}  {image.size}")


if __name__ == "__main__":
    main()
