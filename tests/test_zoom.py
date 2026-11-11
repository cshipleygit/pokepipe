"""Zoom paths (pure geometry), the zoompan expression, and the tracker's box search."""

import numpy as np
import pytest
import torch

from pokepipe.render import ffmpeg
from pokepipe.render.timeline import Segment
from pokepipe.render.zoom import ZoomParams, add_zoom_paths, zoom_path

W, H = 2160, 3840  # vertical phone footage
WINDOW = (W, H, 0, 0)
CARD = (0.30, 0.33, 0.74, 0.67)  # ~44% of the frame width, like Allen's footage


def zseg(start=0.0, end=3.0, crop=(1438, 2556, 400, 640), chained=False, hold=0.0):
    return Segment(start, end, "zoom", 1, "p1-1", crop=crop, window=WINDOW,
                   chained=chained, hold=hold)


def inside(crop, box, pad=0.0):
    w, h, x, y = crop
    x1, y1, x2, y2 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return x <= x1 - px + 1 and y <= y1 - py + 1 and x + w >= x2 + px - 1 and y + h >= y2 + py - 1


def test_zoom_eases_in_from_the_normal_framing_and_out_again():
    track = [(t / 5, CARD) for t in range(15)]
    keys = zoom_path(zseg(), track, W, H, WINDOW, ease_out=True)
    widths = [c[0] for _, c in keys]
    assert widths[0] == W and widths[-1] == W  # starts and ends at the normal framing
    assert min(widths) < W / 1.4  # really zooms in between
    steps = [abs(a - b) for a, b in zip(widths, widths[1:], strict=False)]
    assert max(steps) < W * 0.15  # no jump cut: each keyframe moves a little
    for _, c in keys:
        w, h, x, y = c
        assert w / h == pytest.approx(9 / 16, abs=0.01)
        assert x >= 0 and x + w <= W and y >= 0 and y + h <= H


def test_the_whole_card_stays_inside_the_crop_while_it_moves():
    # The card drifts 25% of the frame to the left during the shot (Arcanine).
    track = []
    for i in range(16):
        dx = -0.25 * i / 15
        track.append((i / 5, (CARD[0] + dx, CARD[1], CARD[2] + dx, CARD[3])))
    keys = zoom_path(zseg(), track, W, H, WINDOW, ease_out=False)
    p = ZoomParams()
    for t, crop in keys:
        if t < p.ease_in:
            continue  # still easing in from the (wider) normal framing
        i = min(range(len(track)), key=lambda k: abs(track[k][0] - t))
        assert inside(crop, track[i][1]), (t, crop, track[i][1])


def test_lost_card_falls_back_to_the_normal_framing():
    track = [(i / 5, CARD if i < 4 else None) for i in range(15)]
    keys = zoom_path(zseg(), track, W, H, WINDOW, ease_out=False)
    assert keys[-1][1] == WINDOW


def test_a_brief_loss_is_bridged_not_zoomed_out():
    track = [(i / 5, None if i == 7 else CARD) for i in range(15)]
    keys = zoom_path(zseg(), track, W, H, WINDOW, ease_out=False)
    mid = [c[0] for t, c in keys if 1.0 < t < 2.0]
    assert max(mid) < W * 0.8


def test_without_a_tracker_the_zoom_aims_at_the_static_crop():
    keys = zoom_path(zseg(), None, W, H, WINDOW, ease_out=False)
    assert keys[-1][1][0] == 1438 and keys[0][1] == WINDOW


def test_chained_zoom_pans_from_the_previous_zoom_without_zooming_out():
    a = zseg(0.0, 3.0)
    b = zseg(3.0, 6.0, crop=(1438, 2556, 100, 900), chained=True)
    hit = Segment(-0.3, 0.0, "hit", 1, "p1-1", crop=WINDOW)
    add_zoom_paths([hit, a, b], W, H)
    assert a.path[-1][1][0] < W  # a doesn't zoom out before b
    assert b.path[0][1] == a.path[-1][1]  # b opens where a ended
    assert b.path[-1][1] == WINDOW  # b eases out at the end


def test_zoom_with_a_freeze_does_not_ease_out():
    s = zseg(hold=1.0)
    add_zoom_paths([s], W, H)
    assert s.path[-1][1][0] < W


def test_zoompan_filter_interpolates_the_path():
    s = zseg()
    s.path = [(0.0, WINDOW), (1.0, (1080, 1920, 540, 960))]
    vf = ffmpeg.segment_filter(s)
    assert vf.startswith(f"crop={W}:{H}:0:0,") and "zoompan=z='" in vf and ":s=1080x1920" in vf
    z = vf.split("z='")[1].split("'")[0]
    assert z.startswith("if(lt((in/30*1.0000),1.000),1.0000+(1.0000)*")  # 1x → 2x over 1 s
    assert z.endswith(",2.0000)")


# --- tracker search, with a fake embedder ---------------------------------------------------


class ColourLayoutEmbedder:
    """Embeds a crop as its 8×8 colour layout, so only the exact card box matches."""

    image_size = 32
    device = "cpu"
    crop_aspect = 63 / 88

    def embed_pixels(self, pixels):
        x = torch.nn.functional.adaptive_avg_pool2d(pixels, 8).flatten(1)
        return torch.nn.functional.normalize(x, dim=-1)


def test_tracker_finds_the_card_outline_from_a_rough_box():
    from pokepipe.cards.index import CardIndex
    from pokepipe.render.track import CardTracker

    h, w = 960, 540
    img = np.zeros((h, w, 3), np.float32)
    bh = 360
    bw = int(bh * 63 / 88)
    x1, y1 = 150, 330
    img[y1:y1 + bh, x1:x1 + bw // 2, 0] = 1  # left half red
    img[y1:y1 + bh, x1 + bw // 2:x1 + bw, 1] = 1  # right half green
    img[y1:y1 + bh // 3, x1:x1 + bw, 2] = 1  # blue band at the top
    b = 14  # a yellow border, like a real card's: a crop inside the card misses it
    for sl in ((slice(y1, y1 + b), slice(x1, x1 + bw)),
               (slice(y1 + bh - b, y1 + bh), slice(x1, x1 + bw)),
               (slice(y1, y1 + bh), slice(x1, x1 + b)),
               (slice(y1, y1 + bh), slice(x1 + bw - b, x1 + bw))):
        img[sl] = (1, 1, 0)
    image = torch.from_numpy(img).permute(2, 0, 1)[None]
    emb = ColourLayoutEmbedder()
    card = image[:, :, y1:y1 + bh, x1:x1 + bw]
    ref = emb.embed_pixels(torch.nn.functional.interpolate(card, size=(32, 32))).numpy()
    tracker = CardTracker(emb, CardIndex(["c1"], ref, "fake", device="cpu"))
    rough = np.asarray([x1 + 45, y1 - 40, x1 + 45 + bw * 0.9, y1 - 40 + bh * 0.9], np.float32)
    box, score = tracker._locate(image, tracker._refs("c1"), w, h, rough, wide=True)
    assert score > 0.97
    # Errs large, never small: the whole card is inside, at most ~12% to spare.
    assert box[0] <= x1 + 4 and box[1] <= y1 + 4
    assert box[2] >= x1 + bw - 4 and box[3] >= y1 + bh - 4
    assert (box[3] - box[1]) <= bh * 1.12
