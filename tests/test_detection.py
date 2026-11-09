import json
import shutil
import subprocess

import numpy as np
import pytest
import torch

from pokepipe.cards.index import CardIndex, Match
from pokepipe.cards.models import Card
from pokepipe.detection.frames import Frame, sample_frames, scaled_size
from pokepipe.detection.matcher import CropGrid, FrameMatcher, FrameResult
from pokepipe.detection.models import PullsFile
from pokepipe.detection.segment import SegmentParams, confidence, segment
from pokepipe.detection.service import build_pulls, video_slug

# --- helpers -------------------------------------------------------------------------


def fr(t: float, card: str, score: float, runner_up: float | None = None) -> FrameResult:
    second = runner_up if runner_up is not None else score - 0.1
    return FrameResult(int(t * 4), t, [Match(card, score), Match("other", second)], (0, 0, 10, 10))


COLORS = {"red": (1.0, 0.1, 0.1), "green": (0.1, 1.0, 0.1), "blue": (0.1, 0.1, 1.0)}


class MeanColorEmbedder:
    """Fake pixel embedder: the crop's mean colour, normalised. Lets us test the GPU
    cropping path on CPU with a ground truth we control."""

    image_size = 16
    device = "cpu"

    def embed_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(pixels.mean(dim=(2, 3)), dim=-1)


def color_index() -> CardIndex:
    vecs = np.array(list(COLORS.values()), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return CardIndex(list(COLORS), vecs, "meancolor", device="cpu")


def frame_with_card(rgb: tuple[float, float, float], index=0, w=320, h=180) -> Frame:
    img = np.full((h, w, 3), 128, np.uint8)  # grey background
    img[30:150, 140:220] = (np.array(rgb) * 255).astype(np.uint8)  # "card" near centre
    return Frame(index, index / 4, img)


# --- crop grid / matcher -------------------------------------------------------------


def test_crop_grid_boxes_are_square_and_inside_frame():
    boxes = CropGrid(sides=(0.5, 1.2), centers_x=(0.0, 0.5, 1.0), centers_y=(0.5,)).boxes(320, 180)
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    np.testing.assert_allclose(w, h)
    assert boxes[:, :2].min() >= 0 and boxes[:, 2].max() <= 320 and boxes[:, 3].max() <= 180


def test_matcher_finds_card_by_colour():
    matcher = FrameMatcher(MeanColorEmbedder(), color_index(), CropGrid(sides=(0.6,)))
    frames = [frame_with_card(COLORS["green"], 0), frame_with_card(COLORS["blue"], 1)]
    results = matcher.match_batch(frames)
    assert [r.best.card_id for r in results] == ["green", "blue"]
    assert results[0].margin > 0
    # best crop should be the one centred on the card (x centre 0.5 of 320 = 160..)
    x1, _, x2, _ = results[0].box
    assert x1 < 180 < x2


def test_match_stream_batches_all_frames():
    matcher = FrameMatcher(MeanColorEmbedder(), color_index())
    frames = (frame_with_card(COLORS["red"], i) for i in range(5))
    assert len(list(matcher.match_stream(frames, batch_size=2))) == 5


# --- segmentation --------------------------------------------------------------------


def test_segment_groups_consecutive_votes_and_drops_flickers():
    frames = [fr(t / 4, "a", 0.8) for t in range(6)]  # 0.0-1.25s: card a
    frames += [fr(2.0, "b", 0.8)]  # single-frame flicker
    frames += [fr(3 + t / 4, "c", 0.85) for t in range(4)]
    segs = segment(frames)
    assert [(s.card_id, len(s.frames)) for s in segs] == [("a", 6), ("c", 4)]


def test_segment_bridges_short_gaps_and_flickers_but_not_long_ones():
    p = SegmentParams(max_gap=1.0)
    frames = [fr(0.0, "a", 0.8), fr(0.25, "a", 0.8), fr(0.5, "b", 0.9), fr(1.0, "a", 0.8)]
    frames += [fr(5.0, "a", 0.8), fr(5.25, "a", 0.8), fr(5.5, "a", 0.8)]
    segs = segment(frames, p)
    assert [(s.card_id, s.start, s.end) for s in segs] == [("a", 0.0, 1.0), ("a", 5.0, 5.5)]


def test_segment_requires_margin_and_peak():
    ambiguous = [fr(t / 4, "a", 0.8, runner_up=0.79) for t in range(5)]
    weak = [fr(5 + t / 4, "b", 0.70) for t in range(5)]
    assert segment(ambiguous + weak) == []


def test_confidence_increases_with_evidence():
    weak = segment([fr(t / 4, "a", 0.73, runner_up=0.69) for t in range(3)])[0]
    strong = segment([fr(t / 4, "a", 0.86, runner_up=0.70) for t in range(10)])[0]
    assert 0 <= confidence(weak) < confidence(strong) <= 1


def test_alternatives_rank_other_cards():
    seg = segment([fr(t / 4, "a", 0.8, runner_up=0.7) for t in range(4)])[0]
    assert seg.alternatives()[:2] == [("a", 0.8), ("other", 0.7)]


# --- service -------------------------------------------------------------------------


def test_build_pulls_serialises_relative_box(tmp_path):
    frames = [
        FrameResult(i, i / 4, [Match("s-1", 0.85), Match("s-2", 0.7)], (100, 50, 300, 250))
        for i in range(4)
    ]
    cards = {"s-1": Card(id="s-1", name="Pikachu", number="25", set_id="s", rarity="Common")}
    [pull] = build_pulls(frames, cards, (400, 300), SegmentParams())
    assert pull.name == "Pikachu" and pull.frames == 4
    assert pull.box == pytest.approx((0.25, 50 / 300, 0.75, 250 / 300))
    assert [a.card_id for a in pull.alternatives] == ["s-2"]
    # round-trips through the on-disk schema
    blob = PullsFile(video="v.mp4", set_id="s", model_tag="m", duration=1.0, pulls=[pull],
                     params={"fps": 4, **SegmentParams().__dict__}).model_dump_json()
    assert PullsFile.model_validate(json.loads(blob)).pulls[0] == pull


def test_video_slug():
    from pathlib import Path

    assert video_slug(Path("Pokémon 151 - EPIC PULL!.mp4")) == "pok-mon-151-epic-pull"


# --- frames (needs ffmpeg) -----------------------------------------------------------


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_sample_frames_from_generated_video(tmp_path):
    video = tmp_path / "test.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=2",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    frames = list(sample_frames(video, fps=5, max_side=320))
    assert len(frames) == 10
    assert frames[0].rgb.shape == (180, 320, 3)
    assert frames[3].time == pytest.approx(0.6)


def test_scaled_size_keeps_even_dims():
    from pokepipe.detection.frames import VideoInfo

    assert scaled_size(VideoInfo(1920, 1080, 1, 60), 960) == (960, 540)
    assert scaled_size(VideoInfo(1080, 1920, 1, 60), 960) == (540, 960)
    assert scaled_size(VideoInfo(640, 360, 1, 30), 960) == (640, 360)  # never upscale
