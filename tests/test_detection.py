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
from pokepipe.detection.models import Pull, PullsFile
from pokepipe.detection.packs import PackParams, group_packs
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
    crop_aspect = 1.0

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


@pytest.mark.parametrize("aspect", [1.0, 63 / 88])
def test_crop_grid_boxes_keep_aspect_and_stay_inside_frame(aspect):
    grid = CropGrid(heights=(0.5, 1.2), centers_x=(0.0, 0.5, 1.0), centers_y=(0.5,), aspect=aspect)
    boxes = grid.boxes(320, 180)
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    np.testing.assert_allclose(w / h, aspect, rtol=1e-2)
    assert boxes[:, :2].min() >= 0 and boxes[:, 2].max() <= 320 and boxes[:, 3].max() <= 180
    assert len(np.unique(boxes, axis=0)) == len(boxes)  # clamped duplicates removed


def test_matcher_rejects_grid_with_wrong_aspect():
    with pytest.raises(ValueError, match="aspect"):
        FrameMatcher(MeanColorEmbedder(), color_index(), CropGrid(aspect=63 / 88))


def test_matcher_finds_card_by_colour():
    grid = CropGrid(heights=(0.6,), aspect=1.0)
    matcher = FrameMatcher(MeanColorEmbedder(), color_index(), grid)
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


def test_decoys_are_dropped_but_real_energy_pulls_are_kept():
    def frames_for(card_id, runner_up):
        return [FrameResult(i, i / 4, [Match(card_id, 0.8), Match(runner_up, 0.6)],
                            (0, 0, 10, 10)) for i in range(4)]

    cards = {"s-e": Card(id="s-e", name="Basic Psychic Energy", number="207", set_id="s",
                         supertype="Energy", rarity="Hyper Rare")}
    # the pack's ordinary Energy matches a decoy: not a pull
    assert build_pulls(frames_for("decoy:sve-5", "s-e"), cards, (40, 40), SegmentParams()) == []
    # a real (gold) Energy pull matches its own card: kept, decoys kept out of alternatives
    [pull] = build_pulls(frames_for("s-e", "decoy:sve-5"), cards, (40, 40), SegmentParams())
    assert pull.card_id == "s-e" and pull.alternatives == []


def make_pull(start: float, end: float | None = None) -> Pull:
    return Pull(card_id="s-1", name="X", number="1", start=start, end=end or start + 1,
                peak_time=start, peak_score=0.8, mean_margin=0.1, frames=4, confidence=0.9,
                box=(0, 0, 1, 1))


def test_group_packs_splits_on_gaps_and_sets_clip_windows():
    # pack A: reveals 20-30s; pack B: 50-55s (20s gap); video is 60s long
    pulls = [make_pull(t) for t in (20, 22, 24, 29)] + [make_pull(t) for t in (50, 52, 54)]
    packs = group_packs(pulls, duration=60, params=PackParams(cards_per_pack=4))
    assert [(p.index, p.cards, p.complete) for p in packs] == [(1, 4, True), (2, 3, False)]
    assert [p.pack for p in pulls] == [1, 1, 1, 1, 2, 2, 2]
    a, b = packs
    assert a.clip_start == 5.0  # 15s lead-in for the tear
    assert a.clip_end == 32.0  # last reveal ends at 30 + 2s tail
    assert b.clip_start == 35.0  # 15s lead-in, still after pack A's clip
    assert b.clip_end == 57.0


def test_group_packs_clips_never_overlap_when_packs_are_close():
    pulls = [make_pull(10), make_pull(12), make_pull(22), make_pull(24)]  # 9s gap
    a, b = group_packs(pulls, duration=26)
    assert a.clip_end <= b.clip_start
    assert b.clip_end == 26  # clamped to video duration


def test_group_packs_empty():
    assert group_packs([], duration=10) == []


def test_openclip_fit_is_validated_before_loading_model():
    from pokepipe.cards.embeddings import OpenClipEmbedder

    with pytest.raises(ValueError, match="fit"):
        OpenClipEmbedder(fit="stretch")


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


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_rotated_phone_video_is_sampled_upright(tmp_path):
    """Phones store vertical video as landscape frames + a rotation flag. Frames must
    come out vertical, not squashed into the stored landscape shape."""
    plain, rotated = tmp_path / "plain.mp4", tmp_path / "rotated.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=640x360:rate=30:duration=1", "-pix_fmt", "yuv420p",
                    str(plain)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "-90", "-i", str(plain),
                    "-c", "copy", str(rotated)], check=True)
    from pokepipe.detection.frames import probe

    info = probe(rotated)
    assert (info.width, info.height, info.rotation) == (360, 640, 90)
    assert info.coded_size == (640, 360)
    frames = list(sample_frames(rotated, fps=5, max_side=320))
    assert frames[0].rgb.shape == (320, 180, 3)  # upright: taller than wide


def test_rotation_flag_parsing():
    from pokepipe.detection.frames import rotation_of

    assert rotation_of({"side_data_list": [{"rotation": -90}]}) == 90  # iPhone display matrix
    assert rotation_of({"side_data_list": [{"rotation": 90}]}) == 270
    assert rotation_of({"tags": {"rotate": "180"}}) == 180  # legacy tag
    assert rotation_of({}) == 0


def test_sampling_tries_gpu_first_then_cpu_and_rotates_after_scaling():
    from pathlib import Path

    from pokepipe.detection.frames import VideoInfo, sample_cmds

    info = VideoInfo(2160, 3840, 10, 60, rotation=90)
    gpu, cpu = sample_cmds(Path("v.mov"), info, 4, (540, 960))
    gpu_vf, cpu_vf = gpu[gpu.index("-vf") + 1], cpu[cpu.index("-vf") + 1]
    assert "cuda" in gpu and "scale_cuda=960:540" in gpu_vf  # scale the stored (landscape) frame
    assert cpu_vf == "fps=4,scale=960:540,transpose=clock"  # then rotate upright
    assert "-noautorotate" in cpu and gpu_vf.endswith("transpose=clock")


def test_portrait_frames_use_portrait_box_sizes():
    grid = CropGrid()
    boxes = grid.boxes(540, 960)
    heights = sorted({round(float(y2 - y1)) for _, y1, _, y2 in boxes})
    assert heights == [270, 335, 405]  # 0.5 / 0.62 / 0.75 of the 540 px short side
    assert boxes[:, 2].max() <= 540 and boxes[:, 3].max() <= 960
    landscape = {round(float(y2 - y1)) for _, y1, _, y2 in grid.boxes(960, 540)}
    assert landscape == {351, 432, 513}  # unchanged for 16:9 footage


def test_scaled_size_keeps_even_dims():
    from pokepipe.detection.frames import VideoInfo

    assert scaled_size(VideoInfo(1920, 1080, 1, 60), 960) == (960, 540)
    assert scaled_size(VideoInfo(1080, 1920, 1, 60), 960) == (540, 960)
    assert scaled_size(VideoInfo(640, 360, 1, 30), 960) == (640, 360)  # never upscale
