"""Stage 2 entry point: video → pulls.json (+ thumbnails for review)."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from pokepipe.cards.cache import SetCache
from pokepipe.cards.embeddings import CARD_ASPECT
from pokepipe.cards.index import CardIndex
from pokepipe.cards.service import is_decoy
from pokepipe.detection.frames import probe, require_tool, sample_frames, scaled_size
from pokepipe.detection.layout import PackLayout, apply_layout, layout_for
from pokepipe.detection.matcher import CropGrid, FrameMatcher, FrameResult, PixelEmbedder
from pokepipe.detection.models import Alternative, DetectionParams, Pack, Pull, PullsFile
from pokepipe.detection.packs import PackParams
from pokepipe.detection.reshows import merge_reshows
from pokepipe.detection.segment import SegmentParams, confidence, segment


def video_slug(video: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", video.stem.lower()).strip("-")[:60] or "video"


def build_pulls(
    frames: list[FrameResult],
    cards_by_id: dict,
    frame_size: tuple[int, int],
    params: SegmentParams,
) -> list[Pull]:
    """Segments → Pull records. A segment whose best match is a decoy (the ordinary
    basic Energy every pack contains, see cards.service.DECOY_SET) is not a pull and is
    dropped. Real Energy pulls match their own card and are kept."""
    w, h = frame_size
    pulls = []
    for seg in segment(frames, params):
        if is_decoy(seg.card_id):
            continue
        card = cards_by_id[seg.card_id]
        peak = seg.peak
        x1, y1, x2, y2 = peak.box
        pulls.append(
            Pull(
                card_id=card.id, name=card.name, number=card.number, rarity=card.rarity,
                start=round(seg.start, 3), end=round(seg.end, 3),
                peak_time=round(peak.time, 3), peak_score=round(peak.best.score, 4),
                mean_margin=round(seg.mean_margin, 4), frames=len(seg.frames),
                confidence=confidence(seg),
                box=(x1 / w, y1 / h, x2 / w, y2 / h),
                alternatives=[Alternative(card_id=c, score=round(s, 4))
                              for c, s in seg.alternatives()
                              if c != card.id and not is_decoy(c)],
            )
        )
    return pulls


def infer_slots(pulls: list[Pull], packs: list[Pack], layout: PackLayout | None):
    """Run pack-layout inference per pack; records the outcome on each Pack."""
    if layout is None:
        return None
    for pack in packs:
        result = apply_layout([p for p in pulls if p.pack == pack.index], layout)
        pack.layout, pack.layout_note = result.mode, result.note
    return layout


def save_thumbnail(video: Path, pull: Pull, dest: Path, video_size: tuple[int, int]) -> None:
    """Full-resolution crop of the peak frame, padded and forced to the card's portrait
    shape (match boxes may be square) so the reviewer sees name, art and rarity."""
    vw, vh = video_size
    x1, y1, x2, y2 = pull.box
    cx, cy, side = (x1 + x2) / 2 * vw, (y1 + y2) / 2 * vh, (x2 - x1) * vw
    cw, ch = side * 1.15, side * 1.15 / CARD_ASPECT
    cw, ch = min(cw, vw), min(ch, vh)
    left = min(max(0, cx - cw / 2), vw - cw)
    top = min(max(0, cy - ch / 2), vh - ch)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            require_tool("ffmpeg"), "-v", "error", "-y", "-ss", f"{pull.peak_time:.3f}",
            "-i", str(video), "-frames:v", "1",
            "-vf", f"crop={int(cw)}:{int(ch)}:{int(left)}:{int(top)},scale=-2:600",
            "-q:v", "3", str(dest),
        ],
        check=True, capture_output=True,
    )


def detect_pulls(
    video: Path,
    set_id: str,
    sets_root: Path,
    index: CardIndex,
    embedder: PixelEmbedder,
    out_dir: Path,
    fps: float = 4.0,
    params: SegmentParams | None = None,
    grid: CropGrid | None = None,
    pack_params: PackParams | None = None,
    thumbnails: bool = True,
    on_frame: Callable[[FrameResult], None] | None = None,
) -> PullsFile:
    params = params or SegmentParams()
    pack_params = pack_params or PackParams()
    info = probe(video)
    size = scaled_size(info, 960)
    matcher = FrameMatcher(embedder, index, grid)
    results = []
    for r in matcher.match_stream(sample_frames(video, fps=fps, max_side=960)):
        results.append(r)
        if on_frame:
            on_frame(r)

    cache = SetCache(sets_root, set_id)
    cards = {c.id: c for c in cache.load_cards()}
    pulls = build_pulls(results, cards, size, params)
    # Repeat showings of a card count once; their footage is kept on the pull.
    pulls, packs = merge_reshows(pulls, info.duration, pack_params)
    layout = infer_slots(pulls, packs, layout_for(cache.load_set().series))
    if thumbnails:
        for i, pull in enumerate(pulls, 1):
            rel = f"thumbs/p{pull.pack:02d}_{i:03d}_{pull.card_id}.jpg"
            save_thumbnail(video, pull, out_dir / rel, (info.width, info.height))
            pull.thumbnail = rel

    result = PullsFile(
        video=str(video.resolve()), set_id=set_id, model_tag=index.model_tag,
        duration=info.duration, packs=packs, pulls=pulls,
        params=DetectionParams(fps=fps, **params.__dict__),
        pack_layout=layout.name if layout else None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pulls.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
