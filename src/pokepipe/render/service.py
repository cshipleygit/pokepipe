"""Stage 6 entry point: one post → posts/<n>/post.mp4 (1080×1920)."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from pokepipe.detection.frames import probe
from pokepipe.detection.models import PullsFile
from pokepipe.narration.models import PostPlan, PostScript, VoiceTiming
from pokepipe.narration.script import post_progress
from pokepipe.narration.service import VideoFiles
from pokepipe.render import ffmpeg
from pokepipe.render.captions import Overlay, build_ass
from pokepipe.render.inputs import pack_windows, render_pulls
from pokepipe.render.timeline import Edit, EditParams, Segment, VoiceLine, plan_edit
from pokepipe.render.zoom import add_zoom_paths
from pokepipe.review.models import ReviewState


class RenderInputError(RuntimeError):
    pass


@dataclass
class RenderResult:
    path: Path
    edit: Edit


def _load(path: Path, model, hint: str):
    if not path.exists():
        raise RenderInputError(f"missing {path.name} — {hint}")
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def render_post(
    files: VideoFiles,
    post_index: int,
    intro: Path | None = None,
    params: EditParams | None = None,
    keep_temp: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    narration: bool = True,
    track: bool = True,
    tracker=None,
) -> RenderResult:
    """Render one post. `narration=False` is the edit alone: no script or voiceover
    needed; the source audio plays under it, speed-matched per shot. `track` follows
    each zoomed card with `render.track` (pass `tracker` to reuse a loaded model)."""
    det = _load(files.root / "pulls.json", PullsFile, "run detection first")
    review = _load(files.root / "review.json", ReviewState, "finish the review first")
    if not review.completed:
        raise RenderInputError("the review isn't finished")
    report = files.load_report()
    plan = _load(files.plan, PostPlan, "run `pokepipe narrate plan`")
    post = next((p for p in plan.posts if p.index == post_index), None)
    if post is None:
        raise RenderInputError(f"no post {post_index} in posts.json")
    pdir = files.post_dir(post_index)
    script = voice = None
    if narration:
        script = _load(pdir / "script.json", PostScript, "run `pokepipe narrate script`")
        voice = _load(pdir / "voice.json", VoiceTiming, "run `pokepipe narrate voice`")

    source = Path(det.video)
    if not source.exists():
        raise RenderInputError(f"source video not found: {source}")
    info = probe(source)
    intro_len = probe(intro).duration if intro and intro.exists() else 0.0
    source_audio = not narration and ffmpeg.has_audio(source)

    target = post.seconds or (max(voice.duration + 1.0, 15.0) if voice else 45.0)
    params = params or EditParams(target=target)
    packs = pack_windows(det, post)
    lines = [VoiceLine(b.anchor, b.text, b.start, b.end) for b in voice.beats] if voice else []
    # Real footage after the post's last pack (until the next pack starts) that the final
    # shot may run into while the outro finishes.
    last_end = max(w.clip_end for w in packs)
    later = [p.clip_start for p in det.packs if p.clip_start >= last_end]
    tail_room = max(0.0, (min(later) if later else info.duration) - last_end)
    edit = plan_edit(
        packs,
        render_pulls(det, review, report),
        lines,
        (info.width, info.height),
        params,
        intro=intro_len,
        tail_room=tail_room,
    )

    by_id = {p.id: p for p in render_pulls(det, review, report)}

    # --- zooms follow the card ---------------------------------------------------------------
    if track and tracker is None and any(s.kind == "zoom" for s in edit.segments):
        from pokepipe.render.track import load_tracker

        tracker = load_tracker(det.set_id)
    track_log: list[dict] = []

    def track_fn(seg: Segment):
        pull = by_id[seg.pull_id]
        box = seg.box or pull.box
        if not (tracker and pull.card_id and box):
            return None
        samples = tracker.track(source, pull.card_id, seg.src_start, seg.src_end, box)
        track_log.append({"pull_id": seg.pull_id, "src_start": seg.src_start,
                          "scores": tracker.last_scores,
                          "lost": sum(b is None for _, b in samples)})
        return samples

    add_zoom_paths(edit.segments, info.width, info.height, track_fn if tracker else None)

    # --- overlay -----------------------------------------------------------------------------
    before, _ = post_progress(report, post)
    labels = [
        (s.out_start, s.out_end, f"{by_id[s.pull_id].label} · {by_id[s.pull_id].name.upper()}")
        for s in edit.segments
        if s.kind == "zoom" and by_id[s.pull_id].label
    ]
    overlay = Overlay(
        hook=script.hook_caption if script else "",
        hook_seconds=3.0,
        master_total=report.after.master_total,
        master_before=before.master_owned,
        master_after_pack={p.index: p.progress_after.master_owned for p in report.packs},
        new_per_pack={p.index: p.new_slots for p in report.packs if p.index in post.packs},
        pack_done_at=edit.pack_done_at,
        labels=labels,
    )

    work = pdir / "render"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "captions.ass").write_text(
        build_ass(edit.lines, voice.words if voice else [], overlay, edit.duration, intro_len),
        encoding="utf-8",
    )
    (work / "edit.json").write_text(
        json.dumps(
            {
                "duration": edit.duration,
                "reveal_speed": edit.reveal_speed,
                "notes": edit.notes,
                "tracking": track_log,
                "segments": [asdict(s) for s in edit.segments],
                "lines": [
                    {
                        "anchor": x.line.anchor,
                        "text": x.line.text,
                        "out_start": x.out_start,
                        "drift": x.drift,
                    }
                    for x in edit.lines
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- pass 1: segments --------------------------------------------------------------------
    clips: list[Path] = []
    total = len(edit.segments) + (1 if intro_len else 0)
    if intro_len:
        clip = work / "seg_000_intro.mp4"
        ffmpeg.run(ffmpeg.intro_cmd(intro, clip, audio=source_audio))
        clips.append(clip)
    for i, seg in enumerate(edit.segments, 1):
        clip = work / f"seg_{i:03d}.mp4"
        ffmpeg.run(ffmpeg.segment_cmd(source, seg, clip, audio=source_audio))
        clips.append(clip)
        if on_progress:
            on_progress(len(clips), total)
    concat = work / "segments.txt"
    concat.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")

    # --- pass 2: assemble ----------------------------------------------------------------------
    out = pdir / "post.mp4"
    if narration:
        cmd = ffmpeg.final_cmd(
            concat, pdir / "voice.mp3", edit.lines, "captions.ass", edit.duration, out
        )
    else:
        cmd = ffmpeg.final_cmd_source_audio(
            concat, "captions.ass", edit.duration, out, source_audio
        )
    ffmpeg.run(cmd, cwd=work)
    if not keep_temp:
        for clip in clips:
            clip.unlink(missing_ok=True)
    return RenderResult(out, edit)
