import shutil
import subprocess

import pytest

from pokepipe.narration.models import TimedWord
from pokepipe.render import ffmpeg
from pokepipe.render.captions import Overlay, build_ass, caption_chunks, ts
from pokepipe.render.timeline import (
    EditParams,
    PackWindow,
    RenderPull,
    Segment,
    VoiceLine,
    build_segments,
    frame_segments,
    place_voice,
    plan_edit,
    solve_speeds,
)

SRC = (1920, 1080)
BOX = (0.40, 0.10, 0.60, 0.90)  # card-ish box in the middle of the frame


def rp(pid, pack, start, rarity="Common", variant="normal", box=BOX, name="Mon"):
    return RenderPull(
        id=pid,
        pack=pack,
        name=name,
        rarity=rarity,
        variant=variant,
        new=True,
        src_start=start,
        src_end=start + 1.0,
        box=box,
    )


PACK = PackWindow(1, 0.0, 30.0)
PULLS = [
    rp("p1-1", 1, 10.0),
    rp("p1-2", 1, 12.0, rarity="Rare", variant="reverse"),  # reverse holo Rare: not a hit
    rp("p1-3", 1, 20.0, rarity="Special Illustration Rare", name="Charizard ex"),
]


def kinds(segs):
    return [(s.kind, s.pull_id) for s in segs]


def test_build_segments_classifies_and_covers_the_window():
    segs = build_segments([PACK], PULLS, EditParams())
    assert kinds(segs) == [
        ("gap", None),
        ("reveal", "p1-1"),
        ("reveal", "p1-2"),
        ("gap", None),
        ("hit", "p1-3"),
        ("gap", None),
    ]
    assert segs[0].src_start == 0.0 and segs[-1].src_end == 30.0
    for a, b in zip(segs, segs[1:], strict=False):
        assert a.src_end <= b.src_start + 1e-9  # no overlaps
    hit = segs[4]
    assert 2.5 <= hit.src_len <= 4.0  # hits: at least min_hit, at most max_hit at 1x


def test_long_held_hit_is_capped_at_max_hit():
    long_hold = [rp("p1-1", 1, 5.0, rarity="Illustration Rare")]
    long_hold = [RenderPull(**{**long_hold[0].__dict__, "src_end": 20.0})]  # held up 15 s
    segs = build_segments([PACK], long_hold, EditParams())
    hit = next(s for s in segs if s.kind == "hit")
    assert hit.src_len == pytest.approx(4.0)
    assert segs[-1].kind == "gap" and segs[-1].src_start == hit.src_end  # rest is flexible


def test_speeds_hit_target_with_hits_at_real_time():
    p = EditParams(target=12.0)
    segs = build_segments([PACK], PULLS, p)
    speed = solve_speeds(segs, p)
    assert speed > 1
    assert all(s.speed == 1.0 for s in segs if s.kind == "hit")
    assert all(
        s.speed == pytest.approx(min(speed * p.gap_factor, p.max_gap_speed))
        for s in segs
        if s.kind == "gap"
    )
    assert sum(s.out_len for s in segs) == pytest.approx(12.0, abs=0.6)


def test_speed_never_slows_footage_and_gaps_are_trimmed_when_capped():
    long = EditParams(target=600.0)
    segs = build_segments([PACK], PULLS, long)
    assert solve_speeds(segs, long) == 1.0  # target longer than footage: no slow-motion

    tight = EditParams(target=6.0, max_speed=2.0, max_gap_speed=3.0)
    segs = build_segments([PACK], PULLS, tight)
    solve_speeds(segs, tight)
    assert all(s.out_len <= tight.gap_cap + 1e-6 for s in segs if s.kind == "gap")


def test_framing_is_vertical_in_bounds_and_hits_zoom():
    p = EditParams(target=12.0)
    segs = build_segments([PACK], PULLS, p)
    solve_speeds(segs, p)
    framed = frame_segments(segs, {x.id: x for x in PULLS}, *SRC, p)
    assert [s.kind for s in framed].count("zoom") == 1
    for s in framed:
        w, h, x, y = s.crop
        assert w % 2 == 0 and h % 2 == 0
        assert x >= 0 and x + w <= 1920 and y >= 0 and y + h <= 1080
        assert w / h == pytest.approx(9 / 16, abs=0.01)
    zoom = next(s for s in framed if s.kind == "zoom")
    normal = next(s for s in framed if s.kind == "hit")
    assert zoom.crop[1] < normal.crop[1]  # tighter crop = zoomed in
    assert normal.src_end == zoom.src_start  # punch-in is a clean cut


def test_voice_lines_start_on_their_card_and_never_overlap():
    reveal = {"p1-1": 4.0, "p1-3": 4.5}
    lines = [
        VoiceLine("intro", "Hook.", 0.0, 2.0),
        VoiceLine("p1-1", "Common.", 2.1, 3.0),
        VoiceLine("p1-3", "Charizard!", 3.1, 6.0),  # its card shows before p1-1's line ends
        VoiceLine("p1-3", "Binder.", 6.1, 7.0),
        VoiceLine("outro", "Out.", 7.1, 8.0),
    ]
    placed = place_voice(lines, reveal, video_end=20.0, start=0.0, p=EditParams())
    starts = [x.out_start for x in placed]
    assert starts[0] == 0.0 and starts[1] == 4.0
    # Charizard appears at 4.5 but the previous line runs to 4.9: it waits (+gap) and drifts.
    assert starts[2] == pytest.approx(5.02) and placed[2].drift == pytest.approx(0.52)
    assert starts[3] == pytest.approx(placed[2].out_end + 0.12)
    # outro follows on (pacing, not placement, makes room for it at the end)
    assert placed[4].out_start == pytest.approx(placed[3].out_end + 0.12)
    assert placed[3].drift == 0  # a second line about the same card isn't "late"
    for a, b in zip(placed, placed[1:], strict=False):
        assert b.out_start >= a.out_end


def test_long_narration_freezes_on_the_hit_not_the_end():
    lines = [VoiceLine("intro", "x", 0.0, 34.0)]  # longer than all footage at 1x
    edit = plan_edit([PACK], PULLS, lines, SRC, EditParams(target=10.0))
    zoom = next(s for s in edit.segments if s.kind == "zoom")
    assert zoom.hold > 0 and edit.segments[-1].hold == 0
    assert edit.duration >= edit.lines[-1].out_end + 0.5
    assert any("held a frame" in n for n in edit.notes)


def test_outro_runs_into_real_footage_before_freezing():
    lines = [VoiceLine("p1-3", "Charizard!", 0.0, 1.0), VoiceLine("outro", "x", 1.1, 16.0)]
    edit = plan_edit([PACK], PULLS, lines, SRC, EditParams(target=20.0), tail_room=30.0)
    assert all(s.hold == 0 for s in edit.segments)
    assert edit.segments[-1].src_end > PACK.clip_end  # ran past the clip window
    assert edit.duration >= edit.lines[-1].out_end + 0.5


def test_narration_drives_pacing_and_keeps_lines_on_their_cards():
    two_packs = [PackWindow(1, 0.0, 30.0), PackWindow(2, 40.0, 70.0)]
    pulls = PULLS + [
        rp("p2-1", 2, 45.0),
        rp("p2-2", 2, 55.0),
        rp("p2-3", 2, 62.0, rarity="Rare"),
    ]
    lines = [
        VoiceLine("intro", "short hook", 0.0, 1.5),
        VoiceLine("p1-3", "a long bit about Charizard", 1.6, 9.0),  # talky region
        VoiceLine("p2-3", "rare", 9.1, 10.0),
    ]
    edit = plan_edit(two_packs, pulls, lines, SRC, EditParams(target=30.0))
    assert edit.duration == pytest.approx(30.0, abs=0.6)
    assert all(x.drift < 0.5 for x in edit.lines)
    talky = [
        s.speed
        for s in edit.segments
        if s.kind == "reveal"
        and s.pack == 2
        and edit.reveal_at[s.pull_id] < edit.lines[2].out_start
    ]
    quiet = [s.speed for s in edit.segments if s.kind == "reveal" and s.pack == 1]
    assert max(talky) < min(quiet)  # region after Charizard slows down for the long line


def test_repeat_showings_become_showcase_shots():
    from dataclasses import replace

    sir = next(x for x in PULLS if x.id == "p1-3")  # Charizard, revealed at 20 s
    shown = replace(sir, showings=(
        (8.0, 9.0, BOX),     # a preview inside the window, before its reveal
        (25.0, 26.0, BOX),   # held up again inside the window
        (90.0, 91.0, BOX),   # an end recap, outside the pack's window
        (95.0, 96.0, BOX),   # beyond the cap of 2 showings
    ))
    pulls = [x for x in PULLS if x.id != "p1-3"] + [shown]
    segs = build_segments([PACK], pulls, EditParams())
    showcases = [s for s in segs if s.showcase]
    assert [(round(s.src_start, 1), s.kind) for s in showcases] == [(7.7, "hit"), (24.7, "hit")]
    assert segs[-1].pack == 1 and not segs[-1].showcase  # cap: the 90 s recap isn't used

    # with a higher cap the recap is spliced in after the pack
    segs = build_segments([PACK], pulls, EditParams(max_showings=3))
    assert segs[-1].showcase and segs[-1].src_start == pytest.approx(89.7)

    # voice sync and the pack counter key off the reveal, never a preview showing
    lines = [VoiceLine("p1-3", "Charizard!", 0.0, 1.0)]
    edit = plan_edit([PACK], pulls, lines, SRC, EditParams(target=20.0))
    reveal = next(s for s in edit.segments if s.pull_id == "p1-3" and not s.showcase)
    assert edit.reveal_at["p1-3"] >= reveal.out_start
    assert edit.lines[0].out_start == edit.reveal_at["p1-3"]


def test_backwards_mention_follows_on_instead_of_jumping():
    lines = [
        VoiceLine("p1-3", "Charizard", 0.0, 1.0),
        VoiceLine("p1-1", "earlier card", 1.1, 2.0),  # mentions a card already passed
    ]
    edit = plan_edit([PACK], PULLS, lines, SRC, EditParams(target=12.0))
    assert edit.lines[1].out_start == pytest.approx(edit.lines[0].out_end + 0.12)


def test_pack_counter_ticks_up_after_last_card_not_at_video_end():
    edit = plan_edit([PACK], PULLS, [], SRC, EditParams(target=12.0))
    done = edit.pack_done_at[1]
    assert done == pytest.approx(edit.reveal_at["p1-3"] + 1.2)
    assert done < edit.duration


def test_plan_edit_with_intro_offsets_everything():
    lines = [VoiceLine("p1-3", "Charizard!", 0.0, 1.0)]
    edit = plan_edit([PACK], PULLS, lines, SRC, EditParams(target=12.0), intro=1.5)
    assert edit.segments[0].out_start == 1.5
    assert edit.lines[0].out_start == edit.reveal_at["p1-3"] >= 1.5


# --- captions ------------------------------------------------------------------------------


def test_ts_and_chunks():
    assert ts(0) == "0:00:00.00" and ts(65.437) == "0:01:05.44"
    words = [
        TimedWord(text=t, start=i, end=i + 0.5)
        for i, t in enumerate(["No", "way,", "Charizard", "ex", "is", "here"])
    ]
    assert [[w.text for w in c] for c in caption_chunks(words)] == [
        ["No", "way,"],
        ["Charizard", "ex", "is"],
        ["here"],
    ]


def test_build_ass_has_highlighted_words_counter_and_labels():
    edit = plan_edit(
        [PACK], PULLS, [VoiceLine("p1-3", "Charizard ex!", 0.0, 1.0)], SRC, EditParams(target=12.0)
    )
    words = [
        TimedWord(text="Charizard", start=0.0, end=0.5),
        TimedWord(text="ex!", start=0.5, end=1.0),
    ]
    overlay = Overlay(
        hook="No way",
        hook_seconds=3.0,
        master_total=360,
        master_before=10,
        master_after_pack={1: 14},
        new_per_pack={1: 4},
        pack_done_at=edit.pack_done_at,
        labels=[(4.0, 6.0, "SIR · CHARIZARD EX")],
    )
    ass = build_ass(edit.lines, words, overlay, edit.duration)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "{\\c&H0000E5FF&}".replace("&}", "}") in ass or "\\c&H0000E5FF" in ass
    assert "CHARIZARD" in ass and "EX!" in ass
    assert "MASTER SET 10/360" in ass and "MASTER SET 14/360" in ass and "+4 NEW" in ass
    assert "SIR · CHARIZARD EX" in ass and "NO WAY" in ass


# --- ffmpeg commands ------------------------------------------------------------------------


def test_segment_filter_and_voice_filter():
    seg = Segment(1.0, 3.0, "reveal", 1, speed=2.0, crop=(606, 1080, 657, 0), hold=0.5)
    vf = ffmpeg.segment_filter(seg)
    assert vf.startswith("crop=606:1080:657:0,scale=1080:1920")
    assert "setpts=(PTS-STARTPTS)/2.0000" in vf and "tpad=stop_mode=clone:stop_duration=0.500" in vf

    edit = plan_edit(
        [PACK],
        PULLS,
        [VoiceLine("intro", "a", 0.0, 1.0), VoiceLine("p1-3", "b", 1.1, 2.0)],
        SRC,
        EditParams(),
    )
    af = ffmpeg.voice_filter(edit.lines)
    assert "asplit=2" in af and "adelay=0:all=1" in af and "amix=inputs=2" in af


# --- real ffmpeg, tiny synthetic render -----------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")


@needs_ffmpeg
def test_end_to_end_render_of_synthetic_clip(tmp_path):
    src, voice = tmp_path / "src.mp4", tmp_path / "voice.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=30:duration=8",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=3", str(voice)], check=True
    )
    pack = PackWindow(1, 0.0, 8.0)
    pulls = [rp("p1-1", 1, 2.0), rp("p1-2", 1, 4.0, rarity="Rare")]
    edit = plan_edit(
        [pack],
        pulls,
        [VoiceLine("intro", "a", 0.0, 1.0), VoiceLine("p1-2", "b", 1.2, 3.0)],
        (1920, 1080),
        EditParams(target=5.0),
    )
    clips = []
    for i, seg in enumerate(edit.segments):
        clip = tmp_path / f"seg_{i}.mp4"
        ffmpeg.run(ffmpeg.segment_cmd(src, seg, clip))
        clips.append(clip)
    concat = tmp_path / "list.txt"
    concat.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips))
    overlay = Overlay("hook", 1.0, 360, 0, {1: 2}, {1: 2}, edit.pack_done_at, [])
    (tmp_path / "c.ass").write_text(
        build_ass(edit.lines, [], overlay, edit.duration), encoding="utf-8"
    )
    out = tmp_path / "post.mp4"
    ffmpeg.run(
        ffmpeg.final_cmd(concat, voice, edit.lines, "c.ass", edit.duration, out), cwd=tmp_path
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height:format=duration",
            "-of",
            "json",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    import json

    info = json.loads(probe)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in info["streams"])
    assert float(info["format"]["duration"]) == pytest.approx(edit.duration, abs=0.3)
