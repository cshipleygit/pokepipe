"""Moment finding (pure parts) and how the series edit uses the clips."""

import numpy as np
import pytest

from pokepipe.detection.moments import (
    Span,
    energy_start,
    intro_moments,
    loud_runs,
    rip_and_cards_out,
)
from pokepipe.narration.script import call_to_action
from pokepipe.render.timeline import Clip, EditParams, PackWindow, RenderPull, plan_edit

QUIET, LOUD = -75.0, -30.0


def level(spans, length=60.0, hop=0.1):
    db = np.full(int(length / hop), QUIET)
    for a, b in spans:
        db[int(a / hop):int(b / hop)] = LOUD
    return db


def test_loud_runs_bridge_short_dips_and_drop_blips():
    db = level([(10, 12), (12.4, 15), (20, 20.5), (30, 33)])
    runs = loud_runs(db, 0.1)
    assert [(round(r.start, 1), round(r.end, 1)) for r in runs] == [(10.0, 15.0), (30.0, 33.0)]


def test_rip_is_the_last_crinkle_before_the_energy():
    runs = [Span(start=5, end=8), Span(start=20, end=30), Span(start=50, end=55)]
    rip, out = rip_and_cards_out(runs, before=40)
    assert rip.start == 20 and rip.length == pytest.approx(0.9)
    assert out.end == pytest.approx(29.2)  # before the crinkle of putting the wrapper down
    assert rip_and_cards_out([Span(start=50, end=55)], before=40) is None


def test_short_crinkle_is_split_in_half():
    rip, out = rip_and_cards_out([Span(start=10, end=11.2)], before=20)
    assert rip.end == out.start == pytest.approx(10.6)


def test_energy_start_is_the_run_just_before_the_first_card():
    samples = [(t / 4, 10 <= t / 4 <= 11 or 14 <= t / 4 <= 16) for t in range(80)]
    assert energy_start(samples, first_card=16.5) == 14.0
    assert energy_start([(1.0, False)], first_card=5) is None


def test_intro_moments_from_scores_and_sound():
    times = np.arange(0, 150, 0.5)
    p_open = np.where((times >= 128) & (times < 130), 0.95, 0.01)
    p_fan = np.where(times == 142, 0.8, 0.05)
    p_table = np.where(times < 2, 0.9, 0.05)
    runs = [Span(start=18, end=20), Span(start=25, end=31), Span(start=134, end=150)]
    m = intro_moments(times, p_open, p_fan, p_table, runs, end=150)
    assert (m.hero.start, m.seal.start, m.open.end, m.packs.start) == (2.5, 25.5, 128.0, 141.6)


def test_intro_moments_skip_what_is_missing():
    times = np.arange(0, 60, 0.5)
    zero = np.zeros_like(times)
    m = intro_moments(times, zero, zero, zero, [], end=60)
    assert m.open is None and m.packs is None and m.seal is None
    assert m.hero.start == 0.5  # the box is in frame from the start


def rp(pid, start, rarity="Common"):
    return RenderPull(pid, 1, "Mon", rarity, "normal", True, start, start + 1.5,
                      (0.3, 0.3, 0.7, 0.7))


def test_lead_and_recap_clips_play_at_real_speed_around_the_pack():
    pulls = [rp(f"p1-{i}", 5 + 3 * i) for i in range(1, 9)] + [rp("p1-9", 32, "Double Rare")]
    lead = [Clip(1.0, 1.9, "rip"), Clip(2.0, 2.9, "cards_out")]
    recap = [Clip(40.0, 42.0, "recap", "p1-9", (0.3, 0.3, 0.7, 0.7))]
    e = plan_edit([PackWindow(1, 7.0, 36.0)], pulls, [], (1080, 1920),
                  EditParams(target=20), lead=lead, recap=recap)
    clips = [s for s in e.segments if s.kind == "clip"]
    assert [s.label for s in clips] == ["rip", "cards_out", "recap"]
    assert all(s.speed == 1.0 for s in clips)
    assert e.segments[0].kind == e.segments[-1].kind == "clip"
    assert e.reveal_at["p1-9"] < e.segments[-1].out_start  # the recap isn't the reveal
    assert e.duration == pytest.approx(20, abs=1.0)


def test_call_to_action_points_to_the_next_part_then_asks_for_a_follow():
    assert "watch part 2" in call_to_action(1, 2) and "follow" in call_to_action(1, 2)
    assert "watch part" not in call_to_action(2, 2) and "follow" in call_to_action(2, 2)


def test_a_talky_ending_stretches_the_recap_instead_of_freezing():
    from pokepipe.render.timeline import VoiceLine

    pulls = [rp(f"p1-{i}", 5 + 3 * i) for i in range(1, 9)] + [rp("p1-9", 32, "Double Rare")]
    recap = [Clip(40.0, 41.2, "recap", "p1-9", (0.3, 0.3, 0.7, 0.7), max_end=49.0),
             Clip(50.0, 51.2, "recap", "p1-1", (0.3, 0.3, 0.7, 0.7), max_end=51.5)]
    lines = [VoiceLine("p1-9", "big hit", 0.0, 4.0), VoiceLine("outro", "follow", 4.0, 12.0)]
    e = plan_edit([PackWindow(1, 7.0, 36.0)], pulls, lines, (1080, 1920),
                  EditParams(target=20), recap=recap)
    assert all(s.hold == 0 for s in e.segments[:-1])  # no frozen frame mid-video
    clips = [s for s in e.segments if s.label == "recap"]
    assert clips[0].src_len > 1.2 and clips[1].src_end <= 51.5 + 1e-6
    assert any("recap shots longer" in n for n in e.notes)
