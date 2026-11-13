"""Missed-card recovery: evidence tools, the agent loop (fake Claude), eval, review."""

from __future__ import annotations

import json
from types import SimpleNamespace

from pokepipe.cards.index import Match
from pokepipe.cards.models import Card
from pokepipe.detection.layout import MODERN, apply_layout
from pokepipe.detection.matcher import FrameResult
from pokepipe.detection.models import DetectionParams, Pack, Pull, PullsFile
from pokepipe.recover.agent import AgentSettings, build_agent, kickoff
from pokepipe.recover.evidence import Evidence, near_misses
from pokepipe.recover.models import RecoveredFile, Suggestion
from pokepipe.recover.service import evaluate
from pokepipe.review.models import ReviewedPull, ReviewState
from pokepipe.review.models import Status as RS

C, U, R, IR = "Common", "Uncommon", "Rare", "Illustration Rare"

CARDS = {
    c.id: c for c in [
        *(Card(id=f"s-c{i}", name=f"Common {i}", number=str(i), set_id="s", rarity=C)
          for i in range(1, 6)),
        *(Card(id=f"s-u{i}", name=f"Uncommon {i}", number=str(10 + i), set_id="s",
               rarity=U) for i in range(1, 5)),
        Card(id="s-u2ir", name="Uncommon 2", number="90", set_id="s", rarity=IR),
        Card(id="s-r1", name="Rare 1", number="20", set_id="s", rarity=R),
    ]
}


def pull(card_id: str, t: float, alts=()) -> Pull:
    card = CARDS[card_id]
    return Pull(card_id=card_id, name=card.name, number=card.number, rarity=card.rarity,
                start=t, end=t + 0.8, peak_time=t + 0.4, peak_score=0.82, mean_margin=0.1,
                frames=4, confidence=0.9, box=(0, 0, 1, 1), pack=1, alternatives=list(alts))


def short_pack() -> PullsFile:
    """Pack 1 with slot 6 (an Uncommon) missing: detection lost a near-tie at ~6 s."""
    order = ["s-c1", "s-c2", "s-c3", "s-c4", "s-u1", "s-u3", "s-c5", "s-u4", "s-r1"]
    times = [1, 2, 3, 4, 5, 7.5, 8.5, 9.5, 10.5]
    pulls = [pull(c, t) for c, t in zip(order, times, strict=True)]
    apply_layout(pulls, MODERN)
    return PullsFile(
        video="v.mp4", set_id="s", model_tag="fake", duration=60.0,
        params=DetectionParams(fps=4, frame_min_score=0.68, frame_min_margin=0.03,
                               max_gap=1, min_frames=3, min_peak=0.72),
        pack_layout="modern",
        packs=[Pack(index=1, start=1, end=11.3, clip_start=0, clip_end=13, cards=9,
                    expected_cards=10)],
        pulls=pulls,
    )


def fr(t: float, *top: tuple[str, float]) -> FrameResult:
    return FrameResult(int(t * 8), t, [Match(c, s) for c, s in top], (0, 0, 1, 1))


class FakeScanner:
    def __init__(self) -> None:
        self.calls = []

    def frames(self, start, end, fps):
        self.calls.append((start, end, fps))
        out = [fr(5.2 + i / 8, ("s-u1", 0.83), ("s-u2", 0.6)) for i in range(4)]
        # The near-tie: Uncommon 2 and its IR trade first place frame to frame.
        for i in range(8):
            a, b = ("s-u2", "s-u2ir") if i % 2 else ("s-u2ir", "s-u2")
            out.append(fr(6.2 + i / 8, (a, 0.79), (b, 0.785), ("s-c1", 0.5)))
        return out


def evidence(scanner=None) -> Evidence:
    return Evidence(short_pack(), CARDS, MODERN, scanner or FakeScanner())


# --- evidence ---------------------------------------------------------------------


def test_near_misses_keep_an_interleaved_tie_together():
    frames = FakeScanner().frames(0, 10, 8)
    misses = near_misses(frames)
    assert len(misses) == 2  # Uncommon 1, then one stretch for the tie (not two)
    tie = misses[1].candidates()
    assert {tie[0][0], tie[1][0]} == {"s-u2", "s-u2ir"}
    assert abs(tie[0][1] - tie[1][1]) < 0.01


def test_overview_names_the_missing_slots():
    ov = evidence().pack_overview(1)
    assert ov["found"] == 9
    missing = ov["missing_slots"]
    assert any(m["kind"] == "uncommon" for m in missing)
    assert ov["gaps"][4]["seconds"] > 1.5  # the hole between Uncommon 1 and 3


def test_scan_flags_already_detected_cards_and_counts_down():
    ev = evidence()
    out = ev.scan_window(1, None, None, 8)
    first, tie = out["card_like_stretches"]
    assert first["already_detected_as"] == "s-u1"
    assert tie["already_detected_as"] is None and tie["lead"] < 0.03
    assert out["scans_left"] == 7


def test_check_fit_uses_the_layout_to_settle_the_tie():
    ev = evidence()
    uncommon = ev.check_fit(1, [("s-u2", 6.5)])
    assert uncommon["fits"] and uncommon["pack_would_have"] == 10
    assert uncommon["added"][0]["slot"] == 6 and not uncommon["added"][0]["reverse_holo"]
    ir = ev.check_fit(1, [("s-u2ir", 6.5)])
    assert not ir["fits"]  # an IR can't sit between two Uncommons


def test_check_fit_takes_two_cards_held_up_together():
    """Delta Reign pack 10: both reverse slots missing, shown side by side, one of them
    a second copy of an Uncommon already in the pack."""
    det = short_pack()
    order = ["s-c1", "s-c2", "s-c3", "s-c4", "s-u1", "s-u2", "s-u3", "s-r1"]  # no 8, 9
    det.pulls = [pull(c, t) for c, t in zip(order, [1, 2, 3, 4, 5, 6.5, 7.5, 10.5],
                                            strict=True)]
    det.packs[0].cards = 8
    ev = Evidence(det, CARDS, MODERN, FakeScanner())
    out = ev.check_fit(1, [("s-c1", 9.5), ("s-u3", 9.6)])
    assert out["fits"] and out["mode"] == "full"
    assert [(a["slot"], a["reverse_holo"]) for a in out["added"]] == [(8, True), (9, True)]
    assert out["added"][1]["already_in_pack_at"] == [7.5]  # a second copy, flagged


# --- agent loop with a scripted model ------------------------------------------------


def tool_use(id_: str, name: str, args: dict):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


def reply(*blocks, stop="tool_use"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                                                 cache_creation_input_tokens=0,
                                                 cache_read_input_tokens=0))


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        return self.replies.pop(0)


def run_agent(client, ev=None, budget=lambda e: None, max_turns=14):
    graph = build_agent(client, ev or evidence(), AgentSettings(max_turns=max_turns), budget)
    return graph.invoke({"messages": [kickoff("v", "Test")]}, {"recursion_limit": 100})


def test_agent_investigates_then_submits():
    found = {"pack": 1, "card_id": "s-u2", "time": 6.5, "reverse_holo": False,
             "confidence": "high", "evidence": "tie with IR; slot 6 must be Uncommon"}
    client = FakeClient([
        reply(tool_use("a", "short_packs", {}), tool_use("b", "pack_overview", {"pack": 1})),
        reply(tool_use("c", "scan_window", {"pack": 1, "start": None, "end": None, "fps": 8})),
        reply(tool_use("d", "check_fit", {"pack": 1, "add": [{"card_id": "s-u2ir",
                                                               "time": 6.5}]}),
              tool_use("e", "check_fit", {"pack": 1, "add": [{"card_id": "s-u2",
                                                               "time": 6.5}]})),
        reply(tool_use("f", "submit_findings", {"suggestions": [found], "notes": ""})),
    ])
    state = run_agent(client)
    assert state["done"] and state["turns"] == 4
    assert [s.card_id for s in state["suggestions"]] == ["s-u2"]
    assert [c.name for c in state["calls"]] == [
        "short_packs", "pack_overview", "scan_window", "check_fit", "check_fit",
        "submit_findings"]
    # Parallel tool calls come back as one message, every result for its tool_use id.
    second = client.requests[1]["messages"][-1]
    assert [r["tool_use_id"] for r in second["content"]] == ["a", "b"]
    assert json.loads(second["content"][0]["content"])["short_packs"][0]["pack"] == 1
    assert client.requests[0]["fallbacks"] == "default"


def test_tool_errors_go_back_to_the_model_not_up():
    client = FakeClient([
        reply(tool_use("a", "check_fit", {"pack": 1, "add": [{"card_id": "nope",
                                                               "time": 1}]})),
        reply(tool_use("b", "submit_findings", {"suggestions": [], "notes": "none"})),
    ])
    state = run_agent(client)
    err = client.requests[1]["messages"][-1]["content"][0]
    assert err["is_error"] and "not a card" in err["content"]
    assert state["calls"][0].ok is False and state["suggestions"] == []


def test_turn_limit_is_a_graph_edge():
    loop = [reply(tool_use(str(i), "short_packs", {})) for i in range(3)]
    client = FakeClient(loop)
    state = run_agent(client, max_turns=3)
    assert state["turns"] == 3 and not state.get("done")
    last_user = client.requests[-1]["messages"][-1]["content"]
    assert last_user[-1]["text"].startswith("Turn limit reached")


def test_budget_stop_and_refusal_end_cleanly():
    def broke(estimate):
        raise RuntimeError("over budget")

    state = run_agent(FakeClient([]), budget=broke)
    assert state["done"] and "over budget" in state["error"]

    state = run_agent(FakeClient([reply(stop="refusal")]))
    assert state["done"] and "refusal" in state["error"]


# --- eval and review ------------------------------------------------------------------


def review_with(*pulls: ReviewedPull) -> ReviewState:
    return ReviewState(video_slug="v", set_id="s", pulls=list(pulls), completed=True)


def test_eval_scores_against_hand_added_pulls():
    review = review_with(
        ReviewedPull(id=0, detected_index=0, detected_card_id="s-c1", card_id="s-c1", pack=1,
                     time=1, status=RS.CONFIRMED),
        ReviewedPull(id=1, card_id="s-u2", pack=1, time=6.5, status=RS.CONFIRMED),
        ReviewedPull(id=2, card_id="s-c5", pack=2, time=20, status=RS.CONFIRMED),
        # an agent suggestion the reviewer kept is not ground truth for grading itself
        ReviewedPull(id=3, card_id="s-u4", pack=2, time=21, status=RS.CONFIRMED,
                     suggested="high confidence: ..."),
    )
    rec = RecoveredFile(video_slug="v", model="m", suggestions=[
        Suggestion(pack=1, card_id="s-u2", time=6.5, confidence="high", evidence=""),
        Suggestion(pack=2, card_id="s-u3", time=22, confidence="medium", evidence=""),
    ])
    r = evaluate(rec, review)
    assert r.found == [(1, "s-u2")] and r.wrong == [(2, "s-u3")] and r.missed == [(2, "s-c5")]
    assert r.recall == 0.5 and r.precision == 0.5


def test_suggestions_seed_a_new_review_as_pending(tmp_path):
    from pokepipe.cards.cache import SetCache
    from pokepipe.cards.models import CardSet
    from pokepipe.review.store import ReviewStore

    sets, videos = tmp_path / "sets", tmp_path / "videos"
    cache = SetCache(sets, "s")
    cache.save(CardSet(id="s", name="Test", series="Scarlet & Violet"), list(CARDS.values()))
    root = videos / "v"
    root.mkdir(parents=True)
    (root / "pulls.json").write_text(short_pack().model_dump_json(), encoding="utf-8")
    rec = RecoveredFile(video_slug="v", model="m", suggestions=[
        Suggestion(pack=1, card_id="s-u2", time=6.5, confidence="high", evidence="slot 6"),
        Suggestion(pack=1, card_id="made-up", time=7, confidence="high", evidence="?"),
    ])
    (root / "recovered.json").write_text(rec.model_dump_json(), encoding="utf-8")

    state = ReviewStore(videos, sets).load("v")
    added = [p for p in state.pulls if p.detected_index is None]
    assert [(p.card_id, p.status, p.pack) for p in added] == [("s-u2", RS.PENDING, 1)]
    assert "slot 6" in added[0].suggested
    assert not state.completed
