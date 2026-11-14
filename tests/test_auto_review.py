import json

import pytest

from pokepipe.cards.cache import SetCache
from pokepipe.cards.models import Card, CardSet
from pokepipe.detection.models import Alternative, DetectionParams, Pack, Pull, PullsFile
from pokepipe.recover.models import RecoveredFile, Suggestion
from pokepipe.review import auto, calibrate
from pokepipe.review.auto import AutoPolicy, auto_review
from pokepipe.review.models import NewPull, PullUpdate, Status
from pokepipe.review.replay import replay
from pokepipe.review.store import ReviewStore

SLUG = "vid"
RARITY = {**{n: "Common" for n in range(1, 7)}, **{n: "Uncommon" for n in range(7, 11)},
          11: "Rare", 12: "Rare", 13: "Double Rare", 14: "Illustration Rare"}
# a normal modern pack in reveal order: 4 C, 3 U, two reverse slots, the rare slot
PACK = [1, 2, 3, 4, 7, 8, 9, 5, 10, 13]
REVERSE_SLOTS = {7, 8}  # 0-based positions of the reverse-holo slots


def pull(n: int, pack: int, t: float, slot: int, conf: float = 0.9,
         runner_up: float = 0.6) -> Pull:
    cid = f"s-{n}"
    return Pull(
        card_id=cid, name=f"Mon{n}", number=str(n), rarity=RARITY[n], start=t, end=t + 1,
        peak_time=t, peak_score=0.85, mean_margin=0.1, frames=8, confidence=conf,
        box=(0, 0, 1, 1), pack=pack, slot=slot + 1,
        reverse_guess=slot in REVERSE_SLOTS and RARITY[n] in ("Common", "Uncommon", "Rare"),
        alternatives=[Alternative(card_id=cid, score=0.85),
                      Alternative(card_id="s-6" if n != 6 else "s-5", score=runner_up)],
    )


def make_pack(pack: int, t0: float, cards=PACK, skip=(), tweak=None) -> list[Pull]:
    tweak = tweak or {}
    return [pull(n, pack, t0 + 2 * i, i, **tweak.get(i, {}))
            for i, n in enumerate(cards) if i not in skip]


def make_env(tmp_path):
    """(store, write(packs, layouts, suggestions), video dir) for a test set of 14 cards."""
    sets_root, videos_root = tmp_path / "sets", tmp_path / "videos"
    cards = [Card(id=f"s-{n}", name=f"Mon{n}", number=str(n), set_id="s", rarity=r)
             for n, r in RARITY.items()]
    SetCache(sets_root, "s").save(CardSet(id="s", name="Test"), cards)
    store = ReviewStore(videos_root, sets_root)

    def write(packs: list[list[Pull]], layouts=None, suggestions=()):
        pulls = [p for pk in packs for p in pk]
        det = PullsFile(
            video="x.mp4", set_id="s", model_tag="m", duration=600, pack_layout="modern",
            params=DetectionParams(fps=4, frame_min_score=0.68, frame_min_margin=0.03,
                                   max_gap=1, min_frames=3, min_peak=0.72),
            packs=[Pack(index=i + 1, start=pk[0].start, end=pk[-1].end, clip_start=0,
                        clip_end=1, cards=len(pk), expected_cards=10,
                        layout=(layouts or {}).get(i + 1, "full" if len(pk) == 10 else
                                                   "partial"))
                   for i, pk in enumerate(packs)],
            pulls=pulls,
        )
        vdir = videos_root / SLUG
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "pulls.json").write_text(det.model_dump_json(), encoding="utf-8")
        if suggestions:
            rec = RecoveredFile(video_slug=SLUG, model="m", suggestions=list(suggestions))
            (vdir / "recovered.json").write_text(rec.model_dump_json(), encoding="utf-8")
        return det

    return store, write, videos_root / SLUG


@pytest.fixture
def env(tmp_path):
    return make_env(tmp_path)


def decide(store, mode="exceptions", policy=AutoPolicy()):
    det = store.detection(SLUG)
    state = store.initial_state(SLUG)
    return auto_review(det, state, store.cards("s"), policy, mode)


# --- deciding ---------------------------------------------------------------------------


def test_clean_packs_are_confirmed_with_reverse_holos_kept(env):
    store, write, _ = env
    write([make_pack(1, 10), make_pack(2, 100)])
    res = decide(store)
    assert res.clean and res.confirmed == 20
    assert all(p.status == Status.CONFIRMED and p.decided_by == "auto" for p in res.state.pulls)
    assert sum(p.reverse_holo for p in res.state.pulls) == 4


def test_a_low_confidence_card_sends_its_whole_pack_to_a_person(env):
    store, write, _ = env
    write([make_pack(1, 10, tweak={3: {"conf": 0.4}}), make_pack(2, 100)])
    res = decide(store)
    pack1 = [p for p in res.state.pulls if p.pack == 1]
    assert all(p.status == Status.PENDING for p in pack1)
    assert "confidence 0.40" in pack1[0].auto_note
    assert res.confirmed == 10 and len(res.exceptions) == 1


def test_near_identical_runner_up_is_escalated(env):
    store, write, _ = env
    write([make_pack(1, 10, tweak={9: {"runner_up": 0.845}})])
    res = decide(store)
    assert not res.clean and "near-identical" in res.exceptions[0]


def test_short_pack_completed_by_a_high_confidence_suggestion(env):
    store, write, _ = env
    # slot 9 (a reverse-holo Uncommon) was missed; the agent found it
    write([make_pack(1, 10, skip={8})], suggestions=[Suggestion(
        pack=1, card_id="s-10", time=26.0, reverse_holo=True, confidence="high",
        evidence="rescan")])
    res = decide(store)
    assert res.clean and res.confirmed == 10
    added = next(p for p in res.state.pulls if p.detected_index is None)
    assert added.status == Status.CONFIRMED and added.reverse_holo
    # re-inferred over the full pack: slot 8 (Common s-5) is a reverse holo too
    assert next(p for p in res.state.pulls if p.card_id == "s-5").reverse_holo


def test_medium_suggestion_and_short_pack_wait_for_a_person(env):
    store, write, _ = env
    write([make_pack(1, 10, skip={8})], suggestions=[Suggestion(
        pack=1, card_id="s-10", time=26.0, confidence="medium", evidence="maybe")])
    res = decide(store)
    assert {"medium" in e for e in res.exceptions} == {True, False}
    assert all(p.status == Status.PENDING for p in res.state.pulls)


def test_no_layout_means_reverse_holos_unknown(env):
    store, write, _ = env
    write([make_pack(1, 10)], layouts={1: None})
    assert "reverse holos are unknown" in decide(store).exceptions[0]
    assert decide(store, policy=AutoPolicy(require_layout=False)).clean


def test_off_mode_decides_everything(env):
    store, write, _ = env
    write([make_pack(1, 10, skip={8}, tweak={0: {"conf": 0.3}})], suggestions=[Suggestion(
        pack=1, card_id="s-10", time=26.0, confidence="medium", evidence="maybe")])
    res = decide(store, mode="off")
    assert not any(p.status == Status.PENDING for p in res.state.pulls)
    agent = next(p for p in res.state.pulls if p.detected_index is None)
    assert agent.status == Status.REJECTED and "UNCHECKED" in agent.auto_note
    assert all("UNCHECKED" in p.auto_note for p in res.state.pulls)


# --- the pipeline step -------------------------------------------------------------------


def test_run_completes_a_clean_review(env):
    store, write, vdir = env
    write([make_pack(1, 10)])
    res = auto.run(store, SLUG, "exceptions", AutoPolicy())
    state = store.load(SLUG)
    assert res.clean and state.completed and state.auto.completed_by_auto
    assert len(state.kept()) == 10


def test_run_leaves_exceptions_pending_and_never_overwrites_a_review(env):
    store, write, vdir = env
    write([make_pack(1, 10), make_pack(2, 100, tweak={0: {"conf": 0.2}})])
    auto.run(store, SLUG, "exceptions", AutoPolicy())
    state = store.load(SLUG)
    assert not state.completed and state.auto.exceptions
    store.confirm_pack(SLUG, 2)  # a person finishes pack 2
    before = (vdir / "review.json").read_text()
    assert auto.run(store, SLUG, "exceptions", AutoPolicy()) is None
    assert (vdir / "review.json").read_text() == before
    assert all(p.decided_by is None for p in store.load(SLUG).pulls if p.pack == 2)


def test_shadow_mode_records_predictions_but_leaves_the_review_alone(env):
    store, write, vdir = env
    write([make_pack(1, 10)])
    auto.run(store, SLUG, "shadow", AutoPolicy())
    assert not (vdir / "review.json").exists()
    shadow = json.loads((vdir / auto.SHADOW_FILE).read_text())
    assert shadow["auto"]["mode"] == "shadow" and shadow["auto"]["confirmed"] == 10
    assert auto.run(store, SLUG, "always", AutoPolicy()) is None


def test_a_human_edit_takes_over_an_auto_decision(env):
    store, write, _ = env
    write([make_pack(1, 10)])
    auto.run(store, SLUG, "exceptions", AutoPolicy())
    store.update(SLUG, 0, PullUpdate(reverse_holo=True))
    p = store.load(SLUG).pulls[0]
    assert p.decided_by is None and not store.load(SLUG).completed


# --- replay -------------------------------------------------------------------------------


def human_review(store, edits=()):
    for pack in sorted({p.pack for p in store.load(SLUG).pulls}):
        store.confirm_pack(SLUG, pack)
    for fn in edits:
        fn()
    store.complete(SLUG)


def test_replay_agrees_with_a_matching_human_review(env):
    store, write, _ = env
    write([make_pack(1, 10), make_pack(2, 100, tweak={4: {"conf": 0.3}})])
    human_review(store)
    r = replay(store, SLUG, AutoPolicy())
    assert r.silent_errors == [] and r.auto_confirmed == 10 and r.escalated == 10
    assert r.escalated_but_right == 10 and r.clean_packs == 1


def test_replay_catches_silent_errors(env):
    store, write, _ = env
    write([make_pack(1, 10)])
    human_review(store, edits=[
        lambda: store.update(SLUG, 2, PullUpdate(status=Status.REJECTED)),
        lambda: store.update(SLUG, 3, PullUpdate(card_id="s-6")),
        lambda: store.add(SLUG, NewPull(card_id="s-12", pack=1, time=30)),
    ])
    errs = replay(store, SLUG, AutoPolicy()).silent_errors
    assert len(errs) == 3
    assert any("rejected" in e for e in errs) and any("person chose Mon6" in e for e in errs)
    assert any("person added it" in e for e in errs)


def test_replay_refuses_unfinished_or_auto_decided_reviews(env):
    store, write, _ = env
    write([make_pack(1, 10)])
    with pytest.raises(ValueError, match="isn't finished"):
        replay(store, SLUG, AutoPolicy())
    auto.run(store, SLUG, "exceptions", AutoPolicy())
    with pytest.raises(ValueError, match="auto decisions"):
        replay(store, SLUG, AutoPolicy())


# --- calibration --------------------------------------------------------------------------


def test_fit_learns_a_threshold_that_lets_no_known_error_through(env):
    store, write, _ = env
    weak = {i: {"conf": 0.3, "runner_up": 0.84} for i in (0, 4, 9)}
    write([make_pack(1, 10, tweak=weak), make_pack(2, 100), make_pack(3, 200)])
    det = store.detection(SLUG)
    for d in det.pulls:  # the weak ones also have weak scores
        if d.confidence < 0.5:
            d.peak_score, d.mean_margin, d.frames = 0.73, 0.01, 3
    (store.video_dir(SLUG) / "pulls.json").write_text(det.model_dump_json())
    human_review(store, edits=[
        lambda: store.update(SLUG, 0, PullUpdate(status=Status.REJECTED)),
        lambda: store.update(SLUG, 4, PullUpdate(card_id="s-6")),
        lambda: store.update(SLUG, 9, PullUpdate(status=Status.REJECTED)),
    ])
    data = calibrate.examples(store.detection(SLUG), store.load(SLUG))
    assert len(data) == 30 and sum(not e.right for e in data) == 3
    cal = calibrate.fit(data)
    wrong = [p for i, p in enumerate(det.pulls) if i in (0, 4, 9)]
    right = [p for i, p in enumerate(det.pulls) if i not in (0, 4, 9)]
    assert all(cal.probability(p) < cal.threshold for p in wrong)
    assert all(cal.probability(p) >= cal.threshold for p in right)
    path = store.videos_root / "cal.json"
    cal.save(path)
    assert calibrate.Calibration.load(path).threshold == cal.threshold
    # the auto-reviewer uses it in place of the raw confidence
    res = decide(store, policy=AutoPolicy(min_confidence=0.0, min_lead=0.0, calibration=cal))
    assert {p.pack for p in res.state.pulls if p.status == Status.PENDING} == {1}
    table = calibrate.threshold_table(data, [0.25, 0.5])
    assert table == [(0.25, 3, 0), (0.5, 0, 0)]


def test_fit_needs_some_mistakes_to_learn_from(env):
    store, write, _ = env
    write([make_pack(1, 10)])
    human_review(store)
    with pytest.raises(ValueError, match="need"):
        calibrate.fit(calibrate.examples(store.detection(SLUG), store.load(SLUG)))
