import json

import pytest
from fastapi.testclient import TestClient

from pokepipe.cards.cache import SetCache
from pokepipe.cards.models import Card, CardSet
from pokepipe.detection.models import DetectionParams, Pack, Pull, PullsFile
from pokepipe.review.app import create_app
from pokepipe.review.models import NewPull, PullUpdate, Status
from pokepipe.review.store import NotFound, ReviewError, ReviewStore
from tests.conftest import png_bytes

SLUG = "test-video"


def pull(i: int, card_id: str, pack: int, t: float) -> Pull:
    return Pull(
        card_id=card_id,
        name=card_id,
        number=card_id.split("-")[1],
        start=t,
        end=t + 1,
        peak_time=t,
        peak_score=0.8,
        mean_margin=0.1,
        frames=4,
        confidence=0.6 + i / 100,
        box=(0, 0, 1, 1),
        pack=pack,
        thumbnail=f"thumbs/p{pack:02d}_{i:03d}_{card_id}.jpg",
    )


@pytest.fixture
def env(tmp_path):
    sets_root, videos_root = tmp_path / "sets", tmp_path / "videos"
    cards = [Card(id=f"s-{n}", name=f"Mon{n}", number=str(n), set_id="s") for n in range(1, 6)]
    cache = SetCache(sets_root, "s")
    cache.save(CardSet(id="s", name="Test"), cards)
    cache.images_dir.mkdir(parents=True)
    cache.image_path("s-1").write_bytes(png_bytes((255, 0, 0)))

    video = tmp_path / "source.mp4"
    video.write_bytes(b"\x00" * 2048)  # fake bytes; enough to test range serving
    pulls = [pull(0, "s-1", 1, 10), pull(1, "s-2", 1, 12), pull(2, "s-3", 2, 40)]
    det = PullsFile(
        video=str(video),
        set_id="s",
        model_tag="m",
        duration=60,
        params=DetectionParams(
            fps=4,
            frame_min_score=0.68,
            frame_min_margin=0.03,
            max_gap=1,
            min_frames=3,
            min_peak=0.72,
        ),
        packs=[
            Pack(index=1, start=10, end=13, clip_start=0, clip_end=15, cards=2, expected_cards=2),
            Pack(index=2, start=40, end=41, clip_start=25, clip_end=43, cards=1, expected_cards=2),
        ],
        pulls=pulls,
    )
    vdir = videos_root / SLUG
    (vdir / "thumbs").mkdir(parents=True)
    (vdir / "pulls.json").write_text(det.model_dump_json(), encoding="utf-8")
    (vdir / pulls[0].thumbnail).write_bytes(b"jpg")
    store = ReviewStore(videos_root, sets_root)
    return store, TestClient(create_app(store)), vdir


# --- store -----------------------------------------------------------------------------


def test_initial_state_mirrors_detection_without_writing(env):
    store, _, vdir = env
    state = store.load(SLUG)
    assert [(p.card_id, p.pack, p.status) for p in state.pulls] == [
        ("s-1", 1, Status.PENDING),
        ("s-2", 1, Status.PENDING),
        ("s-3", 2, Status.PENDING),
    ]
    assert not (vdir / "review.json").exists()


def test_correcting_a_card_confirms_it_and_is_recorded(env):
    store, _, vdir = env
    store.update(SLUG, 1, PullUpdate(card_id="s-4"))
    p = store.load(SLUG).pulls[1]
    assert (p.card_id, p.detected_card_id, p.status, p.corrected) == (
        "s-4",
        "s-2",
        "confirmed",
        True,
    )
    assert json.loads((vdir / "review.json").read_text())["pulls"][1]["card_id"] == "s-4"
    # detection output is never touched
    assert json.loads((vdir / "pulls.json").read_text())["pulls"][1]["card_id"] == "s-2"


def test_unknown_card_and_bad_pack_are_rejected(env):
    store, _, _ = env
    with pytest.raises(ReviewError, match="not a card"):
        store.update(SLUG, 0, PullUpdate(card_id="other-1"))
    with pytest.raises(ReviewError, match="pack"):
        store.add(SLUG, NewPull(card_id="s-1", pack=0))
    with pytest.raises(NotFound):
        store.update(SLUG, 99, PullUpdate(status=Status.CONFIRMED))


def test_pack_summary_counts_added_and_ignores_rejected(env):
    store, _, _ = env
    store.update(SLUG, 0, PullUpdate(status=Status.REJECTED))
    store.add(SLUG, NewPull(card_id="s-5", pack=2, time=42))
    state = store.load(SLUG)
    s1, s2 = store.pack_summaries(state, expected=2)
    assert (s1.cards, s1.pending, s1.complete) == (1, 1, False)
    assert (s2.cards, s2.pending, s2.complete) == (2, 1, True)


def test_only_added_pulls_can_be_deleted(env):
    store, _, _ = env
    added = store.add(SLUG, NewPull(card_id="s-5", pack=1))
    with pytest.raises(ReviewError, match="reject"):
        store.remove(SLUG, 0)
    store.remove(SLUG, added.id)
    assert len(store.load(SLUG).pulls) == 3


def test_complete_requires_no_pending_and_edits_reopen(env):
    store, _, _ = env
    with pytest.raises(ReviewError, match="pending"):
        store.complete(SLUG)
    assert store.confirm_pack(SLUG, 1) == 2
    assert store.confirm_pack(SLUG, 2) == 1
    state = store.complete(SLUG)
    assert state.completed and [p.card_id for p in state.kept()] == ["s-1", "s-2", "s-3"]
    store.update(SLUG, 2, PullUpdate(reverse_holo=True))
    assert store.load(SLUG).completed is False


def test_reverse_guesses_seed_new_reviews_and_prefill_only_pending(env):
    store, _, vdir = env
    det = json.loads((vdir / "pulls.json").read_text())
    det["pulls"][1]["reverse_guess"] = True
    det["pulls"][2]["reverse_guess"] = True
    (vdir / "pulls.json").write_text(json.dumps(det))
    assert [p.reverse_holo for p in store.load(SLUG).pulls] == [False, True, True]

    # an in-progress review saved before guesses existed: prefill skips decided pulls
    store.update(SLUG, 1, PullUpdate(status=Status.CONFIRMED, reverse_holo=False))
    store.update(SLUG, 2, PullUpdate(reverse_holo=False))
    assert store.prefill_reverse(SLUG) == 1
    assert [p.reverse_holo for p in store.load(SLUG).pulls] == [False, False, True]


def test_bad_slugs_never_touch_the_filesystem(env):
    store, _, _ = env
    for slug in ("../etc", "Test-Video", "x" * 100, ""):
        with pytest.raises(NotFound):
            store.video_dir(slug)


# --- API -------------------------------------------------------------------------------


def test_api_lists_videos_and_details(env):
    _, client, _ = env
    [v] = client.get("/api/videos").json()
    assert (v["slug"], v["pulls"], v["pending"], v["completed"]) == (SLUG, 3, 3, False)
    detail = client.get(f"/api/videos/{SLUG}").json()
    assert detail["expected_cards"] == 2
    assert [p["complete"] for p in detail["packs"]] == [True, False]
    assert len(client.get("/api/sets/s/cards").json()) == 5


def test_api_edit_flow(env):
    _, client, _ = env
    r = client.patch(
        f"/api/videos/{SLUG}/pulls/0", json={"status": "confirmed", "reverse_holo": True}
    )
    assert r.status_code == 200 and r.json()["reverse_holo"] is True
    r = client.post(f"/api/videos/{SLUG}/pulls", json={"card_id": "s-5", "pack": 2, "time": 41.5})
    assert r.status_code == 201 and r.json()["status"] == "confirmed"
    assert client.delete(f"/api/videos/{SLUG}/pulls/{r.json()['id']}").status_code == 204
    assert client.post(f"/api/videos/{SLUG}/complete").status_code == 400  # still pending
    client.post(f"/api/videos/{SLUG}/packs/1/confirm")
    client.post(f"/api/videos/{SLUG}/packs/2/confirm")
    assert client.post(f"/api/videos/{SLUG}/complete").json()["completed"] is True


def test_api_errors_map_to_status_codes(env):
    _, client, _ = env
    assert client.get("/api/videos/nope").status_code == 404
    r = client.patch(f"/api/videos/{SLUG}/pulls/0", json={"card_id": "zzz-1"})
    assert r.status_code == 400 and "not a card" in r.json()["detail"]
    assert client.patch(f"/api/videos/{SLUG}/pulls/0", json={"status": "maybe"}).status_code == 422


def test_media_endpoints(env):
    _, client, _ = env
    assert client.get("/").status_code == 200
    js = client.get("/static/app.js")
    assert js.status_code == 200 and js.headers["cache-control"] == "no-cache"
    assert client.get(f"/media/{SLUG}/thumbs/p01_000_s-1.jpg").content == b"jpg"
    assert client.get(f"/media/{SLUG}/thumbs/..%2Fpulls.json").status_code == 404
    assert client.get("/cards/s/s-1.png").headers["content-type"] == "image/png"
    assert client.get("/cards/s/s-2.png").status_code == 404
    ranged = client.get(f"/media/{SLUG}/video", headers={"Range": "bytes=0-99"})
    assert ranged.status_code == 206 and len(ranged.content) == 100
