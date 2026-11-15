"""Control room: resumable uploads, the access code, status view and the HTTP guard."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pokepipe.control import dashboard
from pokepipe.control.access import MAX_FAILURES, Access, normalise
from pokepipe.control.app import COOKIE, ControlConfig, create_app
from pokepipe.control.uploads import (
    OffsetMismatch,
    UploadError,
    UploadStore,
    check_video,
    safe_filename,
)
from pokepipe.watch.intake import Inbox, JobStore
from pokepipe.watch.models import Job, Status

H = {"X-Pokepipe": "1"}


def _store(tmp_path: Path, probe=None, clock=None) -> UploadStore:
    kwargs = {"clock": clock} if clock else {}
    return UploadStore(tmp_path / "uploads", tmp_path / "inbox",
                       valid_sets=lambda: {"sv3pt5", "me6"}, probe=probe, **kwargs)


# --- uploads -------------------------------------------------------------------------


def test_safe_filename_keeps_a_plain_video_name():
    assert safe_filename("IMG_6054.MOV") == "IMG_6054.mov"
    assert safe_filename("../../evil/../My Opening!!.mp4") == "My Opening.mp4"
    assert safe_filename("C:\\x\\clip.m4v") == "clip.m4v"
    with pytest.raises(UploadError):
        safe_filename("notes.txt")
    with pytest.raises(UploadError):
        safe_filename("no_extension")


def test_chunks_resume_and_land_in_the_inbox(tmp_path):
    store = _store(tmp_path)
    data = b"abcdefghij" * 10
    up = store.start("IMG_1.MOV", len(data), "me6")
    store.write(up.id, 0, data[:40])

    # the phone dropped and re-picked the same file: same upload, carrying on at 40
    again = store.start("IMG_1.MOV", len(data), "me6")
    assert again.id == up.id and again.received == 40

    with pytest.raises(OffsetMismatch) as err:
        store.write(up.id, 10, data[10:20])  # a stale retry
    assert err.value.received == 40

    with pytest.raises(UploadError):
        store.finish(up.id)  # not all there yet
    store.write(up.id, 40, data[40:])
    done = store.finish(up.id)

    dest = tmp_path / "inbox" / "me6" / "IMG_1.mov"
    assert Path(done.inbox_path) == dest and dest.read_bytes() == data
    assert not (tmp_path / "uploads" / f"{up.id}.part").exists()
    assert store.finish(up.id).inbox_path == str(dest)  # finishing twice is harmless


def test_finished_upload_is_taken_by_the_watcher(tmp_path):
    store = _store(tmp_path)
    up = store.start("clip.mp4", 3, "sv3pt5")
    store.write(up.id, 0, b"abc")
    store.finish(up.id)

    t = [0.0]
    jobs = JobStore(tmp_path / "videos")
    inbox = Inbox(tmp_path / "inbox", jobs, settle_seconds=10, clock=lambda: t[0])
    assert inbox.ready() == []
    t[0] = 11
    [candidate] = inbox.ready()
    assert candidate.set_id == "sv3pt5"


def test_same_name_twice_gets_a_new_inbox_name(tmp_path):
    store = _store(tmp_path)
    paths = []
    for _ in range(2):
        up = store.start("clip.mp4", 1, "me6")
        store.write(up.id, 0, b"x")
        paths.append(Path(store.finish(up.id).inbox_path).name)
    assert paths == ["clip.mp4", "clip-2.mp4"]


def test_refuses_unknown_sets_oversize_chunks_and_extra_bytes(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(UploadError, match="unknown set"):
        store.start("a.mp4", 10, "nope")
    with pytest.raises(UploadError, match="unknown set"):
        store.start("a.mp4", 10, "../me6")
    up = store.start("a.mp4", 4, "me6")
    with pytest.raises(UploadError, match="more data"):
        store.write(up.id, 0, b"12345")


def test_probe_flags_a_compressed_video_and_rejects_a_broken_one(tmp_path):
    store = _store(tmp_path, probe=lambda p: check_video(720, 1280, 60.0, 30.0))
    up = store.start("a.mov", 1, "me6")
    store.write(up.id, 0, b"x")
    assert "compressed" in store.finish(up.id).check.warning
    assert check_video(2160, 3840, 900, 59.94).warning is None

    def broken(_):
        raise RuntimeError("moov atom not found")

    bad = _store(tmp_path / "b", probe=broken)
    up = bad.start("a.mov", 1, "me6")
    bad.write(up.id, 0, b"x")
    with pytest.raises(UploadError, match="doesn't open"):
        bad.finish(up.id)


def test_stale_unfinished_uploads_are_dropped(tmp_path):
    now = [datetime(2026, 11, 1, tzinfo=UTC)]
    store = _store(tmp_path, clock=lambda: now[0])
    old = store.start("old.mp4", 10, "me6")
    now[0] += timedelta(days=8)
    store.start("new.mp4", 10, "me6")
    assert [u.name for u in store.all()] == ["new.mp4"]
    assert not (tmp_path / "uploads" / f"{old.id}.part").exists()


def test_cancel_removes_everything(tmp_path):
    store = _store(tmp_path)
    up = store.start("a.mp4", 10, "me6")
    store.cancel(up.id)
    assert store.all() == [] and not any((tmp_path / "uploads").iterdir())


# --- access code ---------------------------------------------------------------------


def test_access_code_signs_in_and_persists(tmp_path):
    path = tmp_path / "access.json"
    access = Access(path)
    assert len(access.code) == 8
    token = access.login(access.code.lower()[:4] + "-" + access.code[4:], "phone")
    assert token and access.valid(token)
    assert token not in path.read_text()  # only a hash is stored

    reopened = Access(path)
    assert reopened.code == access.code and reopened.valid(token)
    reopened.reset()
    assert not reopened.valid(token)


def test_wrong_codes_lock_the_client_out(tmp_path):
    t = [0.0]
    access = Access(tmp_path / "a.json", clock=lambda: t[0])
    for _ in range(MAX_FAILURES):
        assert access.login("WRONGWRONG", "phone") is None
    assert access.login(access.code, "phone") is None  # locked even with the right code
    assert access.login(access.code, "other") is not None
    t[0] = 301
    assert access.login(access.code, "phone") is not None


def test_normalise_ignores_spaces_and_case():
    assert normalise(" ab-cd ef ") == "ABCDEF"


# --- status view ---------------------------------------------------------------------


def _job(jobs: JobStore, slug: str, status: Status, failed_at: Status | None = None) -> Job:
    job = Job(slug=slug, source=f"{slug}.mov", set_id="me6")
    job.move(Status.QUEUED, "taken in")
    if status == Status.FAILED:
        job.move(failed_at)
        job.fail(failed_at, "ffmpeg exploded")
    else:
        job.move(status, "working")
    jobs.save(job)
    return job


def test_status_reads_each_stage_files(tmp_path):
    jobs = JobStore(tmp_path / "videos")
    _job(jobs, "a", Status.AWAITING_REVIEW)
    _job(jobs, "b", Status.FAILED, failed_at=Status.RENDERING)
    vid = tmp_path / "videos" / "a"
    (vid / "pulls.json").write_text(json.dumps({"packs": [{}, {}], "pulls": [{}] * 20}))
    (vid / "review.json").write_text(json.dumps(
        {"completed": False, "pulls": [{"status": "pending"}, {"status": "confirmed"}]}))
    (vid / "costs.json").write_text(json.dumps({"entries": [{"usd": 0.1}, {"usd": 0.234}]}))
    (vid / "posts" / "1").mkdir(parents=True)
    (vid / "posts" / "1" / "post.mp4").write_bytes(b"")
    (vid / "posts" / "2").mkdir()  # not rendered yet
    (tmp_path / "sets" / "me6").mkdir(parents=True)
    (tmp_path / "sets" / "me6" / "set.json").write_text(json.dumps({"id": "me6",
                                                                     "name": "Delta Reign"}))

    out = dashboard.build(jobs, tmp_path / "inbox", tmp_path / "sets", None)
    views = {v["slug"]: v for v in out["videos"]}
    a, b = views["a"], views["b"]
    assert (a["packs"], a["pulls"], a["review_pending"], a["cost_usd"]) == (2, 20, 1, 0.33)
    assert a["posts"] == [1] and a["set_name"] == "Delta Reign"
    assert out["steps"][a["step"]] == "Review"
    assert b["failed"] and out["steps"][b["step"]] == "Rendering"
    assert out["watcher"] == {"running": False}


# --- HTTP ----------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "sets" / "me6").mkdir(parents=True)
    (tmp_path / "sets" / "me6" / "set.json").write_text(json.dumps({"id": "me6", "name": "DR"}))
    (tmp_path / "sets" / "sve").mkdir(parents=True)
    (tmp_path / "sets" / "sve" / "set.json").write_text(json.dumps({"id": "sve", "name": "E"}))
    jobs = JobStore(tmp_path / "videos")
    return ControlConfig(
        jobs=jobs, uploads=_store(tmp_path), access=Access(tmp_path / "access.json"),
        inbox=tmp_path / "inbox", sets_root=tmp_path / "sets", decoy_set="sve",
        lan=True, review_url="http://127.0.0.1:8765/",
    )


def test_phone_must_sign_in(cfg):
    phone = TestClient(create_app(cfg), client=("10.0.0.50", 5000))
    assert phone.get("/api/status").status_code == 401
    assert phone.get("/", follow_redirects=False).headers["location"] == "/login"
    assert phone.get("/login").status_code == 200

    assert phone.post("/api/login", json={"code": "nope"}, headers=H).status_code == 401
    res = phone.post("/api/login", json={"code": cfg.access.code}, headers=H)
    assert res.status_code == 200 and COOKIE in res.cookies
    status = phone.get("/api/status").json()
    assert "pc" not in status  # the code and review link stay on the PC screen


def test_pc_is_trusted_and_sees_the_phone_details(cfg):
    pc = TestClient(create_app(cfg), client=("127.0.0.1", 5000))
    status = pc.get("/api/status").json()
    assert status["pc"]["code"] == cfg.access.code
    assert [s["id"] for s in pc.get("/api/sets").json()["sets"]] == ["me6"]  # no decoys


def test_changes_need_the_custom_header(cfg):
    pc = TestClient(create_app(cfg), client=("127.0.0.1", 5000))
    body = {"name": "a.mp4", "size": 3, "set_id": "me6"}
    assert pc.post("/api/uploads", json=body).status_code == 403
    assert pc.post("/api/uploads", json=body, headers=H).status_code == 201


def test_upload_over_http(cfg, tmp_path):
    pc = TestClient(create_app(cfg), client=("127.0.0.1", 5000))
    up = pc.post("/api/uploads", json={"name": "IMG.MOV", "size": 6, "set_id": "me6"},
                 headers=H).json()
    assert pc.put(f"/api/uploads/{up['id']}?offset=0", content=b"abc", headers=H).json() \
        == {"received": 3, "size": 6}
    stale = pc.put(f"/api/uploads/{up['id']}?offset=0", content=b"abc", headers=H)
    assert stale.status_code == 409 and stale.json()["received"] == 3
    pc.put(f"/api/uploads/{up['id']}?offset=3", content=b"def", headers=H)
    done = pc.post(f"/api/uploads/{up['id']}/finish", headers=H).json()
    assert Path(done["inbox_path"]).read_bytes() == b"abcdef"
    assert pc.post("/api/uploads", json={"name": "x.txt", "size": 1, "set_id": "me6"},
                   headers=H).status_code == 400
    assert pc.get("/api/uploads/../../etc").status_code == 404


def test_retry_and_post_video(cfg):
    _job(cfg.jobs, "vid", Status.FAILED, failed_at=Status.RENDERING)
    post = cfg.jobs.root / "vid" / "posts" / "1"
    post.mkdir(parents=True)
    (post / "post.mp4").write_bytes(b"mp4")
    pc = TestClient(create_app(cfg), client=("127.0.0.1", 5000))
    assert pc.get("/media/vid/posts/1.mp4").content == b"mp4"
    assert pc.get("/media/vid/posts/2.mp4").status_code == 404
    assert pc.post("/api/jobs/vid/retry", headers=H).json() == {"status": "rendering"}
    assert cfg.jobs.load("vid").status == Status.RENDERING
    assert pc.post("/api/jobs/vid/retry", headers=H).status_code == 400  # not failed now
