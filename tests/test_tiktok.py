import hashlib
import json
import threading
import time
import urllib.parse

import pytest
import requests
import responses

from pokepipe.narration.models import Post, PostPlan
from pokepipe.tiktok.client import (
    API,
    CHUNK,
    MAX_CHUNK,
    TikTokClient,
    TikTokError,
    Token,
    TokenStore,
    chunk_plan,
    chunks,
)
from pokepipe.tiktok.service import send_drafts

TOKEN = {"access_token": "act.1", "refresh_token": "rft.1", "open_id": "o1",
         "scope": "user.info.basic,video.upload", "expires_in": 86400,
         "refresh_expires_in": 31536000, "token_type": "Bearer"}


def client(tmp_path, token: Token | None = None):
    store = TokenStore(tmp_path / "token.json")
    if token:
        store.save(token)
    return TikTokClient("key", "secret", store)


def fresh():
    return Token.from_response(TOKEN)


def test_chunk_plan_follows_the_media_transfer_rules():
    assert chunk_plan(3_000_000) == (3_000_000, 1)  # small: whole
    assert chunk_plan(MAX_CHUNK) == (MAX_CHUNK, 1)
    size = MAX_CHUNK + 12_345
    assert chunk_plan(size) == (CHUNK, size // CHUNK)
    ranges = chunks(size)
    assert ranges[0] == (0, CHUNK - 1) and ranges[-1][1] == size - 1
    assert all(b + 1 == c for (_, b), (c, _) in zip(ranges, ranges[1:], strict=False))
    assert ranges[-1][1] - ranges[-1][0] + 1 < 2 * CHUNK  # last chunk well under 128 MB


def test_needs_keys():
    with pytest.raises(TikTokError, match="TIKTOK_CLIENT_KEY"):
        TikTokClient(None, None, TokenStore(None))


@responses.activate
def test_login_round_trip_with_pkce(tmp_path):
    seen = {}
    responses.add(responses.POST, f"{API}/oauth/token/", json=TOKEN)
    responses.add_passthru("http://127.0.0.1")
    redirect = "http://127.0.0.1:38457/callback/"

    def browser(url):  # play TikTok: send the user back with a code
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        seen.update(q)
        threading.Thread(target=lambda: (time.sleep(0.2), requests.get(
            f"{redirect}?code=abc&scopes=video.upload&state={q['state']}"))).start()

    token = client(tmp_path).login(redirect, open_browser=browser, wait_seconds=10)
    assert token.access_token == "act.1"
    assert seen["code_challenge_method"] == "S256" and seen["redirect_uri"] == redirect
    form = dict(urllib.parse.parse_qsl(responses.calls[-1].request.body))
    assert form["code"] == "abc" and form["grant_type"] == "authorization_code"
    assert hashlib.sha256(form["code_verifier"].encode()).hexdigest() == seen["code_challenge"]
    assert TokenStore(tmp_path / "token.json").load().refresh_token == "rft.1"


def test_login_refuses_a_non_local_redirect(tmp_path):
    with pytest.raises(TikTokError, match="127.0.0.1"):
        client(tmp_path).login("https://example.com/cb")


@responses.activate
def test_expired_access_token_is_refreshed(tmp_path):
    old = fresh()
    old.expires_at = time.time() - 10
    responses.add(responses.POST, f"{API}/oauth/token/",
                  json={**TOKEN, "access_token": "act.2", "refresh_token": "rft.2"})
    c = client(tmp_path, old)
    assert c.token().access_token == "act.2"
    form = dict(urllib.parse.parse_qsl(responses.calls[0].request.body))
    assert form["grant_type"] == "refresh_token" and form["refresh_token"] == "rft.1"
    assert c.store.load().refresh_token == "rft.2"  # keep the newly returned one


@responses.activate
def test_send_draft_inits_then_uploads_every_byte(tmp_path):
    video = tmp_path / "post.mp4"
    video.write_bytes(b"x" * 1000)
    upload = "https://upload.example/video?token=t"
    responses.add(responses.POST, f"{API}/post/publish/inbox/video/init/",
                  json={"data": {"publish_id": "v_inbox.1", "upload_url": upload},
                        "error": {"code": "ok", "message": ""}})
    responses.add(responses.PUT, upload, status=201)
    assert client(tmp_path, fresh()).send_draft(video) == "v_inbox.1"
    init = json.loads(responses.calls[0].request.body)
    assert init["source_info"] == {"source": "FILE_UPLOAD", "video_size": 1000,
                                   "chunk_size": 1000, "total_chunk_count": 1}
    assert responses.calls[0].request.headers["Authorization"] == "Bearer act.1"
    put = responses.calls[1].request
    assert put.headers["Content-Range"] == "bytes 0-999/1000" and len(put.body) == 1000


@responses.activate
def test_api_errors_are_explained(tmp_path):
    responses.add(responses.POST, f"{API}/post/publish/inbox/video/init/", status=403,
                  json={"error": {"code": "spam_risk_too_many_pending_share",
                                  "message": "too many pending"}})
    video = tmp_path / "post.mp4"
    video.write_bytes(b"x")
    with pytest.raises(TikTokError, match="spam_risk_too_many_pending_share"):
        client(tmp_path, fresh()).send_draft(video)


def test_send_drafts_sends_each_rendered_post_once(tmp_path):
    vdir = tmp_path / "vid"
    plan = PostPlan(video_slug="vid", posts=[Post(index=1, packs=[1]), Post(index=2, packs=[2])])
    (vdir / "posts" / "1").mkdir(parents=True)
    (vdir / "posts.json").write_text(plan.model_dump_json())
    (vdir / "posts" / "1" / "post.mp4").write_bytes(b"v")

    class Fake:
        calls = 0

        def send_draft(self, video):
            Fake.calls += 1
            return f"id{Fake.calls}"

    logs: list[str] = []
    assert send_drafts(vdir, Fake(), log=logs.append) == [1]  # post 2 isn't rendered
    assert json.loads((vdir / "posts" / "1" / "tiktok.json").read_text())["publish_id"] == "id1"
    assert send_drafts(vdir, Fake(), log=logs.append) == []  # never twice
    assert Fake.calls == 1 and any("already sent" in m for m in logs)
