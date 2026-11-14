"""TikTok sign-in (Login Kit for desktop, PKCE) and inbox draft uploads.

Flow, per TikTok's docs (2026-11):
- `login()`: open https://www.tiktok.com/v2/auth/authorize/ in the browser with a PKCE
  challenge (hex SHA-256 of the verifier, as TikTok's desktop guide requires); a tiny
  local server on the registered redirect URI (localhost / 127.0.0.1 with a port)
  catches the code; it is exchanged at /v2/oauth/token/ for an access token (24 h) and a
  refresh token (365 days), saved to data/tiktok/token.json (gitignored).
- `send_draft()`: POST /v2/post/publish/inbox/video/init/ (scope video.upload) with
  FILE_UPLOAD sizes, then PUT the file to upload_url in sequential chunks. The video
  lands in the account's TikTok inbox; the creator opens the notification, adds the
  caption and posts. Drafts carry no caption.

Limits worth knowing: 6 init calls/minute per token; at most 5 pending shares per
24 hours; upload_url is valid for an hour; videos up to 10 minutes / 4 GB.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import requests

API = "https://open.tiktokapis.com/v2"
AUTHORIZE = "https://www.tiktok.com/v2/auth/authorize/"
SCOPES = "user.info.basic,video.upload"
MIN_CHUNK = 5 * 1024 * 1024
MAX_CHUNK = 64 * 1024 * 1024
CHUNK = 10 * 1024 * 1024  # used when a file is too big to send whole


class TikTokError(RuntimeError):
    pass


def chunk_plan(size: int) -> tuple[int, int]:
    """(chunk_size, total_chunk_count) per the media transfer guide: send whole up to
    64 MB; above that, 10 MB chunks with the remainder folded into the last one."""
    if size <= 0:
        raise TikTokError("empty video file")
    if size <= MAX_CHUNK:
        return size, 1
    return CHUNK, size // CHUNK


def chunks(size: int) -> list[tuple[int, int]]:
    """Inclusive byte ranges for each PUT, in order."""
    chunk_size, count = chunk_plan(size)
    ranges = [(i * chunk_size, (i + 1) * chunk_size - 1) for i in range(count)]
    first, _ = ranges[-1]
    ranges[-1] = (first, size - 1)
    return ranges


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:96]
    return verifier, hashlib.sha256(verifier.encode()).hexdigest()


@dataclass
class Token:
    access_token: str
    refresh_token: str
    open_id: str
    scope: str
    expires_at: float
    refresh_expires_at: float

    @classmethod
    def from_response(cls, data: dict, now: float | None = None) -> Token:
        now = time.time() if now is None else now
        return cls(access_token=data["access_token"], refresh_token=data["refresh_token"],
                   open_id=data.get("open_id", ""), scope=data.get("scope", ""),
                   expires_at=now + float(data.get("expires_in", 0)),
                   refresh_expires_at=now + float(data.get("refresh_expires_in", 0)))


class TokenStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Token | None:
        if not self.path.exists():
            return None
        return Token(**json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, token: Token) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(token.__dict__, indent=2), encoding="utf-8")


class TikTokClient:
    def __init__(self, client_key: str | None, client_secret: str | None, store: TokenStore,
                 session: requests.Session | None = None, timeout: float = 60) -> None:
        if not client_key or not client_secret:
            raise TikTokError("set TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET in .env "
                              "(from your app's Sandbox page on developers.tiktok.com)")
        self.client_key, self.client_secret = client_key, client_secret
        self.store = store
        self.session = session or requests.Session()
        self.timeout = timeout

    # --- sign-in ----------------------------------------------------------------------

    def authorize_url(self, redirect_uri: str, state: str, challenge: str) -> str:
        return AUTHORIZE + "?" + urllib.parse.urlencode({
            "client_key": self.client_key, "scope": SCOPES, "redirect_uri": redirect_uri,
            "state": state, "response_type": "code", "code_challenge": challenge,
            "code_challenge_method": "S256",
        })

    def _token_call(self, form: dict) -> Token:
        resp = self.session.post(f"{API}/oauth/token/", timeout=self.timeout, data={
            "client_key": self.client_key, "client_secret": self.client_secret, **form})
        data = resp.json() if resp.content else {}
        if resp.status_code != 200 or "access_token" not in data:
            raise TikTokError(f"token request failed ({resp.status_code}): "
                              f"{data.get('error_description') or data.get('error') or data}")
        token = Token.from_response(data)
        self.store.save(token)
        return token

    def exchange_code(self, code: str, redirect_uri: str, verifier: str) -> Token:
        return self._token_call({"code": code, "grant_type": "authorization_code",
                                 "redirect_uri": redirect_uri, "code_verifier": verifier})

    def login(self, redirect_uri: str, open_browser=webbrowser.open,
              wait_seconds: float = 300) -> Token:
        """Sign in once in the browser; the token is saved for later runs."""
        parsed = urllib.parse.urlparse(redirect_uri)
        if parsed.hostname not in ("localhost", "127.0.0.1") or not parsed.port:
            raise TikTokError("TIKTOK_REDIRECT_URI must be http://localhost:<port>/... or "
                              "http://127.0.0.1:<port>/... and registered on your app")
        verifier, challenge = pkce_pair()
        state = secrets.token_urlsafe(16)
        got: dict[str, str] = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 — http.server's name
                query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
                if "code" in query or "error" in query:
                    got.update(query)
                ok = "code" in query and query.get("state") == state
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(("<h2>Pokepipe is connected to TikTok. You can close this "
                                  "tab.</h2>" if ok else "<h2>Sign-in failed; see the "
                                  "terminal.</h2>").encode())

            def log_message(self, *args):  # keep the terminal quiet
                pass

        server = http.server.HTTPServer((parsed.hostname, parsed.port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            open_browser(self.authorize_url(redirect_uri, state, challenge))
            deadline = time.time() + wait_seconds
            while not got and time.time() < deadline:
                time.sleep(0.2)
        finally:
            server.shutdown()
        if not got:
            raise TikTokError("no answer from TikTok within 5 minutes; try again")
        if "error" in got:
            why = got.get("error_description") or got["error"]
            raise TikTokError(f"TikTok sign-in failed: {why}")
        if got.get("state") != state:
            raise TikTokError("sign-in answer didn't match this request (state); try again")
        return self.exchange_code(got["code"], redirect_uri, verifier)

    def token(self) -> Token:
        """A valid access token, refreshed when it is about to expire."""
        token = self.store.load()
        if token is None:
            raise TikTokError("not signed in: run `pokepipe tiktok login` first")
        if token.expires_at - time.time() > 120:
            return token
        if token.refresh_expires_at <= time.time():
            raise TikTokError("TikTok sign-in expired: run `pokepipe tiktok login` again")
        return self._token_call({"grant_type": "refresh_token",
                                 "refresh_token": token.refresh_token})

    # --- drafts -----------------------------------------------------------------------

    def _api(self, path: str, body: dict) -> dict:
        resp = self.session.post(f"{API}{path}", json=body, timeout=self.timeout, headers={
            "Authorization": f"Bearer {self.token().access_token}",
            "Content-Type": "application/json; charset=UTF-8"})
        data = resp.json() if resp.content else {}
        err = data.get("error", {})
        if resp.status_code != 200 or err.get("code", "ok") != "ok":
            raise TikTokError(f"{path} failed ({resp.status_code} {err.get('code')}): "
                              f"{err.get('message') or data}")
        return data.get("data", {})

    def send_draft(self, video: Path) -> str:
        """Upload a video to the account's TikTok inbox; returns the publish_id."""
        size = video.stat().st_size
        chunk_size, count = chunk_plan(size)
        data = self._api("/post/publish/inbox/video/init/", {"source_info": {
            "source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk_size,
            "total_chunk_count": count}})
        publish_id, upload_url = data["publish_id"], data["upload_url"]
        with video.open("rb") as f:
            for first, last in chunks(size):
                f.seek(first)
                body = f.read(last - first + 1)
                resp = self.session.put(upload_url, data=body, timeout=600, headers={
                    "Content-Type": "video/mp4", "Content-Length": str(len(body)),
                    "Content-Range": f"bytes {first}-{last}/{size}"})
                if resp.status_code not in (200, 201, 206):
                    raise TikTokError(f"upload of bytes {first}-{last} failed "
                                      f"({resp.status_code}): {resp.text[:200]}")
        return publish_id

    def status(self, publish_id: str) -> dict:
        return self._api("/post/publish/status/fetch/", {"publish_id": publish_id})
