"""FastAPI app for the control room. Run via `pokepipe watch run --lan` or `pokepipe ui serve`.

Security model (it may listen on the home network):
- Requests from the PC itself (loopback) are trusted; anything else needs a session
  cookie from the sign-in page (access code, see access.py).
- Every request that changes something must carry the `X-Pokepipe` header. A web page
  on another site can't add a custom header without a CORS preflight this app never
  approves, so a malicious page open on the PC can't use the loopback trust.
- Nothing here serves source footage or edits reviews; that stays on the review page.
"""

from __future__ import annotations

import re
import socket
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pokepipe.control import dashboard
from pokepipe.control.access import Access
from pokepipe.control.uploads import (
    MAX_CHUNK_BYTES,
    OffsetMismatch,
    UploadError,
    UploadNotFound,
    UploadStore,
)
from pokepipe.watch.intake import JobStore
from pokepipe.watch.models import Status
from pokepipe.watch.runner import Runner

STATIC_DIR = Path(__file__).parent / "static"
COOKIE = "pokepipe_session"
CSRF_HEADER = "x-pokepipe"
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
SAFE_SLUG = re.compile(r"^[a-z0-9-]{1,80}$")
OPEN_PATHS = {"/login", "/api/login", "/static/style.css", "/static/login.js"}


@dataclass
class ControlConfig:
    jobs: JobStore
    uploads: UploadStore
    access: Access
    inbox: Path
    sets_root: Path
    decoy_set: str
    default_set: str | None = None
    review_url: str | None = None  # only shown on the PC
    port: int = 8800
    lan: bool = False
    live: dashboard.LiveStatus | None = None


class StartBody(BaseModel):
    name: str
    size: int
    set_id: str


class LoginBody(BaseModel):
    code: str


def is_loopback(request: Request) -> bool:
    return bool(request.client) and request.client.host in LOOPBACK


def lan_addresses() -> list[str]:
    """This PC's address on the home network (the one a phone on the same Wi-Fi uses)."""
    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; picks the LAN interface
            found.append(s.getsockname()[0])
    except OSError:
        pass
    return [a for a in found if not a.startswith("127.")]


def create_app(cfg: ControlConfig) -> FastAPI:
    app = FastAPI(title="pokepipe control room", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        path = request.url.path
        trusted = is_loopback(request) or cfg.access.valid(request.cookies.get(COOKIE))
        if not trusted and path not in OPEN_PATHS:
            if path.startswith("/api/") or path.startswith("/media/"):
                return JSONResponse({"detail": "sign in first"}, status_code=401)
            return RedirectResponse("/login", status_code=303)
        if request.method not in ("GET", "HEAD") and request.headers.get(CSRF_HEADER) != "1":
            return JSONResponse({"detail": "missing X-Pokepipe header"}, status_code=403)
        response = await call_next(request)
        if path in ("/", "/login") or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(UploadNotFound)
    async def _missing(_: Request, exc: UploadNotFound):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(OffsetMismatch)
    async def _offset(_: Request, exc: OffsetMismatch):
        return JSONResponse({"detail": str(exc), "received": exc.received}, status_code=409)

    @app.exception_handler(UploadError)
    async def _bad(_: Request, exc: UploadError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # --- pages -----------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/login", include_in_schema=False)
    def login_page():
        return FileResponse(STATIC_DIR / "login.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.post("/api/login")
    def login(body: LoginBody, request: Request):
        client = request.client.host if request.client else "?"
        if cfg.access.locked_out(client):
            return JSONResponse({"detail": "too many wrong codes; wait 5 minutes"},
                                status_code=429)
        token = cfg.access.login(body.code, client)
        if token is None:
            return JSONResponse({"detail": "wrong code"}, status_code=401)
        response = JSONResponse({"ok": True})
        response.set_cookie(COOKIE, token, max_age=365 * 24 * 3600, httponly=True,
                            samesite="strict")
        return response

    # --- status ----------------------------------------------------------------------

    @app.get("/api/status")
    def status(request: Request):
        out = dashboard.build(cfg.jobs, cfg.inbox, cfg.sets_root, cfg.live)
        out["uploads"] = [u.model_dump(mode="json") for u in cfg.uploads.all()[:10]]
        if is_loopback(request):  # the code and review link are for the PC screen only
            out["pc"] = {
                "review_url": cfg.review_url,
                "lan": cfg.lan,
                "phone_urls": [f"http://{a}:{cfg.port}/" for a in lan_addresses()]
                if cfg.lan else [],
                "code": cfg.access.code if cfg.lan else None,
            }
        return out

    @app.get("/api/sets")
    def sets():
        names = dashboard.set_names(cfg.sets_root)
        names.pop(cfg.decoy_set, None)
        return {
            "default": cfg.default_set,
            "sets": [{"id": k, "name": v} for k, v in sorted(names.items(), key=lambda kv: kv[1])],
        }

    @app.post("/api/jobs/{slug}/retry")
    def retry(slug: str):
        if not SAFE_SLUG.match(slug):
            return JSONResponse({"detail": "no such video"}, status_code=404)
        try:
            job = cfg.jobs.load(slug)
        except FileNotFoundError:
            return JSONResponse({"detail": "no such video"}, status_code=404)
        if job.status != Status.FAILED:
            return JSONResponse({"detail": f"{slug} hasn't failed"}, status_code=400)
        Runner(None, cfg.jobs).retry(job, force_budget=False)  # type: ignore[arg-type]
        return {"status": job.status.value}

    @app.get("/media/{slug}/posts/{n}.mp4", include_in_schema=False)
    def post_video(slug: str, n: int):
        if not SAFE_SLUG.match(slug):
            return JSONResponse({"detail": "no such post"}, status_code=404)
        path = cfg.jobs.root / slug / "posts" / str(n) / "post.mp4"
        if not path.exists():
            return JSONResponse({"detail": "no such post"}, status_code=404)
        return FileResponse(path, media_type="video/mp4",
                            filename=f"{slug}-part{n}.mp4", content_disposition_type="inline")

    # --- uploads ---------------------------------------------------------------------

    @app.post("/api/uploads", status_code=201)
    def start_upload(body: StartBody):
        return cfg.uploads.start(body.name, body.size, body.set_id).model_dump(mode="json")

    @app.get("/api/uploads/{upload_id}")
    def get_upload(upload_id: str):
        return cfg.uploads.get(upload_id).model_dump(mode="json")

    @app.put("/api/uploads/{upload_id}")
    async def put_chunk(upload_id: str, request: Request, offset: int = Query(..., ge=0)):
        declared = int(request.headers.get("content-length") or 0)
        if declared > MAX_CHUNK_BYTES:
            return JSONResponse({"detail": "chunk too large"}, status_code=413)
        data = await request.body()
        up = await run_in_threadpool(cfg.uploads.write, upload_id, offset, data)
        return {"received": up.received, "size": up.size}

    @app.post("/api/uploads/{upload_id}/finish")
    async def finish_upload(upload_id: str):
        up = await run_in_threadpool(cfg.uploads.finish, upload_id)
        return up.model_dump(mode="json")

    @app.delete("/api/uploads/{upload_id}", status_code=204)
    def cancel_upload(upload_id: str):
        cfg.uploads.cancel(upload_id)

    return app
