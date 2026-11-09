"""FastAPI app for the review page. Run via `pokepipe review serve` (localhost only)."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pokepipe.cards.cache import SetCache
from pokepipe.review.models import NewPull, PullUpdate
from pokepipe.review.store import NotFound, ReviewError, ReviewStore

STATIC_DIR = Path(__file__).parent / "static"
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _expected(det) -> int:
    return det.packs[0].expected_cards if det.packs else 10


def create_app(store: ReviewStore) -> FastAPI:
    app = FastAPI(title="pokepipe review", docs_url="/api/docs")

    @app.middleware("http")
    async def _revalidate_ui(request: Request, call_next):
        # The UI ships with the package; make browsers revalidate (cheap ETag check)
        # so an upgraded pokepipe never runs a stale cached app.js.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, exc: NotFound):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ReviewError)
    async def _bad(_: Request, exc: ReviewError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # --- pages & media ---------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/media/{slug}/thumbs/{name}", include_in_schema=False)
    def thumb(slug: str, name: str):
        if not SAFE_NAME.match(name):
            raise NotFound("bad file name")
        path = store.video_dir(slug) / "thumbs" / name
        if not path.exists():
            raise NotFound(name)
        return FileResponse(path, headers={"Cache-Control": "max-age=3600"})

    @app.get("/media/{slug}/video", include_in_schema=False)
    def video(slug: str):
        # Only ever the file detection recorded for this slug, never a client-given path.
        path = Path(store.detection(slug).video)
        if not path.exists():
            raise NotFound("source video has moved or been deleted")
        return FileResponse(path)  # Starlette honours Range requests, so seeking works

    @app.get("/cards/{set_id}/{card_id}.png", include_in_schema=False)
    def card_image(set_id: str, card_id: str):
        if not (SAFE_NAME.match(set_id) and SAFE_NAME.match(card_id)):
            raise NotFound("bad card id")
        path = SetCache(store.sets_root, set_id).image_path(card_id)
        if not path.exists():
            raise NotFound(card_id)
        return FileResponse(path, headers={"Cache-Control": "max-age=86400"})

    # --- API ---------------------------------------------------------------------------

    @app.get("/api/videos")
    def videos():
        out = []
        for slug in store.slugs():
            det, state = store.detection(slug), store.load(slug)
            pending = sum(p.status == "pending" for p in state.pulls)
            out.append({
                "slug": slug, "set_id": det.set_id, "video": Path(det.video).name,
                "packs": len(det.packs), "pulls": len(det.pulls), "pending": pending,
                "completed": state.completed,
            })
        return out

    @app.get("/api/videos/{slug}")
    def video_detail(slug: str):
        det, state = store.detection(slug), store.load(slug)
        expected = _expected(det)
        return {
            "detection": det.model_dump(),
            "review": state.model_dump(),
            "packs": [
                {**s.__dict__, "complete": s.complete}
                for s in store.pack_summaries(state, expected)
            ],
            "expected_cards": expected,
        }

    @app.get("/api/sets/{set_id}/cards")
    def set_cards(set_id: str):
        return [
            {"id": c.id, "name": c.name, "number": c.number, "rarity": c.rarity}
            for c in store.cards(set_id).values()
        ]

    @app.patch("/api/videos/{slug}/pulls/{pull_id}")
    def update_pull(slug: str, pull_id: int, change: PullUpdate):
        return store.update(slug, pull_id, change)

    @app.post("/api/videos/{slug}/pulls", status_code=201)
    def add_pull(slug: str, new: NewPull):
        return store.add(slug, new)

    @app.delete("/api/videos/{slug}/pulls/{pull_id}", status_code=204)
    def delete_pull(slug: str, pull_id: int):
        store.remove(slug, pull_id)

    @app.post("/api/videos/{slug}/packs/{pack}/confirm")
    def confirm_pack(slug: str, pack: int):
        return {"confirmed": store.confirm_pack(slug, pack)}

    @app.post("/api/videos/{slug}/complete")
    def complete(slug: str):
        return store.complete(slug)

    return app
