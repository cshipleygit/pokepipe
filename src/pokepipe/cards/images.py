"""Concurrent, idempotent card-image downloader.

Images are written to a temp file and renamed, so an interrupted run never leaves
a truncated PNG that later looks "cached".
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests

from pokepipe.cards.cache import SetCache
from pokepipe.cards.models import Card


@dataclass
class DownloadReport:
    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def download_images(
    cards: list[Card],
    cache: SetCache,
    session: requests.Session,
    workers: int = 8,
    timeout: float = 60.0,
    on_progress: Callable[[Card], None] | None = None,
) -> DownloadReport:
    report = DownloadReport()
    cache.images_dir.mkdir(parents=True, exist_ok=True)

    todo: list[Card] = []
    for card in cards:
        if cache.image_path(card.id).exists():
            report.skipped.append(card.id)
        elif not card.image_url:
            report.failed[card.id] = "no image url"
        else:
            todo.append(card)

    def fetch(card: Card) -> None:
        resp = session.get(card.image_url, timeout=timeout)
        resp.raise_for_status()
        dest = cache.image_path(card.id)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, c): c for c in todo}
        for fut in as_completed(futures):
            card = futures[fut]
            try:
                fut.result()
                report.downloaded.append(card.id)
            except Exception as exc:  # noqa: BLE001 — record and keep going
                report.failed[card.id] = str(exc)
            if on_progress:
                on_progress(card)
    return report
