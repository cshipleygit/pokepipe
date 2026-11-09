"""Thin client for the free Pokémon TCG API (https://pokemontcg.io, v2).

The API is free but intermittently returns 5xx errors and can be slow, so every
request goes through a session with retry + exponential backoff.
"""

from __future__ import annotations

from collections.abc import Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pokepipe.cards.models import Card, CardSet

PAGE_SIZE = 250  # API maximum


class PokemonTCGError(RuntimeError):
    pass


def build_session(api_key: str | None = None, retries: int = 5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers["User-Agent"] = "pokepipe/0.1"
    if api_key:
        session.headers["X-Api-Key"] = api_key
    return session


class PokemonTCGClient:
    def __init__(
        self,
        base_url: str = "https://api.pokemontcg.io/v2",
        api_key: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or build_session(api_key)
        self.timeout = timeout

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise PokemonTCGError(f"GET {url} failed: {exc}") from exc
        if resp.status_code == 404:
            raise PokemonTCGError(f"Not found: {url}")
        if not resp.ok:
            raise PokemonTCGError(f"GET {url} -> HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def _paginate(self, path: str, params: dict) -> Iterator[dict]:
        page = 1
        while True:
            body = self._get(path, {**params, "page": page, "pageSize": PAGE_SIZE})
            data = body.get("data", [])
            yield from data
            total = body.get("totalCount", 0)
            if not data or page * PAGE_SIZE >= total:
                return
            page += 1

    def get_set(self, set_id: str) -> CardSet:
        return CardSet.from_api(self._get(f"sets/{set_id}")["data"])

    def search_sets(self, query: str | None = None) -> list[CardSet]:
        """List sets, newest first. `query` matches set name or id (case-insensitive)."""
        sets = [CardSet.from_api(s) for s in self._paginate("sets", {"orderBy": "-releaseDate"})]
        if query:
            q = query.lower()
            sets = [s for s in sets if q in s.name.lower() or q in s.id.lower()]
        return sets

    def get_cards(self, set_id: str) -> list[Card]:
        """Full checklist for a set (including secret rares), sorted by collector number."""
        cards = [Card.from_api(c) for c in self._paginate("cards", {"q": f"set.id:{set_id}"})]
        return sorted(cards, key=Card.sort_key)
