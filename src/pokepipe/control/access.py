"""Access code for phones on the home network.

The PC itself (loopback) never needs the code. A phone enters it once on the sign-in
page and gets a long-lived session cookie. Only hashes of session tokens are stored, in
`data/control/access.json`, so sessions survive restarts and the file holds nothing
that signs anyone in. `pokepipe ui code --new` changes the code and signs every phone
out. Wrong guesses are rate-limited per client.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from collections.abc import Callable
from pathlib import Path

ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I/L: easy to type on a phone
CODE_LENGTH = 8
MAX_FAILURES = 5
FAILURE_WINDOW = 300.0  # seconds
MAX_SESSIONS = 50


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_code() -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))


def normalise(code: str) -> str:
    return "".join(ch for ch in code.upper() if ch.isalnum())


class Access:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock
        self._lock = threading.Lock()
        self._failures: dict[str, list[float]] = {}
        data = self._read()
        if not data.get("code"):
            data = {"code": new_code(), "sessions": []}
            self._write(data)
        self._data = data

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def code(self) -> str:
        return self._data["code"]

    def reset(self) -> str:
        """New code; every signed-in phone has to sign in again."""
        with self._lock:
            self._data = {"code": new_code(), "sessions": []}
            self._write(self._data)
            return self.code

    def locked_out(self, client: str) -> bool:
        now = self.clock()
        recent = [t for t in self._failures.get(client, []) if now - t < FAILURE_WINDOW]
        self._failures[client] = recent
        return len(recent) >= MAX_FAILURES

    def login(self, code: str, client: str) -> str | None:
        """A new session token for the right code, else None (and the try counts)."""
        with self._lock:
            if self.locked_out(client):
                return None
            if not hmac.compare_digest(normalise(code), self.code):
                self._failures.setdefault(client, []).append(self.clock())
                return None
            self._failures.pop(client, None)
            token = secrets.token_urlsafe(32)
            sessions = [*self._data["sessions"], _hash(token)][-MAX_SESSIONS:]
            self._data = {**self._data, "sessions": sessions}
            self._write(self._data)
            return token

    def valid(self, token: str | None) -> bool:
        if not token:
            return False
        digest = _hash(token)
        return any(hmac.compare_digest(digest, s) for s in self._data["sessions"])
