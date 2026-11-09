"""ElevenLabs text-to-speech with character timestamps, cached on disk.

One request per post: the beats are joined into one text (so intonation flows across
lines), and the returned per-character timings are mapped back to each beat (for
syncing with the footage) and to each word (for captions).

Every synthesis is cached by a hash of everything that affects the audio, so re-running
a stage never pays twice for the same line.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import requests

from pokepipe.cards.api import build_session
from pokepipe.narration.models import PostScript, TimedBeat, TimedWord

API = "https://api.elevenlabs.io/v1"


class VoiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceSettings:
    voice_id: str
    model_id: str = "eleven_multilingual_v2"
    output_format: str = "mp3_44100_128"
    usd_per_1k_chars: float = 0.30  # set from your ElevenLabs plan; used for budgeting only

    def cost(self, chars: int) -> float:
        return round(chars * self.usd_per_1k_chars / 1000, 5)


class ElevenLabsClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = API,
        session: requests.Session | None = None,
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise VoiceError("ELEVENLABS_API_KEY is not set — add it to .env")
        self.base_url = base_url.rstrip("/")
        self.session = session or build_session()
        self.session.headers["xi-api-key"] = api_key
        self.timeout = timeout

    def _check(self, resp: requests.Response) -> None:
        if resp.ok:
            return
        hint = {
            401: "API key rejected — check ELEVENLABS_API_KEY",
            402: "out of ElevenLabs credits/quota",
            404: "voice not found — check ELEVENLABS_VOICE_ID (`pokepipe narrate voices`)",
            422: "request rejected",
        }.get(resp.status_code, "request failed")
        raise VoiceError(f"ElevenLabs {resp.status_code}: {hint}. {resp.text[:200]}")

    def voices(self) -> list[dict]:
        resp = self.session.get(f"{self.base_url}/voices", timeout=self.timeout)
        self._check(resp)
        return resp.json().get("voices", [])

    def synthesize(self, text: str, settings: VoiceSettings) -> tuple[bytes, dict]:
        resp = self.session.post(
            f"{self.base_url}/text-to-speech/{settings.voice_id}/with-timestamps",
            params={"output_format": settings.output_format},
            json={"text": text, "model_id": settings.model_id},
            timeout=self.timeout,
        )
        self._check(resp)
        body = resp.json()
        alignment = body.get("alignment") or body.get("normalized_alignment")
        if not alignment:
            raise VoiceError("ElevenLabs returned no timing alignment")
        return base64.b64decode(body["audio_base64"]), alignment


class VoiceCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def key(text: str, settings: VoiceSettings) -> str:
        payload = json.dumps([text, settings.voice_id, settings.model_id, settings.output_format])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> tuple[bytes, dict] | None:
        audio, meta = self.root / f"{key}.mp3", self.root / f"{key}.json"
        if audio.exists() and meta.exists():
            return audio.read_bytes(), json.loads(meta.read_text(encoding="utf-8"))
        return None

    def put(self, key: str, audio: bytes, alignment: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{key}.mp3").write_bytes(audio)
        (self.root / f"{key}.json").write_text(json.dumps(alignment), encoding="utf-8")


def join_beats(script: PostScript) -> tuple[str, list[tuple[int, int]]]:
    """Spoken text plus each beat's [start, end) character span within it."""
    parts, spans, pos = [], [], 0
    for beat in script.beats:
        line = beat.text.strip()
        spans.append((pos, pos + len(line)))
        parts.append(line)
        pos += len(line) + 1  # the joining space
    return " ".join(parts), spans


def _char_times(alignment: dict, n_chars: int) -> tuple[list[float], list[float]]:
    starts = alignment["character_start_times_seconds"]
    ends = alignment["character_end_times_seconds"]
    if len(starts) == n_chars:
        return starts, ends
    # Alignment over a slightly different string (text normalisation): map proportionally.
    m = len(starts)
    idx = [min(m - 1, round(i * (m - 1) / max(1, n_chars - 1))) for i in range(n_chars)]
    return [starts[i] for i in idx], [ends[i] for i in idx]


def time_script(
    script: PostScript, alignment: dict
) -> tuple[list[TimedBeat], list[TimedWord], float]:
    text, spans = join_beats(script)
    starts, ends = _char_times(alignment, len(text))
    beats = [
        TimedBeat(
            anchor=b.anchor,
            text=b.text.strip(),
            start=round(starts[s], 3),
            end=round(ends[max(s, e - 1)], 3),
        )
        for b, (s, e) in zip(script.beats, spans, strict=True)
    ]
    words, i = [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        words.append(
            TimedWord(text=text[i:j], start=round(starts[i], 3), end=round(ends[j - 1], 3))
        )
        i = j
    duration = round(max(ends) if ends else 0.0, 3)
    return beats, words, duration
