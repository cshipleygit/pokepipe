"""`pokepipe preflight`: check everything a recording day needs, before the footage
arrives. Each check is independent and returns OK / WARN / FAIL with what to do.

Read-only and free: the API checks only read account state (a model lookup, the
ElevenLabs subscription), they never generate anything.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import StrEnum

from pokepipe.config import PROJECT_ROOT, Settings

# What one video's narration costs in ElevenLabs characters, from the Delta Reign run
# (two 90 s posts, ≈ 1,200 characters each, plus a rewrite's worth of headroom).
CHARS_PER_VIDEO = 3000
GB_PER_VIDEO = 8  # a 16-min 4K60 HDR iPhone video was 6.6 GB, plus outputs


class Level(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class Check:
    name: str
    level: Level
    detail: str


def check_gpu() -> Check:
    try:
        import torch
    except ImportError:
        return Check("GPU", Level.FAIL, "PyTorch isn't installed (see README)")
    if not torch.cuda.is_available():
        return Check("GPU", Level.FAIL, "CUDA not available: install the CUDA build of torch; "
                                        "detection on the CPU is far too slow for 4K video")
    return Check("GPU", Level.OK, torch.cuda.get_device_name(0))


def check_ffmpeg() -> Check:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return Check("ffmpeg", Level.FAIL, "ffmpeg/ffprobe not on PATH")
    from pokepipe.render.ffmpeg import video_encoder

    enc = video_encoder()[1]
    if enc != "h264_nvenc":
        return Check("ffmpeg", Level.WARN, "NVENC unavailable, rendering with x264 (slower)")
    return Check("ffmpeg", Level.OK, "ffmpeg + NVENC")


def check_anthropic(s: Settings) -> Check:
    if not s.narration and not s.recover:
        return Check("Anthropic", Level.OK, "not needed (narration and recovery are off)")
    try:
        from pokepipe.narration.script import anthropic_client

        client = anthropic_client(s.anthropic_api_key, s.anthropic_workspace_id)
        client.models.retrieve(s.script_model)
    except Exception as exc:  # noqa: BLE001 — any failure is the answer
        return Check("Anthropic", Level.FAIL, f"{type(exc).__name__}: {str(exc)[:160]}")
    return Check("Anthropic", Level.OK, f"key works, {s.script_model} available")


def elevenlabs_room(sub: dict, videos: int) -> Check:
    used, limit = sub.get("character_count", 0), sub.get("character_limit", 0)
    left = max(limit - used, 0)
    enough = left // CHARS_PER_VIDEO
    detail = f"{left:,} characters left of {limit:,} (≈ {enough} video(s) of narration)"
    if enough == 0:
        return Check("ElevenLabs", Level.FAIL, detail + " — top up before recording day")
    if enough < videos:
        return Check("ElevenLabs", Level.WARN, detail + f"; you planned {videos}")
    return Check("ElevenLabs", Level.OK, detail)


def check_elevenlabs(s: Settings, videos: int) -> Check:
    if not s.narration:
        return Check("ElevenLabs", Level.OK, "not needed (narration is off)")
    if not s.elevenlabs_api_key or not s.elevenlabs_voice_id:
        return Check("ElevenLabs", Level.FAIL,
                     "set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env")
    try:
        from pokepipe.narration.voice import ElevenLabsClient

        client = ElevenLabsClient(s.elevenlabs_api_key)
        client.voice(s.elevenlabs_voice_id)
        return elevenlabs_room(client.subscription(), videos)
    except Exception as exc:  # noqa: BLE001
        return Check("ElevenLabs", Level.FAIL, f"{str(exc)[:200]}")


def check_set(s: Settings, set_id: str | None) -> Check:
    from pokepipe.cards.cache import SetCache
    from pokepipe.detection.layout import layout_for

    set_id = set_id or s.watch_default_set
    if not set_id:
        return Check("Card set", Level.WARN, "no --set given and no WATCH_DEFAULT_SET: videos "
                                             "must go in data/inbox/<set_id>/")
    cache = SetCache(s.sets_dir, set_id)
    if not cache.exists():
        return Check("Card set", Level.WARN,
                     f"{set_id} not cached yet: run `pokepipe cards fetch {set_id}` and "
                     f"`pokepipe cards embed {set_id}` now, not while the first video waits")
    tag = f"{s.clip_model}__{s.clip_pretrained}" + ("__full" if s.clip_fit == "full" else "")
    card_set = cache.load_set()
    if not cache.index_path(tag).exists():
        return Check("Card set", Level.WARN, f"{card_set.name}: no embedding index; run "
                                             f"`pokepipe cards embed {set_id}`")
    if layout_for(card_set.series) is None:
        return Check("Card set", Level.WARN,
                     f"{card_set.name} ({card_set.series}): pack layout unverified, so no "
                     "reverse-holo inference and auto-review will send every pack to you")
    return Check("Card set", Level.OK, f"{card_set.name} ({card_set.series}): cards, index "
                                       "and pack layout ready")


def check_disk(s: Settings, videos: int) -> Check:
    s.data_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(s.data_dir).free / 1e9
    need = videos * GB_PER_VIDEO
    level = Level.OK if free >= need * 2 else Level.WARN if free >= need else Level.FAIL
    return Check("Disk", level, f"{free:,.0f} GB free in {s.data_dir} (≈ {need} GB for "
                                f"{videos} 4K video(s))")


def check_review(s: Settings) -> Check:
    cal = "calibrated" if s.calibration_path.exists() else "default confidence bar"
    return Check("Review", Level.OK, f"REVIEW_MODE={s.review_mode} ({cal})")


def check_extras(s: Settings) -> Check:
    missing = []
    if not s.style_guide.exists():
        missing.append(f"style guide {s.style_guide}")
    if not (PROJECT_ROOT / "config" / "intro.mp4").exists():
        missing.append("config/intro.mp4 (posts will have no intro)")
    if missing:
        return Check("Extras", Level.WARN, "; ".join(missing))
    return Check("Extras", Level.OK, "style guide and intro found")


def run_all(s: Settings, set_id: str | None, videos: int, online: bool = True) -> list[Check]:
    checks = [check_gpu, check_ffmpeg]
    out: list[Check] = []
    for fn in checks:
        try:
            out.append(fn())
        except Exception as exc:  # noqa: BLE001 — a broken check must not hide the rest
            out.append(Check(fn.__name__.removeprefix("check_"), Level.FAIL, str(exc)[:200]))
    if online:
        out += [check_anthropic(s), check_elevenlabs(s, videos)]
    out += [check_set(s, set_id), check_disk(s, videos), check_review(s), check_extras(s)]
    return out
