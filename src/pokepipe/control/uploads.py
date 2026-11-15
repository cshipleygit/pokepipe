"""Resumable chunked uploads from a phone browser into the watch inbox.

A 4K phone opening is several GB, and a phone browser drops the connection whenever the
screen locks or the tab is backgrounded. So a video is sent in chunks, each written at
an explicit byte offset: after a drop the page asks how much arrived and carries on
from there, and choosing the same file again resumes the unfinished upload.

Unfinished bytes live in `data/uploads/<id>.part`, outside the inbox, so the watcher
never sees half a video. Finishing moves the file to `<inbox>/<set_id>/<name>` in two
steps (first as `<name>.part`, which intake ignores, then renamed), so intake only ever
sees a complete file.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from pokepipe.watch.intake import VIDEO_EXTS

MAX_UPLOAD_BYTES = 40 * 1024**3  # 40 GB: well above a long 4K60 opening
MAX_CHUNK_BYTES = 64 * 1024**2
DISK_MARGIN_BYTES = 2 * 1024**3  # leave room for detection thumbs and renders
STALE_AFTER = timedelta(days=7)
SAFE_ID = re.compile(r"^[a-f0-9]{16}$")
SAFE_SET = re.compile(r"^[A-Za-z0-9._-]{1,40}$")


def _now() -> datetime:
    return datetime.now(UTC)


class UploadError(Exception):
    """A request the store refuses (shown to the person as is)."""


class UploadNotFound(UploadError):
    pass


class OffsetMismatch(UploadError):
    """The chunk doesn't start where the file ends; the client resyncs to `received`."""

    def __init__(self, received: int) -> None:
        super().__init__(f"expected offset {received}")
        self.received = received


class VideoCheck(BaseModel):
    """What ffprobe says about the finished file, so a phone that quietly compressed
    the video (Photos' "Most Compatible") is noticed before a whole run is spent on it."""

    width: int
    height: int
    duration: float
    fps: float
    warning: str | None = None


class Upload(BaseModel):
    id: str
    name: str = Field(description="file name as it will appear in the inbox")
    size: int
    set_id: str
    received: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    inbox_path: str | None = None
    check: VideoCheck | None = None

    @property
    def done(self) -> bool:
        return self.finished_at is not None


def safe_filename(name: str) -> str:
    """A phone's file name reduced to something safe to put in the inbox.

    Keeps only the last path component and plain characters; the extension must be a
    video one the watcher takes."""
    base = re.split(r"[\\/]", name)[-1].strip()
    stem, dot, ext = base.rpartition(".")
    if not dot or f".{ext.lower()}" not in VIDEO_EXTS:
        raise UploadError(
            f"{base or 'that file'} isn't a video the pipeline takes "
            f"({', '.join(sorted(VIDEO_EXTS))})"
        )
    stem = re.sub(r"[^A-Za-z0-9 _()-]+", "-", stem).strip(" .-_") or "video"
    return f"{stem[:80]}.{ext.lower()}"


def check_video(width: int, height: int, duration: float, fps: float) -> VideoCheck:
    short = min(width, height)
    warning = None
    if short < 1080:
        warning = (f"only {width}x{height}: the phone probably compressed it. Send the "
                   "original (Files app, or Photos → Options → Current) for best results")
    return VideoCheck(width=width, height=height, duration=duration, fps=fps, warning=warning)


Probe = Callable[[Path], VideoCheck]


class UploadStore:
    def __init__(
        self,
        root: Path,
        inbox: Path,
        valid_sets: Callable[[], set[str]],
        probe: Probe | None = None,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.root = root
        self.inbox = inbox
        self.valid_sets = valid_sets
        self.probe = probe
        self.clock = clock
        self._lock = threading.Lock()

    # --- paths -----------------------------------------------------------------------

    def _meta_path(self, upload_id: str) -> Path:
        if not SAFE_ID.match(upload_id):
            raise UploadNotFound("no such upload")
        return self.root / f"{upload_id}.json"

    def _data_path(self, upload_id: str) -> Path:
        return self.root / f"{upload_id}.part"

    def _save(self, up: Upload) -> None:
        up.updated_at = self.clock()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._meta_path(up.id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(up.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    # --- reads -----------------------------------------------------------------------

    def get(self, upload_id: str) -> Upload:
        path = self._meta_path(upload_id)
        if not path.exists():
            raise UploadNotFound("no such upload")
        return Upload.model_validate_json(path.read_text(encoding="utf-8"))

    def all(self) -> list[Upload]:
        if not self.root.exists():
            return []
        ups = [Upload.model_validate_json(p.read_text(encoding="utf-8"))
               for p in self.root.glob("*.json")]
        return sorted(ups, key=lambda u: u.created_at, reverse=True)

    # --- writes ----------------------------------------------------------------------

    def start(self, name: str, size: int, set_id: str) -> Upload:
        """Begin an upload, or resume an unfinished one of the same file."""
        clean = safe_filename(name)
        if not SAFE_SET.match(set_id) or set_id not in self.valid_sets():
            raise UploadError(f"unknown set {set_id!r}: fetch it first (pokepipe cards fetch)")
        if not 0 < size <= MAX_UPLOAD_BYTES:
            raise UploadError("file size must be between 1 byte and 40 GB")
        with self._lock:
            self._drop_stale()
            for up in self.all():
                if not up.done and (up.name, up.size, up.set_id) == (clean, size, set_id):
                    return up
            self.root.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(self.root).free
            if free < size + DISK_MARGIN_BYTES:
                raise UploadError(f"not enough disk space: {free / 1e9:.1f} GB free, "
                                  f"need {(size + DISK_MARGIN_BYTES) / 1e9:.1f} GB")
            up = Upload(id=secrets.token_hex(8), name=clean, size=size, set_id=set_id,
                        created_at=self.clock())
            self._data_path(up.id).write_bytes(b"")
            self._save(up)
            return up

    def write(self, upload_id: str, offset: int, data: bytes) -> Upload:
        """Append one chunk that must start exactly where the file ends."""
        if len(data) > MAX_CHUNK_BYTES:
            raise UploadError("chunk too large")
        with self._lock:
            up = self.get(upload_id)
            if up.done:
                raise UploadError("upload already finished")
            if offset != up.received:
                raise OffsetMismatch(up.received)
            if up.received + len(data) > up.size:
                raise UploadError("more data than the file size given at the start")
            with self._data_path(up.id).open("r+b") as f:
                f.seek(offset)
                f.write(data)
                f.truncate()
            up.received += len(data)
            self._save(up)
            return up

    def finish(self, upload_id: str) -> Upload:
        """Check the video and hand it to the watcher's inbox."""
        with self._lock:
            up = self.get(upload_id)
            if up.done:
                return up
            if up.received != up.size:
                raise UploadError(f"only {up.received} of {up.size} bytes arrived")
            data = self._data_path(up.id)
            if self.probe is not None:
                try:
                    up.check = self.probe(data)
                except Exception as exc:  # noqa: BLE001 - a bad file is the person's to fix
                    raise UploadError(f"that file doesn't open as a video: {exc}") from exc
            dest = self._free_name(self.inbox / up.set_id / up.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            staging = dest.with_name(dest.name + ".part")  # intake skips *.part
            shutil.move(str(data), staging)
            os.replace(staging, dest)
            up.inbox_path, up.finished_at = str(dest), self.clock()
            self._save(up)
            return up

    def cancel(self, upload_id: str) -> None:
        with self._lock:
            up = self.get(upload_id)
            self._data_path(up.id).unlink(missing_ok=True)
            self._meta_path(up.id).unlink(missing_ok=True)

    def _free_name(self, path: Path) -> Path:
        n, out = 2, path
        while out.exists() or out.with_name(out.name + ".part").exists():
            out = path.with_name(f"{path.stem}-{n}{path.suffix}")
            n += 1
        return out

    def _drop_stale(self) -> None:
        """Forget unfinished uploads untouched for a week, and old finished records."""
        cutoff = self.clock() - STALE_AFTER
        for up in self.all():
            if up.updated_at < cutoff:
                self._data_path(up.id).unlink(missing_ok=True)
                self._meta_path(up.id).unlink(missing_ok=True)


def ffprobe_check(path: Path) -> VideoCheck:
    from pokepipe.detection.frames import probe

    info = probe(path)
    return check_video(info.width, info.height, info.duration, info.fps)
