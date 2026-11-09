"""Frame sampling: decode a video at a fixed rate via an ffmpeg pipe.

ffmpeg does the decode, fps resampling and downscale in one pass, which is much
faster than decoding full-res 60 fps frames in Python and throwing most away.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float
    fps: float


@dataclass(frozen=True)
class Frame:
    index: int
    time: float  # seconds from start
    rgb: np.ndarray  # (H, W, 3) uint8


def require_tool(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise RuntimeError(f"{tool} not on PATH — install it: winget install Gyan.FFmpeg")
    return path


def probe(video: Path) -> VideoInfo:
    out = subprocess.run(
        [
            require_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate:format=duration",
            "-of", "json", str(video),
        ],
        capture_output=True, check=True, text=True, encoding="utf-8",
    )
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    num, den = (int(x) for x in stream["r_frame_rate"].split("/"))
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration=float(data["format"]["duration"]),
        fps=num / den if den else 0.0,
    )


def scaled_size(info: VideoInfo, max_side: int) -> tuple[int, int]:
    """Output size preserving aspect ratio, with even dimensions (ffmpeg requirement)."""
    scale = min(1.0, max_side / max(info.width, info.height))
    w = int(round(info.width * scale / 2) * 2)
    h = int(round(info.height * scale / 2) * 2)
    return w, h


def sample_frames(
    video: Path,
    fps: float = 4.0,
    max_side: int = 960,
    start: float = 0.0,
    duration: float | None = None,
) -> Iterator[Frame]:
    """Yield RGB frames at `fps`, downscaled so the longest side is `max_side`."""
    info = probe(video)
    w, h = scaled_size(info, max_side)
    cmd = [require_tool("ffmpeg"), "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    frame_bytes = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        index = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                if index == 0:  # nothing decoded: surface ffmpeg's error (tiny at -v error)
                    proc.wait()
                    err = proc.stderr.read().decode("utf-8", "replace").strip()
                    raise RuntimeError(f"ffmpeg produced no frames for {video}: {err}")
                break
            rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            yield Frame(index, start + index / fps, rgb)
            index += 1
    finally:
        proc.stdout.close()
        proc.stderr.close()
        proc.kill()
        proc.wait()
