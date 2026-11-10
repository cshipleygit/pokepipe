"""Frame sampling: decode a video at a fixed rate via an ffmpeg pipe.

ffmpeg does the decode, fps resampling, downscale and rotation in one pass, which is
much faster than decoding full-res frames in Python and throwing most away.

Phone footage needs two extra cares, both found on real 4K iPhone video:
- **Rotation.** Phones store vertical video as landscape frames plus a rotation flag.
  `probe` reports the *displayed* size, and sampling rotates explicitly; otherwise the
  vertical picture gets squashed into a landscape frame.
- **Cost.** 4K60 HEVC decoded on the CPU ran at 0.9x real time. Decoding on the GPU
  (NVDEC), dropping to the sample rate first and scaling on the GPU ran at 6.7x. The
  GPU path is tried first and falls back to the CPU automatically.
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
    width: int  # as displayed (after rotation)
    height: int
    duration: float
    fps: float
    rotation: int = 0  # clockwise degrees needed to display the stored frames: 0/90/180/270

    @property
    def coded_size(self) -> tuple[int, int]:
        """Size of the frames as stored in the file (before rotation)."""
        return (self.height, self.width) if self.rotation in (90, 270) else (self.width,
                                                                             self.height)


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


def rotation_of(stream: dict) -> int:
    """Clockwise degrees to display the stream, from the display matrix (newer files)
    or the legacy `rotate` tag. ffprobe reports the matrix angle counter-clockwise, so
    an iPhone's -90 means "turn 90 degrees clockwise"."""
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            return int(round(-float(side["rotation"]))) % 360
    tag = (stream.get("tags") or {}).get("rotate")
    return int(tag) % 360 if tag else 0


def probe(video: Path) -> VideoInfo:
    out = subprocess.run(
        [
            require_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:stream_side_data=rotation:stream_tags=rotate"
            ":format=duration",
            "-of", "json", str(video),
        ],
        capture_output=True, check=True, text=True, encoding="utf-8",
    )
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    num, den = (int(x) for x in stream["r_frame_rate"].split("/"))
    rotation = rotation_of(stream)
    w, h = int(stream["width"]), int(stream["height"])
    if rotation in (90, 270):
        w, h = h, w
    return VideoInfo(
        width=w,
        height=h,
        duration=float(data["format"]["duration"]),
        fps=num / den if den else 0.0,
        rotation=rotation,
    )


def scaled_size(info: VideoInfo, max_side: int) -> tuple[int, int]:
    """Output size preserving aspect ratio, with even dimensions (ffmpeg requirement)."""
    scale = min(1.0, max_side / max(info.width, info.height))
    w = int(round(info.width * scale / 2) * 2)
    h = int(round(info.height * scale / 2) * 2)
    return w, h


_ROTATE = {90: "transpose=clock", 180: "hflip,vflip", 270: "transpose=cclock"}


def sample_cmds(video: Path, info: VideoInfo, fps: float, size: tuple[int, int],
                start: float = 0.0, duration: float | None = None) -> list[list[str]]:
    """ffmpeg commands to try in order: GPU decode + GPU scale, then CPU."""
    w, h = size
    sw, sh = (h, w) if info.rotation in (90, 270) else (w, h)  # scale before rotating
    rotate = [_ROTATE[info.rotation]] if info.rotation in _ROTATE else []

    def cmd(input_args: list[str], filters: list[str]) -> list[str]:
        args = [require_tool("ffmpeg"), "-v", "error", "-nostdin", *input_args,
                "-noautorotate"]
        if start:
            args += ["-ss", f"{start:.3f}"]
        args += ["-i", str(video)]
        if duration:
            args += ["-t", f"{duration:.3f}"]
        return args + ["-vf", ",".join(filters + rotate), "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "-"]

    gpu = cmd(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"],
              [f"fps={fps}", f"scale_cuda={sw}:{sh}:format=nv12", "hwdownload", "format=nv12"])
    cpu = cmd([], [f"fps={fps}", f"scale={sw}:{sh}"])
    return [gpu, cpu]


def sample_frames(
    video: Path,
    fps: float = 4.0,
    max_side: int = 960,
    start: float = 0.0,
    duration: float | None = None,
) -> Iterator[Frame]:
    """Yield upright RGB frames at `fps`, downscaled so the longest side is `max_side`."""
    info = probe(video)
    w, h = scaled_size(info, max_side)
    frame_bytes = w * h * 3
    errors = []
    for cmd in sample_cmds(video, info, fps, (w, h), start, duration):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:  # this decoder can't handle it: try the next one
                proc.wait()
                errors.append(proc.stderr.read().decode("utf-8", "replace").strip())
                continue
            index = 0
            while len(buf) == frame_bytes:
                yield Frame(index, start + index / fps,
                            np.frombuffer(buf, np.uint8).reshape(h, w, 3))
                index += 1
                buf = proc.stdout.read(frame_bytes)
            return
        finally:
            proc.stdout.close()
            proc.stderr.close()
            proc.kill()
            proc.wait()
    raise RuntimeError(f"ffmpeg produced no frames for {video}: {' | '.join(errors)[-600:]}")
