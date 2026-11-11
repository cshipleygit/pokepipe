"""ffmpeg command builders and runner for stage 6.

Two passes keep each command simple and debuggable:
1. one short vertical clip per timeline segment (seek, crop, scale, speed, fps);
2. concat those clips, burn in the ASS overlay, and lay the voice lines at their times.
NVENC (GPU) is used when available, else libx264.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from pokepipe.detection.frames import require_tool
from pokepipe.render.timeline import OUT_H, OUT_W, PlacedLine, Segment

FPS = 30


class RenderError(RuntimeError):
    pass


@lru_cache
def video_encoder() -> list[str]:
    """GPU H.264 if this machine's ffmpeg + driver can open it, else x264."""
    probe = subprocess.run(
        [
            require_tool("ffmpeg"),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=256x256:d=0.1",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
    )
    if probe.returncode == 0:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "{cq}", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "{cq}"]


def _encoder(cq: int) -> list[str]:
    return [a.replace("{cq}", str(cq)) for a in video_encoder()]


def segment_filter(seg: Segment) -> str:
    w, h, x, y = seg.crop
    parts = [
        f"crop={w}:{h}:{x}:{y}",
        f"scale={OUT_W}:{OUT_H}:flags=lanczos",
        "setsar=1",
        f"setpts=(PTS-STARTPTS)/{seg.speed:.4f}",
        f"fps={FPS}",
    ]
    if seg.hold > 0:
        parts.append(f"tpad=stop_mode=clone:stop_duration={seg.hold:.3f}")
    return ",".join(parts)


AUDIO_OUT = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


def segment_cmd(source: Path, seg: Segment, out: Path, audio: bool = False) -> list[str]:
    """One shot as its own clip. With `audio`, the source's first audio track comes
    along: sped up with pitch-preserving atempo, padded through any freeze-frame, and
    cut to exactly the shot's length so clips concatenate in sync."""
    cmd = [
        require_tool("ffmpeg"), "-v", "error", "-y", "-hwaccel", "auto",
        "-ss", f"{seg.src_start:.3f}", "-t", f"{seg.src_len:.3f}", "-i", str(source),
        "-vf", segment_filter(seg), *_encoder(18), "-pix_fmt", "yuv420p",
    ]
    if audio:
        cmd += ["-map", "0:v:0", "-map", "0:a:0",
                "-af", f"atempo={seg.speed:.4f},apad", *AUDIO_OUT, "-t", f"{seg.out_len:.3f}"]
    else:
        cmd += ["-an"]
    return cmd + [str(out)]


def intro_cmd(intro: Path, out: Path, audio: bool = False) -> list[str]:
    fit = (
        f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
        f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}"
    )
    cmd = [require_tool("ffmpeg"), "-v", "error", "-y", "-i", str(intro)]
    if audio:  # keep the concat's streams uniform: the intro gets silence if it has none
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                "-map", "0:v:0", "-map", "1:a:0", "-shortest", *AUDIO_OUT]
    else:
        cmd += ["-an"]
    return cmd + ["-vf", fit, *_encoder(18), "-pix_fmt", "yuv420p", str(out)]


def has_audio(video: Path) -> bool:
    out = subprocess.run(
        [require_tool("ffprobe"), "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    return bool(out.stdout.strip())


def voice_filter(lines: list[PlacedLine], input_index: int = 1) -> str:
    """Cut each line out of voice.mp3 and delay it to its place on the timeline."""
    n = len(lines)
    chains = (
        [f"[{input_index}:a]asplit={n}" + "".join(f"[v{i}]" for i in range(n))]
        if n > 1
        else [f"[{input_index}:a]anull[v0]"]
    )
    for i, pl in enumerate(lines):
        ms = max(0, round(pl.out_start * 1000))
        chains.append(
            f"[v{i}]atrim=start={pl.line.src_start:.3f}:end={pl.line.src_end + 0.08:.3f},"
            f"asetpts=PTS-STARTPTS,adelay={ms}:all=1[l{i}]"
        )
    chains.append(
        "".join(f"[l{i}]" for i in range(n))
        + f"amix=inputs={n}:normalize=0:dropout_transition=0,apad[aout]"
    )
    return ";".join(chains)


def final_cmd(
    concat_list: Path,
    voice: Path,
    lines: list[PlacedLine],
    ass_name: str,
    duration: float,
    out: Path,
) -> list[str]:
    graph = f"[0:v]subtitles={ass_name}[vout];" + voice_filter(lines)
    return [
        require_tool("ffmpeg"),
        "-v",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-i",
        str(voice),
        "-filter_complex",
        graph,
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-t",
        f"{duration:.3f}",
        *_encoder(21),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(out),
    ]


def final_cmd_source_audio(
    concat_list: Path, ass_name: str, duration: float, out: Path, audio: bool
) -> list[str]:
    """No narration: concat the clips, burn in the overlay, keep the clips' own audio
    (the source's, already speed-matched per shot) or none."""
    cmd = [
        require_tool("ffmpeg"), "-v", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-vf", f"subtitles={ass_name}", "-t", f"{duration:.3f}",
        *_encoder(21), "-pix_fmt", "yuv420p",
    ]
    cmd += ["-map", "0:v:0", "-map", "0:a:0", *AUDIO_OUT] if audio else ["-an"]
    return cmd + ["-movflags", "+faststart", str(out)]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        raise RenderError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")
