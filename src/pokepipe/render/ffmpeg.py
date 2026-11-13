"""ffmpeg command builders and runner for stage 6.

Two passes keep each command simple and debuggable:
1. one short vertical clip per timeline segment (seek, crop, scale, speed, fps);
2. concat those clips, burn in the ASS overlay, and lay the voice lines at their times.
NVENC (GPU) is used when available, else libx264.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
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
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    if probe.returncode == 0:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "{cq}", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "{cq}"]


@lru_cache
def hwaccel() -> list[str]:
    """Hardware decode only where an NVIDIA driver is installed. On a machine without
    a GPU (like a CI runner) `-hwaccel auto` has nothing to gain and can stall."""
    return ["-hwaccel", "auto"] if shutil.which("nvidia-smi") else []


def _encoder(cq: int) -> list[str]:
    return [a.replace("{cq}", str(cq)) for a in video_encoder()]


def _piecewise(times: list[float], values: list[float], t: str) -> str:
    """ffmpeg expression: linear interpolation of `values` at `times`, held at the ends."""
    expr = f"{values[-1]:.4f}"
    for i in range(len(times) - 2, -1, -1):
        t0, t1, v0, v1 = times[i], times[i + 1], values[i], values[i + 1]
        lerp = f"{v0:.4f}+({v1 - v0:.4f})*({t}-{t0:.3f})/{max(t1 - t0, 1e-3):.3f}"
        expr = f"if(lt({t},{t1:.3f}),{lerp},{expr})"
    return expr


def zoompan_filter(seg: Segment) -> str:
    """Animated crop for a zoom shot: crop to the normal framing (`window`), then let
    zoompan move a window through it along `seg.path`, scaled to the output size.
    zoompan's window has its input's aspect, which is why the input is the 9:16 window."""
    ww, wh, wx, wy = seg.window or seg.crop
    times = [t for t, _ in seg.path]
    t = f"(in/{FPS}*{seg.speed:.4f})"  # output frame → seconds into the shot's source
    z = _piecewise(times, [ww / c[0] for _, c in seg.path], t)
    x = _piecewise(times, [c[2] - wx for _, c in seg.path], t)
    y = _piecewise(times, [c[3] - wy for _, c in seg.path], t)
    return (
        f"crop={ww}:{wh}:{wx}:{wy},setpts=(PTS-STARTPTS)/{seg.speed:.4f},fps={FPS},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={OUT_W}x{OUT_H}:fps={FPS},setsar=1"
    )


def segment_filter(seg: Segment) -> str:
    if seg.path:
        parts = [zoompan_filter(seg)]
        if seg.hold > 0:
            parts.append(f"tpad=stop_mode=clone:stop_duration={seg.hold:.3f}")
        return ",".join(parts)
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


def segment_cmd(
    source: Path, seg: Segment, out: Path, audio: bool = False, volume: float = 1.0
) -> list[str]:
    """One shot as its own clip. With `audio`, the source's first audio track comes
    along: sped up with pitch-preserving atempo, set to `volume`, padded through any
    freeze-frame, and cut to exactly the shot's length so clips concatenate in sync."""
    cmd = [
        require_tool("ffmpeg"), "-v", "error", "-y", *hwaccel(),
        "-ss", f"{seg.src_start:.3f}", "-t", f"{seg.src_len:.3f}", "-i", str(source),
        "-vf", segment_filter(seg), *_encoder(18), "-pix_fmt", "yuv420p",
    ]
    if audio:
        # apad with an explicit length: an endless pad relies on -t alone to stop ffmpeg
        cmd += ["-map", "0:v:0", "-map", "0:a:0",
                "-af", f"atempo={seg.speed:.4f},volume={volume:.3f},"
                       f"apad=whole_dur={seg.out_len:.3f}", *AUDIO_OUT,
                "-t", f"{seg.out_len:.3f}"]
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
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60,
    )
    return bool(out.stdout.strip())


def voice_filter(lines: list[PlacedLine], input_index: int = 1, under: str = "") -> str:
    """Cut each line out of voice.mp3 and delay it to its place on the timeline. `under`
    (e.g. "[0:a]"): another audio stream mixed in beneath, the source sound."""
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
        "".join(f"[l{i}]" for i in range(n)) + under
        + f"amix=inputs={n + under.count('[')}:normalize=0:dropout_transition=0,apad[aout]"
    )
    return ";".join(chains)


@dataclass(frozen=True)
class Sprite:
    """A PNG laid over the finished frame, moving along `keys` [(t, x)] at height `y`,
    bobbing `bob` pixels: the progress-bar rocket."""

    image: Path
    keys: list[tuple[float, float]]
    y: float
    bob: float = 4.0


def _video_graph(ass_name: str, sprite: Sprite | None, sprite_input: int) -> str:
    """Burn in the ASS overlay, then the sprite on top. Ends in [vout]."""
    if sprite is None:
        return f"[0:v]subtitles={ass_name}[vout]"
    x = _piecewise([t for t, _ in sprite.keys], [v for _, v in sprite.keys], "t")
    y = f"{sprite.y:.1f}+{sprite.bob:.1f}*sin(2*PI*t*0.9)"
    return (f"[0:v]subtitles={ass_name}[sub];"
            f"[sub][{sprite_input}:v]overlay=x='{x}':y='{y}':eval=frame:shortest=1[vout]")


def _sprite_input(sprite: Sprite | None) -> list[str]:
    return ["-loop", "1", "-i", str(sprite.image)] if sprite else []


def final_cmd(
    concat_list: Path,
    voice: Path,
    lines: list[PlacedLine],
    ass_name: str,
    duration: float,
    out: Path,
    source_audio: bool = False,
    sprite: Sprite | None = None,
    sfx: Path | None = None,
) -> list[str]:
    """`source_audio`: the clips carry the (already levelled) source sound; mix the
    voice over it, plus the `sfx` track. Inputs: 0 clips, 1 voice, 2 sprite, then sfx."""
    sfx_input = 2 + bool(sprite)
    under = ("[0:a]" if source_audio else "") + (f"[{sfx_input}:a]" if sfx else "")
    graph = _video_graph(ass_name, sprite, 2) + ";" + voice_filter(lines, under=under)
    return [
        require_tool("ffmpeg"), "-v", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-i", str(voice), *_sprite_input(sprite), *(["-i", str(sfx)] if sfx else []),
        "-filter_complex", graph, "-map", "[vout]", "-map", "[aout]",
        "-t", f"{duration:.3f}", *_encoder(21), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart", str(out),
    ]


def final_cmd_source_audio(
    concat_list: Path, ass_name: str, duration: float, out: Path, audio: bool,
    sprite: Sprite | None = None, sfx: Path | None = None,
) -> list[str]:
    """No narration: concat the clips, burn in the overlay, keep the clips' own audio
    (the source's, already speed-matched per shot) or none, plus the `sfx` track."""
    sfx_input = 1 + bool(sprite)
    graph = _video_graph(ass_name, sprite, 1)
    streams = ("[0:a]" if audio else "") + (f"[{sfx_input}:a]" if sfx else "")
    if streams:
        graph += (f";{streams}amix=inputs={streams.count('[')}:normalize=0"
                  ":dropout_transition=0,apad[aout]")
    cmd = [
        require_tool("ffmpeg"), "-v", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list), *_sprite_input(sprite),
        *(["-i", str(sfx)] if sfx else []),
        "-filter_complex", graph, "-map", "[vout]",
        "-t", f"{duration:.3f}", *_encoder(21), "-pix_fmt", "yuv420p",
    ]
    cmd += ["-map", "[aout]", *AUDIO_OUT] if streams else ["-an"]
    return cmd + ["-movflags", "+faststart", str(out)]


def run(cmd: list[str], cwd: Path | None = None, timeout: float = 1800) -> None:
    """Run ffmpeg. stdin is closed (ffmpeg otherwise listens for keyboard commands,
    which can stall it without a terminal) and a hung command fails with its name
    instead of blocking the pipeline."""
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"ffmpeg timed out after {timeout:.0f}s: {' '.join(cmd)[:300]}") from exc
    if proc.returncode != 0:
        raise RenderError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")
