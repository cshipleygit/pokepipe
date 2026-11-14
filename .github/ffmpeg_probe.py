"""CI-only diagnostic (runs when tests fail): which part of a shot-with-audio command
hangs this runner's ffmpeg? Results are printed as GitHub annotations, which are
readable through the public API without the job log. Temporary: remove once fixed."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "tests")
from test_render import rp  # noqa: E402

from pokepipe.render import ffmpeg  # noqa: E402
from pokepipe.render.timeline import EditParams, PackWindow, plan_edit  # noqa: E402


def note(title: str, msg: str) -> None:
    msg = msg.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    print(f"::warning title={title}::{msg}", flush=True)


def attempt(cmd: list[str], limit: float = 25) -> tuple[str, float, str]:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           timeout=limit)
        return f"exit {p.returncode}", time.time() - t0, p.stderr[-300:]
    except subprocess.TimeoutExpired as e:
        err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else ""
        return "HUNG", time.time() - t0, err[-600:]


def swap(cmd: list[str], flag: str, value: str | None, last: bool = False) -> list[str]:
    """Replace (or with value=None remove) a flag's argument."""
    idxs = [i for i, a in enumerate(cmd) if a == flag]
    if not idxs:
        return cmd
    i = idxs[-1] if last else idxs[0]
    return cmd[:i] + ([flag, value] if value is not None else []) + cmd[i + 2:]


tmp = Path(tempfile.mkdtemp())
src = tmp / "src.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "testsrc2=size=1920x1080:rate=30:duration=8", "-f", "lavfi", "-i",
                "sine=f=440:d=8", "-pix_fmt", "yuv420p", "-shortest", str(src)], check=True)
note("ffmpeg", subprocess.run(["ffmpeg", "-version"], capture_output=True,
                              text=True).stdout.splitlines()[0])

pulls = [rp("p1-1", 1, 2.0), rp("p1-2", 1, 4.0, rarity="Rare")]
edit = plan_edit([PackWindow(1, 0.0, 8.0)], pulls, [], (1920, 1080), EditParams(target=5.0))
hung = None
for i, seg in enumerate(edit.segments):
    # the three runs render_segment makes for a shot with audio
    cmd = ffmpeg.segment_audio_cmd(src, seg, tmp / f"s{i}.a.m4a")
    res, secs, _ = attempt(cmd)
    for step in (ffmpeg.segment_cmd(src, seg, tmp / f"s{i}.v.mp4"),
                 ffmpeg.mux_cmd(tmp / f"s{i}.v.mp4", tmp / f"s{i}.a.m4a", tmp / f"s{i}.mp4")):
        if res.startswith("exit"):
            res, more, _ = attempt(step)
            secs += more
    note(f"shot {i} {seg.kind}", f"{res} in {secs:.1f}s speed={seg.speed:.3f} "
                                  f"src={seg.src_start:.2f}+{seg.src_len:.2f} out={seg.out_len:.3f}")
    if res == "HUNG" and hung is None:
        hung = (i, seg, cmd)

if hung:
    i, seg, cmd = hung
    note("hung command", " ".join(cmd[1:]))
    af = cmd[cmd.index("-af") + 1]
    length = f"{seg.out_len:.3f}"
    out = str(tmp / "v.mp4")
    variants = {
        "video only (audio=False)": ffmpeg.segment_cmd(src, seg, tmp / "v0.mp4"),
        "no audio filter": swap(cmd, "-af", "anull")[:-1] + [out],
        "no atempo": swap(cmd, "-af", af.split(",", 1)[1])[:-1] + [out],
        "no apad/atrim": swap(cmd, "-af", ",".join(af.split(",")[:2]))[:-1] + [out],
        "no output -t": swap(cmd, "-t", None, last=True)[:-1] + [out],
        "no input -t": swap(cmd, "-t", None)[:-1] + [out],
        "audio only": ["ffmpeg", "-v", "error", "-y", "-ss", f"{seg.src_start:.3f}", "-t",
                       f"{seg.src_len:.3f}", "-i", str(src), "-vn", "-af", af, "-c:a", "aac",
                       "-t", length, str(tmp / "a.m4a")],
        "pcm audio, not aac": [a if a != "aac" else "pcm_s16le" for a in cmd][:-1]
                              + [str(tmp / "v.mkv")],
    }
    for name, c in variants.items():
        res, secs, _ = attempt(c)
        note(f"variant: {name}", f"{res} in {secs:.1f}s")
    _, _, err = attempt(["ffmpeg", "-v", "verbose", "-y", *cmd[4:]], limit=15)
    note("verbose tail of the hung command", err)
else:
    note("probe", "no shot hung in isolation")
