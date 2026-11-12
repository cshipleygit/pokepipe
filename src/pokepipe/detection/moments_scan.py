"""Gather the signals `detection.moments` needs from a video: the high-passed sound
level, Energy matches before each pack and zero-shot scores over the box opening."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch

from pokepipe.cards.index import CardIndex
from pokepipe.cards.service import is_decoy
from pokepipe.detection.frames import require_tool, sample_frames
from pokepipe.detection.matcher import FrameMatcher
from pokepipe.detection.models import PullsFile
from pokepipe.detection.moments import (
    MomentsFile,
    PackMoments,
    SleeveMoment,
    energy_start,
    intro_moments,
    loud_runs,
    rip_and_cards_out,
    sleeve_slide,
)

HOP = 0.1  # seconds per sound level sample
MIN_PACK_PULLS = 3  # smaller groups are cards shown on their own, not opened packs
ENERGY_SCORE = 0.68  # same bar as a detection frame vote

# Softmax over all of these per frame; only "open", "table" and "fan" are used (see
# detection/moments.py for why), the rest give them something to lose to.
PROMPTS = {
    "box": "a large colorful sealed cardboard box with plastic wrap held in hands",
    "unwrap": "hands peeling the plastic shrink wrap off a box",
    "lid": "lifting the lid off a cardboard box",
    "open": "an open box with packs and dividers inside",
    "accessories": "dice, a coin and small accessories on a table",
    "guide": "an open booklet showing a grid of trading cards",
    "fan": "several booster packs fanned out in hands",
    "pack": "a hand holding one sealed foil booster pack",
    "rip": "hands tearing open the top of a foil booster pack",
    "cards_out": "pulling a stack of cards out of an opened foil wrapper",
    "back": "the back of a Pokemon trading card with a blue border",
    "front": "a Pokemon trading card held in hands",
    "sleeve": "sliding a trading card into a clear plastic sleeve",
    "table": "an empty white table",
}


def sound_level(video: Path) -> np.ndarray:
    """dB of the high-passed (>2.5 kHz) first audio track, one value per HOP seconds,
    each averaged over 0.25 s. Tearing foil is broadband; voices and thumps are not."""
    raw = subprocess.run(
        [require_tool("ffmpeg"), "-v", "error", "-i", str(video), "-map", "0:a:0",
         "-ac", "1", "-ar", "16000", "-af", "highpass=f=2500", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    hop, win = int(16000 * HOP), int(16000 * 0.25)
    power = np.convolve(x**2, np.ones(win) / win, "same")[::hop]
    return 20 * np.log10(np.sqrt(power) + 1e-6)


def _frame_tensor(rgb: np.ndarray, device: str, top: float = 0.0) -> torch.Tensor:
    x = torch.from_numpy(rgb.copy()).to(device).permute(2, 0, 1)[None].float() / 255
    if top:
        x = x[:, :, int(x.shape[2] * top):, :]
    return x


def zero_shot(video: Path, embedder, start: float, end: float, fps: float = 2.0):
    """(times, {prompt: probs over the hands region}, {prompt: probs over the full frame})."""
    text = embedder.embed_texts(list(PROMPTS.values()))
    size = embedder.image_size
    times, crop_p, full_p = [], [], []
    for f in sample_frames(video, fps, 448, start, end - start):
        rows = []
        for top in (0.32, 0.0):  # the hands region skips the box display behind the table
            x = torch.nn.functional.interpolate(
                _frame_tensor(f.rgb, embedder.device, top), size=(size, size),
                mode="bilinear")
            rows.append((100 * embedder.embed_pixels(x) @ text.T).softmax(-1)[0].cpu().numpy())
        times.append(f.time)
        crop_p.append(rows[0])
        full_p.append(rows[1])
    names = list(PROMPTS)
    crop_a, full_a = np.asarray(crop_p), np.asarray(full_p)
    return (np.asarray(times), {n: crop_a[:, i] for i, n in enumerate(names)},
            {n: full_a[:, i] for i, n in enumerate(names)})


def find_moments(video: Path, det: PullsFile, embedder, index: CardIndex) -> MomentsFile:
    out = MomentsFile(video=str(video))
    db = sound_level(video)
    runs = loud_runs(db, HOP)
    matcher = FrameMatcher(embedder, index)

    packs = []
    for pack in det.packs:
        pulls = [p for p in det.pulls if p.pack == pack.index]
        if len(pulls) >= MIN_PACK_PULLS:
            packs.append((pack, min(p.start for p in pulls), max(p.end for p in pulls)))
    prev_end = 0.0
    for pack, first, last in packs:
        pm = PackMoments(index=pack.index)
        samples = [
            (r.time, is_decoy(r.best.card_id) and r.best.score >= ENERGY_SCORE)
            for r in matcher.match_stream(sample_frames(video, 4.0, 960, max(0.0, first - 15),
                                                        min(15.0, first)))
        ]
        pm.energy_at = energy_start(samples, first)
        if pm.energy_at is None:
            out.notes.append(f"pack {pack.index}: no Energy found before the first card")
        before = pm.energy_at if pm.energy_at is not None else first
        found = rip_and_cards_out([r for r in runs if r.start >= prev_end], before)
        if found:
            pm.rip, pm.cards_out = found
        else:
            out.notes.append(f"pack {pack.index}: no tearing sound before the first card")
        out.packs.append(pm)
        prev_end = last

    if packs:  # the box: everything before the first opened pack's tear
        first_pack = out.packs[0]
        end = first_pack.rip.start - 1.0 if first_pack.rip else packs[0][0].clip_start
        if end > 10:
            times, crop_p, full_p = zero_shot(video, embedder, 0.0, end)
            # Peeling shrink-wrap pauses more than tearing foil: bridge longer dips.
            wrap_runs = [r for r in loud_runs(db, HOP, max_gap=1.2) if r.end <= end]
            out.intro = intro_moments(times, crop_p["open"], full_p["fan"], crop_p["table"],
                                      wrap_runs, end)
    # Sleeving: a card held up again after its pack was usually just sleeved. Look
    # for the slide in the seconds before that showing.
    for pack, _, last in packs:
        for pull in (p for p in det.pulls if p.pack == pack.index):
            # A hit may be held up before it's sleeved too (Kecleon: shown, then sleeved,
            # then shown again), so try each later showing until a slide is found.
            for showing in [s for s in pull.showings if s.start > last][:3]:
                start = showing.start
                times, crop_p, _ = zero_shot(video, embedder, max(0.0, start - 6), start, fps=4)
                span = sleeve_slide(times, crop_p["sleeve"]) if len(times) else None
                if span:
                    out.sleeves.append(
                        SleeveMoment(pack=pack.index, card_id=pull.card_id, clip=span))
                    break
    out.notes.append(f"sleeving found for {len(out.sleeves)} card(s)")
    found = [k for k, v in out.intro.model_dump().items() if v]
    out.notes.append(f"box shots found: {', '.join(found) or 'none'}")
    return out
