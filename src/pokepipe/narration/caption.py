"""The TikTok caption for each post: `posts/<n>/caption.txt`, ready to paste.

Built from what the pipeline already knows, so it costs nothing extra:

    <hook caption from the script>
    Pulled: Kecleon (Illustration Rare, $31.11), Talonflame ex (Double Rare, $13.75)
    Master set: 80/234 (34%), +12 new
    Part 1 of 3
    <optional footer>
    #pokemon #pokemontcg #deltareign #packopening #masterset

"Pulled" lists the post's best cards (Double Rare and up, or anything worth $5+),
priced when `values.json` has a price. Hashtags are the script's, else a default set.

A channel's own additions stay private: `data/brand/caption.json` (gitignored), e.g.
`{"hashtags": ["#yourchannel"], "footer": "Shop link in bio"}`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pokepipe.cards.rarity import is_special, rank
from pokepipe.collection.models import VideoReport
from pokepipe.collection.values import VideoValues
from pokepipe.narration.models import PostPlan, PostScript

FILENAME = "caption.txt"
MAX_CHARS = 2200  # TikTok allows more, but long captions get cut off in the feed anyway
MAX_HITS = 3
WORTH_MENTIONING_USD = 5.0
DEFAULT_TAGS = ["#pokemon", "#pokemontcg", "#packopening"]


def set_tag(set_name: str) -> str:
    return "#" + re.sub(r"[^a-z0-9]", "", set_name.lower())


def _tags(*groups: list[str]) -> list[str]:
    out: list[str] = []
    for group in groups:
        for tag in group:
            tag = "#" + tag.strip().lstrip("#").replace(" ", "")
            if len(tag) > 1 and tag.lower() not in {t.lower() for t in out}:
                out.append(tag)
    return out


def best_pulls(report: VideoReport, packs: list[int], values: VideoValues | None
               ) -> list[str]:
    """The post's best cards, most valuable first, e.g. "Kecleon (Illustration Rare,
    $31.11)"."""
    prices: dict[tuple[int, str, str], float] = {}
    if values:
        for p in values.pulls:
            if p.usd is not None:
                prices[(p.pack, p.card_id, p.variant)] = p.usd
    picks = []
    for p in report.pulls:
        if p.pack not in packs:
            continue
        usd = prices.get((p.pack, p.card_id, p.variant))
        if is_special(p.rarity) or (usd or 0) >= WORTH_MENTIONING_USD:
            what = [p.rarity] if is_special(p.rarity) else []
            if p.variant == "reverse":
                what.append("reverse holo")
            picks.append(((usd or 0, rank(p.rarity)), p.name, what, usd))
    picks.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _, name, what, usd in picks[:MAX_HITS]:
        if usd is not None:
            what = [*what, f"${usd:,.2f}"]
        out.append(f"{name} ({', '.join(what)})" if what else name)
    return out


def build_caption(report: VideoReport, plan: PostPlan, index: int,
                  script: PostScript | None, values: VideoValues | None,
                  extra: dict | None = None) -> str:
    extra = extra or {}
    post = next(p for p in plan.posts if p.index == index)
    lines = [script.hook_caption.strip() if script else f"{report.set_name} pack opening"]

    hits = best_pulls(report, post.packs, values)
    if hits:
        lines.append("Pulled: " + ", ".join(hits))

    done = [pk for pk in report.packs if pk.index in post.packs]
    if done:
        last = max(done, key=lambda pk: pk.index).progress_after
        pct = round(100 * last.master_owned / max(last.master_total, 1))
        new = sum(pk.new_slots for pk in done)
        lines.append(f"Master set: {last.master_owned}/{last.master_total} ({pct}%)"
                     + (f", +{new} new" if new else ""))

    if len(plan.posts) > 1:
        lines.append(f"Part {sorted(p.index for p in plan.posts).index(index) + 1} "
                     f"of {len(plan.posts)}")
    if extra.get("footer"):
        lines.append(str(extra["footer"]).strip())

    tags = _tags(script.hashtags if script and script.hashtags else
                 [*DEFAULT_TAGS, set_tag(report.set_name)], extra.get("hashtags", []))
    text = "\n".join(lines)
    while tags and len(text) + 1 + len(" ".join(tags)) > MAX_CHARS:
        tags.pop()
    return (text + ("\n" + " ".join(tags) if tags else ""))[:MAX_CHARS]


def load_extra(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_captions(video_dir: Path, extra_path: Path | None = None) -> list[Path]:
    """Write caption.txt for every planned post (cheap and deterministic, so always
    rewritten: a fixed review or new prices show up next time)."""
    report = VideoReport.model_validate_json((video_dir / "collection.json").read_text(
        encoding="utf-8"))
    plan = PostPlan.model_validate_json((video_dir / "posts.json").read_text(encoding="utf-8"))
    values_path = video_dir / "values.json"
    values = (VideoValues.model_validate_json(values_path.read_text(encoding="utf-8"))
              if values_path.exists() else None)
    extra = load_extra(extra_path) if extra_path else {}
    written = []
    for post in plan.posts:
        pdir = video_dir / "posts" / str(post.index)
        script_path = pdir / "script.json"
        script = (PostScript.model_validate_json(script_path.read_text(encoding="utf-8"))
                  if script_path.exists() else None)
        pdir.mkdir(parents=True, exist_ok=True)
        path = pdir / FILENAME
        path.write_text(build_caption(report, plan, post.index, script, values, extra) + "\n",
                        encoding="utf-8")
        written.append(path)
    return written
