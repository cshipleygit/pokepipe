"""Script writing with Claude: facts in, structured beats out.

The model only sees facts we computed (pulls, new/duplicate, master-set progress) plus
the user's style guide, and must return beats anchored to pull ids we gave it. That
keeps narration grounded: it can't invent a pull, and every line can be timed to the
footage in stage 6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pokepipe.collection.models import Progress, PullResult, VideoReport
from pokepipe.narration.models import Post, PostScript, ScriptDraft

FALLBACK_BETA = "server-side-fallback-2026-07-01"
WORDS_PER_SECOND = 2.5  # comfortable TikTok voiceover pace


class ScriptError(RuntimeError):
    pass


def anthropic_client(api_key: str | None = None, workspace_id: str | None = None):
    """Anthropic client from settings. Keys that aren't scoped to a workspace must name
    one on every request via the `anthropic-workspace-id` header."""
    import anthropic

    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    if api_key:
        return anthropic.Anthropic(api_key=api_key, default_headers=headers)
    return anthropic.Anthropic(default_headers=headers)  # env var / `ant auth login` profile


@dataclass(frozen=True)
class ScriptSettings:
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    usd_per_mtok_in: float = 4.0
    usd_per_mtok_out: float = 20.0
    seconds_per_pack: float = 12.0
    min_seconds: float = 15.0
    max_seconds: float = 60.0

    def target_seconds(self, packs: int) -> float:
        return max(self.min_seconds, min(self.max_seconds, self.seconds_per_pack * packs))


def pull_ids(report: VideoReport) -> dict[str, PullResult]:
    """Stable ids 'p<pack>-<n>' (n = reveal order within the pack, 1-based)."""
    ids: dict[str, PullResult] = {}
    for pack in sorted({p.pack for p in report.pulls}):
        in_pack = sorted((p for p in report.pulls if p.pack == pack), key=lambda p: p.time)
        for n, pull in enumerate(in_pack, 1):
            ids[f"p{pack}-{n}"] = pull
    return ids


def _progress_line(p: Progress) -> str:
    return (
        f"{p.master_owned}/{p.master_total} master set (every printing), "
        f"{p.full_owned}/{p.full_total} different cards"
    )


def post_progress(report: VideoReport, post: Post) -> tuple[Progress, Progress]:
    """Master-set progress just before the post's first pack and after its last."""
    by_pack = {p.index: p.progress_after for p in report.packs}
    first, last = min(post.packs), max(post.packs)
    earlier = [i for i in by_pack if i < first]
    before = by_pack[max(earlier)] if earlier else report.before
    return before, by_pack.get(last, report.after)


def build_prompt(
    report: VideoReport, post: Post, total_posts: int, style: str, settings: ScriptSettings
) -> tuple[str, str]:
    system = (
        "You write voiceover scripts for short vertical videos of Pokémon TCG pack openings. "
        "Follow the creator's style guide below. Use only the facts given in the request: "
        "never invent cards, rarities, prices or values.\n\n"
        "<style_guide>\n" + style.strip() + "\n</style_guide>"
    )
    ids = pull_ids(report)
    pulls = []
    for pid, p in ids.items():
        if p.pack not in post.packs:
            continue
        status = (
            "new card"
            if p.new_card
            else "new printing"
            if p.new
            else f"duplicate (copy #{p.copy_number})"
        )
        pulls.append(
            {
                "id": pid,
                "pack": p.pack,
                "card": p.name,
                "number": p.number,
                "rarity": p.rarity,
                "printing": p.variant,
                "collection": status,
            }
        )
    before, after = post_progress(report, post)
    seconds = settings.target_seconds(len(post.packs))
    facts = {
        "set": report.set_name,
        "post": f"{post.index} of {total_posts}",
        "packs_in_this_post": post.packs,
        "pulls_in_reveal_order": pulls,
        "collection_before_post": _progress_line(before),
        "collection_after_post": _progress_line(after),
        "new_master_set_slots_this_post": after.master_owned - before.master_owned,
    }
    user = (
        "Write the voiceover for this post.\n\n"
        f"<facts>\n{json.dumps(facts, indent=2, ensure_ascii=False)}\n</facts>\n\n"
        f"Length: about {round(seconds)} seconds spoken, so at most "
        f"{int(seconds * WORDS_PER_SECOND)} words across all beats.\n"
        "Beats: in the order they are spoken. Anchor each beat to 'intro', 'outro', or the "
        "id of the pull it is about (from the facts); a beat about a card must use that "
        "card's id so it lines up with the reveal on screen."
    )
    return system, user


def generate_script(
    client,
    report: VideoReport,
    post: Post,
    total_posts: int,
    style: str,
    settings: ScriptSettings | None = None,
) -> tuple[PostScript, float, list[str]]:
    """Returns (script, cost in USD, warnings). `client` is an `anthropic.Anthropic`."""
    settings = settings or ScriptSettings()
    system, user = build_prompt(report, post, total_posts, style, settings)
    response = client.beta.messages.parse(
        model=settings.model,
        max_tokens=16000,
        betas=[FALLBACK_BETA],
        fallbacks="default",  # server-side fallback model if a safety classifier declines
        output_config={"effort": settings.effort},
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=ScriptDraft,
    )
    if response.stop_reason == "refusal":
        raise ScriptError("the model declined to write this script (refusal)")
    if response.stop_reason == "max_tokens":
        raise ScriptError("the script was cut off (max_tokens)")
    draft = response.parsed_output
    if draft is None:
        raise ScriptError("no structured script in the response")

    usage = response.usage
    cost = (
        usage.input_tokens * settings.usd_per_mtok_in
        + usage.output_tokens * settings.usd_per_mtok_out
    ) / 1_000_000

    valid = {"intro", "outro"} | {
        pid for pid, p in pull_ids(report).items() if p.pack in post.packs
    }
    warnings = []
    beats = []
    for beat in draft.beats:
        if not beat.text.strip():
            continue
        if beat.anchor not in valid:
            fallback = beats[-1].anchor if beats else "intro"
            warnings.append(f"beat anchored to unknown {beat.anchor!r}; using {fallback!r}")
            beat = beat.model_copy(update={"anchor": fallback})
        beats.append(beat)
    if not beats:
        raise ScriptError("the script had no spoken lines")

    script = PostScript(
        post=post.index,
        model=response.model,
        hook_caption=draft.hook_caption,
        beats=beats,
        hashtags=draft.hashtags,
        words=sum(len(b.text.split()) for b in beats),
    )
    return script, round(cost, 5), warnings
