"""Script writing with Claude: facts in, structured beats out.

The model only sees facts we computed (pulls, new/duplicate, master-set progress) plus
the user's style guide, and must return beats anchored to pull ids we gave it. That
keeps narration grounded: it can't invent a pull, and every line can be timed to the
footage in stage 6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pokepipe.cards.rarity import is_special
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


MIN_OPENED_PACK = 3  # fewer pulls than this: a card shown outside a pack, not a pack


def pack_numbers(report: VideoReport) -> dict[int, int]:
    """Pack index → the number a viewer would say ("pack 3"), counting opened packs only.

    Detection's pack indexes count every group of reveals, so a promo or a card
    shown before the first pack becomes "pack 1" and shifts every real pack by one.
    Narration must use the creator's count instead. Groups too small to be a pack
    (the same cutoff `detection/reshows.py` uses for satellites) get no number.
    """
    counts: dict[int, int] = {}
    for p in report.pulls:
        counts[p.pack] = counts.get(p.pack, 0) + 1
    opened = [i for i in sorted(counts) if counts[i] >= MIN_OPENED_PACK]
    return {index: n for n, index in enumerate(opened, 1)}


CLIP_WORDS = {  # lead clip labels (render.inputs) as the script writer should read them
    "box_hero": "the sealed Elite Trainer Box",
    "box_seal": "ripping the shrink-wrap off the box",
    "box_open": "lifting the lid",
    "box_packs": "the packs fanned out",
    "rip": "a pack being torn open",
    "cards_out": "the cards coming out of the pack",
}


ENGAGEMENT = (
    "\nHook: the intro's first sentence decides whether people keep watching. Make it a "
    "question, challenge or bet aimed at the viewer (guess, predict, comment), not a "
    "summary. The hook_caption works the same way."
    "\nTalk to the viewer: add two or three short interactive moments, such as asking them "
    "to call the rare before a pack's last card or to comment their pick."
    "\nNo spoilers: a card's name or rarity may only be said in its own beat or later. "
    "The hook_caption and intro name no card or rarity, and no line hints that a hit is "
    "coming or that a particular pack is special. Build suspense; the viewer finds out "
    "when the card appears."
)


def is_hit(p: PullResult) -> bool:
    """Same rule as the render's zoom: the rare slot and better, not reverse holos."""
    if p.variant == "reverse":
        return False
    return p.rarity in ("Rare", "Rare Holo") or is_special(p.rarity)


def _name_key(name: str) -> str:
    """'Talonflame ex' -> 'talonflame': what a line would say."""
    return name.lower().replace(" ex", "").split()[0]


def spoilers(draft: ScriptDraft, report: VideoReport, post: Post) -> list[str]:
    """Lines that name a hit, or a special rarity, before the hit appears.

    Lines are positioned by their anchor in reveal order: the hook caption and intro
    come before every card, the outro after all of them."""
    ids = pull_ids(report)
    order = {pid: i for i, pid in enumerate(ids)}
    hits = [(pid, p) for pid, p in ids.items() if p.pack in post.packs and is_hit(p)]
    first_rarity: dict[str, int] = {}
    for pid, p in hits:
        if is_special(p.rarity):
            key = p.rarity.lower()
            first_rarity[key] = min(first_rarity.get(key, order[pid]), order[pid])
    lines = [("hook caption", -1, draft.hook_caption)] + [
        (b.anchor, order.get(b.anchor, -1 if b.anchor == "intro" else len(order)), b.text)
        for b in draft.beats
    ]
    found = []
    for anchor, at, text in lines:
        low = text.lower()
        for pid, p in hits:
            if at < order[pid] and _name_key(p.name) in low:
                found.append(f"{anchor!r} names {p.name} before it appears: {text!r}")
        for rarity, first in first_rarity.items():
            if at < first and rarity in low:
                found.append(f"{anchor!r} says '{rarity}' before one appears: {text!r}")
    return found


def call_to_action(part: int, parts: int) -> str:
    """The creators' chosen CTA: point to the next part, ask for a follow. No selling."""
    follow = "invite viewers to follow the master-set series"
    if part < parts:
        return (f"The outro tells viewers to watch part {part + 1} for the rest of the "
                f"packs and {follow}.")
    return f"The outro should {follow}."


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
    report: VideoReport,
    post: Post,
    total_posts: int,
    style: str,
    settings: ScriptSettings,
    airtime=None,
) -> tuple[str, str]:
    """`airtime` (render.inputs.Airtime, optional): seconds each card is on screen in a
    draft edit, so the script can be written to fit the footage."""
    system = (
        "You write voiceover scripts for short vertical videos of Pokémon TCG pack openings. "
        "Follow the creator's style guide below. Use only the facts given in the request: "
        "never invent cards, rarities, prices or values.\n\n"
        "<style_guide>\n" + style.strip() + "\n</style_guide>"
    )
    ids = pull_ids(report)
    numbers = pack_numbers(report)
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
                "pack": (
                    f"pack {numbers[p.pack]} of {len(numbers)}"
                    if p.pack in numbers
                    else "shown on its own, outside any pack"
                ),
                "card": p.name,
                "number": p.number,
                "rarity": p.rarity,
                "printing": p.variant,
                "collection": status,
            }
        )
    before, after = post_progress(report, post)
    seconds = post.seconds or settings.target_seconds(len(post.packs))
    if airtime is not None:
        for pull in pulls:
            pull["seconds_on_screen"] = airtime.per_card.get(pull["id"])
            shown = getattr(airtime, "shown_again", {}).get(pull["id"], 0)
            if shown:
                pull["shown_again"] = shown  # the creator deliberately shows it off again
    facts = {
        "set": report.set_name,
        "post": f"{post.index} of {total_posts}",
        "packs_in_this_post": [numbers[i] for i in post.packs if i in numbers],
        "packs_opened_in_video": len(numbers),
        "pulls_in_reveal_order": pulls,
        "collection_before_post": _progress_line(before),
        "collection_after_post": _progress_line(after),
        "new_master_set_slots_this_post": after.master_owned - before.master_owned,
    }
    facts["part"] = f"{post.index} of {total_posts}"
    opening = getattr(airtime, "opening", None) or []
    recap = getattr(airtime, "recap", None) or []
    if opening:
        facts["opening_clips"] = [CLIP_WORDS.get(c, c) for c in opening]
    if recap:
        facts["recap_at_end"] = recap
    if airtime is not None:
        facts["footage"] = {
            "video_seconds": round(airtime.duration),
            "seconds_before_first_card": airtime.before_first,
            "seconds_after_last_card": airtime.after_last,
        }
    timing = ""
    if airtime is not None:
        timing = (
            "\nTiming: the video is edited to the footage, so fit the words to it. "
            f"Speech runs about {WORDS_PER_SECOND} words per second. The intro is spoken "
            "before the first card. A line about a card is spoken while it is on screen: "
            "keep it within that card's seconds_on_screen, plus the seconds of any "
            "following cards you don't mention. After the last card there are only "
            "seconds_after_last_card of footage, so everything said after the last card "
            "appears (including the outro) must fit in that. A card with shown_again is "
            "held up again on purpose later in the video; it is still one card, and "
            "its seconds_on_screen already include that extra footage."
        )
    user = (
        "Write the voiceover for this post.\n\n"
        f"<facts>\n{json.dumps(facts, indent=2, ensure_ascii=False)}\n</facts>\n\n"
        f"Length: about {round(seconds)} seconds spoken, so at most "
        f"{int(seconds * WORDS_PER_SECOND)} words across all beats.\n"
        "Beats: in the order they are spoken. Anchor each beat to 'intro', 'outro', or the "
        "id of the pull it is about (from the facts); a beat about a card must use that "
        "card's id so it lines up with the reveal on screen. Ids are internal labels: "
        "their numbers are not pack numbers. When you say a pack number, take it from "
        "the pull's 'pack' field." + timing + _structure(report, opening, recap)
        + "\nCall to action: " + call_to_action(post.index, total_posts)
        + " Don't mention shops, prices or links." + ENGAGEMENT
    )
    return system, user


def _structure(report: VideoReport, opening: list[str], recap: list[str]) -> str:
    out = ""
    if opening:
        out += (
            "\nStructure: the post opens with quick clips (opening_clips) before the first "
            "card; speak the intro over them."
        )
        if any(c.startswith("box_") for c in opening):
            out += (
                f" They show a {report.set_name} Elite Trainer Box being opened: make the "
                "intro a hook about opening it to start the master set."
            )
    if recap:
        out += (
            "\nThe post ends by replaying the hits in recap_at_end (they were already "
            "revealed); the outro plays over that recap."
        )
    return out


def _draft(client, settings: ScriptSettings, system: str, user: str):
    """One structured call → (draft, response, cost in USD)."""
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
    return draft, response, cost


@dataclass(frozen=True)
class CriticSettings:
    """The full critic (narration/critic.py). Without it, only spoilers are checked and
    a spoiling draft gets one rewrite (the original behaviour)."""

    set_card_names: frozenset[str] = frozenset()
    max_rewrites: int = 2
    judge: object | None = None  # critic.JudgeSettings, or None for no style check


def generate_script(
    client,
    report: VideoReport,
    post: Post,
    total_posts: int,
    style: str,
    settings: ScriptSettings | None = None,
    airtime=None,
) -> tuple[PostScript, float, list[str]]:
    """Returns (script, cost in USD, warnings). `client` is an `anthropic.Anthropic`."""
    script, cost, warnings, _ = write_post_script(
        client, report, post, total_posts, style, settings, airtime)
    return script, cost, warnings


def write_post_script(
    client,
    report: VideoReport,
    post: Post,
    total_posts: int,
    style: str,
    settings: ScriptSettings | None = None,
    airtime=None,
    critic: CriticSettings | None = None,
    budget_check=lambda spent: None,
):
    """Draft, critique, rewrite (narration/writer.py). Returns (script, cost, warnings,
    critique record). The kept draft is the attempt with the fewest problems; its
    remaining problems become warnings."""
    from pokepipe.narration import critic as checks
    from pokepipe.narration.writer import best, build_writer, record

    settings = settings or ScriptSettings()
    system, user = build_prompt(report, post, total_posts, style, settings, airtime)

    def write(feedback: str):
        return _draft(client, settings, system, user + feedback)

    def critique(draft: ScriptDraft):
        if critic is None:
            return checks.check_spoilers(draft, report, post), [], 0.0
        problems = checks.check(draft, report, post, set(critic.set_card_names), airtime)
        notes, cost = [], 0.0
        if critic.judge is not None and style.strip():
            major, notes, cost = checks.judge_style(client, draft, style, critic.judge)
            problems += major
        return problems, notes, cost

    graph = build_writer(write, critique, critic.max_rewrites if critic else 1, budget_check)
    state = graph.invoke({})
    attempts = state["attempts"]
    chosen = best(attempts)
    draft, response, cost = chosen.draft, chosen.response, state.get("cost", 0.0)
    warnings = [str(p) for p in chosen.problems] + [f"style (minor): {n}" for n in chosen.notes]
    if state.get("stop"):
        warnings.append(state["stop"])
    critique_record = record(post.index, attempts, chosen, state.get("stop"))

    valid = {"intro", "outro"} | {
        pid for pid, p in pull_ids(report).items() if p.pack in post.packs
    }
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
    return script, round(cost, 5), warnings, critique_record
