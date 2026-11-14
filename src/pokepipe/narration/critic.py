"""The script critic: checks a draft against the facts, the footage and the rules.

Deterministic checks (free, no model):

- **contract**: anchors are known pull ids, beats run in reveal order, lines and the
  hook caption stay within their limits, no money talk.
- **spoilers**: a hit or special rarity named before it appears (`script.spoilers`).
- **facts**: pack numbers match the pull a line is anchored to, no card names from
  outside this post, rarities named match a card already revealed, and "new" /
  "duplicate" claims match the collection.
- **timing**: each stretch of narration fits the footage it plays over (the intro
  before the first card it mentions, each card line until the next mentioned card).
  Lines that don't fit make the render freeze the picture, the most visible failure.

An optional model check (`judge_style`) reads `config/style.md` and the draft and
reports rules the draft breaks. Only *major* breaks send a draft back for a rewrite.

Every problem has a `kind`, so the writer loop can report failure rates per kind, and
`pokepipe narrate critique` can score scripts already on disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from pokepipe.collection.models import VideoReport
from pokepipe.narration.models import Post, ScriptDraft
from pokepipe.narration.script import (
    WORDS_PER_SECOND,
    pack_numbers,
    pull_ids,
    spoilers,
)

Kind = Literal["contract", "spoiler", "fact", "timing", "style"]

TIMING_SLACK = 1.15  # words may run 15% over the airtime before it counts
MAX_BEAT_WORDS = 20
MAX_CAPTION_WORDS = 8

NUMBER_WORDS = {w: i for i, w in enumerate([
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty"])}
RARITY_WORDS = [  # longest first, so "special illustration rare" isn't also "illustration rare"
    "special illustration rare", "mega hyper rare", "hyper rare", "illustration rare",
    "ultra rare", "double rare", "ace spec rare",
]
MONEY = re.compile(r"\$|\bdollars?\b|\bbucks\b|\bworth\b|\bprice[sd]?\b|\bshop\b|\blink\b")
SAYS_NEW = re.compile(r"\b(?:brand[- ]new|new (?:card|one|pull|to (?:me|the binder))|"
                      r"first (?:copy|one)|never (?:had|pulled))\b")
SAYS_DUPE = re.compile(r"\b(?:dupe|duplicate|another copy|already (?:have|had|got|own))\b")
PACK_NO = re.compile(r"\b(?:(\w+) )?pack (\d+|" + "|".join(NUMBER_WORDS) + r")\b")
# "same Vespiquen as pack one", "better than pack two": a look back at another pack,
# not a label for the card the line is about.
REFERS_BACK = {"as", "than", "from", "like", "since", "after", "before", "last"}


@dataclass(frozen=True)
class Problem:
    kind: Kind
    message: str
    anchor: str = ""

    def __str__(self) -> str:
        where = f"[{self.anchor}] " if self.anchor else ""
        return f"{self.kind}: {where}{self.message}"


def _name_key(name: str) -> str:
    return re.sub(r"\s+ex$", "", name.lower().strip())


def _words(text: str) -> int:
    return len(text.split())


# --- deterministic checks -----------------------------------------------------------


def check_contract(draft: ScriptDraft, report: VideoReport, post: Post) -> list[Problem]:
    ids = pull_ids(report)
    order = {pid: i for i, pid in enumerate(ids)}
    valid = {"intro", "outro"} | {pid for pid, p in ids.items() if p.pack in post.packs}
    out: list[Problem] = []
    if _words(draft.hook_caption) > MAX_CAPTION_WORDS:
        out.append(Problem("contract", f"hook caption has {_words(draft.hook_caption)} words "
                                       f"(max {MAX_CAPTION_WORDS})"))
    last = -2
    for b in draft.beats:
        if b.anchor not in valid:
            out.append(Problem("contract", f"anchor {b.anchor!r} is not a pull in this post",
                               b.anchor))
            continue
        at = -1 if b.anchor == "intro" else len(order) if b.anchor == "outro" else order[b.anchor]
        if at < last:
            out.append(Problem("contract", "beat is out of reveal order", b.anchor))
        last = max(last, at)
        if _words(b.text) > MAX_BEAT_WORDS:
            out.append(Problem("contract", f"{_words(b.text)} words (max {MAX_BEAT_WORDS}): "
                                           f"{b.text!r}", b.anchor))
        if MONEY.search(b.text.lower()):
            out.append(Problem("contract", f"mentions money or shops: {b.text!r}", b.anchor))
    return out


def check_spoilers(draft: ScriptDraft, report: VideoReport, post: Post) -> list[Problem]:
    return [Problem("spoiler", s) for s in spoilers(draft, report, post)]


def check_facts(
    draft: ScriptDraft, report: VideoReport, post: Post, set_card_names: set[str]
) -> list[Problem]:
    """`set_card_names`: every card name in the set, to catch cards from outside the post."""
    ids = pull_ids(report)
    in_post = {pid: p for pid, p in ids.items() if p.pack in post.packs}
    order = list(in_post)
    numbers = pack_numbers(report)
    post_names = {_name_key(p.name) for p in in_post.values()}
    # Names that are also ordinary words would false-alarm; only check distinctive ones.
    foreign = {n for n in (_name_key(x) for x in set_card_names)
               if n not in post_names and len(n) >= 5}
    out: list[Problem] = []
    for b in draft.beats:
        low = b.text.lower()
        pull = in_post.get(b.anchor)
        # pack numbers
        for m in PACK_NO.finditer(low):
            word = m.group(2)
            said = int(word) if word.isdigit() else NUMBER_WORDS[word]
            post_packs = {numbers.get(i) for i in post.packs}
            if pull is not None and m.group(1) in REFERS_BACK:
                if said not in numbers.values():
                    out.append(Problem("fact", f"refers to 'pack {word}', which wasn't "
                                               "opened", b.anchor))
            elif pull is not None and numbers.get(pull.pack) != said:
                out.append(Problem("fact", f"says 'pack {word}' but this card is in "
                                           f"pack {numbers.get(pull.pack)}", b.anchor))
            elif pull is None and said not in post_packs:
                out.append(Problem("fact", f"says 'pack {word}', not a pack in this "
                                           "post", b.anchor))
        # cards from outside the post
        for name in foreign:
            if re.search(rf"\b{re.escape(name)}\b", low):
                out.append(Problem("fact", f"names {name.title()!r}, which isn't pulled in "
                                           "this post", b.anchor))
        if pull is None:
            continue
        # rarity named must belong to a card revealed by now
        revealed = order[: order.index(b.anchor) + 1]
        seen = {(in_post[pid].rarity or "").lower() for pid in revealed}
        for rarity in RARITY_WORDS:
            if rarity in low:
                if rarity not in seen:
                    out.append(Problem("fact", f"says '{rarity}', but no card so far is one",
                                       b.anchor))
                low = low.replace(rarity, " ")
        # collection claims about the anchored card
        if SAYS_NEW.search(low) and not pull.new:
            out.append(Problem("fact", f"calls {pull.name} new, but it's copy "
                                       f"#{pull.copy_number}", b.anchor))
        # "I already had the reverse, so this holo is new" is true: only a card that is
        # new in every printing can't be called a duplicate.
        if SAYS_DUPE.search(low) and pull.new_card:
            out.append(Problem("fact", f"calls {pull.name} a duplicate, but it's the first "
                                       "copy of that card", b.anchor))
    return out


def check_timing(draft: ScriptDraft, report: VideoReport, post: Post, airtime) -> list[Problem]:
    """Each stretch from one mentioned moment to the next must hold its words.

    Matches how render cuts the edit: a region starts at the card a line is about and
    runs until the next card that has a line; the intro region runs from the start to
    the first such card; the outro is spoken in the last region."""
    if airtime is None:
        return []
    ids = [pid for pid, p in pull_ids(report).items() if p.pack in post.packs]
    words: dict[str, int] = {}
    for b in draft.beats:
        key = ids[-1] if b.anchor == "outro" and ids else b.anchor
        words[key] = words.get(key, 0) + _words(b.text)

    regions: list[tuple[str, float]] = [("intro", airtime.before_first)]
    for pid in ids:
        secs = airtime.per_card.get(pid, 0.0)
        if pid in words:
            regions.append((pid, secs))
        else:  # no line here: its footage belongs to the region before it
            name, total = regions[-1]
            regions[-1] = (name, total + secs)
    out = []
    for name, secs in regions:
        n = words.get(name, 0)
        room = secs * WORDS_PER_SECOND * TIMING_SLACK
        if n > room:
            spoken = n / WORDS_PER_SECOND
            has_outro = any(b.anchor == "outro" for b in draft.beats)
            label = f"{name} + outro" if ids and name == ids[-1] and has_outro else name
            out.append(Problem("timing", f"{n} words (~{spoken:.1f}s) over {secs:.1f}s of "
                                         f"footage; cut about {n - int(room)} words", label))
    return out


def check(
    draft: ScriptDraft, report: VideoReport, post: Post, set_card_names: set[str],
    airtime=None,
) -> list[Problem]:
    return [
        *check_contract(draft, report, post),
        *check_spoilers(draft, report, post),
        *check_facts(draft, report, post, set_card_names),
        *check_timing(draft, report, post, airtime),
    ]


# --- model check: the style guide ------------------------------------------------------


class StyleBreak(BaseModel):
    rule: str = Field(description="the style-guide rule, quoted or closely paraphrased")
    quote: str = Field(description="the words in the draft that break it")
    severity: Literal["major", "minor"] = Field(
        description="major = a reader would notice the guide was ignored; minor = nitpick")


class StyleVerdict(BaseModel):
    breaks: list[StyleBreak]


JUDGE_SYSTEM = (
    "You check a short voiceover script against the creator's style guide. Report only "
    "clear breaks of a rule that is written in the guide, quoting the draft. Do not "
    "suggest improvements that no rule asks for. Facts, timing and spoilers are "
    "checked elsewhere: ignore them. An empty list is a good answer."
)


@dataclass(frozen=True)
class JudgeSettings:
    model: str = "claude-opus-5-5"
    effort: str = "low"
    usd_per_mtok_in: float = 4.0
    usd_per_mtok_out: float = 20.0


def judge_style(client, draft: ScriptDraft, style: str,
                settings: JudgeSettings) -> tuple[list[Problem], list[str], float]:
    """→ (major breaks as problems, minor breaks as notes, cost USD)."""
    from pokepipe.narration.script import FALLBACK_BETA

    lines = "\n".join(f"[{b.anchor}] {b.text}" for b in draft.beats)
    response = client.beta.messages.parse(
        model=settings.model,
        max_tokens=4000,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        output_config={"effort": settings.effort},
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": (
            f"<style_guide>\n{style.strip()}\n</style_guide>\n\n"
            f"<draft>\nhook caption: {draft.hook_caption}\nhashtags: "
            f"{' '.join(draft.hashtags)}\n{lines}\n</draft>")}],
        output_format=StyleVerdict,
    )
    usage = response.usage
    cost = (usage.input_tokens * settings.usd_per_mtok_in
            + usage.output_tokens * settings.usd_per_mtok_out) / 1e6
    verdict = response.parsed_output
    if response.stop_reason in ("refusal", "max_tokens") or verdict is None:
        return [], [f"style check skipped ({response.stop_reason})"], cost
    major = [Problem("style", f"{b.rule} — {b.quote!r}") for b in verdict.breaks
             if b.severity == "major"]
    minor = [f"{b.rule} — {b.quote!r}" for b in verdict.breaks if b.severity == "minor"]
    return major, minor, cost
