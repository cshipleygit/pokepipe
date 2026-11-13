"""Script critic checks and the writer ⇄ critic loop (fake Claude)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pokepipe.collection.models import PackResult, Progress, PullResult, VideoReport
from pokepipe.narration import critic
from pokepipe.narration.models import Beat, Post, ScriptDraft
from pokepipe.narration.script import CriticSettings, write_post_script
from pokepipe.narration.writer import Attempt, best


def prog(n: int) -> Progress:
    return Progress(master_owned=n, master_total=360, full_owned=n, full_total=207,
                    base_owned=n, base_total=165)


def pull(pack, t, name, rarity="Common", new=True, new_card=None, variant="normal", copy=1):
    return PullResult(card_id=f"s-{name}", name=name, number="1", rarity=rarity,
                      variant=variant, pack=pack, time=t, new=new,
                      new_card=new if new_card is None else new_card, copy_number=copy)


# Pack 1 is a lone promo (not an opened pack), so detection's pack 2 is "pack 1".
REPORT = VideoReport(
    video_slug="v", collection="test", set_id="s", set_name="Test", before=prog(0),
    after=prog(7),
    packs=[PackResult(index=i, new_slots=1, progress_after=prog(i)) for i in (1, 2, 3)],
    pulls=[
        pull(1, 5, "Mew", rarity="Promo"),
        pull(2, 10, "Pidgey"), pull(2, 12, "Zubat", new=False, copy=2),
        pull(2, 14, "Charizard ex", rarity="Special Illustration Rare"),
        pull(3, 40, "Ditto"), pull(3, 42, "Oddish"),
        pull(3, 44, "Cacturne", rarity="Rare", new=True, new_card=False),  # new printing
    ],
)
POST = Post(index=1, packs=[2, 3])
NAMES = {"Pidgey", "Zubat", "Charizard ex", "Ditto", "Oddish", "Cacturne", "Mew",
         "Blastoise ex"}


def draft(*beats: tuple[str, str], caption="Guess my best pull") -> ScriptDraft:
    return ScriptDraft(hook_caption=caption, hashtags=["#pokemon"],
                       beats=[Beat(anchor=a, text=t) for a, t in beats])


def kinds(problems) -> list[str]:
    return [p.kind for p in problems]


def test_a_clean_script_passes():
    d = draft(("intro", "Call the hit in the comments."), ("p2-1", "Pack one, Pidgey."),
              ("p2-3", "Special illustration rare Charizard!"),
              ("p3-3", "Holo Cacturne, I already had the reverse."),
              ("outro", "Follow for part two."))
    assert critic.check(d, REPORT, POST, NAMES) == []


def test_contract_anchor_order_length_and_money():
    d = draft(("p2-3", "Charizard first."), ("p2-1", "Then Pidgey."),
              ("p7-1", "Where am I?"), ("p3-1", "This Ditto is worth twenty dollars."),
              ("outro", " ".join(["word"] * 21)), caption="one two three four five six seven "
                                                          "eight nine")
    msgs = [p.message for p in critic.check_contract(d, REPORT, POST)]
    assert any("hook caption" in m for m in msgs)
    assert any("out of reveal order" in m for m in msgs)
    assert any("'p7-1'" in m for m in msgs)
    assert any("money" in m for m in msgs)
    assert any("21 words" in m for m in msgs)


def test_facts_pack_numbers_count_opened_packs_only():
    ok = draft(("p2-1", "Pack one starts with Pidgey."), ("p3-1", "Pack 2, Ditto."))
    assert critic.check_facts(ok, REPORT, POST, NAMES) == []
    bad = draft(("p2-1", "Pack two starts with Pidgey."))
    assert "pack 1" in critic.check_facts(bad, REPORT, POST, NAMES)[0].message


def test_facts_cards_outside_the_post_and_wrong_rarity():
    d = draft(("p2-1", "Not a Blastoise, just Pidgey."),
              ("p2-2", "An ultra rare Zubat? No, a reverse."))
    msgs = [p.message for p in critic.check_facts(d, REPORT, POST, NAMES)]
    assert any("Blastoise" in m for m in msgs)
    assert any("'ultra rare'" in m for m in msgs)


def test_facts_new_and_duplicate_claims():
    d = draft(("p2-2", "Zubat, a brand new card for me."),
              ("p3-1", "Ditto, another duplicate."),
              ("p3-3", "Cacturne, I already had the reverse, so a new printing."))
    msgs = [p.message for p in critic.check_facts(d, REPORT, POST, NAMES)]
    assert len(msgs) == 2  # the Cacturne line is true
    assert "copy #2" in msgs[0] and "first copy" in msgs[1]


def test_timing_regions_absorb_cards_without_lines():
    air = SimpleNamespace(before_first=2.0, per_card={
        "p2-1": 2.0, "p2-2": 2.0, "p2-3": 4.0, "p3-1": 1.0, "p3-2": 1.0, "p3-3": 3.0},
        after_last=3.0, duration=20.0)
    # Room = seconds x 2.5 words/s x 1.15 slack.
    d = draft(("intro", "one two three four five six"),  # 2 s alone would be over (5.75)...
              # ...but p2-1 and p2-2 have no line, so their 4 s join the intro: 6 s, fine
              ("p2-3", "Charizard " * 10 + "!"),  # 11 words in 4+1+1 = 6 s (17.25): fine
              ("p3-3", "Cacturne nice."), ("outro", "word " * 8))  # 10 words in 3 s (8.6): over
    probs = critic.check_timing(d, REPORT, POST, air)
    assert [p.anchor for p in probs] == ["p3-3 + outro"]
    assert critic.check_timing(d, REPORT, POST, None) == []


# --- writer ⇄ critic loop ------------------------------------------------------------


class FakeClaude:
    """Scripted drafts for the writer; a scripted verdict for the style judge."""

    def __init__(self, drafts, verdicts=()):
        self.drafts, self.verdicts, self.calls = list(drafts), list(verdicts), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kw):
        self.calls.append(kw)
        usage = SimpleNamespace(input_tokens=1000, output_tokens=200)
        if kw["output_format"] is critic.StyleVerdict:
            out = self.verdicts.pop(0) if self.verdicts else critic.StyleVerdict(breaks=[])
        else:
            out = self.drafts.pop(0)
        return SimpleNamespace(stop_reason="end_turn", parsed_output=out, model=kw["model"],
                               usage=usage)


SPOILER = draft(("intro", "Charizard is in here."), ("p2-3", "There it is."))
WRONG_PACK = draft(("intro", "Guess the hit."), ("p2-1", "Pack two, Pidgey."),
                   ("p2-3", "Charizard!"))
CLEAN = draft(("intro", "Guess the hit."), ("p2-1", "Pack one, Pidgey."),
              ("p2-3", "Charizard!"))
FULL = CriticSettings(set_card_names=frozenset(NAMES), max_rewrites=2)


def run(client, critic_settings=FULL, budget=lambda spent: None, style=""):
    return write_post_script(client, REPORT, POST, 2, style, critic=critic_settings,
                             budget_check=budget)


def test_rewrites_until_clean_with_every_problem_fed_back():
    client = FakeClaude([SPOILER, WRONG_PACK, CLEAN])
    script, cost, warnings, rec = run(client)
    assert len(client.calls) == 3 and not warnings and rec.passed and rec.chosen == 2
    assert [a.kinds for a in rec.attempts] == [["spoiler"], ["fact"], []]
    second, third = (c["messages"][0]["content"] for c in client.calls[1:])
    assert "no-spoiler rule" in second and "<previous_draft>" in second
    assert "fact rules" in third and "says 'pack two'" in third
    assert script.beats[1].text == "Pack one, Pidgey."


def test_stops_after_max_rewrites_and_keeps_the_best_attempt():
    worse = draft(("intro", "Charizard ex, special illustration rare, inside!"),
                  ("p2-1", "Pack two."), ("p2-3", "Yes."))
    client = FakeClaude([WRONG_PACK, worse, worse])
    script, _, warnings, rec = run(client)
    assert len(client.calls) == 3 and rec.chosen == 0  # first draft had fewest problems
    assert warnings == ["fact: [p2-1] says 'pack two' but this card is in pack 1"]


def test_budget_stops_rewrites_but_keeps_the_draft():
    def broke(spent):
        raise RuntimeError("over budget")

    client = FakeClaude([WRONG_PACK])
    script, _, warnings, rec = run(client, budget=broke)
    assert len(client.calls) == 1 and "over budget" in rec.stop
    assert any("over budget" in w for w in warnings)


def test_style_judge_major_breaks_trigger_a_rewrite_minor_ones_dont():
    judge = critic.JudgeSettings()
    settings = CriticSettings(set_card_names=frozenset(NAMES), max_rewrites=2, judge=judge)
    major = critic.StyleVerdict(breaks=[critic.StyleBreak(
        rule="Never start with 'Hey guys'", quote="Hey guys", severity="major")])
    minor = critic.StyleVerdict(breaks=[critic.StyleBreak(
        rule="at most one catchphrase", quote="Into the binder", severity="minor")])
    client = FakeClaude([CLEAN, CLEAN], [major, minor])
    _, cost, warnings, rec = run(client, settings, style="Never start with 'Hey guys'.")
    assert len(client.calls) == 4  # draft, judge, rewrite, judge
    assert rec.attempts[0].kinds == ["style"] and rec.passed
    assert warnings == ["style (minor): at most one catchphrase — 'Into the binder'"]
    assert client.calls[1]["output_config"] == {"effort": "low"}
    assert cost == pytest.approx(4 * (1000 * 4e-6 + 200 * 20e-6))


def test_without_critic_settings_only_spoilers_are_checked_once():
    client = FakeClaude([WRONG_PACK])
    _, _, warnings, rec = run(client, critic_settings=None)
    assert len(client.calls) == 1 and not warnings  # the pack number isn't checked


def test_best_prefers_fewest_problems_then_latest():
    a = Attempt(CLEAN, None, [critic.Problem("fact", "x")])
    b = Attempt(CLEAN, None, [critic.Problem("fact", "y")])
    c = Attempt(CLEAN, None, [critic.Problem("fact", "y"), critic.Problem("timing", "z")])
    assert best([a, b, c]) is b
