import base64
import json
from types import SimpleNamespace

import pytest
import requests
import responses

from pokepipe.collection.models import PackResult, Progress, PullResult, VideoReport
from pokepipe.narration import posts as P
from pokepipe.narration import service
from pokepipe.narration.models import Beat, Post, PostScript, ScriptDraft
from pokepipe.narration.script import ScriptError, ScriptSettings, build_prompt, generate_script
from pokepipe.narration.voice import (
    API,
    ElevenLabsClient,
    VoiceCache,
    VoiceError,
    VoiceSettings,
    join_beats,
    time_script,
)


def prog(n: int) -> Progress:
    return Progress(
        master_owned=n, master_total=360, full_owned=n, full_total=207, base_owned=n, base_total=165
    )


def pull(pack, t, name, rarity="Common", new=True, variant="normal", copy=1):
    return PullResult(
        card_id=f"s-{name}",
        name=name,
        number="1",
        rarity=rarity,
        variant=variant,
        pack=pack,
        time=t,
        new=new,
        new_card=new and variant != "reverse",
        copy_number=copy,
    )


REPORT = VideoReport(
    video_slug="v",
    collection="test",
    set_id="sv3pt5",
    set_name="151",
    before=prog(10),
    after=prog(16),
    packs=[
        PackResult(index=1, new_slots=2, progress_after=prog(12)),
        PackResult(index=2, new_slots=3, progress_after=prog(15)),
        PackResult(index=3, new_slots=1, progress_after=prog(16)),
    ],
    pulls=[
        pull(1, 10, "Pidgey"),
        pull(1, 12, "Zubat", variant="reverse"),
        pull(2, 40, "Rattata", new=False, copy=2),
        pull(2, 44, "Charizard ex", rarity="Special Illustration Rare"),
        pull(3, 70, "Ditto", rarity="Rare"),
    ],
)


# --- posts ---------------------------------------------------------------------------------


def test_parse_spec_ranges_singles_and_combined():
    assert P.parse_spec("1-3, 5, 6+8", set(range(1, 9))) == [[1, 2, 3], [5], [6, 8]]


@pytest.mark.parametrize(
    "spec,msg",
    [
        ("1-3, 3", "more than one post"),
        ("13", "no pack"),
        ("3-1", "backwards"),
        ("a", "can't read"),
        (" , ", "no posts"),
    ],
)
def test_parse_spec_errors(spec, msg):
    with pytest.raises(P.PostSpecError, match=msg):
        P.parse_spec(spec, set(range(1, 13)))


def test_suggest_isolates_hit_packs_and_groups_the_rest():
    rarities = {i: ["Common", "Rare"] for i in range(1, 13)}
    rarities[2].append("Illustration Rare")
    rarities[9].append("Special Illustration Rare")
    groups = P.suggest(rarities)
    assert groups == [[1], [2], [3, 4, 5], [6, 7, 8], [9], [10, 11, 12]]
    assert P.to_spec(groups) == "1, 2, 3-5, 6-8, 9, 10-12"
    assert P.to_spec([[6, 8]]) == "6+8"


# --- prompt & script -----------------------------------------------------------------------


def test_prompt_contains_only_this_posts_facts():
    system, user = build_prompt(REPORT, Post(index=2, packs=[2]), 3, "Be funny.", ScriptSettings())
    assert "Be funny." in system and "never invent" in system
    facts = json.loads(user.split("<facts>")[1].split("</facts>")[0])
    assert [p["id"] for p in facts["pulls_in_reveal_order"]] == ["p2-1", "p2-2"]
    assert facts["pulls_in_reveal_order"][0]["collection"] == "duplicate (copy #2)"
    assert "12/360" in facts["collection_before_post"]  # progress after pack 1
    assert "15/360" in facts["collection_after_post"]
    assert facts["new_master_set_slots_this_post"] == 3
    assert "at most 37 words" in user  # 15 s minimum x 2.5 words/s


class FakeAnthropic:
    def __init__(self, draft=None, stop_reason="end_turn"):
        self.calls = []
        self.draft, self.stop_reason = draft, stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            parsed_output=self.draft,
            model=kwargs["model"],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500),
        )


DRAFT = ScriptDraft(
    hook_caption="This pack had no business being this good",
    beats=[
        Beat(anchor="intro", text="Let's see what this one's hiding."),
        Beat(anchor="p2-2", text="Special illustration rare Charizard!"),
        Beat(anchor="p7-1", text="Into the binder it goes."),
    ],  # bad anchor
    hashtags=["#pokemon", "#pokemon151"],
)


def test_generate_script_calls_claude_correctly_and_validates_anchors():
    client = FakeAnthropic(DRAFT)
    script, cost, warnings = generate_script(client, REPORT, Post(index=2, packs=[2]), 3, "")
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_format"] is ScriptDraft
    assert call["output_config"] == {"effort": "medium"}
    assert cost == pytest.approx(1000 * 4e-6 + 500 * 20e-6)
    assert [b.anchor for b in script.beats] == ["intro", "p2-2", "p2-2"]
    assert "p7-1" in warnings[0]
    assert script.words == 15


def test_anthropic_client_sends_workspace_header_only_when_set():
    from pokepipe.narration.script import anthropic_client

    scoped = anthropic_client("sk-test", "wrkspc_123")
    assert scoped.default_headers["anthropic-workspace-id"] == "wrkspc_123"
    plain = anthropic_client("sk-test", None)
    assert "anthropic-workspace-id" not in plain.default_headers


def test_refusal_raises():
    with pytest.raises(ScriptError, match="declined"):
        generate_script(FakeAnthropic(DRAFT, "refusal"), REPORT, Post(index=1, packs=[1]), 1, "")


# --- voice timing --------------------------------------------------------------------------

SCRIPT = PostScript(
    post=1,
    model="m",
    words=4,
    hook_caption="h",
    hashtags=[],
    beats=[Beat(anchor="intro", text="Hi there"), Beat(anchor="p1-1", text="Pidgey!")],
)


def alignment_for(text: str, step: float = 0.1) -> dict:
    return {
        "characters": list(text),
        "character_start_times_seconds": [i * step for i in range(len(text))],
        "character_end_times_seconds": [(i + 1) * step for i in range(len(text))],
    }


def test_time_script_maps_characters_to_beats_and_words():
    text, spans = join_beats(SCRIPT)
    assert text == "Hi there Pidgey!" and spans == [(0, 8), (9, 16)]
    beats, words, duration = time_script(SCRIPT, alignment_for(text))
    assert [(b.anchor, b.start, b.end) for b in beats] == [("intro", 0.0, 0.8), ("p1-1", 0.9, 1.6)]
    assert [(w.text, w.start, w.end) for w in words] == [
        ("Hi", 0.0, 0.2),
        ("there", 0.3, 0.8),
        ("Pidgey!", 0.9, 1.6),
    ]
    assert duration == 1.6


def test_time_script_tolerates_normalised_alignment_length():
    text, _ = join_beats(SCRIPT)
    beats, _, duration = time_script(SCRIPT, alignment_for(text + "xxxx"))
    assert beats[0].start == 0 and duration == pytest.approx(2.0)


# --- ElevenLabs client ---------------------------------------------------------------------

SETTINGS = VoiceSettings(voice_id="voice123", usd_per_1k_chars=0.30)


def mock_tts(text="Hi there Pidgey!"):
    responses.post(
        f"{API}/text-to-speech/voice123/with-timestamps",
        json={
            "audio_base64": base64.b64encode(b"MP3DATA").decode(),
            "alignment": alignment_for(text),
        },
    )


@responses.activate
def test_synthesize_sends_key_and_returns_audio():
    mock_tts()
    client = ElevenLabsClient("key-abc", session=requests.Session())
    audio, alignment = client.synthesize("Hi there Pidgey!", SETTINGS)
    req = responses.calls[0].request
    assert req.headers["xi-api-key"] == "key-abc"
    assert "output_format=mp3_44100_128" in req.url
    assert json.loads(req.body) == {
        "text": "Hi there Pidgey!",
        "model_id": "eleven_multilingual_v2",
    }
    assert audio == b"MP3DATA" and len(alignment["characters"]) == 16


@responses.activate
def test_elevenlabs_errors_are_explained():
    responses.post(f"{API}/text-to-speech/voice123/with-timestamps", status=401, json={})
    with pytest.raises(VoiceError, match="API key rejected"):
        ElevenLabsClient("bad", session=requests.Session()).synthesize("x", SETTINGS)
    with pytest.raises(VoiceError, match="not set"):
        ElevenLabsClient("")


# --- end to end with budget and cache ----------------------------------------------------------


@pytest.fixture
def files(tmp_path):
    root = tmp_path / "videos" / "v"
    root.mkdir(parents=True)
    (root / "collection.json").write_text(REPORT.model_dump_json())
    return service.VideoFiles(root)


def test_plan_write_scripts_and_ledger(files):
    plan = service.plan_posts(files, "v")
    assert [p.packs for p in plan.posts] == [[1], [2], [3]]  # pack 2 has the SIR
    plan = service.plan_posts(files, "v", "1+3, 2")
    assert [p.packs for p in plan.posts] == [[1, 3], [2]]

    client = FakeAnthropic(DRAFT)
    runs = service.write_scripts(files, client, "", ScriptSettings(), service.Budget(1.0), [2])
    assert len(runs) == 1 and (files.post_dir(2) / "script.json").exists()
    assert files.load_ledger().total == pytest.approx(0.014)


def test_budget_blocks_before_paying(files):
    service.plan_posts(files, "v", "1, 2")
    with pytest.raises(service.BudgetError, match="over the \\$0.01 budget"):
        service.write_scripts(
            files, FakeAnthropic(DRAFT), "", ScriptSettings(), service.Budget(0.01), [1]
        )
    assert files.load_ledger().total == 0


@responses.activate
def test_voice_is_cached_and_budgeted(files, tmp_path):
    service.plan_posts(files, "v", "1")
    files.save(files.post_dir(1) / "script.json", SCRIPT)
    mock_tts()
    client = ElevenLabsClient("k", session=requests.Session())
    cache = VoiceCache(tmp_path / "tts")

    [t1] = service.voice_posts(files, client, cache, SETTINGS, service.Budget(1.0))
    assert not t1.cached and (files.post_dir(1) / "voice.mp3").read_bytes() == b"MP3DATA"
    assert files.load_ledger().total == pytest.approx(16 * 0.30 / 1000)

    [t2] = service.voice_posts(files, client, cache, SETTINGS, service.Budget(1.0))
    assert t2.cached and len(responses.calls) == 1  # no second paid call
    assert files.load_ledger().total == pytest.approx(16 * 0.30 / 1000)
    assert [b.anchor for b in t2.beats] == ["intro", "p1-1"]


def test_voice_budget_refuses_and_force_overrides(files, tmp_path):
    service.plan_posts(files, "v", "1")
    files.save(files.post_dir(1) / "script.json", SCRIPT)
    tight = VoiceSettings(voice_id="voice123", usd_per_1k_chars=1000.0)
    with pytest.raises(service.BudgetError):
        service.voice_posts(files, None, VoiceCache(tmp_path / "tts"), tight, service.Budget(1.0))
