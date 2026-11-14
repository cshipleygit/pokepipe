import json

from pokepipe.collection.models import PackResult, Progress, PullResult, VideoReport
from pokepipe.collection.values import PricedPull, VideoValues
from pokepipe.narration.caption import MAX_CHARS, build_caption, set_tag, write_captions
from pokepipe.narration.models import Post, PostPlan, PostScript


def prog(owned):
    return Progress(master_owned=owned, master_total=200, full_owned=0, full_total=1,
                    base_owned=0, base_total=1)


def pull(cid, name, rarity, pack, variant="normal"):
    return PullResult(card_id=cid, name=name, number="1", rarity=rarity, variant=variant,
                      pack=pack, time=1.0, new=True, new_card=True, copy_number=1)


REPORT = VideoReport(
    video_slug="v", collection="main", set_id="me6", set_name="Delta Reign",
    before=prog(0), after=prog(30),
    packs=[PackResult(index=1, new_slots=10, progress_after=prog(10)),
           PackResult(index=2, new_slots=9, progress_after=prog(19)),
           PackResult(index=3, new_slots=11, progress_after=prog(30))],
    pulls=[pull("a", "Pikachu", "Common", 1), pull("b", "Talonflame ex", "Double Rare", 1),
           pull("c", "Kecleon", "Illustration Rare", 2), pull("d", "Onix", "Rare", 2),
           pull("e", "Zarude", "Uncommon", 3, "reverse")],
)
PLAN = PostPlan(video_slug="v", posts=[Post(index=1, packs=[1, 2]), Post(index=2, packs=[3])])
VALUES = VideoValues(video_slug="v", set_id="me6", total_usd=45, priced=2, unpriced=3,
                     per_pack_usd={}, pulls=[
                         PricedPull(pull_id="1", pack=1, card_id="b", name="Talonflame ex",
                                    rarity="Double Rare", variant="normal", usd=13.75),
                         PricedPull(pull_id="2", pack=2, card_id="c", name="Kecleon",
                                    rarity="Illustration Rare", variant="normal", usd=31.11),
                         PricedPull(pull_id="3", pack=3, card_id="e", name="Zarude",
                                    rarity="Uncommon", variant="reverse", usd=6.0)])
SCRIPT = PostScript(hook_caption="Bet you can't guess my best pull", beats=[],
                    hashtags=["pokemon", "#PokemonTCG", "#pokemontcg"], post=1, model="m",
                    words=0)


def test_caption_names_the_best_pulls_progress_part_and_hashtags():
    text = build_caption(REPORT, PLAN, 1, SCRIPT, VALUES)
    assert text.splitlines() == [
        "Bet you can't guess my best pull",
        "Pulled: Kecleon (Illustration Rare, $31.11), Talonflame ex (Double Rare, $13.75)",
        "Master set: 19/200 (10%), +19 new",
        "Part 1 of 2",
        "#pokemon #PokemonTCG",  # normalised and de-duplicated
    ]


def test_a_valuable_base_card_is_mentioned_and_defaults_fill_in_without_a_script():
    text = build_caption(REPORT, PLAN, 2, None, VALUES,
                         {"hashtags": ["#mychannel"], "footer": "Shop link in bio"})
    lines = text.splitlines()
    assert lines[0] == "Delta Reign pack opening"
    assert lines[1] == "Pulled: Zarude (reverse holo, $6.00)"
    assert lines[3:] == ["Part 2 of 2", "Shop link in bio",
                         f"#pokemon #pokemontcg #packopening {set_tag('Delta Reign')} #mychannel"]
    assert set_tag("Scarlet & Violet—151") == "#scarletviolet151"


def test_caption_fits_tiktoks_limit():
    many = SCRIPT.model_copy(update={"hashtags": [f"#tag{i}" for i in range(2000)]})
    assert len(build_caption(REPORT, PLAN, 1, many, None)) <= MAX_CHARS


def test_write_captions_writes_one_file_per_post(tmp_path):
    (tmp_path / "collection.json").write_text(REPORT.model_dump_json())
    (tmp_path / "posts.json").write_text(PLAN.model_dump_json())
    (tmp_path / "posts" / "1").mkdir(parents=True)
    (tmp_path / "posts" / "1" / "script.json").write_text(SCRIPT.model_dump_json())
    extra = tmp_path / "caption.json"
    extra.write_text(json.dumps({"hashtags": ["#mychannel"]}))
    paths = write_captions(tmp_path, extra)
    assert [p.parent.name for p in paths] == ["1", "2"]
    assert paths[0].read_text().startswith("Bet you can't guess")
    assert "#mychannel" in paths[1].read_text()
