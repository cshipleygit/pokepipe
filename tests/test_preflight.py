from pokepipe.cards.cache import SetCache
from pokepipe.cards.models import Card, CardSet
from pokepipe.config import Settings
from pokepipe.preflight import Level, check_set, elevenlabs_room, run_all


def test_elevenlabs_room_counts_videos_of_narration():
    assert elevenlabs_room({"character_count": 0, "character_limit": 30000}, 3).level == Level.OK
    assert elevenlabs_room({"character_count": 25000, "character_limit": 30000},
                           3).level == Level.WARN
    low = elevenlabs_room({"character_count": 29000, "character_limit": 30000}, 1)
    assert low.level == Level.FAIL and "top up" in low.detail


def test_set_check_explains_what_to_run(tmp_path):
    s = Settings(data_dir=tmp_path, _env_file=None)
    assert "cards fetch sv9" in check_set(s, "sv9").detail
    cache = SetCache(s.sets_dir, "sv9")
    cache.save(CardSet(id="sv9", name="Journey Together", series="Scarlet & Violet"),
               [Card(id="sv9-1", name="A", number="1", set_id="sv9")])
    assert "cards embed sv9" in check_set(s, "sv9").detail
    tag = f"{s.clip_model}__{s.clip_pretrained}__full"
    cache.index_path(tag).parent.mkdir(parents=True)
    cache.index_path(tag).write_bytes(b"")
    assert check_set(s, "sv9").level == Level.OK
    other = SetCache(s.sets_dir, "old")
    other.save(CardSet(id="old", name="Old", series="Sun & Moon"), [])
    other.index_path(tag).parent.mkdir(parents=True)
    other.index_path(tag).write_bytes(b"")
    assert "unverified" in check_set(s, "old").detail


def test_offline_run_never_raises(tmp_path):
    checks = run_all(Settings(data_dir=tmp_path, _env_file=None), None, 2, online=False)
    assert {c.name for c in checks} >= {"GPU", "ffmpeg", "Card set", "Disk", "Review"}
