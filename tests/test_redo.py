from pokepipe.narration.models import Post, PostPlan
from pokepipe.narration.service import VideoFiles
from pokepipe.review.models import PullUpdate, Status
from pokepipe.watch.redo import clear, stale_posts, stamp
from tests.test_auto_review import SLUG, human_review, make_env, make_pack


def test_only_posts_the_fix_touches_are_redone(tmp_path):
    store, write, vdir = make_env(tmp_path)
    write([make_pack(1, 10), make_pack(2, 100), make_pack(3, 200)])
    human_review(store)
    files = VideoFiles(vdir)
    plan = PostPlan(video_slug=SLUG, posts=[Post(index=1, packs=[1]),
                                            Post(index=2, packs=[2, 3])])
    stamp(files, plan, store.load(SLUG))
    (files.post_dir(1) / "post.mp4").write_bytes(b"x")
    assert stale_posts(files, plan, store.load(SLUG)) == []

    # a wrong card in pack 2 is fixed: post 2 changes, post 1 (pack 1 only) doesn't
    pull = next(p for p in store.load(SLUG).pulls if p.pack == 2)
    store.update(SLUG, pull.id, PullUpdate(card_id="s-6"))
    store.complete(SLUG)
    assert stale_posts(files, plan, store.load(SLUG)) == [2]

    # a fix in pack 1 changes the progress shown in every later post too
    store.update(SLUG, 0, PullUpdate(status=Status.REJECTED))
    store.complete(SLUG)
    assert stale_posts(files, plan, store.load(SLUG)) == [1, 2]
    clear(files, [1, 2])
    assert not files.post_dir(1).exists()


def test_posts_dropped_from_the_plan_are_stale(tmp_path):
    store, write, vdir = make_env(tmp_path)
    write([make_pack(1, 10), make_pack(2, 100)])
    human_review(store)
    files = VideoFiles(vdir)
    old = PostPlan(video_slug=SLUG, posts=[Post(index=1, packs=[1]), Post(index=2, packs=[2])])
    stamp(files, old, store.load(SLUG))
    new = PostPlan(video_slug=SLUG, posts=[Post(index=1, packs=[1, 2])])
    assert stale_posts(files, new, store.load(SLUG)) == [1, 2]
