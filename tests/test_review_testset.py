"""Replay the auto-reviewer against real finished reviews frozen in tests/fixtures/
(`pokepipe review export-testset`). Skipped until some are exported."""

from pathlib import Path

import pytest

from pokepipe.review.auto import AutoPolicy
from pokepipe.review.replay import replay
from pokepipe.review.testset import export, fixture_store, load_baseline

FIXTURES = Path(__file__).parent / "fixtures"
SLUGS = fixture_store(FIXTURES).slugs()


@pytest.mark.skipif(not SLUGS, reason="no reviews exported to tests/fixtures yet")
@pytest.mark.parametrize("slug", SLUGS)
def test_auto_review_makes_no_new_silent_errors_on_real_reviews(slug):
    # Default policy, no calibration file: the test set judges the rules themselves.
    result = replay(fixture_store(FIXTURES), slug, AutoPolicy())
    allowed = load_baseline(FIXTURES).get(slug, 0)
    assert len(result.silent_errors) <= allowed, "\n".join(result.silent_errors)


def test_export_freezes_a_review_without_local_paths(tmp_path):
    from tests.test_auto_review import SLUG, human_review, make_env, make_pack

    store, write, vdir = make_env(tmp_path / "data")
    det = write([make_pack(1, 10)])
    det.video = "C:\\Users\\chance\\Videos\\opening.mp4"
    (vdir / "pulls.json").write_text(det.model_dump_json())
    human_review(store)
    dest = tmp_path / "fixtures"
    assert export(store, SLUG, dest, AutoPolicy()) == 0
    frozen = fixture_store(dest)
    assert frozen.slugs() == [SLUG] and frozen.detection(SLUG).video == "opening.mp4"
    assert load_baseline(dest) == {SLUG: 0}
    assert replay(frozen, SLUG, AutoPolicy()).silent_errors == []
