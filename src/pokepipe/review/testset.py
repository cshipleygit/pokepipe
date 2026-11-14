"""Freeze finished human reviews into a test set (`pokepipe review export-testset`).

A finished review plus detection output is a labelled example of every call a person
made. Copying them into `tests/fixtures/` (JSON only: no footage, frames or card
images) lets `tests/test_review_testset.py` replay the auto-reviewer against real
decisions on every test run.

`baseline.json` holds the silent errors each video is allowed. It starts at whatever
the current policy scores, and is a ratchet: lower it as the auto-reviewer improves,
and a change that lets a new mistake through fails the tests.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pokepipe.cards.cache import SetCache
from pokepipe.detection.models import PullsFile
from pokepipe.recover.models import FILENAME as RECOVERED
from pokepipe.review.auto import AutoPolicy
from pokepipe.review.replay import replay
from pokepipe.review.store import ReviewStore

BASELINE = "baseline.json"


def export(store: ReviewStore, slug: str, dest: Path, policy: AutoPolicy) -> int:
    """Copy one finished human review into `dest`; returns its silent errors now."""
    result = replay(store, slug, policy)  # also refuses unfinished / auto-decided reviews
    src = store.video_dir(slug)
    out = dest / "reviews" / slug
    out.mkdir(parents=True, exist_ok=True)
    det = PullsFile.model_validate_json((src / "pulls.json").read_text(encoding="utf-8"))
    det.video = Path(det.video.replace("\\", "/")).name  # no local paths in the repo
    (out / "pulls.json").write_text(det.model_dump_json(indent=1), encoding="utf-8")
    shutil.copyfile(src / "review.json", out / "review.json")
    if (src / RECOVERED).exists():
        shutil.copyfile(src / RECOVERED, out / RECOVERED)

    cache = SetCache(store.sets_root, det.set_id)
    sets_out = dest / "sets" / cache.dir.name
    sets_out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache.set_path, sets_out / "set.json")
    shutil.copyfile(cache.cards_path, sets_out / "cards.json")

    path = dest / "reviews" / BASELINE
    baseline = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    baseline[slug] = len(result.silent_errors)
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(result.silent_errors)


def fixture_store(root: Path) -> ReviewStore:
    return ReviewStore(root / "reviews", root / "sets")


def load_baseline(root: Path) -> dict[str, int]:
    path = root / "reviews" / BASELINE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
