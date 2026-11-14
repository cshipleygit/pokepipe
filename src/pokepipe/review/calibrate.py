"""Fit detection confidence on finished human reviews (`pokepipe review fit`).

`detection.segment.confidence` is hand-weighted. Every finished review labels each
detected pull: kept as detected (right) or rejected / corrected to another card
(wrong). This fits a small logistic regression on the signals detection already
records and picks the cut-off the auto-reviewer uses.

Only *human* decisions are used: pulls the auto-reviewer confirmed (`decided_by ==
"auto"`) are skipped, so it never learns from its own guesses.

The threshold is chosen to let **no known wrong pull through**: just above the
highest-scoring wrong pull, then reported with how many right pulls that sends to a
person. With a few hundred labels and a handful of errors this is a starting point,
not a guarantee — re-fit as reviews accumulate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pokepipe.detection.models import Pull, PullsFile
from pokepipe.review.models import ReviewState, Status

FILENAME = "review_calibration.json"
FEATURES = ("peak_score", "mean_margin", "log_frames", "lead")
MIN_WRONG = 3  # fewer wrong examples than this and there is nothing to fit


def _lead(pull: Pull) -> float:
    others = [a.score for a in pull.alternatives
              if a.card_id != pull.card_id and not a.card_id.startswith("decoy:")]
    return min(pull.peak_score - max(others), 0.2) if others else 0.2


def features(pull: Pull) -> list[float]:
    return [pull.peak_score, pull.mean_margin, math.log(max(pull.frames, 1)), _lead(pull)]


@dataclass
class Example:
    slug: str
    card: str
    confidence: float  # detection's heuristic, for the threshold table
    x: list[float]
    right: bool
    why: str = ""  # for wrong ones: rejected / corrected to X


def examples(det: PullsFile, review: ReviewState) -> list[Example]:
    """Labelled detected pulls from one finished review (human decisions only)."""
    out = []
    if not review.completed:
        return out
    for p in review.pulls:
        if p.detected_index is None or p.decided_by == "auto" or p.status == Status.PENDING:
            continue
        pull = det.pulls[p.detected_index]
        rejected = p.status == Status.REJECTED
        why = "rejected" if rejected else (f"corrected to {p.card_id}" if p.corrected else "")
        out.append(Example(review.video_slug, pull.name, pull.confidence, features(pull),
                           right=not why, why=why))
    return out


@dataclass
class Calibration:
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float
    threshold: float
    n: int
    wrong: int
    fitted_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def probability(self, pull: Pull) -> float:
        z = (np.asarray(features(pull)) - self.mean) / self.std
        return float(1 / (1 + np.exp(-(z @ np.asarray(self.weights) + self.bias))))

    def summary(self) -> dict:
        return {"threshold": round(self.threshold, 4), "n": self.n, "wrong": self.wrong,
                "fitted_at": self.fitted_at}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Calibration | None:
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def fit(data: list[Example], l2: float = 1.0, steps: int = 50) -> Calibration:
    """L2-regularised logistic regression (Newton's method), threshold chosen so no
    training example that was wrong scores at or above it."""
    wrong = sum(not e.right for e in data)
    if wrong < MIN_WRONG:
        raise ValueError(f"only {wrong} wrong pulls in {len(data)} labelled; need "
                         f"{MIN_WRONG}+ to fit (use the confidence table instead)")
    X = np.asarray([e.x for e in data], dtype=float)
    y = np.asarray([e.right for e in data], dtype=float)
    mean, std = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mean) / std, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    reg = np.eye(len(w)) * l2
    reg[-1, -1] = 0  # don't shrink the bias
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Z @ w))
        grad = Z.T @ (p - y) + reg @ w
        hess = Z.T @ (Z * (p * (1 - p))[:, None]) + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    probs = 1 / (1 + np.exp(-Z @ w))
    worst_wrong = max(pr for pr, e in zip(probs, data, strict=True) if not e.right)
    return Calibration(mean=mean.tolist(), std=std.tolist(), weights=w[:-1].tolist(),
                       bias=float(w[-1]), threshold=float(min(worst_wrong + 1e-4, 0.9999)),
                       n=len(data), wrong=wrong)


def threshold_table(data: list[Example], cuts: list[float]) -> list[tuple[float, int, int]]:
    """For each cut on detection's heuristic confidence: (cut, wrong pulls that would be
    auto-confirmed, right pulls that would go to a person)."""
    return [(c, sum(e.confidence >= c and not e.right for e in data),
             sum(e.confidence < c and e.right for e in data)) for c in cuts]
