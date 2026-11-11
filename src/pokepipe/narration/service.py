"""Stage 5 orchestration: plan posts → write scripts → synthesise voice, within budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pokepipe.collection.models import VideoReport
from pokepipe.narration import posts as posts_mod
from pokepipe.narration.models import CostEntry, CostLedger, PostPlan, PostScript, VoiceTiming
from pokepipe.narration.script import ScriptSettings, generate_script
from pokepipe.narration.voice import (
    ElevenLabsClient,
    VoiceCache,
    VoiceSettings,
    join_beats,
    time_script,
)


class NarrationError(RuntimeError):
    pass


class BudgetError(NarrationError):
    pass


@dataclass
class VideoFiles:
    root: Path  # data/videos/<slug>

    @property
    def report(self) -> Path:
        return self.root / "collection.json"

    @property
    def plan(self) -> Path:
        return self.root / "posts.json"

    @property
    def ledger(self) -> Path:
        return self.root / "costs.json"

    def post_dir(self, n: int) -> Path:
        return self.root / "posts" / str(n)

    def load_report(self) -> VideoReport:
        if not self.report.exists():
            raise NarrationError(
                "no collection.json — finish the review and run `pokepipe collection apply` first"
            )
        return VideoReport.model_validate_json(self.report.read_text(encoding="utf-8"))

    def load_plan(self) -> PostPlan:
        if not self.plan.exists():
            raise NarrationError("no posts.json — run `pokepipe narrate plan` first")
        return PostPlan.model_validate_json(self.plan.read_text(encoding="utf-8"))

    def load_ledger(self) -> CostLedger:
        if self.ledger.exists():
            return CostLedger.model_validate_json(self.ledger.read_text(encoding="utf-8"))
        return CostLedger()

    def save(self, path: Path, model) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


@dataclass
class Budget:
    usd: float = 1.0
    force: bool = False

    def check(self, ledger: CostLedger, estimate: float, what: str) -> None:
        if self.force or ledger.total + estimate <= self.usd:
            return
        raise BudgetError(
            f"{what} (~${estimate:.3f}) would take this video to "
            f"${ledger.total + estimate:.3f}, over the ${self.usd:.2f} budget. "
            "Shorten the scripts, or pass --force."
        )


# --- posts ------------------------------------------------------------------------------------


def plan_posts(
    files: VideoFiles, slug: str, spec: str | None = None, seconds: float | None = None
) -> PostPlan:
    report = files.load_report()
    rarities: dict[int, list[str | None]] = {}
    for p in report.pulls:
        rarities.setdefault(p.pack, []).append(p.rarity)
    groups = posts_mod.parse_spec(spec, set(rarities)) if spec else posts_mod.suggest(rarities)
    plan = posts_mod.make_plan(slug, groups, seconds)
    files.save(files.plan, plan)
    return plan


def _select(plan: PostPlan, only: list[int] | None):
    chosen = [p for p in plan.posts if not only or p.index in only]
    unknown = set(only or []) - {p.index for p in plan.posts}
    if unknown:
        raise NarrationError(f"no post(s) {sorted(unknown)} in posts.json")
    return chosen


# --- scripts ----------------------------------------------------------------------------------

SCRIPT_ESTIMATE_USD = 0.05  # generous per-call ceiling used only for the pre-call budget check


@dataclass
class ScriptRun:
    script: PostScript
    cost: float
    warnings: list[str] = field(default_factory=list)


def _airtime(files: VideoFiles, report: VideoReport, post, settings: ScriptSettings):
    """Seconds each card is on screen in a draft edit, so the script fits the footage.
    None when the footage data isn't available (the script is then length-only)."""
    from pokepipe.detection.models import PullsFile
    from pokepipe.render.inputs import airtime, load_moments
    from pokepipe.review.models import ReviewState

    det_path, review_path = files.root / "pulls.json", files.root / "review.json"
    if not (det_path.exists() and review_path.exists()):
        return None
    det = PullsFile.model_validate_json(det_path.read_text(encoding="utf-8"))
    review = ReviewState.model_validate_json(review_path.read_text(encoding="utf-8"))
    target = post.seconds or settings.target_seconds(len(post.packs))
    try:
        return airtime(det, review, report, post, target, load_moments(files.root))
    except ValueError:  # no footage for these packs
        return None


def write_scripts(
    files: VideoFiles,
    client,
    style: str,
    settings: ScriptSettings,
    budget: Budget,
    only: list[int] | None = None,
) -> list[ScriptRun]:
    report, plan, ledger = files.load_report(), files.load_plan(), files.load_ledger()
    runs = []
    for post in _select(plan, only):
        budget.check(ledger, SCRIPT_ESTIMATE_USD, f"script for post {post.index}")
        script, cost, warnings = generate_script(
            client,
            report,
            post,
            len(plan.posts),
            style,
            settings,
            airtime=_airtime(files, report, post, settings),
        )
        files.save(files.post_dir(post.index) / "script.json", script)
        ledger.entries.append(
            CostEntry(
                kind="script",
                post=post.index,
                usd=cost,
                detail=f"{script.model}, {script.words} words",
            )
        )
        files.save(files.ledger, ledger)
        runs.append(ScriptRun(script, cost, warnings))
    return runs


# --- voice --------------------------------------------------------------------------------------


def voice_posts(
    files: VideoFiles,
    client: ElevenLabsClient | None,
    cache: VoiceCache,
    settings: VoiceSettings,
    budget: Budget,
    only: list[int] | None = None,
) -> list[VoiceTiming]:
    plan, ledger = files.load_plan(), files.load_ledger()
    out = []
    for post in _select(plan, only):
        script_path = files.post_dir(post.index) / "script.json"
        if not script_path.exists():
            raise NarrationError(f"post {post.index} has no script — run `pokepipe narrate script`")
        script = PostScript.model_validate_json(script_path.read_text(encoding="utf-8"))
        text, _ = join_beats(script)
        key = VoiceCache.key(text, settings)
        hit = cache.get(key)
        if hit:
            audio, alignment = hit
        else:
            cost = settings.cost(len(text))
            budget.check(ledger, cost, f"voice for post {post.index} ({len(text)} characters)")
            if client is None:
                raise NarrationError("ElevenLabs client not configured")
            audio, alignment = client.synthesize(text, settings)
            cache.put(key, audio, alignment)
            ledger.entries.append(
                CostEntry(
                    kind="voice",
                    post=post.index,
                    usd=cost,
                    detail=f"{len(text)} chars, {settings.model_id}",
                )
            )
            files.save(files.ledger, ledger)
        beats, words, duration = time_script(script, alignment)
        timing = VoiceTiming(
            post=post.index,
            voice_id=settings.voice_id,
            model_id=settings.model_id,
            duration=duration,
            characters=len(text),
            beats=beats,
            words=words,
            cached=hit is not None,
        )
        pdir = files.post_dir(post.index)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "voice.mp3").write_bytes(audio)
        files.save(pdir / "voice.json", timing)
        out.append(timing)
    return out
