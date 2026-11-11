"""The real `Ops`: thin adapters over each stage's service.

Every method checks for its outputs first and only does missing work, which is what
makes the runner's steps safe to repeat after a crash, a restart or a retry.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pokepipe.config import PROJECT_ROOT, Settings
from pokepipe.narration.service import Budget, VideoFiles
from pokepipe.watch.models import Job


class PipelineOps:
    def __init__(self, settings: Settings, log: Callable[[str], None] = print) -> None:
        self.s = settings
        self.log = log
        self._embedder = None  # CLIP stays loaded between videos

    # --- helpers --------------------------------------------------------------------------

    def _files(self, job: Job) -> VideoFiles:
        return VideoFiles(self.s.videos_dir / job.slug)

    def _budget(self, job: Job) -> Budget:
        return Budget(usd=self.s.budget_usd_per_video, force=job.force_budget)

    def embedder(self):
        if self._embedder is None:
            from pokepipe.cards.embeddings import OpenClipEmbedder

            self._embedder = OpenClipEmbedder(
                self.s.clip_model, self.s.clip_pretrained, fit=self.s.clip_fit
            )
        return self._embedder

    def _tracker(self, job: Job):
        """Zoom tracking reuses the CLIP model already loaded for detection."""
        from pokepipe.cards import service as cards
        from pokepipe.render.track import CardTracker

        emb = self.embedder()
        return CardTracker(emb, cards.load_index(self.s.sets_dir, job.set_id, emb.model_tag))

    # --- stages ---------------------------------------------------------------------------

    def ensure_set(self, job: Job) -> None:
        from pokepipe.cards import service as cards
        from pokepipe.cards.api import PokemonTCGClient
        from pokepipe.cards.cache import SetCache

        if not SetCache(self.s.sets_dir, job.set_id).exists():
            self.log(f"fetching card data for {job.set_id}")
            client = PokemonTCGClient(self.s.pokemontcg_base_url, api_key=self.s.pokemontcg_api_key)
            result = cards.fetch_set(
                client, self.s.sets_dir, job.set_id, workers=self.s.download_workers
            )
            if result.images.failed:
                raise RuntimeError(f"{len(result.images.failed)} card images failed to download")
        emb = self.embedder()
        index_path = SetCache(self.s.sets_dir, job.set_id).index_path(emb.model_tag)
        stale = index_path.exists() and not cards.index_has_decoys(
            cards.load_index(self.s.sets_dir, job.set_id, emb.model_tag)
        )
        if not index_path.exists() or stale:
            self.log(f"building the {job.set_id} card index")
            client = PokemonTCGClient(self.s.pokemontcg_base_url,
                                      api_key=self.s.pokemontcg_api_key)
            decoys = cards.ensure_decoys(client, self.s.sets_dir, self.s.download_workers)
            cards.build_index(self.s.sets_dir, job.set_id, emb, decoys=decoys)

    def detect(self, job: Job) -> None:
        from pokepipe.cards import service as cards
        from pokepipe.detection import service as detection

        files = self._files(job)
        if (files.root / "pulls.json").exists():
            return
        emb = self.embedder()
        index = cards.load_index(self.s.sets_dir, job.set_id, emb.model_tag)
        result = detection.detect_pulls(
            Path(job.source), job.set_id, self.s.sets_dir, index, emb, files.root
        )
        self.log(f"{job.slug}: {len(result.pulls)} pulls in {len(result.packs)} packs")

    def review_done(self, job: Job) -> bool:
        from pokepipe.review.store import ReviewStore

        path = self._files(job).root / "review.json"
        if not path.exists():
            return False
        return ReviewStore(self.s.videos_dir, self.s.sets_dir).load(job.slug).completed

    def apply_collection(self, job: Job) -> None:
        from pokepipe.collection import service as collection
        from pokepipe.review.store import ReviewStore

        report = collection.apply_video(
            collection.CollectionStore(self.s.data_dir / "collection"),
            ReviewStore(self.s.videos_dir, self.s.sets_dir),
            self.s.sets_dir,
            job.slug,
            self.s.watch_collection,
        )
        self.log(
            f"{job.slug}: +{report.new_count} new for the "
            f"{report.set_name} master set ({report.after.master_owned}/"
            f"{report.after.master_total})"
        )

    def narrate(self, job: Job) -> None:
        from pokepipe.narration import service as narration
        from pokepipe.narration.script import ScriptSettings, anthropic_client
        from pokepipe.narration.voice import ElevenLabsClient, VoiceCache, VoiceSettings

        files = self._files(job)
        if not files.plan.exists():  # not chosen in review: use the suggested grouping
            narration.plan_posts(files, job.slug, None, self.s.default_post_seconds)
        if not self.s.narration:  # edit-only posts: no script, no voice, no API spend
            self.log(f"{job.slug}: narration off — skipping scripts and voice")
            return
        plan = files.load_plan()

        need_script = [
            p.index for p in plan.posts if not (files.post_dir(p.index) / "script.json").exists()
        ]
        if need_script:
            style = (
                self.s.style_guide.read_text(encoding="utf-8")
                if self.s.style_guide.exists()
                else ""
            )
            narration.write_scripts(
                files,
                anthropic_client(self.s.anthropic_api_key, self.s.anthropic_workspace_id),
                style,
                ScriptSettings(model=self.s.script_model, effort=self.s.script_effort),
                self._budget(job),
                need_script,
            )

        need_voice = [
            p.index for p in plan.posts if not (files.post_dir(p.index) / "voice.json").exists()
        ]
        if need_voice:
            if not self.s.elevenlabs_voice_id:
                raise RuntimeError("ELEVENLABS_VOICE_ID is not set")
            narration.voice_posts(
                files,
                ElevenLabsClient(self.s.elevenlabs_api_key or ""),
                VoiceCache(self.s.data_dir / "tts_cache"),
                VoiceSettings(
                    voice_id=self.s.elevenlabs_voice_id,
                    model_id=self.s.elevenlabs_model_id,
                    usd_per_1k_chars=self.s.elevenlabs_usd_per_1k_chars,
                ),
                self._budget(job),
                need_voice,
            )
        self.log(f"{job.slug}: narration ready (${files.load_ledger().total:.2f} so far)")

    def render(self, job: Job) -> None:
        from pokepipe.render.service import render_post

        files = self._files(job)
        intro = PROJECT_ROOT / "config" / "intro.mp4"
        for post in files.load_plan().posts:
            if (files.post_dir(post.index) / "post.mp4").exists():
                continue
            result = render_post(files, post.index, intro=intro if intro.exists() else None,
                                 narration=self.s.narration, track=self.s.render_track,
                                 tracker=self._tracker(job) if self.s.render_track else None)
            self.log(f"{job.slug}: post {post.index} → {result.path} ({result.edit.duration:.0f}s)")
