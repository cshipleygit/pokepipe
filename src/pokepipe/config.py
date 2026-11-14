"""Runtime settings, loaded from environment variables or a local `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"

    # Stage 1 — card data
    pokemontcg_api_key: str | None = None
    pokemontcg_base_url: str = "https://api.pokemontcg.io/v2"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    clip_fit: str = "full"  # "full" card squashed to square, or stock "center" crop
    download_workers: int = 8

    # Stage 5 — script + voice
    narration: bool = True  # False: posts are the edit alone, with the source audio
    render_track: bool = True  # follow the card through zooms (loads CLIP on the GPU)
    series_edit: bool = True  # watcher finds box/tear/sleeve moments → the series edit
    budget_usd_per_video: float = 1.0
    style_guide: Path = PROJECT_ROOT / "config" / "style.md"
    anthropic_api_key: str | None = None
    # Needed only for API keys not scoped to a workspace (sent as anthropic-workspace-id).
    anthropic_workspace_id: str | None = None
    script_model: str = "claude-opus-5-5"
    script_effort: str = "medium"
    # Writer ⇄ critic loop (narration/writer.py): checks every draft, rewrites on problems
    script_critic: bool = True
    script_rewrites: int = 2
    script_style_judge: bool = True  # a cheap model check of the draft against style.md
    script_judge_effort: str = "low"
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    # Used only for the budget check — set it from your ElevenLabs plan's per-character rate.
    elevenlabs_usd_per_1k_chars: float = 0.30

    # Stage 7 — watch folder
    watch_inbox: Path | None = None  # default: <data_dir>/inbox
    watch_default_set: str | None = None  # for videos dropped straight into the inbox
    watch_collection: str = "main"  # collection that finished reviews are applied to
    watch_poll_seconds: float = 5.0
    watch_settle_seconds: float = 10.0  # a file must stop changing this long before intake
    default_post_seconds: float = 45.0
    notify: bool = True  # Windows toast: review waiting / posts ready / failed

    # Auto-review (review/auto.py): always | shadow | exceptions | off
    review_mode: Literal["always", "shadow", "exceptions", "off"] = "shadow"
    auto_min_confidence: float = 0.6
    auto_min_lead: float = 0.02
    auto_accept_suggestions: bool = True  # the recovery agent's high-confidence ones
    auto_require_layout: bool = True
    auto_use_calibration: bool = True  # use data/review_calibration.json once fitted

    # Missed-card recovery agent (runs in the graph engine, before review)
    recover: bool = True
    recover_model: str = "claude-opus-5-5"
    recover_effort: str = "medium"
    recover_max_turns: int = 14

    @property
    def sets_dir(self) -> Path:
        return self.data_dir / "sets"

    @property
    def prices_dir(self) -> Path:
        return self.data_dir / "prices"

    @property
    def brand_fonts_dir(self) -> Path:
        """A private theme's font files (gitignored; see render/theme.py)."""
        return self.data_dir / "brand" / "fonts"

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def inbox_dir(self) -> Path:
        return self.watch_inbox or self.data_dir / "inbox"

    @property
    def calibration_path(self) -> Path:
        return self.data_dir / "review_calibration.json"

    def auto_policy(self):
        from pokepipe.review.auto import AutoPolicy
        from pokepipe.review.calibrate import Calibration

        cal = Calibration.load(self.calibration_path) if self.auto_use_calibration else None
        return AutoPolicy(min_confidence=self.auto_min_confidence, min_lead=self.auto_min_lead,
                          accept_suggestions=self.auto_accept_suggestions,
                          require_layout=self.auto_require_layout, calibration=cal)

    @property
    def graph_checkpoints(self) -> Path:
        return self.data_dir / "graph" / "checkpoints.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()
