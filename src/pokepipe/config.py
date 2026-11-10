"""Runtime settings, loaded from environment variables or a local `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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
    budget_usd_per_video: float = 1.0
    style_guide: Path = PROJECT_ROOT / "config" / "style.md"
    anthropic_api_key: str | None = None
    # Needed only for API keys not scoped to a workspace (sent as anthropic-workspace-id).
    anthropic_workspace_id: str | None = None
    script_model: str = "claude-opus-5-5"
    script_effort: str = "medium"
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

    @property
    def sets_dir(self) -> Path:
        return self.data_dir / "sets"

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def inbox_dir(self) -> Path:
        return self.watch_inbox or self.data_dir / "inbox"


@lru_cache
def get_settings() -> Settings:
    return Settings()
