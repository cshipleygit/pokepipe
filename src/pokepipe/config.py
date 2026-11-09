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
    download_workers: int = 8

    # Later stages
    anthropic_api_key: str | None = None
    elevenlabs_api_key: str | None = None

    @property
    def sets_dir(self) -> Path:
        return self.data_dir / "sets"


@lru_cache
def get_settings() -> Settings:
    return Settings()
