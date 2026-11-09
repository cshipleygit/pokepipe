from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence

import numpy as np
import pytest
from PIL import Image


def card_payload(set_id: str, number: str, name: str, rarity: str = "Common") -> dict:
    return {
        "id": f"{set_id}-{number}",
        "name": name,
        "number": number,
        "rarity": rarity,
        "supertype": "Pokémon",
        "subtypes": ["Basic"],
        "set": {"id": set_id, "name": "Test Set"},
        "images": {
            "small": f"https://images.example/{set_id}/{number}.png",
            "large": f"https://images.example/{set_id}/{number}_hires.png",
        },
    }


def png_bytes(color: tuple[int, int, int], size=(40, 56), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color if mode == "RGB" else (*color, 255)).save(buf, "PNG")
    return buf.getvalue()


class FakeEmbedder:
    """Deterministic embedder: a vector derived from the image's mean colour, so the
    same image always embeds identically and different colours land far apart."""

    model_tag = "fake__v1"
    dim = 16

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        rows = []
        for im in images:
            mean = np.asarray(im.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
            seed = int(hashlib.md5(mean.round().tobytes()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).standard_normal(self.dim))
        v = np.asarray(rows, dtype=np.float32).reshape(-1, self.dim)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
