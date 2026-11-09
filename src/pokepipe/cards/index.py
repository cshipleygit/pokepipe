"""CardIndex: a tiny, exact nearest-neighbour index over card embeddings.

A set has at most a few hundred cards, so brute-force cosine similarity on the GPU
is both exact and sub-millisecond — no need for FAISS or an ANN library.

The index may hold several vectors per card (e.g. augmented views added in stage 2);
search scores each card by its best-matching vector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Match:
    card_id: str
    score: float  # cosine similarity in [-1, 1]


class CardIndex:
    def __init__(
        self,
        card_ids: list[str],
        vectors: np.ndarray,
        model_tag: str,
        device: str | None = None,
    ) -> None:
        if len(card_ids) != len(vectors):
            raise ValueError(f"{len(card_ids)} ids but {len(vectors)} vectors")
        self.card_ids = list(card_ids)
        self.model_tag = model_tag
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.unique_ids = list(dict.fromkeys(self.card_ids))  # stable order
        position = {cid: i for i, cid in enumerate(self.unique_ids)}
        self._owner = torch.tensor([position[c] for c in self.card_ids], device=self.device)
        self._vectors = torch.as_tensor(np.asarray(vectors, dtype=np.float32), device=self.device)

    def __len__(self) -> int:
        return len(self.unique_ids)

    @property
    def dim(self) -> int:
        return int(self._vectors.shape[1])

    def card_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """(Q, D) normalised queries → (Q, n_cards) best cosine score per card, on device.

        Columns follow `self.unique_ids`. Used directly by detection to stay on the GPU.
        """
        q = queries.to(self.device, torch.float32)
        if q.shape[1] != self.dim:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.dim}")
        sims = q @ self._vectors.T  # (Q, N)
        per_card = torch.full((q.shape[0], len(self)), -torch.inf, device=self.device)
        per_card.scatter_reduce_(1, self._owner.expand_as(sims), sims, reduce="amax")
        return per_card

    def top_k(self, per_card: torch.Tensor, k: int = 5) -> list[list[Match]]:
        scores, idx = per_card.topk(min(k, len(self)), dim=1)
        return [
            [Match(self.unique_ids[i], float(s)) for s, i in zip(row_s, row_i, strict=True)]
            for row_s, row_i in zip(scores.tolist(), idx.tolist(), strict=True)
        ]

    def search(self, queries: np.ndarray, top_k: int = 5) -> list[list[Match]]:
        """Top-k cards for each query vector (queries must be L2-normalised)."""
        q = torch.as_tensor(np.atleast_2d(queries).astype(np.float32))
        return self.top_k(self.card_scores(q), top_k)

    # --- persistence -------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        vectors = self._vectors.cpu().numpy().astype(np.float16)
        np.savez_compressed(path, vectors=vectors, card_ids=np.array(self.card_ids))
        manifest = {
            "model_tag": self.model_tag,
            "dim": self.dim,
            "n_vectors": len(self.card_ids),
            "n_cards": len(self),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> CardIndex:
        with np.load(path, allow_pickle=False) as data:
            vectors = data["vectors"].astype(np.float32)
            card_ids = [str(c) for c in data["card_ids"]]
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(card_ids, vectors, manifest["model_tag"], device=device)
