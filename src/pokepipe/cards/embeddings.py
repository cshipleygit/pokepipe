"""Image embedders.

Everything downstream depends only on the `Embedder` protocol, so the model can be
swapped (or faked in tests) without touching the index or detection code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image


@runtime_checkable
class Embedder(Protocol):
    """Maps images to L2-normalised float32 vectors of shape (N, dim)."""

    model_tag: str
    dim: int

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def to_rgb(img: Image.Image) -> Image.Image:
    """Card scans are RGBA PNGs with transparent rounded corners; flatten onto white
    rather than letting PIL drop alpha to black, which skews the embedding."""
    if img.mode == "RGB":
        return img
    rgba = img.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


CARD_ASPECT = 63 / 88  # width / height of a standard trading card


class OpenClipEmbedder:
    """CLIP image encoder via `open_clip`, run on CUDA in fp16 when available.

    `fit` controls how a portrait card becomes CLIP's square input:
      - "full" (default): squash the whole card to a square. Keeps the name bar, HP
        and attack text, which separate look-alike cards (e.g. Squirtle vs Pikachu).
      - "center": OpenCLIP's stock centre-crop, which keeps only the middle square.
    Detection crops frames with the same geometry (`crop_aspect`), so reference and
    query images are distorted identically.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        batch_size: int = 64,
        fit: str = "full",
    ) -> None:
        if fit not in ("full", "center"):
            raise ValueError(f"fit must be 'full' or 'center', not {fit!r}")
        import open_clip  # heavy imports kept local so the package imports fast
        import torch

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.fit = fit
        self.crop_aspect = CARD_ASPECT if fit == "full" else 1.0
        self.model_tag = f"{model_name}__{pretrained}" + ("__full" if fit == "full" else "")
        self.dim = int(self.model.visual.output_dim)

        visual = self.model.visual
        size = getattr(visual, "image_size", 224)
        self.image_size = int(size[0] if isinstance(size, tuple | list) else size)
        mean = getattr(visual, "image_mean", None) or open_clip.OPENAI_DATASET_MEAN
        std = getattr(visual, "image_std", None) or open_clip.OPENAI_DATASET_STD
        self._mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

    def embed_pixels(self, pixels):
        """GPU fast path: (N, 3, S, S) float tensor in [0, 1], already cropped/resized
        to `image_size` → (N, D) L2-normalised float32 tensor on the same device.

        Detection uses this to crop frames with `roi_align` on the GPU and skip the
        PIL round-trip of `embed_images`.
        """
        torch = self._torch
        x = (pixels.to(self.device) - self._mean) / self._std
        with torch.no_grad(), torch.autocast("cuda", enabled=self.device.startswith("cuda")):
            feats = self.model.encode_image(x).float()
        return torch.nn.functional.normalize(feats, dim=-1)

    def _to_pixels(self, im: Image.Image):
        torch = self._torch
        if self.fit == "center":
            x = self.preprocess(to_rgb(im))  # resize + centre-crop + normalise
            return x * self._std[0].cpu() + self._mean[0].cpu()  # undo normalise
        s = self.image_size
        arr = np.asarray(to_rgb(im).resize((s, s), Image.Resampling.BICUBIC), np.float32)
        return torch.from_numpy(arr / 255.0).permute(2, 0, 1)

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            pixels = torch.stack([self._to_pixels(im) for im in batch])
            out.append(self.embed_pixels(pixels).cpu().numpy())
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.concatenate(out)).astype(np.float32)
