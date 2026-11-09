"""Real-model test: run with `pytest -m gpu` (downloads ~600 MB of CLIP weights once)."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def test_openclip_embeds_on_gpu_and_is_normalised():
    from pokepipe.cards.embeddings import OpenClipEmbedder

    emb = OpenClipEmbedder()
    assert emb.device == "cuda"
    a = Image.new("RGB", (245, 342), (220, 40, 40))
    ImageDraw.Draw(a).ellipse((60, 80, 180, 200), fill=(250, 220, 0))
    b = Image.new("RGB", (245, 342), (20, 60, 200))
    v = emb.embed_images([a, a.rotate(3), b])
    assert v.shape == (3, emb.dim)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)
    assert v[0] @ v[1] > v[0] @ v[2]  # slight rotation stays closer than a different image
