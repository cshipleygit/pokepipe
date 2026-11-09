# pokepipe

Local Python pipeline that turns raw Pokémon card-opening videos into edited
vertical TikTok shorts — card recognition on a local GPU, a human review step,
master-set tracking, LLM narration + TTS, and ffmpeg rendering.

See [CLAUDE.md](CLAUDE.md) for the full architecture and stage plan.

## Status

| Stage | Status |
|-------|--------|
| 1. Card data + CLIP embeddings | ✅ |
| 2. Pull detection | ⏳ |
| 3. Review UI | ⏳ |
| 4. Collection tracking | ⏳ |
| 5. Script + voice | ⏳ |
| 6. Render | ⏳ |
| 7. Watch folder | ⏳ |

## Setup (Windows, NVIDIA GPU)

Requires Python 3.12, ffmpeg on PATH, and an NVIDIA driver.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
copy .env.example .env   # optional API keys
```

## Stage 1: card data

```bash
pokepipe cards sets --search "151"     # find a set id
pokepipe cards fetch sv3pt5            # checklist + hi-res images → data/sets/sv3pt5
pokepipe cards embed sv3pt5            # CLIP (ViT-B/32) embeddings on the GPU
pokepipe cards match sv3pt5 card.jpg   # top-5 matches for a photo
```

Design notes:

- **Any set:** everything is keyed by the pokemontcg.io set id; the full checklist
  (including secret rares) is paginated and sorted by collector number.
- **Resilient fetching:** the free API returns intermittent 5xx errors, so requests
  retry with exponential backoff; image downloads are concurrent, idempotent and
  atomic (temp file + rename).
- **Swappable model:** the index only depends on an `Embedder` protocol; tests use a
  deterministic fake, production uses OpenCLIP in fp16 on CUDA.
- **Exact search:** a set is a few hundred cards, so brute-force cosine similarity on
  the GPU is exact and sub-millisecond. The index supports several vectors per card
  so later stages can add augmented views.

## Tests

```bash
pytest          # fast unit tests — no network, no GPU
pytest -m gpu   # real CLIP model on CUDA
ruff check src tests
```
