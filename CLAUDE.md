# CLAUDE.md — Pokémon Pack-Opening → TikTok Pipeline

A local Python pipeline that turns raw Pokémon card-opening videos into edited
vertical TikTok shorts. It is also a **portfolio project for AI Engineer roles**:
code must be modular, typed, tested and documented. Optimise for readability by a
reviewer as much as for function.

## Constraints

- **Machine:** Windows 11, NVIDIA RTX 3060 (12 GB VRAM), driver supports CUDA 13.x.
- **Runtime:** Python 3.12 in `.venv/`, PyTorch CUDA wheels (cu128 index), ffmpeg 9 on PATH.
- **Budget:** under ~$1 of API spend per video (LLM narration + ElevenLabs TTS).
  Vision/matching runs locally on the GPU — never send video frames to a paid API.
- **Local-first:** all caches, collection state and outputs live under `data/` (gitignored).

## Stages (each its own subpackage in `src/pokepipe/`)

| # | Package | Status | Purpose |
|---|---------|--------|---------|
| 1 | `cards/` | **built** | Fetch any set's checklist + images from pokemontcg.io, cache locally, build CLIP image embeddings, GPU nearest-neighbour index. |
| 2 | `detection/` | todo | Sample frames, detect card-reveal moments, crop the card, match against the stage-1 index → pulls with confidence + timestamps. |
| 3 | `review/` | todo | Small local web page to confirm/fix detected pulls before rendering. The only human-in-the-loop pause. |
| 4 | `collection/` | todo | Local master-set progress per set, updated from confirmed pulls of each video. |
| 5 | `narration/` | todo | LLM script from the user's style guidelines + ElevenLabs TTS voiceover. |
| 6 | `render/` | todo | ffmpeg: 4–6 vertical clips, speed ramps, zoom on hits, reusable intro, captions, master-set progress overlay. |
| 7 | `watch/` | todo | Watch folder: drop a video in → run stages 2→7, pausing only at review. |

Work proceeds **one stage at a time**; stop and show the user after each stage
before starting the next.

## Data flow

```
video.mp4 ─► detection ──pulls.json──► review (human) ──confirmed.json──► collection
                ▲                                              │
     cards index (stage 1)                                     ▼
                                                 narration (script.txt, voice.mp3)
                                                               │
                                                               ▼
                                                    render ─► short_01..06.mp4
```

Stages communicate through **JSON files on disk** validated by Pydantic models,
so any stage can be re-run in isolation and inspected by hand.

## Repo layout

```
src/pokepipe/
  config.py          # Settings (env vars / .env), data paths
  cli.py             # Typer root CLI: `pokepipe <stage> <command>`
  cards/             # Stage 1
    models.py        #   Card / CardSet Pydantic models
    api.py           #   pokemontcg.io v2 client (pagination, retries, optional API key)
    cache.py         #   On-disk set cache (JSON + images)
    images.py        #   Concurrent image downloader (idempotent)
    embeddings.py    #   Embedder protocol + OpenCLIP implementation
    index.py         #   CardIndex: save/load .npz, GPU cosine top-k search
    service.py       #   High-level ops used by CLI and later stages
    cli.py           #   `pokepipe cards ...`
  detection/ review/ collection/ narration/ render/ watch/   # later stages
tests/               # pytest; network mocked with `responses`; GPU tests marked `gpu`
data/                # gitignored runtime data
```

### Stage 1 on-disk cache

```
data/sets/<set_id>/
  set.json                      # CardSet metadata
  cards.json                    # list[Card], sorted by collector number
  images/<card_id>.png          # large (hi-res) card scans
  index/<model_tag>.npz         # embeddings (float16, L2-normalised) + card_ids
  index/<model_tag>.json        # index manifest (model, dims, created_at, n_vectors)
```

The index supports **multiple vectors per card** (`card_ids` may repeat) so stage 2
can add augmented views (crops, glare, perspective) without changing the format;
search aggregates by max score per card.

## Commands

```bash
.venv\Scripts\activate
pokepipe cards sets --search "151"        # find a set id
pokepipe cards fetch sv3pt5               # checklist + images → data/sets/sv3pt5
pokepipe cards embed sv3pt5               # build GPU embeddings index
pokepipe cards match sv3pt5 photo.jpg     # sanity-check: top-5 matches for an image
pytest                                    # unit tests (no network, no GPU)
pytest -m gpu                             # GPU/model tests
ruff check src tests
```

## Conventions

- Type hints everywhere; Pydantic v2 for anything serialised to disk.
- Pure logic separated from I/O so it is unit-testable; network calls go through
  one client class; the embedding model sits behind the `Embedder` protocol so
  tests use a deterministic fake.
- No network or GPU in default `pytest`. Mark real-model tests `@pytest.mark.gpu`.
- Secrets (`POKEMONTCG_API_KEY`, `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY`) come
  from env / `.env`, never committed.
- pokemontcg.io is free but flaky (intermittent 502s) — always retry with backoff.
  An API key is optional and only raises rate limits.
- Prefer small, well-named modules over clever code; docstrings explain *why*.

## Open decisions (revisit at the relevant stage)

- Stage 2: CLIP zero-shot on raw scans vs. adding augmented reference views vs.
  a light fine-tune — measure on real footage first.
- Stage 5: model choice for narration (cheap Claude model is likely enough) and
  ElevenLabs character budget per video to stay under $1.
