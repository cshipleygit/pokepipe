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
| 2 | `detection/` | **built** | Sample frames, detect card-reveal moments, crop the card, match against the stage-1 index → pulls with confidence + timestamps. |
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

### Stage 2 design and calibration

- `frames.py` decodes via an ffmpeg rawvideo pipe at 4 fps, longest side 960 px.
- **Whole-card embeddings (`clip_fit="full"`, the default).** Reference scans are
  squashed to CLIP's square input rather than OpenCLIP's stock centre-crop, and
  `matcher.py` crops frames with card-aspect boxes squashed the same way. The stock
  centre-crop dropped the name bar and attack text, which caused look-alike misses
  (Squirtle vs Pikachu) and weak scores on trainers. Synthetic benchmark: top-1 87.9% →
  92.8%, top-5 93.7% → 99.5%. Indexes are tagged `__full` so both modes can coexist.
- `matcher.py` scores a grid of card-aspect crops per frame in one batched
  `torchvision.ops.roi_align` call, so frames stay on the GPU.
- `segment.py` turns per-frame votes into reveals (pure functions, unit-tested).
  Defaults were calibrated on a third-party 10.6 min 1080p60 opening of 12 packs of 151
  (`--dump-frames` writes the per-frame data used). Genuine reveals peak at 0.80–0.89
  cosine. Sealed product, hands, card backs and code cards stay below about 0.71.
  Hence: frame vote ≥ 0.68 with ≥ 0.03 lead over the runner-up, segment peak ≥ 0.72,
  ≥ 3 frames, 1 s gap bridging. Basic Energy is matched but excluded from pulls.
- `packs.py` groups pulls into packs on gaps > 8 s, the unit we split posts by. Each
  pack gets a clip window: up to 15 s lead-in for the tear, 2 s tail, never overlapping.
  A pack whose count ≠ `cards_per_pack` (default 10) is flagged for review.
- **Ground-truth result (12 packs = 120 cards): 120/120 found, 12 packs × 10, in 89 s on
  the RTX 3060.** The first version (centre-crop) found 114. All 114 were correct, and
  the 6 misses were 2 trainers, Snorlax, Squirtle ×2 and Caterpie.
- Output: `data/videos/<slug>/pulls.json` (`detection/models.py:PullsFile`: packs +
  pulls) plus `thumbs/pNN_NNN_<card_id>.jpg` full-res crops for the review UI.

Known limitations, to handle in later stages:
- One video is not a benchmark. Validate on the user's own footage (different
  lighting, framing and camera) before trusting the thresholds, and keep a labelled
  pack list per test video so recall stays measurable.
- Detection confidence is lowest on full-art hits (Charizard ex SIR: 0.62) because foil
  texture differs from the scan. Review should show hits first.
- **Reverse holos** match the same card id as the regular print, so detection cannot
  tell them apart. That matters for master-set tracking (stage 4); the review UI should
  let the user mark a pull as reverse holo.
- The same card shown twice (e.g. re-shown at the end) appears as two pulls. Review
  decides whether it is a duplicate copy.
- `confidence` is a hand-weighted heuristic. Fit it on confirmed labels from review.
- The test video is someone else's YouTube upload: use it for development only and
  never in rendered output.

## Commands

```bash
.venv\Scripts\activate
pokepipe cards sets --search "151"        # find a set id
pokepipe cards fetch sv3pt5               # checklist + images → data/sets/sv3pt5
pokepipe cards embed sv3pt5               # build GPU embeddings index
pokepipe cards match sv3pt5 photo.jpg     # sanity-check: top-5 matches for an image
pokepipe detect run video.mp4 --set sv3pt5 # → data/videos/<slug>/pulls.json + thumbs/
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
