# pokepipe

Local Python pipeline that turns raw Pokémon card-opening videos into edited
vertical TikTok shorts — card recognition on a local GPU, a human review step,
master-set tracking, LLM narration + TTS, and ffmpeg rendering.

See [CLAUDE.md](CLAUDE.md) for the full architecture and stage plan.

## Status

| Stage | Status |
|-------|--------|
| 1. Card data + CLIP embeddings | ✅ |
| 2. Pull detection | ✅ |
| 3. Review UI | ✅ |
| 4. Collection tracking | ✅ |
| 5. Script + voice | ✅ |
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

## Stage 2: pull detection

```bash
pokepipe detect run opening.mp4 --set sv3pt5
```

Frames are sampled with ffmpeg at 4 fps. Each frame is scored as a grid of
card-shaped crops in a single batched GPU pass, against embeddings of the whole card
(not CLIP's default centre crop, which loses the name bar). Per-frame votes are
grouped into reveals using a calibrated score, a minimum lead over the runner-up
card, and duration thresholds. Reveals are then grouped into **packs**, each with a
clip window, so one opening can be split into several posts.

**Reverse holos come from pack order.** Image matching can't tell a reverse holo from
the regular print, but unshuffled packs follow a fixed order: 4 commons, 3 uncommons,
2 reverse-holo slots, then the rare. A Common, Uncommon or Rare in slot 8 or 9 is
therefore a reverse holo. This was verified on all 12 test packs and inferred 22
reverse holos automatically. A pack that doesn't fit the order gets no guesses and is
flagged for review.

On a 10.6-minute 1080p60 opening of 12 packs of 151 (120 cards), it found
**120/120 cards, 10 per pack, in 89 s** on an RTX 3060. Switching from centre-crop
to whole-card matching recovered the 6 cards the first version missed.

## Stage 3: review

```bash
pokepipe review serve      # opens http://127.0.0.1:8765
```

A local page for checking detections pack by pack. Each card's crop from the video
sits next to its reference scan, and the player jumps to the moment it was shown.
Keep, reject, change the card, mark reverse holo, or add a card detection missed.
Keyboard shortcuts make a 120-card review quick. Decisions are saved to
`review.json` beside the untouched `pulls.json`, so the two files together form
labelled data for recalibrating detection. The server binds to localhost only.

## Stage 4: collection tracking

```bash
pokepipe collection apply <video-slug>    # after the review is finished
pokepipe collection status sv3pt5
pokepipe collection missing sv3pt5 --variant reverse
```

Tracks master-set progress for any set. Which printings each card comes in (normal,
holo, reverse holo) comes from the API data rather than hard-coded rules. Every
applied video is stored as a separate contribution, so re-applying a re-reviewed
video never double-counts and any video can be removed. Each video gets a
`collection.json` marking which pulls are new and the progress after each pack,
which is the number shown on that pack's post. Trial runs go in a separate named
collection (`--name test`).

## Stage 5: script and voice

```bash
pokepipe narrate plan <slug> --posts "1-3, 9, 10-12"   # which packs go in each post
pokepipe narrate script <slug>                          # Claude writes the voiceover
pokepipe narrate voice <slug>                           # ElevenLabs reads it
```

- **Choose the posts.** You decide which packs go into each TikTok post.
- **Grounded scripts.** Claude (`claude-opus-5-5`, structured output) writes each
  post's script from the facts only: the pulls, which ones are new, and master-set
  progress. It follows your style guide in `config/style.md`. Every line points at a
  specific pull, so it can't invent a card and the line can be timed to that reveal.
- **Voice with timing.** ElevenLabs returns timings for every character, which are
  turned into timings per line and per word for captions.
- **Cost control.** Audio is cached, so re-runs are free. Every paid call is logged,
  and a check before each call keeps a video under the $1 budget.

## Tests

```bash
pytest          # fast unit tests — no network, no GPU
pytest -m gpu   # real CLIP model on CUDA
ruff check src tests
```
