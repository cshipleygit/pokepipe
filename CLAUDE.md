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
| 3 | `review/` | **built** | Small local web page to confirm/fix detected pulls before rendering. The only human-in-the-loop pause. |
| 4 | `collection/` | **built** | Local master-set progress per set, updated from confirmed pulls of each video. |
| 5 | `narration/` | **built** | LLM script from the user's style guidelines + ElevenLabs TTS voiceover. |
| 6 | `render/` | **built** | ffmpeg: one vertical post per pack group, narration-driven speed ramps, zoom on hits, optional reusable intro, word captions, master-set progress overlay. |
| 7 | `watch/` | todo | Watch folder: drop a video in → run stages 2→7, pausing only at review. |

Work proceeds **one stage at a time**; stop and show the user after each stage
before starting the next.

## Data flow

```
video.mp4 ─► detection ──pulls.json──► review (human) ──review.json──► collection
                ▲                                                       │
     cards index (stage 1)                                    collection.json
                                                                        ▼
                         narration: posts.json → posts/<n>/script.json, voice.mp3/.json
                                                                        │
                                                                        ▼
                                              render ─► posts/<n>/post.mp4 (1080×1920)
```

All per-video files live in `data/videos/<slug>/`.

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
- Calibration footage may be third-party: use it for development only, and never
  commit or publish renders made from footage the user doesn't own.

### Pack layouts → automatic reverse holos (`detection/layout.py`)

Unshuffled packs reveal cards in print order. Verified on all 12 packs of the test
video: slots 1–4 Common, 5–7 Uncommon, 8–9 reverse-holo slots (C/U/R; slot 9 also gave
an IR and an SIR), 10 the rare slot (R, or a UR). Basic Energy and the code card come
outside these 10 and are not pulls. CLIP can't see reverse holo (same card id, same
art), so a C/U/R in slot 8 or 9 is marked `reverse_guess`. Result: 22 reverse holos
inferred, which moved the test master set from 93 to 100/360.

- Conservative by design: only a pack whose rarities fit the layout gets guesses. A
  short pack aligns from the end (the rare is last, the reverse slots just before it).
  A shuffled or over-long pack gets no guesses and a `layout_note` shown in review.
- The layout is chosen by series (`layout_for`). Only Scarlet & Violet is defined and
  verified, on 151. Other sets in the series may differ (e.g. ACE SPEC or special slots),
  so verify a new set's packs before trusting it, and add layouts per era as needed.
- Review starts with the guess filled in ("reverse holo · auto"); the reviewer can flip
  it. `pokepipe detect layout <slug>` re-runs inference without re-scanning, and
  `pokepipe review prefill <slug>` applies guesses to a review already in progress
  (pending pulls only).
- A community guide says packs have 5 commons. On this footage it's 4; its 5th
  "common" is the basic Energy, which here comes after the rare. The same guide says
  IRs only appear in god packs, but Tangela IR appeared in slot 9 of a normal pack.

### Set code + collector number (OCR) — measured, not adopted yet

The bottom-left of every card shows regulation mark, set code + language (`MEW EN`),
`number/printed total` (`001/165`) and a rarity symbol. Experiment on the test video
(rapidocr, CPU, about 6 frames per reveal, majority vote; script not in repo):

| | CLIP (current) | corner OCR |
|---|---|---|
| correct card | 120/120 | 97/120 (8 misreads, 15 unread) |
| time | 89 s (GPU) | 456 s (CPU) |
| set code "MEW" read | n/a | 41/120 reveals |

At 1080p the text is about 20 px tall and often motion-blurred or under the opener's
thumb, so OCR is slower and less accurate than CLIP as the primary identifier. Planned
uses instead:
1. **Automatic set detection.** Vote on set code + printed total across a video's reads
   (41 good reads here). That removes `--set` and supports videos mixing sets.
2. **Tie-breaker** when CLIP's top two are close or the art is shared, e.g. the same
   art reprinted in another set or a promo. Only OCR the sharpest few frames of those
   reveals, on GPU, to keep the cost down.

### Stage 3 design

- `pokepipe review serve` runs FastAPI and uvicorn on **127.0.0.1 only**. The API can
  stream the source video and write review state, so it must never listen on the
  network. The UI is `review/static/` (vanilla HTML/JS/CSS, no build step) and is
  served with `Cache-Control: no-cache`.
- `pulls.json` is never modified. Human decisions live in `review.json`
  (`review/models.py:ReviewState`): each pull is pending / confirmed / rejected, may
  be corrected to another card (the detected card is kept), or flagged reverse holo.
  Reviewers can add cards detection missed. Detected pulls can only be rejected, not
  deleted, so `pulls.json` + `review.json` together form labelled data for
  recalibrating detection.
- All edit rules live in `review/store.py` (atomic writes, thread lock, card ids
  validated against the set, any edit reopens a finished review). `app.py` is a thin
  layer over it. A review is **complete** only when nothing is pending. Stages 4+ and
  the watch folder read `review.json` and wait for `completed: true`.
- UI: packs sidebar (red when a pack's count ≠ expected), video player that jumps to
  each card, video crop next to reference scan, hit badges, low-confidence "check"
  badges, and a card picker seeded with the detector's runner-ups. Keyboard shortcuts:
  j/k move, c/x/r/e act, space plays from the card.

### Stage 4 design

- **Master set is data-driven.** pokemontcg.io only exposes which printings exist
  through TCGplayer price keys (`normal`, `holofoil`, `reverseHolofoil`, ...).
  `Card.variants` stores them, mapped to `normal` / `holo` / `reverse` / 1st-edition.
  If a card has no price data, a rarity rule fills in: C/U → normal + reverse, Rare →
  holo + reverse, rarer → holo. 151 = 360 slots (207 card numbers, 165 in the base set).
- Progress is reported at three levels: master (every printing), full (every card
  number including secret rares) and base (numbers ≤ the printed total).
- `data/collection/<name>/<set_id>.json`. Ownership is a sum of **contributions**
  (one per applied video, plus `manual`), never a bare counter. Re-applying a video
  replaces its contribution, and `collection remove` undoes it exactly. Named
  collections keep trial runs (`--name test`) out of the real one (`main`).
- `collection apply <slug>` requires a finished review. It writes
  `data/videos/<slug>/collection.json` (`VideoReport`) with, per pull, the printing,
  `new` (first copy of that printing), `new_card` and `copy_number`, plus progress
  before and after the video and **after each pack** (the overlay number for that
  pack's post).
- Test video, all 120 detections confirmed and the 22 layout-inferred reverse holos
  kept: +100 new, master 100/360 (93/207 card numbers).

Known limitations:
- #207 in 151 is a gold Basic Psychic Energy (Hyper Rare). Detection drops all Energy
  because each pack's ordinary basic Energy matches #207 (the only Energy in the
  index). A real gold #207 therefore has to be added by hand in review. The proper
  fix is to index the ordinary basic Energies as decoys.
- "Before" progress counts every other applied video, not only earlier ones. Apply
  videos in the order they were opened.

### Stage 5 design

- **Posts.** The user decides which packs go into each TikTok post
  (`narrate plan <slug> --posts "1-3, 9, 10+12"`). With no spec, a pack containing a
  Double Rare or better gets its own post and the others are grouped in threes.
  Saved to `posts.json`.
- **Scripts:** Claude `claude-opus-5-5` via `client.beta.messages.parse(...,
  output_format=ScriptDraft)`, effort `medium`, server-side `fallbacks="default"` (beta
  `server-side-fallback-2026-07-01`), refusals and truncation handled.
  - The model sees only computed facts (pulls with `p<pack>-<n>` ids, new/duplicate
    flags, master-set progress before and after the post) plus `config/style.md`.
  - Every beat must be anchored to `intro`, `outro` or one of those pull ids, so it
    can't invent a card, and stage 6 can time each line to its reveal. Unknown anchors
    are repaired with a warning.
  - Length target: 12 s per pack (clamped 15–60 s) at 2.5 words/s.
- **Voice:** ElevenLabs `POST /v1/text-to-speech/{voice}/with-timestamps` (header
  `xi-api-key`), one request per post with the beats joined. The character alignment
  is mapped back to per-beat and per-word timings (`voice.json`) for syncing and
  captions in stage 6. Audio is cached in `data/tts_cache/` by hash of
  text+voice+model+format, so re-runs are free.
- **Budget:** every paid call is logged to `costs.json`. The check runs *before* each
  call and refuses to push the video over `BUDGET_USD_PER_VIDEO` (default $1) unless
  `--force` is passed. Claude cost comes from actual usage at $4/$20 per MTok. Voice
  cost is an estimate from `ELEVENLABS_USD_PER_1K_CHARS`; set it from your plan.
- Keys live in `.env` (gitignored): `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY`,
  `ELEVENLABS_VOICE_ID`. Tests mock both APIs, so no key is needed for `pytest`.

### Stage 6 design (render/)

- `timeline.py` is pure edit logic. Each pack's clip window becomes gap, reveal and
  hit shots.
  - **Hits** are the rare slot and better, excluding reverse-holo Rares. They play at
    1× for 2.5–4 s with a punch-in zoom; the rest of a long hold speeds up.
  - **Framing:** shots are vertical 9:16 crops centred on the card.
- **Narration drives the pacing.**
  - Regions are cut at each card a line is about. A region must last as long as its
    narration, quiet regions run up to 4× (gaps 2.5× faster), and spare time up to
    the post's target is shared out.
  - If a region can't fit its lines even at 1×, its hit freezes. The final shot may
    run into real footage after the pack (`tail_room`) before freezing.
  - Each voice line is cut from `voice.mp3` and starts when its card appears, and
    drift is reported.
- **Scripts are written to fit:** before Claude writes, `render/inputs.airtime()`
  drafts the edit and gives each card's seconds on screen plus the seconds after the
  last card. This removed the end-of-video freeze (first render 58 s with a 7 s
  freeze, now 45.2 s with every line in sync).
- `captions.py` writes an ASS file burned in by libass: word-by-word captions with
  the current word highlighted, the hook caption, a master-set counter that ticks up
  1.2 s after each pack's last card plus a "+N NEW" badge, and rarity labels during
  big-hit zooms. Everything stays inside TikTok's safe area.
- `ffmpeg.py` renders in two passes. First, one clip per shot (NVENC when available,
  else x264). Then concat, burn in the ASS file, and mix the voice lines
  (atrim + adelay + amix). An optional `config/intro.mp4` is prepended.
  `edit.json` records every decision.
- Result on packs 1–3 of the test video: 45.2 s, 41 shots, rendered in 39 s on the
  RTX 3060. A clean run of one 45 s post costs about $0.20 (script + voice).
- To tune next: bigger counter/badge (the badge overlaps the card name), optional
  source audio (pack-rip sounds), and motion blur on commons at ~3.6×.

## Commands

```bash
.venv\Scripts\activate
pokepipe cards sets --search "151"        # find a set id
pokepipe cards fetch sv3pt5               # checklist + images → data/sets/sv3pt5
pokepipe cards embed sv3pt5               # build GPU embeddings index
pokepipe cards match sv3pt5 photo.jpg     # sanity-check: top-5 matches for an image
pokepipe detect run video.mp4 --set sv3pt5 # → data/videos/<slug>/pulls.json + thumbs/
pokepipe detect layout <slug>             # re-run pack-slot / reverse-holo inference
pokepipe review serve                     # review page at http://127.0.0.1:8765
pokepipe review prefill <slug>            # copy reverse guesses onto pending pulls
pokepipe review status                    # progress per video
pokepipe collection apply <slug>          # add a finished review → collection.json
pokepipe collection status sv3pt5         # master-set progress by rarity
pokepipe collection missing sv3pt5 --variant reverse
pokepipe collection add sv3pt5 sv3pt5-25 --variant reverse   # cards owned already
pokepipe narrate plan <slug> --posts "1-3, 9, 10-12"  # packs per post (omit = suggest)
pokepipe narrate script <slug> [-p 2]     # Claude writes each post's voiceover
pokepipe narrate voices                   # list ElevenLabs voices → ELEVENLABS_VOICE_ID
pokepipe narrate voice <slug> [-p 2]      # synthesise (cached, budget-checked)
pokepipe narrate cost <slug>              # spend so far vs the per-video budget
pokepipe render post <slug> [-p 1]        # → posts/<n>/post.mp4 (1080x1920)
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
- Stage 5: narration uses `claude-opus-5-5` (≈ $0.04 per post). Revisit only if
  cost matters, by measuring script quality against a cheaper model.
- Creator profiles: per-creator style guide, voice, intro and collection, kept out
  of the public repo (profile *system* in code, profile *contents* gitignored).
