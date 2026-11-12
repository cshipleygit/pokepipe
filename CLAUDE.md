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
| 7 | `watch/` | **built** | Watch folder: drop a video in → run stages 2→6, pausing only at review. |

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
  ≥ 3 frames, 1 s gap bridging.
- **Basic-Energy decoys.** Every pack has an ordinary basic Energy that isn't a pull,
  but real pulls can be Energy too (151's gold #207 Hyper Rare, Delta Reign's special
  Energy Ultra Rares). So every index also embeds the `sve` basic Energies as
  `decoy:`-prefixed vectors (`cards/service.py`). A reveal whose best match is a decoy
  is dropped, and decoys never appear as review suggestions. On the 151 video this
  gave the identical 120 pulls as before, with each pack's Energy now matching a decoy
  instead of #207. Mega Evolution-era basic Energy art is unverified until the first
  real footage: watch for 11-card packs in review.
- **Rarities** live in one place (`cards/rarity.py`). Any unknown rarity containing
  "Rare" that isn't a base rarity counts as special, because every era invents new
  chase rarities (Mega Evolution added "Mega Hyper Rare").
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

### Phone footage (first real creator video: Delta Reign ETB, 9 packs = 90 cards)

Vertical iPhone, 4K60 HEVC 10-bit HDR (HLG), 16.4 min, 6.6 GB. Three fixes:
- **Rotation:** phones store vertical video as landscape frames plus a display-matrix
  flag. `probe` reports the displayed size and sampling rotates explicitly; before, the
  picture was squashed into 16:9.
- **Decode speed:** NVDEC decode, fps first, `scale_cuda` to 8-bit, rotate last: 0.9x →
  6.6x real time (whole video about 4 min including CLIP). Falls back to the CPU.
- **Vertical crop grid:** card height ≈ 0.6–0.7 of the frame width, so box sizes are
  relative to the short side, with portrait-specific sizes and positions.

**Repeat showings** (`detection/reshows.py`). The creator deliberately re-shows cards:
hits held up again, a preview before a pack, an end recap.
- A repeat is merged into the card's pull as `Pull.showings`, so it counts once in
  review and the collection.
- Render uses up to `max_showings` (2) as showcase shots: hits play at 1× with a zoom,
  others sped up. A showing inside the pack plays in place; one from elsewhere is
  spliced in after its pack.
- Script facts carry `shown_again`. Pack grouping treats showings as activity, so
  merging doesn't split a pack.
- `pokepipe detect regroup <slug>` re-applies this without re-scanning.
- Result: 104 detections in 14 groups → 88 pulls in 10 packs (9 real + a card shown
  before the first pack), with 16 showings kept. Packs 2–8 are all 10/10.

**Recall against ground truth (90):** 86 found. The 4 misses, all explained:
- Enamorus Uncommon vs Enamorus IR: near-identical art, lead 0.007 < 0.03.
- Nidoran♀: borderline, clear on a rescan of that window.
- A second Tornadus: a reverse-holo copy in slot 8, wrongly merged as a repeat. A
  reverse slot can hold a copy of a card already in the pack.

Delta Reign also has split-art twins (Legendary Trench #98/#99) that look like one card
read twice.

Next improvements, ranked:
1. Use the pack layout to settle near-ties (slot 5 must be an Uncommon, so Enamorus U).
2. Don't merge an in-pack repeat when the pack is short and the repeat sits in a
   reverse slot.
3. Re-scan short packs' windows with a frame offset to catch borderline cards.

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
- The layout (`MODERN`) is chosen by series (`layout_for`): Scarlet & Violet (verified
  on 151) and Mega Evolution (verified on the readable packs of a Delta Reign ETB).
  Slot 9 and the rare slot accept any special rarity via `cards/rarity.py`. Other eras
  get no inference until verified.
- Short packs: every order-preserving placement of the detected cards into the 10
  slots is tried, and a slot or reverse guess is kept only if *all* fitting placements
  agree. Ambiguity goes to review instead of a wrong guess.
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
- Brand-new sets have no TCGplayer price data yet (Delta Reign, 3 days old: 0 of 141
  cards), so printings come from the rarity rule and master-set totals are
  approximate until the API catches up. Re-run `cards fetch <set> --refresh` later.
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
- **Zooms follow the card and never cut it off** (`render/track.py`, `render/zoom.py`).
  - Detection boxes are grid cells, too rough to frame a card, so the tracker re-finds
    the known card 5×/s during each zoom: coarse-to-fine box search scored against that
    one card's embedding on the GPU. CLIP scores the artwork alone about as high as the
    whole card, so it takes the *largest* box within 0.02 of the best score, kept to
    0.7–1.35× the detection box (full-art cards score flatly and would grow otherwise).
  - `zoom.py` (pure): crop width = card width × 1.5 (≤ 1.8× in), moving average, then
    any keyframe that would clip the card is widened until it fits. A lost card (score
    < 0.55) eases back to the normal framing; 1–2 lost samples are bridged.
  - Zooms ease in (0.6 s) and out (0.5 s) with zoompan instead of hard cuts. One zoom
    per card: repeat showings stay at normal framing. A hit straight after another zoom
    pans across from it instead of zooming out and back in.
  - Allen's posts: 16 zooms → 10, every zoom frame shows the whole card. Tracking adds
    about 10 s per post. `--no-track` / `RENDER_TRACK=false` skips it.
- **Pack numbers in scripts count opened packs only** (`narration/script.pack_numbers`): a group of fewer than 3 pulls, like a promo shown before the first pack, gets no number. Pull ids (`p<index>-<n>`) are unchanged.
- **Narration off** (`render post --no-narration`, or `NARRATION=false` for the watcher):
  the edit alone, with no script, voice or API spend. The source audio plays under it,
  sped up per shot with pitch-preserving `atempo` and padded through freezes, so it stays
  in sync. There's no hook and no word captions; the counter and labels stay. Allen's Delta
  Reign posts (90 s each) rendered this way in about 5 min each.
- To tune next: bigger counter/badge (the badge overlaps the card name), optional
  source audio (pack-rip sounds), and motion blur on commons at ~3.6×.

### Series edit (moments.json)

`pokepipe detect moments <slug>` (`detection/moments.py` pure, `moments_scan.py` signals)
finds footage Allen films on purpose. When `moments.json` exists, render and the script
writer use the series edit (`render/inputs.edit_inputs`):
- **Opening montage, ~1 s each with sound:** the sealed box, shrink-wrap coming off, lid
  off, packs fanned (only in the post with the first pack), then every pack's tear and
  cards coming out.
- **Packs start at their Energy,** which cuts the straightening, the code card and early
  peeks. Repeat showings are dropped from packs.
- **End recap:** each hit again (2 s for Double Rare+, 1.2 s otherwise), from its sleeving
  footage when there is some, under the outro. CTA: part *n*+1, then follow the series.
- **Sound:** source audio at 0.22 under the voice, 0.8 on the quick clips.
- **Each pack's tear plays right before that pack** (not all up front: hits already on
  the table would show early). The packs run card to card (`EditParams.cut_gaps`):
  straightening, slips and re-shown cards between reveals are cut (part 1: 48 s of such
  footage), and so is the tail after the last card.
- **Clean cuts:** in the series edit a card's shot starts and ends exactly where it is
  recognised (`reveal_lead`/`reveal_tail` = 0, and short gaps are cut, not absorbed).
  The default margins are precisely when the next card's edge peeks out from behind the
  current one. A colour check (next card's hues around the current card) was tried and
  dropped: hands and foil glare moved it as much as a real peek. Checked by eye on the
  last frames of all 40 card shots of part 1.
- **Sleeving:** after a hit, a 2 s clip of it sliding into its sleeve (2x volume: it is
  recorded very quietly). Silent, so found visually: CLIP's "sliding a card into a
  sleeve" score is high while the card goes in and drops once it's in, in the 6 s before
  a post-pack showing (each showing is tried; Kecleon was held up before sleeving).
  3/3 sleeved hits found, Kyogre (not sleeved) correctly skipped.
- **On screen:** a master-set progress bar (ASS vector drawing, its fill clip animated
  when a pack is done), a "PACK 3 OF 9" title over each tear, and spoken captions at
  about 75% of the height, above TikTok's own caption and username.
- **Script:** the first line is a hook aimed at the viewer, with 2–3 interactive
  prompts. **No spoilers:** `narration/script.spoilers` flags any line (and the
  hook caption) that names a hit, or a special rarity, before it appears. A spoiling
  draft is rewritten once with the exact problems listed; a second miss is a warning.

Signals, measured on the Delta Reign ETB video:
- **Sound** (high-passed >2.5 kHz): tearing foil and shrink-wrap are -25 to -40 dB in an
  otherwise quiet room (about -75 dB). The last crinkle of ≥1.5 s before a pack's Energy
  is its tear then cards out. 9/9 tears correct, 9/9 cards-out after one fix (the crinkle
  tails off past the cards).
- **Energy start:** basic-Energy decoy matches. 9/9.
- **CLIP zero-shot** only for "open box with packs inside" (0.94–0.98) and "packs fanned".
  Generic prompts fired everywhere because the ETB display wall is behind every shot.
  The longest wrap crinkle before the lid comes off is the seal. All 4 box shots correct.
- Tuned on one video: check `moments.json` on new footage (a different room, music, or
  talking would break the sound rules).

### Stage 7 design (watch/)

- `pokepipe watch run` = watcher loop + the review page (thread, 127.0.0.1) in one
  command. Drop videos in `<inbox>/<set_id>/` (default `data/inbox`). A video dropped in
  the inbox root uses `WATCH_DEFAULT_SET`.
- `intake.py`: a file is taken only after its size and mtime are unchanged for
  `WATCH_SETTLE_SECONDS` *and* it opens for writing (Windows locks files still being
  copied). Partial downloads are ignored. It is **moved** to
  `data/videos/<slug>/source.<ext>` so its path never changes, with a unique slug.
- `runner.py`: a state machine over an `Ops` protocol (queued → detecting →
  awaiting_review → collecting → narrating → rendering → done / failed).
  - `job.json` is the only watcher state; real outputs stay in each stage's files.
  - Every op skips work whose outputs exist, so restarts and retries are safe.
  - A failure records the stage and error; `watch retry <slug> [--force-budget]`
    resumes from that stage.
  - Single-threaded on purpose: detection and render both want the GPU.
- **One human stop.** The review page's Finish step also chooses which packs go into
  each post (`posts.json`), pre-filled with a suggestion. If none is saved, the
  suggestion is used.
- End to end on a 3-minute, 3-pack clip: taken in 8 s after dropping, detection 27 s
  (30 pulls), then after review: collection, script + voice ($0.19), render. Result:
  a 46.7 s post, 10/10 lines in sync, no freeze, about 1 min of machine time after
  review.
- Not done yet: editing a finished review re-opens it, but already-rendered posts are
  not invalidated (delete `posts/<n>/` to redo one). No desktop notification when a
  review is waiting.

## Commands

```bash
.venv\Scripts\activate
pokepipe cards sets --search "151"        # find a set id
pokepipe cards fetch sv3pt5               # checklist + images → data/sets/sv3pt5
pokepipe cards embed sv3pt5               # build GPU embeddings index
pokepipe cards match sv3pt5 photo.jpg     # sanity-check: top-5 matches for an image
pokepipe detect run video.mp4 --set sv3pt5 # → data/videos/<slug>/pulls.json + thumbs/
pokepipe detect layout <slug>             # re-run pack-slot / reverse-holo inference
pokepipe detect regroup <slug>            # merge repeat showings + regroup + layout
pokepipe detect moments <slug>            # box / tear / cards-out / Energy → series edit
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
pokepipe watch run                        # inbox watcher + review page, end to end
pokepipe watch status                     # where every video is
pokepipe watch retry <slug>               # resume a failed video at its failed stage
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
