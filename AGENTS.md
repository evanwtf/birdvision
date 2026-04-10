# BirdVision — Agent Context

## What this project is

Bird species identification from video and photos. Point a camera at a bird,
run the pipeline, get species predictions weighted by visual similarity plus
optional eBird location/season frequency data when the media falls inside a
supported region.

## Hardware

- RTX 3080 Ti (12GB VRAM), AMD Ryzen 9 7900X, 32GB RAM
- Location: 40.7, -73.5 (Long Island / Nassau County, NY)

## Architecture

```
src/
  detector.py            — YOLOv8 object detection (COCO bird class), bbox + crop
  tracker.py             — multi-frame tracker; IoU + centroid-distance matching;
                           stores raw and weighted prediction history per track
  classifier.py          — BioCLIP zero-shot classifier; pre-computed text
                           embeddings; batched inference
  hf_classifier.py       — generic HuggingFace image-classification backend;
                           wraps AutoModelForImageClassification with fuzzy
                           label mapping to the project species list
  gemma_classifier.py    — Gemma 4 vision-language classifier backend
  ensemble_classifier.py — weighted geometric-mean ensemble of BioCLIP + a
                           secondary HF classifier; per-model score breakdown
                           in results JSON
  metadata.py            — eBird bar chart priors; seasonal or location-only
                           mode; local prior overrides via YAML file; Long
                           Island bounding-box gating for GPS-driven jobs
  pipeline.py            — orchestrates detect/track/classify; center-weighted
                           scoring; adaptive crop padding; video-level summaries
                           + per-track detail; image jobs emit per-photo
                           summaries, annotated JPEGs, and overlay metadata;
                           video jobs extract annotated stills
  pipeline_defaults.py   — default species list (Northeast NA, ~150 species)
  tuner.py               — single-video parameter tuner; grid-searches
                           hot-reloadable params against a known-species asset
  video_metadata.py      — ExifTool/OpenCV metadata helpers
  webapp.py              — FastAPI web UI + JSON API; content-addressed asset
                           store; two-phase upload; Google OAuth gating;
                           paginated job listing; friendly slug URLs; Open Graph
                           metadata; theme switcher; per-request access logging.
                           API: POST /api/v1/videos (token-authenticated)

scripts/
  serve.py                  — uvicorn entry point (port 3587)
  identify_videos.py        — CLI batch processor
  import_ebird_barchart.py  — eBird bar chart TSVs -> SQLite DB
  tune_single_video.py      — CLI for tuner.py

eval/                       — model comparison eval container
  eval_runner.py            — runs multiple classifier backends on a test set
  report_generator.py       — generates comparison reports from eval results
  Dockerfile, docker-compose.yml, config.yaml

templates/
  base.html, index.html, job.html  — Jinja2, mobile-friendly

ebird_data/
  ebird_US-NY-{047,059,061,081,103}__*_barchart.txt
  — Kings, Nassau, Manhattan, Queens, Suffolk counties
  — Imported at Docker build time -> data/ebird_priors.db (gitignored)

data/
  species_lists/north_america_common.txt  — species list (text format)
  ebird_priors.db  — generated, not committed
```

## Key design decisions

- **No video output** — text logs + JSON results + JPEG crops only
- **Classify every 10 frames** per track, not every frame
- **Center weighting** — classification events weighted by Gaussian based on
  bbox center distance from frame center (`scoring.center_weight_strength`)
- **Adaptive crop padding** — smaller/distant birds get more context
- **Ensemble classifier** — BioCLIP + secondary model combined via weighted
  geometric mean; per-model score breakdown preserved in results JSON;
  species outside the secondary model's vocabulary get a uniform floor
- **Video-level summary + per-track detail** — UI leads with video-level
  species summary; fragmented tracks available as debug detail
- **Image jobs are photo-first** — per-photo species tables, original photo
  with overlaid detection boxes, CSS overlay labels; no-detection photos
  show original in a collapsed view
- **Video jobs have annotated stills** — evenly-spaced stills snapped to
  tracked frames, annotated with bounding boxes; video embedded as player
- **Upload flow is inspect then finalize** — surfaces duplicate status +
  metadata; Select All / Select None / Select New controls; multiple videos
  create one job per video
- **Content-addressed asset store** — uploads stored by sha256; deduplication
  across browser and API uploads
- **API ingest** — `POST /api/v1/videos` with `X-API-Token` header validated
  against `webapp.api_tokens_file`; jobs tagged `submitted_by=<name>@api`
- **eBird priors** — multiplicative (visual * frequency, renormalized);
  zero-frequency species floored at 0.01; supports seasonal (default) or
  location-only mode via `prior_mode` config
- **Local prior overrides** — YAML file with per-location species frequency
  adjustments, applied on top of eBird data
- **Long Island-only GPS gating** — priors applied inside a rough Long Island
  bounding box (Kings/Queens/Nassau/Suffolk average); visual-only outside
- **Friendly job URLs** — jobs get slug URLs; bare ID links redirect
- **Open Graph metadata** — job pages include share card metadata
- **Paginated job listing** — with media type labels and species summaries
- **Job state** — in-memory, reconstructed from results JSON on startup;
  auto-refreshes while jobs are pending/running
- **Recording date from metadata** — EXIF/QuickTime dates for seasonal priors
- **Video codec warning** — non-Safari codecs detected at upload time
- **Theme switcher** — Default, Birdy, Super Birdy; cookie-backed
- **OAuth callback override** — `auth.redirect_uri` for reverse-proxied deploys
- **Access logs** — proxy IP, forwarded client IP, signed-in email
- **Config hot-reload** — re-read before each job; model/device changes
  require restart
- **Species name normalization** at eBird import time (NAME_OVERRIDES dict)

## Docker

- Base image: `cgr.dev/chainguard/python:latest-dev` (Wolfi/Alpine)
- `USER root` before `apk add`; drop to `nonroot` before CMD
- venv layer separate from source layers for fast rebuilds
- `config.yaml` bind-mounted (`./config.yaml:/data/config.yaml:ro`)
- Writable volumes: `./videos`, `./results`, `./models` (chmod 777 on host)
- eBird SQLite DB built during `docker compose build`

## Config (config.yaml)

Treat `config.yaml` as the source of truth for current defaults.

Hot-reloadable: detector threshold, classifier cadence/padding/gates, tracker
thresholds, scoring weights, metadata coords/FIPS/results dir, auth emails,
prior_mode, local_priors_file.

Require restart: model path/name/device, OAuth credentials, session secret.

## Workflow notes for AI/code assistants

- **Unit tests**: `uv run pytest` (243 tests, ~4s, no GPU needed). Tests cover
  tracker, metadata priors, pipeline helpers, video metadata, webapp
  auth/upload, ensemble classifier. Heavy deps monkeypatched in fixtures.
- **Smoke checks**:
  - `uv run scripts/import_ebird_barchart.py ebird_data/ --db /tmp/ebird_priors.db`
  - `uv run scripts/serve.py --config config.yaml --port 3587`
  - `uv run scripts/identify_videos.py <video-or-dir> --config config.yaml`
- **Pipeline startup is heavyweight**: model init/download at runtime. Avoid
  unnecessary full runs for small edits.
- **Default config targets container paths** (`/data/...`). For local runs use
  local path overrides or an alternate config file.
- **Result filename conventions** (used by startup reconstruction):
  - JSON: `{job_id}_{original_stem}_results.json`
  - Video crops: `<results_dir>/<video_stem>_crops/track_<track_id>.jpg`
  - Image crops: `<results_dir>/<job_id>_crops/` (bird crops + annotated JPEGs)
  - Video stills: `still_{idx}_{frame}_annotated.jpg` in crops dir
- **Jobs reference explicit asset records**, not `{job_id}_filename` paths.
- **Serial processing** (`ThreadPoolExecutor(max_workers=1)`). Changing
  concurrency affects queue behavior and GPU memory.
- **Keep out of commits**: model weights, `results/`, `videos/` (including
  `videos/assets/` and `videos/asset_index.json`), `data/ebird_priors.db`.

## GitHub

Repo: https://github.com/evandhoffman/birdvision (private)
CI: GitHub Actions runs unit tests on push/PR (manual dispatch only).

## What's not done yet

Key open issues (see GitHub for full backlog):

- Broader eBird region coverage beyond Long Island (#18)
- Video-level summary robustness to noisy track fragments (#9)
- Species-group rollups and UI (#33, #32)
- Small-bird recall via tiled/zoomed fallback detection (#26)
- Tuner improvements: species-group optimization (#34), trial logging (#28)
- Fine-tuning detector/classifier on BirdVision data (#7, #30)
- Human-in-the-loop active learning workflow (#31)
- Live camera feed support (currently batch/upload only)
