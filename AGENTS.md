# BirdVision — Claude Code Context

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
  detector.py       — YOLOv8 object detection (COCO bird class), returns bbox + crop
  tracker.py        — multi-frame tracker with IoU matching plus centroid-
                      distance fallback; stores both raw (pre-prior) and
                      weighted (post-prior) prediction history per track
  classifier.py     — BioCLIP zero-shot classifier; pre-computes text embeddings
                      for all species at startup; batched inference
  metadata.py       — eBird bar chart priors; maps date → 1-of-48 annual periods,
                      applies priors only inside a rough Long Island bounding box,
                      averages Kings/Queens/Nassau/Suffolk frequencies there, and
                      otherwise falls back to visual-only scoring; zero-frequency
                      species floored at 0.01
  pipeline.py       — orchestrates all stages; center-weights classification
                      events by bbox proximity to frame center; applies adaptive
                      crop padding + confidence/crop-size gates; generates both
                      video-level species summaries and per-track explanations;
                      for image jobs emits per-photo summaries, annotated
                      full-photo JPEGs, and browser overlay metadata; for video
                      jobs extracts annotated stills snapped to tracked frames
  video_metadata.py — ExifTool/OpenCV metadata helpers; extracts recording date,
                      GPS, camera info, video codec, dimensions, and duration/fps
  webapp.py         — FastAPI web UI; content-addressed asset store + persisted
                      hash index; two-phase inspect/finalize upload flow;
                      multi-video uploads split into one job per video;
                      in-memory job queue restored from results JSON on startup;
                      hot-reloads config before each job

scripts/
  serve.py                    — uvicorn entry point for web UI (port 3587)
  identify_videos.py          — CLI batch processor
  import_ebird_barchart.py    — parses eBird bar chart TSVs → SQLite DB

templates/
  base.html, index.html, job.html  — Jinja2, mobile-friendly

ebird_data/
  ebird_US-NY-{047,059,061,081,103}__*_barchart.txt
  — Kings, Nassau, Manhattan, Queens, Suffolk counties
  — Imported at Docker build time → data/ebird_priors.db (gitignored)

data/
  species_lists/north_america_common.txt  — 238 Northeast species
  ebird_priors.db  — generated, not committed
```

## Key design decisions

- **No video output** — text logs + JSON results + JPEG crops only
- **Classify every 15 frames** per track, not every frame
- **Center weighting** — classification events weighted by Gaussian based on
  distance of bbox center from frame center (`scoring.center_weight_strength`)
- **Adaptive crop padding** — smaller/distant birds get more surrounding
  context; close-up birds get less background
- **Video-level summary + per-track detail** — UI and console now lead with a
  video-level species summary; fragmented tracks remain available as debug detail
- **Image jobs are photo-first** — the job page shows per-photo species tables,
  the original photo with overlaid detection boxes, and CSS overlay labels tied
  to each classified box; photos with no detections show the original in a
  collapsed view
- **Video jobs have annotated stills** — evenly-spaced stills (min(10,
  duration/2)) are extracted and snapped to the nearest tracked frame within
  +/-1s, then annotated with the same bounding box overlay as photos; the video
  is also embedded as a player on the results page
- **Upload flow is inspect then finalize** — the upload page inspects each
  candidate asset, surfaces duplicate status plus metadata, and offers
  Select All / Select None / Select New batch controls; multiple videos create
  one job per video
- **Managed uploads are content-addressed** — uploaded media are stored and
  reused by `sha256` under the webapp upload directory, with a persisted index;
  duplicate uploads reuse existing canonical files instead of creating copies
- **eBird priors** are multiplicative — visual prob × frequency, then renormalized;
  species with 0 frequency are floored at `zero_floor=0.01` (not zeroed out)
- **Long Island-only eBird gating for GPS-driven jobs** — when photo/video GPS is
  present, priors are only applied inside a rough Long Island bounding box using
  a virtual region averaged across Kings, Queens, Nassau, and Suffolk; outside
  that area the system stays visual-only
- **Job state** is in-memory but reconstructed from results JSON on startup;
  the job listing auto-refreshes while any job is pending/running
- **Job listing shows species** — completed jobs display date + top species
  (e.g. "2024-02-03: Mourning Dove, Blue Jay, 2 others") instead of filename
- **Recording date comes from media metadata** — EXIF/QuickTime dates are used
  for seasonal eBird weighting; no manual date input in the upload form
- **Video codec warning** — non-Safari-compatible codecs (VP9, AV1) are detected
  at upload time and a warning is shown on the job page
- **Config hot-reload** — config.yaml is re-read before each job; model/device
  changes still require restart
- **Species name normalization** happens at eBird import time (NAME_OVERRIDES
  dict in import_ebird_barchart.py), not at query time

## Docker

- Base image: `cgr.dev/chainguard/python:latest-dev` (Wolfi/Alpine)
- `USER root` required before `apk add`; drop back to `nonroot` before CMD
- venv layer is separate from source layers — dependency changes are slow,
  source-only changes are fast
- `config.yaml` is bind-mounted from host (`./config.yaml:/data/config.yaml:ro`)
- Writable volumes: `./videos`, `./results`, `./models` — must be chmod 777
  on host (Docker creates them as root)
- eBird SQLite DB is built during `docker compose build` via import script

## Config (config.yaml)

Current defaults change as tuning evolves; treat `config.yaml` as the source of
truth rather than this document.

Hot-reloadable settings currently include:

- detector confidence threshold
- classifier cadence, crop padding, crop-size gate, and event-confidence gate
- tracker disappearance/IoU/centroid thresholds and reporting thresholds
- scoring.center-weight strength
- metadata latitude/longitude/FIPS and results directory

Model path / model name / device changes still require restart.

## Workflow notes for AI/code assistants

- **No automated test suite is currently checked in** (`tests/` is absent).
  Prefer focused smoke checks:
  - `uv run scripts/import_ebird_barchart.py ebird_data/ --db /tmp/ebird_priors.db`
  - `uv run scripts/serve.py --config config.yaml --port 3587`
  - `uv run scripts/identify_videos.py <video-or-dir> --config config.yaml`
- **Pipeline startup is heavyweight**: model init/download happens at runtime
  (BioCLIP + YOLO). Avoid unnecessary full end-to-end runs for small edits.
- **Default `config.yaml` targets container paths** (`/data/...`). For local uv
  runs, use local path overrides (`videos/`, `results/`, `data/ebird_priors.db`)
  or an alternate config file.
- **Preserve result filename conventions** used by startup reconstruction:
  - JSON pattern: `{job_id}_{original_stem}_results.json`
  - Video track crops: `<results_dir>/<video_stem>_crops/track_<track_id>.jpg`
  - Image-job artifacts live under `<results_dir>/<job_id>_crops/` and
    include per-bird crops plus annotated full-photo JPEGs
  - Video stills: `still_{idx}_{frame}_annotated.jpg` in the crops dir
- **Jobs now reference explicit asset records**, not `{job_id}_filename`
  upload paths. Reprocess should reuse the same canonical asset descriptors.
- **Webapp processing is intentionally serial** (`ThreadPoolExecutor(max_workers=1)`).
  Changing concurrency affects queue behavior and GPU memory pressure.
- **Keep generated artifacts out of commits**: model weights, `results/`,
  `videos/` (including `videos/assets/` and `videos/asset_index.json`), and
  generated `data/ebird_priors.db` are intentionally gitignored.

## GitHub

Repo: https://github.com/evandhoffman/birdvision (private)

## What's not done yet

- Broader eBird region coverage and more generic GPS -> place naming are still
  open; GPS-driven priors currently only recognize rough Long Island coverage
- Live camera feed support (currently batch/upload only)
- Bird-specific classifier replacement/evaluation is still open work; see GitHub
  issues for the current backlog rather than relying on this file
