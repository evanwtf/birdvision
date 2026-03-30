# BirdVision — Claude Code Context

## What this project is

Bird species identification from video. Point a camera at a bird, run the
pipeline, get species predictions weighted by visual similarity + eBird
location/season frequency data.

## Hardware

- RTX 3080 Ti (12GB VRAM), AMD Ryzen 9 7900X, 32GB RAM
- Location: 40.7, -73.5 (Long Island / Nassau County, NY)

## Architecture

```
src/
  detector.py       — YOLOv8 object detection (COCO bird class), returns bbox + crop
  tracker.py        — IoU-based multi-frame tracker with stable IDs; stores
                      both raw (pre-prior) and weighted (post-prior) prediction
                      history per track; weighted average for final result
  classifier.py     — BioCLIP zero-shot classifier; pre-computes text embeddings
                      for all species at startup; batched inference
  metadata.py       — eBird bar chart priors; maps date → 1-of-48 annual periods,
                      multiplies classifier probs by observed frequency per county;
                      zero-frequency species floored at 0.01
  pipeline.py       — orchestrates all stages; center-weights classification
                      events by bbox proximity to frame center; generates plain-
                      English explanation per track comparing visual vs prior scores
  video_metadata.py — ExifTool wrapper; extracts recording date + GPS from video
  webapp.py         — FastAPI web UI; in-memory job queue; restores completed jobs
                      from results JSON on startup; hot-reloads config before each job

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
- **eBird priors** are multiplicative — visual prob × frequency, then renormalized;
  species with 0 frequency are floored at `zero_floor=0.01` (not zeroed out)
- **Job state** is in-memory but reconstructed from results JSON on startup
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

## Config (config.yaml) — hot-reloadable settings

| Key | Default | Notes |
|-----|---------|-------|
| `detector.confidence` | `0.4` | Raise to cut false positives |
| `tracker.min_frames_to_report` | `3` | Filter short/noisy tracks |
| `tracker.min_confidence_to_report` | `0.6` | Bypass min_frames if confidence high |
| `scoring.center_weight_strength` | `2.0` | Gaussian weight for center-frame events |
| `metadata.ebird_fips` | `US-NY-059` | Nassau County |
| `metadata.ebird_db` | `/app/data/ebird_priors.db` | Built at image build time |

## GitHub

Repo: https://github.com/evandhoffman/birdvision (private)

## Open issues

- evandhoffman/birdvision#1 — eBird priors integration (now done; issue can be closed)

## What's not done yet

- eBird API integration for dynamic location-based county selection
  (currently hardcoded to Nassau County FIPS)
- Live camera feed support (currently batch/upload only)
- Results persistence across container restarts beyond JSON reconstruction
  (no database; in-memory only)
