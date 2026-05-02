# BirdVision — Agent Context

This file is the authoritative context document for AI coding agents working
on this repository. It is symlinked as `CLAUDE.md` for Claude Code compatibility.

## What this project is

Bird species identification from video and photos. Point a camera at a bird,
run the pipeline, get species predictions weighted by visual similarity plus
optional eBird location/season frequency data when the media falls inside a
supported region.

## Hardware

**Desktop (webapp / training / model compilation)**
- RTX 3080 Ti (12GB VRAM), AMD Ryzen 9 7900X, 32GB RAM, x86_64 Ubuntu
- Location: 40.7, -73.5 (Long Island / Nassau County, NY)
- GitHub username: evandhoffman, HuggingFace username: k10z

**Raspberry Pi 5 (real-time streaming pipeline)**
- Hailo-8 AI Processor (PCIe, 26 TOPS INT8), 8GB RAM, aarch64 Ubuntu 24.04
- Elgato Cam Link 4K (HDMI→USB capture), Samsung camcorder via HDMI
- HailoRT firmware: 4.23.0

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
  pipeline_defaults.py   — default species list (Northeast NA, ~242 species)
  tuner.py               — single-video parameter tuner; grid-searches
                           hot-reloadable params against a known-species asset
  video_metadata.py      — ExifTool/OpenCV metadata helpers
  webapp.py              — FastAPI web UI + JSON API; content-addressed asset
                           store; two-phase upload; Google OAuth gating;
                           paginated job listing; friendly slug URLs; Open Graph
                           metadata; theme switcher; per-request access logging.
                           API: POST /api/v1/videos (token-authenticated)

  [Pi-only — do not import from webapp or existing pipeline]
  hailo_detector.py      — YOLOv8n detection via Hailo-8 HEF; same interface
                           as detector.py; shares VDevice with classifier
  hailo_classifier.py    — EfficientNet-S classification via Hailo-8 HEF;
                           same interface as classifier.py; shares VDevice
  stream_capture.py      — V4L2 live frame source (Cam Link 4K); yields
                           (frame_no, bgr) iterator; YUYV capture
  ws_frame_source.py     — Starlette/uvicorn WebSocket frame source for phone
                           sidecar mode; serves static browser client; accepts
                           JPEG frames + metadata; sends normalized detections;
                           exposes POST /upload for sidecar file identification
  file_frame_source.py   — video-file frame source with the same iterator
                           interface, useful for Pi pipeline debugging
  display_overlay.py     — Pi Touch Display 2 framebuffer writer; live feed,
                           detection boxes, CC-style species caption on /dev/fb0
  realtime_pipeline.py   — orchestrates V4L2/WebSocket source → hailo_detector →
                           tracker → hailo_classifier; optional display overlay;
                           sidecar upload processing + debug crops; 1s summary
                           logs; system stats every 30s; clean shutdown

scripts/
  serve.py                       — uvicorn entry point (port 3587)
  identify_videos.py             — CLI batch processor
  import_ebird_barchart.py       — eBird bar chart TSVs -> SQLite DB
  tune_single_video.py           — CLI for tuner.py
  realtime_identify.py           — [Pi] real-time streaming entry point
  download_inat_training_data.py — fetch iNaturalist training photos by species
  train_efficientnet.py          — fine-tune EfficientNet-V2-S, export ONNX
  compile_efficientnet_hef.py    — compile ONNX to Hailo HEF via DFC SDK
  verify_efficientnet_onnx.py    — onnxruntime shape/label-count verification
  upload_model_to_hf.py          — upload model artifacts to HuggingFace Hub
  ws_test_client.py              — [Pi] stream a video file to sidecar WebSocket
                                  server for repeatable phone-free testing
  autostart.sh                   — [Pi] install/uninstall systemd service wrapper
  run_training.sh                — convenience wrapper for train_efficientnet.py
  run_verify_efficientnet_onnx.sh — logged wrapper for ONNX verification
  run_compile_efficientnet_hef.sh — logged wrapper for HEF compilation
  log_utils.py                   — shared file logging + ETA helpers for long jobs

eval/                       — model comparison eval container
  eval_runner.py            — runs multiple classifier backends on a test set
  report_generator.py       — generates comparison reports from eval results
  Dockerfile, docker-compose.yml, config.yaml

templates/
  base.html, index.html, job.html  — Jinja2, mobile-friendly

static/
  index.html                  — Pi sidecar browser camera/upload client

ebird_data/
  ebird_US-NY-{047,059,061,081,103}__*_barchart.txt
  — Kings, Nassau, Manhattan, Queens, Suffolk counties
  — Imported at Docker build time -> data/ebird_priors.db (gitignored)

data/
  species_lists/north_america_common.txt  — 242-species list (Northeast NA)
  ebird_priors.db                         — generated, not committed

pi/
  README.md                    — Pi setup, hardware, model locations, run instructions
  hailo_hef_compile_status.md  — YOLOv8n + EfficientNet-S compile history and results
  display_test/                — standalone framebuffer/camera display test container
  sidecar_plan.md              — implementation notes for phone-to-Pi sidecar mode
  models/                      — .hef + .har + .onnx + species_labels.json (all gitignored)

config.pi.yaml         — Pi-specific config (stream device, Hailo HEF paths, realtime settings)
                         Separate from config.yaml; do not merge them.
config.pi.sidecar.yaml — Pi sidecar config; WebSocket source, browser static files, display off.
Dockerfile.pi          — arm64 image; Hailo + V4L2 device passthrough
docker-compose.pi.yml  — Pi compose profiles: backyard (Hailo + video + fb0)
                         and sidecar (Hailo + Caddy HTTPS reverse proxy)
Caddyfile.sidecar      — sidecar HTTPS reverse proxy; self-signed internal certs
```

## Models

### Desktop pipeline (webapp)
- **Detector**: YOLOv8s (Ultralytics, COCO)
- **Classifier**: BioCLIP zero-shot, optionally ensembled with a secondary HF model

### Raspberry Pi pipeline
| File | Description | Benchmark |
|---|---|---|
| `pi/models/yolov8n.hef` | YOLOv8n detector, Hailo-8 INT8 | 212 FPS on Pi |
| `pi/models/efficientnet_s_birds.hef` | EfficientNet-V2-S classifier, 237 species | 22 FPS / 44ms |
| `pi/models/species_labels.json` | 237-entry class index for classifier | — |

Trained model + HEF: https://huggingface.co/k10z/birdvision-efficientnet-s

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
- **Safari transcoding** — VP9/AV1 videos can be transcoded to H.264 before
  ML processing and playback when `webapp.transcode_incompatible_video` is true
- **Theme switcher** — Default, Birdy, Super Birdy; cookie-backed
- **OAuth callback override** — `auth.redirect_uri` for reverse-proxied deploys
- **Access logs** — proxy IP, forwarded client IP, signed-in email
- **Config hot-reload** — re-read before each job; model/device changes
  require restart
- **Species name normalization** at eBird import time (NAME_OVERRIDES dict)
- **Pi has two runtime modes** — backyard mode reads the Cam Link 4K over V4L2;
  sidecar mode serves a phone browser client over HTTPS/WebSocket and receives
  JPEG frames + GPS metadata from the phone
- **Pi Touch Display overlay** — optional `/dev/fb0` writer shows the live feed,
  boxes, and a bottom caption; no X11/Wayland required
- **Pi sidecar uploads** — browser client can upload photo/video files to the
  Pi; upload path logs copyable prediction summaries and saves classifier debug
  crops under `results/upload_debug/` or `/tmp/birdvision_upload_debug/`
- **Hailo classifier input layout** — compiled EfficientNet-S HEF expects NHWC
  `(224, 224, 3)` tensors; do not change preprocessing back to CHW
- **Pi classifies at 22 FPS** — sufficient for `classify_every_n_frames: 10`
  at 60fps capture (max 6 classifications/sec per track needed)

## Docker

**Webapp (x86_64)**
- Base image: `cgr.dev/chainguard/python:latest-dev` (Wolfi/Alpine)
- `USER root` before `apk add`; drop to `nonroot` before CMD
- venv layer separate from source layers for fast rebuilds
- `config.yaml` bind-mounted (`./config.yaml:/data/config.yaml:ro`)
- Writable volumes: `./videos`, `./results`, `./models` (chmod 777 on host)
- eBird SQLite DB built during `docker compose build`
- Run: `docker compose up`

**Raspberry Pi (arm64)**
- `Dockerfile.pi` + `docker-compose.pi.yml` — separate files, do not modify main compose
- Backyard profile device passthrough: `/dev/hailo0`, `/dev/video0`, `/dev/video1`,
  and optional `/dev/fb0` for Pi Touch Display 2
- Sidecar profile uses `/dev/hailo0` plus Caddy on ports 80/443; phone opens
  `https://<pi-ip>/` and accepts the one-time self-signed cert warning
- Configs: `config.pi.yaml` for backyard, `config.pi.sidecar.yaml` for sidecar;
  both are bind-mounted as `/app/config.pi.yaml`
- Run backyard: `docker compose -f docker-compose.pi.yml --profile backyard up`
- Run sidecar: `docker compose -f docker-compose.pi.yml --profile sidecar up`

## Config (config.yaml)

Treat `config.yaml` as the source of truth for current defaults.

Hot-reloadable: detector threshold, classifier cadence/padding/gates, tracker
thresholds, scoring weights, metadata coords/FIPS/results dir, auth emails,
prior_mode, local_priors_file.

Require restart: model path/name/device, OAuth credentials, session secret.

## Workflow notes for AI/code agents

- **Unit tests**: `uv run pytest` (279 tests collected, no GPU needed). Tests cover
  tracker, metadata priors, pipeline helpers, video metadata, webapp
  auth/upload, ensemble classifier, WebSocket frame source, and Hailo classifier
  preprocessing. Heavy deps monkeypatched in fixtures.
- **Smoke checks**:
  - `uv run scripts/import_ebird_barchart.py ebird_data/ --db /tmp/ebird_priors.db`
  - `uv run scripts/serve.py --config config.yaml --port 3587`
  - `uv run scripts/identify_videos.py <video-or-dir> --config config.yaml`
  - Direct sidecar WebSocket dev: `uv run --no-project --with websockets,opencv-python-headless scripts/ws_test_client.py <video> --server ws://<pi-ip>:8765/ws`
- **Pipeline startup is heavyweight**: model init/download at runtime. Avoid
  unnecessary full runs for small edits.
- **Default config targets container paths** (`/data/...`). For local runs use
  local path overrides or an alternate config file.
- **Pi dependency isolation**: `hailort` and other Pi-only packages live in
  `[dependency-groups] pi` in `pyproject.toml`. Never add them to
  `[project.dependencies]` — they only install on Linux aarch64 and will break
  CI and desktop `uv sync`. Install on Pi with `uv sync --group pi`.
- **`[tool.uv] default-groups = ["dev"]`** in `pyproject.toml` — prevents uv
  from trying to resolve `hailort` (not on PyPI) during normal `uv run` /
  `uv sync`. Do not remove this.
- **Pi scripts that don't need the project**: use `uv run --no-project --with <pkg>`
  to avoid project dep resolution entirely (e.g. training and download scripts).
- **Long-running retraining steps log to files**: `run_training.sh`,
  `run_verify_efficientnet_onnx.sh`, and `run_compile_efficientnet_hef.sh`
  print a timestamped log file under `logs/retraining/` so users can `tail -f`
  while Codex-launched jobs are running.
- **Pi code is additive**: `src/hailo_*.py`, `src/stream_capture.py`,
  `src/ws_frame_source.py`, `src/file_frame_source.py`, `src/display_overlay.py`,
  and `src/realtime_pipeline.py` are Pi-focused. Do not pull Pi-only deps into
  desktop modules.
- **Three configs, never merged**: `config.yaml` = webapp,
  `config.pi.yaml` = Pi backyard, `config.pi.sidecar.yaml` = Pi sidecar.
  Do not add Pi-specific keys to `config.yaml`.
- **Result filename conventions** (used by startup reconstruction):
  - JSON: `{job_id}_{original_stem}_results.json`
  - Video crops: `<results_dir>/<video_stem>_crops/track_<track_id>.jpg`
  - Image crops: `<results_dir>/<job_id>_crops/` (bird crops + annotated JPEGs)
  - Video stills: `still_{idx}_{frame}_annotated.jpg` in crops dir
- **Jobs reference explicit asset records**, not `{job_id}_filename` paths.
- **Serial processing** (`ThreadPoolExecutor(max_workers=1)`). Changing
  concurrency affects queue behavior and GPU memory.
- **Keep out of commits**: model weights, `results/`, `videos/` (including
  `videos/assets/` and `videos/asset_index.json`), `data/ebird_priors.db`,
  `pi/models/*.hef`, `pi/models/*.har`, `pi/models/*.onnx`,
  `pi/models/species_labels.json`, `pi/deps/*.whl`, `pi/deps/*.deb`, `certs/`.
- **Hailo DFC compilation** requires x86_64 Linux + DFC 3.33.1 from
  https://hailo.ai/developer-zone/. ONNX must be exported with `dynamo=False`
  (legacy TorchScript exporter) for DFC compatibility. Calibration data must
  be NHWC `(N, 224, 224, 3)` float32.

## GitHub

Repo: https://github.com/evanwtf/birdvision (private)
CI: GitHub Actions runs unit tests on push/PR (manual dispatch only).

## What's not done yet

Key open issues (see GitHub for full backlog):

- Broader eBird region coverage beyond Long Island (#18)
- Video-level summary robustness to noisy track fragments (#9)
- Species-group rollups and UI (#33, #32)
- Small-bird recall via tiled/zoomed fallback detection (#26)
- Tuner improvements: species-group optimization (#34), benchmark workflow (#12)
- Fine-tuning detector/classifier on BirdVision data (#7, #30)
- Human-in-the-loop active learning workflow (#31)
- Pi field sidecar networking/hotspot/autostart docs (#94, #97-#101)
- Pi power monitoring and model comparison eval (#86, #85)
- Native iPhone or hybrid phone/server exploration (#90)

**Raspberry Pi real-time sub-project** (#70) — v0.3.1, pipeline working end-to-end:
- ~~Scaffold Pi monorepo structure (#81)~~ done
- ~~OS packages on Pi (#71)~~ done
- ~~Hailo PCIe kernel driver on Pi host (#72)~~ done — `/dev/hailo0` present
- ~~Cam Link 4K verification (#73)~~ done — `/dev/video0` present
- ~~YOLOv8n to Hailo HEF (#74)~~ done — `pi/models/yolov8n.hef`, 212 FPS on Pi
- ~~EfficientNet-S fine-tune (#75)~~ done — 80.7% top-1, 94.0% top-5, 237 species
- ~~EfficientNet-S to HEF + classifier backend (#77)~~ done — 22 FPS / 44ms on Pi
- ~~Upload model to HuggingFace (#82)~~ done — https://huggingface.co/k10z/birdvision-efficientnet-s
- ~~Hailo-8 detection backend hailo_detector.py (#76)~~ done — YOLOv8n via shared VDevice
- ~~Live video capture stream_capture.py (#78)~~ done — YUYV V4L2 at 1920×1080 60fps
- ~~Real-time pipeline realtime_pipeline.py (#79)~~ done — ~27–34 FPS end-to-end, verified live
- ~~ARM64 Docker image (#80)~~ done — Dockerfile.pi + docker-compose.pi.yml
- ~~Pi Touch Display test container (#89)~~ done — live video + fake captions on `/dev/fb0`
- ~~Pi Touch Display overlay (#87)~~ done — live feed, boxes, CC-style species label
- ~~WebSocket phone sidecar mode (PR #95/#96)~~ done — browser camera client streams JPEG frames + GPS to Pi over WebSocket
- ~~Sidecar HTTPS via Caddy (v0.3.0)~~ done — phone opens `https://<pi-ip>/`; HTTP redirects to HTTPS
- ~~Sidecar file upload identification (#102/#103, v0.3.1)~~ done — photo/video upload path, copyable summaries, debug crops
- ~~Hailo EfficientNet input layout fix (#103)~~ done — classifier preprocessing now emits NHWC `(224, 224, 3)`

**Next Pi milestones:**
- Pi WiFi hotspot + sidecar field workflow (#94, #97-#101)
- Power monitoring via INA219/USB-C meter for battery runtime estimation (#86)
- Evaluate retrained 237-class EfficientNet against the current desktop classifier (#85)
- Close or reconcile stale tracking issues now implemented by PR #95/#96 (#91-#93, #88)
