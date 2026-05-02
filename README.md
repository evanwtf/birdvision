# BirdVision

Bird species identification from video and photos using local computer vision models.
Aimed at the Long Island / Northeast US region, no cloud dependencies.

Primary deployment targets:
- **Desktop / webapp** — upload video or photos via browser or API; BioCLIP + ensemble classifier on GPU
- **Raspberry Pi 5 backyard mode** — real-time inference from a live HDMI capture feed using a Hailo-8 AI accelerator, with optional Touch Display overlay
- **Raspberry Pi 5 sidecar mode** — phone browser streams camera frames to the Pi over HTTPS/WebSocket and receives live boxes + species labels; no native app required
- **Native iOS / hybrid app** (exploratory) — future mobile bird ID work may run on-device, server-side, or split detection/classification across phone and server

## How it works

1. **Detection** — finds birds in each frame
2. **Tracking** — assigns stable IDs across frames, classifies every N frames
3. **Classification** — zero-shot species ID from cropped bird images via BioCLIP, optionally ensembled with a secondary HuggingFace classifier (weighted geometric mean); events are weighted by proximity to frame center (centered bird = better crop = more weight)
4. **eBird priors** — observed species frequency for the recording location and week re-ranks predictions; supports seasonal (default) or location-only mode, plus user-defined local prior overrides via YAML
5. **Explanation** — each result includes a plain-English summary of what the model saw visually vs. what the location/season data contributed

Desktop models download automatically on first run. Pi HEF models and HailoRT
packages are placed manually because Hailo's runtime packages are proprietary.

## Setup

### Docker (recommended)

Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker compose build
docker compose up
```

Open `http://localhost:3587` in a browser or on your phone. Upload video or
photos and view results with species predictions, confidence scores, annotated
image/video artifacts, and links to Cornell All About Birds for each candidate
species.

If you want Google OAuth login for uploads, export secrets before `docker compose up`:

```bash
export GOOGLE_CLIENT_ID="your-client-id.apps.googleusercontent.com"
export GOOGLE_CLIENT_SECRET="your-client-secret"
export SESSION_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

If BirdVision is served behind HTTPS through a reverse proxy, also set the
exact callback URL under `auth.redirect_uri` in `config.yaml`, for example
`https://birdvision.example.com/auth/callback`.

Managed uploaded media, the asset index, and any legacy uploads persist in
`./videos/` on the host. Results persist in `./results/`.
Model weights are cached in `./models/` and reused across restarts.

### Local (uv)

```bash
uv sync
uv run scripts/serve.py          # web UI on :3587
uv run scripts/serve.py --debug  # local debug mode, auth bypassed
uv run scripts/identify_videos.py videos/   # CLI batch mode
uv run scripts/tune_single_video.py   # single-video tuner
```

## HTTP API

In addition to the browser UI, BirdVision exposes a small JSON API for
external clients (e.g. [birdcamgrabber](https://github.com/evandhoffman/birdcamgrabber)
posting motion-event clips).

### Auth

API requests must include an `X-API-Token` header. Tokens are loaded from a
YAML file referenced by `webapp.api_tokens_file` in `config.yaml`:

```yaml
# api_tokens.yaml — gitignored, mounted into the container
tokens:
  - name: birdcamgrabber
    token: <hex>
```

Generate a token with `python -c 'import secrets; print(secrets.token_hex(32))'`.
The browser-facing `/upload` route stays gated by Google OAuth as before;
only the `/api/v1/*` endpoints check tokens. If `api_tokens_file` is unset
or empty, `/api/v1/videos` returns `503`.

### `POST /api/v1/videos`

Multipart form upload. Submits a video clip for processing and returns a
job id immediately (processing is async). The clip is ingested through the
same content-addressed asset store as browser uploads, so identical clips
deduplicate on disk.

| Field             | Required | Description                                          |
|-------------------|----------|------------------------------------------------------|
| `file`            | yes      | Video file                                            |
| `captured_at`     | yes      | ISO-8601 timestamp of the source event               |
| `latitude`        | no       | Decimal degrees; falls back to `metadata.latitude`   |
| `longitude`       | no       | Decimal degrees; falls back to `metadata.longitude`  |
| `source`          | no       | Short client name (e.g. `birdcamgrabber`)            |
| `source_event_id` | no       | Opaque upstream id; surfaced on the job page         |

Response: `202 Accepted` with `{"job_id": "...", "url": "/jobs/<id>", "status": "pending"}`.
Errors: `400` for malformed/non-video uploads, `401` for bad auth, `503` if
no tokens are configured.

```bash
curl -X POST https://birdvision.example.com/api/v1/videos \
  -H "X-API-Token: $BIRDVISION_TOKEN" \
  -F "file=@clip.mp4" \
  -F "captured_at=2026-04-07T12:34:56Z" \
  -F "latitude=40.77" -F "longitude=-73.97" \
  -F "source=birdcamgrabber" -F "source_event_id=evt-abc"
```

API-uploaded jobs are tagged with `submitted_by=<source>@api` so they appear
in the existing "Submitted by" line in the UI without special-casing.

## eBird priors

Species frequency data is sourced from eBird bar chart downloads for the
following counties:

| File | County |
|------|--------|
| `US-NY-047` | Kings (Brooklyn) |
| `US-NY-059` | Nassau |
| `US-NY-061` | New York (Manhattan) |
| `US-NY-081` | Queens |
| `US-NY-103` | Suffolk |

For GPS-driven jobs, BirdVision currently applies priors only inside a rough
Long Island bounding box. Within that area, it uses a virtual `Long Island`
region averaged across Kings, Queens, Nassau, and Suffolk. Outside that area,
the system falls back to visual-only scoring.

If GPS is unavailable, the fallback county is set via `metadata.ebird_fips` in
`config.yaml`.

Set `metadata.prior_mode` to `location_only` to use annual frequency averages
instead of the default seasonal (per-week) weighting.

### Local prior overrides

You can supply a `metadata.local_priors_file` pointing to a YAML file with
per-location species frequency adjustments. These are applied on top of the
eBird data, letting you boost or suppress species you know are present or
absent at a specific feeder or site.

To add more counties, download the bar chart from `ebird.org/barchart`,
drop the file in `ebird_data/`, and rebuild the image.

## Configuration

`config.yaml` is mounted from the host. Mutable settings are re-read before the
next job, but model/device changes still require a restart. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `detector.model` | `/data/models/yolov8s.pt` | Desktop YOLO detector weights |
| `detector.confidence` | `0.4` | Detection threshold (raise to reduce false positives) |
| `detector.enable_small_bird_zoom_fallback` | `true` | Try a center zoom detection pass periodically when full-frame detection misses small birds |
| `detector.small_bird_fallback_every_n_frames` | `5` | Frame cadence for the small-bird zoom fallback |
| `classifier.classify_every_n_frames` | `10` | Run classification every N frames for each active track |
| `classifier.crop_padding_ratio` | `0.18` | Maximum crop padding for smaller/distant birds |
| `classifier.crop_padding_ratio_min` | `0.04` | Minimum crop padding once the bird already fills a large share of the frame |
| `classifier.crop_closeup_area_ratio` | `0.06` | When bbox area reaches this fraction of the frame, padding ramps down to the minimum |
| `classifier.min_crop_area` | `2500` | Skip classification for tiny detections that are mostly noise when upscaled |
| `classifier.min_event_confidence` | `0.35` | Discard low-confidence visual classification events instead of averaging them into a track |
| `tracker.centroid_max_distance` | `0.18` | Fallback match radius as a fraction of frame diagonal when IoU matching fails |
| `tracker.min_frames_to_report` | `8` | Minimum frames tracked to include in results |
| `tracker.min_confidence_to_report` | `0.6` | Override min_frames for high-confidence single detections |
| `scoring.center_weight_strength` | `2.0` | How much to favor center-frame detections (0 = off) |
| `metadata.ebird_fips` | `US-NY-059` | Fallback county for eBird priors when GPS is unavailable |
| `metadata.prior_mode` | `seasonal` | `seasonal` (per-week) or `location_only` (annual average) eBird weighting |
| `metadata.local_priors_file` | `""` | Path to YAML file with per-location species frequency overrides |
| `webapp.debug` | `false` | Local/debug bypass for auth checks on upload and reprocess endpoints |
| `webapp.transcode_incompatible_video` | `true` | Transcode VP9/AV1 uploads to H.264 for Safari/iOS playback and processing |
| `webapp.api_tokens_file` | `/data/api_tokens.yaml` | YAML token file for `POST /api/v1/videos` |
| `auth.redirect_uri` | `""` | Optional explicit OAuth callback URL to use behind HTTPS reverse proxies |
| `auth.allowed_emails` | `[]` | Signed-in Google accounts allowed to upload and reprocess jobs |

Google OAuth client ID, client secret, redirect URI, and session secret can
live either in `config.yaml` or the `GOOGLE_CLIENT_ID`,
`GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`, and `SESSION_SECRET`
environment variables. Environment variables take precedence so you can keep
secrets out of git. Changes to `auth.allowed_emails` are picked up without
restart; changing OAuth credentials, redirect URI, or the session secret still
requires restarting the web server.

For local work, set `webapp.debug: true`, run `uv run scripts/serve.py --debug`,
or export `BIRDVISION_DEBUG=1` to bypass auth entirely.

## Google OAuth Setup

Uploads and job reprocessing can be gated behind Google sign-in plus an email
whitelist. Browsing the job list and results pages stays public.

Full setup guide: [docs/google_oauth_setup.md](docs/google_oauth_setup.md).
That doc covers creating the Google Cloud OAuth client, setting the callback
URI, generating `SESSION_SECRET`, and wiring the values into BirdVision.

## Themes

Use the header theme switcher to change the UI appearance. BirdVision currently
ships with:

- `Default` — the current restrained green UI
- `Birdy` — a warmer, more natural bird-book palette
- `Super Birdy` — a loud Big Bird-style yellow/orange/blue palette with animated
  bird icons drifting in the background

The selected theme is stored in a cookie and applies to both the upload page
and results pages.

## Request Logging

BirdVision writes an application-level access log entry for each request. In
addition to method, path, status, and duration, it logs:

- `proxy_ip` — the direct client address seen by the app, usually the reverse proxy
- `real_ip` — `X-Real-IP` when present, otherwise the first `X-Forwarded-For` IP
- `x_forwarded_for` and `x_real_ip` — the raw forwarded headers
- `email` — the signed-in Google account email from the session, or `-` for anonymous requests

For accurate client IP logging behind a reverse proxy, configure the proxy to
send `X-Forwarded-For` and/or `X-Real-IP`.

## Results display

### Photos

- each uploaded photo gets its own `Visual` vs `+ eBird` species table
- the saved result includes an annotated full-photo JPEG with detection boxes
- the web UI adds a CSS overlay on top of the scaled image so box labels remain
  readable at browser size
- photo metadata shows whether the image is considered `On Long Island` or
  `Not on Long Island` for eBird gating
- photos with no detections show the original image in a collapsed view

### Videos

- aggregate species score at the top of the results page, with per-model
  score breakdown when using the ensemble classifier
- embedded video player with codec compatibility warning for Safari
- annotated stills extracted at evenly-spaced intervals, snapped to frames
  with active tracked detections, rendered with the same bounding box overlay
  style as photos
- representative frame gallery and per-track detail in a collapsible section
- camera, date, and GPS metadata displayed when available from EXIF

## Web UI features

- **Friendly job URLs** — jobs get human-readable slug URLs (e.g.
  `/jobs/abc123-mourning-dove-blue-jay`); bare ID links redirect
- **Open Graph metadata** — job pages include share card metadata for social
  media previews
- **Paginated job listing** — with media type labels (photo/video) and species
  summaries; jobs sorted by creation date
- **Job attribution** — uploaded jobs show the submitter's email (visible to
  signed-in users)

## Upload review and duplicate handling

The web upload flow is two-stage:

1. choose files or drag-and-drop them onto the upload page
2. BirdVision inspects each asset and shows a review list before processing
3. use Select All / Select None / Select New buttons to quickly filter assets
4. submit the checked assets for processing

Multiple videos create one job per video, queued and processed serially.
Photos are batched into a single job. Mixed image/video selections are rejected.

During inspection, BirdVision reports filename, filesize, media type, dimensions,
and when available duration, framerate, GPS, and recorded date. The backend
computes a `sha256` for each upload and stores media in a content-addressed
asset store under `videos/assets/`.

If the same bytes were already uploaded before, BirdVision marks the item as a
duplicate and transparently reuses the existing canonical file instead of
writing another copy. The "Select New" button quickly selects only files that
have not been previously uploaded.

Recording date for eBird seasonal weighting is derived from embedded media
metadata (EXIF/QuickTime). If no date is present in the metadata, seasonal
weighting is skipped.

## CLI batch mode

```bash
# Process a directory of videos
docker compose --profile cli run birdvision

# With explicit date
docker compose --profile cli run birdvision \
  /data/videos --date 2026-04-15 --config /data/config.yaml
```

Results are written as JSON to `results/{job_id}_{display_stem}_results.json`.
Image-job artifacts are saved under `results/<job_id>_crops/`.

## Model comparison eval

The `eval/` container runs every video in the asset library through multiple
classifier backends in sequence and produces a self-contained HTML report so
you can visually compare what each model predicted for each clip.

### Quick start (local, no Docker)

The BioCLIP pass reads your existing results JSONs — no GPU, no model load:

```bash
uv run eval/eval_runner.py --config eval/config-local.yaml
uv run eval/report_generator.py --config eval/config-local.yaml
# open eval/report/report.html in a browser
```

To limit to a subset while iterating:

```bash
uv run eval/eval_runner.py --config eval/config-local.yaml --max-clips 20
```

### Docker workflow

Three compose services, run in order:

```bash
# 1. Download / verify all configured models (fast on repeat runs)
docker compose -f eval/docker-compose.yml run --rm prefetch

# 2. Run inference over all clips, write sidecar JSONs
#    The report/ output directory must be writable by the container's
#    nonroot user (uid 65532). If it was created by a local uv run, fix first:
#      chmod 777 eval/report
docker compose -f eval/docker-compose.yml up eval

# 3. Generate report.html from sidecars (no GPU needed)
docker compose -f eval/docker-compose.yml --profile report up report
```

The report is written to `eval/report/report.html` alongside a `crops/`
directory of copied bird crop images. The HTML file is self-contained and
portable — open it directly in a browser.

### Adding a model

Enable a backend in `eval/config.yaml`:

```yaml
models:
  - id: gemma4_e4b
    label: "Gemma 4 E4B-it"
    backend: gemma4
    model: google/gemma-4-E4B-it
    enabled: true          # flip this when the backend is implemented
```

Then run the prefetch → eval → report sequence above. The `prefetch` service
will download the model weights before the eval run starts.

### Report features

- **Cards sorted by disagreement** — clips where models disagree appear first
- **Implausible-species flagging** — picks that are ecologically unlikely at a
  backyard feeder (Redhead, Horned Grebe, Atlantic Puffin, etc.) are highlighted
  in red
- **Filter buttons** — show all clips, disagreements only, or implausible only
- **Per-model top-5 predictions** with confidence scores and individual model
  score breakdowns per track

## Single-video tuner

The tuner runs repeated video jobs against one target species, reusing the
normal BirdVision pipeline and saving each trial under `results/tuning/`.

```bash
# Local baseline gull case
uv run scripts/tune_single_video.py \
  videos/assets/e84e85beb30cf97e7ccce5a1fe0a6b1bd705b5e81b282cd9a30008c9860cc3c6.mov \
  --target-species "Herring Gull" \
  --stop-confidence 0.60 \
  --time-budget-minutes 30

# Docker Compose tuner service
docker compose --profile tuner run --rm tuner \
  /data/videos/assets/e84e85beb30cf97e7ccce5a1fe0a6b1bd705b5e81b282cd9a30008c9860cc3c6.mov \
  --target-species "Herring Gull" \
  --stop-confidence 0.60 \
  --time-budget-minutes 30
```

The first trial is always the current baseline config. After that, the runner
does a bounded coordinate search over BirdVision's hot-reloadable detector,
classifier, tracker, and scoring settings. It stops when the target confidence
threshold is hit, the time budget is exhausted before launching another trial,
or no single-parameter improvement remains.

## Raspberry Pi real-time pipeline

BirdVision runs on a Raspberry Pi 5 with a Hailo-8 AI accelerator. The Pi
pipeline has two modes:

- **Backyard mode** — reads the Elgato Cam Link 4K over V4L2, runs real-time
  detection/classification, and can draw a live overlay to the Pi Touch Display
  2 framebuffer (`/dev/fb0`).
- **Sidecar mode** — serves a phone browser client over HTTPS. The phone
  streams camera frames and GPS metadata to the Pi over WebSocket, then receives
  normalized boxes and species labels for live overlay.

The backyard pipeline runs at ~27–34 FPS end-to-end, classifying every 10
frames per tracked bird. eBird seasonal priors and local feeder overrides are
applied to re-rank visual predictions, matching the same prior system used by
the desktop pipeline.

### How it works

```
V4L2 camera or phone WebSocket → YOLOv8n (Hailo-8) → BirdTracker → EfficientNet-S (Hailo-8)
          60fps / JPEG frames          212 FPS          IoU+centroid       22 FPS / 44ms
                                                                            ↓
                                              eBird priors + local overrides + 1s log summary
```

Both models share a single `VDevice` handle because the Hailo-8 chip can only
be opened once per process. Detection and classification run as separate
network groups on the same hardware.

### Models

| Model | File | Performance |
|---|---|---|
| YOLOv8n detector | `pi/models/yolov8n.hef` | 212 FPS on Hailo-8 |
| EfficientNet-V2-S classifier | `pi/models/efficientnet_s_birds.hef` | 22 FPS / 44ms |

The EfficientNet-S classifier is fine-tuned on 237 North American bird species
using iNaturalist research-grade photos (New York state).
Trained weights and HEF: https://huggingface.co/k10z/birdvision-efficientnet-s

**Validation accuracy:** 80.7% top-1, 94.0% top-5

### Log format

```
# Every ~1 second:
top_species=Mourning Dove confidence=0.63 tracks=1 fps=27.8
no_detection tracks=0 fps=33.4

# Every 30 seconds:
system temp=70.5C load=4.58/3.65 cpu=43% freq=1500MHz mem=15% fan=7610rpm
```

### Setup

See `pi/README.md` for full setup instructions, model download, and run commands.

```bash
# Build on Pi
docker compose -f docker-compose.pi.yml build

# Backyard USB camera mode
docker compose -f docker-compose.pi.yml --profile backyard up

# Phone sidecar mode
docker compose -f docker-compose.pi.yml --profile sidecar up
```

Only one Pi profile should run at a time because Hailo-8 access is exclusive.

In sidecar mode, open `https://<pi-ip>/` on the phone. Caddy terminates HTTPS
with an internal self-signed certificate; accept the one-time browser warning,
tap **Start Camera**, and allow camera/GPS access. HTTP on port 80 redirects to
HTTPS automatically.

The sidecar page also supports photo/video file upload. Uploaded media runs
through the same Pi Hailo pipeline and returns image boxes, video species
summaries, per-track details, and copyable plain-text results. Upload
classification logs and debug crops are written under `results/upload_debug/`
or `/tmp/birdvision_upload_debug/` when the results volume is not writable.

For repeatable development without a phone when connecting directly to the
sidecar WebSocket server:

```bash
uv run --no-project --with websockets,opencv-python-headless \
  scripts/ws_test_client.py test_video.mp4 --server ws://<pi-ip>:8765/ws --fps 5
```

### Touch Display overlay

Backyard mode can write directly to the Pi Touch Display 2 framebuffer with no
X11/Wayland. `src/display_overlay.py` scales the live feed, draws detection
boxes, and shows a closed-caption-style species label at the visual bottom.
Set `display.enabled: false` in `config.pi.yaml` for headless deployments.

### Retraining

```bash
./scripts/run_training.sh
./scripts/run_verify_efficientnet_onnx.sh
./scripts/run_compile_efficientnet_hef.sh \
  --onnx pi/models/efficientnet_s_birds.onnx \
  --train-dir train_data \
  --output pi/models/efficientnet_s_birds.hef
```

Each script writes a timestamped log file under `logs/retraining/` and prints
the `tail -f` command at startup.
