# BirdVision

Bird species identification from video and photos using local computer vision models.
Aimed at the Long Island / Northeast US region, no cloud dependencies.

## How it works

1. **Detection** — finds birds in each frame
2. **Tracking** — assigns stable IDs across frames, classifies every N frames
3. **Classification** — zero-shot species ID from cropped bird images; events are weighted by proximity to frame center (centered bird = better crop = more weight)
4. **eBird priors** — observed species frequency for the recording location and week re-ranks predictions; e.g. Herring Gull at 44% checklist frequency in Nassau County in March outweighs a visually-similar but implausible species
5. **Explanation** — each result includes a plain-English summary of what the model saw visually vs. what the location/season data contributed

Models download automatically on first run (~600 MB + ~6 MB).

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

Managed uploaded media, the asset index, and any legacy uploads persist in
`./videos/` on the host. Results persist in `./results/`.
Model weights are cached in `./models/` and reused across restarts.

### Local (uv)

```bash
uv sync
uv run scripts/serve.py          # web UI on :3587
uv run scripts/identify_videos.py videos/   # CLI batch mode
uv run scripts/tune_single_video.py   # single-video tuner
```

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

To add more counties, download the bar chart from `ebird.org/barchart`,
drop the file in `ebird_data/`, and rebuild the image.

## Configuration

`config.yaml` is mounted from the host. Mutable settings are re-read before the
next job, but model/device changes still require a restart. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `detector.confidence` | `0.4` | Detection threshold (raise to reduce false positives) |
| `classifier.crop_padding_ratio` | `0.12` | Maximum crop padding for smaller/distant birds |
| `classifier.crop_padding_ratio_min` | `0.04` | Minimum crop padding once the bird already fills a large share of the frame |
| `classifier.crop_closeup_area_ratio` | `0.10` | When bbox area reaches this fraction of the frame, padding ramps down to the minimum |
| `classifier.min_crop_area` | `2500` | Skip classification for tiny detections that are mostly noise when upscaled |
| `classifier.min_event_confidence` | `0.30` | Discard low-confidence visual classification events instead of averaging them into a track |
| `tracker.centroid_max_distance` | `0.18` | Fallback match radius as a fraction of frame diagonal when IoU matching fails |
| `tracker.min_frames_to_report` | `8` | Minimum frames tracked to include in results |
| `tracker.min_confidence_to_report` | `0.6` | Override min_frames for high-confidence single detections |
| `scoring.center_weight_strength` | `2.0` | How much to favor center-frame detections (0 = off) |
| `metadata.ebird_fips` | `US-NY-059` | Fallback county for eBird priors when GPS is unavailable |

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

- aggregate species score at the top of the results page
- embedded video player with codec compatibility warning for Safari
- annotated stills extracted at evenly-spaced intervals, snapped to frames
  with active tracked detections, rendered with the same bounding box overlay
  style as photos
- representative frame gallery and per-track detail in a collapsible section
- camera, date, and GPS metadata displayed when available from EXIF

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
