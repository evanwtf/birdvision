# BirdVision

Bird species identification from video using local computer vision models.
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

Open `http://localhost:3587` in a browser or on your phone. Upload a video,
view results with species predictions, confidence scores, crop thumbnails,
and links to Cornell All About Birds for each candidate species.

Videos and results persist in `./videos/` and `./results/` on the host.
Model weights are cached in `./models/` and reused across restarts.

### Local (uv)

```bash
uv sync
uv run scripts/serve.py          # web UI on :3587
uv run scripts/identify_videos.py videos/   # CLI batch mode
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

The active county is set via `metadata.ebird_fips` in `config.yaml`.
To add more counties, download the bar chart from `ebird.org/barchart`,
drop the file in `ebird_data/`, and rebuild the image.

## Configuration

`config.yaml` is mounted from the host — edit it and the next job picks up
changes automatically (no restart needed). Key settings:

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
| `metadata.ebird_fips` | `US-NY-059` | County for eBird frequency priors |

## CLI batch mode

```bash
# Process a directory of videos
docker compose --profile cli run birdvision

# With explicit date (if not embedded in video metadata)
docker compose --profile cli run birdvision \
  /data/videos --date 2026-04-15 --config /data/config.yaml
```

Results are written as JSON to `results/<videoname>_results.json`.
