# BirdVision

Proof-of-concept bird species identification from video using local computer vision models.
Aimed at the Long Island / Northeast US region, no cloud dependencies.

## How it works

1. **Detection** — finds birds in each frame
2. **Tracking** — assigns stable IDs across frames
3. **Classification** — identifies the species from each cropped bird
4. **Metadata weighting** — location + date priors re-rank predictions (stub → eBird integration planned)

Models download automatically on first run.

## Setup

### Local (uv)

```bash
uv sync
uv run scripts/identify_videos.py videos/
```

### Docker

Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
# Build
docker compose build

# Drop videos into ./videos/, then run
docker compose run --rm birdvision

# With a recording date
docker compose run --rm birdvision /data/videos --date 2026-04-15
```

Models are cached in `./models/` on the host and reused across runs.

Requires CUDA. Tested on RTX 3080 Ti.

## Usage

```bash
# Process a directory of videos
uv run scripts/identify_videos.py videos/

# With a recording date (improves seasonal priors)
uv run scripts/identify_videos.py videos/ --date 2026-04-15
```

Results are written as JSON to `results/<videoname>_results.json`.

## Test videos

Good sources for bird footage to test with:
- [Xeno-canto](https://xeno-canto.org) — bird recordings (some have video)
- YouTube: search "birds Long Island backyard" and download with `yt-dlp`
- Your own backyard camera footage

## Configuration

Edit `config.yaml` to adjust detection confidence, GPU device, output paths, etc.
The species list at `data/species_lists/north_america_common.txt` can be trimmed
to your local expected species for faster and more accurate classification.
