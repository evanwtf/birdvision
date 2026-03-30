# BirdVision

Proof-of-concept bird species identification from video using local computer vision models.
Aimed at the Long Island / Northeast US region, no cloud dependencies.

## How it works

1. **Detection** — YOLOv8 finds birds in each frame
2. **Tracking** — IoU tracker assigns stable IDs across frames
3. **Classification** — [BioCLIP](https://huggingface.co/imageomics/bioclip) identifies the species from each cropped bird
4. **Metadata weighting** — location + date priors re-rank predictions (stub → eBird integration planned)

Models download automatically on first run (~600 MB BioCLIP + ~6 MB YOLOv8n).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires CUDA. Tested on RTX 3080 Ti.

## Usage

```bash
# Process a directory of videos
python scripts/identify_videos.py videos/

# With a recording date (improves seasonal priors)
python scripts/identify_videos.py videos/ --date 2026-04-15
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
