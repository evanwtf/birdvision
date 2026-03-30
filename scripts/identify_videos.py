#!/usr/bin/env python3
"""
BirdVision — identify bird species in a directory of video files.

Usage:
    python scripts/identify_videos.py videos/
    python scripts/identify_videos.py videos/ --date 2026-04-15
    python scripts/identify_videos.py videos/ --config config.yaml

Each video produces a JSON file in the results/ directory with per-track
species predictions and a timestamped event log.
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.pipeline import BirdIdentificationPipeline

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("birdvision")


def find_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    videos = sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    return videos


def main():
    parser = argparse.ArgumentParser(description="Identify birds in video files")
    parser.add_argument("input", help="Video file or directory of videos")
    parser.add_argument("--config", default="config.yaml", help="Config YAML file")
    parser.add_argument("--date", help="Recording date YYYY-MM-DD (for seasonal priors)")
    parser.add_argument("--species", help="Path to species list file (overrides config)")
    parser.add_argument("--results-dir", help="Output directory for JSON results")
    args = parser.parse_args()

    # Load config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning(f"Config file not found: {args.config} — using defaults")

    if args.results_dir:
        config.setdefault("output", {})["results_dir"] = args.results_dir

    video_date = None
    if args.date:
        video_date = datetime.strptime(args.date, "%Y-%m-%d")

    videos = find_videos(Path(args.input))
    if not videos:
        logger.error(f"No video files found in: {args.input}")
        sys.exit(1)

    logger.info(f"Found {len(videos)} video(s) to process")

    pipeline = BirdIdentificationPipeline(config)
    species_file = args.species or config.get("species", {}).get("list_file")
    pipeline.load_species(species_file)

    for i, video_path in enumerate(videos, 1):
        logger.info(f"\n[{i}/{len(videos)}] Processing: {video_path.name}")
        try:
            pipeline.process_video(str(video_path), video_date=video_date)
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}", exc_info=True)
            continue

    logger.info("All done.")


if __name__ == "__main__":
    main()
