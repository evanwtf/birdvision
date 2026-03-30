#!/usr/bin/env python3
"""
BirdVision single-video parameter tuner.

Searches a bounded set of hot-reloadable parameters for one video and one
target species, stopping when the requested confidence is reached or the time
budget expires.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.tuner import (
    BASELINE_TARGET_SPECIES,
    default_baseline_video_path,
    SingleVideoTuningRunner,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("birdvision.tuner")


def _rewrite_path_if_missing(config: dict, dotted_key: str, local_path: str) -> None:
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            return
        current = current[part]

    leaf = parts[-1]
    existing = current.get(leaf)
    if not existing:
        return
    if Path(existing).exists():
        return
    if Path(local_path).exists():
        logger.info("Using local path override for %s: %s -> %s", dotted_key, existing, local_path)
        current[leaf] = local_path


def normalize_local_paths(config: dict) -> dict:
    normalized = dict(config)
    normalized["detector"] = dict(config.get("detector", {}))
    normalized["metadata"] = dict(config.get("metadata", {}))
    normalized["output"] = dict(config.get("output", {}))
    normalized["webapp"] = dict(config.get("webapp", {}))

    _rewrite_path_if_missing(normalized, "detector.model", "models/yolov8s.pt")
    _rewrite_path_if_missing(normalized, "metadata.ebird_db", "data/ebird_priors.db")
    _rewrite_path_if_missing(normalized, "output.results_dir", "results")
    _rewrite_path_if_missing(normalized, "webapp.upload_dir", "videos")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune BirdVision parameters for one video and one target species")
    parser.add_argument(
        "video",
        nargs="?",
        default=default_baseline_video_path(),
        help="Video file to tune (defaults to the baseline gull case)",
    )
    parser.add_argument(
        "--target-species",
        default=BASELINE_TARGET_SPECIES,
        help="Species to optimize at the video level",
    )
    parser.add_argument(
        "--stop-confidence",
        type=float,
        default=0.60,
        help="Stop early once the target species reaches this video-level confidence",
    )
    parser.add_argument(
        "--time-budget-minutes",
        type=float,
        default=30.0,
        help="Stop launching new trials once this overall wall-clock budget is exhausted",
    )
    parser.add_argument("--max-trials", type=int, help="Optional hard cap on trial count")
    parser.add_argument("--config", default="config.yaml", help="Config YAML file")
    parser.add_argument("--results-dir", help="Override output root for tuning artifacts")
    parser.add_argument("--date", help="Recording date YYYY-MM-DD for eBird priors")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning("Config file not found: %s; using defaults", args.config)
    config = normalize_local_paths(config)

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return 1

    video_date = None
    if args.date:
        video_date = datetime.strptime(args.date, "%Y-%m-%d")

    runner = SingleVideoTuningRunner(
        config=config,
        video_path=str(video_path),
        target_species=args.target_species,
        stop_confidence=args.stop_confidence,
        time_budget_s=args.time_budget_minutes * 60.0,
        results_dir=args.results_dir,
        max_trials=args.max_trials,
        video_date=video_date,
    )
    result = runner.run()

    best = result["best_trial"]
    print(json.dumps(result, indent=2))
    logger.info(
        "Best trial %s: %s %.1f%%, top result=%s %.1f%%, stop_reason=%s",
        best["trial_index"],
        best["target_species"],
        best["target_confidence"] * 100.0,
        best["top_species"],
        best["top_confidence"] * 100.0,
        result["stop_reason"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
