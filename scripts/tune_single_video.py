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
from typing import Any

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


def _set_nested(config: dict, dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _format_diff_lines(original: dict[str, Any], best: dict[str, Any]) -> list[str]:
    lines = []
    for key in original:
        original_value = original[key]
        best_value = best.get(key)
        if original_value != best_value:
            lines.append(f"  {key}: {original_value} -> {best_value}")
    return lines


def _prompt_apply_config_updates(config_path: Path, raw_config: dict, best_values: dict[str, Any], diff_lines: list[str]) -> None:
    if not diff_lines:
        print("\nParameter diff: no tuning-space changes from baseline.")
        return

    print("\nParameter diff:")
    for line in diff_lines:
        print(line)

    if not sys.stdin.isatty():
        print("\nInteractive config update skipped because stdin is not a TTY.")
        return
    if not config_path.exists():
        print(f"\nInteractive config update skipped because {config_path} does not exist.")
        return

    response = input(f"\nUpdate {config_path} in place with these best values? [y/N] ").strip().lower()
    if response not in {"y", "yes"}:
        print("Config left unchanged.")
        return

    updated_config = json.loads(json.dumps(raw_config))
    for key, value in best_values.items():
        _set_nested(updated_config, key, value)

    with open(config_path, "w") as f:
        yaml.safe_dump(updated_config, f, sort_keys=False)
    print(f"Updated {config_path} with best tuning values.")


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
        "--success-species-contains",
        help="Optional case-insensitive species-name substring that counts for stop-rule success, for example Gull",
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

    raw_config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            raw_config = yaml.safe_load(f) or {}
    else:
        logger.warning("Config file not found: %s; using defaults", args.config)
    config = normalize_local_paths(raw_config)

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
        success_species_contains=args.success_species_contains,
        stop_confidence=args.stop_confidence,
        time_budget_s=args.time_budget_minutes * 60.0,
        results_dir=args.results_dir,
        max_trials=args.max_trials,
        video_date=video_date,
    )
    result = runner.run()

    best = result["best_trial"]
    baseline = runner.trials[0].tuned_params
    best_values = best["tuned_params"]
    diff_lines = _format_diff_lines(baseline, best_values)

    print(json.dumps(result, indent=2))
    _prompt_apply_config_updates(config_path, raw_config, best_values, diff_lines)
    logger.info(
        "Best trial %s: %s %.1f%%, success=%s %.1f%%, top result=%s %.1f%%, stop_reason=%s",
        best["trial_index"],
        best["target_species"],
        best["target_confidence"] * 100.0,
        best["success_species"] or "none",
        best["success_confidence"] * 100.0,
        best["top_species"],
        best["top_confidence"] * 100.0,
        result["stop_reason"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
