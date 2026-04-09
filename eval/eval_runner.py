#!/usr/bin/env python3
"""
BirdVision model comparison runner.

For each clip that has already been processed by BioCLIP:
  1. Extract BioCLIP track predictions from the existing results JSON and
     save them as an eval sidecar so the report generator has a uniform
     data format across all models.
  2. For each additional enabled model, load the track crop images and run
     inference, saving a sidecar JSON per model.

Sidecar format (eval/<asset_sha>_<model_id>.json):
  {
    "model_id": "bioclip",
    "model_label": "BioCLIP",
    "asset_sha": "<sha256>",
    "source_results_file": "<basename>",
    "tracks": [
      {
        "track_id": 0,
        "crop_file": "track_0.jpg",          # relative to <asset_sha>_crops/
        "top_species": [
          {"species": "House Sparrow", "score": 0.6415},
          ...
        ]
      }
    ]
  }

Usage:
    uv run eval/eval_runner.py --config eval/config.yaml
    uv run eval/eval_runner.py --config eval/config.yaml --max-clips 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_runner")

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Classifier backends
# ---------------------------------------------------------------------------

class BioCLIPBackend:
    """Reads predictions from an existing results JSON; no GPU inference."""

    def __init__(self, cfg: dict):
        pass

    def classify_crops(self, crops_dir: Path, crop_files: list[str]) -> list[list[tuple[str, float]]]:
        raise RuntimeError("BioCLIP backend uses existing results; call extract_bioclip_tracks instead.")


def _load_classifier_backend(model_cfg: dict):
    backend = model_cfg.get("backend", "bioclip")
    if backend == "bioclip":
        return BioCLIPBackend(model_cfg)
    if backend == "gemma4":
        try:
            from src.gemma_classifier import GemmaClassifier
        except ImportError:
            logger.warning(
                "gemma_classifier module not found — Gemma 4 backend requires issue #58 to be merged. "
                "Skipping model: %s", model_cfg.get("id")
            )
            return None
        # Species list is loaded separately; classifier is initialized lazily
        return model_cfg
    raise ValueError(f"Unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# Sidecar helpers
# ---------------------------------------------------------------------------

def sidecar_path(eval_dir: Path, asset_sha: str, model_id: str) -> Path:
    return eval_dir / f"{asset_sha}_{model_id}.json"


def write_sidecar(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    logger.debug("Wrote sidecar: %s", path)


def read_sidecar(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


# ---------------------------------------------------------------------------
# Extract BioCLIP tracks from existing results JSON
# ---------------------------------------------------------------------------

def extract_bioclip_tracks(result: dict) -> list[dict]:
    """Pull per-track predictions out of an existing pipeline results JSON."""
    tracks_out = []
    for track in result.get("tracks", []):
        avg_preds = track.get("averaged_predictions", [])
        top_species = [
            {"species": p["species"], "score": round(p["probability"], 4)}
            for p in avg_preds[:10]
        ]
        if not top_species:
            continue
        tracks_out.append({
            "track_id": track.get("track_id", 0),
            "crop_file": track.get("crop", ""),
            "top_species": top_species,
        })
    return tracks_out


# ---------------------------------------------------------------------------
# Gemma inference on annotated stills
# ---------------------------------------------------------------------------

def _run_gemma_on_stills(
    result: dict,
    crops_dir: Path,
    classifier,
    model_cfg: dict,
) -> list[dict]:
    """
    For each detection in video_stills, load the annotated full frame, draw a
    bright red highlight box around the specific detection, and run Gemma.
    Returns a list of track-sidecar dicts (one per detection).

    Falls back to track crops if no video_stills are present.
    """
    import cv2

    tracks_out = []
    video_stills = result.get("video_stills", [])

    if not video_stills:
        # Fallback: use track crops (old behaviour)
        for track in result.get("tracks", []):
            crop_file = track.get("crop", "")
            crop_path = crops_dir / crop_file
            if not crop_path.exists():
                continue
            frame_bgr = cv2.imread(str(crop_path))
            if frame_bgr is None:
                continue
            scores = classifier.classify(frame_bgr)
            top_species = [
                {"species": sp, "score": round(sc, 4)}
                for sp, sc in sorted(scores.items(), key=lambda x: -x[1])
                if sc > 0
            ][:10]
            tracks_out.append({
                "track_id": track.get("track_id", 0),
                "crop_file": crop_file,
                "top_species": top_species,
            })
        return tracks_out

    seen_track_ids: set[int] = set()

    for still in video_stills:
        annotated_file = still.get("annotated_file", "")
        still_path = crops_dir / annotated_file
        if not annotated_file or not still_path.exists():
            continue

        base_frame = cv2.imread(str(still_path))
        if base_frame is None:
            logger.warning("Could not read still: %s", still_path)
            continue

        for det in still.get("detections", []):
            # Use detection_index as a proxy for track_id when track_id absent
            track_id = det.get("track_id", det.get("detection_index", 0))
            if track_id in seen_track_ids:
                continue  # already have a result for this track
            seen_track_ids.add(track_id)

            bbox = det.get("bbox")  # [x1, y1, x2, y2] in pixels
            if not bbox or len(bbox) != 4:
                continue

            # Draw a thick red box around the specific bird we want identified
            frame = base_frame.copy()
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

            scores = classifier.classify(frame)
            top_species = [
                {"species": sp, "score": round(sc, 4)}
                for sp, sc in sorted(scores.items(), key=lambda x: -x[1])
                if sc > 0
            ][:10]

            tracks_out.append({
                "track_id": track_id,
                "crop_file": annotated_file,
                "top_species": top_species,
            })

    return tracks_out


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_results(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*_results.json"))


def asset_sha_from_result(result: dict) -> str | None:
    records = result.get("asset_records", [])
    if records:
        return records[0].get("sha256")
    # Fallback: derive from video path basename
    video = result.get("video", "")
    if video:
        stem = Path(video).stem
        if len(stem) == 64:  # SHA256 hex
            return stem
    return None


def run(cfg: dict, max_clips: int | None = None) -> None:
    results_dir = Path(cfg["results_dir"])
    report_dir = Path(cfg["report_dir"])
    eval_dir = report_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    skip_no_detections = cfg.get("skip_no_detections", True)
    model_cfgs = cfg.get("models", [])

    result_files = find_results(results_dir)
    logger.info("Found %d result files in %s", len(result_files), results_dir)

    if skip_no_detections:
        result_files = [
            f for f in result_files
            if json.loads(f.read_text()).get("video_predictions")
        ]
        logger.info("%d results have bird detections (skip_no_detections=true)", len(result_files))

    if max_clips is not None:
        result_files = result_files[:max_clips]
        logger.info("Capped to %d clips (--max-clips)", max_clips)

    # ---- BioCLIP pass: extract from existing JSONs -------------------------
    bioclip_cfg = next((m for m in model_cfgs if m.get("backend") == "bioclip"), None)
    if bioclip_cfg:
        model_id = bioclip_cfg["id"]
        label = bioclip_cfg.get("label", model_id)
        logger.info("Extracting BioCLIP predictions from existing results...")
        extracted = 0
        for result_file in result_files:
            result = json.loads(result_file.read_text())
            asset_sha = asset_sha_from_result(result)
            if not asset_sha:
                logger.warning("Could not determine asset SHA for %s — skipping", result_file.name)
                continue

            sc_path = sidecar_path(eval_dir, asset_sha, model_id)
            if sc_path.exists() and not bioclip_cfg.get("rerun", False):
                continue

            tracks = extract_bioclip_tracks(result)
            if not tracks:
                continue

            write_sidecar(sc_path, {
                "model_id": model_id,
                "model_label": label,
                "asset_sha": asset_sha,
                "source_results_file": result_file.name,
                "tracks": tracks,
            })
            extracted += 1

        logger.info("BioCLIP: wrote/verified %d sidecars", extracted)

    # ---- Additional model passes -------------------------------------------
    for model_cfg in model_cfgs:
        if model_cfg.get("backend") == "bioclip":
            continue
        if not model_cfg.get("enabled", True):
            logger.info("Model %r is disabled in config — skipping", model_cfg.get("id"))
            continue

        model_id = model_cfg["id"]
        label = model_cfg.get("label", model_id)
        logger.info("Running model: %s (%s)", label, model_id)

        backend = _load_classifier_backend(model_cfg)
        if backend is None:
            continue

        # Determine which clips still need inference
        pending = []
        for result_file in result_files:
            result = json.loads(result_file.read_text())
            asset_sha = asset_sha_from_result(result)
            if not asset_sha:
                continue
            sc_path = sidecar_path(eval_dir, asset_sha, model_id)
            if not sc_path.exists():
                pending.append((result_file, result, asset_sha))

        logger.info("%s: %d clips need inference", model_id, len(pending))

        if not pending:
            continue

        # Lazy-init Gemma classifier
        if model_cfg.get("backend") == "gemma4":
            from src.gemma_classifier import GemmaClassifier
            species_file = Path(
                cfg.get("species_list_file")
                or Path(__file__).parent.parent / "data" / "species_lists" / "north_america_common.txt"
            )
            all_species = [l.strip() for l in species_file.read_text().splitlines() if l.strip()]
            logger.info("Loaded %d species from %s", len(all_species), species_file)
            classifier = GemmaClassifier(
                model_name=model_cfg["model"],
                species_list=all_species,
                location_hint=model_cfg.get("location_hint", ""),
            )

        for result_file, result, asset_sha in pending:
            crops_dir = results_dir / f"{asset_sha}_crops"
            if not crops_dir.exists():
                logger.debug("No crops dir for %s — skipping %s", asset_sha[:16], model_id)
                continue

            if model_cfg.get("backend") == "gemma4":
                tracks_out = _run_gemma_on_stills(
                    result, crops_dir, classifier, model_cfg,
                )
            else:
                logger.error("Unhandled backend: %s", model_cfg.get("backend"))
                continue

            if tracks_out:
                sc_path = sidecar_path(eval_dir, asset_sha, model_id)
                write_sidecar(sc_path, {
                    "model_id": model_id,
                    "model_label": label,
                    "asset_sha": asset_sha,
                    "source_results_file": result_file.name,
                    "tracks": tracks_out,
                })

        logger.info("%s: done", model_id)

    logger.info("Eval runner complete. Sidecars in: %s", eval_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdVision model comparison runner")
    parser.add_argument("--config", default="eval/config.yaml")
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    max_clips = args.max_clips or cfg.get("max_clips")
    run(cfg, max_clips=max_clips)


if __name__ == "__main__":
    main()
