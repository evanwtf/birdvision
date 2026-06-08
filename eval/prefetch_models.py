#!/usr/bin/env python3
"""
Pre-download all models required by the configured eval backends.

Run this before eval_runner.py to ensure all models are present in the
local cache (HF_HOME / TORCH_HOME). Safe to re-run — already-cached
models are verified without re-downloading.

Usage:
    uv run eval/prefetch_models.py --config eval/config.yaml
    uv run eval/prefetch_models.py --config eval/config-local.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prefetch_models")

sys.path.insert(0, str(Path(__file__).parent.parent))


def prefetch_bioclip(model_name: str) -> None:
    logger.info("Checking BioCLIP model: %s", model_name)
    import open_clip

    # open_clip downloads to torch hub cache on create; this is the same call
    # the classifier makes, so the cache hit path is identical.
    model, _, _ = open_clip.create_model_and_transforms(model_name)
    del model
    logger.info("BioCLIP OK: %s", model_name)


def prefetch_gemma(model_name: str) -> None:
    logger.info("Checking Gemma model: %s  (this may download several GB)", model_name)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Download processor first (small) — confirms the model ID is valid before
    # we pull the weights.
    logger.info("Fetching processor...")
    AutoProcessor.from_pretrained(model_name)
    logger.info("Processor OK")

    # Pull weights. device_map="cpu" avoids requiring a GPU just for the cache
    # check, but still validates the full checkpoint.
    logger.info("Fetching model weights (device=cpu for cache verification)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    del model
    logger.info("Gemma OK: %s", model_name)


def prefetch_hf_image_classifier(model_name: str) -> None:
    logger.info("Checking HF image classifier: %s", model_name)
    from PIL import Image
    from transformers import pipeline as hf_pipeline

    pipe = hf_pipeline("image-classification", model=model_name, device=-1, top_k=1)
    pipe(Image.new("RGB", (224, 224)))
    del pipe
    logger.info("HF image classifier OK: %s", model_name)


def prefetch_yolo(model_path: str) -> None:
    """YOLO downloads yolov8n.pt (etc.) from ultralytics on first use."""
    from ultralytics import YOLO

    logger.info("Checking YOLO model: %s", model_path)
    YOLO(model_path)
    logger.info("YOLO OK: %s", model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download eval models")
    parser.add_argument("--config", default="eval/config.yaml")
    parser.add_argument(
        "--skip-bioclip",
        action="store_true",
        help="Skip BioCLIP check (not needed when reusing existing results JSONs)",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO check (not needed when reusing existing results JSONs)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfgs = cfg.get("models", [])
    errors = []

    for model_cfg in model_cfgs:
        if not model_cfg.get("enabled", True):
            logger.info("Model %r is disabled — skipping", model_cfg.get("id"))
            continue

        backend = model_cfg.get("backend", "bioclip")
        model_name = model_cfg.get("model", "hf-hub:imageomics/bioclip")

        try:
            if backend == "bioclip":
                if args.skip_bioclip:
                    logger.info("Skipping BioCLIP (--skip-bioclip)")
                    continue
                prefetch_bioclip(model_name)

            elif backend == "gemma4":
                prefetch_gemma(model_name)

            elif backend == "hf_image_classifier":
                prefetch_hf_image_classifier(model_name)

            else:
                logger.warning("Unknown backend %r — skipping prefetch", backend)

        except Exception as e:
            logger.error("Failed to prefetch %s (%s): %s", model_cfg.get("id"), backend, e)
            errors.append(model_cfg.get("id"))

    # Always check YOLO unless explicitly skipped, since future eval backends
    # may need to re-run detection on clips that have no existing results.
    if not args.skip_yolo:
        try:
            # Read YOLO model path from a parent config if available, else default
            yolo_model = "yolov8n.pt"
            prefetch_yolo(yolo_model)
        except Exception as e:
            logger.error("Failed to prefetch YOLO: %s", e)
            errors.append("yolo")

    if errors:
        logger.error("Prefetch failed for: %s", ", ".join(errors))
        sys.exit(1)

    logger.info("All models verified.")


if __name__ == "__main__":
    main()
