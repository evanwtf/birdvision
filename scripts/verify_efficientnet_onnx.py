"""Verify the exported EfficientNet ONNX model and class count."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from log_utils import add_logging_args, configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify EfficientNet ONNX export")
    parser.add_argument("--onnx", type=Path, default=Path("pi/models/efficientnet_s_birds.onnx"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("pi/models/species_labels.json"),
        help="Path to species_labels.json to verify output class count",
    )
    add_logging_args(parser)
    args = parser.parse_args()

    log_path = configure_logging(
        "verify_efficientnet_onnx",
        log_file=args.log_file,
        log_dir=args.log_dir,
    )
    if log_path:
        logger.info("File logging enabled: %s", log_path)

    if not args.onnx.exists():
        raise SystemExit(f"ONNX model not found: {args.onnx}")
    if not args.labels.exists():
        raise SystemExit(f"Species labels not found: {args.labels}")

    labels = json.loads(args.labels.read_text())
    logger.info("Loaded %d labels from %s", len(labels), args.labels)

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    output = session.run(None, {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})[0]
    logger.info("ONNX inference OK: shape=%s dtype=%s", tuple(output.shape), output.dtype)

    expected_shape = (1, len(labels))
    if tuple(output.shape) != expected_shape:
        raise SystemExit(f"Unexpected output shape: {tuple(output.shape)} != {expected_shape}")

    logger.info("Output shape matches species_labels.json (%d classes)", len(labels))


if __name__ == "__main__":
    main()
