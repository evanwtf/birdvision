"""Compile efficientnet_s_birds.onnx to Hailo HEF using the Dataflow Compiler.

Requires the Hailo Dataflow Compiler (x86_64 only) — install from:
  https://hailo.ai/developer-zone/

DFC version must match HailoRT on the Pi (4.23.0).

Workflow:
  1. Sample calibration images from train_data (or a custom dir)
  2. Translate ONNX → Hailo native model
  3. Optimize (INT8 quantization with calibration data)
  4. Compile → HEF

Usage:
    uv run --no-project --with numpy --with pillow --with tqdm \\
        scripts/compile_efficientnet_hef.py \\
        --onnx pi/models/efficientnet_s_birds.onnx \\
        --train-dir train_data \\
        --output pi/models/efficientnet_s_birds.hef
"""

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from log_utils import add_logging_args, configure_logging, estimate_remaining, format_duration

logger = logging.getLogger(__name__)

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(path: Path) -> np.ndarray:
    """Load and preprocess one image to float32 HWC, ImageNet-normalized.

    DFC expects calibration data in NHWC format (H, W, C).
    """
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0         # HWC [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD             # normalize, stay HWC
    return arr


def sample_calibration_images(
    train_dir: Path,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Sample images from train_data and return float32 array (N, C, H, W)."""
    all_images: list[Path] = []
    for cls_dir in train_dir.iterdir():
        if cls_dir.is_dir():
            all_images.extend(cls_dir.glob("*.jpg"))

    random.seed(seed)
    chosen = random.sample(all_images, min(n_samples, len(all_images)))
    logger.info("Sampling %d calibration images from %d total", len(chosen), len(all_images))

    arrays = []
    failed = 0
    start_time = time.time()
    for path in tqdm(chosen, desc="Preprocessing calibration images"):
        try:
            arrays.append(preprocess_image(path))
        except Exception as exc:
            logger.debug("Skipping %s: %s", path, exc)
            failed += 1
        processed = len(arrays) + failed
        if processed and (processed % 50 == 0 or processed == len(chosen)):
            elapsed = time.time() - start_time
            eta = estimate_remaining(elapsed, processed, len(chosen))
            logger.info(
                "Calibration preprocessing %d/%d  elapsed=%s  eta=%s",
                processed,
                len(chosen),
                format_duration(elapsed),
                format_duration(eta),
            )

    if failed:
        logger.warning("Skipped %d images during calibration preprocessing", failed)

    calib = np.stack(arrays)  # (N, 3, 224, 224)
    logger.info("Calibration dataset: shape=%s  dtype=%s  min=%.3f  max=%.3f",
                calib.shape, calib.dtype, calib.min(), calib.max())
    return calib


def compile_hef(onnx_path: Path, calib_data: np.ndarray, output_path: Path) -> None:
    """Translate, optimize, and compile ONNX → HEF using the Hailo DFC Python SDK."""
    try:
        from hailo_sdk_client import ClientRunner
    except ImportError:
        raise SystemExit(
            "hailo_sdk_client not found — install the Hailo Dataflow Compiler from "
            "https://hailo.ai/developer-zone/ (x86_64 only, requires account)"
        )

    logger.info("Initializing DFC for hailo8...")
    runner = ClientRunner(hw_arch="hailo8")

    logger.info("Translating ONNX: %s", onnx_path)
    runner.translate_onnx_model(
        str(onnx_path),
        "efficientnet_s_birds",
        start_node_names=["input"],
        end_node_names=["output"],
        net_input_shapes={"input": [1, 3, IMG_SIZE, IMG_SIZE]},
    )
    logger.info("Translation complete.")

    # Save HAR (intermediate artifact) alongside the HEF
    har_path = output_path.with_suffix(".har")
    runner.save_har(str(har_path))
    logger.info("HAR saved: %s", har_path)

    logger.info("Optimizing (INT8 quantization) with %d calibration samples...", len(calib_data))
    runner.optimize(calib_data)
    logger.info("Optimization complete.")

    logger.info("Compiling to HEF...")
    hef_bytes = runner.compile()

    output_path.write_bytes(hef_bytes)
    size_mb = output_path.stat().st_size / 1e6
    logger.info("HEF written: %s (%.1f MB)", output_path, size_mb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile EfficientNet-S ONNX to Hailo HEF")
    parser.add_argument("--onnx", type=Path, default=Path("pi/models/efficientnet_s_birds.onnx"))
    parser.add_argument("--output", type=Path, default=Path("pi/models/efficientnet_s_birds.hef"))
    parser.add_argument("--train-dir", type=Path, default=Path("train_data"),
                        help="Training data directory to sample calibration images from")
    parser.add_argument("--calib-npy", type=Path,
                        help="Pre-computed calibration .npy file (N,3,224,224) — skips sampling")
    parser.add_argument("--n-calib", type=int, default=500,
                        help="Number of calibration images to sample (default: 500)")
    parser.add_argument("--save-calib", type=Path,
                        help="Save calibration numpy array to this path for reuse")
    add_logging_args(parser)
    args = parser.parse_args()

    log_path = configure_logging(
        "compile_efficientnet_hef",
        log_file=args.log_file,
        log_dir=args.log_dir,
    )
    if log_path:
        logger.info("File logging enabled: %s", log_path)

    if not args.onnx.exists():
        raise SystemExit(f"ONNX model not found: {args.onnx}")

    # Calibration data
    if args.calib_npy:
        logger.info("Loading pre-computed calibration data: %s", args.calib_npy)
        calib_data = np.load(str(args.calib_npy))
    else:
        if not args.train_dir.exists():
            raise SystemExit(f"Training data directory not found: {args.train_dir}")
        calib_data = sample_calibration_images(args.train_dir, args.n_calib)
        if args.save_calib:
            np.save(str(args.save_calib), calib_data)
            logger.info("Calibration data saved to %s for reuse", args.save_calib)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    compile_hef(args.onnx, calib_data, args.output)
    logger.info("Done. Run on Pi to verify:")
    logger.info("  hailortcli benchmark --hef %s", args.output)


if __name__ == "__main__":
    main()
