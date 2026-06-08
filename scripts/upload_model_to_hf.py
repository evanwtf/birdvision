"""Upload BirdVision EfficientNet-S model artifacts to Hugging Face Hub.

Uploads:
  - efficientnet_s_birds.onnx       (main inference artifact)
  - species_labels.json             (class index → species name)
  - efficientnet_s_birds_best.pt    (best PyTorch checkpoint, for resuming)
  - efficientnet_s_birds_phase1.pt  (phase 1 checkpoint, optional)

Creates the repo if it doesn't exist.  Run once after training completes.

Usage:
    uv run --no-project --with huggingface_hub \\
        scripts/upload_model_to_hf.py \\
        --model-dir ./pi/models \\
        --repo evandhoffman/birdvision-efficientnet-s

You'll be prompted to log in if no HF token is found.  Or set HF_TOKEN env var.
"""

import argparse
import json
import logging
import os
from pathlib import Path

from log_utils import add_logging_args, configure_logging

logger = logging.getLogger(__name__)

MODEL_CARD_TEMPLATE = """---
license: cc-by-nc-4.0
tags:
  - image-classification
  - birds
  - efficientnet
  - onnx
  - wildlife
datasets:
  - inaturalist
language: []
---

# BirdVision — EfficientNet-V2-S Bird Species Classifier

Fine-tuned [EfficientNet-V2-S](https://arxiv.org/abs/2104.00298) for bird species
classification across {num_classes} North American species (Northeast / Long Island focus).

Part of the [BirdVision](https://github.com/evandhoffman/birdvision) project —
real-time bird species identification from video using a Raspberry Pi 5 + Hailo-8
AI accelerator.

## Model details

| | |
|---|---|
| Base model | EfficientNet-V2-S (ImageNet-1K pretrained) |
| Input | 224×224 RGB, ImageNet normalization |
| Output | {num_classes}-class softmax logits |
| Training data | iNaturalist research-grade observations, New York state |
| Training images | ~{approx_images} photos across {num_classes} species |
| Val top-1 accuracy | {top1} |
| Val top-5 accuracy | {top5} |

## Training

Two-phase fine-tune on an NVIDIA RTX 3080 Ti:
- **Phase 1** (5 epochs, head only): frozen backbone, LR=1e-3
- **Phase 2** (15 epochs, full): all layers unfrozen, LR=5e-5, cosine annealing

Augmentation: random resized crop, horizontal flip, rotation ±20°, color jitter.

## Usage

```python
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download

# Load model and labels
onnx_path = hf_hub_download("{repo_id}", "efficientnet_s_birds.onnx")
labels_path = hf_hub_download("{repo_id}", "species_labels.json")

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
species = json.loads(open(labels_path).read())

# Preprocess image (224×224, ImageNet normalization)
def preprocess(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None]  # NCHW

# Run inference
logits = session.run(None, {{"input": preprocess("bird.jpg")}})[0][0]
top5 = np.argsort(logits)[::-1][:5]
for i in top5:
    print(f"{{species[i]:40s}} {{logits[i]:.3f}}")
```

## Species list

{num_classes} species — Northeast North America focus (Long Island / NY area).
See `species_labels.json` for the full list.

## Hailo-8 HEF (Raspberry Pi 5)

A compiled `efficientnet_s_birds.hef` for the [Hailo-8](https://hailo.ai/products/hailo-8/)
AI accelerator is included in this repo.

Benchmark on Raspberry Pi 5 (HailoRT 4.23.0):
- **22.3 FPS** hardware throughput
- **43.7 ms** hardware latency
- 4 contexts, 8 clusters

## License

Model weights derived from iNaturalist training data licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) —
**non-commercial use only**.
"""


def make_model_card(repo_id: str, labels: list[str], metrics: dict) -> str:
    approx_images = f"{len(labels) * 400:,}"  # rough estimate
    return MODEL_CARD_TEMPLATE.format(
        repo_id=repo_id,
        num_classes=len(labels),
        approx_images=approx_images,
        top1=metrics.get("top1", "see training logs"),
        top5=metrics.get("top5", "see training logs"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload BirdVision model to Hugging Face Hub")
    parser.add_argument("--model-dir", type=Path, default=Path("pi/models"))
    parser.add_argument(
        "--repo",
        default="k10z/birdvision-efficientnet-s",
        help="HF repo id (default: k10z/birdvision-efficientnet-s)",
    )
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument("--top1", type=float, help="Val top-1 accuracy to include in model card")
    parser.add_argument("--top5", type=float, help="Val top-5 accuracy to include in model card")
    add_logging_args(parser)
    args = parser.parse_args()

    log_path = configure_logging(
        "upload_model_to_hf",
        log_file=args.log_file,
        log_dir=args.log_dir,
    )
    if log_path:
        logger.info("File logging enabled: %s", log_path)

    from huggingface_hub import HfApi, login

    token = os.environ.get("HF_TOKEN")
    if not token:
        login()  # interactive prompt
    else:
        login(token=token)

    api = HfApi()

    # Create repo if needed
    try:
        api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
        logger.info("Repo ready: https://huggingface.co/%s", args.repo)
    except Exception as exc:
        logger.error("Failed to create repo: %s", exc)
        raise

    # Files to upload
    uploads = [
        ("efficientnet_s_birds.onnx", "efficientnet_s_birds.onnx"),
        ("efficientnet_s_birds.hef", "efficientnet_s_birds.hef"),
        ("species_labels.json", "species_labels.json"),
        ("efficientnet_s_birds_best.pt", "efficientnet_s_birds_best.pt"),
    ]
    # Phase 1 checkpoint is optional
    phase1 = args.model_dir / "efficientnet_s_birds_phase1.pt"
    if phase1.exists():
        uploads.append(("efficientnet_s_birds_phase1.pt", "efficientnet_s_birds_phase1.pt"))

    for filename, path_in_repo in uploads:
        local = args.model_dir / filename
        if not local.exists():
            logger.warning("Skipping missing file: %s", local)
            continue
        size_mb = local.stat().st_size / 1e6
        logger.info("Uploading %s (%.1f MB) → %s", filename, size_mb, path_in_repo)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            repo_id=args.repo,
            repo_type="model",
        )
        logger.info("  ✓ %s", filename)

    # Generate and upload model card
    labels_path = args.model_dir / "species_labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else []
    metrics = {}
    if args.top1:
        metrics["top1"] = f"{args.top1:.1%}"
    if args.top5:
        metrics["top5"] = f"{args.top5:.1%}"

    card = make_model_card(args.repo, labels, metrics)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )
    logger.info("Model card uploaded.")
    logger.info("Done: https://huggingface.co/%s", args.repo)


if __name__ == "__main__":
    main()
