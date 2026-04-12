#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${1:-$REPO_DIR/train_data}"
OUTPUT_DIR="${2:-$REPO_DIR/pi/models}"

echo "Data dir:   $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"

uv run --no-project \
    --with torch \
    --with torchvision \
    --with tqdm \
    --with onnxscript \
    "$SCRIPT_DIR/train_efficientnet.py" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "${@:3}"
