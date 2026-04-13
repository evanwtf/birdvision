#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${1:-$REPO_DIR/train_data}"
OUTPUT_DIR="${2:-$REPO_DIR/pi/models}"
LOG_DIR="${BIRDVISION_SCRIPT_LOG_DIR:-$REPO_DIR/logs/retraining}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_efficientnet_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "Data dir:   $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Log file:   $LOG_FILE"
echo "Tail with:  tail -f $LOG_FILE"

uv run --no-project \
    --with torch \
    --with torchvision \
    --with tqdm \
    --with onnxscript \
    "$SCRIPT_DIR/train_efficientnet.py" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --log-file "$LOG_FILE" \
    "${@:3}"
