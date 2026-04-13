#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

LOG_DIR="${BIRDVISION_SCRIPT_LOG_DIR:-$REPO_DIR/logs/retraining}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/verify_efficientnet_onnx_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "Log file:   $LOG_FILE"
echo "Tail with:  tail -f $LOG_FILE"

uv run --no-project \
    --with onnxruntime \
    --with numpy \
    "$SCRIPT_DIR/verify_efficientnet_onnx.py" \
    --log-file "$LOG_FILE" \
    "$@"
