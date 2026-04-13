# BirdVision — Raspberry Pi 5 Real-Time Pipeline

## Hardware

- Raspberry Pi 5 (8GB RAM, aarch64 Ubuntu 24.04)
- Hailo-8 AI Processor (PCIe, 26 TOPS INT8) — `/dev/hailo0`
- Elgato Cam Link 4K (HDMI→USB capture) — `/dev/video0`
- Samsung camcorder via HDMI → Cam Link

## Capture device

```
v4l2-ctl --list-formats-ext -d /dev/video0
```

Supported formats: YUYV 4:2:2, NV12, YU12 — all at 1920×1080 @ 59.94 fps. No MJPEG.

Test a single frame:

```bash
ffmpeg -f v4l2 -input_format yuyv422 -video_size 1920x1080 -framerate 60 \
  -i /dev/video0 -frames:v 1 -update 1 test_frame.jpg
```

## Host requirements

Only two things must be installed on the Pi host (not in the container):

| Requirement | Status | Notes |
|---|---|---|
| Hailo PCIe kernel driver | present | `/dev/hailo0` exists |
| V4L2 (Cam Link 4K) | present | `/dev/video0` exists |

Everything else (HailoRT library, Python bindings, app code) runs in the container.

## Models

HEF files are gitignored. Download from HuggingFace or recompile from source:

| File | Description | Benchmark | Source |
|---|---|---|---|
| `pi/models/yolov8n.hef` | YOLOv8n detector, Hailo-8 INT8 | 212 FPS | Compiled per #74 |
| `pi/models/efficientnet_s_birds.hef` | EfficientNet-V2-S classifier, 237 species | 22 FPS / 44ms | Retrained after #75 and recompiled |
| `pi/models/species_labels.json` | 237-entry class index (must match training) | — | Produced by latest retrain |

Download from HuggingFace:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('k10z/birdvision-efficientnet-s', 'efficientnet_s_birds.hef', local_dir='pi/models')
hf_hub_download('k10z/birdvision-efficientnet-s', 'species_labels.json', local_dir='pi/models')
"
```

The YOLOv8n HEF is not on HuggingFace — recompile using `scripts/compile_yolov8n_hef.sh`
or retrieve from a previous build.

## Compilation (on desktop x86_64)

See `pi/hailo_hef_compile_status.md` for full notes on both model compilations.

**YOLOv8n HEF** — compiled via `hailomz compile yolov8n --hw-arch hailo8 --calib-path ~/calib_images/`

**EfficientNet-S HEF:**
```bash
# 1. Train (produces pi/models/efficientnet_s_birds.onnx + species_labels.json)
./scripts/run_training.sh

# 2. Verify ONNX export
./scripts/run_verify_efficientnet_onnx.sh

# 3. Compile to HEF
./scripts/run_compile_efficientnet_hef.sh \
    --onnx pi/models/efficientnet_s_birds.onnx \
    --train-dir train_data \
    --output pi/models/efficientnet_s_birds.hef
```

These wrapper scripts write timestamped logs under `logs/retraining/` and print
the exact `tail -f` command at startup.

Requires Hailo Dataflow Compiler 3.33.1 (x86_64 only, from https://hailo.ai/developer-zone/).

## Python setup

Pi-only packages (HailoRT Python bindings) live in the `pi` dependency group:

```bash
# On the Pi only:
uv sync --group pi
```

Do not add Pi packages to `[project.dependencies]` — `hailort` is not on PyPI
and will break `uv sync` on any non-Pi machine.

## Running (Docker)

```bash
docker compose -f docker-compose.pi.yml up
```

Requires `/dev/hailo0` and `/dev/video0` on the host.

## Running (bare metal, for development)

```bash
uv run scripts/realtime_identify.py --config config.pi.yaml
```

## Config

`config.pi.yaml` — Pi-specific settings (stream device, HEF paths, Hailo device).
Never merge keys into `config.yaml` (webapp config).
