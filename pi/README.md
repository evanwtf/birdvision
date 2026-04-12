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
| Hailo PCIe kernel driver | ✓ present | `/dev/hailo0` exists |
| V4L2 (Cam Link 4K) | ✓ present | `/dev/video0` exists |

Everything else (HailoRT library, Python bindings, app code) runs in the container.

## Models

Place compiled HEF files in `pi/models/` (gitignored):

| File | Description | Source |
|---|---|---|
| `yolov8n_hailo8.hef` | YOLOv8n bird detector | Compiled per #74 |
| `efficientnet_s_birds_hailo8.hef` | EfficientNet-S classifier | Compiled per #77 |
| `species_labels.json` | Species label order (must match training) | Produced per #75 |

## Python setup

Python dependencies are managed with `uv`. Pi-only packages (HailoRT bindings) live in
the `pi` dependency group and must not be installed on x86_64:

```bash
# On the Pi only:
uv sync --group pi
```

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
