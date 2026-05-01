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
| Framebuffer (Touch Display 2) | optional | `/dev/fb0`; set `display.enabled: false` for headless |

Everything else (HailoRT library, Python bindings, app code) runs in the container.

---

## Docker setup (recommended)

### Step 1 — Place assets

Three sets of files must be present before building the image. None are committed to git.

#### 1a. HailoRT packages

The HailoRT packages are not on PyPI. Download both from:

> https://hailo.ai/developer-zone/ → SW Downloads → HailoRT

Place both files in `pi/deps/`:

```
pi/deps/hailort_4.23.0_arm64.deb             ← system runtime library (libhailort.so)
pi/deps/hailort-4.23.0-cp3XX-cp3XX-linux_aarch64.whl  ← Python bindings
```

Select the `.whl` matching your Python version (cp312 or cp313).
Only one `.whl` file should be in `pi/deps/` at a time.

#### 1b. EfficientNet-S classifier HEF + labels

Download from HuggingFace (run from the repo root):

```bash
pip install huggingface_hub   # or: uv pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('k10z/birdvision-efficientnet-s', 'efficientnet_s_birds.hef', local_dir='pi/models')
hf_hub_download('k10z/birdvision-efficientnet-s', 'species_labels.json', local_dir='pi/models')
"
```

#### 1c. YOLOv8n detector HEF

The YOLOv8n HEF is not on HuggingFace. Copy it from a previous build or recompile:

```bash
# If you have a previous build on this machine:
cp ~/yolov8n.hef pi/models/yolov8n.hef

# To recompile from scratch (x86_64 desktop, Hailo DFC 3.33.1 required):
hailomz compile yolov8n --hw-arch hailo8 --calib-path ~/calib_images/
cp yolov8n.hef pi/models/yolov8n.hef
```

See `pi/hailo_hef_compile_status.md` for full compilation notes.

#### Asset checklist

Before building, verify all required files are in place:

```
pi/
  deps/
    hailort_4.23.0_arm64.deb             ← from hailo.ai/developer-zone
    hailort-4.23.0-*-linux_aarch64.whl   ← from hailo.ai/developer-zone
  models/
    yolov8n.hef                           ← compiled (see above)
    efficientnet_s_birds.hef              ← from HuggingFace
    species_labels.json                   ← from HuggingFace
```

### Step 2 — Verify device permissions

Check the permissions on the Pi:

```bash
ls -l /dev/hailo0 /dev/video0
```

Expected (confirmed on this hardware):
```
crw-rw-rw- 1 root root  234, 0  /dev/hailo0   ← world r/w, no group needed
crw-rw---- 1 root video  81, 19  /dev/video0   ← video group required
crw-rw---- 1 root video  29,  0  /dev/fb0     ← video group (Touch Display 2 framebuffer)
```

`docker-compose.pi.yml` adds the container user to the `video` group, which
covers both `/dev/video0` and `/dev/fb0`. Since `/dev/hailo0` is world-accessible,
no extra group is needed for Hailo. If your setup differs, adjust `group_add` in
the compose file.

If the Pi has no Touch Display 2 attached, set `display.enabled: false` in
`config.pi.yaml`; the container will still run and `/dev/fb0` passthrough is
harmless when the node is absent (compose will fail at startup though — comment
out the `/dev/fb0` line for headless hosts).

### Step 3 — Build and run

```bash
# Build the image (run from repo root)
docker compose -f docker-compose.pi.yml build
```

The Pi compose file defines two services behind separate profiles. Only one
can run at a time — both require exclusive access to the Hailo-8 chip.

#### Backyard mode (USB camera)

Captures live video from the Elgato Cam Link 4K (`/dev/video0`) connected to
a camcorder pointed at a bird feeder. Detections and species predictions are
logged to the console and written to `./results/`. Optionally overlays
bounding boxes on a Pi Touch Display 2 (`/dev/fb0`).

```bash
docker compose -f docker-compose.pi.yml --profile backyard up
```

Config: `config.pi.yaml` (bind-mounted read-only).

#### Sidecar mode (phone camera via WebSocket)

No USB camera needed. A phone opens `http://<pi-ip>:8765/` in its browser,
streams camera frames over WebSocket to the Pi for inference, and receives
bounding boxes + species labels overlaid on the live preview. Works on
iPhone Safari and Android Chrome.

```bash
docker compose -f docker-compose.pi.yml --profile sidecar up
```

Config: `config.pi.sidecar.yaml` (bind-mounted as `config.pi.yaml` inside
the container).

The sidecar auto-generates a self-signed TLS certificate on first start
(persisted in `./certs/`). On the phone, open **`https://<pi-ip>:8765/`**
(note: HTTPS, not HTTP). Safari will show a certificate warning — tap
"Show Details" → "visit this website" to accept it once. After that, tap
"Start Camera" and allow camera access when prompted.

#### Common commands

```bash
# Run in background
docker compose -f docker-compose.pi.yml --profile backyard up -d

# Follow logs
docker compose -f docker-compose.pi.yml --profile backyard logs -f

# Stop
docker compose -f docker-compose.pi.yml --profile backyard down
```

Results are written to `./results/` on the host.

---

## Models

| File | Description | Benchmark | Source |
|---|---|---|---|
| `pi/models/yolov8n.hef` | YOLOv8n detector, Hailo-8 INT8 | 212 FPS | Compiled per #74 |
| `pi/models/efficientnet_s_birds.hef` | EfficientNet-V2-S classifier, 237 species | 22 FPS / 44ms | HuggingFace k10z/birdvision-efficientnet-s |
| `pi/models/species_labels.json` | 237-entry class index (must match HEF) | — | HuggingFace k10z/birdvision-efficientnet-s |

---

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

---

## Running (bare metal, for development)

Install Pi-only packages on the Pi first:

```bash
# Install hailort wheel directly (not via uv sync — it's not on PyPI)
uv pip install pi/deps/hailort-4.23.0-*-linux_aarch64.whl

# Run
uv run scripts/realtime_identify.py --config config.pi.yaml

# Debug logging
uv run scripts/realtime_identify.py --config config.pi.yaml --debug
```

---

## Config

Two Pi configs exist — one per mode. Never merge keys into `config.yaml` (webapp config).

| File | Mode | Key difference |
|---|---|---|
| `config.pi.yaml` | Backyard | `stream.source: v4l2` — reads from Cam Link 4K |
| `config.pi.sidecar.yaml` | Sidecar | `stream.source: websocket` — receives frames over WebSocket on port 8765 |

Key sections (both files):
- `stream:` — source type, V4L2 device or WebSocket host/port
- `detector:` — YOLOv8n HEF path, confidence threshold
- `classifier:` — EfficientNet HEF path, labels path, classify cadence
- `tracker:` — IoU/centroid thresholds, disappear timeout
- `metadata:` — eBird priors DB, FIPS code, lat/lon, prior mode
- `display:` — framebuffer overlay (backyard only; disabled in sidecar)
- `output:` — results dir, log interval
