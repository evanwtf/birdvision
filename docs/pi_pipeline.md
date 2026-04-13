# BirdVision Pi Pipeline — Build Summary

This document captures what was built to make the Raspberry Pi 5 + Hailo-8
real-time pipeline work end-to-end, the key engineering decisions made along
the way, and the problems that had to be solved. Released as **v0.2.0**.

---

## What was built

A complete real-time bird identification system running entirely on a Raspberry
Pi 5, using a Hailo-8 AI accelerator for hardware-accelerated inference and an
Elgato Cam Link 4K for HDMI video capture.

**Verified live results (2026-04-13):**
```
top_species=Mourning Dove confidence=0.63 tracks=1 fps=27.8
top_species=Mourning Dove confidence=0.57 tracks=1 fps=27.6
top_species=Mourning Dove confidence=0.45 tracks=1 fps=27.9
```

---

## New files

### `src/hailo_detector.py`

YOLOv8n bird detection via Hailo-8 HEF. Key design points:

- Defines `Detection` dataclass locally — does **not** import `detector.py`,
  which would transitively pull in `ultralytics` (not installed on Pi)
- Accepts a `vdevice=` parameter so the caller can pass in a shared `VDevice`;
  creates its own only if not provided
- YOLOv8n NMS output format from Hailo is unusual: the output tensor is a
  Python list-of-1 wrapping 80 per-class arrays, each shaped `(N, 5)` with
  columns `[y1, x1, y2, x2, score]` in normalized `[0, 1]` coordinates.
  Bird class index is 14 (COCO). The parser unwraps `output[tensor_name][0]`,
  indexes by `BIRD_CLASS_ID`, and converts normalized coords to pixel space.

### `src/stream_capture.py`

V4L2 live frame source for the Elgato Cam Link 4K. Key points:

- Cam Link 4K does not support MJPEG — YUYV 4:2:2 only at 1920×1080 @ 59.94fps
- Uses `cv2.VideoCapture` with `cv2.CAP_V4L2` backend and explicit YUYV fourcc
- Yields `(frame_no, bgr_array)` tuples; OpenCV handles YUYV→BGR conversion
- `stop()` sets a flag checked at the top of the frame loop for clean shutdown

### `src/realtime_pipeline.py`

Main orchestrator. Key design points:

- Creates a **single** `VDevice` and passes it to both detector and classifier.
  The Hailo-8 chip can only be opened once per process; opening it twice raises
  `HAILO_OUT_OF_PHYSICAL_DEVICES`. Both models run as separate network groups
  on the same device handle.
- Classifies every `classify_every_n_frames` (default: 10) per tracked bird,
  not every frame — keeps inference load well within the 22 FPS classifier budget
- Applies eBird seasonal priors + local feeder overrides via `MetadataPrior`
  (same class as the desktop pipeline, `.apply()` method)
- Logs a 1-second summary: best species + confidence seen in the window, active
  track count, FPS
- Logs system health every `stats_interval_seconds` (default: 30) via psutil:
  CPU temp, load average, utilization, frequency, memory, fan RPM
- Handles `SIGINT` and `SIGTERM` for clean Docker shutdown

### `scripts/realtime_identify.py`

Entry point. Loads `config.pi.yaml`, instantiates `RealtimePipeline`, calls
`pipeline.run()`. Supports `--debug` flag for `DEBUG`-level logging.

### `Dockerfile.pi` + `docker-compose.pi.yml`

ARM64 Docker image. Key engineering decisions:

- Base: `ubuntu:24.04` (not Chainguard) — HailoRT `.deb` needs `apt` runtime
  libraries and its post-install script uses `systemctl`
- Python 3.13 via `uv python install 3.13` with `UV_PYTHON_INSTALL_DIR=/opt/uv-python`.
  ubuntu:24.04 ships Python 3.12; the HailoRT wheel is `cp313`-only.
- The uv-managed Python dir is COPYed from builder to runtime at the **same
  path** so venv symlinks resolve correctly
- HailoRT `.deb` installed via `dpkg --unpack` + manual `ldconfig` to avoid
  running the post-install script (which calls `systemctl`, absent in containers)
- Runtime user UID 65532 (not 1000 — the HailoRT `.deb` creates a user at
  UID 1000 during unpack, making that UID unavailable)
- `PYTHONPATH=/app` required because `scripts/` runs directly rather than via
  `uv run` with the project installed
- Device passthrough: `/dev/hailo0` (world-accessible, no group needed),
  `/dev/video0` (needs `video` group — added via `group_add`)
- Sysfs volumes for system stats: `/sys/class/thermal` and `/sys/class/hwmon`
  bind-mounted read-only so psutil can read temperature and fan data

---

## Problems solved

| Problem | Root cause | Fix |
|---|---|---|
| `cp313` wheel, `cp312` Python | ubuntu:24.04 ships 3.12; HailoRT wheel requires 3.13 | `uv python install 3.13` + copy managed Python to runtime |
| `dpkg` post-install exits 127 | Post-install script calls `systemctl`, not in containers | `dpkg --unpack` + `ldconfig` instead of `dpkg -i` |
| UID 1000 conflict | HailoRT `.deb` creates a system user at UID 1000 during unpack | Use UID 65532 for `nonroot` |
| `ModuleNotFoundError: src` | `PYTHONPATH` not set in container | `ENV PYTHONPATH=/app` |
| `ModuleNotFoundError: ultralytics` | `hailo_detector.py` originally imported `detector.py` which imports ultralytics at module level | Define `Detection` inline in `hailo_detector.py` |
| `HAILO_OUT_OF_PHYSICAL_DEVICES` | Detector and classifier each called `VDevice()` | Create one `VDevice` in `RealtimePipeline`, pass via `vdevice=` |
| `AttributeError: list has no .shape` | Assumed Hailo NMS output was ndarray; it's a Python list | Updated parser to unwrap the list structure |
| `ValueError: inhomogeneous shape (80,)` | Tried to stack 80 per-class arrays of different lengths | Index by `BIRD_CLASS_ID` first, then parse that array only |
| `C_CONTIGUOUS` OpenCV warnings | OpenCV crop slices produce non-contiguous arrays | `np.ascontiguousarray(batch)` before inference |

---

## Performance (Raspberry Pi 5, as measured)

| Stage | Rate |
|---|---|
| V4L2 capture | 59.94 fps |
| YOLOv8n detection | 212 FPS (Hailo-8) |
| EfficientNet-S classification | 22 FPS / 44ms (Hailo-8) |
| End-to-end pipeline | ~27–34 FPS |
| System temperature at load | ~70°C |
| Memory usage | ~15% of 8GB |

The pipeline runs comfortably within thermal and memory limits. The fan reaches
~7600 RPM under sustained load and keeps the CPU at or below 70°C.

---

## Configuration (`config.pi.yaml`)

Key sections and their purpose:

```yaml
stream:
  device: /dev/video0
  format: yuyv422
  width: 1920
  height: 1080
  framerate: 59.94

detector:
  hef: /app/pi/models/yolov8n.hef
  confidence: 0.4

classifier:
  hef: /app/pi/models/efficientnet_s_birds.hef
  labels: /app/pi/models/species_labels.json
  top_k: 20
  classify_every_n_frames: 10   # classify each track at most once per 10 frames

tracker:
  max_disappeared: 120          # frames before a track is dropped
  iou_threshold: 0.2
  centroid_max_distance: 0.18   # fallback match radius (fraction of frame diagonal)

metadata:
  ebird_db: /app/data/ebird_priors.db
  ebird_fips: US-NY-059         # Nassau County fallback
  prior_mode: seasonal          # per-week eBird frequencies
  local_priors_file: /app/data/local_priors.yaml   # backyard feeder overrides

output:
  results_dir: /data/results
  log_interval_seconds: 1
  stats_interval_seconds: 30
```

---

## What's next

### Raspberry Pi Touch Display

The next hardware addition is a Raspberry Pi official Touch Display (7" or
larger). Goals:

- Show a live detection overlay: bounding boxes, track IDs, species label +
  confidence — without routing video through the Pi's CPU for every frame
- Render a persistent sidebar showing the best-confidence species seen in the
  last N seconds / minutes
- Potentially show JPEG crops of recently classified birds

**Open questions to investigate:**

1. **Display connection** — the official Pi Touch Display connects via DSI
   ribbon cable to the Pi 5's DSI port. Pi 5 has two DSI connectors; confirm
   which one is the primary. A larger HDMI monitor is an alternative if DSI
   is problematic with the current camera setup.

2. **Rendering approach** — options in rough order of complexity:
   - **OpenCV `imshow`** with `DISPLAY=:0` — simplest; requires an X server
     or Wayland compositor running on the Pi. Works if the Pi is running a
     desktop environment.
   - **DRM/KMS framebuffer** — render directly to `/dev/dri/card0` without a
     compositor; more complex but no desktop environment needed.
   - **Lightweight GUI** — `pygame`, `tkinter`, or `kivy` window displaying
     JPEG crops and overlaid text; friendlier than raw DRM.
   - **Web UI served from Pi** — FastAPI serving a small page with auto-refresh
     or WebSocket push; viewable on any device on the local network as well as
     the touch display via a kiosk browser.

3. **Docker vs. bare metal for display** — running a GUI app inside Docker
   requires passing through the display socket (`-e DISPLAY -v /tmp/.X11-unix`)
   or using DRM device passthrough (`/dev/dri/card0`). May be simpler to run
   the display component outside Docker while keeping inference containerized.

4. **Frame rate for display** — the display doesn't need to update at 30fps.
   A 5–10fps annotation overlay is smooth enough and avoids burning CPU on
   rendering. The pipeline's frame loop can push frames to a display thread
   at a reduced rate.

**Suggested approach to start:**

Run a minimal Wayland/X11 session on the Pi host, pass `DISPLAY` through to
the container, and add an optional OpenCV `imshow` display mode activated by a
`--show` flag in `realtime_identify.py`. That gives a fast path to something
working without committing to a specific UI framework.
