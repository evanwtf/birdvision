# BirdVision — Raspberry Pi / Hailo-8 (`pi/`)

Deployment assets for the Pi 5 real-time pipeline: proprietary HailoRT
packages (`pi/deps/`), compiled models (`pi/models/`), setup docs, and a
display test harness. Pipeline code itself lives in `src/` and `scripts/`
and is governed by the root `AGENTS.md` (import boundaries, config
separation, style). This file covers the Pi container, its assets, and HEF
compilation.

## Environment & Dependencies

- Hardware: Pi 5 (8 GB, aarch64 Ubuntu 24.04), Hailo-8 PCIe (`/dev/hailo0`),
  Elgato Cam Link 4K (`/dev/video0` + `/dev/video1`; YUYV/NV12/YU12 at
  1080p, no MJPEG), optional Touch Display 2 framebuffer (`/dev/fb0`).
- `pi/deps/` (gitignored; download from hailo.ai/developer-zone):
  `hailort_4.23.0_arm64.deb` plus exactly one
  `hailort-4.23.0-*-linux_aarch64.whl`. The container venv is Python 3.13,
  so the image build needs the cp313 wheel.
- `pi/models/` (gitignored): `yolov8n.hef`, `efficientnet_s_birds.hef`,
  `species_labels.json` (237 classes; labels must match the HEF). Fetch:
  `hf download k10z/birdvision-efficientnet-s yolov8n.hef efficientnet_s_birds.hef species_labels.json --local-dir pi/models`
- `/dev/video0` and `/dev/fb0` need group `video` (compose `group_add`
  covers it); `/dev/hailo0` is world-readable/writable.

## Commands

```bash
docker compose -f docker-compose.pi.yml build                  # from repo root
docker compose -f docker-compose.pi.yml --profile sidecar up   # primary mode
docker compose -f docker-compose.pi.yml --profile backyard up  # USB capture
./scripts/autostart.sh install   # systemd unit wrapping the Pi compose file

# Bare metal (dev) — the hailort wheel installs directly, never via uv sync
uv pip install pi/deps/hailort-4.23.0-*-linux_aarch64.whl
uv run scripts/realtime_identify.py --config config.pi.yaml [--debug]
```

`pi/display_test/` is a standalone framebuffer plumbing test with its own
Dockerfile and compose file — no Hailo, no ML.

## Compiling HEFs (x86_64 Linux only)

- Requires Hailo Dataflow Compiler 3.33.1. ONNX must be exported with
  `dynamo=False` (legacy TorchScript exporter); calibration data must be
  NHWC `(N, 224, 224, 3)` float32.
- YOLOv8n: `hailomz compile yolov8n --hw-arch hailo8 --calib-path <dir>`.
  The model-zoo wheel is missing postprocess configs — use a git editable
  install of `hailo_model_zoo` v2.18 (see `pi/hailo_hef_compile_status.md`).
- EfficientNet-S: `./scripts/run_training.sh`, then
  `./scripts/run_verify_efficientnet_onnx.sh`, then
  `./scripts/run_compile_efficientnet_hef.sh --onnx … --train-dir … --output …`.
  These wrappers log to timestamped files under `logs/retraining/` —
  `tail -f` that file, not the launching terminal.

## Guardrails

- Run only one compose profile at a time — backyard and sidecar both need
  exclusive Hailo-8 access.
- Sidecar bind-mounts `config.pi.sidecar.yaml` as `/app/config.pi.yaml`
  in-container; phones require the Caddy HTTPS front (`Caddyfile.sidecar`)
  for browser camera access.
- `Dockerfile.pi` installs the HailoRT `.deb` with `dpkg --unpack` because
  the post-install script calls `systemctl` and fails in containers — keep it.
- HEF files are bind-mounted at runtime, never baked into the image.
- Headless host (no Touch Display 2): set `display.enabled: false` in
  `config.pi.yaml` AND remove the `/dev/fb0` device line from the compose
  file, or container startup fails.

## Agent Notes

Symlinked as `CLAUDE.md` and `GEMINI.md`; keep instructions tool-neutral.
