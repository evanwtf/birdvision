# Getting Started

A practical walkthrough from a fresh `git clone` to the first bird ID on
your phone. The primary deployment is **Pi sidecar mode** — phone is the
camera, Pi does the inference, results stream back. Other deployments
are covered in §3 and §4.

If you're just curious about the project, the [README](../README.md)
covers what it does and [`docs/project_timeline.md`](project_timeline.md)
covers how it was built.

---

## 0. Bill of materials

Everything you need to gather **before** starting. The hardware
arrives in days; the software downloads take minutes; the models are
~30 MB.

### Hardware (sidecar mode — minimum)

| Item | Approx. price | Source | Notes |
|---|---|---|---|
| Raspberry Pi 5 8 GB AI Kit (26 TOPS) | $379 | [CanaKit](https://www.canakit.com/raspberry-pi-5-8gb-quick-start-kit-ai-128gb-26t.html) | Includes Pi 5, Hailo-8 M.2 accelerator, 128 GB storage, case with active cooling, PSU |
| Phone with camera | $0 | already own | iOS Safari 14.5+ or Android Chrome — anything from the last ~5 years |
| WiFi network | $0 | already own | Pi and phone must be on the same LAN |

That's it. The Pi 5 AI Kit is the only meaningful purchase.

### Hardware (backyard mode — adds the following)

| Item | Approx. price | Source | Notes |
|---|---|---|---|
| HDMI → USB capture card | ~$120 | Elgato Cam Link 4K or equivalent | YUYV 4:2:2 / NV12 at 1080p60 |
| Camera with HDMI out | varies | camcorder, DSLR, action cam | Any camera that does live HDMI output |
| Pi Touch Display 2 (optional) | $52 | [Raspberry Pi store](https://www.raspberrypi.com/products/touch-display-2/) | For live overlay; not required if running headless |

### Hardware (desktop webapp — separate machine)

| Item | Notes |
|---|---|
| NVIDIA GPU + Linux | tested on RTX 3080 Ti / Ubuntu; anything with `nvidia-container-toolkit` support |

### Software — installed on the Pi

| Package | Source | Notes |
|---|---|---|
| Ubuntu 24.04 aarch64 | [raspberrypi.com](https://www.raspberrypi.com/software/operating-systems/) | Pi 5 official image |
| Docker + Docker Compose | `apt install docker.io docker-compose-v2` | Standard |
| HailoRT 4.23.0 `.deb` | [hailo.ai/developer-zone](https://hailo.ai/developer-zone/) → SW Downloads → HailoRT | Free account required; not on PyPI |
| HailoRT 4.23.0 `.whl` (aarch64) | same | Match your Python version (cp312 / cp313) |

### Software — installed on your laptop / dev machine

| Package | Why | Install |
|---|---|---|
| `git` | clone the repo | already have |
| `ssh` | deploy to the Pi | already have |
| Hugging Face CLI (`hf`) | download the models | `uv tool install huggingface-hub` |
| (Optional) `op` CLI | resolve secrets at deploy time | `brew install --cask 1password-cli` |

### Models — downloaded from Hugging Face

All three live in
[`k10z/birdvision-efficientnet-s`](https://huggingface.co/k10z/birdvision-efficientnet-s).

| File | Size | Purpose |
|---|---|---|
| `yolov8n.hef` | 4 MB | Bird detector (Hailo-8 INT8) |
| `efficientnet_s_birds.hef` | 23 MB | Species classifier (237 species, Hailo-8 INT8) |
| `species_labels.json` | 5 KB | Class index → species name map |

One `hf download` command fetches all three (see §1.3).

---

## 1. Pi sidecar mode (phone streaming, recommended)

### 1.1 Prepare the Pi

Flash Ubuntu 24.04 aarch64 to the storage, boot, set a hostname (this
guide assumes `birdvision-pi`), and verify the Hailo PCIe driver loaded:

```bash
ls -l /dev/hailo0
# crw-rw-rw- 1 root root 234, 0 ... /dev/hailo0
```

Install Docker:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
# log out and back in for the group change to take effect
```

### 1.2 Clone the repo

```bash
git clone https://github.com/evanwtf/birdvision.git
cd birdvision
cp config.pi.sidecar.yaml.example config.pi.sidecar.yaml
```

Edit `config.pi.sidecar.yaml` and set `metadata.latitude` /
`metadata.longitude` to your camera location (used for eBird priors).

### 1.3 Drop in the HailoRT packages and models

Download HailoRT from
[hailo.ai/developer-zone](https://hailo.ai/developer-zone/) → SW
Downloads → HailoRT 4.23.0:

```bash
# On the Pi, in the repo root:
cp ~/Downloads/hailort_4.23.0_arm64.deb pi/deps/
cp ~/Downloads/hailort-4.23.0-cp312-cp312-linux_aarch64.whl pi/deps/
# (Use the cp312 or cp313 wheel matching your Python.)
```

Download the models from Hugging Face (the `hf` CLI is the cleanest
path; alternatively you can drag-drop from the HF web UI):

```bash
uv tool install huggingface-hub   # one-time, installs `hf`
hf download k10z/birdvision-efficientnet-s \
  yolov8n.hef efficientnet_s_birds.hef species_labels.json \
  --local-dir pi/models
```

### 1.4 Build and launch

```bash
docker compose -f docker-compose.pi.yml build
docker compose -f docker-compose.pi.yml --profile sidecar up
```

The Pi serves an HTTPS page on port 8765 via a Caddy reverse proxy
(self-signed cert; phone shows a one-time warning).

### 1.5 Connect the phone

On any phone on the same WiFi: visit `https://<pi-ip>/` in Safari or
Chrome. Accept the cert warning once, then tap **Start Camera** and
grant camera access. Bird detections render as overlay boxes with
species labels, in real time.

That's the whole thing.

---

## 2. What's happening under the hood

When the phone streams a frame to the Pi:

1. The browser captures a frame from `getUserMedia`, JPEG-encodes it,
   and ships it over the WebSocket.
2. The Pi decodes the JPEG, runs YOLOv8n on Hailo (~5 ms), tracks
   detections across recent frames, and every Nth frame runs the
   EfficientNet classifier on each crop (~44 ms).
3. eBird priors re-rank predictions by what's actually plausible for
   the location and week.
4. Boxes + species labels go back to the phone, which overlays them
   on the live preview.

End-to-end latency target: well under one frame at 10–15 FPS on the
phone side.

---

## 3. Pi backyard mode (USB capture, optional)

Same Pi, no phone in the loop. The Pi reads HDMI through a USB capture
card connected to a permanent camera, runs the same pipeline, and
optionally writes a live overlay to a Pi Touch Display 2.

```bash
cp config.pi.yaml.example config.pi.yaml
# edit metadata.latitude/longitude and stream.device if your capture
# card isn't at /dev/video0
docker compose -f docker-compose.pi.yml --profile backyard up
```

You can't run backyard and sidecar at the same time — both need
exclusive access to the Hailo chip.

See [`docs/pi_pipeline.md`](pi_pipeline.md) for model retraining,
Hailo DFC compilation, and the deeper hardware notes.

---

## 4. Desktop webapp (NVIDIA GPU, optional)

Browser UI for batch-processing video and photos. Detection runs on
YOLOv8 (PyTorch), classification on BioCLIP (zero-shot) with an
optional secondary HuggingFace classifier in an ensemble. Includes a
token-authenticated JSON API (`POST /api/v1/videos`) for motion-event
cameras that want to post clips directly.

```bash
git clone https://github.com/evanwtf/birdvision.git
cd birdvision
cp config.yaml.example config.yaml
docker compose build
docker compose up
```

First request triggers ~5 GB of model downloads (BioCLIP + the
ensemble classifier). Models cache to `./models/` and reuse across
restarts. Open `http://localhost:3587` to upload your first clip.

For Google OAuth login, see [`docs/google_oauth_setup.md`](google_oauth_setup.md)
and the [Secrets section](../README.md#secrets) in the README.

---

## 5. Where things go on disk

| Path | Contents |
|---|---|
| `pi/models/` | HEFs + species label map (downloaded in §1.3) |
| `pi/deps/` | HailoRT `.deb` + `.whl` (placed in §1.3) |
| `./results/` | Per-job JSON + crops + annotated stills (backyard / desktop) |
| `./videos/assets/` | Content-addressed uploaded clips (desktop) |
| `./models/` | Cached PyTorch / BioCLIP weights (desktop) |
| `./data/ebird_priors.db` | Built at image-build time from `ebird_data/` |

All gitignored.

---

## 6. Troubleshooting

- **`docker compose up` fails with "config.pi.sidecar.yaml not found":**
  you skipped §1.2. Copy from the `.example`.
- **Browser warns about the cert:** expected — Caddy issues a
  self-signed cert on demand. Accept it once.
- **Pi container can't open `/dev/hailo0`:** confirm `hailort` loaded
  (`lsmod | grep hailo`) and that you placed the `.deb` + `.whl` in
  `pi/deps/` before the Docker build.
- **`hf download` fails with 401:** the model files in
  `k10z/birdvision-efficientnet-s` are public; if `hf` is asking for
  auth it likely means you previously logged in with an expired
  token. Run `hf auth logout` or just delete `~/.cache/huggingface/token`.
- **Backyard mode says `/dev/video0` not found:** confirm the USB
  capture card is connected and recognized (`v4l2-ctl
  --list-devices`).
- **Coverage gate fails on a PR (dev only):** run `uv run pytest
  --cov=src --cov-report=term-missing` locally to see which file
  dropped coverage.

For anything else, the issue tracker is the right place.
