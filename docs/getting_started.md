# Getting Started

A practical walkthrough from a fresh `git clone` to your first identified
bird. This doc assumes you've already decided which target you want — a
desktop webapp, a Raspberry Pi 5 backyard camera, or the phone-streaming
sidecar — and walks through each.

If you're just curious about the project, the [README](../README.md)
covers what it does and the [project timeline](project_timeline.md)
covers how it was built.

---

## 1. Pick your target

| Target | What you need | When to pick it |
|---|---|---|
| **Desktop webapp** | NVIDIA GPU + Docker | You want a browser UI to upload videos and photos and get results |
| **Pi backyard mode** | Raspberry Pi 5 + Hailo-8 + USB capture card | You want live ID from a permanent camera feed |
| **Pi sidecar mode** | Raspberry Pi 5 + Hailo-8 + phone | You want to point a phone at birds and see results in real time |

The rest of this doc has one section per target. Start with the desktop
webapp if you're not sure — it's the lowest-friction way to see the
pipeline work.

---

## 2. Desktop webapp

### 2.1 Prerequisites

- Linux host with an NVIDIA GPU
- Docker + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 10 GB free disk for model weights and cached uploads
- (Optional) A Google Cloud project with an OAuth client if you want
  login-gated uploads. See [docs/google_oauth_setup.md](google_oauth_setup.md).

### 2.2 First boot

```bash
git clone https://github.com/evanwtf/birdvision.git
cd birdvision
cp config.yaml.example config.yaml
docker compose build
docker compose up
```

Open `http://localhost:3587`. The first request triggers model downloads
(BioCLIP + the ensemble classifier, ~5 GB total). They cache to `./models/`
and reuse across restarts.

Upload a video or photo. You should see a job appear, transition from
`pending` → `running` → `complete`, and land on a results page with
species predictions, confidence scores, annotated crops, and links to
Cornell All About Birds.

### 2.3 Adding OAuth (optional)

By default the upload route is open. For multi-user deploys you want
Google OAuth.

1. Follow [docs/google_oauth_setup.md](google_oauth_setup.md) to mint a
   client ID + secret in Google Cloud.
2. Store them in [1Password](https://1password.com) (recommended) or your
   secret manager of choice. The repo includes
   [`.env.op.example`](../.env.op.example) for the `op run` pattern.
3. Add allowed uploader emails to `auth.allowed_emails` in `config.yaml`.
4. Launch with `op run --env-file .env.op.example -- docker compose up`.

### 2.4 The HTTP API

Once OAuth is set up, the `/api/v1/videos` endpoint lets external clients
post motion-event clips. Token-based auth. See the
[HTTP API](../README.md#http-api) section in the README for the request
shape and an example.

---

## 3. Pi backyard mode

Real-time inference from a permanent camera. The Pi reads HDMI through a
USB capture card and writes detections to a Touch Display 2 overlay (or
runs headless).

### 3.1 Hardware

- Raspberry Pi 5, 8 GB RAM, aarch64 Ubuntu 24.04
- Hailo-8 AI accelerator (PCIe M.2) — included in the CanaKit AI kits
- USB capture card (e.g. Elgato Cam Link 4K)
- (Optional) Pi Touch Display 2 — 5" portrait

### 3.2 Setup

```bash
git clone https://github.com/evanwtf/birdvision.git
cd birdvision
cp config.pi.yaml.example config.pi.yaml
```

Drop the Hailo packages into `pi/deps/` (they aren't on PyPI — download
from [hailo.ai/developer-zone](https://hailo.ai/developer-zone/)):

- `hailort_4.23.0_arm64.deb`
- `hailort-4.23.0-…aarch64.whl`

Drop the compiled models into `pi/models/`:

- `yolov8n.hef` — bird detector
- `efficientnet_s_birds.hef` — species classifier
- `species_labels.json` — class index → species name

The trained EfficientNet weights and HEF are published at
[huggingface.co/k10z/birdvision-efficientnet-s](https://huggingface.co/k10z/birdvision-efficientnet-s).

Then:

```bash
docker compose -f docker-compose.pi.yml --profile backyard up
```

For deeper detail (model retraining, Hailo DFC compilation), see
[docs/pi_pipeline.md](pi_pipeline.md).

---

## 4. Pi sidecar mode

Phone browser streams camera frames to the Pi over HTTPS/WebSocket; the
Pi runs the same Hailo pipeline and pushes detections back. No native
app.

Same hardware as backyard mode minus the capture card and display.

```bash
git clone https://github.com/evanwtf/birdvision.git
cd birdvision
cp config.pi.sidecar.yaml.example config.pi.sidecar.yaml
docker compose -f docker-compose.pi.yml --profile sidecar up
```

The Pi serves an HTTPS page on port 8765 via Caddy (self-signed cert,
phone shows a one-time warning). Phone visits
`https://<pi-ip>/` → grants camera permission → starts streaming.
Detections render as boxes + species labels on the phone view.

---

## 5. Where things go on disk

| Path | Contents | Persists across restarts? |
|---|---|---|
| `./videos/assets/` | Content-addressed (sha256) ingested clips | Yes |
| `./videos/asset_index.json` | Asset metadata index | Yes |
| `./results/` | Per-job JSON + crops + annotated stills | Yes |
| `./models/` | Cached model weights | Yes |
| `./logs/retraining/` | Training script output | Yes |
| `./data/ebird_priors.db` | Built at image-build time from `ebird_data/` | Rebuilt on `docker compose build` |
| `pi/models/` | HEF + label artifacts (Pi only) | Yes |
| `pi/deps/` | HailoRT `.whl` + `.deb` | Yes |

Nothing in those paths is gitignored data you'd want to back up unless
the source clips themselves matter to you.

---

## 6. Troubleshooting

- **`docker compose up` fails with "config.yaml not found":** you skipped
  step 2.2 — `cp config.yaml.example config.yaml`.
- **Model downloads stall on first request:** check `./models/` exists and
  is writable; the container runs as a non-root user.
- **OAuth callback returns "redirect_uri_mismatch":** the URI you
  authorized in Google Cloud must exactly match `auth.redirect_uri` in
  `config.yaml`, scheme and trailing slash included.
- **Pi container can't access `/dev/hailo0`:** confirm `hailort` is
  loaded (`lsmod | grep hailo`) and the device exists; the compose file
  passes it through explicitly.
- **Coverage gate fails on a PR:** run `uv run pytest --cov=src
  --cov-report=term-missing` locally to see which file dropped coverage.

For anything else, the issue tracker is the right place.
