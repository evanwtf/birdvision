# BirdVision — Project Timeline

A detailed outline of what got built and when, sourced from the full GitHub
issue history (88 issues) and git log (168 commits). This is raw material for
a blog post — structured chronologically, grouped by theme, with commit SHAs
and issue numbers so we can drill down later.

**Span:** 2026-03-30 through 2026-05-02 — 34 days.
88 issues opened, 63 closed, 18 PRs merged.

## Hardware used

**Desktop (webapp / training / Hailo HEF compilation)**
- NVIDIA RTX 3080 Ti, 12 GB VRAM
- AMD Ryzen 9 7900X
- 32 GB RAM
- x86_64 Ubuntu
- Location: Long Island / Nassau County, NY (40.7, -73.5)

**Raspberry Pi 5 (real-time edge pipeline)**
- **CanaKit Raspberry Pi 5 8GB Quick-Start AI Kit — 26 TOPS** (SKU `PI5-8GB-AI128-C4-WHT-26T`) — $379.95
  - Raspberry Pi 5, 8 GB RAM, aarch64 Ubuntu 24.04
  - Hailo-8 AI accelerator (PCIe M.2, 26 TOPS INT8). HailoRT firmware 4.23.0.
  - 128 GB storage, case with active cooling
- **Raspberry Pi Touch Display 2, 5" Portrait** (SKU `RSP-DISPLAY-V2-5`) — $52.95
- Elgato Cam Link 4K (HDMI→USB capture) reading a Samsung camcorder over HDMI
- Ordered Tue Apr 7, 2026 — total **$454.85** including shipping. Project
  started three days before the hardware arrived; code stack was ready when
  the box showed up.

## Trained model, published artifacts

- **Hugging Face repo:** https://huggingface.co/k10z/birdvision-efficientnet-s
- **GitHub repo:** https://github.com/evandhoffman/birdvision (private)
- Artifacts published to HF: `efficientnet_s_birds.onnx`, `efficientnet_s_birds.hef`
  (Hailo-compiled), `species_labels.json`, PyTorch best + phase-1 checkpoints,
  auto-generated model card with Pi benchmarks.
- License: **CC-BY-NC-4.0** — inherited from iNaturalist training data.
- HF username `k10z`; GitHub username `evandhoffman`.

---

## Act I — Day 1: From Nothing to a Working Pipeline (Sun Mar 30)

The most remarkable day of the project. A working detection + classification
pipeline, web UI, Docker packaging, eBird priors, a tuner, OAuth, photo
support, video stills, and a canonical asset store — all landed on March 30.
49 commits in a single day. 35 issues created, 26 closed same-day.

### Core pipeline stands up (Mar 30 ~01:00 UTC)
- `870d3dc` Initial framework: **YOLOv8 detection + BioCLIP zero-shot classification**
- `b88c6d8` Switch to **uv** + Dockerfile + docker-compose from day one
- `ec352a8` Strip model names from user-facing copy — describe capability, not implementation

### Web UI arrives same day (Mar 30 ~01:20–01:48 UTC)
- `2043d8e` FastAPI + Jinja upload/results UI
- Chainguard runtime wrestling: `a09c93d` apk perms, `0934a5e` `--chown` venv speedup,
  `98e3946` entrypoint override, `a348c86` `mesa-gl` for OpenCV
- `37ce4b1`/`bd5c9f9` writable model volume + host-mounted config
- `71820cc` split venv/source layers for fast rebuilds
- `d15b62c` TemplateResponse signature fix for Starlette 0.36+

### Tracking, results display, metadata (Mar 30 ~01:55–02:35 UTC)
- `df5c61a` Archive pruned tracks instead of deleting them — fixed zero-tracks summaries
- `48be974` Save and display best crop per track
- `14f3e52` Restore completed jobs from disk on startup (no DB — JSON on disk is state)
- `f30641f` Extract video metadata (date, GPS) + OSM links
- `2560a70` Link species to Cornell All About Birds

### First round of threshold tuning (Mar 30 ~02:40–02:52 UTC)
- `87aa7db` Raise det. conf 0.5, add `min_frames_to_report`
- `a7afadf` Tune to 0.4 / `min_frames=3`
- `e6f9869` Reprocess button — rerun pipeline on existing upload
- `587b59a` **Config hot-reload** — change thresholds without restart (issue #2)
- `d450a56` `min_confidence_to_report` — rescue high-conf short tracks
- `c2ce16a` **Center weighting** — Gaussian weight by bbox distance from frame center

### eBird priors (issue #1, Mar 30 ~03:00 UTC)
- `c9898f8` Raw eBird bar-chart TSVs committed
- `a087de6` Import pipeline: bar charts → SQLite; location + season priors
- `2dbd281` Per-track explanation: visual vs prior-adjusted scores (issue #11)

### Docs (Mar 30 ~03:10–04:00 UTC)
- `52166b4`/`e7b6d06`/`2dfedec`/`4c46505` README + CLAUDE.md + AGENTS.md — project context for future AI sessions

### More pipeline work (Mar 30 ~04:00–04:16 UTC)
- `9abe555` Improve tracking + classification filtering
- `c4ea020` Tune for mockingbird videos
- `7d081fb` Video-level species summary (issue #9's first pass)
- `639f90c` **Adaptive crop padding** — smaller/distant birds get more context (issue #4)

### Detector upgrade (Mar 30 ~13:27 UTC)
- `9a55be9` Switch detector yolov8n → yolov8s for better boxes (issue #3)

### Photo uploads (Mar 30 ~13:41–15:48 UTC, issues #13, #14, #15, #16, #17)
- `9107217` Photo uploads for multi-image classification
- `2bf7b59`/`1ddb70e` Smart-quote cleanups in templates
- `ea3aca5` Per-image metadata, merged priors, visual vs weighted tables
- `3590d39` Long Island eBird gating + better photo UI

### Upload review + canonical asset store (Mar 30 ~16:47–22:40 UTC, issues #22, #23, #24, #25)
- `8b8414f` **Content-addressed storage by sha256** — dedup across browser/API uploads
- `ccdd267` Multi-video uploads; remove manual date input
- `4191c50` Select All / None / New controls
- `2959a78` Warn on Safari-incompatible codecs
- `a53fe62` Collapsed "no bird" photo view

### Tuner + video stills (Mar 30 ~18:16–21:54 UTC, issues #27, #29, #35)
- `d01d98d` Single-video tuner + compose service
- `5c8ff80` Gull baseline tuning
- `3750151` Annotated video stills with bboxes
- `0ccc69e` Fix startup job restoration
- `727978b` Auto-refresh while jobs pending/running
- `a931c30` Embed video on job detail page
- `0c59ea6` Camera info in metadata display

### Auth (Mar 30 ~23:19–23:56 UTC, issues #36, #37, #38, #39)
- `51e194f` Google OAuth2 flow + setup docs
- `6a474b8` OAuth + themes + request logging
- `a0636f8` v0.1.0 docs

---

## Act II — Day 2: Tests, Pagination, CI (Mon Mar 31)

4 commits. Testing harness, pagination, and CI — the "make it real" day.

- `4e07631` Docs for auth/themes/logging (Mar 31 00:02 UTC)
- `497bc37` **pytest harness + unit tests** across pipeline/metadata/tracker/webapp (Mar 31 03:54 UTC, issues #40–#44)
- `5b3541e` Job list pagination, media type labels, mixed uploads (Mar 31 03:54 UTC, issue #45)
- `eb088f9` GitHub Actions CI on push/PR (Mar 31 10:11 UTC)

---

## Act III — Apr 3–5: Polish Sprint (Thu–Sat)

Issues #46–#50 created Apr 3, all closed by Apr 5. Product polish — making
it a real site, not a demo. 4 PRs merged on Apr 5 (PR #51–#54).

- `e70663e` More allowlisted emails (Apr 5 14:58 UTC)
- `037495f` Sort job listing by created_at desc (Apr 5 15:08 UTC, issue #46)
- `80c7560` **Open Graph share cards** on job pages (Apr 5 15:12 UTC, issue #47, PR #51)
- `8ad9913` **Friendly slug URLs** with bare-ID redirects (Apr 5, issue #48, PR #52)
- `8b899d3` **Location-only prior mode** — no seasonal weighting (Apr 5, issue #49, PR #53)
- `cd5de19` Persist + display submitter email (Apr 5, issue #50, PR #54)
- `7645bd1` Switch CI to manual dispatch only (Apr 5 15:27 UTC)

---

## Act IV — Apr 7–11: API, Eval Harness, Ensemble, Gemma (Mon–Fri)

Shifts from product features toward model quality and comparison infrastructure.

### API ingest (Apr 7–8, issue #55, PR #57)
- `82a312a` **`POST /api/v1/videos`** with `X-API-Token` header (merged Apr 8)

### Model comparison eval container (Apr 8–9, issues #58, #59, #60, PR #62)
- `d0a1d29` Eval container (merged Apr 9 01:49 UTC)
- Includes: prefetch models compose service, eval docs, ENTRYPOINT fix,
  transformers + accelerate for Gemma, chmod 777 for nonroot user

### Gemma 4 vision-language classifier saga (Apr 8–9, issue #58)
- `ba440c2` First cut — `GemmaClassifier` wired into eval runner
- `c3147fa` Strip special tokens; fix generation_config deprecation
- `dde089c`/`74b...`/`e6d3a83` Fight with `generation_config` max_length/warnings
- `4fba1d0` 4-bit quantization to fit 12GB VRAM
- `68ec428` OOM fix: `device_map` in model_kwargs so pipeline doesn't re-`.to(device)`
- `59054c8`/`05b9dfc` Simplify prompt; filter warnings via `logging.Filter`
- `e7146ab` Tighten prompt: regional framing + North American common name
- `48825c1` Feed Gemma **full annotated still with red highlight box** — better than raw crop

### Ensemble + local priors (Apr 9, issues #61, #65, PRs #63, #64, #65)
- `7e8e5e4` EfficientNet-BIRDS525 backend + **weighted geometric-mean ensemble** (PR #63, merged Apr 9 02:24 UTC)
- `0f3d2af` **User-defined local species priors** on top of eBird (PR #64, merged Apr 9 02:40 UTC)
- `34e4607` Per-model score breakdown in results (PR #65, merged Apr 9 10:19 UTC)
- `3fbab64`/`551928b` config wiring
- `ade83ff` iOS job list layout tightening (issue #66, PR #68, merged Apr 9 23:02 UTC)

### Cleanup (Apr 10–11)
- `3117c4a` Refresh AGENTS.md + README (Apr 10)
- `a9d8348` Reject + clean up invalid uploads (PR #69, merged Apr 11 20:12 UTC)

---

## Act V — Apr 12–14: The Raspberry Pi Subproject (Sat–Mon)

The big one. Parent issue **#70** — "Adapt project for real-time streaming on Pi."
14 sub-issues (#71–#86) created Apr 12, nearly all closed by Apr 14.
Three days from blank scaffold to 30 FPS end-to-end bird ID on a Pi 5 +
framebuffer display overlay.

### Apr 12 — hardware setup + training (issues #71–#75, #81, #82)
- `ded283e` "Recent Jobs" → "Recent Visitors" + listing thumbnails (Apr 12 15:41 UTC)
- `6747823` Scaffold Pi sub-project (#81) — separate `Dockerfile.pi`, `docker-compose.pi.yml`, `config.pi.yaml` (Apr 12 17:50 UTC)
- `fa2d41b`/`b5b5ccb`/`67ac6d3` **YOLOv8n HEF compile — 212 FPS on Hailo-8** (#74, Apr 12 18:57–19:50 UTC)
- `2f5084c` EfficientNet-S training pipeline (#75, Apr 12 20:03 UTC)
- `ab5ee54`/`e78a9f8` Pin uv to Linux + exclude pi group from default resolution (hailort not on PyPI) — critical for keeping desktop/CI green (Apr 12 20:05–20:07 UTC)
- Training cleanup: `767ea7e` v2 model name fix, `65ef10f` skip empty class dirs, `2ce9455` tqdm + ETA,
  `f28b32c` replace_all bug, `e615abd` `run_training.sh`, `c7543fa` batch size 32 for phase-2 VRAM (Apr 12 22:14–22:29 UTC)
- `74bc405` **HF Hub upload script** (`scripts/upload_model_to_hf.py`) — issue #82 (Apr 12 22:43 UTC)
- `d373076` onnxscript dep for torch.onnx.export (Apr 12 23:22 UTC)
- `2bd7469` **EfficientNet-S training done — 80.3% top-1** (236 classes, pre-Blue-Jay fix) (Apr 12 23:32 UTC)

### Training recipe (issue #75)
- Base: `torchvision.models.efficientnet_v2_s`, ImageNet-pretrained
- Head: linear layer to N species; 224×224 input
- Data: iNaturalist research-grade photos, filtered to New York (`place_id=48`)
  via `scripts/download_inat_training_data.py`
- Phase 1 (head only) + Phase 2 (fine-tune full network, batch size 32 for VRAM)
- Augment: random crop, hflip, color jitter, rotation; label smoothing 0.1
- Final run: **80.7% top-1, 94.0% top-5** on validation after Blue Jay fix

### Apr 12–13 — EfficientNet → HEF (issue #77)
- `ecb03ea` HEF compile script + `hailo_classifier.py` (Apr 12 23:41 UTC)
- `031ceaf` `dynamo=False` — Hailo DFC 3.33.1 needs legacy ONNX exporter (Apr 12 23:45 UTC)
- `23a3eef` **Calibration NCHW → NHWC** (DFC expects HWC) — classic gotcha (Apr 13 00:16 UTC)
- `1499dc7` HEF on HF; Pi benchmarks in model card (Apr 13 00:45 UTC)
- `b90ffe7` Final: **22 FPS / 44ms on Pi** for 237-species classifier (Apr 13 00:46 UTC)

### Apr 13 — the Blue Jay bug + retrain (issues #83, #84)
The first 236-class model silently dropped **Blue Jay** — the most
recognizable bird on the property. Root cause: `download_inat_training_data.py`
was querying iNaturalist by raw `taxon_name` string instead of resolving to a
taxon id first, so `Cyanocitta cristata` (id `8229`) came back empty under the
NY `place_id=48` filter. Once queries were resolved to taxon ids, 500 Blue Jay
photos came down immediately.
- `0caa274` Fixed downloader; improved retraining logging (Apr 13 16:40 UTC)
- `4092477` Updated docs for latest retrain state (Apr 13 20:57 UTC)
- Regenerated `species_labels.json` (236 → **237 classes**) and ONNX
- Recompiled the Hailo HEF from the new ONNX
- Re-uploaded artifacts + updated model card to HF (`k10z/birdvision-efficientnet-s`)
- Remaining zero-data classes under NY filter: Atlantic Puffin, Carolina Chickadee (documented, acceptable)

### Apr 13 — real-time pipeline (issues #76, #78, #79, #80)
- `5ca9651` **Real-time pipeline**: `hailo_detector.py` + `stream_capture.py` + `realtime_pipeline.py` + Docker (Apr 13 21:20 UTC)
- Runtime debugging marathon (Apr 13 21:33–22:18 UTC):
  - `0287b17` Install HailoRT .deb in runtime image; device group config
  - `d312ec1` uv-installed cp313 for hailort wheel
  - `ad9ebce` `dpkg --unpack` to skip post-install scripts
  - `e303745` UID collision — hailort claims 1000, use 65532
  - `c2361ce` `PYTHONPATH=/app`
  - `b00e44b` Define Detection locally — drop cross-import of desktop `detector.py`
  - `80cf3b4` **Share single VDevice** between detector + classifier
  - `5347eae`/`405a0f2`/`4c1c826` NMS output format — list-of-80-per-class, not ndarray
  - `63ee9d2` C_CONTIGUOUS warning fix
- `10c4d7d` System stats (temp/load/cpu/fan) every 30s (Apr 13 22:23 UTC)
- `bfa520c` Local priors file wired into Pi pipeline (Apr 13 22:30 UTC)
- `046d3d1` **Pi pipeline milestone** — ~27–34 FPS end-to-end, verified live (Apr 13 22:40 UTC)

### Apr 14 — display overlay (issue #89)
- `1fd768b` Pi display test container (Apr 14 04:45 UTC)
- `50943dd` `reset_display.sh` to blank fb0 / restore console (Apr 14 04:50 UTC)
- `8e2fddb` **Framebuffer overlay for Pi Touch Display 2** (Apr 14 10:14 UTC)
- `da8456b` Always show caption — fall back to "No Bird Detected" (Apr 14 10:31 UTC)

### Apr 14 — timeline + blog post draft
- `7c47372`/`ff3fb38`/`82ddd7b` Project timeline outline for blog post (Apr 14 18:46–20:14 UTC)
- `f8c189c`/`154e7c3` First blog post draft: "Teaching a Raspberry Pi to Identify Birds" (Apr 14 20:23–20:25 UTC)

---

## Interlude — Apr 19–27: Maintenance + Pi Merge

A quieter stretch between the Act V sprint and the sidecar work.

- `79616d1` **Fix EMFILE leak**: persistent ExifToolHelper singleton (Apr 19 20:13 UTC)
- `3e6191d`/`3a240de` Merge pi-setup branch into main; remove hailort from pyproject.toml
  to fix cross-platform resolution (Apr 26)
- `ecfab49` README update (Apr 26)
- Issue #87 closed Apr 27 — Pi display overlay complete
- Issue #90 opened Apr 27 — "Investigate iPhone bird ID streaming" — seeds the sidecar idea

---

## Act VI — Apr 30–May 2: Phone Sidecar Mode (Wed–Fri)

The second major feature arc. The Pi gains a second runtime mode: instead of
reading a wired HDMI camera, it serves a browser-based camera client over
HTTPS/WebSocket. Point your phone at a bird, the Pi identifies it.

Issues #91–#94 opened May 1. PR #95 (1,622 additions) + PR #96 merged May 1.
PR #103 (upload fixes), PR #106, PR #107 merged May 2.

### Apr 30–May 1 — WebSocket sidecar core (PRs #95, #96)
- `b67e3ea` **WebSocket sidecar mode**: `ws_frame_source.py` + browser camera client
  `static/index.html` + `file_frame_source.py` test harness + `ws_test_client.py` (Apr 30 22:50 EDT)
- `afb3934` `autostart.sh` — systemd service wrapper for Pi (May 1 19:40 UTC)
- `246db97` Fix PR review issues: profile isolation, HTTPS detection, bbox alignment,
  GPS priors, frame/result queue safety (May 1 16:42 EDT)
- `f4a4a13` Merged PR #95: WebSocket sidecar mode for phone-to-Pi streaming (May 1 16:44 EDT)

### May 1 — HTTPS via Caddy (PR #96)
- `ea47a36` Add self-signed HTTPS and Start Camera button (May 1 16:59 EDT)
- `cb87d64` Add openssl to Pi runtime image (May 1 17:12 EDT)
- `cf80e45` HTTP redirect server → HTTPS link (May 1 17:14 EDT)
- `27e04bb` **Replace in-app TLS with Caddy reverse proxy** — cleaner HTTPS handling (May 1 17:29 EDT)
- `abad530`/`245f8ae` Fix Caddy TLS for IP-address access (May 1 20:58–20:59 EDT)
- PR #96 merged May 1 21:09 UTC — **sidecar v0.3.0**: phone opens `https://<pi-ip>/`,
  accepts one-time self-signed cert, streams JPEG frames + GPS to Pi over WebSocket

### May 2 — Upload identification + sidecar hardening (PRs #103, #104, #106, #107)
- `7b736d1` Sidecar file upload path (May 2 03:58 UTC)
- `de4297f` Log copyable upload identification results (May 2 04:10 UTC)
- `708abf6` Save upload classification debug crops (May 2 04:16 UTC)
- `13b2a84`/`d020306` Graceful fallback when debug crop dir is unwritable (May 2 04:21–04:26 UTC)
- `72b3e98` **Fix Hailo classifier input layout** — NHWC preprocessing fix (May 2 04:32 UTC)
- Issue #102 closed (May 2 04:36 UTC) — sidecar file upload identification complete
- PR #103 merged May 2 — **sidecar v0.3.1**: photo/video upload path with copyable
  summaries and debug crops
- `3c2764d` Publish Pi sidecar upstream port (May 2, PR #106)
- `26bce19` **Restrict sidecar streaming to one client** — reject extra WebSocket
  connections with error JSON + 1013 close code (May 2 11:29 EDT)
- `867267c`/`ff1059c`/`d1e2a90` **Camera zoom controls** — native track zoom with
  digital crop fallback, 0.5x steps to 15x, press-and-hold repeat (May 2 19:28–19:44 EDT)
- `80e68de` Fix rejection race, zoom state leak, will-change perf (May 2 19:59 EDT)
- PR #107 merged May 3 00:44 UTC — single-client guard + zoom controls

---

## Cross-Cutting Themes for the Blog Post

### 1. Product velocity enabled by AI coding agents
88 issues opened, 63 closed, 168 commits in 34 days. The Day-1 commit graph
alone — 49 commits building a complete pipeline, web UI, eBird priors, asset
store, OAuth, and photo support — is roughly what a team would ship in a
quarter. Worth naming: this was written alongside Claude Code and Codex as
collaborators.

### 2. Two very different inference environments, one codebase
- Desktop webapp: YOLOv8s + BioCLIP (+ ensemble with EfficientNet-BIRDS525 or Gemma 4)
- Pi edge: YOLOv8n HEF + fine-tuned EfficientNet-V2-S HEF on Hailo-8
- Kept cleanly separated: `Dockerfile.pi` / `config.pi.yaml` / `src/hailo_*.py` / `src/realtime_pipeline.py`.
  No changes to existing `src/` modules for Pi work — interfaces compatible instead.

### 3. Priors as a first-class signal
eBird bar-chart data → SQLite → multiplicative priors with Long Island
bounding-box gating → per-track explainability (visual vs weighted) → local
prior override YAML for specific locations. Not many bird ID apps expose this.

### 4. Content-addressed asset store + job state reconstruction
No database. Assets keyed by sha256; jobs reconstructed from JSON on startup.
Kept the system simple and crash-tolerant.

### 5. Model evaluation as infrastructure
A dedicated eval container runs multiple classifier backends over a fixed test
set and emits HTML reports. Made ensemble + Gemma comparison tractable.

### 6. The Gemma 4 subplot
VLM-as-classifier experiment. Most commits were plumbing fights — 4-bit quant,
device_map, generation_config warnings — but the interesting finding was
`48825c1`: feeding Gemma the full annotated still with a red box beat feeding
it the raw crop.

### 7. Hailo HEF compilation — the real gotchas
- `dynamo=False` in `torch.onnx.export` for DFC 3.33.1
- Calibration data must be NHWC, not NCHW
- HailoRT wheel needs cp313
- Shared VDevice between detector + classifier, or resource conflicts
- NMS output is `list[list[ndarray]]`, one per class

### 8. Two Pi modes: backyard and sidecar
The Apr 14 framebuffer work closes the backyard loop — camera-to-screen with
live species captions, no cloud, no network. The May 1–2 sidecar work opens a
second mode — phone-as-camera over WebSocket/HTTPS, turning the Pi into a
portable field device. Both modes share the same Hailo pipeline; only the
frame source and display differ.

### 9. Closing the phone loop
The sidecar arc (Apr 30–May 2) went from "investigate iPhone streaming" (#90)
to a working phone-to-Pi pipeline in 3 days: browser camera client, Caddy
HTTPS, file uploads, single-client guard, zoom controls. 5 PRs merged.

---

## Open Threads (Good "What's Next" Section)

- `#7, #30` Fine-tune detector + classifier on BirdVision data
- `#9` Video-level summary robustness to noisy track fragments
- `#18` Broader eBird region coverage + GPS-to-place lookup
- `#26` Small-bird recall via tiled/zoomed fallback
- `#34` Tuner: species-group optimization
- `#31` Human-in-the-loop active learning
- `#32, #33` Species-group rollups in UI
- `#85` Evaluate retrained 237-class EfficientNet against desktop classifier
- `#86` Power monitoring for battery field use
- `#88` Pi Docker: pass /dev/fb0 to container for Touch Display 2
- `#90` Native iPhone or hybrid phone/server exploration
- `#91–#93` Reconcile stale tracking issues (implemented by PR #95/#96)
- `#94, #97–#101` Pi WiFi hotspot for field sidecar use (5 sub-issues open)
- `#105` Log sidecar bandwidth

---

## Suggested Blog Post Structure

1. **Hook** — "I pointed a camera at my backyard and the Raspberry Pi told me what was there." Pi demo video.
2. **What it is** — 2 paragraphs. Webapp + Pi edge device; priors from eBird; ensemble classification.
3. **Act I: Day One** — the pace. What fell out of the first 49 commits.
4. **Getting classification right** — BioCLIP → ensemble → Gemma experiment. The priors story.
5. **Getting the product right** — OAuth, asset dedup, slug URLs, share cards, API ingest.
6. **Taking it to the edge** — Pi subproject. Training, HEF compilation gotchas, real-time pipeline, framebuffer overlay.
7. **Phone as camera** — Sidecar mode. WebSocket streaming, Caddy HTTPS, zoom controls. From issue to working demo in 3 days.
8. **Reflection** — AI-assisted velocity; what that changes about scope.
9. **What's next** — the open issues list.
