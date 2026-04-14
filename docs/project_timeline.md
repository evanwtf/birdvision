# BirdVision — Project Timeline

A detailed outline of what got built and when, sourced from the full GitHub
issue history (89 issues) and git log (~160 commits). This is raw material for
a blog post — structured chronologically, grouped by theme, with commit SHAs
and issue numbers so we can drill down later.

**Span:** 2026-03-30 through 2026-04-14 — about two weeks.

## Hardware used

**Desktop (webapp / training / Hailo HEF compilation)**
- NVIDIA RTX 3080 Ti, 12 GB VRAM
- AMD Ryzen 9 7900X
- 32 GB RAM
- x86_64 Ubuntu
- Location: Long Island / Nassau County, NY (40.7, -73.5)

**Raspberry Pi 5 (real-time edge pipeline)**
- Raspberry Pi 5, 8 GB RAM, aarch64 Ubuntu 24.04
- Hailo-8 AI accelerator (PCIe M.2, 26 TOPS INT8). HailoRT firmware 4.23.0.
- Elgato Cam Link 4K (HDMI→USB capture) reading a Samsung camcorder over HDMI
- Raspberry Pi Touch Display 2 for live caption overlay

## Trained model, published artifacts

- **Hugging Face repo:** https://huggingface.co/k10z/birdvision-efficientnet-s
- **GitHub repo:** https://github.com/evandhoffman/birdvision (private)
- Artifacts published to HF: `efficientnet_s_birds.onnx`, `efficientnet_s_birds.hef`
  (Hailo-compiled), `species_labels.json`, PyTorch best + phase-1 checkpoints,
  auto-generated model card with Pi benchmarks.
- License: **CC-BY-NC-4.0** — inherited from iNaturalist training data.
- HF username `k10z`; GitHub username `evandhoffman`.

---

## Act I — Day 1: From Nothing to a Working Pipeline (Mar 30)

The most remarkable day of the project. A working detection + classification
pipeline, web UI, Docker packaging, eBird priors, a tuner, OAuth, photo
support, video stills, and a canonical asset store — all landed on March 30.

### Core pipeline stands up
- `870d3dc` Initial framework: **YOLOv8 detection + BioCLIP zero-shot classification**
- `b88c6d8` Switch to **uv** + Dockerfile + docker-compose from day one
- `ec352a8` Strip model names from user-facing copy — describe capability, not implementation

### Web UI arrives same day
- `2043d8e` FastAPI + Jinja upload/results UI
- Chainguard runtime wrestling: `a09c93d` apk perms, `0934a5e` `--chown` venv speedup,
  `98e3946` entrypoint override, `a348c86` `mesa-gl` for OpenCV
- `37ce4b1`/`bd5c9f9` writable model volume + host-mounted config
- `71820cc` split venv/source layers for fast rebuilds
- `d15b62c` TemplateResponse signature fix for Starlette 0.36+

### Tracking, results display, metadata
- `df5c61a` Archive pruned tracks instead of deleting them — fixed zero-tracks summaries
- `48be974` Save and display best crop per track
- `14f3e52` Restore completed jobs from disk on startup (no DB — JSON on disk is state)
- `f30641f` Extract video metadata (date, GPS) + OSM links
- `2560a70` Link species to Cornell All About Birds

### First round of threshold tuning (same day)
- `87aa7db` Raise det. conf 0.5, add `min_frames_to_report`
- `a7afadf` Tune to 0.4 / `min_frames=3`
- `e6f9869` Reprocess button — rerun pipeline on existing upload
- `587b59a` **Config hot-reload** — change thresholds without restart (issue #2)
- `d450a56` `min_confidence_to_report` — rescue high-conf short tracks
- `c2ce16a` **Center weighting** — Gaussian weight by bbox distance from frame center

### eBird priors (issue #1, day 1)
- `c9898f8` Raw eBird bar-chart TSVs committed
- `a087de6` Import pipeline: bar charts → SQLite; location + season priors
- `2dbd281` Per-track explanation: visual vs prior-adjusted scores (issue #11)

### Docs
- `52166b4`/`e7b6d06`/`2dfedec`/`4c46505` README + CLAUDE.md + AGENTS.md — project context for future AI sessions

### Still same day: more pipeline work
- `9abe555` Improve tracking + classification filtering
- `c4ea020` Tune for mockingbird videos
- `7d081fb` Video-level species summary (issue #9's first pass)
- `639f90c` **Adaptive crop padding** — smaller/distant birds get more context (issue #4)
- `9a55be9` Switch detector yolov8n → yolov8s for better boxes (issue #3)

### Photo uploads (issues #13, #14, #15, #16, #17)
- `9107217` Photo uploads for multi-image classification
- `2bf7b59`/`1ddb70e` Smart-quote cleanups in templates
- `ea3aca5` Per-image metadata, merged priors, visual vs weighted tables
- `3590d39` Long Island eBird gating + better photo UI

### Upload review + canonical asset store (issues #22, #23, #24, #25)
- `8b8414f` **Content-addressed storage by sha256** — dedup across browser/API uploads
- `ccdd267` Multi-video uploads; remove manual date input
- `4191c50` Select All / None / New controls
- `2959a78` Warn on Safari-incompatible codecs
- `a53fe62` Collapsed "no bird" photo view

### Tuner + video stills (issues #27, #29, #35)
- `d01d98d` Single-video tuner + compose service
- `5c8ff80` Gull baseline tuning
- `3750151` Annotated video stills with bboxes
- `0ccc69e` Fix startup job restoration
- `727978b` Auto-refresh while jobs pending/running
- `a931c30` Embed video on job detail page
- `0c59ea6` Camera info in metadata display

### Auth (issues #36, #37, #38, #39)
- `51e194f` Google OAuth2 flow + setup docs
- `6a474b8` OAuth + themes + request logging
- `a0636f8` v0.1.0 docs

---

## Act II — Mar 31: Tests, Pagination, CI

- `4e07631` Docs for auth/themes/logging
- `497bc37` **pytest harness + unit tests** across pipeline/metadata/tracker/webapp (issues #40–#44)
- `5b3541e` Job list pagination, media type labels, mixed uploads (issue #45)
- `eb088f9` GitHub Actions CI on push/PR

---

## Act III — Apr 3–5: Polish Sprint

Product polish — making it a real site, not a demo.

- `e70663e` More allowlisted emails
- `037495f` Sort job listing by created_at desc (issue #46)
- `80c7560` **Open Graph share cards** on job pages (issue #47)
- `a290037` / `8ad9913` **Friendly slug URLs** with bare-ID redirects (issue #48, PR #52)
- `400e2cb` / `8b899d3` **Location-only prior mode** — no seasonal weighting (issue #49/#53)
- `cd5de19` Persist + display submitter email (issue #50)
- `7645bd1` Switch CI to manual dispatch only

---

## Act IV — Apr 7–11: API, Eval Harness, Ensemble, Gemma

Shifts from product features toward model quality and comparison infrastructure.

### API ingest (issue #55)
- `56aa809` / `82a312a` **`POST /api/v1/videos`** with `X-API-Token` header

### Model comparison eval container (issues #58, #59, #60)
- `a5dcef4` Eval container
- `2c451ac` Prefetch models compose service
- `18362ba` Eval docs in README
- `dbbb8a6` ENTRYPOINT [] for Chainguard
- `e4bed96` transformers + accelerate for Gemma
- `2728672` Document chmod 777 for nonroot user

### Gemma 4 vision-language classifier saga (issue #58)
- `ba440c2` First cut — `GemmaClassifier` wired into eval runner
- `c3147fa` Strip special tokens; fix generation_config deprecation
- `dde089c`/`74b...`/`e6d3a83` Fight with `generation_config` max_length/warnings
- `4fba1d0` 4-bit quantization to fit 12GB VRAM
- `68ec428` OOM fix: `device_map` in model_kwargs so pipeline doesn't re-`.to(device)`
- `59054c8`/`05b9dfc` Simplify prompt; filter warnings via `logging.Filter`
- `e7146ab` Tighten prompt: regional framing + North American common name
- `48825c1` Feed Gemma **full annotated still with red highlight box** — better than raw crop

### Ensemble + local priors (issues #61, #65)
- `7e8e5e4` EfficientNet-BIRDS525 backend + **weighted geometric-mean ensemble** (PR #63)
- `0f3d2af` **User-defined local species priors** on top of eBird (issue #61, PR #64)
- `34e4607` Per-model score breakdown in results (PR #65)
- `3fbab64`/`551928b` config wiring
- `ade83ff` iOS job list layout tightening (issue #66, PR #68)
- `3117c4a` Refresh AGENTS.md + README
- `a9d8348` Reject + clean up invalid uploads (PR #69)
- `ded283e` "Recent Jobs" → "Recent Visitors" + listing thumbnails

---

## Act V — Apr 12–14: The Raspberry Pi Subproject

The big one. Parent issue **#70** — "Adapt project for real-time streaming on Pi."
14 sub-issues (#71–#86) closed in 3 days.

### Apr 12 — hardware setup + training (issues #71–#75, #81, #82)
- `6747823` Scaffold Pi sub-project (#81) — separate `Dockerfile.pi`, `docker-compose.pi.yml`, `config.pi.yaml`
- `fa2d41b`/`b5b5ccb`/`67ac6d3` **YOLOv8n HEF compile — 212 FPS on Hailo-8** (#74)
- `2f5084c` EfficientNet-S training pipeline (#75)
- `ab5ee54`/`e78a9f8` Pin uv to Linux + exclude pi group from default resolution (hailort not on PyPI) — critical for keeping desktop/CI green
- Training cleanup: `767ea7e` v2 model name fix, `65ef10f` skip empty class dirs, `2ce9455` tqdm + ETA,
  `f28b32c` replace_all bug, `e615abd` `run_training.sh`, `c7543fa` batch size 32 for phase-2 VRAM
- `74bc405` **HF Hub upload script** (`scripts/upload_model_to_hf.py`) — issue #82
- `d373076` onnxscript dep for torch.onnx.export
- `2bd7469` **EfficientNet-S training done — 80.3% top-1** (236 classes, pre-Blue-Jay fix)

### Training recipe (issue #75)
- Base: `torchvision.models.efficientnet_v2_s`, ImageNet-pretrained
- Head: linear layer to N species; 224×224 input
- Data: iNaturalist research-grade photos, filtered to New York (`place_id=48`)
  via `scripts/download_inat_training_data.py`
- Phase 1 (head only) + Phase 2 (fine-tune full network, batch size 32 for VRAM)
- Augment: random crop, hflip, color jitter, rotation; label smoothing 0.1
- Final run: **80.7% top-1, 94.0% top-5** on validation after Blue Jay fix

### Apr 12–13 — EfficientNet → HEF (issue #77)
- `ecb03ea` HEF compile script + `hailo_classifier.py`
- `031ceaf` `dynamo=False` — Hailo DFC 3.33.1 needs legacy ONNX exporter
- `23a3eef` **Calibration NCHW → NHWC** (DFC expects HWC) — classic gotcha
- `1499dc7` HEF on HF; Pi benchmarks in model card
- `b90ffe7` Final: **22 FPS / 44ms on Pi** for 237-species classifier

### Apr 13 — the Blue Jay bug + retrain (issues #83, #84)
The first 236-class model silently dropped **Blue Jay** — the most
recognizable bird on the property. Root cause: `download_inat_training_data.py`
was querying iNaturalist by raw `taxon_name` string instead of resolving to a
taxon id first, so `Cyanocitta cristata` (id `8229`) came back empty under the
NY `place_id=48` filter. Once queries were resolved to taxon ids, 500 Blue Jay
photos came down immediately.
- `0caa274`/`4092477` Fixed downloader; re-ran phase-1 + phase-2 training
- Regenerated `species_labels.json` (236 → **237 classes**) and ONNX
- Recompiled the Hailo HEF from the new ONNX
- Re-uploaded artifacts + updated model card to HF (`k10z/birdvision-efficientnet-s`)
- Remaining zero-data classes under NY filter: Atlantic Puffin, Carolina Chickadee (documented, acceptable)
- `530004e` Rename CLAUDE.md → AGENTS.md with symlink for compat

### Apr 13 — real-time pipeline (issues #76, #78, #79, #80)
- `5ca9651` **Real-time pipeline**: `hailo_detector.py` + `stream_capture.py` + `realtime_pipeline.py` + Docker
- Runtime debugging marathon:
  - `0287b17` Install HailoRT .deb in runtime image; device group config
  - `d312ec1` uv-installed cp313 for hailort wheel
  - `ad9ebce` `dpkg --unpack` to skip post-install scripts
  - `e303745` UID collision — hailort claims 1000, use 65532
  - `c2361ce` `PYTHONPATH=/app`
  - `b00e44b` Define Detection locally — drop cross-import of desktop `detector.py`
  - `80cf3b4` **Share single VDevice** between detector + classifier
  - `5347eae`/`405a0f2`/`4c1c826` NMS output format — list-of-80-per-class, not ndarray
  - `63ee9d2` C_CONTIGUOUS warning fix
- `10c4d7d` System stats (temp/load/cpu/fan) every 30s
- `bfa520c` Local priors file wired into Pi pipeline
- `046d3d1` **Pi pipeline milestone** — ~27–34 FPS end-to-end, verified live

### Apr 14 — display overlay (issue #89)
- `1fd768b` Pi display test container
- `50943dd` `reset_display.sh` to blank fb0 / restore console
- `8e2fddb` **Framebuffer overlay for Pi Touch Display 2**
- `da8456b` Always show caption — fall back to "No Bird Detected"

---

## Cross-Cutting Themes for the Blog Post

### 1. Product velocity enabled by AI coding agents
89 issues opened, 74 closed, in 15 days. The Day-1 commit graph alone is
roughly what a team would ship in a quarter. Worth naming: this was written
alongside Claude Code and Codex as collaborators.

### 2. Two very different inference environments, one codebase
- Desktop webapp: YOLOv8n + BioCLIP (+ ensemble with EfficientNet-BIRDS525 or Gemma 4)
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

### 8. Closing loop: from capture to local display
The Apr 14 framebuffer work closes the loop — a camera-to-screen system with
live species captions, no cloud, no network. A good narrative endpoint for
the post.

---

## Open Threads (Good "What's Next" Section)

- `#7, #30` Fine-tune detector + classifier on BirdVision data
- `#9` Video-level summary robustness to noisy track fragments
- `#18` Broader eBird region coverage + GPS-to-place lookup
- `#26` Small-bird recall via tiled/zoomed fallback
- `#28, #34` Tuner: trial logging + species-group optimization
- `#31` Human-in-the-loop active learning
- `#32, #33` Species-group rollups in UI
- `#85` Evaluate retrained 237-class EfficientNet against desktop classifier
- `#86` Power monitoring for battery field use
- `#87, #88` Pi display overlay — full live feed + CC label on Touch Display 2

---

## Suggested Blog Post Structure

1. **Hook** — "I pointed a camera at my backyard and the Raspberry Pi told me what was there." Pi demo video.
2. **What it is** — 2 paragraphs. Webapp + Pi edge device; priors from eBird; ensemble classification.
3. **Act I: Day One** — the pace. What fell out of the first day.
4. **Getting classification right** — BioCLIP → ensemble → Gemma experiment. The priors story.
5. **Getting the product right** — OAuth, asset dedup, slug URLs, share cards, API ingest.
6. **Taking it to the edge** — Pi subproject. Training, HEF compilation gotchas, real-time pipeline, framebuffer overlay.
7. **Reflection** — AI-assisted velocity; what that changes about scope.
8. **What's next** — the open issues list.
