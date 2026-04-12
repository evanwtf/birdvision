# Hailo HEF Compilation — Status & Next Steps

## What's been done (✅ COMPLETE)

- HailoRT 4.23.0 confirmed on Pi (`hailortcli fw-control identify`)
- Downloaded `hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl` ✓
- Downloaded `hailo_model_zoo-2.18.0-py3-none-any.whl` ✓
- Installed `libgraphviz-dev graphviz` ✓
- `hailomz info yolov8n` works — model recognized, hailo8 supported ✓
- Prepared 500 calibration images in `~/calib_images/` from birdvision stills ✓
- **Git editable install of hailo_model_zoo v2.18** ✓ (with `--no-build-isolation`)
- **Installed torch + torchvision** (required by v2.18) ✓
- **Fixed import compatibility issues** — disabled torch_infer.py (not needed for compilation) ✓
- **Compilation running successfully** — Started 2026-04-12 19:46 (CPU mode, ~60 min expected)

## Issues Encountered & Solutions

### Issue 1: Pre-built Wheel Missing Files
**Problem:** `hailo_model_zoo-2.18.0-py3-none-any.whl` did not include the `cfg/postprocess_config/` 
directory, causing `hailomz compile` to fail with:
```
AllocatorScriptParserException: Post-process config file isn't found in
.../postprocess_config/yolov8n_nms_config.json
```

**Solution:** Switched from wheel install to git editable install:
```bash
pip uninstall hailo-model-zoo -y
cd ~/git/hailo_model_zoo && git checkout v2.18
pip install --no-build-isolation -e .
```
The git repo v2.18 tag includes all config files and works as a standalone clone (no external version.py dependency).

### Issue 2: Missing torch Dependency
**Problem:** v2.18's setup.py doesn't explicitly list torch as a dependency, but the code imports it:
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:** Manually installed torch + torchvision:
```bash
pip install torch torchvision
```

### Issue 3: Incompatible hailo_model_optimization API
**Problem:** v2.18 code tries to import `TorchInferenceModel` from hailo_model_optimization, but the 
installed version has a different API:
```
ImportError: cannot import name 'TorchInferenceModel' from 'hailo_model_optimization.flows.inference_flow'
(did you mean: 'HWInferenceModel'?)
```

**Root cause:** Version mismatch between hailo_model_zoo v2.18 (expects older API) and the installed 
hailo_model_optimization (newer/different API). No newer version of hailo_model_zoo was compatible with 
the installed DFC 3.33.1.

**Solution:** Disabled torch_infer.py module (it's only used for evaluation, not compilation):
```bash
mv ~/git/hailo_model_zoo/hailo_model_zoo/core/infer/torch_infer.py \
   ~/git/hailo_model_zoo/hailo_model_zoo/core/infer/torch_infer.py.disabled
```

The inference plugins are auto-discovered in infer_factory.py, so disabling one module doesn't break 
the compilation pipeline.

### Why v2.18?
- The git repo includes all config files (unlike the wheel)
- v2.18 is compatible with HailoRT 4.23.0 on the Pi
- Later versions (master branch) have different setup requirements (versions.py) and would need more complex workarounds

## Solution applied

The pre-built wheel (`hailo_model_zoo-2.18.0-py3-none-any.whl`) was missing the
`cfg/postprocess_config/` directory. Used git repo with editable install:

```bash
# Completed steps:
pip uninstall hailo-model-zoo -y
cd ~/git/hailo_model_zoo && git checkout v2.18
pip install torch torchvision
pip install --no-build-isolation -e .
mv hailo_model_zoo/core/infer/torch_infer.py hailo_model_zoo/core/infer/torch_infer.py.disabled
```

The torch_infer.py module had version incompatibility with installed hailo_model_optimization 
(expects saitama submodule that doesn't exist). Since it's only for evaluation (not compilation),
we disabled it.

**Compilation command:**
```bash
hailomz compile yolov8n --hw-arch hailo8 --calib-path ~/calib_images/
```

**Status:** ✅ **COMPLETE** | File: `~/yolov8n.hef` (4.2M) | Compiled: 2026-04-12 19:46

## Compilation & Testing Complete ✅

**Output file:**
- Path: `pi/models/yolov8n.hef`
- Size: 4.2M
- Compiled: 2026-04-12 19:46 (6 minutes on CPU)
- Architecture: Hailo-8 INT8 quantized, COCO dataset

**Test results on Pi (192.168.1.180):**
```
Model: yolov8n/yolov8n
Frames processed: 1062
FPS: 212.14
Send Rate: 2085.40 Mbit/s
Recv Rate: 2072.37 Mbit/s
Status: ✅ PASS
```

The HEF loads successfully and delivers excellent inference performance on the Hailo-8 accelerator.

## Deployment

The model is now in the repo at `pi/models/yolov8n.hef`.

`config.pi.yaml` references it at `/app/pi/models/yolov8n.hef` (already correct).

## Next Steps

- **EfficientNet-S fine-tune + HEF compilation** — See GitHub issues #75, #77
- **Realtime pipeline integration** — See GitHub issue #78

---

# EfficientNet-S Fine-Tune — Status & Next Steps

## What's been done (✅ COMPLETE)

- `scripts/download_inat_training_data.py` — fetches research-grade bird photos from
  iNaturalist API by species name, ImageFolder layout, resumable ✓
- `scripts/train_efficientnet.py` — two-phase EfficientNet-S fine-tune (head-only then
  full), ONNX export, `species_labels.json` output ✓
- iNaturalist download running on desktop: place_id=48 (New York state),
  quality_grade=research, captive=false, max 500 photos/species, 239 species ✓

## Results ✅ COMPLETE (2026-04-12)

- **val top-1: 80.3%** (target: ≥70%) ✓
- **val top-5: 94.2%** (target: ≥90%) ✓
- 236 species trained (3 skipped — no NY iNaturalist data: Atlantic Puffin, Blue Jay, Carolina Chickadee)
- Training data: ~9.2GB, iNaturalist NY (place_id=48), research-grade, ~500 photos/species
- Hardware: RTX 3080 Ti, ~55 min total (5 phase-1 + 15 phase-2 epochs)

**Artifacts:**
- `pi/models/efficientnet_s_birds.onnx`
- `pi/models/species_labels.json` (236 classes)
- `pi/models/efficientnet_s_birds_best.pt`
- HuggingFace backup: https://huggingface.co/k10z/birdvision-efficientnet-s

## Next Steps

- **Compile to HEF** — See issue #77 (same hailomz workflow as YOLOv8n)
