# Hailo HEF Compilation — Status & Next Steps

## What's been done

- HailoRT 4.23.0 confirmed on Pi (`hailortcli fw-control identify`)
- Downloaded `hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl` ✓
- Downloaded `hailo_model_zoo-2.18.0-py3-none-any.whl` ✓
- Installed `libgraphviz-dev graphviz` ✓
- `hailomz info yolov8n` works — model recognized, hailo8 supported ✓
- Prepared 500 calibration images in `~/calib_images/` from birdvision stills ✓

## Current blocker

`hailomz compile yolov8n --hw-arch hailo8 --calib-path ~/calib_images/` fails:

```
AllocatorScriptParserException: Post-process config file isn't found in
.../hailo_model_zoo/cfg/alls/generic/../../postprocess_config/yolov8n_nms_config.json
```

The pre-built wheel (`hailo_model_zoo-2.18.0-py3-none-any.whl`) is missing the
`cfg/postprocess_config/` directory — it wasn't packed into the wheel correctly.

## Fix

The git repo at `~/git/hailo_model_zoo` (currently on `main`/v5.3.0) has the files.
Switch to v2.18 and do an editable install instead of using the wheel:

```bash
source ~/venv/bin/activate
pip uninstall hailo-model-zoo -y   # remove broken wheel install (if it registered)
cd ~/git/hailo_model_zoo
git checkout v2.18
pip install -e .
```

v2.18's `setup.py` has versions hardcoded — no `versions.py` dependency, so this
works as a standalone git clone.

Then retry:

```bash
hailomz compile yolov8n --hw-arch hailo8 --calib-path ~/calib_images/
```

## After compile succeeds

1. `yolov8n.hef` will be written to the current directory (~30–60 min on CPU)
2. Copy to Pi: `scp yolov8n.hef pi@<pi-ip>:~/`
3. Test on Pi: `hailortcli run yolov8n.hef`
4. If it loads, copy to `pi/models/yolov8_birds.hef` in the repo
