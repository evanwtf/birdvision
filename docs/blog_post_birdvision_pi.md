# Teaching a Raspberry Pi to identify birds in my backyard

I've got a lot of birds in my backyard. We have a feeder, a bird bath, and a
perpetually-annoyed cat watching from the window. The birds don't care about
the cat. Over the past couple of years I've gotten slightly obsessed with
figuring out what the visitors actually are — beyond "sparrow, sparrow,
sparrow, huh that's a weird one."

So a few weeks ago I decided to build something that could identify them
automatically. Over the course of two weeks it grew into a real thing: a web
app on my desktop for uploading videos, and — more interestingly — a little
Raspberry Pi with a Hailo-8 AI accelerator that can watch a live video feed
and tell me what it's looking at in real time, locally, no cloud. This post
is mostly about the Pi side.

## The hardware

I ordered this on April 7 from CanaKit:

- **CanaKit Raspberry Pi 5 8GB Quick-Start AI Kit — 26 TOPS** — $379.95
  (Pi 5, 8GB RAM, Hailo-8 M.2 accelerator, 128GB storage, case with active cooling)
- **Raspberry Pi Touch Display 2, 5" Portrait** — $52.95
- Shipping, tax, etc — about $22

Total $454.85, arrived a few days later. The Hailo-8 is the interesting piece
— it's a PCIe M.2 card that does 26 TOPS of INT8 inference. On paper that's
not far off what an old desktop GPU can do, except it pulls about 2 watts and
lives on top of a credit-card-sized computer.

I also had a Samsung camcorder lying around from a previous life, and an
Elgato Cam Link 4K which turns any HDMI output into a USB webcam. This ended
up being important — the Pi's own camera modules are fine, but nothing I own
has real zoom lenses for them, and the camcorder does. So the pipeline is:

    camcorder → HDMI → Cam Link 4K → USB → /dev/video0 on the Pi

## The desktop app, as prologue

Before the Pi even showed up, I had about a week of work on the desktop side.
I have an RTX 3080 Ti in my gaming PC which, it turns out, is pretty great
for fine-tuning small vision models. The desktop app is a FastAPI web thing
where I can upload a video or photo, and it runs a pipeline like:

1. YOLOv8n (the "nano" size, because we don't need anything bigger for a
   single bird class) detects the birds.
2. A multi-frame tracker links detections together so a single bird visible
   for 4 seconds doesn't produce 120 separate results.
3. Every 10 frames per track, it runs a classifier on the cropped bird region
   and accumulates predictions.
4. Then it applies **eBird priors** — if you're filming in my backyard in
   Long Island in April, some birds are a lot more likely than others. I
   imported bar-chart data from eBird for my local counties and the pipeline
   multiplies visual scores by regional frequency.

The whole thing fits in about 1200 lines of Python. The classifier started
out as [BioCLIP](https://imageomics.github.io/bioclip/), a zero-shot CLIP
model trained on biological imagery, which is kind of amazing — you hand it
a list of species and a photo, and it'll score each species by how well the
photo matches. No training required.

So by the time the Pi arrived, I had a working system. The problem is I
couldn't carry my desktop out into the backyard.

## The Pi as an edge device

The plan was always: take the pipeline I have on the desktop and make it run
on the Pi, live, off a camera, cheap enough in power that I can run it off a
battery in the yard. The trouble is that BioCLIP is way too big for a Pi
— it needs a GPU to be interactive, and a CPU-only Pi would give you maybe
one classification every few seconds. The Hailo-8 changes that, but only if
you can get your model onto it, and the Hailo compiler is picky.

A Hailo-8 wants `.hef` files — architecture-specific INT8 binaries. You
compile them on an x86 Linux box with the Hailo Dataflow Compiler (DFC), a
proprietary tool from their developer zone. DFC takes ONNX in, runs
quantization calibration on sample images, and emits an HEF.

So the plan was:

1. Get YOLOv8n compiled to HEF and running at a reasonable framerate.
2. Fine-tune a smaller classifier — something like EfficientNet-V2-S — on the
   ~237 species I actually care about, and get that compiled to HEF.
3. Wire both together in a pipeline that reads frames off the Cam Link and
   writes results somewhere useful.

## Getting YOLO onto the Hailo

This one I was lucky on. Hailo publishes a model zoo with a pre-compiled
`yolov8n.hef`. I grabbed it, copied it to `pi/models/`, and wrote a small
detector wrapper that loads the HEF, shares a `VDevice` (the Hailo runtime's
handle to the chip), and runs inference. I benchmarked it at **212 FPS** on
the Pi, which is way more than I need — the Cam Link tops out at 60 FPS
capture anyway.

There were two small gotchas. First, the NMS output format surprised me.
I expected a flat numpy array of `[x, y, w, h, conf, class]` rows. What you
actually get back is a nested structure — a list of length 1, then inside
that a list of length 80 (one per COCO class), then inside that arrays of
detections for that class:

```python
# This is what val = output_tensors[0] gives you:
# val is a list of length 1 (per-batch)
# val[0] is a list of length 80 (per-class)
# val[0][class_id] is an ndarray of shape (num_detections, 5)
#                   with columns [y_min, x_min, y_max, x_max, confidence]
for class_id, per_class in enumerate(val[0]):
    if class_id != BIRD_CLASS_ID:
        continue
    for det in per_class:
        y1, x1, y2, x2, conf = det
        if conf >= threshold:
            ...
```

Took me three commits to nail down exactly what shape I was looking at, all
within a couple hours of reading Hailo forum posts. The other gotcha was
that I needed to share one `VDevice` between the detector and the classifier
— trying to open two of them against the same Hailo chip from the same
process just fails in weird ways.

## Fine-tuning a bird classifier

Now the hard part. I wanted an EfficientNet-V2-S fine-tuned on 237 species
of birds that are plausible to see in the northeastern US. The base model is
an ImageNet-pretrained network; I needed to replace the final classifier
layer with one sized to my species list and then do two phases of training —
first just the head, then the whole network at a lower learning rate.

The question was where to get the training data. I settled on
[iNaturalist](https://www.inaturalist.org/), which has an open API and a ton
of research-grade photos geotagged all over the world. Their API lets you
query by `taxon_id` and `place_id`, and for training I wanted New York State
observations (`place_id=48`) filtered to research-grade photos only.

I wrote `scripts/download_inat_training_data.py` to pull ~500 photos per
species into an `ImageFolder`-compatible directory tree:

```
train_data/
    american_robin/
        123456789.jpg
        ...
    northern_cardinal/
        ...
    blue_jay/
        ...
```

The iNat API rate-limits you at 100 requests/minute unauthenticated, so the
script paces itself at ~67 requests/minute with a `time.sleep(0.9)` between
calls — well under the limit but not so slow that the downloads take forever.
Downloaded a total of about 105,000 photos over a few hours.

Then training. I wrote `scripts/train_efficientnet.py` that does the usual
dance — `torchvision.models.efficientnet_v2_s(weights=IMAGENET1K_V1)`,
replace `classifier[1]` with `nn.Linear(1280, num_classes)`, and train in two
phases:

- **Phase 1**: Freeze everything but the new head. Train at `lr=1e-3` for a
  few epochs. This gets you most of the way there.
- **Phase 2**: Unfreeze everything and fine-tune at `lr=1e-4`. This is the
  slow and expensive part and eats all 12GB of VRAM at batch size 32.

The 3080 Ti chews through this fast enough that I was getting a full training
run in about an hour. The first run I did gave me **80.3% top-1** and **94%
top-5** accuracy on the held-out validation set, which for a 236-class
problem I'll take.

Except for one thing.

## The Blue Jay that wasn't

I tested the trained model on one of my own bird videos and it insisted a
Blue Jay was a Steller's Jay. Steller's Jay is a west-coast bird — beautiful,
but emphatically not in my yard. I figured, OK, Blue Jay probably just has a
few rough photos in the training set, I'll check.

When I looked at `train_data/blue_jay/`, it was empty. Zero photos.

This was confusing because Blue Jays are maybe the most-photographed bird in
New York State. So I went back to my download script and traced through
what it was doing. It turned out that when I built the iNat query URL, I
was passing `taxon_name=Blue Jay` as a query parameter. iNat's API does
accept that — but it's a fuzzy text match, not a taxon lookup, and for some
pairs of taxa where the common name collides with a sibling or a subspecies,
it just returns nothing.

The fix was to do an explicit taxon lookup first:

```python
def resolve_taxon_id(common_name: str) -> int:
    """Look up the iNat taxon id for a common name like 'Blue Jay'."""
    r = httpx.get(
        f"{INAT_API}/taxa",
        params={"q": common_name, "rank": "species", "per_page": 1},
    )
    results = r.json()["results"]
    if not results:
        raise LookupError(f"no iNat taxon for {common_name!r}")
    return results[0]["id"]
```

Once the downloader resolved Blue Jay to taxon id `8229` first, and then
queried observations with `taxon_id=8229&place_id=48`, 500 photos came down
in a couple of minutes.

I re-ran the whole training pipeline from scratch — phase 1, phase 2, ONNX
export — and retrained with 237 classes including Blue Jay. Final metrics
bumped up a notch to **80.7% top-1, 94.0% top-5**. The Steller's Jay
misclassification went away on my test video, and I felt silly for a
couple of hours.

A few species still had zero training photos after the fix — Atlantic Puffin
and Carolina Chickadee, neither of which anyone has photographed in New York
State and marked research-grade. Fine. Those just won't be predicted and
that's a reasonable tradeoff.

## The HEF compile, and its peculiarities

Next up: turn `efficientnet_s_birds.onnx` into `efficientnet_s_birds.hef`.
The Hailo DFC has a Python API that, roughly, goes like this:

```python
runner = ClientRunner(hw_arch="hailo8")
runner.translate_onnx_model(onnx_path, model_name,
                            start_node_names=[input_name],
                            end_node_names=[output_name])
runner.optimize(calibration_data)  # needs real sample images
hef = runner.compile()
```

It's three steps: translate, optimize, compile. Translate moves ONNX ops into
Hailo's IR. Optimize does INT8 quantization with your calibration data.
Compile lowers the optimized graph to an actual chip binary. Translate is
instant. Optimize is ~30 minutes with 256 calibration samples. Compile is
another 30-60 minutes. A full round trip takes the better part of two hours.

Two things bit me here, both annoying and both undocumented in the places I
was looking:

### 1. ONNX export must use the legacy exporter

PyTorch 2.x added a new `dynamo=True` ONNX exporter which is supposed to be
the future. Hailo DFC 3.33.1 does not support it. If you use it, the translate
step silently produces a broken model that the optimize step fails on with a
cryptic error. You have to explicitly do:

```python
torch.onnx.export(
    model, dummy_input, onnx_path,
    opset_version=11,
    input_names=["input"], output_names=["output"],
    dynamo=False,                       # ← the important part
)
```

This cost me about two hours of staring at DFC error messages that had
nothing to do with the actual problem.

### 2. Calibration data must be NHWC, not NCHW

PyTorch is NCHW — `(batch, channels, height, width)`. TensorFlow is NHWC —
`(batch, height, width, channels)`. ONNX is typically NCHW. Hailo's DFC, for
reasons best known to Hailo, expects the calibration numpy array in NHWC.

If you hand it NCHW, it doesn't complain loudly. The optimize step runs,
emits warnings, and the resulting HEF produces total garbage predictions.
You find out when you test the HEF on the Pi and discover that 95% of the
time it's telling you your bird is a Laughing Gull. (I don't live anywhere
near a beach. There are no Laughing Gulls here.)

Fix:

```python
# Wrong (NCHW — what you'd naively do in a PyTorch workflow):
calib = np.stack([preprocess(img) for img in samples]).astype(np.float32)
# calib.shape == (256, 3, 224, 224)

# Right (NHWC — what DFC actually expects):
calib = np.stack([preprocess(img) for img in samples]).astype(np.float32)
calib = calib.transpose(0, 2, 3, 1)
# calib.shape == (256, 224, 224, 3)
runner.optimize(calib)
```

Once I fixed that, the HEF came out sensible. Benchmarked on the Pi: **22
FPS / 44 ms per inference** for 237-way classification. Since the pipeline
only classifies every 10 frames per track, that's enough headroom to handle
several birds in frame simultaneously at 60 FPS capture.

## Publishing the model

I pushed the model artifacts to Hugging Face:
**https://huggingface.co/k10z/birdvision-efficientnet-s**

What's up there:

- `efficientnet_s_birds.onnx` — the portable ONNX artifact, runs on CPU/GPU
  via `onnxruntime` if you want to use this without a Hailo
- `efficientnet_s_birds.hef` — the Hailo-compiled binary (Hailo-8 only)
- `species_labels.json` — class index → species name, must match training order
- PyTorch checkpoints for both phases, in case anyone wants to resume training
- A model card with Pi benchmarks, training details, and an ONNX inference
  example

The license is **CC-BY-NC-4.0**, which it has to be — iNaturalist photos are
CC-BY-NC, so anything derived from them is too. No commercial use. Fine by
me; I'm not trying to sell birds.

## The real-time pipeline

With both HEFs on the Pi, the pipeline is straightforward enough:

```
V4L2FrameSource → HailoDetector → BirdTracker → HailoClassifier → eBird priors
                                                                  → 1-second log
                                                                  → display overlay
```

`V4L2FrameSource` is a thin wrapper around `v4l2-ctl` / OpenCV's V4L2 backend
that reads YUYV frames from `/dev/video0` at 1920×1080, 60 FPS, and yields
them as BGR numpy arrays. Every frame goes through the detector. Detections
go into the tracker, which links them across frames by IoU + centroid
distance so a single bird produces one track rather than 120 independent
results. Every 10th frame per track, a crop of that bird gets sent to the
classifier. Classifier outputs get weighted by a Gaussian based on where the
bird is in the frame (center birds count more than edge birds), then
multiplied by eBird regional priors, then accumulated onto the track.

The whole thing runs at about **27-34 FPS end-to-end**. The bottleneck isn't
the Hailo at all — it's the YUYV decode and OpenCV frame copies on the CPU.
I could probably squeeze more out of it with a proper GStreamer pipeline,
but 30 FPS is plenty for birds, which move slower than cars.

I also added a status logger that dumps the Pi's temp, load average, fan
state, and CPU usage every 30 seconds. I was a little worried about the
Hailo + active inference pegging the SoC and causing thermal throttle, but
in practice the Pi sits around 55-60°C with the kit's active cooler. Fine.

## The Touch Display overlay

The last piece: I bought a 5" Raspberry Pi Touch Display 2 to mount next to
the setup so I can see what the pipeline is seeing without SSHing in. The
Pi exposes it as a framebuffer at `/dev/fb0` — you can literally write RGB
bytes to that file and they show up on the screen. I wrote a small overlay
module that draws the latest frame onto the framebuffer with the current
top-prediction caption underneath:

```
+------------------------------------+
|                                    |
|     [ bird crop, highlighted ]     |
|                                    |
|                                    |
|   Northern Cardinal  (conf 0.91)   |
+------------------------------------+
```

There's a tiny subtlety: when the container exits, you want the framebuffer
reset so the Pi's normal console comes back — otherwise you're stuck looking
at a frozen bird frame forever. `reset_display.sh` blanks `/dev/fb0` and
restores the console cursor, called from the container's shutdown hook.

The other subtlety: if there's no bird in frame, I want the caption to still
show — "No Bird Detected" — rather than just blanking to the last caption.
Otherwise you stare at the screen and have no idea if the pipeline is
running or has crashed.

## What I learned

A few scattered observations after two weeks:

**Edge inference hardware has gotten really good and really cheap.** A $400
Pi+Hailo kit does real-time 237-way fine-grained image classification at 30
FPS. Five years ago this was Jetson or GPU-workstation territory.

**Species priors are underrated.** My eBird-prior layer is maybe 200 lines
of code and it fixes a lot of the "technically plausible but you'd never see
that bird in Nassau County" mistakes. When visual evidence is marginal, a
prior pulls you toward the plausible answer. When visual evidence is strong,
the prior barely matters. This is just Bayes, and more people should do it.

**Publishing the model was the easy part.** The iNaturalist data, the
training code, the HEF compile gotchas — those all took real time. Pushing
the final artifacts to HuggingFace was like ten minutes. HuggingFace is
great.

**Most of the real pain was in data.** The Blue Jay bug was the single
biggest debugging rabbit hole of the project, and it had nothing to do with
model architecture or compile pipelines or any of the interesting parts. It
was a `params={"taxon_name": ...}` vs `params={"taxon_id": ...}` call. The
lesson is to always spot-check your training data before you train on it.

**What's next?** A few things. I want to evaluate the retrained 237-class
model against the desktop BioCLIP ensemble to see how close it actually is
(the validation accuracy is a proxy, not the real thing). I want to hook up
a USB power meter so I can tell people how long this thing runs off a USB-C
battery pack — my guess is 5-6 hours of continuous inference. And I want to
get the live video stream onto the display, not just the caption, so I can
set the whole thing up on the deck and use it as a weird birdwatching
monitor.

But the core system is working. I pointed it at my window this morning,
walked away for a cup of coffee, and came back to a log full of House
Sparrows, a Mourning Dove, and a Red-Bellied Woodpecker. Which, as far as
I can tell, is exactly what was outside.

Not bad for a $450 kit and a couple of weeks.
