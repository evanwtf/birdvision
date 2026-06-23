> **Draft.** Work-in-progress blog post kept in the repo for editing
> convenience. First-person origin-story version. The final published version
> will live elsewhere. (A separate, more technical Pi-focused draft lives in
> `blog_post_birdvision_pi.md`.)

# Building BirdVision

Growing up, my mom always had lots of bird feeders in our yard. Filling them
was one of my chores, so as you might imagine, we had lots of different birds
around. Cardinals, finches, goldfinches, sparrows, chickadees, blue jays,
seagulls, woodpeckers, crows, and hummingbirds were all pretty common sights.
I've always had an interest in birds, mostly just because they were always
there.

A few months ago I was looking up at a bird in the sky and wondering what it
was. Was it a bald eagle? A turkey vulture? An osprey? I've only ever seen one
confirmed bald eagle in my whole life outside of a zoo, and standing there
squinting at this thing, I started thinking how cool it would be if there were
a "smart scope" — something you could look through like a telescope that
identified birds in real time, locally, on-device. Ideally it would accept
existing telephoto lenses from a big camera maker like Sony or Canon, so you
could get really good optical zoom without reinventing the glass.

I pondered this for a few days and couldn't come up with a good reason it
wouldn't work, at least as a prototype. A GPU, an image sensor, a 3D-printed
housing to mount the lens, and some software to mash it all together. Why not?
V1 would probably be clunky and expensive, but it seemed very doable, and more
importantly it seemed fun.

I consulted my buddy Claude for options and figured the easiest starting point
for a hardware platform was a Raspberry Pi with a Hailo-8 accelerator. I'd
never heard of Hailo, but the website said the chip could do 26 TOPS, and the
kit from CanaKit ran about $400, which seemed reasonable for something that
would sit in a 3D-printed box in my backyard.

I ordered the kit, but while I waited for it to arrive I figured I'd see
whether I could prototype the whole idea on my old Windows gaming PC. I'd built
that machine a few years back — 32 GB of RAM, a Ryzen 7900, and a GeForce 3080
Ti — and I hadn't turned it on in almost a year. I'd quit World of Warcraft and
switched to a MacBook Pro as my daily driver, so I was pretty confident there
was nothing on it I needed. I wiped it and installed Ubuntu 24.04.

Once that was up, I decided the fastest way to prove out bird species
identification was to skip the hardware entirely for now and just build a web
app where I could upload photos and videos and have it ID the birds. I didn't
really know where to begin, but Claude suggested YOLOv8 for detection and
BioCLIP for classification — YOLO finds the bird in the frame, BioCLIP figures
out what species it is.

At this point I have to admit I basically let Claude take over driving the
development. My role became less "programmer" and more "product owner." I
defined what I wanted in GitHub issues and had Claude Code work through them as
a queue. I had a few standing preferences: the code should be Python, use `uv`
as the package manager, and run in a container. I also had an idea I was pretty
attached to — that the results would be a lot more useful if I weighted the
species suggestions by location. On Long Island it would be vanishingly rare to
see a penguin but extremely common to see a herring gull, and the model should
know that.

Initial dev was fast. By the end of the first day I had something surprisingly
competent running on that old desktop:

- **Detection and classification** with YOLOv8 + BioCLIP, working on both
  photos and video — and for video, IDing birds either per-frame or per-track,
  so a single bird moving across the frame got treated as one identity instead
  of a new bird every frame.
- **Geographic weighting** built on eBird data, abstracted to a simple "on Long
  Island / not on Long Island" signal that nudged the species rankings toward
  what you'd actually expect to see there.
- **A web UI** for uploading photos and videos, with async job processing so
  the page didn't hang while a model chewed through a clip.
- **A metadata processor** for pulling geodata out of photos and videos, which
  fed the geographic weighting.
- **A container** built on the Chainguard dev image.
- **Google OAuth login** with a static allowlist — anybody could view results,
  but only I could upload.

For a single day of work it was wildly more than I expected. The "Long Island
vs. not" weighting was crude, the UI was bare, and none of it was anywhere near
a scope you could hold up to the sky — but the core loop worked. You could hand
it a bird and it would tell you, with reasonable confidence, what it was
looking at. That was enough to convince me the rest was worth building.

Over the next week or so the desktop app filled out. The classifier grew into
an ensemble — a weighted geometric mean of BioCLIP and a second fine-tuned
classifier, which smoothed out the cases where one model was confidently wrong.
I built an eval harness in its own container so I could run different
classifier backends over a fixed test set and actually compare them instead of
eyeballing it. (I even spent an afternoon wiring up Gemma as a vision-language
classifier — feeding it a still with a red box drawn around the bird and asking
it for a North American common name. It worked better than I expected, but it
was slow and heavy, so it stayed an experiment.) The eBird priors got more
serious too: real county bar-chart data imported into SQLite, per-week seasonal
weighting, and a per-track explanation showing the raw visual score next to the
prior-adjusted one. And a pile of product polish — content-addressed upload
storage so the same video uploaded twice didn't get processed twice, slug URLs,
share cards, an API endpoint for ingesting videos.

It was a real little app at that point. The only problem was that I couldn't
carry my desktop out into the backyard.

## The Pi shows up

The CanaKit kit arrived a few days later. The whole order came to about $455 —
the Pi 5 AI kit was around $380, and I'd also thrown in a 5" Raspberry Pi Touch
Display 2 so the thing could show me what it was seeing without me having to
SSH in.

The interesting piece is the Hailo-8. It's a little PCIe M.2 card that does
26 TOPS of INT8 inference and pulls something like 2 watts. On paper that's not
a million miles off what an old desktop GPU manages, except it lives on top of
a credit-card-sized computer and barely sips power. That's exactly the trade
you want for something that's eventually going to run off a battery on a deck
railing.

The camera question I solved with stuff I already had lying around. I had an
old Samsung camcorder from a previous life, and an Elgato Cam Link 4K, which is
a little dongle that turns any HDMI output into a USB webcam. This turned out
to be the key to the whole "use real camera glass" idea. The Pi's own camera
modules are fine, but nothing I own has proper zoom lenses for them — the
camcorder does. So the capture path became:

    camcorder → HDMI → Cam Link 4K → USB → /dev/video0 on the Pi

I started calling this "backyard mode": a wired camera pointed out the window,
feeding frames into the Pi over a USB capture card. The whole pipeline runs on
the Pi, nothing leaves the device.

## Getting the models onto the Hailo

The catch with the Hailo-8 is that it doesn't run PyTorch models directly.
BioCLIP, the classifier that made the desktop app work, is way too big for a Pi
anyway — it really wants a GPU to be interactive. The Hailo can be fast, but
only if you compile your model into its own format first: a `.hef` file, which
is an architecture-specific INT8 binary. You build those on an x86 Linux box
with Hailo's Dataflow Compiler, which takes an ONNX model in, runs quantization
calibration on some sample images, and spits out an HEF.

So the plan had three parts: get a detector compiled and running, fine-tune a
small classifier on the species I actually care about and compile that too,
then wire both together into a live pipeline reading off the Cam Link.

The detector I got lucky on. Hailo publishes a model zoo with a pre-compiled
`yolov8n.hef`, so I grabbed it, wrote a small wrapper to load it and run
inference, and benchmarked it at **212 FPS** on the Pi — wildly more than I
need, since the Cam Link only captures at 60 FPS anyway.

The classifier was the real work. I wanted an EfficientNet-V2-S fine-tuned on
the ~237 species that are plausible in the northeastern US. For training data I
used iNaturalist, which has an open API and a huge pile of research-grade,
geotagged photos. I wrote a downloader to pull a few hundred photos per species
filtered to New York observations, ended up with around 105,000 images, and let
the 3080 Ti chew through training — a full run took about an hour. First model
came back at roughly **80% top-1, 94% top-5** accuracy on the held-out set,
which for a 237-way problem I'll happily take.

Then I hit the single dumbest, most time-consuming bug of the entire project,
and it had nothing to do with any of the interesting parts. I tested the
trained model on one of my own videos and it insisted a Blue Jay was a Steller's
Jay — a beautiful west-coast bird that is emphatically not in my yard. I went to
look at the Blue Jay folder in my training data and it was *empty*. Zero photos
of the single most recognizable bird on the property.

It turned out my downloader was querying iNaturalist by the common name string
("Blue Jay") instead of resolving it to a proper taxon ID first. iNat accepts
the text query, but it's a fuzzy match, and for some species it just quietly
returns nothing. The fix was to look up the taxon ID first and then query
observations by ID — once I did that, 500 Blue Jay photos came down in a couple
of minutes. I retrained from scratch, the Steller's Jay problem went away, and I
felt silly for a couple of hours. The lesson, which I will now never forget, is
to spot-check your training data before you train on it.

Compiling the classifier to an HEF had two more gotchas waiting, both annoying
and both undocumented in the places I was looking. First, the ONNX export has
to use PyTorch's *legacy* exporter (`dynamo=False`) — the shiny new one
produces a model the Hailo compiler silently chokes on. Second, the calibration
data has to be in NHWC layout, not the NCHW that PyTorch hands you by default.
Get that wrong and it doesn't error — it just produces an HEF that's confidently
wrong about everything, which you only discover when the Pi starts telling you
every bird is a Laughing Gull. (I don't live near a beach. There are no
Laughing Gulls here.) Once I sorted both out, the compiled classifier ran at
about **22 FPS** on the Pi, plenty for the job.

I pushed all the model artifacts — the ONNX, the HEF, the labels, the
checkpoints, a model card — up to Hugging Face under `k10z/birdvision-efficientnet-s`.
It's licensed CC-BY-NC-4.0, which it has to be: iNaturalist photos are
non-commercial, so anything trained on them inherits that. Fine by me — I'm not
trying to sell birds.

## Real-time in the backyard

With both HEFs on the Pi, the live pipeline came together. It reads frames off
the Cam Link at 1080p, runs every frame through the detector, feeds detections
into the same IoU + centroid tracker the desktop app uses so a single bird
becomes one track instead of a hundred, classifies a crop every tenth frame per
track, weights the result by how centered the bird is in the frame, multiplies
in the eBird priors, and accumulates it all onto the track. Same conceptual
pipeline as the desktop, just with the Hailo doing the heavy lifting instead of
a GPU.

End-to-end it runs at about **27–34 FPS**, which is plenty — birds move a lot
slower than cars. The fun part is that the Hailo isn't even the bottleneck;
it's the CPU-side work of decoding the raw video frames and copying them
around. The accelerator is barely breaking a sweat. I added a little status
logger that prints the Pi's temperature, load, and fan state every 30 seconds
because I was worried about thermal throttling, but with the kit's active
cooler it just sits around 55–60°C and shrugs.

The last backyard piece was the Touch Display 2. The Pi exposes it as a plain
framebuffer at `/dev/fb0` — you can literally write pixels to a file and they
show up on the screen — so I wrote a small overlay that draws the current frame
with the top species prediction captioned underneath. A couple of small but
important details: when the program exits you have to blank the framebuffer and
restore the console, or you're stuck staring at a frozen bird forever; and when
there's no bird in frame it shows "No Bird Detected" rather than the last
caption, so you can always tell the thing is actually running and hasn't
crashed.

At this point the original idea basically worked. I pointed the camera out the
kitchen window one morning, went to make coffee, and came back to a log full of
House Sparrows, a Mourning Dove, and a Red-Bellied Woodpecker — which, as far
as I could tell, was exactly what was outside. A $455 kit, a couple of weeks,
and no cloud anywhere in the loop.

## Phone as the camera

The backyard setup had one obvious limitation: it's tethered. A camcorder, a
capture card, a wall outlet, and a little screen, all sitting in a window.
Great for the kitchen, useless for actually walking around. And the camcorder,
while it has real zoom, is a clunky thing to aim.

Meanwhile I was carrying around a device with an excellent camera, a GPS, a
screen, and a touchscreen for zoom controls — my phone. So the next idea was
obvious in hindsight: what if the phone *was* the camera, and the Pi was just
the brain? Point the phone at a bird, stream the video to the Pi over the local
network, let the Pi do the identification, and send the results back.

This became "sidecar mode." Instead of reading a wired HDMI camera, the Pi runs
a little web server and serves a browser-based camera client. You open a page on
your phone, hit "Start Camera," and it streams JPEG frames — plus the phone's
GPS coordinates — to the Pi over a WebSocket. The Pi runs the exact same Hailo
pipeline as backyard mode; the only thing that changed was the source of the
frames. Sending GPS along turned out to be a nice bonus: instead of the crude
"on Long Island / not" gate, the priors could use the phone's actual location.

The one genuinely fiddly part was HTTPS. Browsers refuse to give a web page
access to the camera unless the page is served over HTTPS, even on your own
local network. I tried to handle TLS inside the app and it was a mess, so I put
a Caddy reverse proxy in front to terminate HTTPS with a self-signed cert. You
accept the certificate warning once and from then on the phone can open the
camera. Not pretty, but it's a device on my own network, not a public website.

From there the sidecar grew the features that made it actually usable in the
field. A file-upload path, so you can hand it a photo or video instead of a live
stream and get back a copyable identification summary. A single-client guard,
because the Hailo can only do one stream at a time and two phones connecting at
once made a mess — now the second connection gets politely rejected. And zoom
controls, which mattered more than I expected: the client uses the camera's
native zoom where the phone supports it and falls back to a digital crop where
it doesn't, with press-and-hold buttons stepping from 0.5x up to 15x. That last
one got me a lot closer to the original "smart scope" feeling — hold the phone
up, zoom in on a far-off bird, and read the species off the screen.

Sidecar mode is now the primary way I run the thing. It's the closest the
project has come to the picture in my head from that first afternoon: hold
something up, point it at a bird, and have it tell you what you're looking at,
right there, no cloud involved.

## What I'd still like to do

A few loose ends. The biggest is power — I'd love to run the whole thing off a
battery so it's genuinely portable, but the Pi 5 is picky. It really wants the
official 27W supply, or at least a USB-C battery that can negotiate 5V at 5
amps, and most portable packs I own top out at 3 amps. At boot the Pi flashes a
warning and refuses to enable the full power budget for accessories, which
matters once you hang a capture card off it. For now it stays plugged into a
wall, which is fine in a window but not on a hike.

I'd also like to properly evaluate the retrained 237-class model against the
desktop ensemble — validation accuracy is a proxy, not the real thing — and get
a WiFi hotspot mode working so the phone-to-Pi link doesn't depend on a network
being around.

But the core of it works, and works better than I had any right to expect from
a $455 kit and a few weeks of evenings. I set out to build a thing you could
point at a bird to find out what it is, and now I have two of them — one bolted
to a window, one running off the phone in my pocket. Not bad.
