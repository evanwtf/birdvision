# BirdVision Sidecar Mode — iPhone → Pi Streaming Plan

## Overview

Add a "sidecar" mode to the existing Raspberry Pi pipeline: instead of
reading frames from a USB capture device (V4L2), the Pi accepts JPEG frames
over a WebSocket from an iPhone app (or any client). The Pi runs detection
and classification on its Hailo-8 as usual and streams results back to the
client for overlay rendering.

For field use, the Pi can broadcast its own WiFi network via hostapd so the
iPhone can connect without any external infrastructure. For development and
home use, the phone and Pi communicate over the regular home LAN.

```
┌─────────────┐        WiFi (home LAN or AP)   ┌──────────────────────┐
│  iPhone app  │  ── JPEG frames (WebSocket) ─→│  Raspberry Pi 5      │
│  (or test    │  ←─ JSON results ────────────│  + Hailo-8            │
│   script)    │                               │  (same container)     │
└─────────────┘                                └──────────────────────┘
```

Same Pi, same Hailo, same models, same container image. The frame source
is selected by a config key — no separate compose file or image needed.

---

## Operating modes

Controlled by `stream.source` in `config.pi.yaml`:

| Mode | `stream.source` | Frame source | When to use |
|---|---|---|---|
| **Backyard** (existing) | `v4l2` (default) | USB camera | Stationary camera pointed at a feeder |
| **Sidecar** (new) | `websocket` | iPhone / test client | Handheld birding with the phone |

Only one mode runs at a time — the Hailo VDevice can only be opened by one
process.

### Config example

To switch modes, change one line in `config.pi.yaml` and restart the
container:

```yaml
# Backyard mode (default — unchanged from today)
stream:
  source: v4l2
  device: /dev/video0
  width: 1920
  height: 1080
  framerate: 59.94

# Sidecar mode — comment out the above, uncomment below:
# stream:
#   source: websocket
#   ws_port: 8765
#   ws_host: 0.0.0.0
```

Everything else in the config (detector, classifier, tracker, metadata,
display, output) stays the same regardless of mode.

---

## Components to build

### 1. WebSocket frame source (`src/ws_frame_source.py`)

A new frame source class that implements the same `frames()` iterator
interface as `V4L2FrameSource`:

```python
class WebSocketFrameSource:
    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_number, bgr) from JPEG frames received over WebSocket."""
        ...

    def stop(self) -> None: ...
```

- Runs a lightweight ASGI server (Starlette/uvicorn) on a configurable port
  (default 8765).
- Accepts a single WebSocket connection at `ws://<pi-ip>:8765/stream`.
- Each incoming message is a binary JPEG blob.
- Decodes with `cv2.imdecode()` and yields `(frame_no, bgr)`.
- Sends JSON result messages back on the same WebSocket after each
  classification cycle (bounding boxes + species predictions).

Estimated size: ~80-100 lines.

### 2. Pipeline config switch (~10 lines changed in `realtime_pipeline.py`)

`realtime_pipeline.py.__init__` currently hardcodes `V4L2FrameSource`.
Change it to check `stream.source`:

```python
if stream_cfg.get("source", "v4l2") == "websocket":
    from .ws_frame_source import WebSocketFrameSource
    self._source = WebSocketFrameSource(
        host=stream_cfg.get("ws_host", "0.0.0.0"),
        port=stream_cfg.get("ws_port", 8765),
    )
else:
    self._source = V4L2FrameSource(...)
```

The result callback (sending JSON back to the WebSocket client) needs a
small hook in the pipeline's classification section — after predictions are
computed, serialize and send them back. The WebSocket source exposes a
`send_result()` method that the pipeline calls. In V4L2 mode this is a
no-op.

### 3. Expose WebSocket port in `docker-compose.pi.yml`

Add port 8765 to the existing compose file. It's harmless in V4L2 mode
(nothing listens) and avoids needing a second compose file:

```yaml
ports:
  - "8765:8765"
```

### 4. Test client script (`scripts/ws_test_client.py`)

A Python script that reads a local video file and streams its frames over
WebSocket to the Pi, simulating an iPhone connection. This is critical for
development — avoids needing a physical iPhone for every test cycle.

```
uv run scripts/ws_test_client.py test_video.mp4 --server ws://pi-ip:8765/stream --fps 5
```

- Opens the video with OpenCV.
- Encodes each frame as JPEG.
- Sends over WebSocket at the specified FPS.
- Prints received JSON results to stdout.

Estimated size: ~60-80 lines.

### 5. iPhone app (`ios/BirdVision/`)

Minimal SwiftUI app (lives in this repo):

- **Camera capture:** `AVCaptureSession` with `AVCaptureVideoDataOutput`.
- **Frame streaming:** Encode sample buffers as JPEG, send over
  `URLSessionWebSocketTask` to the Pi.
- **Result overlay:** Decode JSON responses, draw bounding boxes and species
  labels on a transparent overlay above the camera preview.
- **Server config:** Text field for the Pi's IP/port (defaulting to
  `ws://10.0.0.1:8765/stream` for AP mode). Could persist in UserDefaults.
- **Frame rate control:** Target 3-5 FPS send rate (adjustable). The phone
  captures at 30 FPS but only sends every Nth frame.

Resolution: 720p is plenty. At JPEG quality 70%, each frame is ~50-80 KB.
At 5 FPS that's ~2-3 Mbps — trivial over local WiFi.

Estimated size: ~300-400 lines of Swift across 3-4 files.

### 6. hostapd setup on Pi host (one-time, not containerized)

Run hostapd + dnsmasq on the Pi **host OS**, not inside Docker. The
container just binds to `0.0.0.0:8765` and doesn't need to know how the
phone connected.

Required host config:

- `/etc/hostapd/hostapd.conf` — SSID (e.g. `BirdVision`), WPA2 passphrase,
  WiFi channel, `interface=wlan0`.
- `/etc/dnsmasq.d/birdvision.conf` — DHCP range for the AP subnet
  (e.g. `10.0.0.10,10.0.0.50`), bind to `wlan0`.
- Static IP on `wlan0` (e.g. `10.0.0.1/24`).
- `systemctl enable hostapd dnsmasq` for field use; disable when on home
  WiFi.

Toggle scripts for convenience:

```bash
# field-mode-on.sh
sudo systemctl start hostapd dnsmasq

# field-mode-off.sh
sudo systemctl stop hostapd dnsmasq
```

The Pi's Ethernet or USB tethering can still provide internet access while
the WiFi interface runs as an AP (if needed for pulling containers, etc.).

---

## What does NOT change

- `src/hailo_detector.py` — unchanged
- `src/hailo_classifier.py` — unchanged
- `src/tracker.py` — unchanged
- `src/metadata.py` — unchanged
- `src/stream_capture.py` — unchanged (still used by backyard mode)
- `src/realtime_pipeline.py` — small change to source selection in
  `__init__`, small hook for result callback; `run()` loop untouched
- Pi models (`yolov8n.hef`, `efficientnet_s_birds.hef`) — unchanged
- `config.pi.yaml` — add `stream.source` key; existing V4L2 settings
  remain and are still the default
- `docker-compose.pi.yml` — add port 8765 exposure; otherwise unchanged

---

## Development sequence

### Phase 1: WebSocket frame source + test client (no iPhone needed)

1. Create `src/ws_frame_source.py` with the `WebSocketFrameSource` class.
2. Add `stream.source` config switch to `realtime_pipeline.py`.
3. Add result callback mechanism (send predictions back over WebSocket).
4. Add `ws_port: 8765` exposure to `docker-compose.pi.yml`.
5. Create `scripts/ws_test_client.py` (feed video files to the Pi).
6. Test end-to-end over home LAN: run test client on desktop with a video
   file → Pi processes frames → results appear in test client stdout.

**Deliverable:** Working WebSocket mode testable from any machine with a
video file. No iPhone or hostapd needed.

### Phase 2: iPhone app

1. Scaffold Xcode project in `ios/BirdVision/`.
2. Implement camera capture + JPEG encoding.
3. Implement WebSocket client (send frames, receive results).
4. Implement bounding box + label overlay on camera preview.
5. Add server address configuration UI.
6. Test on iPhone 15 Pro over home WiFi, pointing at Pi's LAN IP.

**Deliverable:** Point phone at bird, see species label on screen.

### Phase 3: hostapd for field use

1. Configure hostapd + dnsmasq on the Pi host.
2. Write toggle scripts (`field-mode-on.sh`, `field-mode-off.sh`).
3. Test: iPhone joins Pi AP, streams frames, gets results.
4. Document the setup in `pi/README.md`.

**Deliverable:** Fully portable, self-contained bird ID rig — Pi in a
backpack, phone in hand, no external network needed.

---

## Open questions

- **Result format:** What JSON shape should the Pi send back? Minimal
  proposal: `{"detections": [{"bbox": [x1,y1,x2,y2], "species": "...",
  "score": 0.85}], "fps": 12.3}`. The phone draws boxes and labels from
  this.
- **Multiple clients:** For now, single client only. The Hailo pipeline
  runs one stream. Supporting multiple phones would require frame
  multiplexing.
- **Audio:** Could the phone also stream audio for call-based ID in the
  future? Out of scope for v1 but worth noting.
- **GPS passthrough:** The phone has GPS. Passing coordinates with each
  frame (or periodically) would let the Pi use location-aware eBird priors
  even in the field. Low effort, high value — consider adding in phase 1.
