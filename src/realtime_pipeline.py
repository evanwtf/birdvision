"""Real-time bird identification pipeline for Raspberry Pi 5 + Hailo-8.

Orchestrates:
    V4L2FrameSource → HailoDetector → BirdTracker → HailoClassifier
                                                     ↳ eBird priors (optional)
                                                     ↳ 1-second log summary

Log format (one line per log_interval_seconds):
    top_species=American Robin confidence=0.87 tracks=2 fps=12.3
    no_detection tracks=0 fps=13.1

Designed to be instantiated and called from scripts/realtime_identify.py.
"""

import logging
import signal
import time
from typing import Optional

import numpy as np
import psutil

from .tracker import BirdTracker
from .stream_capture import V4L2FrameSource
from .hailo_detector import HailoDetector
from .hailo_classifier import HailoClassifier
from .display_overlay import DisplayOverlay

logger = logging.getLogger(__name__)


def _expanded_crop(
    frame: np.ndarray,
    bbox: np.ndarray,
    padding_ratio: float = 0.18,
    padding_ratio_min: float = 0.04,
    closeup_area_ratio: float = 0.06,
    min_crop_area: int = 2500,
) -> Optional[np.ndarray]:
    """Extract a padded crop from frame.  Mirrors pipeline.py _expanded_crop logic."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    box_area = box_w * box_h
    if box_area < min_crop_area:
        return None

    frame_area = max(1, frame.shape[0] * frame.shape[1])
    area_ratio = box_area / frame_area

    if closeup_area_ratio > 0:
        progress = min(area_ratio / closeup_area_ratio, 1.0)
    else:
        progress = 0.0
    pad_ratio = padding_ratio - (padding_ratio - padding_ratio_min) * progress
    pad_x = int(round(box_w * pad_ratio))
    pad_y = int(round(box_h * pad_ratio))

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(frame.shape[1], x2 + pad_x)
    cy2 = min(frame.shape[0], y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame[cy1:cy2, cx1:cx2]


class RealtimePipeline:
    """End-to-end real-time bird identification pipeline.

    Args:
        config: Parsed config.pi.yaml dict (as returned by yaml.safe_load)
    """

    def __init__(self, config: dict):
        self._config = config
        self._stop   = False

        stream_cfg = config.get("stream", {})
        det_cfg    = config.get("detector", {})
        cls_cfg    = config.get("classifier", {})
        trk_cfg    = config.get("tracker", {})
        out_cfg    = config.get("output", {})

        # Capture source — V4L2 (backyard camera) or WebSocket (phone sidecar)
        source_type = stream_cfg.get("source", "v4l2")
        if source_type == "websocket":
            from .ws_frame_source import WebSocketFrameSource
            self._source = WebSocketFrameSource(
                host=stream_cfg.get("ws_host", "0.0.0.0"),
                port=stream_cfg.get("ws_port", 8765),
                static_dir=stream_cfg.get("ws_static_dir"),
                ssl_certfile=stream_cfg.get("ws_ssl_certfile"),
                ssl_keyfile=stream_cfg.get("ws_ssl_keyfile"),
                ssl_cert_dir=stream_cfg.get("ws_ssl_cert_dir", "/data/certs"),
            )
        else:
            self._source = V4L2FrameSource(
                device=stream_cfg.get("device", "/dev/video0"),
                width=stream_cfg.get("width", 1920),
                height=stream_cfg.get("height", 1080),
                fps=int(stream_cfg.get("framerate", 60)),
            )

        # Single VDevice shared by detector and classifier — the Hailo-8 chip
        # can only be opened once; both models run as separate network groups.
        from hailo_platform import VDevice
        logger.info("Opening Hailo VDevice")
        self._vdevice = VDevice()

        # Hailo detector
        self._detector = HailoDetector(
            hef_path=det_cfg["hef"],
            threshold=det_cfg.get("confidence", 0.4),
            vdevice=self._vdevice,
        )

        # Hailo classifier
        self._classifier = HailoClassifier(
            hef_path=cls_cfg["hef"],
            labels_path=cls_cfg["labels"],
            top_k=cls_cfg.get("top_k", 20),
            vdevice=self._vdevice,
        )

        # Tracker
        self._tracker = BirdTracker(
            max_disappeared=trk_cfg.get("max_disappeared", 120),
            iou_threshold=trk_cfg.get("iou_threshold", 0.2),
            centroid_max_distance=trk_cfg.get("centroid_max_distance", 0.18),
        )

        # Classifier settings
        self._classify_every_n    = cls_cfg.get("classify_every_n_frames", 10)
        self._crop_padding        = cls_cfg.get("crop_padding_ratio", 0.18)
        self._crop_padding_min    = cls_cfg.get("crop_padding_ratio_min", 0.04)
        self._crop_closeup_ratio  = cls_cfg.get("crop_closeup_area_ratio", 0.06)
        self._min_crop_area       = cls_cfg.get("min_crop_area", 2500)
        self._min_event_conf      = cls_cfg.get("min_event_confidence", 0.35)
        self._log_interval        = out_cfg.get("log_interval_seconds", 1.0)
        self._stats_interval      = out_cfg.get("stats_interval_seconds", 30.0)

        # eBird priors (optional — skip if db not present)
        self._metadata = None
        meta_cfg = config.get("metadata", {})
        if meta_cfg.get("ebird_db"):
            try:
                from .metadata import MetadataPrior
                self._metadata = MetadataPrior(
                    db_path=meta_cfg["ebird_db"],
                    fips=meta_cfg.get("ebird_fips", "US-NY-059"),
                    prior_mode=meta_cfg.get("prior_mode", "seasonal"),
                    latitude=meta_cfg.get("latitude"),
                    longitude=meta_cfg.get("longitude"),
                    local_priors_file=meta_cfg.get("local_priors_file"),
                )
                logger.info("eBird priors loaded from %s", meta_cfg["ebird_db"])
            except Exception as exc:
                logger.warning("Could not load eBird priors: %s — running without", exc)

        # Optional framebuffer display overlay (Pi Touch Display 2)
        self._display: Optional[DisplayOverlay] = None
        disp_cfg = config.get("display", {})
        if disp_cfg.get("enabled", False):
            self._display = DisplayOverlay(
                device=disp_cfg.get("device", "/dev/fb0"),
                rotate=int(disp_cfg.get("rotate", 0)),
                show_fps=disp_cfg.get("show_fps", False),
                show_boxes=disp_cfg.get("show_boxes", True),
            )
        self._caption_ttl = float(disp_cfg.get("caption_ttl_seconds", 3.0))

        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the pipeline until SIGINT/SIGTERM or stop() is called."""
        logger.info("Real-time pipeline starting")

        frame_count         = 0
        t_window_start      = time.monotonic()
        t_stats_last        = time.monotonic()
        window_events: list = []   # (species, score) from classify events this window
        caption_species: Optional[str]  = None
        caption_score:   Optional[float] = None
        caption_expiry:  float           = 0.0
        current_fps:     float           = 0.0

        for frame_no, frame in self._source.frames():
            if self._stop:
                break

            frame_count += 1
            frame_h, frame_w = frame.shape[:2]

            # --- Detection ---
            detections = self._detector.detect(frame)

            # --- Tracking ---
            tracks = self._tracker.update(detections, frame_size=(frame_w, frame_h))

            # --- Classification (every N frames per track) ---
            crops_to_classify = []
            track_ids_to_classify = []

            for tid, track in tracks.items():
                if track.matched_detection_idx is None:
                    continue
                if (frame_no - track.last_classified_frame) < self._classify_every_n:
                    continue
                det = detections[track.matched_detection_idx]
                crop = _expanded_crop(
                    frame, det.bbox,
                    padding_ratio=self._crop_padding,
                    padding_ratio_min=self._crop_padding_min,
                    closeup_area_ratio=self._crop_closeup_ratio,
                    min_crop_area=self._min_crop_area,
                )
                if crop is None:
                    continue
                crops_to_classify.append(crop)
                track_ids_to_classify.append(tid)

            if crops_to_classify:
                results = self._classifier.classify_batch(crops_to_classify)

                client_meta = getattr(self._source, "client_metadata", {})
                meta_lat = client_meta.get("lat")
                meta_lon = client_meta.get("lon")

                for tid, preds in zip(track_ids_to_classify, results):
                    track = tracks[tid]
                    track.last_classified_frame = frame_no

                    if self._metadata is not None:
                        preds = self._metadata.apply(
                            preds, latitude=meta_lat, longitude=meta_lon,
                        )

                    top_species, top_score = preds[0] if preds else ("unknown", 0.0)
                    if top_score >= self._min_event_conf:
                        track.prediction_history.append(preds)
                        window_events.append((top_species, top_score))
                        logger.debug(
                            "track=%d  %s  %.3f", tid, top_species, top_score
                        )
                        caption_species = top_species
                        caption_score   = top_score
                        caption_expiry  = time.monotonic() + self._caption_ttl

            # --- Push to framebuffer display (if enabled) ---
            if self._display is not None and self._display.enabled:
                now_display = time.monotonic()
                if caption_expiry and now_display >= caption_expiry:
                    caption_species = None
                    caption_score   = None
                bboxes = [d.bbox for d in detections]
                self._display.post(
                    frame, bboxes,
                    top_species=caption_species,
                    top_score=caption_score,
                    fps=current_fps,
                )

            # --- Send results to WebSocket client (sidecar mode) ---
            if hasattr(self._source, "send_result"):
                ws_detections = []
                for tid, track in tracks.items():
                    if track.matched_detection_idx is None:
                        continue
                    det = detections[track.matched_detection_idx]
                    bbox_norm = [
                        float(det.bbox[0]) / frame_w,
                        float(det.bbox[1]) / frame_h,
                        float(det.bbox[2]) / frame_w,
                        float(det.bbox[3]) / frame_h,
                    ]
                    species = None
                    species_score = None
                    if track.prediction_history:
                        latest = track.prediction_history[-1]
                        if latest:
                            species = latest[0][0]
                            species_score = round(latest[0][1], 3)
                    ws_detections.append({
                        "bbox": [round(c, 4) for c in bbox_norm],
                        "track_id": tid,
                        "confidence": round(float(det.confidence), 3),
                        "species": species,
                        "species_score": species_score,
                    })
                client_meta = self._source.client_metadata
                self._source.send_result({
                    "frame_id": client_meta.get("frame_id", frame_no),
                    "detections": ws_detections,
                    "fps": current_fps,
                })

            # --- 1-second summary log ---
            now = time.monotonic()
            elapsed = now - t_window_start
            if elapsed >= self._log_interval:
                fps = frame_count / elapsed
                current_fps = fps
                active_tracks = len([t for t in tracks.values() if t.disappeared == 0])

                if window_events:
                    # Best species in the window = highest single-event confidence
                    best_species, best_score = max(window_events, key=lambda x: x[1])
                    logger.info(
                        "top_species=%s confidence=%.2f tracks=%d fps=%.1f",
                        best_species, best_score, active_tracks, fps,
                    )
                else:
                    logger.info("no_detection tracks=%d fps=%.1f", active_tracks, fps)

                # Reset window
                frame_count    = 0
                t_window_start = now
                window_events  = []

            # --- System stats (every stats_interval_seconds) ---
            if now - t_stats_last >= self._stats_interval:
                self._log_system_stats()
                t_stats_last = now

        self._source.stop()
        if self._display is not None:
            self._display.close()
        logger.info("Real-time pipeline stopped")

    def _log_system_stats(self) -> None:
        """Log Pi system health: CPU temp, load, frequency, memory."""
        parts = []

        # CPU temperature (reads /sys/class/thermal via psutil)
        try:
            temps = psutil.sensors_temperatures()
            # Pi 5 exposes temp under 'cpu_thermal' or 'coretemp'
            for key in ("cpu_thermal", "coretemp", "soc_thermal"):
                if key in temps and temps[key]:
                    parts.append(f"temp={temps[key][0].current:.1f}C")
                    break
        except (AttributeError, OSError):
            pass

        # CPU load (1-minute average from /proc/loadavg)
        load1, load5, *_ = psutil.getloadavg()
        parts.append(f"load={load1:.2f}/{load5:.2f}")

        # CPU utilization %
        cpu_pct = psutil.cpu_percent(interval=None)
        parts.append(f"cpu={cpu_pct:.0f}%")

        # CPU frequency
        try:
            freq = psutil.cpu_freq()
            if freq:
                parts.append(f"freq={freq.current:.0f}MHz")
        except (AttributeError, OSError):
            pass

        # Memory
        mem = psutil.virtual_memory()
        parts.append(f"mem={mem.percent:.0f}%")

        # Fan speed (Pi 5 fan via hwmon — not always present)
        try:
            fans = psutil.sensors_fans()
            for entries in fans.values():
                if entries:
                    parts.append(f"fan={entries[0].current:.0f}rpm")
                    break
        except (AttributeError, OSError):
            pass

        logger.info("system %s", " ".join(parts))

    def stop(self) -> None:
        """Request a clean shutdown after the current frame."""
        self._stop = True
        self._source.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Signal %d received — shutting down", signum)
        self.stop()
