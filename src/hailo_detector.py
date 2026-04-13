"""Hailo-8 YOLOv8n detection backend for the Pi real-time pipeline.

Implements the same interface as detector.py (BirdDetector.detect) so it can
be swapped in by realtime_pipeline.py without changing call sites.

Requires hailort Python bindings (hailo_platform) — Pi only.
Install: uv sync --group pi

Output format note
------------------
hailomz-compiled yolov8n produces NMS-post-processed output tensors.
The Hailo model zoo v2.18 NMS config bakes per-class detections into the HEF.
Two formats are possible depending on how the NMS was linked:

  A) Per-class mode (80 output tensors, one per COCO class):
     Each tensor shape: (1, 1, max_dets_per_class, 5)
     Values: [y_min, x_min, y_max, x_max, confidence]  — normalized [0, 1]

  B) Combined mode (single output tensor):
     Shape: (1, max_dets, 6)
     Values: [y_min, x_min, y_max, x_max, confidence, class_id]

The init method logs all output layer names + shapes on first inference so that
if the format differs from the above, it can be diagnosed from the Pi logs.
"""

import logging
import time
from typing import List

import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    """Bird detection result — mirrors detector.py Detection for interface compatibility."""
    bbox: np.ndarray   # [x1, y1, x2, y2] pixel coords
    confidence: float
    crop: np.ndarray   # BGR image crop

logger = logging.getLogger(__name__)

BIRD_CLASS_ID = 14   # COCO class index for "bird"
INPUT_SIZE    = 640  # YOLOv8 canonical input resolution


def _preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Resize to 640×640, convert BGR→RGB, return NHWC uint8 (1, 640, 640, 3).

    The hailomz-compiled yolov8n model has normalization baked in during
    quantization calibration, so we pass raw uint8 pixel values.
    """
    import cv2
    resized = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb[np.newaxis].astype(np.uint8)  # (1, 640, 640, 3)


class HailoDetector:
    """YOLOv8n bird detector via Hailo-8 HEF.

    Drop-in replacement for BirdDetector from detector.py (detect() method).

    Args:
        hef_path:   Path to yolov8n.hef (compiled by hailomz for hailo8)
        threshold:  Minimum confidence to keep a detection (default 0.3)
    """

    def __init__(self, hef_path: str, threshold: float = 0.3):
        self.hef_path  = str(hef_path)
        self.threshold = threshold
        self._output_logged = False
        self._init_hailo()

    def _init_hailo(self) -> None:
        from hailo_platform import (
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        logger.info("Loading YOLOv8n HEF: %s", self.hef_path)
        hef = HEF(self.hef_path)

        self._target = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef=hef, interface=HailoStreamInterface.PCIe
        )
        network_groups         = self._target.configure(hef, configure_params)
        self._network_group    = network_groups[0]
        self._network_group_params = self._network_group.create_params()

        # UINT8 input — normalization is baked into the compiled model
        self._input_vstream_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.UINT8
        )
        # FLOAT32 output for easy post-processing
        self._output_vstream_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )

        input_info  = hef.get_input_vstream_infos()
        output_info = hef.get_output_vstream_infos()
        self._input_name   = input_info[0].name
        self._output_names = [o.name for o in output_info]
        logger.info("Hailo detector — input: %s", self._input_name)
        logger.info("Hailo detector — outputs (%d): %s", len(self._output_names), self._output_names)
        logger.info("HailoDetector ready (threshold=%.2f)", self.threshold)

    # ------------------------------------------------------------------
    # Public interface (matches detector.py BirdDetector)
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLOv8 inference and return bird detections.

        Args:
            frame: BGR frame from cv2 (any resolution)

        Returns:
            List of Detection(bbox, confidence, crop) filtered to birds only.
        """
        frame_h, frame_w = frame.shape[:2]
        input_batch = _preprocess_frame(frame)

        t0 = time.perf_counter()
        raw_output = self._run_inference(input_batch)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not self._output_logged:
            for name, arr in raw_output.items():
                logger.info("Output tensor '%s' shape=%s dtype=%s", name, arr.shape, arr.dtype)
            self._output_logged = True

        logger.debug("Hailo detect: %.1f ms", elapsed_ms)

        detections = self._parse_detections(raw_output, frame_h, frame_w)
        crops_added = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(frame_w, x2);  y2 = min(frame_h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops_added.append(Detection(
                    bbox=np.array([x1, y1, x2, y2]),
                    confidence=det.confidence,
                    crop=crop,
                ))
        logger.debug("Detections: %d birds above threshold %.2f", len(crops_added), self.threshold)
        return crops_added

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_inference(self, input_batch: np.ndarray) -> dict:
        from hailo_platform import InferVStreams
        with InferVStreams(
            self._network_group,
            self._input_vstream_params,
            self._output_vstream_params,
        ) as pipeline:
            with self._network_group.activate(self._network_group_params):
                return pipeline.infer({self._input_name: input_batch})

    def _parse_detections(self, raw_output: dict, frame_h: int, frame_w: int) -> List[Detection]:
        """Decode Hailo NMS output into Detection objects.

        Tries two formats emitted by hailomz-compiled yolov8n:
          A) Per-class tensors (one entry per COCO class, index 14 = bird)
          B) Combined tensor (all classes, class_id in last column)

        Coordinates are assumed to be normalized [0, 1]; multiplied by frame dims
        to get pixel coords. If detections look like pixel coords already (values
        consistently > 1.0) they are used as-is.
        """
        detections: List[Detection] = []

        # ---- Format A: per-class output (80 tensors) ---- #
        if len(raw_output) >= 80:
            # Keys may be ordered; bird class is at index BIRD_CLASS_ID (14)
            keys = list(raw_output.keys())
            if BIRD_CLASS_ID < len(keys):
                bird_tensor = raw_output[keys[BIRD_CLASS_ID]]
                # Shape variants: (1,1,max_dets,5) or (1,max_dets,5)
                arr = bird_tensor.reshape(-1, 5) if bird_tensor.ndim >= 3 else bird_tensor.reshape(-1, 5)
                for row in arr:
                    score = float(row[4])
                    if score < self.threshold:
                        continue
                    y1, x1, y2, x2 = row[0], row[1], row[2], row[3]
                    detections.append(self._make_detection(x1, y1, x2, y2, score, frame_w, frame_h))
            return [d for d in detections if d is not None]

        # ---- Format B: single combined output tensor ---- #
        if len(raw_output) == 1:
            arr = next(iter(raw_output.values()))
            # Shape: (1, max_dets, 6) or (1, 1, max_dets, 6)
            arr = arr.reshape(-1, arr.shape[-1])
            ncols = arr.shape[1]
            for row in arr:
                if ncols == 6:
                    y1, x1, y2, x2, score, cls = row
                    if int(cls) != BIRD_CLASS_ID:
                        continue
                elif ncols == 5:
                    # No class column — assume bird-only model
                    y1, x1, y2, x2, score = row
                else:
                    logger.warning("Unexpected detection row length %d — skipping", ncols)
                    continue
                if float(score) < self.threshold:
                    continue
                detections.append(self._make_detection(x1, y1, x2, y2, float(score), frame_w, frame_h))
            return [d for d in detections if d is not None]

        # ---- Fallback: log and return empty ---- #
        logger.warning(
            "Unrecognized output format: %d tensors with shapes %s — "
            "edit _parse_detections() to match. Returning no detections.",
            len(raw_output),
            {k: v.shape for k, v in raw_output.items()},
        )
        return []

    @staticmethod
    def _make_detection(
        x1: float, y1: float, x2: float, y2: float,
        score: float, frame_w: int, frame_h: int,
    ):
        # If coords look normalized [0,1], scale to pixels
        if max(x1, y1, x2, y2) <= 1.0:
            x1, x2 = x1 * frame_w, x2 * frame_w
            y1, y2 = y1 * frame_h, y2 * frame_h
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return Detection(
            bbox=np.array([x1, y1, x2, y2]),
            confidence=score,
            crop=np.empty((0,)),  # filled by caller with original-res frame
        )
