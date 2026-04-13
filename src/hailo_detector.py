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

    def __init__(self, hef_path: str, threshold: float = 0.3, vdevice=None):
        self.hef_path  = str(hef_path)
        self.threshold = threshold
        self._output_logged = False
        self._shared_vdevice = vdevice
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

        if self._shared_vdevice is not None:
            self._target = self._shared_vdevice
        else:
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
            for name, val in raw_output.items():
                if isinstance(val, list):
                    logger.info("Output '%s': list len=%d, element type=%s",
                                name, len(val), type(val[0]).__name__ if val else "empty")
                    if val and hasattr(val[0], 'shape'):
                        logger.info("  element[0].shape=%s dtype=%s", val[0].shape, val[0].dtype)
                else:
                    logger.info("Output tensor '%s' shape=%s dtype=%s", name, val.shape, val.dtype)
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

        yolov8_nms_postprocess returns a single key whose value is a Python list
        of per-class numpy arrays (one array per COCO class, 80 total):
            raw_output['yolov8n/yolov8_nms_postprocess'] → list[ndarray]
            list[14] → bird class array, shape (N, 5): [y1, x1, y2, x2, score]

        Coordinates are normalized [0, 1].

        Fallback: if the value is a numpy array instead of a list (single combined
        tensor), tries (N, 6) format with [y1, x1, y2, x2, score, class_id].
        """
        detections: List[Detection] = []

        val = next(iter(raw_output.values()))

        # ---- Primary: list output from yolov8_nms_postprocess ---- #
        # Two sub-formats observed:
        #   A) list of 80 per-class arrays: val[14] → (N,5) [y1,x1,y2,x2,score]
        #   B) list of 1 element which is itself a list of all-class detections:
        #      val[0] → [[y1,x1,y2,x2,score,class_id], ...]
        if isinstance(val, list):
            # Sub-format B: outer list has 1 element containing all detections
            if len(val) == 1 and isinstance(val[0], (list, np.ndarray)):
                inner = val[0]
                if len(inner) == 0:
                    return []
                arr = np.array(inner).reshape(-1, np.array(inner).shape[-1] if hasattr(inner, '__len__') else 1)
                for row in arr:
                    if len(row) >= 6:
                        y1, x1, y2, x2, score, cls = row[0], row[1], row[2], row[3], row[4], row[5]
                        if int(cls) != BIRD_CLASS_ID:
                            continue
                    elif len(row) == 5:
                        y1, x1, y2, x2, score = row
                    else:
                        continue
                    if float(score) < self.threshold:
                        continue
                    d = self._make_detection(x1, y1, x2, y2, float(score), frame_w, frame_h)
                    if d is not None:
                        detections.append(d)
                return detections

            # Sub-format A: 80 per-class arrays
            if BIRD_CLASS_ID >= len(val):
                logger.warning("NMS output list has %d classes, expected >= %d", len(val), BIRD_CLASS_ID + 1)
                return []
            bird_arr = val[BIRD_CLASS_ID]  # (N, 5): [y1, x1, y2, x2, score]
            if bird_arr is None or len(bird_arr) == 0:
                return []
            arr = np.array(bird_arr).reshape(-1, 5)
            for row in arr:
                score = float(row[4])
                if score < self.threshold:
                    continue
                y1, x1, y2, x2 = row[0], row[1], row[2], row[3]
                d = self._make_detection(x1, y1, x2, y2, score, frame_w, frame_h)
                if d is not None:
                    detections.append(d)
            return detections

        # ---- Fallback: single combined numpy tensor ---- #
        if isinstance(val, np.ndarray):
            arr = val.reshape(-1, val.shape[-1])
            ncols = arr.shape[1]
            for row in arr:
                if ncols == 6:
                    y1, x1, y2, x2, score, cls = row
                    if int(cls) != BIRD_CLASS_ID:
                        continue
                elif ncols == 5:
                    y1, x1, y2, x2, score = row
                else:
                    logger.warning("Unexpected detection row width %d", ncols)
                    continue
                if float(score) < self.threshold:
                    continue
                d = self._make_detection(x1, y1, x2, y2, float(score), frame_w, frame_h)
                if d is not None:
                    detections.append(d)
            return detections

        logger.warning("Unrecognized NMS output type: %s — returning no detections", type(val))
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
