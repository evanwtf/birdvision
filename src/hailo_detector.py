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
from typing import List, Tuple

import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    """Bird detection result — mirrors detector.py Detection for interface compatibility."""

    bbox: np.ndarray  # [x1, y1, x2, y2] pixel coords
    confidence: float
    crop: np.ndarray  # BGR image crop


logger = logging.getLogger(__name__)

BIRD_CLASS_ID = 14  # COCO class index for "bird"
INPUT_SIZE = 640  # YOLOv8 canonical input resolution


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

    def __init__(
        self,
        hef_path: str,
        threshold: float = 0.3,
        vdevice=None,
        fallback_crop_ratio: float = 0.5,
        dedupe_iou_threshold: float = 0.5,
    ):
        self.hef_path = str(hef_path)
        self.threshold = threshold
        self.fallback_crop_ratio = fallback_crop_ratio
        self.dedupe_iou_threshold = dedupe_iou_threshold
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
        network_groups = self._target.configure(hef, configure_params)
        self._network_group = network_groups[0]
        self._network_group_params = self._network_group.create_params()

        # UINT8 input — normalization is baked into the compiled model
        self._input_vstream_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.UINT8
        )
        # FLOAT32 output for easy post-processing
        self._output_vstream_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )

        input_info = hef.get_input_vstream_infos()
        output_info = hef.get_output_vstream_infos()
        self._input_name = input_info[0].name
        self._output_names = [o.name for o in output_info]
        logger.info("Hailo detector — input: %s", self._input_name)
        logger.info(
            "Hailo detector — outputs (%d): %s", len(self._output_names), self._output_names
        )
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
                    logger.info(
                        "Output '%s': list len=%d, element type=%s",
                        name,
                        len(val),
                        type(val[0]).__name__ if val else "empty",
                    )
                    if val and hasattr(val[0], "shape"):
                        logger.info("  element[0].shape=%s dtype=%s", val[0].shape, val[0].dtype)
                else:
                    logger.info("Output tensor '%s' shape=%s dtype=%s", name, val.shape, val.dtype)
            self._output_logged = True

        logger.debug("Hailo detect: %.1f ms", elapsed_ms)

        detections = self._parse_detections(raw_output, frame_h, frame_w)
        crops_added = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_w, x2)
            y2 = min(frame_h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops_added.append(
                    Detection(
                        bbox=np.array([x1, y1, x2, y2]),
                        confidence=det.confidence,
                        crop=crop,
                    )
                )
        logger.debug("Detections: %d birds above threshold %.2f", len(crops_added), self.threshold)
        return crops_added

    def detect_zoomed(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on the center crop and map boxes to full-frame coords."""
        region, origin = self._center_crop_region(frame)
        detections = self.detect(region)
        origin_x, origin_y = origin
        remapped: List[Detection] = []

        frame_h, frame_w = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            gx1 = max(0, x1 + origin_x)
            gy1 = max(0, y1 + origin_y)
            gx2 = min(frame_w, x2 + origin_x)
            gy2 = min(frame_h, y2 + origin_y)
            if gx2 <= gx1 or gy2 <= gy1:
                continue
            remapped.append(
                Detection(
                    bbox=np.array([gx1, gy1, gx2, gy2]),
                    confidence=det.confidence,
                    crop=frame[gy1:gy2, gx1:gx2],
                )
            )

        return self._dedupe(remapped)

    def _center_crop_region(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = frame.shape[:2]
        crop_ratio = min(max(self.fallback_crop_ratio, 0.2), 1.0)
        crop_w = max(1, int(round(w * crop_ratio)))
        crop_h = max(1, int(round(h * crop_ratio)))
        x1 = max(0, (w - crop_w) // 2)
        y1 = max(0, (h - crop_h) // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        return frame[y1:y2, x1:x2], (x1, y1)

    def _dedupe(self, detections: List[Detection]) -> List[Detection]:
        kept: List[Detection] = []
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if any(
                self._iou(det.bbox, existing.bbox) >= self.dedupe_iou_threshold for existing in kept
            ):
                continue
            kept.append(det)
        return kept

    @staticmethod
    def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / max(area1 + area2 - inter, 1)

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
        # Confirmed format: val = [per_class_list]  (outer list length 1)
        #   per_class_list = [class0_arr, class1_arr, ..., class79_arr]  (80 elements)
        #   class_arr = ndarray shape (N, 5): [y1, x1, y2, x2, score], normalized [0,1]
        if isinstance(val, list):
            # Unwrap outer list if needed
            per_class = val[0] if (len(val) == 1 and isinstance(val[0], list)) else val

            if BIRD_CLASS_ID >= len(per_class):
                logger.warning(
                    "NMS per-class list has %d entries, expected >= %d",
                    len(per_class),
                    BIRD_CLASS_ID + 1,
                )
                return []

            bird_arr = per_class[BIRD_CLASS_ID]
            if bird_arr is None or (hasattr(bird_arr, "__len__") and len(bird_arr) == 0):
                return []

            arr = np.array(bird_arr, dtype=np.float32).reshape(-1, 5)
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
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        score: float,
        frame_w: int,
        frame_h: int,
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
