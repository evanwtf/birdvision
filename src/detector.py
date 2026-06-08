from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

BIRD_CLASS_ID = 14  # COCO class index for "bird"


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2] in pixels
    confidence: float
    crop: np.ndarray  # BGR image crop


class BirdDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.3,
        device: str = "cuda",
        fallback_crop_ratio: float = 0.5,
        dedupe_iou_threshold: float = 0.5,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.fallback_crop_ratio = fallback_crop_ratio
        self.dedupe_iou_threshold = dedupe_iou_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self._detect_in_region(frame, origin=(0, 0))

    def detect_zoomed(self, frame: np.ndarray) -> list[Detection]:
        region, origin = self._center_crop_region(frame)
        detections = self._detect_in_region(region, origin=origin, full_frame=frame)
        return self._dedupe(detections)

    def _center_crop_region(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = frame.shape[:2]
        crop_ratio = min(max(self.fallback_crop_ratio, 0.2), 1.0)
        crop_w = max(1, int(round(w * crop_ratio)))
        crop_h = max(1, int(round(h * crop_ratio)))
        x1 = max(0, (w - crop_w) // 2)
        y1 = max(0, (h - crop_h) // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        return frame[y1:y2, x1:x2], (x1, y1)

    def _detect_in_region(
        self,
        frame: np.ndarray,
        *,
        origin: tuple[int, int],
        full_frame: np.ndarray | None = None,
    ) -> list[Detection]:
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[BIRD_CLASS_ID],
            device=self.device,
            verbose=False,
        )
        detections = []
        origin_x, origin_y = origin
        target_frame = full_frame if full_frame is not None else frame
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    gx1 = max(0, x1 + origin_x)
                    gy1 = max(0, y1 + origin_y)
                    gx2 = min(target_frame.shape[1], x2 + origin_x)
                    gy2 = min(target_frame.shape[0], y2 + origin_y)
                    detections.append(
                        Detection(
                            bbox=np.array([gx1, gy1, gx2, gy2]),
                            confidence=float(box.conf[0]),
                            crop=target_frame[gy1:gy2, gx1:gx2],
                        )
                    )
        return detections

    def _dedupe(self, detections: list[Detection]) -> list[Detection]:
        kept: list[Detection] = []
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
