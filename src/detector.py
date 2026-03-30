import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List

BIRD_CLASS_ID = 14  # COCO class index for "bird"


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2] in pixels
    confidence: float
    crop: np.ndarray  # BGR image crop


class BirdDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.3, device: str = "cuda"):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[BIRD_CLASS_ID],
            device=self.device,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    detections.append(Detection(
                        bbox=np.array([x1, y1, x2, y2]),
                        confidence=float(box.conf[0]),
                        crop=frame[y1:y2, x1:x2],
                    ))
        return detections
