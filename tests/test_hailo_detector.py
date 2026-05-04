"""Tests for Pi Hailo detector helpers."""

import numpy as np

from src.hailo_detector import Detection, HailoDetector


class StubHailoDetector(HailoDetector):
    def _init_hailo(self) -> None:
        pass


def test_detect_zoomed_remaps_center_crop_boxes_to_full_frame():
    detector = StubHailoDetector("dummy.hef", fallback_crop_ratio=0.5)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    calls = []

    def fake_detect(region):
        calls.append(region.shape)
        return [
            Detection(
                bbox=np.array([10, 20, 30, 40]),
                confidence=0.8,
                crop=region[20:40, 10:30],
            )
        ]

    detector.detect = fake_detect

    detections = detector.detect_zoomed(frame)

    assert calls == [(50, 100, 3)]
    assert len(detections) == 1
    assert detections[0].bbox.tolist() == [60, 45, 80, 65]
    assert detections[0].crop.shape == (20, 20, 3)


def test_detect_zoomed_dedupes_overlapping_remapped_boxes():
    detector = StubHailoDetector("dummy.hef", fallback_crop_ratio=0.5)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    def fake_detect(region):
        return [
            Detection(bbox=np.array([10, 20, 30, 40]), confidence=0.7, crop=region[20:40, 10:30]),
            Detection(bbox=np.array([11, 21, 31, 41]), confidence=0.9, crop=region[21:41, 11:31]),
        ]

    detector.detect = fake_detect

    detections = detector.detect_zoomed(frame)

    assert len(detections) == 1
    assert detections[0].confidence == 0.9
    assert detections[0].bbox.tolist() == [61, 46, 81, 66]
