from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    disappeared: int = 0
    frame_count: int = 0
    last_classified_frame: int = -9999
    matched_detection_idx: Optional[int] = None
    # Weighted (post-prior) predictions, one list per classification event
    prediction_history: List[List[Tuple[str, float]]] = field(default_factory=list)
    prediction_weights: List[float] = field(default_factory=list)
    # Raw visual predictions (pre-prior), parallel to prediction_history
    raw_prediction_history: List[List[Tuple[str, float]]] = field(default_factory=list)

    @property
    def best_raw_prediction(self) -> Optional[List[Tuple[str, float]]]:
        """Weighted average of raw visual scores (before eBird priors)."""
        if not self.raw_prediction_history:
            return None
        weights = self.prediction_weights if self.prediction_weights else [1.0] * len(self.raw_prediction_history)
        total_weight = sum(weights)
        scores: Dict[str, float] = {}
        for preds, w in zip(self.raw_prediction_history, weights):
            for species, prob in preds:
                scores[species] = scores.get(species, 0.0) + prob * w
        averaged = [(s, p / total_weight) for s, p in scores.items()]
        return sorted(averaged, key=lambda x: -x[1])

    @property
    def best_prediction(self) -> Optional[List[Tuple[str, float]]]:
        """Weighted average of probabilities across all classification events."""
        if not self.prediction_history:
            return None
        weights = self.prediction_weights if self.prediction_weights else [1.0] * len(self.prediction_history)
        total_weight = sum(weights)
        scores: Dict[str, float] = {}
        for preds, w in zip(self.prediction_history, weights):
            for species, prob in preds:
                scores[species] = scores.get(species, 0.0) + prob * w
        averaged = [(s, p / total_weight) for s, p in scores.items()]
        return sorted(averaged, key=lambda x: -x[1])


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


def centroid_distance(box1: np.ndarray, box2: np.ndarray) -> float:
    c1x = (box1[0] + box1[2]) / 2.0
    c1y = (box1[1] + box1[3]) / 2.0
    c2x = (box2[0] + box2[2]) / 2.0
    c2y = (box2[1] + box2[3]) / 2.0
    return float(np.hypot(c1x - c2x, c1y - c2y))


class BirdTracker:
    def __init__(
        self,
        max_disappeared: int = 30,
        iou_threshold: float = 0.3,
        centroid_max_distance: float = 0.15,
    ):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.centroid_max_distance = centroid_max_distance
        self.next_id = 0
        self.tracks: Dict[int, Track] = OrderedDict()
        self.completed_tracks: Dict[int, Track] = {}  # pruned tracks, kept for summary

    def update(self, detections, frame_size: Optional[Tuple[int, int]] = None) -> Dict[int, Track]:
        """Update tracks with new detections. Returns all active tracks."""
        for track in self.tracks.values():
            track.matched_detection_idx = None

        if not detections:
            for track in self.tracks.values():
                track.disappeared += 1
            self._prune()
            return self.tracks

        if not self.tracks:
            for di, det in enumerate(detections):
                self._new_track(det.bbox, matched_detection_idx=di)
            return self.tracks

        track_ids = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[tid].bbox for tid in track_ids])
        det_boxes = np.array([d.bbox for d in detections])

        # IoU matrix: tracks × detections
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        for i, tb in enumerate(track_boxes):
            for j, db in enumerate(det_boxes):
                iou_matrix[i, j] = iou(tb, db)

        matched_tracks, matched_dets = set(), set()

        def match_detection(track_idx: int, det_idx: int):
            tid = track_ids[track_idx]
            track = self.tracks[tid]
            track.bbox = detections[det_idx].bbox
            track.disappeared = 0
            track.frame_count += 1
            track.matched_detection_idx = det_idx
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)

        while True:
            if iou_matrix.size == 0 or iou_matrix.max() < self.iou_threshold:
                break
            ti, di = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if ti not in matched_tracks and di not in matched_dets:
                match_detection(ti, di)
            iou_matrix[ti, :] = 0
            iou_matrix[:, di] = 0

        if frame_size and self.centroid_max_distance > 0:
            max_distance_px = self.centroid_max_distance * float(np.hypot(*frame_size))
            unmatched_track_indices = [ti for ti in range(len(track_ids)) if ti not in matched_tracks]
            unmatched_det_indices = [di for di in range(len(detections)) if di not in matched_dets]
            if unmatched_track_indices and unmatched_det_indices:
                distance_matrix = np.full(
                    (len(unmatched_track_indices), len(unmatched_det_indices)),
                    np.inf,
                    dtype=float,
                )
                for row_idx, ti in enumerate(unmatched_track_indices):
                    for col_idx, di in enumerate(unmatched_det_indices):
                        distance_matrix[row_idx, col_idx] = centroid_distance(
                            track_boxes[ti],
                            det_boxes[di],
                        )

                while np.isfinite(distance_matrix).any():
                    row_idx, col_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
                    if distance_matrix[row_idx, col_idx] > max_distance_px:
                        break
                    ti = unmatched_track_indices[row_idx]
                    di = unmatched_det_indices[col_idx]
                    if ti not in matched_tracks and di not in matched_dets:
                        match_detection(ti, di)
                    distance_matrix[row_idx, :] = np.inf
                    distance_matrix[:, col_idx] = np.inf

        for ti, tid in enumerate(track_ids):
            if ti not in matched_tracks:
                self.tracks[tid].disappeared += 1

        for di, det in enumerate(detections):
            if di not in matched_dets:
                self._new_track(det.bbox, matched_detection_idx=di)

        self._prune()
        return self.tracks

    def _new_track(self, bbox: np.ndarray, matched_detection_idx: Optional[int] = None):
        self.tracks[self.next_id] = Track(
            track_id=self.next_id,
            bbox=bbox,
            frame_count=1,
            matched_detection_idx=matched_detection_idx,
        )
        self.next_id += 1

    def _prune(self):
        surviving = {}
        for k, v in self.tracks.items():
            if v.disappeared <= self.max_disappeared:
                surviving[k] = v
            else:
                self.completed_tracks[k] = v
        self.tracks = surviving
