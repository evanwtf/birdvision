"""Unit tests for src/tracker.py — IoU, centroid distance, Track predictions,
BirdTracker update and pruning logic."""

from dataclasses import dataclass

import numpy as np
import pytest

from src.tracker import BirdTracker, Track, centroid_distance, iou


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------


class TestIoU:
    def test_identical_boxes(self):
        box = np.array([10, 10, 50, 50])
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 30, 30])
        assert iou(box1, box2) == 0.0

    def test_partial_overlap(self):
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([5, 5, 15, 15])
        # Intersection: 5x5=25, Union: 100+100-25=175
        assert iou(box1, box2) == pytest.approx(25.0 / 175.0)

    def test_one_inside_other(self):
        outer = np.array([0, 0, 100, 100])
        inner = np.array([10, 10, 20, 20])
        # Intersection = inner area = 100, Union = 10000+100-100=10000
        assert iou(outer, inner) == pytest.approx(100.0 / 10000.0)

    def test_touching_edges(self):
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([10, 0, 20, 10])
        assert iou(box1, box2) == 0.0


# ---------------------------------------------------------------------------
# centroid_distance
# ---------------------------------------------------------------------------


class TestCentroidDistance:
    def test_same_box(self):
        box = np.array([0, 0, 10, 10])
        assert centroid_distance(box, box) == pytest.approx(0.0)

    def test_horizontal_offset(self):
        box1 = np.array([0, 0, 10, 10])  # centroid (5, 5)
        box2 = np.array([10, 0, 20, 10])  # centroid (15, 5)
        assert centroid_distance(box1, box2) == pytest.approx(10.0)

    def test_diagonal_offset(self):
        box1 = np.array([0, 0, 10, 10])  # centroid (5, 5)
        box2 = np.array([6, 8, 16, 18])  # centroid (11, 13)
        expected = float(np.hypot(6.0, 8.0))  # 10.0
        assert centroid_distance(box1, box2) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Track.best_prediction / best_raw_prediction
# ---------------------------------------------------------------------------


class TestTrackPredictions:
    def test_no_history_returns_none(self):
        t = Track(track_id=0, bbox=np.array([0, 0, 1, 1]))
        assert t.best_prediction is None
        assert t.best_raw_prediction is None

    def test_single_event_uniform_weight(self):
        t = Track(track_id=0, bbox=np.array([0, 0, 1, 1]))
        t.prediction_history = [[("Robin", 0.8), ("Sparrow", 0.2)]]
        t.prediction_weights = [1.0]
        preds = t.best_prediction
        assert preds[0] == ("Robin", pytest.approx(0.8))
        assert preds[1] == ("Sparrow", pytest.approx(0.2))

    def test_weighted_average_across_events(self):
        t = Track(track_id=0, bbox=np.array([0, 0, 1, 1]))
        t.prediction_history = [
            [("Robin", 0.6), ("Sparrow", 0.4)],
            [("Robin", 0.2), ("Sparrow", 0.8)],
        ]
        t.prediction_weights = [3.0, 1.0]
        preds = dict(t.best_prediction)
        # Robin: (0.6*3 + 0.2*1) / 4 = 2.0/4 = 0.5
        # Sparrow: (0.4*3 + 0.8*1) / 4 = 2.0/4 = 0.5
        assert preds["Robin"] == pytest.approx(0.5)
        assert preds["Sparrow"] == pytest.approx(0.5)

    def test_raw_prediction_independent(self):
        t = Track(track_id=0, bbox=np.array([0, 0, 1, 1]))
        t.prediction_history = [[("Robin", 0.9)]]
        t.raw_prediction_history = [[("Robin", 0.5), ("Sparrow", 0.5)]]
        t.prediction_weights = [1.0]
        raw = dict(t.best_raw_prediction)
        assert raw["Robin"] == pytest.approx(0.5)

    def test_empty_weights_defaults_to_uniform(self):
        t = Track(track_id=0, bbox=np.array([0, 0, 1, 1]))
        t.prediction_history = [
            [("Robin", 1.0)],
            [("Robin", 0.0)],
        ]
        t.prediction_weights = []  # empty -> uniform
        preds = dict(t.best_prediction)
        assert preds["Robin"] == pytest.approx(0.5)

    def test_sorted_descending(self):
        t = Track(track_id=0, bbox=np.array([0, 0, 1, 1]))
        t.prediction_history = [[("A", 0.1), ("B", 0.9)]]
        t.prediction_weights = [1.0]
        preds = t.best_prediction
        assert preds[0][0] == "B"
        assert preds[1][0] == "A"


# ---------------------------------------------------------------------------
# BirdTracker.update
# ---------------------------------------------------------------------------


@dataclass
class FakeDetection:
    bbox: np.ndarray


class TestBirdTrackerUpdate:
    def test_first_frame_creates_tracks(self):
        tracker = BirdTracker()
        dets = [FakeDetection(np.array([0, 0, 10, 10])), FakeDetection(np.array([50, 50, 60, 60]))]
        tracks = tracker.update(dets)
        assert len(tracks) == 2
        assert all(t.frame_count == 1 for t in tracks.values())

    def test_empty_detections_increments_disappeared(self):
        tracker = BirdTracker(max_disappeared=5)
        tracker.update([FakeDetection(np.array([0, 0, 10, 10]))])
        tracker.update([])
        track = list(tracker.tracks.values())[0]
        assert track.disappeared == 1

    def test_iou_matching(self):
        tracker = BirdTracker()
        tracker.update([FakeDetection(np.array([10, 10, 50, 50]))])
        tid = list(tracker.tracks.keys())[0]
        # Slightly shifted detection should match
        tracker.update([FakeDetection(np.array([12, 12, 52, 52]))])
        assert tid in tracker.tracks
        assert tracker.tracks[tid].frame_count == 2
        assert tracker.tracks[tid].disappeared == 0

    def test_centroid_fallback_matching(self):
        tracker = BirdTracker(iou_threshold=0.99, centroid_max_distance=0.5)
        tracker.update(
            [FakeDetection(np.array([100, 100, 110, 110]))],
            frame_size=(1920, 1080),
        )
        tid = list(tracker.tracks.keys())[0]
        # Non-overlapping but close detection
        tracker.update(
            [FakeDetection(np.array([112, 112, 122, 122]))],
            frame_size=(1920, 1080),
        )
        assert tid in tracker.tracks
        assert tracker.tracks[tid].frame_count == 2

    def test_unmatched_detection_creates_new_track(self):
        tracker = BirdTracker()
        tracker.update([FakeDetection(np.array([0, 0, 10, 10]))])
        # Far away detection should create a new track
        tracker.update([FakeDetection(np.array([500, 500, 510, 510]))])
        assert len(tracker.tracks) == 2

    def test_pruning_after_max_disappeared(self):
        tracker = BirdTracker(max_disappeared=2)
        tracker.update([FakeDetection(np.array([0, 0, 10, 10]))])
        tid = list(tracker.tracks.keys())[0]
        tracker.update([])  # disappeared=1
        tracker.update([])  # disappeared=2
        assert tid in tracker.tracks  # still within limit
        tracker.update([])  # disappeared=3 -> pruned
        assert tid not in tracker.tracks
        assert tid in tracker.completed_tracks

    def test_completed_tracks_retained(self):
        tracker = BirdTracker(max_disappeared=0)
        tracker.update([FakeDetection(np.array([0, 0, 10, 10]))])
        tid = list(tracker.tracks.keys())[0]
        tracker.update([])  # immediately exceeds max_disappeared=0
        assert tid in tracker.completed_tracks
        assert tracker.completed_tracks[tid].track_id == tid

    def test_matched_detection_idx_set(self):
        tracker = BirdTracker()
        dets = [
            FakeDetection(np.array([0, 0, 10, 10])),
            FakeDetection(np.array([50, 50, 60, 60])),
        ]
        tracks = tracker.update(dets)
        idxs = {t.matched_detection_idx for t in tracks.values()}
        assert idxs == {0, 1}

    def test_matched_detection_idx_cleared_each_frame(self):
        tracker = BirdTracker()
        tracker.update([FakeDetection(np.array([0, 0, 10, 10]))])
        # No detections this frame
        tracker.update([])
        track = list(tracker.tracks.values())[0]
        assert track.matched_detection_idx is None
