"""Tests that Hailo detector/classifier serialize device access via a lock.

In sidecar mode the /upload handler runs Hailo inference on a thread-pool
thread while the live frame loop calls the same detector/classifier instances.
Both re-enter network_group.activate() on the shared VDevice, which is not
safe to do concurrently. A shared inference lock must serialize those calls.

These tests stub out the HailoRT bindings (not installed in CI) by injecting a
fake hailo_platform module, and verify the real lock is held across infer().
"""

import sys
import threading
from unittest.mock import MagicMock, patch

import numpy as np

from src.hailo_classifier import HailoClassifier
from src.hailo_detector import HailoDetector


class StubHailoDetector(HailoDetector):
    def _init_hailo(self) -> None:  # bypass real HailoRT init
        pass


class StubHailoClassifier(HailoClassifier):
    def __init__(self, *args, **kwargs):  # bypass label load + HailoRT init
        self.top_k = kwargs.get("top_k", 5)
        self.species_names = ["a", "b", "c"]
        self._inference_lock = kwargs.get("inference_lock") or threading.Lock()


def _fake_hailo_platform(infer_return, on_infer):
    """Build a fake hailo_platform module whose InferVStreams is a no-op CM."""
    module = MagicMock()
    pipeline = MagicMock()

    def infer(_data):
        on_infer()
        return infer_return

    pipeline.infer.side_effect = infer
    cm = MagicMock()
    cm.__enter__.return_value = pipeline
    cm.__exit__.return_value = False
    module.InferVStreams.return_value = cm
    return module


def test_detector_accepts_and_holds_shared_inference_lock():
    lock = threading.Lock()
    detector = StubHailoDetector("dummy.hef", inference_lock=lock)
    detector._network_group = MagicMock()
    detector._network_group_params = MagicMock()
    detector._input_vstream_params = MagicMock()
    detector._output_vstream_params = MagicMock()
    detector._input_name = "in"

    assert detector._inference_lock is lock

    held = []
    fake = _fake_hailo_platform({"out": np.zeros((1, 5))}, lambda: held.append(lock.locked()))
    with patch.dict(sys.modules, {"hailo_platform": fake}):
        detector._run_inference(np.zeros((1, 640, 640, 3), dtype=np.uint8))

    assert held == [True]  # lock was held during infer()
    assert not lock.locked()  # and released afterward


def test_classifier_accepts_and_holds_shared_inference_lock():
    lock = threading.Lock()
    classifier = StubHailoClassifier("dummy.hef", "labels.json", inference_lock=lock)
    classifier._network_group = MagicMock()
    classifier._network_group_params = MagicMock()
    classifier._input_vstream_params = MagicMock()
    classifier._output_vstream_params = MagicMock()
    classifier._input_name = "in"
    classifier._output_name = "out"

    assert classifier._inference_lock is lock

    held = []
    fake = _fake_hailo_platform(
        {"out": np.zeros((1, 3), dtype=np.float32)}, lambda: held.append(lock.locked())
    )
    with patch.dict(sys.modules, {"hailo_platform": fake}):
        classifier._run_inference([np.zeros((10, 10, 3), dtype=np.uint8)])

    assert held == [True]
    assert not lock.locked()


def test_shared_lock_serializes_detector_and_classifier():
    """One shared lock makes detector and classifier mutually exclusive."""
    lock = threading.Lock()
    detector = StubHailoDetector("dummy.hef", inference_lock=lock)
    classifier = StubHailoClassifier("dummy.hef", "labels.json", inference_lock=lock)
    assert detector._inference_lock is classifier._inference_lock

    # While the detector holds the lock, the classifier cannot acquire it.
    with detector._inference_lock:
        assert classifier._inference_lock.acquire(blocking=False) is False
