"""Regression test: the sidecar video-upload path must release its
cv2.VideoCapture even when frame processing raises.

RealtimePipeline is Pi-only and excluded from coverage, but it imports cleanly
without HailoRT (hardware bindings are imported lazily inside methods), so we
can construct a bare instance and drive _process_video_upload directly.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.realtime_pipeline as rp_module
from src.realtime_pipeline import RealtimePipeline


def _bare_pipeline(tmp_path: Path) -> RealtimePipeline:
    pipe = object.__new__(RealtimePipeline)
    pipe._config = {"tracker": {}}
    pipe._classify_every_n = 10
    pipe._crop_padding = 0.18
    pipe._crop_padding_min = 0.04
    pipe._crop_closeup_ratio = 0.06
    pipe._min_crop_area = 2500
    pipe._min_event_conf = 0.35
    pipe._metadata = None
    pipe._results_dir = tmp_path
    pipe._detector = MagicMock()
    pipe._classifier = MagicMock()
    pipe._enable_small_bird_zoom_fallback = False
    pipe._small_bird_fallback_every_n = 5
    return pipe


class _OneFrameSource:
    client_metadata = {"lat": 40.7, "lon": -73.5, "frame_id": "frame-1"}

    def __init__(self):
        self.stopped = False

    def frames(self):
        yield 10, np.zeros((100, 100, 3), dtype=np.uint8)

    def stop(self):
        self.stopped = True


def test_live_stream_metadata_prior_receives_date(tmp_path):
    pipe = _bare_pipeline(tmp_path)
    pipe._stop = False
    pipe._source = _OneFrameSource()
    pipe._display = None
    pipe._caption_ttl = 3.0
    pipe._log_interval = 999.0
    pipe._stats_interval = 999.0

    detection = SimpleNamespace(
        bbox=np.array([10, 10, 90, 90]),
        confidence=0.95,
    )
    track = SimpleNamespace(
        matched_detection_idx=0,
        last_classified_frame=-100,
        prediction_history=[],
        disappeared=0,
    )
    pipe._detector.detect.return_value = [detection]
    pipe._tracker = MagicMock()
    pipe._tracker.update.return_value = {1: track}
    pipe._classifier.classify_batch.return_value = [[("Mourning Dove", 0.9), ("Blue Jay", 0.1)]]
    pipe._metadata = MagicMock()
    pipe._metadata.apply.side_effect = lambda preds, **_: preds

    pipe.run()

    assert pipe._source.stopped is True
    _, kwargs = pipe._metadata.apply.call_args
    assert kwargs["latitude"] == 40.7
    assert kwargs["longitude"] == -73.5
    assert kwargs["dt"] is not None


def test_video_capture_released_when_processing_raises(tmp_path):
    pipe = _bare_pipeline(tmp_path)

    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.return_value = 10  # frame count / fps / width / height
    cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))

    # Force an exception partway through the frame loop.
    pipe._detector.detect.side_effect = RuntimeError("boom")

    with (
        patch.object(rp_module.cv2, "VideoCapture", return_value=cap),
        pytest.raises(RuntimeError),
    ):
        pipe._process_video_upload(b"video-bytes", "clip.mp4")

    # The capture handle must be released even though processing raised.
    cap.release.assert_called_once()
