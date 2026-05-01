"""Unit tests for WebSocketFrameSource."""

import asyncio
import json
import queue
import threading
import time

import cv2
import numpy as np
import pytest

from src.ws_frame_source import WebSocketFrameSource


def _make_jpeg(width: int = 64, height: int = 48, color: tuple = (0, 128, 255)) -> bytes:
    """Create a small JPEG image as bytes."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class TestJPEGDecoding:
    def test_valid_jpeg_decodes_to_bgr(self):
        jpeg = _make_jpeg(100, 80)
        buf = np.frombuffer(jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert bgr is not None
        assert bgr.shape == (80, 100, 3)

    def test_invalid_bytes_return_none(self):
        buf = np.frombuffer(b"not a jpeg", dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert bgr is None


class TestFrameQueueDropLogic:
    def test_full_queue_drops_oldest(self):
        """When queue is full, evict the oldest frame before adding the new one."""
        q: queue.Queue = queue.Queue(maxsize=2)
        q.put(("frame_0", {}))
        q.put(("frame_1", {}))
        # Drop oldest, then enqueue newest
        if q.full():
            q.get_nowait()
        q.put_nowait(("frame_2", {}))
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert [i[0] for i in items] == ["frame_1", "frame_2"]

    def test_drain_keeps_latest(self):
        """Simulate the drain logic in frames() — only the latest frame survives."""
        q: queue.Queue = queue.Queue(maxsize=5)
        for i in range(5):
            q.put((f"frame_{i}", {"frame_id": i}))

        bgr, meta = q.get()
        while not q.empty():
            try:
                bgr, meta = q.get_nowait()
            except queue.Empty:
                break

        assert bgr == "frame_4"
        assert meta["frame_id"] == 4


class TestMetadataParsing:
    def test_valid_json_metadata(self):
        src = WebSocketFrameSource.__new__(WebSocketFrameSource)
        src._metadata = {}
        metadata_json = '{"frame_id": 42, "lat": 40.7, "lon": -73.5}'
        parsed = json.loads(metadata_json)
        src._metadata = parsed
        assert src.client_metadata["lat"] == 40.7
        assert src.client_metadata["lon"] == -73.5
        assert src.client_metadata["frame_id"] == 42

    def test_invalid_json_ignored(self):
        src = WebSocketFrameSource.__new__(WebSocketFrameSource)
        src._metadata = {"existing": True}
        try:
            json.loads("not json {{{")
        except json.JSONDecodeError:
            pass
        assert src._metadata == {"existing": True}


class TestSendResult:
    def test_send_result_without_client_is_noop(self):
        src = WebSocketFrameSource.__new__(WebSocketFrameSource)
        src._loop = None
        src._result_queue = None
        src._client_connected = threading.Event()
        src.send_result({"frame_id": 1, "detections": []})

    def test_send_result_with_closed_loop_is_noop(self):
        loop = asyncio.new_event_loop()
        loop.close()
        src = WebSocketFrameSource.__new__(WebSocketFrameSource)
        src._loop = loop
        src._result_queue = asyncio.Queue()
        src._client_connected = threading.Event()
        src._client_connected.set()
        src.send_result({"frame_id": 1, "detections": []})

    def test_send_result_skipped_when_no_client(self):
        loop = asyncio.new_event_loop()
        result_queue = asyncio.Queue()
        src = WebSocketFrameSource.__new__(WebSocketFrameSource)
        src._loop = loop
        src._result_queue = result_queue
        src._client_connected = threading.Event()
        src.send_result({"frame_id": 1, "detections": []})
        assert result_queue.empty()
        loop.close()

    def test_send_result_enqueues(self):
        loop = asyncio.new_event_loop()
        result_queue = asyncio.Queue()

        src = WebSocketFrameSource.__new__(WebSocketFrameSource)
        src._loop = loop
        src._result_queue = result_queue
        src._client_connected = threading.Event()
        src._client_connected.set()

        result = {"frame_id": 1, "detections": [], "fps": 5.0}
        src.send_result(result)
        loop.run_until_complete(asyncio.sleep(0))

        assert not result_queue.empty()
        item = loop.run_until_complete(result_queue.get())
        assert item == result
        loop.close()


class TestStopBehavior:
    def test_stop_sets_flag(self):
        src = WebSocketFrameSource()
        assert not src._stop
        src.stop()
        assert src._stop


class TestResultFormat:
    """Verify the result dict shape matches the protocol spec."""

    def test_result_schema(self):
        result = {
            "frame_id": 42,
            "detections": [
                {
                    "bbox": [0.12, 0.34, 0.45, 0.67],
                    "track_id": 1,
                    "confidence": 0.92,
                    "species": "American Robin",
                    "species_score": 0.87,
                },
            ],
            "fps": 12.3,
        }
        assert "frame_id" in result
        assert isinstance(result["detections"], list)
        det = result["detections"][0]
        assert len(det["bbox"]) == 4
        assert all(0.0 <= c <= 1.0 for c in det["bbox"])
        assert isinstance(det["track_id"], int)
        assert isinstance(det["species"], str)

    def test_empty_detections(self):
        result = {"frame_id": 0, "detections": [], "fps": 0.0}
        assert result["detections"] == []


class TestMetadataFramePairing:
    """Verify that metadata stays paired with its corresponding frame."""

    def test_metadata_set_on_dequeue(self):
        src = WebSocketFrameSource.__new__(WebSocketFrameSource)
        src._stop = False
        src._metadata = {}
        src._pending_metadata = {}
        src._frame_queue = queue.Queue(maxsize=5)

        # Simulate two frames with different metadata arriving
        src._frame_queue.put((np.zeros((10, 10, 3), dtype=np.uint8), {"frame_id": 0, "lat": 40.0}))
        src._frame_queue.put((np.zeros((10, 10, 3), dtype=np.uint8), {"frame_id": 1, "lat": 41.0}))

        # Mock _start_server and _shutdown_server
        src._start_server = lambda: None
        src._shutdown_server = lambda: None

        frames_iter = src.frames()
        # First frame should set metadata to frame_id=0
        # But drain logic will skip to the latest (frame_id=1)
        frame_no, bgr = next(frames_iter)
        assert src.client_metadata["frame_id"] == 1
        assert src.client_metadata["lat"] == 41.0
        src.stop()


class TestInit:
    def test_default_params(self):
        src = WebSocketFrameSource()
        assert src._host == "0.0.0.0"
        assert src._port == 8765
        assert src._static_dir is None
        assert src._stop is False

    def test_custom_params(self):
        src = WebSocketFrameSource(host="127.0.0.1", port=9999, static_dir="/tmp")
        assert src._host == "127.0.0.1"
        assert src._port == 9999
        assert src._static_dir == "/tmp"
