"""Tests for moving blocking upload work off the event loop.

AssetStore.inspect_bytes / ingest_path do synchronous heavy work (sha256 over
the payload, a 50 MB disk write, an exiftool subprocess, OpenCV probes). Run on
the event loop they stall every other request. They must be dispatched to a
worker thread via _run_off_event_loop.
"""

import asyncio
import json
import os
import threading

import src.webapp as webapp_module
from src.video_metadata import MediaMetadata
from src.webapp import AssetStore, _run_off_event_loop


def _video_meta(_path):
    return MediaMetadata(width=10, height=10, duration_s=1.0, fps=30.0, video_codec="avc1")


def test_run_off_event_loop_executes_on_worker_thread():
    main_thread = threading.get_ident()
    seen: dict[str, int] = {}

    def blocking(value):
        seen["thread"] = threading.get_ident()
        return value * 2

    result = asyncio.run(_run_off_event_loop(blocking, 21))

    assert result == 42
    # The callable ran on a thread-pool thread, not the event loop thread.
    assert seen["thread"] != main_thread


def test_run_off_event_loop_forwards_kwargs():
    def add(a, *, b):
        return a + b

    assert asyncio.run(_run_off_event_loop(add, 2, b=3)) == 5


def test_asset_store_inspect_bytes_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        webapp_module,
        "inspect_media",
        lambda _p: MediaMetadata(width=10, height=10, duration_s=1.0, fps=30.0, video_codec="avc1"),
    )
    store = AssetStore(tmp_path / "videos")

    inspected = store.inspect_bytes(
        original_filename="clip.mp4",
        content_type="video/mp4",
        data=b"payload",
        client_metadata=None,
    )

    assert inspected["media_type"] == "video"
    assert store.get(inspected["sha256"]) is not None


def test_save_index_uses_atomic_replace(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp_module, "inspect_media", _video_meta)
    store = AssetStore(tmp_path / "videos")

    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src, dst):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(webapp_module.os, "replace", spy)
    store.inspect_bytes(
        original_filename="a.mp4", content_type="video/mp4", data=b"x", client_metadata=None
    )

    # The index was written to a temp file and atomically replaced into place.
    assert any(dst.endswith("asset_index.json") and src.endswith(".tmp") for src, dst in calls)
    # No temp file is left behind, and the index parses.
    index_dir = tmp_path / "videos"
    assert not list(index_dir.glob("*.tmp"))
    payload = json.loads((index_dir / "asset_index.json").read_text())
    assert len(payload["assets"]) == 1


def test_load_index_preserves_corrupt_file(tmp_path):
    upload = tmp_path / "videos"
    upload.mkdir()
    corrupt = upload / "asset_index.json"
    corrupt.write_text("{ this is not valid json")

    store = AssetStore(upload)

    # Starts empty rather than wiping the corrupt file silently…
    assert store._assets == {}
    # …and the corrupt content is preserved aside for recovery.
    backups = list(upload.glob("asset_index.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == "{ this is not valid json"
    # The live index path is now free for a fresh, valid save.
    assert not corrupt.exists()
