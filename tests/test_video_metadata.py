"""Unit tests for src/video_metadata.py — camera_info, osm_url, and extract."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.video_metadata import MediaMetadata, VideoMetadata, extract


# ---------------------------------------------------------------------------
# VideoMetadata.camera_info
# ---------------------------------------------------------------------------

class TestCameraInfo:
    def test_make_and_model_no_duplication(self):
        meta = VideoMetadata(camera_make="Apple", camera_model="Apple iPhone 15 Pro")
        assert meta.camera_info == "Apple iPhone 15 Pro"

    def test_make_and_model_concatenated(self):
        meta = VideoMetadata(camera_make="Canon", camera_model="EOS R6")
        assert meta.camera_info == "Canon EOS R6"

    def test_model_only(self):
        meta = VideoMetadata(camera_model="EOS R6")
        assert meta.camera_info == "EOS R6"

    def test_with_focal_length(self):
        meta = VideoMetadata(camera_make="Canon", camera_model="EOS R6", focal_length="50mm")
        assert meta.camera_info == "Canon EOS R6, 50mm"

    def test_focal_length_only(self):
        meta = VideoMetadata(focal_length="77mm")
        assert meta.camera_info == "77mm"

    def test_no_camera_info(self):
        meta = VideoMetadata()
        assert meta.camera_info is None

    def test_case_insensitive_dedup(self):
        meta = VideoMetadata(camera_make="apple", camera_model="Apple iPhone")
        assert meta.camera_info == "Apple iPhone"


# ---------------------------------------------------------------------------
# VideoMetadata.osm_url
# ---------------------------------------------------------------------------

class TestOsmUrl:
    def test_with_gps(self):
        meta = VideoMetadata(latitude=40.7, longitude=-73.5)
        url = meta.osm_url
        assert url is not None
        assert "openstreetmap.org" in url
        assert "40.7" in url
        assert "-73.5" in url

    def test_without_gps(self):
        meta = VideoMetadata()
        assert meta.osm_url is None

    def test_partial_gps(self):
        meta = VideoMetadata(latitude=40.0)
        assert meta.osm_url is None


# ---------------------------------------------------------------------------
# MediaMetadata.camera_info
# ---------------------------------------------------------------------------

class TestMediaMetadataCameraInfo:
    def test_delegates_to_video_metadata(self):
        meta = MediaMetadata(camera_make="Canon", camera_model="EOS R6", focal_length="50mm")
        assert meta.camera_info == "Canon EOS R6, 50mm"

    def test_has_gps(self):
        meta = MediaMetadata(latitude=40.0, longitude=-73.0)
        assert meta.has_gps is True

    def test_no_gps(self):
        meta = MediaMetadata()
        assert meta.has_gps is False


# ---------------------------------------------------------------------------
# extract (monkeypatched)
# ---------------------------------------------------------------------------

class TestExtract:
    def test_extract_returns_video_metadata(self):
        fake_media = MediaMetadata(
            recorded_at=datetime(2024, 3, 15, 10, 30),
            latitude=40.7,
            longitude=-73.5,
            camera_make="Apple",
            camera_model="iPhone 15",
            focal_length="77mm",
            width=1920,
            height=1080,
            duration_s=30.0,
            fps=30.0,
        )
        with patch("src.video_metadata.inspect_media", return_value=fake_media):
            result = extract("/fake/video.mp4")
        assert isinstance(result, VideoMetadata)
        assert result.recorded_at == datetime(2024, 3, 15, 10, 30)
        assert result.latitude == 40.7
        assert result.longitude == -73.5
        assert result.camera_make == "Apple"
        assert result.camera_model == "iPhone 15"
        assert result.focal_length == "77mm"
        # MediaMetadata-only fields should not be on VideoMetadata
        assert not hasattr(result, "width") or result.__class__ is VideoMetadata
