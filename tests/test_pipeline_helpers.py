"""Unit tests for pure helper functions in src/pipeline.py.

Heavy dependencies (YOLO, BioCLIP) are monkeypatched so tests stay CPU-only
and do not download models.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pipeline import (
    BirdIdentificationPipeline,
    compact_path,
    resolution_warning_text,
)

# ---------------------------------------------------------------------------
# compact_path
# ---------------------------------------------------------------------------


class TestCompactPath:
    def test_short_path_unchanged(self):
        assert compact_path("a/b") == "a/b"

    def test_absolute_long_path(self):
        result = compact_path("/home/user/videos/bird.mp4")
        assert result == "/{...}/videos/bird.mp4"

    def test_relative_long_path(self):
        result = compact_path("home/user/videos/bird.mp4")
        assert result == "{...}/videos/bird.mp4"

    def test_custom_keep_parts(self):
        result = compact_path("/a/b/c/d/e", keep_parts=3)
        assert result == "/{...}/c/d/e"

    def test_exact_keep_parts(self):
        # Path with exactly keep_parts parts should not be abbreviated
        assert compact_path("a/b", keep_parts=2) == "a/b"

    def test_single_component(self):
        assert compact_path("file.txt") == "file.txt"


# ---------------------------------------------------------------------------
# resolution_warning_text
# ---------------------------------------------------------------------------


class TestResolutionWarningText:
    def test_none_dimensions(self):
        assert resolution_warning_text(media_type="video", width=None, height=None) is None

    def test_video_high_res_no_warning(self):
        assert resolution_warning_text(media_type="video", width=1920, height=1080) is None

    def test_video_low_res_warning(self):
        result = resolution_warning_text(media_type="video", width=640, height=480)
        assert result is not None
        assert "video" in result.lower()

    def test_image_high_res_no_warning(self):
        assert resolution_warning_text(media_type="image", width=4000, height=3000) is None

    def test_image_low_res_warning(self):
        result = resolution_warning_text(media_type="image", width=800, height=600)
        assert result is not None
        assert "photo" in result.lower()

    def test_video_borderline_passes(self):
        # Exactly at thresholds (1280x720)
        assert resolution_warning_text(media_type="video", width=1280, height=720) is None

    def test_video_borderline_fails(self):
        assert resolution_warning_text(media_type="video", width=1279, height=720) is not None

    def test_unknown_media_type_no_warning(self):
        assert resolution_warning_text(media_type="other", width=100, height=100) is None


# ---------------------------------------------------------------------------
# Pipeline helpers — requires monkeypatching heavy constructors
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipeline():
    """Create a pipeline with mocked detector/classifier/prior."""
    with (
        patch("src.pipeline.BirdDetector") as mock_det,
        patch("src.pipeline.BirdClassifier") as mock_cls,
        patch("src.pipeline.MetadataPrior") as mock_prior,
    ):
        mock_det.return_value = MagicMock(confidence=0.3)
        mock_cls.return_value = MagicMock(top_k=5)
        mock_prior.return_value = MagicMock(latitude=None, longitude=None)
        pipe = BirdIdentificationPipeline({})
    return pipe


class TestExpandedCrop:
    def test_returns_none_for_tiny_box(self, pipeline):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        bbox = np.array([0, 0, 5, 5])  # area=25, below min_crop_area=2500
        assert pipeline._expanded_crop(frame, bbox) is None

    def test_returns_crop_for_valid_box(self, pipeline):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 200, 200])  # area=10000
        crop = pipeline._expanded_crop(frame, bbox)
        assert crop is not None
        assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_crop_has_padding(self, pipeline):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 200, 200])
        crop = pipeline._expanded_crop(frame, bbox)
        # Crop should be larger than the raw bbox
        assert crop.shape[0] > 100
        assert crop.shape[1] > 100

    def test_crop_clamped_to_frame(self, pipeline):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = np.array([0, 0, 90, 90])  # Near frame edges
        crop = pipeline._expanded_crop(frame, bbox)
        assert crop is not None
        assert crop.shape[0] <= 100
        assert crop.shape[1] <= 100

    def test_closeup_gets_less_padding(self, pipeline):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Small bird: small fraction of frame
        small_bbox = np.array([500, 500, 560, 560])
        small_crop = pipeline._expanded_crop(frame, small_bbox)
        # Large bird: big fraction of frame
        big_bbox = np.array([100, 100, 900, 900])
        big_crop = pipeline._expanded_crop(frame, big_bbox)
        if small_crop is not None and big_crop is not None:
            small_padding_ratio = (small_crop.shape[0] - 60) / 60.0
            big_padding_ratio = (big_crop.shape[0] - 800) / 800.0
            assert small_padding_ratio > big_padding_ratio


class TestBuildVideoPredictions:
    def test_single_track(self, pipeline):
        track_preds = [
            [("Robin", 0.8), ("Sparrow", 0.2)],
        ]
        result = pipeline._build_video_predictions(track_preds)
        assert len(result) >= 1
        assert result[0]["species"] == "Robin"
        assert result[0]["presence_probability"] == 0.8
        assert result[0]["supporting_tracks"] == 1

    def test_multiple_tracks_max_probability(self, pipeline):
        track_preds = [
            [("Robin", 0.6)],
            [("Robin", 0.9)],
        ]
        result = pipeline._build_video_predictions(track_preds)
        robin = next(r for r in result if r["species"] == "Robin")
        assert robin["presence_probability"] == 0.9
        assert robin["supporting_tracks"] == 2

    def test_top_5_limit(self, pipeline):
        track_preds = [
            [(f"Species{i}", 1.0 / (i + 1)) for i in range(10)],
        ]
        result = pipeline._build_video_predictions(track_preds)
        assert len(result) <= 5

    def test_empty_input(self, pipeline):
        assert pipeline._build_video_predictions([]) == []


class TestBuildSpeciesSummary:
    def test_basic_summary(self, pipeline):
        weighted = {"Robin": 0.7, "Sparrow": 0.3}
        raw = {"Robin": 0.5, "Sparrow": 0.5}
        result = pipeline._build_species_summary(weighted, raw)
        assert result[0]["species"] == "Robin"
        assert result[0]["probability"] == 0.7
        assert result[0]["raw_probability"] == 0.5

    def test_top_5_limit(self, pipeline):
        weighted = {f"Species{i}": 1.0 / (i + 1) for i in range(10)}
        raw = weighted.copy()
        result = pipeline._build_species_summary(weighted, raw)
        assert len(result) <= 5

    def test_missing_raw_defaults_to_zero(self, pipeline):
        weighted = {"Robin": 0.9}
        raw = {}
        result = pipeline._build_species_summary(weighted, raw)
        assert result[0]["raw_probability"] == 0.0


class TestWaterbirdShapeAdjustment:
    def test_no_adjustment_without_both_swan_and_gull(self, pipeline):
        preds = [("Mute Swan", 0.8), ("Robin", 0.2)]
        result = pipeline._apply_waterbird_shape_adjustment(
            preds,
            bbox=np.array([0, 0, 200, 50]),
            frame_width=1920,
            frame_height=1080,
        )
        assert dict(result) == dict(preds)

    def test_wide_box_boosts_gull(self, pipeline):
        preds = [("Mute Swan", 0.5), ("Herring Gull", 0.5)]
        # Wide, low-profile box: aspect_ratio > 1.15, relative_height < 0.23
        result = pipeline._apply_waterbird_shape_adjustment(
            preds,
            bbox=np.array([100, 500, 400, 550]),  # 300w x 50h, AR=6.0
            frame_width=1920,
            frame_height=1080,
        )
        result_dict = dict(result)
        assert result_dict["Herring Gull"] > result_dict["Mute Swan"]

    def test_tall_box_no_adjustment(self, pipeline):
        preds = [("Mute Swan", 0.5), ("Herring Gull", 0.5)]
        # Tall box: aspect_ratio < 1.15 -> no adjustment
        result = pipeline._apply_waterbird_shape_adjustment(
            preds,
            bbox=np.array([100, 100, 200, 400]),  # 100w x 300h
            frame_width=1920,
            frame_height=1080,
        )
        result_dict = dict(result)
        assert result_dict["Mute Swan"] == pytest.approx(0.5)

    def test_empty_preds(self, pipeline):
        assert (
            pipeline._apply_waterbird_shape_adjustment(
                [],
                bbox=np.array([0, 0, 1, 1]),
                frame_width=100,
                frame_height=100,
            )
            == []
        )

    def test_adjustment_renormalizes(self, pipeline):
        preds = [("Tundra Swan", 0.4), ("Herring Gull", 0.4), ("Robin", 0.2)]
        result = pipeline._apply_waterbird_shape_adjustment(
            preds,
            bbox=np.array([100, 500, 400, 550]),
            frame_width=1920,
            frame_height=1080,
        )
        total = sum(p for _, p in result)
        assert total == pytest.approx(1.0)


class TestSelectVideoGalleryPlan:
    def test_empty_candidates_with_fallback(self, pipeline):
        result = pipeline._select_video_gallery_plan(
            [],
            total_frames=100,
            fps=30.0,
            min_frames=3,
        )
        # With only 100 frames and min_gap constraints, fewer than min_frames
        # may be returned; verify fallback frames were generated
        assert len(result) >= 1
        assert all(r["score"] == 0.0 for r in result)

    def test_zero_total_frames(self, pipeline):
        result = pipeline._select_video_gallery_plan(
            [],
            total_frames=0,
            fps=30.0,
        )
        assert result == []

    def test_candidates_selected_by_score(self, pipeline):
        candidates = [
            {"frame": 10, "score": 0.9, "timestamp_s": 0.33, "track_id": 0, "species": "Robin"},
            {"frame": 500, "score": 0.5, "timestamp_s": 16.67, "track_id": 1, "species": "Sparrow"},
            {"frame": 900, "score": 0.1, "timestamp_s": 30.0, "track_id": 2, "species": "Dove"},
        ]
        result = pipeline._select_video_gallery_plan(
            candidates,
            total_frames=1000,
            fps=30.0,
            max_frames=3,
        )
        assert len(result) <= 3
        # Should be sorted by frame number
        frames = [r["frame"] for r in result]
        assert frames == sorted(frames)

    def test_respects_max_frames(self, pipeline):
        candidates = [
            {"frame": i * 100, "score": 0.5, "timestamp_s": i, "track_id": i, "species": "Robin"}
            for i in range(20)
        ]
        result = pipeline._select_video_gallery_plan(
            candidates,
            total_frames=2000,
            fps=30.0,
            max_frames=6,
        )
        assert len(result) <= 6

    def test_fallback_frames_added(self, pipeline):
        # No good candidates -> fallback frames fill to min_frames
        result = pipeline._select_video_gallery_plan(
            [],
            total_frames=3000,
            fps=30.0,
            min_frames=4,
            max_frames=6,
        )
        assert len(result) >= 4
        assert all(r["score"] == 0.0 for r in result)
