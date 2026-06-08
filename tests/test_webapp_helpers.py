"""Unit tests for pure helper / auth logic in src/webapp.py."""

import json
import os
import subprocess
from unittest.mock import MagicMock, patch

from src.webapp import (
    SAFARI_COMPATIBLE_CODECS,
    AuthSettings,
    Job,
    _load_api_tokens,
    _load_existing_jobs,
    _persist_source_event_id,
    _persist_submitted_by,
    build_auth_settings,
    build_job_display_name,
    can_upload_email,
    classify_media_type,
    current_user_email,
    debug_mode_enabled,
    normalize_email,
    normalize_secret,
    parse_bool,
    require_upload_access,
    resolution_warning_text,
    result_name_seed,
    slugify_result_name,
    transcode_to_h264,
    validate_asset_batch,
)

# ---------------------------------------------------------------------------
# parse_bool
# ---------------------------------------------------------------------------


class TestParseBool:
    def test_true_bool(self):
        assert parse_bool(True) is True

    def test_false_bool(self):
        assert parse_bool(False) is False

    def test_string_true(self):
        assert parse_bool("true") is True
        assert parse_bool("True") is True
        assert parse_bool("TRUE") is True

    def test_string_yes(self):
        assert parse_bool("yes") is True

    def test_string_one(self):
        assert parse_bool("1") is True

    def test_string_on(self):
        assert parse_bool("on") is True

    def test_string_debug(self):
        assert parse_bool("debug") is True

    def test_string_false(self):
        assert parse_bool("false") is False

    def test_string_no(self):
        assert parse_bool("no") is False

    def test_integer_truthy(self):
        assert parse_bool(1) is True

    def test_integer_falsy(self):
        assert parse_bool(0) is False

    def test_whitespace_stripped(self):
        assert parse_bool("  true  ") is True


# ---------------------------------------------------------------------------
# normalize_secret
# ---------------------------------------------------------------------------


class TestNormalizeSecret:
    def test_valid_string(self):
        assert normalize_secret("my-secret") == "my-secret"

    def test_strips_whitespace(self):
        assert normalize_secret("  secret  ") == "secret"

    def test_empty_string(self):
        assert normalize_secret("") is None

    def test_whitespace_only(self):
        assert normalize_secret("   ") is None

    def test_none(self):
        assert normalize_secret(None) is None

    def test_non_string(self):
        assert normalize_secret(123) is None


# ---------------------------------------------------------------------------
# normalize_email
# ---------------------------------------------------------------------------


class TestNormalizeEmail:
    def test_lowercases(self):
        assert normalize_email("User@Example.COM") == "user@example.com"

    def test_strips_whitespace(self):
        assert normalize_email("  user@test.com  ") == "user@test.com"

    def test_empty_string(self):
        assert normalize_email("") is None

    def test_none(self):
        assert normalize_email(None) is None

    def test_non_string(self):
        assert normalize_email(42) is None


# ---------------------------------------------------------------------------
# debug_mode_enabled
# ---------------------------------------------------------------------------


class TestDebugModeEnabled:
    def test_env_var_override(self):
        with patch.dict(os.environ, {"BIRDVISION_DEBUG": "true"}):
            assert debug_mode_enabled({}) is True

    def test_env_var_false(self):
        with patch.dict(os.environ, {"BIRDVISION_DEBUG": "false"}):
            assert debug_mode_enabled({}) is False

    def test_config_webapp_debug(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BIRDVISION_DEBUG", None)
            assert debug_mode_enabled({"webapp": {"debug": True}}) is True

    def test_config_no_debug(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BIRDVISION_DEBUG", None)
            assert debug_mode_enabled({}) is False

    def test_invalid_webapp_config(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BIRDVISION_DEBUG", None)
            assert debug_mode_enabled({"webapp": "invalid"}) is False


# ---------------------------------------------------------------------------
# build_auth_settings
# ---------------------------------------------------------------------------


class TestBuildAuthSettings:
    def test_enabled_with_all_fields(self):
        with patch.dict(os.environ, {}, clear=True):
            for key in (
                "BIRDVISION_DEBUG",
                "GOOGLE_CLIENT_ID",
                "GOOGLE_CLIENT_SECRET",
                "SESSION_SECRET",
                "GOOGLE_REDIRECT_URI",
            ):
                os.environ.pop(key, None)
            config = {
                "auth": {
                    "google_client_id": "id123",
                    "google_client_secret": "secret456",
                    "session_secret": "sess789",
                    "allowed_emails": ["user@test.com"],
                }
            }
            settings = build_auth_settings(config)
        assert settings.enabled is True
        assert settings.google_client_id == "id123"
        assert "user@test.com" in settings.allowed_emails

    def test_disabled_without_required_fields(self):
        with patch.dict(os.environ, {}, clear=True):
            for key in (
                "BIRDVISION_DEBUG",
                "GOOGLE_CLIENT_ID",
                "GOOGLE_CLIENT_SECRET",
                "SESSION_SECRET",
            ):
                os.environ.pop(key, None)
            settings = build_auth_settings({})
        assert settings.enabled is False

    def test_disabled_in_debug_mode(self):
        with patch.dict(os.environ, {"BIRDVISION_DEBUG": "true"}, clear=True):
            for key in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "SESSION_SECRET"):
                os.environ.pop(key, None)
            config = {
                "auth": {
                    "google_client_id": "id123",
                    "google_client_secret": "secret456",
                    "session_secret": "sess789",
                }
            }
            settings = build_auth_settings(config)
        assert settings.enabled is False
        assert settings.debug_mode is True

    def test_env_vars_override_config(self):
        with patch.dict(
            os.environ,
            {
                "GOOGLE_CLIENT_ID": "env-id",
                "GOOGLE_CLIENT_SECRET": "env-secret",
                "SESSION_SECRET": "env-session",
            },
            clear=True,
        ):
            os.environ.pop("BIRDVISION_DEBUG", None)
            config = {
                "auth": {
                    "google_client_id": "config-id",
                    "google_client_secret": "config-secret",
                    "session_secret": "config-session",
                }
            }
            settings = build_auth_settings(config)
        assert settings.google_client_id == "env-id"
        assert settings.google_client_secret == "env-secret"
        assert settings.session_secret == "env-session"

    def test_normalizes_emails(self):
        with patch.dict(os.environ, {}, clear=True):
            for key in (
                "BIRDVISION_DEBUG",
                "GOOGLE_CLIENT_ID",
                "GOOGLE_CLIENT_SECRET",
                "SESSION_SECRET",
            ):
                os.environ.pop(key, None)
            config = {
                "auth": {
                    "google_client_id": "id",
                    "google_client_secret": "secret",
                    "session_secret": "sess",
                    "allowed_emails": ["  User@Test.COM  ", 42, ""],
                }
            }
            settings = build_auth_settings(config)
        assert "user@test.com" in settings.allowed_emails
        assert len(settings.allowed_emails) == 1


# ---------------------------------------------------------------------------
# can_upload_email
# ---------------------------------------------------------------------------


class TestCanUploadEmail:
    def test_auth_disabled_always_allowed(self):
        settings = AuthSettings(
            enabled=False,
            debug_mode=True,
            google_client_id=None,
            google_client_secret=None,
            redirect_uri=None,
            session_secret=None,
            allowed_emails=set(),
        )
        assert can_upload_email(None, settings) is True

    def test_allowed_email(self):
        settings = AuthSettings(
            enabled=True,
            debug_mode=False,
            google_client_id="x",
            google_client_secret="x",
            redirect_uri=None,
            session_secret="x",
            allowed_emails={"user@test.com"},
        )
        assert can_upload_email("user@test.com", settings) is True

    def test_disallowed_email(self):
        settings = AuthSettings(
            enabled=True,
            debug_mode=False,
            google_client_id="x",
            google_client_secret="x",
            redirect_uri=None,
            session_secret="x",
            allowed_emails={"user@test.com"},
        )
        assert can_upload_email("other@test.com", settings) is False

    def test_none_email_when_auth_enabled(self):
        settings = AuthSettings(
            enabled=True,
            debug_mode=False,
            google_client_id="x",
            google_client_secret="x",
            redirect_uri=None,
            session_secret="x",
            allowed_emails={"user@test.com"},
        )
        assert can_upload_email(None, settings) is False


# ---------------------------------------------------------------------------
# require_upload_access
# ---------------------------------------------------------------------------


def _make_request(path="/upload", session=None):
    request = MagicMock()
    request.url.path = path
    request.session = session or {}
    return request


class TestRequireUploadAccess:
    def test_auth_disabled_returns_none(self):
        settings = AuthSettings(
            enabled=False,
            debug_mode=True,
            google_client_id=None,
            google_client_secret=None,
            redirect_uri=None,
            session_secret=None,
            allowed_emails=set(),
        )
        assert require_upload_access(_make_request(), settings) is None

    def test_no_email_page_redirects(self):
        settings = AuthSettings(
            enabled=True,
            debug_mode=False,
            google_client_id="x",
            google_client_secret="x",
            redirect_uri=None,
            session_secret="x",
            allowed_emails=set(),
        )
        result = require_upload_access(_make_request("/upload"), settings)
        assert result is not None
        assert result.status_code == 303

    def test_no_email_api_returns_401(self):
        settings = AuthSettings(
            enabled=True,
            debug_mode=False,
            google_client_id="x",
            google_client_secret="x",
            redirect_uri=None,
            session_secret="x",
            allowed_emails=set(),
        )
        result = require_upload_access(_make_request("/api/upload"), settings)
        assert result is not None
        assert result.status_code == 401

    def test_non_whitelisted_api_returns_403(self):
        settings = AuthSettings(
            enabled=True,
            debug_mode=False,
            google_client_id="x",
            google_client_secret="x",
            redirect_uri=None,
            session_secret="x",
            allowed_emails={"allowed@test.com"},
        )
        request = _make_request("/api/upload", session={"email": "other@test.com"})
        result = require_upload_access(request, settings)
        assert result is not None
        assert result.status_code == 403

    def test_whitelisted_returns_none(self):
        settings = AuthSettings(
            enabled=True,
            debug_mode=False,
            google_client_id="x",
            google_client_secret="x",
            redirect_uri=None,
            session_secret="x",
            allowed_emails={"user@test.com"},
        )
        request = _make_request("/upload", session={"email": "user@test.com"})
        assert require_upload_access(request, settings) is None


# ---------------------------------------------------------------------------
# validate_asset_batch
# ---------------------------------------------------------------------------


class TestValidateAssetBatch:
    def test_empty_list(self):
        result = validate_asset_batch([])
        assert result["valid"] is False

    def test_all_images(self):
        assets = [{"media_type": "image"}, {"media_type": "image"}]
        result = validate_asset_batch(assets)
        assert result["valid"] is True
        assert result["media_type"] == "image"

    def test_all_videos(self):
        assets = [{"media_type": "video"}]
        result = validate_asset_batch(assets)
        assert result["valid"] is True
        assert result["media_type"] == "video"

    def test_mixed_returns_mixed(self):
        assets = [{"media_type": "image"}, {"media_type": "video"}]
        result = validate_asset_batch(assets)
        assert result["valid"] is True
        assert result["media_type"] == "mixed"

    def test_unknown_types_only(self):
        assets = [{"media_type": "unknown"}]
        result = validate_asset_batch(assets)
        assert result["valid"] is False

    def test_unprocessable_video_only_returns_info(self):
        assets = [{"media_type": "video", "processable": False}]
        result = validate_asset_batch(assets)
        assert result["valid"] is False
        assert result["message_level"] == "info"
        assert "could not be opened for processing" in result["error"]

    def test_processable_assets_ignore_unprocessable_ones(self):
        assets = [
            {"media_type": "image", "processable": True},
            {"media_type": "video", "processable": False},
        ]
        result = validate_asset_batch(assets)
        assert result["valid"] is True
        assert result["media_type"] == "image"
        assert "could not be opened for processing" in result["info"]


# ---------------------------------------------------------------------------
# classify_media_type
# ---------------------------------------------------------------------------


class TestClassifyMediaType:
    def test_video_extension(self):
        assert classify_media_type("bird.mp4", None) == "video"
        assert classify_media_type("bird.MOV", None) == "video"

    def test_image_extension(self):
        assert classify_media_type("bird.jpg", None) == "image"
        assert classify_media_type("bird.PNG", None) == "image"

    def test_unknown_extension_with_content_type(self):
        assert classify_media_type("bird.dat", "video/mp4") == "video"
        assert classify_media_type("bird.dat", "image/jpeg") == "image"

    def test_unknown(self):
        assert classify_media_type("bird.dat", None) == "unknown"
        assert classify_media_type("bird.dat", "application/octet-stream") == "unknown"

    def test_heic(self):
        assert classify_media_type("photo.heic", None) == "image"

    def test_webp(self):
        assert classify_media_type("photo.webp", None) == "image"


# ---------------------------------------------------------------------------
# build_job_display_name
# ---------------------------------------------------------------------------


class TestBuildJobDisplayName:
    def test_single_image(self):
        assets = [{"original_filename": "robin.jpg"}]
        assert build_job_display_name(assets, "images") == "robin.jpg"

    def test_multiple_images(self):
        assets = [{"original_filename": "a.jpg"}, {"original_filename": "b.jpg"}]
        assert build_job_display_name(assets, "images") == "2 photos"

    def test_video(self):
        assets = [{"original_filename": "video.mp4"}]
        assert build_job_display_name(assets, "video") == "video.mp4"


# ---------------------------------------------------------------------------
# slugify_result_name
# ---------------------------------------------------------------------------


class TestSlugifyResultName:
    def test_simple_filename(self):
        assert slugify_result_name("bird_video.mp4") == "bird_video"

    def test_special_characters(self):
        result = slugify_result_name("bird (1) video!.mp4")
        assert "(" not in result
        assert "!" not in result
        assert result == "bird_1_video"

    def test_empty_result_defaults_to_upload(self):
        assert slugify_result_name("...") == "upload"

    def test_truncated_at_80(self):
        long_name = "a" * 100 + ".mp4"
        result = slugify_result_name(long_name)
        assert len(result) <= 80

    def test_path_uses_stem(self):
        assert slugify_result_name("/path/to/video.mp4") == "video"


# ---------------------------------------------------------------------------
# result_name_seed
# ---------------------------------------------------------------------------


class TestResultNameSeed:
    def test_single_image(self):
        assets = [{"original_filename": "robin.jpg"}]
        assert result_name_seed(assets, "images") == "robin.jpg"

    def test_multiple_images(self):
        assets = [{"original_filename": "a.jpg"}, {"original_filename": "b.jpg"}]
        assert result_name_seed(assets, "images") == "photos"

    def test_video(self):
        assets = [{"original_filename": "vid.mp4"}]
        assert result_name_seed(assets, "video") == "vid.mp4"


# ---------------------------------------------------------------------------
# resolution_warning_text (webapp version)
# ---------------------------------------------------------------------------


class TestWebappResolutionWarningText:
    def test_none_dimensions(self):
        assert resolution_warning_text(media_type="video", width=None, height=None) is None

    def test_video_low_res(self):
        result = resolution_warning_text(media_type="video", width=640, height=480)
        assert result is not None

    def test_image_low_res(self):
        result = resolution_warning_text(media_type="image", width=800, height=600)
        assert result is not None

    def test_adequate_video(self):
        assert resolution_warning_text(media_type="video", width=1920, height=1080) is None

    def test_adequate_image(self):
        assert resolution_warning_text(media_type="image", width=4000, height=3000) is None


# ---------------------------------------------------------------------------
# Job.slug
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Job.has_detections
# ---------------------------------------------------------------------------


class TestJobHasDetections:
    def _video_job(self, status="done", tracks=None):
        job = Job(id="abc", filename="clip.mp4", media_type="video")
        job.status = status
        if status == "done":
            job.result = {"type": "video", "tracks": tracks if tracks is not None else []}
        return job

    def _image_job(self, status="done", images=None):
        job = Job(id="abc", filename="photo.jpg", media_type="images")
        job.status = status
        if status == "done":
            job.result = {"type": "images", "images": images if images is not None else []}
        return job

    def test_pending_job_always_visible(self):
        job = self._video_job(status="pending")
        job.result = None
        assert job.has_detections is True

    def test_running_job_always_visible(self):
        job = self._video_job(status="running")
        job.result = None
        assert job.has_detections is True

    def test_error_job_always_visible(self):
        job = self._video_job(status="error")
        job.result = None
        assert job.has_detections is True

    def test_video_with_tracks_is_visible(self):
        job = self._video_job(tracks=[{"track_id": 1}])
        assert job.has_detections is True

    def test_video_with_no_tracks_is_hidden(self):
        job = self._video_job(tracks=[])
        assert job.has_detections is False

    def test_video_with_tracks_but_no_species_is_visible(self):
        # Detected but unidentified - still show it
        job = self._video_job(tracks=[{"track_id": 1}])
        job.result["video_predictions"] = []
        assert job.has_detections is True

    def test_image_with_detections_is_visible(self):
        job = self._image_job(images=[{"detections": [{"bbox": [0, 0, 10, 10]}]}])
        assert job.has_detections is True

    def test_image_with_no_detections_is_hidden(self):
        job = self._image_job(images=[{"detections": []}])
        assert job.has_detections is False

    def test_image_with_mixed_photos_is_visible(self):
        # At least one photo has a detection
        job = self._image_job(
            images=[
                {"detections": []},
                {"detections": [{"bbox": [0, 0, 10, 10]}]},
            ]
        )
        assert job.has_detections is True

    def test_image_with_no_images_is_hidden(self):
        job = self._image_job(images=[])
        assert job.has_detections is False

    def test_done_job_with_no_result_is_visible(self):
        job = self._video_job(status="done")
        job.result = None
        assert job.has_detections is True


class TestJobSlug:
    def _make_job(self, filename: str) -> Job:
        return Job(id="abc123", filename=filename)

    def _make_done_video_job(self, filename: str, species: str, probability: float) -> Job:
        job = self._make_job(filename)
        job.status = "done"
        job.result = {
            "type": "video",
            "tracks": [{"track_id": 1}],
            "video_predictions": [
                {
                    "species": species,
                    "presence_probability": probability,
                    "supporting_tracks": 1,
                }
            ],
        }
        return job

    def _make_done_image_job(self, filename: str, species: str, probability: float) -> Job:
        job = Job(id="abc123", filename=filename, media_type="images")
        job.status = "done"
        job.result = {
            "type": "images",
            "images": [
                {
                    "species_summary": [
                        {
                            "species": species,
                            "probability": probability,
                        }
                    ],
                    "detections": [{"bbox": [0, 0, 10, 10]}],
                }
            ],
        }
        return job

    def test_simple_filename(self):
        assert self._make_job("blue-jay.mp4").slug == "blue-jay"

    def test_uppercase_lowercased(self):
        assert self._make_job("Blue_Jay.MOV").slug == "blue-jay"

    def test_spaces_become_hyphens(self):
        assert self._make_job("morning dove 2024.mp4").slug == "morning-dove-2024"

    def test_special_chars_stripped(self):
        slug = self._make_job("bird (1) video!.mp4").slug
        assert "(" not in slug
        assert "!" not in slug
        assert slug == "bird-1-video"

    def test_leading_trailing_hyphens_stripped(self):
        slug = self._make_job("--bird--.mp4").slug
        assert not slug.startswith("-")
        assert not slug.endswith("-")

    def test_empty_stem_defaults_to_job(self):
        assert self._make_job("...mp4").slug == "job"

    def test_path_uses_stem_only(self):
        assert self._make_job("/path/to/cardinal.mp4").slug == "cardinal"

    def test_only_alphanumeric_and_hyphens(self):
        import re as _re

        slug = self._make_job("bird@photo#2024.jpg").slug
        assert _re.match(r"^[a-z0-9-]+$", slug)

    def test_done_video_uses_top_species_when_above_90_percent(self):
        slug = self._make_done_video_job("8a7c4f19.mp4", "Blue Jay", 0.9342).slug
        assert slug == "blue-jay"

    def test_done_video_with_species_id_style_still_slugifies_cleanly(self):
        slug = self._make_done_video_job("8a7c4f19.mp4", "Blue_Jay", 0.9342).slug
        assert slug == "blue-jay"

    def test_done_video_falls_back_at_90_percent(self):
        slug = self._make_done_video_job("8a7c4f19.mp4", "Blue Jay", 0.9).slug
        assert slug == "8a7c4f19"

    def test_done_image_uses_top_species_when_above_90_percent(self):
        slug = self._make_done_image_job("d2f01a7b.jpg", "Northern Cardinal", 0.951).slug
        assert slug == "northern-cardinal"


# ---------------------------------------------------------------------------
# Job.thumbnail_url
# ---------------------------------------------------------------------------


class TestJobThumbnailUrl:
    def test_pending_job_has_no_thumbnail(self):
        job = Job(id="abc123", filename="clip.mp4", media_type="video")
        assert job.thumbnail_url is None

    def test_image_job_uses_highest_confidence_annotated_photo(self):
        job = Job(id="abc123", filename="batch.jpg", media_type="images")
        job.status = "done"
        job.result = {
            "type": "images",
            "images": [
                {
                    "annotated_file": "img0_annotated.jpg",
                    "species_summary": [{"species": "Blue Jay", "probability": 0.91}],
                    "detections": [{"bbox": [0, 0, 10, 10]}],
                },
                {
                    "annotated_file": "img1_annotated.jpg",
                    "species_summary": [{"species": "Northern Cardinal", "probability": 0.96}],
                    "detections": [{"bbox": [0, 0, 10, 10]}],
                },
            ],
        }

        assert job.thumbnail_url == "/jobs/abc123/crops/img1_annotated.jpg"

    def test_video_job_uses_highest_confidence_annotated_still(self):
        job = Job(id="abc123", filename="clip.mp4", media_type="video")
        job.status = "done"
        job.result = {
            "type": "video",
            "tracks": [{"track_id": 1}],
            "video_stills": [
                {
                    "annotated_file": "still_00_000010_annotated.jpg",
                    "detections": [
                        {"species": [{"species": "Blue Jay", "probability": 0.88}]},
                    ],
                },
                {
                    "annotated_file": "still_01_000020_annotated.jpg",
                    "detections": [
                        {"species": [{"species": "Northern Cardinal", "probability": 0.93}]},
                        {"species": [{"species": "Mourning Dove", "probability": 0.97}]},
                    ],
                },
            ],
        }

        assert job.thumbnail_url == "/jobs/abc123/crops/still_01_000020_annotated.jpg"

    def test_thumbnail_url_quotes_special_characters(self):
        job = Job(id="abc123", filename="batch.jpg", media_type="images")
        job.status = "done"
        job.result = {
            "type": "images",
            "images": [
                {
                    "annotated_file": "img 1 annotated.jpg",
                    "species_summary": [{"species": "Blue Jay", "probability": 0.92}],
                    "detections": [{"bbox": [0, 0, 10, 10]}],
                },
            ],
        }

        assert job.thumbnail_url == "/jobs/abc123/crops/img%201%20annotated.jpg"


# ---------------------------------------------------------------------------
# Job.submitted_by
# ---------------------------------------------------------------------------


class TestJobSubmittedBy:
    def test_default_is_none(self):
        job = Job(id="abc123", filename="bird.mp4", media_type="video")
        assert job.submitted_by is None

    def test_can_be_set(self):
        job = Job(id="abc123", filename="bird.mp4", media_type="video")
        job.submitted_by = "user@example.com"
        assert job.submitted_by == "user@example.com"

    def test_can_be_set_to_none(self):
        job = Job(id="abc123", filename="bird.mp4", media_type="video")
        job.submitted_by = "user@example.com"
        job.submitted_by = None
        assert job.submitted_by is None


# ---------------------------------------------------------------------------
# _persist_submitted_by
# ---------------------------------------------------------------------------


class TestPersistSubmittedBy:
    def test_writes_submitted_by_to_json(self, tmp_path):
        stem = "abc123_bird_video"
        results_dir = tmp_path
        json_path = results_dir / f"{stem}_results.json"
        initial_data = {"video": "/path/to/bird.mp4", "tracks": []}
        json_path.write_text(json.dumps(initial_data))

        _persist_submitted_by(results_dir, stem, "user@example.com")

        updated = json.loads(json_path.read_text())
        assert updated["submitted_by"] == "user@example.com"
        assert updated["video"] == "/path/to/bird.mp4"

    def test_missing_json_is_a_noop(self, tmp_path):
        # Should not raise even when the file doesn't exist
        _persist_submitted_by(tmp_path, "nonexistent_stem", "user@example.com")

    def test_overwrites_existing_submitted_by(self, tmp_path):
        stem = "def456_vid"
        json_path = tmp_path / f"{stem}_results.json"
        json_path.write_text(json.dumps({"submitted_by": "old@example.com"}))

        _persist_submitted_by(tmp_path, stem, "new@example.com")

        updated = json.loads(json_path.read_text())
        assert updated["submitted_by"] == "new@example.com"


# ---------------------------------------------------------------------------
# _load_existing_jobs restores submitted_by
# ---------------------------------------------------------------------------


class TestLoadExistingJobsSubmittedBy:
    def test_submitted_by_restored_from_json(self, tmp_path, monkeypatch):
        import src.webapp as webapp_module

        # Clear and patch the module-level _jobs dict
        monkeypatch.setattr(webapp_module, "_jobs", {})

        job_id = "a" * 32
        stem = f"{job_id}_birdvideo"
        result_data = {
            "type": "video",
            "video": "/path/to/bird.mp4",
            "source_filename": "bird.mp4",
            "display_name": "bird.mp4",
            "submitted_by": "tester@example.com",
            "tracks": [],
            "video_predictions": [],
        }
        result_file = tmp_path / f"{stem}_results.json"
        result_file.write_text(json.dumps(result_data))

        _load_existing_jobs(tmp_path)

        assert job_id in webapp_module._jobs
        assert webapp_module._jobs[job_id].submitted_by == "tester@example.com"

    def test_submitted_by_none_when_absent_from_json(self, tmp_path, monkeypatch):
        import src.webapp as webapp_module

        monkeypatch.setattr(webapp_module, "_jobs", {})

        job_id = "b" * 32
        stem = f"{job_id}_oldvideo"
        result_data = {
            "type": "video",
            "video": "/path/to/old.mp4",
            "source_filename": "old.mp4",
            "display_name": "old.mp4",
            "tracks": [],
            "video_predictions": [],
        }
        result_file = tmp_path / f"{stem}_results.json"
        result_file.write_text(json.dumps(result_data))

        _load_existing_jobs(tmp_path)

        assert job_id in webapp_module._jobs
        assert webapp_module._jobs[job_id].submitted_by is None


# ---------------------------------------------------------------------------
# _load_api_tokens
# ---------------------------------------------------------------------------


class TestLoadApiTokens:
    def test_missing_file_returns_empty(self, tmp_path):
        assert _load_api_tokens(tmp_path / "nope.yaml") == {}

    def test_well_formed_file(self, tmp_path):
        f = tmp_path / "tokens.yaml"
        f.write_text(
            "tokens:\n"
            "  - name: birdcamgrabber\n"
            "    token: abc123\n"
            "  - name: other-client\n"
            "    token: def456\n"
        )
        result = _load_api_tokens(f)
        assert result == {"abc123": "birdcamgrabber", "def456": "other-client"}

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "tokens.yaml"
        f.write_text("")
        assert _load_api_tokens(f) == {}

    def test_missing_tokens_key(self, tmp_path):
        f = tmp_path / "tokens.yaml"
        f.write_text("other: value\n")
        assert _load_api_tokens(f) == {}

    def test_skips_entries_without_token(self, tmp_path):
        f = tmp_path / "tokens.yaml"
        f.write_text(
            "tokens:\n"
            "  - name: good\n"
            "    token: t1\n"
            "  - name: bad-no-token\n"
            "  - name: bad-empty-token\n"
            '    token: ""\n'
        )
        assert _load_api_tokens(f) == {"t1": "good"}

    def test_unknown_name_falls_back(self, tmp_path):
        f = tmp_path / "tokens.yaml"
        f.write_text("tokens:\n  - token: just-a-token\n")
        assert _load_api_tokens(f) == {"just-a-token": "unknown"}

    def test_malformed_yaml_returns_empty(self, tmp_path):
        f = tmp_path / "tokens.yaml"
        f.write_text("tokens: [this is not: valid yaml\n")
        assert _load_api_tokens(f) == {}

    def test_tokens_not_a_list(self, tmp_path):
        f = tmp_path / "tokens.yaml"
        f.write_text("tokens: not-a-list\n")
        assert _load_api_tokens(f) == {}


# ---------------------------------------------------------------------------
# _persist_source_event_id
# ---------------------------------------------------------------------------


class TestPersistSourceEventId:
    def test_writes_source_event_id_to_json(self, tmp_path):
        stem = "abc123_clip"
        json_path = tmp_path / f"{stem}_results.json"
        json_path.write_text(json.dumps({"video": "/x.mp4"}))

        _persist_source_event_id(tmp_path, stem, "evt-xyz")

        updated = json.loads(json_path.read_text())
        assert updated["source_event_id"] == "evt-xyz"
        assert updated["video"] == "/x.mp4"

    def test_missing_json_is_a_noop(self, tmp_path):
        _persist_source_event_id(tmp_path, "nonexistent", "evt-xyz")

    def test_overwrites_existing_value(self, tmp_path):
        stem = "abc123_clip"
        json_path = tmp_path / f"{stem}_results.json"
        json_path.write_text(json.dumps({"source_event_id": "old"}))

        _persist_source_event_id(tmp_path, stem, "new")

        assert json.loads(json_path.read_text())["source_event_id"] == "new"


# ---------------------------------------------------------------------------
# _load_existing_jobs restores source_event_id
# ---------------------------------------------------------------------------


class TestLoadExistingJobsSourceEventId:
    def test_source_event_id_restored_from_json(self, tmp_path, monkeypatch):
        import src.webapp as webapp_module

        monkeypatch.setattr(webapp_module, "_jobs", {})

        job_id = "c" * 32
        stem = f"{job_id}_clip"
        result_data = {
            "type": "video",
            "video": "/path/to/clip.mp4",
            "source_filename": "clip.mp4",
            "display_name": "clip.mp4",
            "source_event_id": "evt-12345",
            "tracks": [],
            "video_predictions": [],
        }
        (tmp_path / f"{stem}_results.json").write_text(json.dumps(result_data))

        _load_existing_jobs(tmp_path)

        assert webapp_module._jobs[job_id].source_event_id == "evt-12345"

    def test_source_event_id_none_when_absent(self, tmp_path, monkeypatch):
        import src.webapp as webapp_module

        monkeypatch.setattr(webapp_module, "_jobs", {})

        job_id = "d" * 32
        stem = f"{job_id}_clip"
        (tmp_path / f"{stem}_results.json").write_text(
            json.dumps(
                {
                    "type": "video",
                    "video": "/p.mp4",
                    "source_filename": "p.mp4",
                    "display_name": "p.mp4",
                    "tracks": [],
                    "video_predictions": [],
                }
            )
        )

        _load_existing_jobs(tmp_path)
        assert webapp_module._jobs[job_id].source_event_id is None


# ---------------------------------------------------------------------------
# current_user_email
# ---------------------------------------------------------------------------


class TestCurrentUserEmail:
    def test_returns_email_from_session(self):
        request = MagicMock()
        request.session = {"email": "User@Example.COM"}
        assert current_user_email(request) == "user@example.com"

    def test_returns_none_when_no_session_email(self):
        request = MagicMock()
        request.session = {}
        assert current_user_email(request) is None


# ---------------------------------------------------------------------------
# SAFARI_COMPATIBLE_CODECS
# ---------------------------------------------------------------------------


class TestSafariCompatibleCodecs:
    def test_h264_is_compatible(self):
        assert "avc1" in SAFARI_COMPATIBLE_CODECS

    def test_hevc_variants_are_compatible(self):
        assert "hvc1" in SAFARI_COMPATIBLE_CODECS
        assert "hev1" in SAFARI_COMPATIBLE_CODECS

    def test_vp9_not_compatible(self):
        assert "VP90" not in SAFARI_COMPATIBLE_CODECS

    def test_av1_not_compatible(self):
        assert "av01" not in SAFARI_COMPATIBLE_CODECS


# ---------------------------------------------------------------------------
# transcode_to_h264
# ---------------------------------------------------------------------------


class TestTranscodeToH264:
    def test_calls_ffmpeg_with_expected_args(self, tmp_path):
        input_file = tmp_path / "clip.webm"
        input_file.write_bytes(b"fake video data")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # Simulate ffmpeg creating the tmp file
            tmp_out = tmp_path / "clip_h264.tmp.mp4"
            mock_run.side_effect = lambda *a, **kw: (
                tmp_out.write_bytes(b"fake h264"),
                MagicMock(returncode=0),
            )[1]

            transcode_to_h264(input_file)

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-movflags" in cmd
        assert "+faststart" in cmd

    def test_returns_output_path_on_success(self, tmp_path):
        input_file = tmp_path / "clip.webm"
        input_file.write_bytes(b"fake")
        expected_output = tmp_path / "clip_h264.mp4"

        def fake_run(*args, **kwargs):
            # ffmpeg writes tmp file
            (tmp_path / "clip_h264.tmp.mp4").write_bytes(b"fake h264")
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=fake_run):
            result = transcode_to_h264(input_file)

        assert result == expected_output
        assert expected_output.exists()

    def test_returns_none_on_ffmpeg_failure(self, tmp_path):
        input_file = tmp_path / "clip.webm"
        input_file.write_bytes(b"fake")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr=b"error")
            result = transcode_to_h264(input_file)

        assert result is None

    def test_returns_none_when_ffmpeg_not_found(self, tmp_path):
        input_file = tmp_path / "clip.webm"
        input_file.write_bytes(b"fake")

        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = transcode_to_h264(input_file)

        assert result is None

    def test_returns_none_on_timeout(self, tmp_path):
        input_file = tmp_path / "clip.webm"
        input_file.write_bytes(b"fake")

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 600)):
            result = transcode_to_h264(input_file)

        assert result is None

    def test_reuses_existing_output(self, tmp_path):
        input_file = tmp_path / "clip.webm"
        input_file.write_bytes(b"fake")
        output_file = tmp_path / "clip_h264.mp4"
        output_file.write_bytes(b"already transcoded")

        with patch("subprocess.run") as mock_run:
            result = transcode_to_h264(input_file)

        mock_run.assert_not_called()
        assert result == output_file

    def test_cleans_up_tmp_on_failure(self, tmp_path):
        input_file = tmp_path / "clip.webm"
        input_file.write_bytes(b"fake")
        tmp_out = tmp_path / "clip_h264.tmp.mp4"

        def fake_run(*args, **kwargs):
            tmp_out.write_bytes(b"partial")
            return MagicMock(returncode=1, stderr=b"error")

        with patch("subprocess.run", side_effect=fake_run):
            transcode_to_h264(input_file)

        assert not tmp_out.exists()
