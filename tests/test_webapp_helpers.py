"""Unit tests for pure helper / auth logic in src/webapp.py."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.webapp import (
    AuthSettings,
    build_auth_settings,
    build_job_display_name,
    can_upload_email,
    classify_media_type,
    debug_mode_enabled,
    normalize_email,
    normalize_secret,
    parse_bool,
    require_upload_access,
    resolution_warning_text,
    result_name_seed,
    slugify_result_name,
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
            for key in ("BIRDVISION_DEBUG", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "SESSION_SECRET", "GOOGLE_REDIRECT_URI"):
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
            for key in ("BIRDVISION_DEBUG", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "SESSION_SECRET"):
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
        with patch.dict(os.environ, {
            "GOOGLE_CLIENT_ID": "env-id",
            "GOOGLE_CLIENT_SECRET": "env-secret",
            "SESSION_SECRET": "env-session",
        }, clear=True):
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
            for key in ("BIRDVISION_DEBUG", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "SESSION_SECRET"):
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
            enabled=False, debug_mode=True, google_client_id=None,
            google_client_secret=None, redirect_uri=None,
            session_secret=None, allowed_emails=set(),
        )
        assert can_upload_email(None, settings) is True

    def test_allowed_email(self):
        settings = AuthSettings(
            enabled=True, debug_mode=False, google_client_id="x",
            google_client_secret="x", redirect_uri=None,
            session_secret="x", allowed_emails={"user@test.com"},
        )
        assert can_upload_email("user@test.com", settings) is True

    def test_disallowed_email(self):
        settings = AuthSettings(
            enabled=True, debug_mode=False, google_client_id="x",
            google_client_secret="x", redirect_uri=None,
            session_secret="x", allowed_emails={"user@test.com"},
        )
        assert can_upload_email("other@test.com", settings) is False

    def test_none_email_when_auth_enabled(self):
        settings = AuthSettings(
            enabled=True, debug_mode=False, google_client_id="x",
            google_client_secret="x", redirect_uri=None,
            session_secret="x", allowed_emails={"user@test.com"},
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
            enabled=False, debug_mode=True, google_client_id=None,
            google_client_secret=None, redirect_uri=None,
            session_secret=None, allowed_emails=set(),
        )
        assert require_upload_access(_make_request(), settings) is None

    def test_no_email_page_redirects(self):
        settings = AuthSettings(
            enabled=True, debug_mode=False, google_client_id="x",
            google_client_secret="x", redirect_uri=None,
            session_secret="x", allowed_emails=set(),
        )
        result = require_upload_access(_make_request("/upload"), settings)
        assert result is not None
        assert result.status_code == 303

    def test_no_email_api_returns_401(self):
        settings = AuthSettings(
            enabled=True, debug_mode=False, google_client_id="x",
            google_client_secret="x", redirect_uri=None,
            session_secret="x", allowed_emails=set(),
        )
        result = require_upload_access(_make_request("/api/upload"), settings)
        assert result is not None
        assert result.status_code == 401

    def test_non_whitelisted_api_returns_403(self):
        settings = AuthSettings(
            enabled=True, debug_mode=False, google_client_id="x",
            google_client_secret="x", redirect_uri=None,
            session_secret="x", allowed_emails={"allowed@test.com"},
        )
        request = _make_request("/api/upload", session={"email": "other@test.com"})
        result = require_upload_access(request, settings)
        assert result is not None
        assert result.status_code == 403

    def test_whitelisted_returns_none(self):
        settings = AuthSettings(
            enabled=True, debug_mode=False, google_client_id="x",
            google_client_secret="x", redirect_uri=None,
            session_secret="x", allowed_emails={"user@test.com"},
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
