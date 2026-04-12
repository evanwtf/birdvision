"""Integration tests for POST /api/v1/videos."""

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import src.webapp as webapp_module
from src.webapp import Job
from src.video_metadata import MediaMetadata


class _FakePipeline:
    """Stand-in pipeline so create_app() doesn't try to load BioCLIP/YOLO."""

    def apply_config(self, config):
        pass

    def process_video(self, *args, **kwargs):
        return {}

    def process_images(self, *args, **kwargs):
        return {}


@pytest.fixture(autouse=True)
def _reset_jobs(monkeypatch):
    """Reset module-level _jobs and stub the pipeline factory for every test."""
    monkeypatch.setattr(webapp_module, "_jobs", {})
    monkeypatch.setattr(webapp_module, "_init_pipeline", lambda c: _FakePipeline())
    monkeypatch.setattr(
        webapp_module,
        "inspect_media",
        lambda path: MediaMetadata(
            width=1920,
            height=1080,
            duration_s=1.0,
            fps=30.0,
            video_codec="avc1",
        ),
    )


def _make_client(tmp_path: Path, *, with_tokens: bool = True) -> TestClient:
    """Build a TestClient with isolated upload/results dirs."""
    cfg: dict = {
        "webapp": {
            "upload_dir": str(tmp_path / "videos"),
        },
        "output": {"results_dir": str(tmp_path / "results")},
        "species": {},
    }
    if with_tokens:
        tokens_file = tmp_path / "tokens.yaml"
        tokens_file.write_text(yaml.safe_dump({
            "tokens": [{"name": "birdcamgrabber", "token": "secret-token"}]
        }))
        cfg["webapp"]["api_tokens_file"] = str(tokens_file)

    app = webapp_module.create_app(cfg, templates_dir="templates")
    return TestClient(app)


_VALID_TS = "2026-04-07T12:34:56Z"


class TestApiVideoUpload:
    def test_missing_token_returns_401(self, tmp_path):
        with _make_client(tmp_path) as client:
            r = client.post(
                "/api/v1/videos",
                files={"file": ("clip.mp4", b"fakebytes", "video/mp4")},
                data={"captured_at": _VALID_TS},
            )
            assert r.status_code == 401

    def test_bad_token_returns_401(self, tmp_path):
        with _make_client(tmp_path) as client:
            r = client.post(
                "/api/v1/videos",
                headers={"X-API-Token": "wrong-token"},
                files={"file": ("clip.mp4", b"fakebytes", "video/mp4")},
                data={"captured_at": _VALID_TS},
            )
            assert r.status_code == 401

    def test_no_tokens_configured_returns_503(self, tmp_path):
        with _make_client(tmp_path, with_tokens=False) as client:
            r = client.post(
                "/api/v1/videos",
                headers={"X-API-Token": "anything"},
                files={"file": ("clip.mp4", b"x", "video/mp4")},
                data={"captured_at": _VALID_TS},
            )
            assert r.status_code == 503

    def test_bad_captured_at_returns_400(self, tmp_path):
        with _make_client(tmp_path) as client:
            r = client.post(
                "/api/v1/videos",
                headers={"X-API-Token": "secret-token"},
                files={"file": ("clip.mp4", b"fakebytes", "video/mp4")},
                data={"captured_at": "not-a-timestamp"},
            )
            assert r.status_code == 400
            assert "captured_at" in r.json()["detail"]

    def test_empty_file_returns_400(self, tmp_path):
        with _make_client(tmp_path) as client:
            r = client.post(
                "/api/v1/videos",
                headers={"X-API-Token": "secret-token"},
                files={"file": ("clip.mp4", b"", "video/mp4")},
                data={"captured_at": _VALID_TS},
            )
            assert r.status_code == 400

    def test_non_video_extension_returns_400(self, tmp_path):
        with _make_client(tmp_path) as client:
            r = client.post(
                "/api/v1/videos",
                headers={"X-API-Token": "secret-token"},
                files={"file": ("clip.txt", b"hello", "text/plain")},
                data={"captured_at": _VALID_TS},
            )
            assert r.status_code == 400
            assert "unsupported file extension" in r.json()["detail"]

    def test_happy_path_returns_202_and_creates_job(self, tmp_path):
        with _make_client(tmp_path) as client:
            r = client.post(
                "/api/v1/videos",
                headers={"X-API-Token": "secret-token"},
                files={"file": ("clip.mp4", b"fake-mp4-bytes", "video/mp4")},
                data={
                    "captured_at": _VALID_TS,
                    "latitude": "40.77",
                    "longitude": "-73.97",
                    "source": "birdcamgrabber",
                    "source_event_id": "evt-abc-123",
                },
            )
            assert r.status_code == 202, r.text
            body = r.json()
            assert "job_id" in body
            assert body["url"] == f"/jobs/{body['job_id']}"
            assert body["status"] == "pending"

            job = webapp_module._jobs[body["job_id"]]
            assert job.media_type == "video"
            assert job.source_event_id == "evt-abc-123"
            assert job.submitted_by == "birdcamgrabber@api"
            assert job.video_meta is not None
            assert job.video_meta.latitude == pytest.approx(40.77)
            assert job.video_meta.longitude == pytest.approx(-73.97)
            assert job.selected_date is not None
            assert job.selected_date.year == 2026

            # File should have been written under the asset store
            asset_dir = tmp_path / "videos" / "assets"
            stored = list(asset_dir.glob("*.mp4"))
            assert len(stored) == 1
            assert stored[0].read_bytes() == b"fake-mp4-bytes"

    def test_submitted_by_uses_token_name_when_source_omitted(self, tmp_path):
        with _make_client(tmp_path) as client:
            r = client.post(
                "/api/v1/videos",
                headers={"X-API-Token": "secret-token"},
                files={"file": ("clip.mp4", b"bytes", "video/mp4")},
                data={"captured_at": _VALID_TS},
            )
            assert r.status_code == 202
            job = webapp_module._jobs[r.json()["job_id"]]
            assert job.submitted_by == "birdcamgrabber@api"

    def test_dedup_via_asset_store(self, tmp_path):
        """Re-uploading identical bytes reuses the same asset on disk."""
        with _make_client(tmp_path) as client:
            for _ in range(2):
                r = client.post(
                    "/api/v1/videos",
                    headers={"X-API-Token": "secret-token"},
                    files={"file": ("clip.mp4", b"identical-bytes", "video/mp4")},
                    data={"captured_at": _VALID_TS},
                )
                assert r.status_code == 202

            asset_dir = tmp_path / "videos" / "assets"
            stored = list(asset_dir.glob("*.mp4"))
            assert len(stored) == 1, "asset store should dedup identical uploads"
            # Two distinct jobs were still created
            assert len(webapp_module._jobs) == 2

    def test_api_rejects_unreadable_video_and_deletes_asset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webapp_module, "inspect_media", lambda path: MediaMetadata())

        with _make_client(tmp_path) as client:
            response = client.post(
                "/api/v1/videos",
                headers={"X-API-Token": "secret-token"},
                files={"file": ("clip.mp4", b"not-a-real-video", "video/mp4")},
                data={"captured_at": _VALID_TS},
            )

        assert response.status_code == 400
        assert "could not be opened for processing" in response.json()["detail"]
        assert not list((tmp_path / "videos" / "assets").glob("*"))
        payload = json.loads((tmp_path / "videos" / "asset_index.json").read_text())
        assert payload["assets"] == {}

    def test_browser_inspect_reports_bad_video_without_persisting_asset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webapp_module, "inspect_media", lambda path: MediaMetadata())

        with _make_client(tmp_path) as client:
            inspect_response = client.post(
                "/api/uploads/inspect",
                files={"file": ("clip.mp4", b"not-a-real-video", "video/mp4")},
            )

        assert inspect_response.status_code == 200
        body = inspect_response.json()
        assert body["assets"][0]["processable"] is False
        assert "could not be opened for processing" in body["assets"][0]["processing_issue"]
        assert body["batch"]["valid"] is False
        assert body["batch"]["message_level"] == "info"
        assert not list((tmp_path / "videos" / "assets").glob("*"))
        payload = json.loads((tmp_path / "videos" / "asset_index.json").read_text())
        assert payload["assets"] == {}

    def test_browser_finalize_returns_info_when_only_bad_assets_selected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webapp_module, "inspect_media", lambda path: MediaMetadata())

        with _make_client(tmp_path) as client:
            inspect_response = client.post(
                "/api/uploads/inspect",
                files={"file": ("clip.mp4", b"not-a-real-video", "video/mp4")},
            )
            asset = inspect_response.json()["assets"][0]
            finalize_response = client.post(
                "/api/uploads/finalize",
                json={
                    "assets": [{
                        "sha256": asset["sha256"],
                        "original_filename": asset["original_filename"],
                        "selected": True,
                    }]
                },
            )

        assert finalize_response.status_code == 200
        body = finalize_response.json()
        assert body["jobs_created"] == 0
        assert "could not be opened for processing" in body["info_message"]

    def test_browser_inspect_rejects_more_than_twenty_photos(self, tmp_path):
        files = [
            ("file", (f"photo-{idx}.jpg", b"jpeg-bytes", "image/jpeg"))
            for idx in range(21)
        ]

        with _make_client(tmp_path) as client:
            response = client.post("/api/uploads/inspect", files=files)

        assert response.status_code == 400
        assert response.json()["error"] == "Select at most 20 photos per upload."

    def test_browser_inspect_rejects_unsupported_extension(self, tmp_path):
        with _make_client(tmp_path) as client:
            response = client.post(
                "/api/uploads/inspect",
                files={"file": ("payload.exe", b"junk", "application/octet-stream")},
            )

        assert response.status_code == 400
        assert "unsupported file extension" in response.json()["error"]

    def test_browser_inspect_rejects_oversized_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webapp_module, "MAX_UPLOAD_FILE_BYTES", 4)

        with _make_client(tmp_path) as client:
            response = client.post(
                "/api/uploads/inspect",
                files={"file": ("clip.mp4", b"12345", "video/mp4")},
            )

        assert response.status_code == 400
        assert "50 MB upload limit" in response.json()["error"] or "upload limit" in response.json()["error"]


class TestJobSlugRoutes:
    def test_job_redirect_uses_high_confidence_species_slug(self, tmp_path):
        with _make_client(tmp_path) as client:
            job = Job(id="a" * 32, filename="8a7c4f19.mp4", media_type="video")
            job.status = "done"
            job.result = {
                "type": "video",
                "tracks": [{"track_id": 1}],
                "video_predictions": [{
                    "species": "Blue Jay",
                    "presence_probability": 0.9731,
                    "supporting_tracks": 1,
                }],
            }
            webapp_module._jobs[job.id] = job

            response = client.get(f"/jobs/{job.id}", follow_redirects=False)

            assert response.status_code == 301
            assert response.headers["location"] == f"/jobs/{job.id}/blue-jay"


class TestRecentVisitorsListing:
    def test_index_and_api_include_recent_visitors_thumbnail(self, tmp_path):
        with _make_client(tmp_path) as client:
            job = Job(id="c" * 32, filename="clip.mp4", media_type="video")
            job.status = "done"
            job.result = {
                "type": "video",
                "tracks": [{"track_id": 1}],
                "video_predictions": [{
                    "species": "Blue Jay",
                    "presence_probability": 0.9731,
                    "supporting_tracks": 1,
                }],
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
                            {"species": [{"species": "Northern Cardinal", "probability": 0.94}]},
                        ],
                    },
                ],
            }
            webapp_module._jobs[job.id] = job

            page = client.get("/")
            assert page.status_code == 200
            assert "Recent Visitors" in page.text
            assert f'/jobs/{job.id}/crops/still_01_000020_annotated.jpg' in page.text
            assert "Recent Jobs" not in page.text

            api = client.get("/api/jobs")
            assert api.status_code == 200
            payload = api.json()
            assert payload["jobs"][0]["thumbnail_url"] == (
                f"/jobs/{job.id}/crops/still_01_000020_annotated.jpg"
            )

    def test_noncanonical_job_slug_redirects_to_current_species_slug(self, tmp_path):
        with _make_client(tmp_path) as client:
            job = Job(id="b" * 32, filename="8a7c4f19.mp4", media_type="video")
            job.status = "done"
            job.result = {
                "type": "video",
                "tracks": [{"track_id": 1}],
                "video_predictions": [{
                    "species": "Blue Jay",
                    "presence_probability": 0.9731,
                    "supporting_tracks": 1,
                }],
            }
            webapp_module._jobs[job.id] = job

            response = client.get(f"/jobs/{job.id}/random-gibberish", follow_redirects=False)

            assert response.status_code == 301
            assert response.headers["location"] == f"/jobs/{job.id}/blue-jay"
