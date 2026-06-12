"""
BirdVision web interface — inspect uploads, create jobs, view results.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from time import perf_counter
from typing import Any, Optional
from urllib.parse import quote

import yaml
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.middleware.sessions import SessionMiddleware

try:
    from authlib.integrations.starlette_client import OAuth
except ImportError:  # pragma: no cover - keeps app importable until deps are installed
    OAuth = None

import contextlib

from .pipeline import BirdIdentificationPipeline
from .video_metadata import MediaMetadata, VideoMetadata, inspect_media

logger = logging.getLogger(__name__)
access_logger = logging.getLogger("birdvision.access")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
KNOWN_UPLOAD_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
MAX_IMAGE_UPLOAD_COUNT = 20
MAX_UPLOAD_FILE_BYTES = 50 * 1024 * 1024
GOOGLE_SERVER_METADATA_URL = "https://accounts.google.com/.well-known/openid-configuration"
THEME_COOKIE_NAME = "birdvision_theme"
THEME_OPTIONS = (
    {"id": "super-birdy", "label": "Super Birdy"},
    {"id": "boring", "label": "Boring"},
)
DEFAULT_THEME_ID = "super-birdy"
VALID_THEME_IDS = {theme["id"] for theme in THEME_OPTIONS}


def slugify_job_label(label: str) -> str:
    """Normalize a filename or species label into a URL slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or "job"


def safe_redirect_path(value: str, default: str = "/") -> str:
    """Return ``value`` only if it is a safe same-site relative path.

    Guards against open redirects: a plain ``startswith("/")`` check accepts
    protocol-relative targets like ``//evil.example`` (and ``/\\evil.example``,
    which some browsers normalize to ``//``), which browsers resolve to an
    external origin. Anything that is not a single-slash-rooted path falls back
    to ``default``.
    """
    if not value.startswith("/"):
        return default
    if value.startswith("//") or value.startswith("/\\"):
        return default
    return value


async def _run_off_event_loop(func, *args, **kwargs):
    """Run a blocking callable on the default thread pool, awaiting the result.

    Upload inspection/ingest does synchronous heavy work (hashing, large disk
    writes, an exiftool subprocess, OpenCV probes); calling it directly from an
    async handler stalls the whole event loop. The default executor is used
    (not the single-worker pipeline executor) so uploads don't queue behind a
    running inference job.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


@dataclass
class Job:
    id: str
    filename: str
    media_type: str = "video"

    def __post_init__(self):
        self.status = "pending"  # pending | running | done | error
        self.result: dict | None = None
        self.error: str | None = None
        self.video_meta: VideoMetadata | None = None
        self.image_paths: list[str] = []
        self.assets: list[dict[str, Any]] = []
        self.created_at: datetime = datetime.now(UTC)
        self.selected_date: datetime | None = None
        self.result_stem: str | None = None
        self.submitted_by: str | None = None
        self.source_event_id: str | None = None  # opaque upstream id (API uploads only)

    @property
    def slug(self) -> str:
        """URL-friendly slug derived from a high-confidence species or filename."""
        return slugify_job_label(self._slug_label())

    def _slug_label(self) -> str:
        species_label = self._high_confidence_species_label()
        if species_label:
            return species_label
        return Path(self.filename).stem

    def _high_confidence_species_label(self) -> str | None:
        """Return the top species label when confidence is strictly above 90%."""
        if self.status != "done" or not self.result:
            return None

        best_species = ""
        best_prob = 0.0
        if self.result.get("type") == "images":
            for img in self.result.get("images", []):
                for pred in img.get("species_summary", []):
                    species = pred.get("species", "")
                    prob = pred.get("probability", 0)
                    if species and prob > best_prob:
                        best_species = species
                        best_prob = prob
        else:
            top_prediction = next(iter(self.result.get("video_predictions", [])), None)
            if top_prediction:
                best_species = top_prediction.get("species", "")
                best_prob = top_prediction.get("presence_probability", 0)

        if best_species and best_prob > 0.9:
            return best_species
        return None

    @property
    def has_detections(self) -> bool:
        """False only for completed jobs where the detector found nothing at all.
        Pending, running, and error jobs always return True so they remain visible.
        A job with detections but no confident ID also returns True.
        """
        if self.status != "done" or self.result is None:
            return True
        if self.result.get("type") == "images":
            return any(img.get("detections") for img in self.result.get("images", []))
        return bool(self.result.get("tracks"))

    @property
    def media_label(self) -> str:
        if self.media_type == "images":
            n = self._image_count()
            return f"📷×{n}" if n > 1 else "📷"
        return "🎥"

    @property
    def summary(self) -> str:
        label = self._species_summary_label()
        if label:
            return label

        if self.media_type == "images":
            if self._image_count() == 1:
                return self.filename
            return "Photo batch"
        return self.filename

    def _image_count(self) -> int:
        n = len(self.assets) or len(self.image_paths)
        if n == 0 and self.result and self.result.get("image_info"):
            n = self.result["image_info"].get("count", 0)
        return n

    def _species_summary_label(self) -> str | None:
        """Build a label like 'Mourning Dove (82%), Blue Jay (74%), 2 others'."""
        if not self.result or self.status != "done":
            return None

        # Collect top (species, prob) pairs
        top_species: list[tuple[str, float]] = []
        if self.result.get("type") == "images":
            seen: set[str] = set()
            for img in self.result.get("images", []):
                for pred in img.get("species_summary", []):
                    sp = pred.get("species", "")
                    prob = pred.get("probability", 0)
                    if sp and prob >= 0.15 and sp not in seen:
                        seen.add(sp)
                        top_species.append((sp, prob))
        else:
            for pred in self.result.get("video_predictions", []):
                sp = pred.get("species", "")
                prob = pred.get("presence_probability", 0)
                if sp and prob >= 0.15:
                    top_species.append((sp, prob))

        if not top_species:
            return None

        def fmt(sp: str, prob: float) -> str:
            return f"{sp} ({round(prob * 100):.0f}%)"

        # Build the species part: show up to 2 with %, then "N others"
        if len(top_species) <= 2:
            species_text = ", ".join(fmt(sp, prob) for sp, prob in top_species)
        else:
            others = len(top_species) - 2
            species_text = (
                f"{fmt(*top_species[0])}, {fmt(*top_species[1])}, "
                f"{others} other{'s' if others != 1 else ''}"
            )

        return species_text

    @property
    def thumbnail_url(self) -> str | None:
        filename = self._thumbnail_filename()
        if not filename:
            return None
        return f"/jobs/{self.id}/crops/{quote(filename)}"

    def _thumbnail_filename(self) -> str | None:
        if not self.result or self.status != "done":
            return None

        best_filename = None
        best_prob = -1.0

        if self.result.get("type") == "images":
            for img in self.result.get("images", []):
                annotated_file = img.get("annotated_file")
                if not annotated_file:
                    continue
                prob = max(
                    (
                        float(pred.get("probability", 0) or 0)
                        for pred in img.get("species_summary", [])
                    ),
                    default=0.0,
                )
                if prob > best_prob:
                    best_prob = prob
                    best_filename = annotated_file
        else:
            for still in self.result.get("video_stills", []):
                annotated_file = still.get("annotated_file")
                if not annotated_file:
                    continue
                prob = max(
                    (
                        float(pred.get("probability", 0) or 0)
                        for det in still.get("detections", [])
                        for pred in det.get("species", [])
                    ),
                    default=0.0,
                )
                if prob > best_prob:
                    best_prob = prob
                    best_filename = annotated_file

        return best_filename


@dataclass(frozen=True)
class AuthSettings:
    enabled: bool
    debug_mode: bool
    google_client_id: str | None
    google_client_secret: str | None
    redirect_uri: str | None
    session_secret: str | None
    allowed_emails: set[str]


SAFARI_COMPATIBLE_CODECS = {"avc1", "hvc1", "hev1"}


def transcode_to_h264(input_path: Path) -> Path | None:
    """Transcode a video to H.264/AAC in an MP4 container using ffmpeg.

    Returns the output path on success, or None if ffmpeg is unavailable or fails.
    If the output file already exists, returns it immediately without re-transcoding.
    """
    output_path = input_path.parent / (input_path.stem + "_h264.mp4")
    if output_path.exists():
        logger.info(f"Transcoded file already exists, reusing: {output_path.name}")
        return output_path

    tmp_path = input_path.parent / (input_path.stem + "_h264.tmp.mp4")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(tmp_path),
            ],
            capture_output=True,
            timeout=600,
        )
        if result.returncode != 0:
            logger.error(
                f"ffmpeg transcode failed for {input_path.name}: "
                f"{result.stderr.decode(errors='replace')[-500:]}"
            )
            tmp_path.unlink(missing_ok=True)
            return None
        tmp_path.rename(output_path)
        logger.info(f"Transcoded {input_path.name} -> {output_path.name}")
        return output_path
    except FileNotFoundError:
        logger.error("ffmpeg not found; cannot transcode video")
        tmp_path.unlink(missing_ok=True)
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg timed out transcoding {input_path.name}")
        tmp_path.unlink(missing_ok=True)
        return None
    except Exception as exc:
        logger.error(f"Unexpected error transcoding {input_path.name}: {exc}")
        tmp_path.unlink(missing_ok=True)
        return None


class AssetStore:
    def __init__(self, upload_dir: Path):
        self.root = upload_dir
        self.asset_dir = upload_dir / "assets"
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = upload_dir / "asset_index.json"
        # Inspect/ingest now runs on thread-pool workers (see
        # _run_off_event_loop), so concurrent requests can touch the in-memory
        # index and its JSON file at once. Guard the read-modify-write regions.
        # Reentrant because inspect_bytes calls get() while already holding it.
        self._lock = threading.RLock()
        self._assets = self._load_index()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.exists():
            return {}
        try:
            payload = json.loads(self.index_path.read_text())
        except Exception as exc:
            logger.warning(f"Could not load asset index {self.index_path}: {exc}")
            self._preserve_corrupt_index()
            return {}
        assets = payload.get("assets", {})
        if isinstance(assets, dict):
            return assets
        return {}

    def _preserve_corrupt_index(self) -> None:
        """Move an unparseable index aside so the next save can't overwrite it.

        Keeps the bytes around for recovery (the index is also reconstructible
        from the content-addressed files under assets/).
        """
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        backup = self.index_path.parent / f"{self.index_path.name}.corrupt-{stamp}"
        try:
            os.replace(self.index_path, backup)
            logger.warning("Preserved corrupt asset index as %s", backup)
        except OSError as exc:
            logger.warning(f"Could not preserve corrupt asset index {self.index_path}: {exc}")

    def _save_index(self):
        payload = {
            "version": 1,
            "assets": self._assets,
        }
        # Atomic write: a crash mid-write must not truncate the live index
        # (which would orphan every stored asset on the next startup). Write to
        # a temp file in the same directory, then os.replace() onto the target.
        tmp_path = self.index_path.parent / f"{self.index_path.name}.tmp"
        tmp_path.write_text(json.dumps(payload, indent=2))
        os.replace(tmp_path, self.index_path)

    def get(self, sha256: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._assets.get(sha256)
            if record is None:
                return None
            stored_path = Path(record["stored_path"])
            if stored_path.exists():
                return record
            logger.warning(f"Indexed asset missing on disk, dropping stale entry: {stored_path}")
            self._assets.pop(sha256, None)
            self._save_index()
            return None

    def inspect_bytes(
        self,
        *,
        original_filename: str,
        content_type: str | None,
        data: bytes,
        client_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_name = Path(original_filename or "upload").name or "upload"
        ext = Path(safe_name).suffix.lower()
        media_type = classify_media_type(safe_name, content_type)
        sha256 = hashlib.sha256(data).hexdigest()

        existing = self.get(sha256)
        duplicate = existing is not None
        stored_ext = ext or (Path(existing["stored_path"]).suffix.lower() if existing else "")
        stored_path = self.asset_dir / f"{sha256}{stored_ext}"

        if not duplicate or not stored_path.exists():
            stored_path.write_bytes(data)

        inspected = inspect_media(str(stored_path))
        processing_issue = asset_processing_issue(media_type, inspected)
        record = {
            "sha256": sha256,
            "stored_path": str(stored_path),
            "media_type": media_type,
            "size_bytes": len(data),
            "extension": stored_ext,
            "width": first_value(inspected.width, client_metadata, "width"),
            "height": first_value(inspected.height, client_metadata, "height"),
            "duration_s": first_value(inspected.duration_s, client_metadata, "duration_s"),
            "fps": first_value(inspected.fps, client_metadata, "fps"),
            "recorded_at": inspected.recorded_at.isoformat() if inspected.recorded_at else None,
            "latitude": first_value(inspected.latitude, client_metadata, "latitude"),
            "longitude": first_value(inspected.longitude, client_metadata, "longitude"),
            "camera_info": inspected.camera_info,
            "video_codec": inspected.video_codec,
            "metadata_error": inspected.metadata_error,
            "processable": processing_issue is None,
            "processing_issue": processing_issue,
            "original_names": sorted(
                {
                    safe_name,
                    *(existing.get("original_names", []) if existing else []),
                }
            ),
            "created_at": (existing or {}).get("created_at") or utc_now_iso(),
            "last_seen_at": utc_now_iso(),
        }
        with self._lock:
            if processing_issue is not None:
                stored_path.unlink(missing_ok=True)
                self._assets.pop(sha256, None)
                self._save_index()
            else:
                self._assets[sha256] = record
                self._save_index()

        return {
            "sha256": sha256,
            "original_filename": safe_name,
            "stored_path": str(stored_path) if processing_issue is None else None,
            "media_type": media_type,
            "size_bytes": len(data),
            "width": record["width"],
            "height": record["height"],
            "duration_s": record["duration_s"],
            "fps": record["fps"],
            "recorded_at": record["recorded_at"],
            "latitude": record["latitude"],
            "longitude": record["longitude"],
            "camera_info": record["camera_info"],
            "metadata_status": "partial" if record["metadata_error"] else "ok",
            "metadata_error": record["metadata_error"],
            "processable": record["processable"],
            "processing_issue": record["processing_issue"],
            "resolution_warning": resolution_warning_text(
                media_type=media_type,
                width=record["width"],
                height=record["height"],
            ),
            "duplicate": duplicate if processing_issue is None else False,
            "duplicate_status": ("existing" if duplicate else "new")
            if processing_issue is None
            else "rejected",
            "canonical_path": str(stored_path) if processing_issue is None else None,
        }

    def ingest_path(self, path: str, original_filename: str | None = None) -> dict[str, Any]:
        source = Path(path)
        return self.inspect_bytes(
            original_filename=original_filename or source.name,
            content_type=None,
            data=source.read_bytes(),
            client_metadata=None,
        )

    def update_transcoded_path(self, sha256: str, new_path: str, new_codec: str) -> None:
        """Update the stored path and codec for an asset after transcoding."""
        with self._lock:
            record = self._assets.get(sha256)
            if record is None:
                return
            record["stored_path"] = new_path
            record["video_codec"] = new_codec
            self._save_index()


_jobs: dict[str, Job] = {}
_queue: asyncio.Queue
_executor = ThreadPoolExecutor(max_workers=1)


def create_app(
    config: dict, templates_dir: str = "templates", config_path: Path | None = None
) -> FastAPI:
    app = FastAPI(title="BirdVision")
    templates = Jinja2Templates(directory=templates_dir)
    initial_auth_settings = build_auth_settings(config)
    if initial_auth_settings.debug_mode:
        logger.warning("Webapp debug mode is enabled; auth gating is disabled.")
    elif not initial_auth_settings.enabled and any(
        (
            initial_auth_settings.google_client_id,
            initial_auth_settings.google_client_secret,
            initial_auth_settings.session_secret,
        )
    ):
        logger.warning(
            "Auth configuration is incomplete; authentication routes and upload gating are disabled."
        )

    app.add_middleware(
        SessionMiddleware,
        secret_key=initial_auth_settings.session_secret or secrets.token_hex(32),
        same_site="lax",
    )

    upload_dir = Path(config.get("webapp", {}).get("upload_dir", "videos"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.get("output", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    asset_store = AssetStore(upload_dir)

    api_tokens_path_str = config.get("webapp", {}).get("api_tokens_file")
    api_tokens: dict[str, str] = {}
    if api_tokens_path_str:
        api_tokens = _load_api_tokens(Path(api_tokens_path_str))
        if api_tokens:
            logger.info(f"Loaded {len(api_tokens)} API token(s) from {api_tokens_path_str}")
        else:
            logger.warning(
                f"API tokens file {api_tokens_path_str} missing or empty — "
                f"/api/v1/videos will return 503 until tokens are configured"
            )

    pipeline: BirdIdentificationPipeline | None = None

    def current_config() -> dict[str, Any]:
        return load_runtime_config(config, config_path)

    def current_auth_settings() -> AuthSettings:
        return build_auth_settings(current_config())

    def render_template(
        request: Request, template_name: str, context: dict[str, Any]
    ) -> HTMLResponse:
        merged = {
            **context,
            **build_template_auth_context(request, current_auth_settings()),
            **build_template_theme_context(request),
        }
        return templates.TemplateResponse(request, template_name, merged)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (perf_counter() - started_at) * 1000
            proxy_ip, real_ip, forwarded_for, x_real_ip = extract_request_ips(request)
            access_logger.exception(
                "request_failed method=%s path=%s status=500 duration_ms=%.1f proxy_ip=%s real_ip=%s x_forwarded_for=%s x_real_ip=%s email=%s",
                request.method,
                request.url.path,
                duration_ms,
                proxy_ip,
                real_ip,
                forwarded_for,
                x_real_ip,
                safe_current_user_email(request),
            )
            raise

        duration_ms = (perf_counter() - started_at) * 1000
        proxy_ip, real_ip, forwarded_for, x_real_ip = extract_request_ips(request)
        access_logger.info(
            "request method=%s path=%s status=%s duration_ms=%.1f proxy_ip=%s real_ip=%s x_forwarded_for=%s x_real_ip=%s email=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            proxy_ip,
            real_ip,
            forwarded_for,
            x_real_ip,
            safe_current_user_email(request),
        )
        return response

    @app.on_event("startup")
    async def startup():
        nonlocal pipeline
        global _queue
        _queue = asyncio.Queue()

        _load_existing_jobs(results_dir)

        logger.info("Loading pipeline (models may download on first run)…")
        loop = asyncio.get_event_loop()
        pipeline = await loop.run_in_executor(_executor, lambda: _init_pipeline(config))
        logger.info("Pipeline ready.")
        asyncio.create_task(_worker(loop, pipeline, config_path, results_dir, asset_store))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, page: int = 1):
        page_size = 20
        all_jobs = sorted(
            (j for j in _jobs.values() if j.has_detections),
            key=lambda j: j.created_at,
            reverse=True,
        )
        total = len(all_jobs)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        jobs_page = all_jobs[start : start + page_size]
        return render_template(
            request,
            "index.html",
            {
                "jobs": jobs_page,
                "page": page,
                "total_pages": total_pages,
                "total_jobs": total,
            },
        )

    @app.get("/api/jobs")
    async def api_jobs(page: int = 1):
        page_size = 20
        all_jobs = sorted(
            (j for j in _jobs.values() if j.has_detections),
            key=lambda j: j.created_at,
            reverse=True,
        )
        total = len(all_jobs)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        jobs_page = all_jobs[start : start + page_size]
        return {
            "jobs": [
                {
                    "id": j.id,
                    "slug": j.slug,
                    "status": j.status,
                    "media_label": j.media_label,
                    "media_type": j.media_type,
                    "summary": j.summary,
                    "thumbnail_url": j.thumbnail_url,
                    "created_at": j.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                for j in jobs_page
            ],
            "page": page,
            "total_pages": total_pages,
            "total_jobs": total,
            "has_active": any(j.status in ("pending", "running") for j in all_jobs),
        }

    @app.get("/login")
    async def login(request: Request):
        settings = current_auth_settings()
        if not settings.enabled:
            return RedirectResponse("/", status_code=303)
        try:
            google = build_google_oauth_client(settings)
        except RuntimeError as exc:
            return HTMLResponse(str(exc), status_code=500)
        redirect_uri = settings.redirect_uri or str(request.url_for("auth_callback"))
        return await google.authorize_redirect(request, redirect_uri)

    @app.get("/auth/callback")
    async def auth_callback(request: Request):
        settings = current_auth_settings()
        if not settings.enabled:
            return RedirectResponse("/", status_code=303)
        try:
            google = build_google_oauth_client(settings)
        except RuntimeError as exc:
            return HTMLResponse(str(exc), status_code=500)
        token = await google.authorize_access_token(request)
        user_info = token.get("userinfo")
        if not user_info:
            try:
                user_info = await google.parse_id_token(request, token)
            except Exception as exc:
                logger.warning(f"Could not parse Google ID token: {exc}")
                user_info = None

        email = normalize_email((user_info or {}).get("email"))
        if not email:
            return HTMLResponse("Google login did not return an email address.", status_code=400)

        request.session.clear()
        request.session["email"] = email
        return RedirectResponse("/", status_code=303)

    @app.get("/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/", status_code=303)

    @app.get("/theme/{theme_name}")
    async def set_theme(request: Request, theme_name: str, next: str = "/"):
        redirect_to = safe_redirect_path(next)
        normalized_theme = normalize_theme(theme_name)
        response = RedirectResponse(redirect_to, status_code=303)
        if normalized_theme == DEFAULT_THEME_ID:
            response.delete_cookie(THEME_COOKIE_NAME)
        else:
            response.set_cookie(
                THEME_COOKIE_NAME,
                normalized_theme,
                max_age=60 * 60 * 24 * 365,
                samesite="lax",
            )
        return response

    @app.post("/api/uploads/inspect")
    async def inspect_upload_candidates(request: Request):
        auth_response = require_upload_access(request, current_auth_settings())
        if auth_response is not None:
            return auth_response
        files, client_metadata = await _parse_upload_form(request)
        if not files:
            return JSONResponse({"error": "No file uploaded"}, status_code=HTTPStatus.BAD_REQUEST)
        selection_error = validate_upload_selection(files)
        if selection_error is not None:
            await log_rejected_upload_batch(request, files, reason=selection_error)
            return JSONResponse({"error": selection_error}, status_code=HTTPStatus.BAD_REQUEST)

        try:
            inspected_assets = await _inspect_files(files, client_metadata, asset_store)
        except RejectedUploadError as exc:
            log_rejected_upload(
                request,
                reason=exc.reason,
                filename=exc.filename,
                size_bytes=exc.size_bytes,
                sha256=exc.sha256,
                content_type=exc.content_type,
            )
            return JSONResponse({"error": exc.reason}, status_code=HTTPStatus.BAD_REQUEST)
        for asset in inspected_assets:
            if not asset_is_processable(asset):
                log_rejected_upload(
                    request,
                    reason=asset["processing_issue"],
                    filename=asset["original_filename"],
                    size_bytes=asset["size_bytes"],
                    sha256=asset["sha256"],
                    content_type=None,
                )
        batch = validate_asset_batch(inspected_assets)
        return JSONResponse(
            {
                "assets": inspected_assets,
                "batch": batch,
            }
        )

    @app.post("/api/uploads/finalize")
    async def finalize_upload(request: Request):
        auth_response = require_upload_access(request, current_auth_settings())
        if auth_response is not None:
            return auth_response
        payload = await request.json()
        selected_assets = payload.get("assets") or []
        if not isinstance(selected_assets, list):
            return JSONResponse(
                {"error": "Invalid asset selection"}, status_code=HTTPStatus.BAD_REQUEST
            )

        # Split multiple videos into one job per video
        groups = _split_asset_groups(selected_assets, asset_store)
        if not groups:
            selected_count = sum(1 for asset in selected_assets if asset.get("selected", True))
            if selected_count == 0:
                return JSONResponse(
                    {"error": "Select at least one asset to process."},
                    status_code=HTTPStatus.BAD_REQUEST,
                )
            return JSONResponse(
                {
                    "job_id": None,
                    "redirect_url": None,
                    "jobs_created": 0,
                    "info_message": unprocessable_assets_message(selected_count),
                }
            )

        submitter_email = current_user_email(request)
        created_jobs: list[Job] = []
        for group in groups:
            job = await _create_job_from_selection(
                selected_assets=group,
                asset_store=asset_store,
            )
            if isinstance(job, JSONResponse):
                return job
            job.submitted_by = submitter_email
            _jobs[job.id] = job
            await _queue.put(job)
            created_jobs.append(job)

        if len(created_jobs) == 1:
            return JSONResponse(
                {
                    "job_id": created_jobs[0].id,
                    "redirect_url": f"/jobs/{created_jobs[0].id}",
                }
            )
        return JSONResponse(
            {
                "job_id": created_jobs[0].id,
                "redirect_url": "/",
                "jobs_created": len(created_jobs),
            }
        )

    @app.post("/upload")
    async def upload(request: Request):
        auth_response = require_upload_access(request, current_auth_settings())
        if auth_response is not None:
            return auth_response
        files, client_metadata = await _parse_upload_form(request)
        if not files:
            return HTMLResponse("No file uploaded", status_code=HTTPStatus.BAD_REQUEST)
        selection_error = validate_upload_selection(files)
        if selection_error is not None:
            await log_rejected_upload_batch(request, files, reason=selection_error)
            return HTMLResponse(selection_error, status_code=HTTPStatus.BAD_REQUEST)

        try:
            inspected_assets = await _inspect_files(files, client_metadata, asset_store)
        except RejectedUploadError as exc:
            log_rejected_upload(
                request,
                reason=exc.reason,
                filename=exc.filename,
                size_bytes=exc.size_bytes,
                sha256=exc.sha256,
                content_type=exc.content_type,
            )
            return HTMLResponse(exc.reason, status_code=HTTPStatus.BAD_REQUEST)
        for asset in inspected_assets:
            if not asset_is_processable(asset):
                log_rejected_upload(
                    request,
                    reason=asset["processing_issue"],
                    filename=asset["original_filename"],
                    size_bytes=asset["size_bytes"],
                    sha256=asset["sha256"],
                    content_type=None,
                )
        batch = validate_asset_batch(inspected_assets)
        if not batch["valid"]:
            status_code = (
                HTTPStatus.OK if batch.get("message_level") == "info" else HTTPStatus.BAD_REQUEST
            )
            return HTMLResponse(batch["error"], status_code=status_code)

        selected_assets = [
            {
                "sha256": asset["sha256"],
                "original_filename": asset["original_filename"],
                "selected": True,
            }
            for asset in inspected_assets
        ]

        submitter_email = current_user_email(request)
        groups = _split_asset_groups(selected_assets, asset_store)
        if not groups:
            return HTMLResponse(
                batch.get("info") or "No processable assets selected.", status_code=HTTPStatus.OK
            )
        created_jobs: list[Job] = []
        for group in groups:
            job = await _create_job_from_selection(
                selected_assets=group,
                asset_store=asset_store,
            )
            if isinstance(job, JSONResponse):
                return HTMLResponse(job.body.decode(), status_code=job.status_code)
            job.submitted_by = submitter_email
            _jobs[job.id] = job
            await _queue.put(job)
            created_jobs.append(job)

        if len(created_jobs) == 1:
            return RedirectResponse(f"/jobs/{created_jobs[0].id}", status_code=303)
        return RedirectResponse("/", status_code=303)

    @app.post("/api/v1/videos")
    async def api_upload_video(
        request: Request,
        file: UploadFile = File(...),
        captured_at: str = Form(...),
        latitude: float | None = Form(None),
        longitude: float | None = Form(None),
        source: str | None = Form(None),
        source_event_id: str | None = Form(None),
        x_api_token: str | None = Header(None, alias="X-API-Token"),
    ):
        # ── Auth ──────────────────────────────────────────────────────────
        # Token-based auth, separate from the browser /upload Google OAuth
        # gate. If no tokens are configured the endpoint is disabled.
        if not api_tokens:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="API ingest is not configured (set webapp.api_tokens_file)",
            )
        if not x_api_token or x_api_token not in api_tokens:
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED,
                detail="Invalid or missing X-API-Token",
            )
        client_name = api_tokens[x_api_token]
        actor = f"{source or client_name}@api"

        # ── Validate captured_at ──────────────────────────────────────────
        try:
            video_date = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        except ValueError as err:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"captured_at must be ISO-8601, got: {captured_at!r}",
            ) from err

        # ── Read file and ingest into the content-addressed asset store ───
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Empty file")
        original_filename = Path(file.filename or "upload.mp4").name
        sha256 = hashlib.sha256(contents).hexdigest()

        extension_error = validate_upload_extension(original_filename)
        if extension_error is not None:
            log_rejected_upload(
                request,
                reason=extension_error,
                filename=original_filename,
                size_bytes=len(contents),
                sha256=sha256,
                content_type=file.content_type,
                actor=actor,
            )
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=extension_error)

        size_error = validate_upload_size(original_filename, len(contents))
        if size_error is not None:
            log_rejected_upload(
                request,
                reason=size_error,
                filename=original_filename,
                size_bytes=len(contents),
                sha256=sha256,
                content_type=file.content_type,
                actor=actor,
            )
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=size_error)

        if Path(original_filename).suffix.lower() not in VIDEO_EXTENSIONS:
            reason = "Expected a video upload with a supported video file extension."
            log_rejected_upload(
                request,
                reason=reason,
                filename=original_filename,
                size_bytes=len(contents),
                sha256=sha256,
                content_type=file.content_type,
                actor=actor,
            )
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=reason)
        inspected = await _run_off_event_loop(
            asset_store.inspect_bytes,
            original_filename=original_filename,
            content_type=file.content_type,
            data=contents,
            client_metadata=None,
        )
        if inspected.get("media_type") != "video":
            reason = f"Expected a video upload, got media_type={inspected.get('media_type')}"
            log_rejected_upload(
                request,
                reason=reason,
                filename=original_filename,
                size_bytes=len(contents),
                sha256=inspected["sha256"],
                content_type=file.content_type,
                actor=actor,
            )
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=reason,
            )
        if not asset_is_processable(inspected):
            log_rejected_upload(
                request,
                reason=inspected["processing_issue"],
                filename=original_filename,
                size_bytes=len(contents),
                sha256=inspected["sha256"],
                content_type=file.content_type,
                actor=actor,
            )
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=inspected["processing_issue"],
            )

        # ── Build a Job through the same path as the browser flow ────────
        selected_assets = [
            {
                "sha256": inspected["sha256"],
                "original_filename": inspected["original_filename"],
                "selected": True,
            }
        ]
        job_or_error = await _create_job_from_selection(
            selected_assets=selected_assets,
            asset_store=asset_store,
        )
        if isinstance(job_or_error, JSONResponse):
            return job_or_error
        job = job_or_error

        # Override metadata with what the API client told us. The clip is
        # likely a re-encoded MP4 with no exif/QuickTime tags, so the
        # asset_store's auto-extracted values are usually None.
        job.selected_date = video_date
        if latitude is not None or longitude is not None:
            job.video_meta = VideoMetadata(
                recorded_at=video_date,
                latitude=latitude,
                longitude=longitude,
            )
        elif job.video_meta is None:
            job.video_meta = VideoMetadata(recorded_at=video_date)
        else:
            job.video_meta.recorded_at = video_date

        # Surface the API client as the submitter so the existing
        # "Submitted by" UI works without special-casing API jobs.
        job.submitted_by = actor
        job.source_event_id = source_event_id

        _jobs[job.id] = job
        await _queue.put(job)
        logger.info(
            f"API video ingest from {client_name}: job={job.id} "
            f"sha256={inspected['sha256'][:12]} "
            f"event_id={source_event_id} captured_at={video_date.isoformat()}"
        )

        return JSONResponse(
            status_code=202,
            content={
                "job_id": job.id,
                "url": f"/jobs/{job.id}",
                "status": job.status,
            },
        )

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_detail_redirect(request: Request, job_id: str):
        job = _jobs.get(job_id)
        if not job:
            return HTMLResponse("Job not found", status_code=404)
        return RedirectResponse(url=f"/jobs/{job_id}/{job.slug}", status_code=301)

    @app.post("/jobs/{job_id}/reprocess")
    async def reprocess(request: Request, job_id: str):
        auth_response = require_upload_access(request, current_auth_settings())
        if auth_response is not None:
            return auth_response
        old_job = _jobs.get(job_id)
        if old_job is None:
            return HTMLResponse("Job not found", status_code=404)
        if old_job.status in ("pending", "running"):
            return RedirectResponse(f"/jobs/{job_id}", status_code=303)

        if old_job.assets:
            selected_assets = [
                {
                    "sha256": asset["sha256"],
                    "original_filename": asset["original_filename"],
                    "selected": True,
                }
                for asset in old_job.assets
            ]
            new_job = await _create_job_from_selection(
                selected_assets=selected_assets,
                asset_store=asset_store,
            )
            if isinstance(new_job, JSONResponse):
                return HTMLResponse(new_job.body.decode(), status_code=new_job.status_code)
            _jobs[new_job.id] = new_job
            await _queue.put(new_job)
            return RedirectResponse(f"/jobs/{new_job.id}", status_code=303)

        is_image_job = old_job.media_type == "images"
        if is_image_job:
            image_paths = old_job.image_paths
            if not image_paths:
                return HTMLResponse("Original image files not found", status_code=404)
            selected_assets = []
            for img_path in image_paths:
                img_file = Path(img_path)
                inspected = await _run_off_event_loop(
                    asset_store.ingest_path,
                    str(img_file),
                    old_job.filename if len(image_paths) == 1 else img_file.name,
                )
                selected_assets.append(
                    {
                        "sha256": inspected["sha256"],
                        "original_filename": inspected["original_filename"],
                        "selected": True,
                    }
                )
        else:
            first_path = None
            if old_job.result:
                candidate = Path(old_job.result["video"])
                if candidate.exists():
                    first_path = str(candidate)
            if first_path is None:
                return HTMLResponse("Original video file not found", status_code=404)
            inspected = await _run_off_event_loop(
                asset_store.ingest_path, first_path, old_job.filename
            )
            selected_assets = [
                {
                    "sha256": inspected["sha256"],
                    "original_filename": inspected["original_filename"],
                    "selected": True,
                }
            ]

        new_job = await _create_job_from_selection(
            selected_assets=selected_assets,
            asset_store=asset_store,
        )
        if isinstance(new_job, JSONResponse):
            return HTMLResponse(new_job.body.decode(), status_code=new_job.status_code)

        _jobs[new_job.id] = new_job
        await _queue.put(new_job)
        return RedirectResponse(f"/jobs/{new_job.id}", status_code=303)

    @app.get("/jobs/{job_id}/crops/{filename}")
    async def serve_crop(job_id: str, filename: str):
        job = _jobs.get(job_id)
        if job is None or job.result is None:
            return HTMLResponse("Not found", status_code=404)
        if job.result.get("type") == "images":
            crops_stem = job.id
        else:
            crops_stem = Path(job.result["video"]).stem
        crop_path = results_dir / f"{crops_stem}_crops" / filename
        if not crop_path.exists():
            return HTMLResponse("Not found", status_code=404)
        return FileResponse(crop_path, media_type="image/jpeg")

    @app.get("/jobs/{job_id}/video")
    async def serve_video(job_id: str):
        job = _jobs.get(job_id)
        if job is None or job.media_type == "images":
            return HTMLResponse("Not found", status_code=404)
        video_path = None
        if job.assets:
            video_path = job.assets[0].get("stored_path")
        elif job.result and job.result.get("video"):
            video_path = job.result["video"]
        if not video_path or not Path(video_path).exists():
            return HTMLResponse("Video file not found", status_code=404)
        return FileResponse(video_path)

    @app.get("/jobs/{job_id}/photo/{img_idx}")
    async def serve_photo(job_id: str, img_idx: int):
        job = _jobs.get(job_id)
        if job is None or job.media_type != "images":
            return HTMLResponse("Not found", status_code=404)
        if not job.image_paths or img_idx < 0 or img_idx >= len(job.image_paths):
            return HTMLResponse("Not found", status_code=404)
        photo_path = Path(job.image_paths[img_idx])
        if not photo_path.exists():
            return HTMLResponse("Photo not found", status_code=404)
        return FileResponse(photo_path)

    # This catch-all must be registered AFTER all specific /jobs/{job_id}/...
    # sub-routes, otherwise FastAPI matches it first.
    @app.get("/jobs/{job_id}/{slug}", response_class=HTMLResponse)
    async def job_detail(request: Request, job_id: str, slug: str):
        job = _jobs.get(job_id)
        if job is None:
            return HTMLResponse("Job not found", status_code=404)
        expected_slug = job.slug
        if slug != expected_slug:
            return RedirectResponse(url=f"/jobs/{job_id}/{expected_slug}", status_code=301)

        base = str(request.base_url).rstrip("/")
        og_url = f"{base}/jobs/{job_id}"

        if job.status == "done" and job.result:
            og_title = f"BirdVision: {job.summary}"
            # Build description from top species in the summary label
            species_label = job._species_summary_label()
            if species_label:
                og_description = f"Bird identification results: {species_label}"
            else:
                og_description = "Bird identification results from BirdVision."

            # Pick a representative annotated image
            og_image_url = None
            if job.media_type == "images":
                images = job.result.get("images") or []
                for img in images:
                    af = img.get("annotated_file")
                    if af:
                        og_image_url = f"{base}/jobs/{job_id}/crops/{af}"
                        break
            else:
                gallery = job.result.get("frame_gallery") or []
                if gallery:
                    first_file = gallery[0].get("file")
                    if first_file:
                        og_image_url = f"{base}/jobs/{job_id}/crops/{first_file}"
        else:
            status_label = job.status
            og_title = f"BirdVision \u2014 Job {status_label}"
            og_description = f"Job is {status_label}."
            og_image_url = None

        return render_template(
            request,
            "job.html",
            {
                "job": job,
                "og_title": og_title,
                "og_description": og_description,
                "og_url": og_url,
                "og_image_url": og_image_url,
            },
        )

    return app


def load_runtime_config(initial_config: dict[str, Any], config_path: Path | None) -> dict[str, Any]:
    if config_path and config_path.exists():
        try:
            return yaml.safe_load(config_path.read_text()) or {}
        except Exception as exc:
            logger.warning(f"Could not reload config from {config_path}: {exc}")
    return initial_config


def build_auth_settings(config: dict[str, Any]) -> AuthSettings:
    auth_config = config.get("auth", {})
    if not isinstance(auth_config, dict):
        auth_config = {}
    debug_mode = debug_mode_enabled(config)

    google_client_id = normalize_secret(os.getenv("GOOGLE_CLIENT_ID")) or normalize_secret(
        auth_config.get("google_client_id")
    )
    google_client_secret = normalize_secret(os.getenv("GOOGLE_CLIENT_SECRET")) or normalize_secret(
        auth_config.get("google_client_secret")
    )
    redirect_uri = normalize_secret(os.getenv("GOOGLE_REDIRECT_URI")) or normalize_secret(
        auth_config.get("redirect_uri")
    )
    session_secret = normalize_secret(os.getenv("SESSION_SECRET")) or normalize_secret(
        auth_config.get("session_secret")
    )

    raw_allowed_emails = auth_config.get("allowed_emails", [])
    if not isinstance(raw_allowed_emails, list):
        raw_allowed_emails = []
    allowed_emails = {
        email for email in (normalize_email(item) for item in raw_allowed_emails) if email
    }

    return AuthSettings(
        enabled=(not debug_mode)
        and bool(google_client_id and google_client_secret and session_secret),
        debug_mode=debug_mode,
        google_client_id=google_client_id,
        google_client_secret=google_client_secret,
        redirect_uri=redirect_uri,
        session_secret=session_secret,
        allowed_emails=allowed_emails,
    )


def debug_mode_enabled(config: dict[str, Any]) -> bool:
    env_value = os.getenv("BIRDVISION_DEBUG")
    if env_value is not None:
        return parse_bool(env_value)
    webapp_config = config.get("webapp", {})
    if not isinstance(webapp_config, dict):
        return False
    return parse_bool(webapp_config.get("debug", False))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "debug"}
    return bool(value)


def normalize_secret(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def normalize_email(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip().lower()
    return stripped or None


def current_user_email(request: Request) -> str | None:
    return normalize_email(request.session.get("email"))


def safe_current_user_email(request: Request) -> str:
    session = request.scope.get("session")
    if not isinstance(session, dict):
        return "-"
    return normalize_email(session.get("email")) or "-"


def log_rejected_upload(
    request: Request,
    *,
    reason: str,
    filename: str,
    size_bytes: int,
    sha256: str,
    content_type: str | None,
    actor: str | None = None,
) -> None:
    proxy_ip, real_ip, forwarded_for, x_real_ip = extract_request_ips(request)
    logger.warning(
        "rejected_upload reason=%s filename=%s sha256=%s size_bytes=%s content_type=%s proxy_ip=%s real_ip=%s x_forwarded_for=%s x_real_ip=%s email=%s",
        reason,
        filename,
        sha256,
        size_bytes,
        content_type or "-",
        proxy_ip,
        real_ip,
        forwarded_for,
        x_real_ip,
        actor or safe_current_user_email(request),
    )


async def log_rejected_upload_batch(
    request: Request,
    files: list[StarletteUploadFile],
    *,
    reason: str,
    actor: str | None = None,
) -> None:
    for upload in files:
        filename = Path(upload.filename or "upload").name
        contents = await upload.read()
        sha256 = hashlib.sha256(contents).hexdigest() if contents else "-"
        log_rejected_upload(
            request,
            reason=reason,
            filename=filename,
            size_bytes=len(contents),
            sha256=sha256,
            content_type=upload.content_type,
            actor=actor,
        )


def split_forwarded_for(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def extract_request_ips(request: Request) -> tuple[str, str, str, str]:
    proxy_ip = request.client.host if request.client else "-"
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    x_real_ip = request.headers.get("x-real-ip", "").strip()
    forwarded_chain = split_forwarded_for(forwarded_for)
    real_ip = x_real_ip or (forwarded_chain[0] if forwarded_chain else proxy_ip)
    return (
        proxy_ip,
        real_ip or "-",
        forwarded_for or "-",
        x_real_ip or "-",
    )


def can_upload_email(email: str | None, settings: AuthSettings) -> bool:
    if not settings.enabled:
        return True
    return bool(email and email in settings.allowed_emails)


@dataclass
class RejectedUploadError(Exception):
    reason: str
    filename: str
    size_bytes: int
    sha256: str
    content_type: str | None = None


def normalize_theme(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_THEME_IDS:
            return normalized
    return DEFAULT_THEME_ID


def current_theme(request: Request) -> str:
    return normalize_theme(request.cookies.get(THEME_COOKIE_NAME))


def current_path_with_query(request: Request) -> str:
    path = request.url.path or "/"
    if request.url.query:
        return f"{path}?{request.url.query}"
    return path


def build_template_theme_context(request: Request) -> dict[str, Any]:
    return {
        "active_theme": current_theme(request),
        "theme_options": THEME_OPTIONS,
        "theme_return_to": current_path_with_query(request),
        "theme_return_to_encoded": quote(current_path_with_query(request), safe="/"),
    }


def build_template_auth_context(request: Request, settings: AuthSettings) -> dict[str, Any]:
    email = current_user_email(request)
    return {
        "auth_enabled": settings.enabled,
        "user_email": email,
        "can_upload": can_upload_email(email, settings),
    }


def require_upload_access(
    request: Request, settings: AuthSettings
) -> JSONResponse | RedirectResponse | None:
    if not settings.enabled:
        return None

    email = current_user_email(request)
    is_api = request.url.path.startswith("/api/")

    if not email:
        if is_api:
            return JSONResponse({"error": "Authentication required."}, status_code=401)
        return RedirectResponse("/login", status_code=303)

    if email not in settings.allowed_emails:
        if is_api:
            return JSONResponse(
                {"error": "Signed-in user is not authorized to upload."}, status_code=403
            )
        return RedirectResponse("/", status_code=303)

    return None


def build_google_oauth_client(settings: AuthSettings):
    if OAuth is None:
        raise RuntimeError("Auth dependencies are not installed. Run `uv sync` to install authlib.")
    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url=GOOGLE_SERVER_METADATA_URL,
        client_kwargs={"scope": "openid email profile"},
    )
    return oauth.create_client("google")


def _load_api_tokens(path: Path) -> dict[str, str]:
    """Load API tokens from a YAML file. Returns ``{token: client_name}``.

    File format::

        tokens:
          - name: birdcamgrabber
            token: <random-hex>

    Returns an empty dict if the file is missing, unreadable, or empty.
    """
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        logger.warning(f"Could not parse api_tokens_file {path}: {exc}")
        return {}
    out: dict[str, str] = {}
    entries = data.get("tokens") or []
    if not isinstance(entries, list):
        return {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        token = entry.get("token")
        name = entry.get("name") or "unknown"
        if isinstance(token, str) and token:
            out[token] = name
    return out


def _load_existing_jobs(results_dir: Path):
    pattern = re.compile(r"^([0-9a-f]{32})_(.+)_results\.json$")
    legacy_pattern = re.compile(r"^([0-9a-f]{32})_results\.json$")
    loaded = 0
    for result_file in sorted(results_dir.glob("*_results.json")):
        m = pattern.match(result_file.name)
        if m:
            job_id = m.group(1)
        else:
            legacy = legacy_pattern.match(result_file.name)
            if not legacy:
                continue
            job_id = legacy.group(1)
        if job_id in _jobs:
            continue
        try:
            result = json.loads(result_file.read_text())
            is_image = result.get("type") == "images"
            if result.get("display_name"):
                original_filename = result["display_name"]
            elif is_image:
                info = result.get("image_info", {})
                count = info.get("count", 0)
                names = info.get("filenames", [])
                original_filename = (
                    f"{count} photos" if count > 1 else (names[0] if names else "photos")
                )
            else:
                original_filename = (
                    result.get("source_filename") or Path(result.get("video", "")).name
                )
                if original_filename.startswith(job_id + "_"):
                    original_filename = original_filename[len(job_id) + 1 :]

            media_type = "images" if is_image else "video"
            job = Job(id=job_id, filename=original_filename, media_type=media_type)
            job.status = "done"
            job.result = result
            job.created_at = datetime.fromtimestamp(result_file.stat().st_mtime, tz=UTC)
            job.assets = result.get("asset_records", [])
            if is_image:
                job.image_paths = [
                    asset["stored_path"] for asset in job.assets if asset.get("stored_path")
                ]
            if result.get("date"):
                with contextlib.suppress(ValueError):
                    job.selected_date = datetime.fromisoformat(result["date"])
            job.submitted_by = result.get("submitted_by")
            job.source_event_id = result.get("source_event_id")
            if job.assets:
                job.video_meta = build_video_meta_from_asset(job.assets[0])
            elif result.get("latitude") is not None and result.get("longitude") is not None:
                vm = VideoMetadata(
                    latitude=result["latitude"],
                    longitude=result["longitude"],
                )
                if result.get("date"):
                    with contextlib.suppress(ValueError):
                        vm.recorded_at = datetime.fromisoformat(result["date"])
                job.video_meta = vm
            _jobs[job_id] = job
            loaded += 1
        except Exception as e:
            logger.warning(f"Could not load result {result_file.name}: {e}")
    if loaded:
        logger.info(f"Restored {loaded} completed job(s) from disk.")


def _init_pipeline(config: dict) -> BirdIdentificationPipeline:
    p = BirdIdentificationPipeline(config)
    species_file = config.get("species", {}).get("list_file")
    p.load_species(species_file)
    return p


def _persist_submitted_by(results_dir: Path, result_stem: str, submitted_by: str) -> None:
    """Append submitted_by to the already-written results JSON file."""
    json_path = results_dir / f"{result_stem}_results.json"
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text())
        data["submitted_by"] = submitted_by
        json_path.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        logger.warning(f"Could not persist submitted_by to {json_path}: {exc}")


def _persist_source_event_id(results_dir: Path, result_stem: str, source_event_id: str) -> None:
    """Append source_event_id to the already-written results JSON file."""
    json_path = results_dir / f"{result_stem}_results.json"
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text())
        data["source_event_id"] = source_event_id
        json_path.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        logger.warning(f"Could not persist source_event_id to {json_path}: {exc}")


async def _worker(
    loop: asyncio.AbstractEventLoop,
    pipeline: BirdIdentificationPipeline,
    config_path: Path | None = None,
    results_dir: Path | None = None,
    asset_store: Optional["AssetStore"] = None,
):
    while True:
        job = await _queue.get()
        job.status = "running"
        logger.info(f"Processing job {job.id}: {job.filename}")
        try:
            fresh_config: dict = {}
            if config_path and config_path.exists():
                fresh_config = yaml.safe_load(config_path.read_text()) or {}
                await loop.run_in_executor(_executor, lambda: pipeline.apply_config(fresh_config))
                logger.info("Config reloaded.")

            if job.media_type == "images":
                result = await loop.run_in_executor(
                    _executor,
                    lambda: pipeline.process_images(
                        job.image_paths,
                        source_filenames=[asset["original_filename"] for asset in job.assets],
                        video_date=job.selected_date,
                        latitude=job.video_meta.latitude if job.video_meta else None,
                        longitude=job.video_meta.longitude if job.video_meta else None,
                        job_id=job.id,
                        result_stem=job.result_stem,
                        display_name=job.filename,
                        asset_records=job.assets,
                    ),
                )
            else:
                video_asset = job.assets[0]
                transcode_enabled = fresh_config.get("webapp", {}).get(
                    "transcode_incompatible_video", True
                )
                codec = video_asset.get("video_codec")
                if transcode_enabled and codec and codec not in SAFARI_COMPATIBLE_CODECS:
                    logger.info(
                        f"Job {job.id}: codec {codec!r} is not Safari-compatible, transcoding to H.264"
                    )
                    transcoded = await loop.run_in_executor(
                        _executor,
                        lambda: transcode_to_h264(Path(video_asset["stored_path"])),
                    )
                    if transcoded:
                        video_asset["stored_path"] = str(transcoded)
                        video_asset["video_codec"] = "avc1"
                        if asset_store:
                            asset_store.update_transcoded_path(
                                video_asset["sha256"], str(transcoded), "avc1"
                            )
                    else:
                        logger.warning(
                            f"Job {job.id}: transcode failed, proceeding with original {codec!r} file"
                        )
                result = await loop.run_in_executor(
                    _executor,
                    lambda: pipeline.process_video(
                        video_asset["stored_path"],
                        video_date=job.selected_date,
                        latitude=job.video_meta.latitude if job.video_meta else None,
                        longitude=job.video_meta.longitude if job.video_meta else None,
                        result_stem=job.result_stem,
                        source_filename=video_asset["original_filename"],
                        display_name=job.filename,
                        asset_records=job.assets,
                    ),
                )
            # Inject submitted_by into result dict and persist it in the JSON file
            if job.submitted_by:
                result["submitted_by"] = job.submitted_by
                if results_dir and job.result_stem:
                    _persist_submitted_by(results_dir, job.result_stem, job.submitted_by)
            if job.source_event_id:
                result["source_event_id"] = job.source_event_id
                if results_dir and job.result_stem:
                    _persist_source_event_id(results_dir, job.result_stem, job.source_event_id)
            job.result = result
            job.status = "done"
            logger.info(f"Job {job.id} done.")
        except Exception as exc:
            job.error = str(exc)
            job.status = "error"
            logger.exception(f"Job {job.id} failed: {exc}")
        finally:
            _queue.task_done()


async def _parse_upload_form(
    request: Request,
) -> tuple[list[StarletteUploadFile], list[dict[str, Any]]]:
    form = await request.form()
    files = form.getlist("file")
    valid_files = [f for f in files if isinstance(f, StarletteUploadFile)]
    client_metadata_raw = form.get("client_metadata")
    client_metadata: list[dict[str, Any]] = []
    if isinstance(client_metadata_raw, str) and client_metadata_raw:
        try:
            parsed = json.loads(client_metadata_raw)
            if isinstance(parsed, list):
                client_metadata = [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            logger.warning("Invalid client metadata payload")
    return valid_files, client_metadata


async def _inspect_files(
    files: list[StarletteUploadFile],
    client_metadata: list[dict[str, Any]],
    asset_store: AssetStore,
) -> list[dict[str, Any]]:
    inspected_assets = []
    for idx, upload in enumerate(files):
        client_meta = client_metadata[idx] if idx < len(client_metadata) else None
        contents = await upload.read()
        filename = upload.filename or "upload"
        sha256 = hashlib.sha256(contents).hexdigest() if contents else "-"
        extension_error = validate_upload_extension(filename)
        if extension_error is not None:
            raise RejectedUploadError(
                reason=extension_error,
                filename=Path(filename).name,
                size_bytes=len(contents),
                sha256=sha256,
                content_type=upload.content_type,
            )
        size_error = validate_upload_size(filename, len(contents))
        if size_error is not None:
            raise RejectedUploadError(
                reason=size_error,
                filename=Path(filename).name,
                size_bytes=len(contents),
                sha256=sha256,
                content_type=upload.content_type,
            )
        inspected_assets.append(
            await _run_off_event_loop(
                asset_store.inspect_bytes,
                original_filename=filename,
                content_type=upload.content_type,
                data=contents,
                client_metadata=client_meta,
            )
        )
    return inspected_assets


async def _create_job_from_selection(
    *,
    selected_assets: list[dict[str, Any]],
    asset_store: AssetStore,
) -> Job | JSONResponse:
    chosen = [asset for asset in selected_assets if asset.get("selected", True)]
    if not chosen:
        return JSONResponse({"error": "Select at least one asset to process."}, status_code=400)

    resolved_assets = []
    for selection in chosen:
        sha256 = selection.get("sha256")
        if not isinstance(sha256, str):
            return JSONResponse({"error": "Invalid asset selection."}, status_code=400)
        indexed = asset_store.get(sha256)
        if indexed is None:
            return JSONResponse(
                {"error": f"Asset {sha256} is no longer available."}, status_code=404
            )
        resolved_assets.append(build_job_asset(indexed, selection.get("original_filename")))

    processable_assets = [asset for asset in resolved_assets if asset_is_processable(asset)]
    batch = validate_asset_batch(resolved_assets)
    if not batch["valid"]:
        return JSONResponse({"error": batch["error"]}, status_code=400)

    media_type = "images" if batch["media_type"] == "image" else "video"
    job_id = uuid.uuid4().hex
    display_name = build_job_display_name(processable_assets, media_type)
    job = Job(job_id, display_name, media_type)
    job.assets = processable_assets
    job.image_paths = [
        asset["stored_path"] for asset in processable_assets if asset["media_type"] == "image"
    ]
    job.video_meta = build_video_meta_from_asset(processable_assets[0])
    job.selected_date = (
        datetime.fromisoformat(processable_assets[0]["recorded_at"])
        if processable_assets[0].get("recorded_at")
        else None
    )
    job.result_stem = (
        f"{job_id}_{slugify_result_name(result_name_seed(processable_assets, media_type))}"
    )
    return job


def validate_asset_batch(assets: list[dict[str, Any]]) -> dict[str, Any]:
    processable_assets = [asset for asset in assets if asset_is_processable(asset)]
    media_types = {
        asset.get("media_type")
        for asset in processable_assets
        if asset.get("media_type") in {"image", "video"}
    }
    unprocessable_count = len(assets) - len(processable_assets)
    if not assets:
        return {"valid": False, "error": "No assets selected.", "media_type": None}
    if not processable_assets:
        return {
            "valid": False,
            "error": unprocessable_assets_message(unprocessable_count),
            "media_type": None,
            "message_level": "info",
        }
    if not media_types:
        return {
            "valid": False,
            "error": "BirdVision only supports common image and video uploads.",
            "media_type": None,
        }
    if len(media_types) > 1:
        result = {
            "valid": True,
            "error": None,
            "media_type": "mixed",
        }
        if unprocessable_count:
            result["info"] = unprocessable_assets_message(unprocessable_count)
        return result
    media_type = next(iter(media_types))
    result = {"valid": True, "error": None, "media_type": media_type}
    if unprocessable_count:
        result["info"] = unprocessable_assets_message(unprocessable_count)
    return result


def _split_asset_groups(
    selected_assets: list[dict[str, Any]],
    asset_store: AssetStore,
) -> list[list[dict[str, Any]]]:
    """Split selected assets into job groups: one job per video, all images in one job."""
    videos = []
    images = []
    for asset in selected_assets:
        if not asset.get("selected", True):
            continue
        indexed = asset_store.get(asset.get("sha256", ""))
        if not indexed or not asset_is_processable(indexed):
            continue
        if indexed and indexed.get("media_type") == "video":
            videos.append(asset)
        else:
            images.append(asset)

    groups: list[list[dict[str, Any]]] = []
    for video in videos:
        groups.append([video])
    if images:
        groups.append(images)
    return groups


def classify_media_type(filename: str, content_type: str | None) -> str:
    ext = Path(filename).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if content_type:
        if content_type.startswith("video/"):
            return "video"
        if content_type.startswith("image/"):
            return "image"
    return "unknown"


def build_job_asset(indexed: dict[str, Any], original_filename: str | None) -> dict[str, Any]:
    return {
        "sha256": indexed["sha256"],
        "stored_path": indexed["stored_path"],
        "media_type": indexed["media_type"],
        "original_filename": Path(original_filename or indexed["original_names"][0]).name,
        "size_bytes": indexed["size_bytes"],
        "width": indexed.get("width"),
        "height": indexed.get("height"),
        "duration_s": indexed.get("duration_s"),
        "fps": indexed.get("fps"),
        "recorded_at": indexed.get("recorded_at"),
        "latitude": indexed.get("latitude"),
        "longitude": indexed.get("longitude"),
        "video_codec": indexed.get("video_codec"),
        "processable": indexed.get("processable", True),
        "processing_issue": indexed.get("processing_issue"),
        "resolution_warning": resolution_warning_text(
            media_type=indexed["media_type"],
            width=indexed.get("width"),
            height=indexed.get("height"),
        ),
    }


def build_job_display_name(assets: list[dict[str, Any]], media_type: str) -> str:
    if media_type == "images":
        return assets[0]["original_filename"] if len(assets) == 1 else f"{len(assets)} photos"
    return assets[0]["original_filename"]


def build_video_meta_from_asset(asset: dict[str, Any]) -> VideoMetadata:
    meta = VideoMetadata(
        latitude=asset.get("latitude"),
        longitude=asset.get("longitude"),
    )
    recorded_at = asset.get("recorded_at")
    if recorded_at:
        with contextlib.suppress(ValueError):
            meta.recorded_at = datetime.fromisoformat(recorded_at)
    # camera_info is stored pre-formatted in the asset index; parse it
    # back into make/model so VideoMetadata.camera_info property works
    camera_info = asset.get("camera_info")
    if camera_info:
        meta.camera_model = camera_info
    return meta


def slugify_result_name(name: str) -> str:
    stem = Path(name).stem or name or "upload"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return slug[:80] or "upload"


def result_name_seed(assets: list[dict[str, Any]], media_type: str) -> str:
    if media_type == "images":
        return assets[0]["original_filename"] if len(assets) == 1 else "photos"
    return assets[0]["original_filename"]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def first_value(server_value: Any, client_metadata: dict[str, Any] | None, key: str) -> Any:
    if server_value is not None:
        return server_value
    if client_metadata and client_metadata.get(key) is not None:
        return client_metadata[key]
    return None


def asset_is_processable(asset: dict[str, Any]) -> bool:
    return asset.get("processable", True)


def asset_processing_issue(media_type: str, inspected: MediaMetadata) -> str | None:
    if media_type != "video":
        return None
    if any(
        value is not None
        for value in (
            inspected.width,
            inspected.height,
            inspected.duration_s,
            inspected.fps,
            inspected.video_codec,
        )
    ):
        return None
    return (
        "This video file could not be opened for processing. "
        "It may be corrupt, incomplete, or unsupported."
    )


def unprocessable_assets_message(count: int) -> str:
    noun = "file" if count == 1 else "files"
    return (
        f"{count} selected {noun} could not be opened for processing. "
        "They may be corrupt, incomplete, or unsupported."
    )


def validate_upload_selection(files: list[StarletteUploadFile]) -> str | None:
    image_count = sum(
        1 for upload in files if Path(upload.filename or "").suffix.lower() in IMAGE_EXTENSIONS
    )
    if image_count > MAX_IMAGE_UPLOAD_COUNT:
        return f"Select at most {MAX_IMAGE_UPLOAD_COUNT} photos per upload."
    return None


def validate_upload_size(filename: str, size_bytes: int) -> str | None:
    if size_bytes <= MAX_UPLOAD_FILE_BYTES:
        return None
    return f"{Path(filename).name} exceeds the 50 MB upload limit."


def validate_upload_extension(filename: str) -> str | None:
    if Path(filename).suffix.lower() in KNOWN_UPLOAD_EXTENSIONS:
        return None
    return (
        f"{Path(filename).name} has an unsupported file extension. "
        "Upload a common photo or video format."
    )


def resolution_warning_text(
    *,
    media_type: str | None,
    width: int | None,
    height: int | None,
) -> str | None:
    if width is None or height is None:
        return None
    long_edge = max(width, height)
    short_edge = min(width, height)
    if media_type == "video" and (long_edge < 1280 or short_edge < 720):
        return "Low-resolution video can reduce bird detection recall, especially for small or distant birds."
    if media_type == "image" and (long_edge < 1600 or short_edge < 900):
        return "Low-resolution photos can reduce bird detection recall, especially for small or distant birds."
    return None
