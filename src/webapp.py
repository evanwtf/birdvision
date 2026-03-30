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
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.datastructures import UploadFile as StarletteUploadFile

try:
    from authlib.integrations.starlette_client import OAuth
except ImportError:  # pragma: no cover - keeps app importable until deps are installed
    OAuth = None

from .pipeline import BirdIdentificationPipeline
from .video_metadata import VideoMetadata, inspect_media

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
GOOGLE_SERVER_METADATA_URL = "https://accounts.google.com/.well-known/openid-configuration"


@dataclass
class Job:
    id: str
    filename: str
    media_type: str = "video"

    def __post_init__(self):
        self.status = "pending"   # pending | running | done | error
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.video_meta: Optional[VideoMetadata] = None
        self.image_paths: list[str] = []
        self.assets: list[dict[str, Any]] = []
        self.created_at: datetime = datetime.now()
        self.selected_date: Optional[datetime] = None
        self.result_stem: Optional[str] = None

    @property
    def summary(self) -> str:
        label = self._species_summary_label()
        if label:
            return label

        if self.media_type == "images":
            n = len(self.assets) or len(self.image_paths)
            if n == 0 and self.result and self.result.get("image_info"):
                n = self.result["image_info"].get("count", 0)
            return f"{n} photo{'s' if n != 1 else ''}"
        return f"1 video ({self.filename})"

    def _species_summary_label(self) -> Optional[str]:
        """Build a label like '2024-02-03: Mourning Dove, Blue Jay, 2 others'."""
        if not self.result or self.status != "done":
            return None

        # Collect top species names
        top_species: list[str] = []
        if self.result.get("type") == "images":
            seen: set[str] = set()
            for img in self.result.get("images", []):
                for pred in img.get("species_summary", []):
                    sp = pred.get("species", "")
                    prob = pred.get("probability", 0)
                    if sp and prob >= 0.15 and sp not in seen:
                        seen.add(sp)
                        top_species.append(sp)
        else:
            for pred in self.result.get("video_predictions", []):
                sp = pred.get("species", "")
                prob = pred.get("presence_probability", 0)
                if sp and prob >= 0.15:
                    top_species.append(sp)

        if not top_species:
            return None

        # Build the species part: show up to 2, then "N others"
        if len(top_species) <= 2:
            species_text = ", ".join(top_species)
        else:
            species_text = f"{top_species[0]}, {top_species[1]}, {len(top_species) - 2} other{'s' if len(top_species) - 2 != 1 else ''}"

        # Prefix with date if available
        date_str = self.result.get("date")
        if date_str:
            date_prefix = date_str[:10]
            return f"{date_prefix}: {species_text}"
        return species_text


@dataclass(frozen=True)
class AuthSettings:
    enabled: bool
    debug_mode: bool
    google_client_id: Optional[str]
    google_client_secret: Optional[str]
    session_secret: Optional[str]
    allowed_emails: set[str]


class AssetStore:
    def __init__(self, upload_dir: Path):
        self.root = upload_dir
        self.asset_dir = upload_dir / "assets"
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = upload_dir / "asset_index.json"
        self._assets = self._load_index()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.exists():
            return {}
        try:
            payload = json.loads(self.index_path.read_text())
            assets = payload.get("assets", {})
            if isinstance(assets, dict):
                return assets
        except Exception as exc:
            logger.warning(f"Could not load asset index {self.index_path}: {exc}")
        return {}

    def _save_index(self):
        payload = {
            "version": 1,
            "assets": self._assets,
        }
        self.index_path.write_text(json.dumps(payload, indent=2))

    def get(self, sha256: str) -> Optional[dict[str, Any]]:
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
        content_type: Optional[str],
        data: bytes,
        client_metadata: Optional[dict[str, Any]] = None,
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
            "original_names": sorted({
                safe_name,
                *(existing.get("original_names", []) if existing else []),
            }),
            "created_at": (existing or {}).get("created_at") or utc_now_iso(),
            "last_seen_at": utc_now_iso(),
        }
        self._assets[sha256] = record
        self._save_index()

        return {
            "sha256": sha256,
            "original_filename": safe_name,
            "stored_path": str(stored_path),
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
            "resolution_warning": resolution_warning_text(
                media_type=media_type,
                width=record["width"],
                height=record["height"],
            ),
            "duplicate": duplicate,
            "duplicate_status": "existing" if duplicate else "new",
            "canonical_path": str(stored_path),
        }

    def ingest_path(self, path: str, original_filename: Optional[str] = None) -> dict[str, Any]:
        source = Path(path)
        return self.inspect_bytes(
            original_filename=original_filename or source.name,
            content_type=None,
            data=source.read_bytes(),
            client_metadata=None,
        )


_jobs: dict[str, Job] = {}
_queue: asyncio.Queue
_executor = ThreadPoolExecutor(max_workers=1)


def create_app(config: dict, templates_dir: str = "templates", config_path: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="BirdVision")
    templates = Jinja2Templates(directory=templates_dir)
    initial_auth_settings = build_auth_settings(config)
    if initial_auth_settings.debug_mode:
        logger.warning("Webapp debug mode is enabled; auth gating is disabled.")
    elif not initial_auth_settings.enabled and any((
        initial_auth_settings.google_client_id,
        initial_auth_settings.google_client_secret,
        initial_auth_settings.session_secret,
    )):
        logger.warning("Auth configuration is incomplete; authentication routes and upload gating are disabled.")

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

    pipeline: Optional[BirdIdentificationPipeline] = None

    def current_config() -> dict[str, Any]:
        return load_runtime_config(config, config_path)

    def current_auth_settings() -> AuthSettings:
        return build_auth_settings(current_config())

    def render_template(request: Request, template_name: str, context: dict[str, Any]) -> HTMLResponse:
        merged = {
            **context,
            **build_template_auth_context(request, current_auth_settings()),
        }
        return templates.TemplateResponse(request, template_name, merged)

    @app.on_event("startup")
    async def startup():
        nonlocal pipeline
        global _queue
        _queue = asyncio.Queue()

        _load_existing_jobs(results_dir)

        logger.info("Loading pipeline (models may download on first run)…")
        loop = asyncio.get_event_loop()
        pipeline = await loop.run_in_executor(
            _executor, lambda: _init_pipeline(config)
        )
        logger.info("Pipeline ready.")
        asyncio.create_task(_worker(loop, pipeline, config_path))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        recent = list(reversed(list(_jobs.values())))[:20]
        return render_template(request, "index.html", {
            "jobs": recent,
        })

    @app.get("/login")
    async def login(request: Request):
        settings = current_auth_settings()
        if not settings.enabled:
            return RedirectResponse("/", status_code=303)
        try:
            google = build_google_oauth_client(settings)
        except RuntimeError as exc:
            return HTMLResponse(str(exc), status_code=500)
        redirect_uri = str(request.url_for("auth_callback"))
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

    @app.post("/api/uploads/inspect")
    async def inspect_upload_candidates(request: Request):
        auth_response = require_upload_access(request, current_auth_settings())
        if auth_response is not None:
            return auth_response
        files, client_metadata = await _parse_upload_form(request)
        if not files:
            return JSONResponse({"error": "No file uploaded"}, status_code=400)

        inspected_assets = await _inspect_files(files, client_metadata, asset_store)
        batch = validate_asset_batch(inspected_assets)
        return JSONResponse({
            "assets": inspected_assets,
            "batch": batch,
        })

    @app.post("/api/uploads/finalize")
    async def finalize_upload(request: Request):
        auth_response = require_upload_access(request, current_auth_settings())
        if auth_response is not None:
            return auth_response
        payload = await request.json()
        selected_assets = payload.get("assets") or []
        if not isinstance(selected_assets, list):
            return JSONResponse({"error": "Invalid asset selection"}, status_code=400)

        # Split multiple videos into one job per video
        groups = _split_asset_groups(selected_assets, asset_store)

        created_jobs: list[Job] = []
        for group in groups:
            job = await _create_job_from_selection(
                selected_assets=group,
                asset_store=asset_store,
            )
            if isinstance(job, JSONResponse):
                return job
            _jobs[job.id] = job
            await _queue.put(job)
            created_jobs.append(job)

        if len(created_jobs) == 1:
            return JSONResponse({
                "job_id": created_jobs[0].id,
                "redirect_url": f"/jobs/{created_jobs[0].id}",
            })
        return JSONResponse({
            "job_id": created_jobs[0].id,
            "redirect_url": "/",
            "jobs_created": len(created_jobs),
        })

    @app.post("/upload")
    async def upload(request: Request):
        auth_response = require_upload_access(request, current_auth_settings())
        if auth_response is not None:
            return auth_response
        files, client_metadata = await _parse_upload_form(request)
        if not files:
            return HTMLResponse("No file uploaded", status_code=400)

        inspected_assets = await _inspect_files(files, client_metadata, asset_store)
        batch = validate_asset_batch(inspected_assets)
        if not batch["valid"]:
            return HTMLResponse(batch["error"], status_code=400)

        selected_assets = [
            {
                "sha256": asset["sha256"],
                "original_filename": asset["original_filename"],
                "selected": True,
            }
            for asset in inspected_assets
        ]

        groups = _split_asset_groups(selected_assets, asset_store)
        created_jobs: list[Job] = []
        for group in groups:
            job = await _create_job_from_selection(
                selected_assets=group,
                asset_store=asset_store,
            )
            if isinstance(job, JSONResponse):
                return HTMLResponse(job.body.decode(), status_code=job.status_code)
            _jobs[job.id] = job
            await _queue.put(job)
            created_jobs.append(job)

        if len(created_jobs) == 1:
            return RedirectResponse(f"/jobs/{created_jobs[0].id}", status_code=303)
        return RedirectResponse("/", status_code=303)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_detail(request: Request, job_id: str):
        job = _jobs.get(job_id)
        if job is None:
            return HTMLResponse("Job not found", status_code=404)
        return render_template(request, "job.html", {
            "job": job,
        })

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
                inspected = asset_store.ingest_path(str(img_file), old_job.filename if len(image_paths) == 1 else img_file.name)
                selected_assets.append({
                    "sha256": inspected["sha256"],
                    "original_filename": inspected["original_filename"],
                    "selected": True,
                })
        else:
            first_path = None
            if old_job.result:
                candidate = Path(old_job.result["video"])
                if candidate.exists():
                    first_path = str(candidate)
            if first_path is None:
                return HTMLResponse("Original video file not found", status_code=404)
            inspected = asset_store.ingest_path(first_path, old_job.filename)
            selected_assets = [{
                "sha256": inspected["sha256"],
                "original_filename": inspected["original_filename"],
                "selected": True,
            }]

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

    return app


def load_runtime_config(initial_config: dict[str, Any], config_path: Optional[Path]) -> dict[str, Any]:
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

    google_client_id = normalize_secret(os.getenv("GOOGLE_CLIENT_ID")) or normalize_secret(auth_config.get("google_client_id"))
    google_client_secret = normalize_secret(os.getenv("GOOGLE_CLIENT_SECRET")) or normalize_secret(auth_config.get("google_client_secret"))
    session_secret = normalize_secret(os.getenv("SESSION_SECRET")) or normalize_secret(auth_config.get("session_secret"))

    raw_allowed_emails = auth_config.get("allowed_emails", [])
    if not isinstance(raw_allowed_emails, list):
        raw_allowed_emails = []
    allowed_emails = {
        email for email in (normalize_email(item) for item in raw_allowed_emails)
        if email
    }

    return AuthSettings(
        enabled=(not debug_mode) and bool(google_client_id and google_client_secret and session_secret),
        debug_mode=debug_mode,
        google_client_id=google_client_id,
        google_client_secret=google_client_secret,
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


def normalize_secret(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def normalize_email(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip().lower()
    return stripped or None


def current_user_email(request: Request) -> Optional[str]:
    return normalize_email(request.session.get("email"))


def can_upload_email(email: Optional[str], settings: AuthSettings) -> bool:
    if not settings.enabled:
        return True
    return bool(email and email in settings.allowed_emails)


def build_template_auth_context(request: Request, settings: AuthSettings) -> dict[str, Any]:
    email = current_user_email(request)
    return {
        "auth_enabled": settings.enabled,
        "user_email": email,
        "can_upload": can_upload_email(email, settings),
    }


def require_upload_access(request: Request, settings: AuthSettings) -> Optional[JSONResponse | RedirectResponse]:
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
            return JSONResponse({"error": "Signed-in user is not authorized to upload."}, status_code=403)
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


def _load_existing_jobs(results_dir: Path):
    pattern = re.compile(r'^([0-9a-f]{32})_(.+)_results\.json$')
    legacy_pattern = re.compile(r'^([0-9a-f]{32})_results\.json$')
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
                original_filename = f"{count} photos" if count > 1 else (names[0] if names else "photos")
            else:
                original_filename = result.get("source_filename") or Path(result.get("video", "")).name
                if original_filename.startswith(job_id + "_"):
                    original_filename = original_filename[len(job_id) + 1:]

            media_type = "images" if is_image else "video"
            job = Job(id=job_id, filename=original_filename, media_type=media_type)
            job.status = "done"
            job.result = result
            job.created_at = datetime.fromtimestamp(result_file.stat().st_mtime)
            job.assets = result.get("asset_records", [])
            if is_image:
                job.image_paths = [asset["stored_path"] for asset in job.assets if asset.get("stored_path")]
            if result.get("date"):
                try:
                    job.selected_date = datetime.fromisoformat(result["date"])
                except ValueError:
                    pass
            if job.assets:
                job.video_meta = build_video_meta_from_asset(job.assets[0])
            elif result.get("latitude") is not None and result.get("longitude") is not None:
                vm = VideoMetadata(
                    latitude=result["latitude"],
                    longitude=result["longitude"],
                )
                if result.get("date"):
                    try:
                        vm.recorded_at = datetime.fromisoformat(result["date"])
                    except ValueError:
                        pass
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


async def _worker(loop: asyncio.AbstractEventLoop, pipeline: BirdIdentificationPipeline, config_path: Optional[Path] = None):
    while True:
        job = await _queue.get()
        job.status = "running"
        logger.info(f"Processing job {job.id}: {job.filename}")
        try:
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
            job.result = result
            job.status = "done"
            logger.info(f"Job {job.id} done.")
        except Exception as exc:
            job.error = str(exc)
            job.status = "error"
            logger.exception(f"Job {job.id} failed: {exc}")
        finally:
            _queue.task_done()


async def _parse_upload_form(request: Request) -> tuple[list[StarletteUploadFile], list[dict[str, Any]]]:
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
        inspected_assets.append(asset_store.inspect_bytes(
            original_filename=upload.filename or "upload",
            content_type=upload.content_type,
            data=contents,
            client_metadata=client_meta,
        ))
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
            return JSONResponse({"error": f"Asset {sha256} is no longer available."}, status_code=404)
        resolved_assets.append(build_job_asset(indexed, selection.get("original_filename")))

    batch = validate_asset_batch(resolved_assets)
    if not batch["valid"]:
        return JSONResponse({"error": batch["error"]}, status_code=400)

    media_type = "images" if batch["media_type"] == "image" else "video"
    job_id = uuid.uuid4().hex
    display_name = build_job_display_name(resolved_assets, media_type)
    job = Job(job_id, display_name, media_type)
    job.assets = resolved_assets
    job.image_paths = [asset["stored_path"] for asset in resolved_assets if asset["media_type"] == "image"]
    job.video_meta = build_video_meta_from_asset(resolved_assets[0])
    job.selected_date = (
        datetime.fromisoformat(resolved_assets[0]["recorded_at"])
        if resolved_assets[0].get("recorded_at") else None
    )
    job.result_stem = f"{job_id}_{slugify_result_name(result_name_seed(resolved_assets, media_type))}"
    return job


def validate_asset_batch(assets: list[dict[str, Any]]) -> dict[str, Any]:
    media_types = {asset.get("media_type") for asset in assets if asset.get("media_type") in {"image", "video"}}
    if not assets:
        return {"valid": False, "error": "No assets selected.", "media_type": None}
    if not media_types:
        return {"valid": False, "error": "BirdVision only supports common image and video uploads.", "media_type": None}
    if len(media_types) > 1:
        return {
            "valid": False,
            "error": "Please upload either videos or photos, not both at once.",
            "media_type": None,
        }
    media_type = next(iter(media_types))
    return {"valid": True, "error": None, "media_type": media_type}


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


def classify_media_type(filename: str, content_type: Optional[str]) -> str:
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


def build_job_asset(indexed: dict[str, Any], original_filename: Optional[str]) -> dict[str, Any]:
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
        try:
            meta.recorded_at = datetime.fromisoformat(recorded_at)
        except ValueError:
            pass
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
    return datetime.now(timezone.utc).isoformat()


def first_value(server_value: Any, client_metadata: Optional[dict[str, Any]], key: str) -> Any:
    if server_value is not None:
        return server_value
    if client_metadata and client_metadata.get(key) is not None:
        return client_metadata[key]
    return None


def resolution_warning_text(
    *,
    media_type: Optional[str],
    width: Optional[int],
    height: Optional[int],
) -> Optional[str]:
    if width is None or height is None:
        return None
    long_edge = max(width, height)
    short_edge = min(width, height)
    if media_type == "video" and (long_edge < 1280 or short_edge < 720):
        return (
            "Low-resolution video can reduce bird detection recall, especially for small or distant birds."
        )
    if media_type == "image" and (long_edge < 1600 or short_edge < 900):
        return (
            "Low-resolution photos can reduce bird detection recall, especially for small or distant birds."
        )
    return None
