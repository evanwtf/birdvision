"""
BirdVision web interface — inspect uploads, create jobs, view results.
"""
import asyncio
import hashlib
import json
import logging
import re
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
from starlette.datastructures import UploadFile as StarletteUploadFile

from .pipeline import BirdIdentificationPipeline
from .video_metadata import VideoMetadata, inspect_media

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


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

    upload_dir = Path(config.get("webapp", {}).get("upload_dir", "videos"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.get("output", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    asset_store = AssetStore(upload_dir)

    pipeline: Optional[BirdIdentificationPipeline] = None

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
        return templates.TemplateResponse(request, "index.html", {
            "jobs": recent,
        })

    @app.post("/api/uploads/inspect")
    async def inspect_upload_candidates(request: Request):
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
        return templates.TemplateResponse(request, "job.html", {
            "job": job,
        })

    @app.post("/jobs/{job_id}/reprocess")
    async def reprocess(job_id: str):
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

    return app


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
