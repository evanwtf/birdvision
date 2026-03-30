"""
BirdVision web interface — upload videos, view results.
"""
import asyncio
import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import yaml

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi.templating import Jinja2Templates

from .pipeline import BirdIdentificationPipeline
from .video_metadata import VideoMetadata, extract as extract_video_metadata

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}

# ---------------------------------------------------------------------------
# Job state (in-memory; fine for a single-host home tool)
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, job_id: str, filename: str, media_type: str = "video"):
        self.id = job_id
        self.filename = filename
        self.media_type = media_type  # "video" or "images"
        self.status = "pending"   # pending | running | done | error
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.video_meta: Optional[VideoMetadata] = None
        self.image_paths: list[str] = []  # for image jobs


_jobs: dict[str, Job] = {}        # job_id → Job
_queue: asyncio.Queue             # filled at startup
_executor = ThreadPoolExecutor(max_workers=1)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: dict, templates_dir: str = "templates", config_path: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="BirdVision")
    templates = Jinja2Templates(directory=templates_dir)

    upload_dir = Path(config.get("webapp", {}).get("upload_dir", "videos"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.get("output", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    pipeline: Optional[BirdIdentificationPipeline] = None

    # ── startup ──────────────────────────────────────────────────────────────

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

    # ── routes ───────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        recent = list(reversed(list(_jobs.values())))[:20]
        return templates.TemplateResponse(request, "index.html", {
            "jobs": recent,
            "today": date.today().isoformat(),
        })

    @app.post("/upload")
    async def upload(
        request: Request,
        date_str: Optional[str] = Form(None, alias="date"),
    ):
        form = await request.form()
        files = form.getlist("file")
        if not files or not isinstance(files[0], StarletteUploadFile):
            return HTMLResponse("No file uploaded", status_code=400)

        job_id = uuid.uuid4().hex

        # Classify each file as video or image by extension
        file_types = set()
        for f in files:
            ext = Path(f.filename).suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                file_types.add("video")
            elif ext in IMAGE_EXTENSIONS:
                file_types.add("image")

        if len(file_types) > 1:
            return HTMLResponse(
                "Please upload either videos or photos, not both at once.",
                status_code=400,
            )

        is_image_job = "image" in file_types

        # Save uploaded files
        saved_paths: list[Path] = []
        for f in files:
            safe_name = Path(f.filename).name
            dest = upload_dir / f"{job_id}_{safe_name}"
            contents = await f.read()
            dest.write_bytes(contents)
            logger.info(f"Saved upload: {dest} ({len(contents):,} bytes)")
            saved_paths.append(dest)

        # Extract metadata from the first file
        loop = asyncio.get_event_loop()
        meta_path = saved_paths[0]
        video_meta = await loop.run_in_executor(
            _executor, lambda: extract_video_metadata(str(meta_path))
        )
        logger.info(
            f"Media metadata: date={video_meta.recorded_at} "
            f"gps={'yes' if video_meta.has_gps else 'no'}"
        )

        # Manual date overrides embedded date
        video_date = None
        if date_str:
            try:
                video_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                pass
        if video_date is None and video_meta.recorded_at:
            video_date = video_meta.recorded_at

        display_name = Path(files[0].filename).name
        if is_image_job and len(files) > 1:
            display_name = f"{len(files)} photos"

        media_type = "images" if is_image_job else "video"
        job = Job(job_id=job_id, filename=display_name, media_type=media_type)
        job.video_meta = video_meta
        if is_image_job:
            job.image_paths = [str(p) for p in saved_paths]
        _jobs[job_id] = job
        await _queue.put((job, str(saved_paths[0]), video_date, video_meta))

        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

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

        is_image_job = old_job.media_type == "images"

        if is_image_job:
            # For image jobs, find original files by old job_id prefix
            image_paths = old_job.image_paths
            if not image_paths:
                matches = sorted(upload_dir.glob(f"{job_id}_*"))
                image_paths = [str(p) for p in matches]
            if not image_paths:
                return HTMLResponse("Original image files not found", status_code=404)
            first_path = image_paths[0]
        else:
            # The video path is stored in the result; fall back to scanning upload_dir
            first_path = None
            if old_job.result:
                candidate = Path(old_job.result["video"])
                if candidate.exists():
                    first_path = str(candidate)
            if first_path is None:
                matches = list(upload_dir.glob(f"{job_id}_*"))
                if matches:
                    first_path = str(matches[0])
            if first_path is None:
                return HTMLResponse("Original video file not found", status_code=404)

        new_job_id = uuid.uuid4().hex
        new_job = Job(job_id=new_job_id, filename=old_job.filename, media_type=old_job.media_type)
        new_job.video_meta = old_job.video_meta
        if is_image_job:
            new_job.image_paths = image_paths
        _jobs[new_job_id] = new_job

        video_date = old_job.video_meta.recorded_at if old_job.video_meta else None
        await _queue.put((new_job, first_path, video_date, old_job.video_meta))

        return RedirectResponse(f"/jobs/{new_job_id}", status_code=303)

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

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_existing_jobs(results_dir: Path):
    """Reconstruct completed jobs from results JSON files on disk."""
    # Result files are named {job_id}_{original_stem}_results.json
    # job_id is a 32-char hex string
    pattern = re.compile(r'^([0-9a-f]{32})_(.+)_results\.json$')
    loaded = 0
    for result_file in sorted(results_dir.glob("*_results.json")):
        m = pattern.match(result_file.name)
        if not m:
            continue
        job_id, _ = m.group(1), m.group(2)
        if job_id in _jobs:
            continue
        try:
            result = json.loads(result_file.read_text())
            is_image = result.get("type") == "images"
            if is_image:
                info = result.get("image_info", {})
                count = info.get("count", 0)
                names = info.get("filenames", [])
                original_filename = f"{count} photos" if count > 1 else (names[0] if names else "photos")
            else:
                original_filename = Path(result.get("video", "")).name
                # Strip the job_id prefix to get the original upload name
                if original_filename.startswith(job_id + "_"):
                    original_filename = original_filename[len(job_id) + 1:]
            media_type = "images" if is_image else "video"
            job = Job(job_id=job_id, filename=original_filename, media_type=media_type)
            job.status = "done"
            job.result = result
            # Reconstruct video_meta from saved coords so the OSM link works
            if result.get("latitude") and result.get("longitude"):
                from .video_metadata import VideoMetadata
                from datetime import datetime as dt
                vm = VideoMetadata(
                    latitude=result["latitude"],
                    longitude=result["longitude"],
                )
                if result.get("date"):
                    try:
                        vm.recorded_at = dt.fromisoformat(result["date"])
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
    """Process jobs from the queue one at a time in a thread-pool executor."""
    while True:
        job, video_path, video_date, video_meta = await _queue.get()
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
                        video_date=video_date,
                        latitude=video_meta.latitude if video_meta else None,
                        longitude=video_meta.longitude if video_meta else None,
                        job_id=job.id,
                    ),
                )
            else:
                result = await loop.run_in_executor(
                    _executor,
                    lambda: pipeline.process_video(
                        video_path,
                        video_date=video_date,
                        latitude=video_meta.latitude if video_meta else None,
                        longitude=video_meta.longitude if video_meta else None,
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
