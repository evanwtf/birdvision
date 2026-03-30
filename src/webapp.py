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
from fastapi.templating import Jinja2Templates

from .pipeline import BirdIdentificationPipeline
from .video_metadata import VideoMetadata, extract as extract_video_metadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job state (in-memory; fine for a single-host home tool)
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, job_id: str, filename: str):
        self.id = job_id
        self.filename = filename
        self.status = "pending"   # pending | running | done | error
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.video_meta: Optional[VideoMetadata] = None


_jobs: dict[str, Job] = {}        # job_id → Job
_queue: asyncio.Queue             # filled at startup
_executor = ThreadPoolExecutor(max_workers=1)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: dict, templates_dir: str = "templates") -> FastAPI:
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
        asyncio.create_task(_worker(loop, pipeline))

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
        file: UploadFile = File(...),
        date_str: Optional[str] = Form(None, alias="date"),
    ):
        job_id = uuid.uuid4().hex
        safe_name = Path(file.filename).name
        dest = upload_dir / f"{job_id}_{safe_name}"

        contents = await file.read()
        dest.write_bytes(contents)
        logger.info(f"Saved upload: {dest} ({len(contents):,} bytes)")

        # Extract embedded metadata (date, GPS) from the video file
        loop = asyncio.get_event_loop()
        video_meta = await loop.run_in_executor(_executor, lambda: extract_video_metadata(str(dest)))
        logger.info(
            f"Video metadata: date={video_meta.recorded_at} "
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

        job = Job(job_id=job_id, filename=safe_name)
        job.video_meta = video_meta
        _jobs[job_id] = job
        await _queue.put((job, str(dest), video_date, video_meta))

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

        # The video path is stored in the result; fall back to scanning upload_dir
        video_path = None
        if old_job.result:
            candidate = Path(old_job.result["video"])
            if candidate.exists():
                video_path = candidate
        if video_path is None:
            # Try to find it in upload_dir by job_id prefix
            matches = list(upload_dir.glob(f"{job_id}_*"))
            if matches:
                video_path = matches[0]
        if video_path is None:
            return HTMLResponse("Original video file not found", status_code=404)

        new_job_id = uuid.uuid4().hex
        new_job = Job(job_id=new_job_id, filename=old_job.filename)
        new_job.video_meta = old_job.video_meta
        _jobs[new_job_id] = new_job

        video_date = old_job.video_meta.recorded_at if old_job.video_meta else None
        await _queue.put((new_job, str(video_path), video_date, old_job.video_meta))

        return RedirectResponse(f"/jobs/{new_job_id}", status_code=303)

    @app.get("/jobs/{job_id}/crops/{filename}")
    async def serve_crop(job_id: str, filename: str):
        job = _jobs.get(job_id)
        if job is None or job.result is None:
            return HTMLResponse("Not found", status_code=404)
        video_stem = Path(job.result["video"]).stem
        crop_path = results_dir / f"{video_stem}_crops" / filename
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
            original_filename = Path(result.get("video", "")).name
            # Strip the job_id prefix to get the original upload name
            if original_filename.startswith(job_id + "_"):
                original_filename = original_filename[len(job_id) + 1:]
            job = Job(job_id=job_id, filename=original_filename)
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


async def _worker(loop: asyncio.AbstractEventLoop, pipeline: BirdIdentificationPipeline):
    """Process jobs from the queue one at a time in a thread-pool executor."""
    while True:
        job, video_path, video_date, video_meta = await _queue.get()
        job.status = "running"
        logger.info(f"Processing job {job.id}: {job.filename}")
        try:
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
