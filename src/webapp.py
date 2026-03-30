"""
BirdVision web interface — upload videos, view results.
"""
import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .pipeline import BirdIdentificationPipeline

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

    pipeline: Optional[BirdIdentificationPipeline] = None

    # ── startup ──────────────────────────────────────────────────────────────

    @app.on_event("startup")
    async def startup():
        nonlocal pipeline
        global _queue
        _queue = asyncio.Queue()

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
        return templates.TemplateResponse("index.html", {
            "request": request,
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

        video_date = None
        if date_str:
            try:
                video_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                pass

        job = Job(job_id=job_id, filename=safe_name)
        _jobs[job_id] = job
        await _queue.put((job, str(dest), video_date))

        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_detail(request: Request, job_id: str):
        job = _jobs.get(job_id)
        if job is None:
            return HTMLResponse("Job not found", status_code=404)
        return templates.TemplateResponse("job.html", {
            "request": request,
            "job": job,
        })

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_pipeline(config: dict) -> BirdIdentificationPipeline:
    p = BirdIdentificationPipeline(config)
    species_file = config.get("species", {}).get("list_file")
    p.load_species(species_file)
    return p


async def _worker(loop: asyncio.AbstractEventLoop, pipeline: BirdIdentificationPipeline):
    """Process jobs from the queue one at a time in a thread-pool executor."""
    while True:
        job, video_path, video_date = await _queue.get()
        job.status = "running"
        logger.info(f"Processing job {job.id}: {job.filename}")
        try:
            result = await loop.run_in_executor(
                _executor,
                lambda: pipeline.process_video(video_path, video_date=video_date),
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
