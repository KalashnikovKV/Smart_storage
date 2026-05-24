"""Training API routes."""

from __future__ import annotations

import sys

from fastapi import APIRouter, HTTPException, Request

from src.web.schemas.jobs import JobResponse, JobStatusResponse, TrainRequest

router = APIRouter(prefix="/api/train", tags=["train"])


@router.post("/start", response_model=JobResponse)
def start_training(request: Request, body: TrainRequest):
    settings = request.app.state.settings
    command = [
        sys.executable,
        str(settings.project_root / "training" / "train_yolo.py"),
        "--data",
        str(settings.dataset_yaml),
        "--model",
        body.model,
        "--epochs",
        str(body.epochs),
        "--imgsz",
        str(body.imgsz),
        "--batch",
        str(body.batch),
        "--device",
        body.device,
    ]
    if body.validate_only:
        command.append("--validate-only")

    job = request.app.state.job_store.start(command, cwd=settings.project_root)
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        command=" ".join(command),
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_train_job(request: Request, job_id: str):
    job = request.app.state.job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        command=" ".join(job.command),
        output=job.output,
        exit_code=job.exit_code,
    )
