"""Dataset build API routes."""

from __future__ import annotations

import sys

from fastapi import APIRouter, Request

from src.web.schemas.jobs import DatasetBuildRequest, JobResponse, JobStatusResponse

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.post("/build", response_model=JobResponse)
def build_dataset(request: Request, body: DatasetBuildRequest):
    settings = request.app.state.settings
    command = [
        sys.executable,
        str(settings.project_root / "training" / "build_dataset.py"),
        "--csv",
        str(settings.csv_path),
        "--images-dir",
        str(settings.images_dir),
        "--output",
        str(settings.dataset_dir),
        "--val-ratio",
        str(body.val_ratio),
    ]
    if body.clean:
        command.append("--clean")
    if body.require_ground_truth:
        command.append("--require-ground-truth")
    if body.regenerate_masks:
        command.append("--regenerate-masks")
    if body.min_confidence > 0:
        command.extend(["--min-confidence", str(body.min_confidence)])

    job = request.app.state.job_store.start(command, cwd=settings.project_root)
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        command=" ".join(command),
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_dataset_job(request: Request, job_id: str):
    return _job_status(request, job_id)


def _job_status(request: Request, job_id: str) -> JobStatusResponse:
    job = request.app.state.job_store.get(job_id)
    if job is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        command=" ".join(job.command),
        output=job.output,
        exit_code=job.exit_code,
    )
