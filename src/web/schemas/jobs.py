"""Background job API models."""

from __future__ import annotations

from pydantic import BaseModel


class DatasetBuildRequest(BaseModel):
    clean: bool = False
    require_ground_truth: bool = False
    regenerate_masks: bool = False
    min_confidence: float = 0.0
    val_ratio: float = 0.2


class TrainRequest(BaseModel):
    validate_only: bool = False
    model: str = "yolo11n-seg.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 8
    device: str = "cpu"


class JobResponse(BaseModel):
    job_id: str
    status: str
    command: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    command: str
    output: str
    exit_code: int | None = None
