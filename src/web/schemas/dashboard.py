"""Dashboard API models."""

from __future__ import annotations

from pydantic import BaseModel


class ClassProgress(BaseModel):
    slug: str
    have: int
    need: int
    ok: bool


class DashboardStats(BaseModel):
    total_images: int
    labeled_images: int
    csv_rows: int
    dataset_train: int
    dataset_val: int
    class_progress: list[ClassProgress]
    checklist: dict[str, bool]
