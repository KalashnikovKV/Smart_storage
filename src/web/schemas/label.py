"""Label API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LabelClassOption(BaseModel):
    hotkey: str
    label: str
    value: str


class LabelItem(BaseModel):
    image_name: str
    image_url: str
    preview_url: str
    predicted_category: str = ""
    confidence: float = 0.0
    color: str = ""
    size: str = ""
    method_used: str = ""
    processing_time_ms: float = 0.0
    has_detection: bool = False


class LabelQueueStats(BaseModel):
    total: int
    labeled: int
    remaining: int
    filter: str


class LabelNextResponse(BaseModel):
    item: LabelItem | None
    stats: LabelQueueStats
    classes: list[LabelClassOption]


class LabelSubmitRequest(BaseModel):
    image_name: str
    ground_truth: str = Field(
        description="Canonical class label, or empty to confirm prediction.",
    )
    confirm_prediction: bool = False
    skip: bool = False


class LabelSubmitResponse(BaseModel):
    saved_ground_truth: str
    next_item: LabelItem | None
    stats: LabelQueueStats
