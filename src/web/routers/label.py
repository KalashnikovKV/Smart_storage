"""Labeling API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from src.app.data_exporter import DataExporter
from src.app.labeling import resolve_ground_truth
from src.modes.common import export_result, resolve_csv_path
from src.web.schemas.label import (
    LabelNextResponse,
    LabelQueueStats,
    LabelSubmitRequest,
    LabelSubmitResponse,
)
from src.web.services.label_classes import list_label_classes
from src.web.services.label_queue import build_queue, queue_stats
from src.web.services.label_runner import (
    build_label_item,
    encode_jpeg,
    read_image,
    render_preview,
    run_pipeline,
)

router = APIRouter(prefix="/api/label", tags=["label"])

_VALID_FILTERS = {"unlabeled", "low_confidence", "all"}


def _stats(request: Request, label_filter: str) -> LabelQueueStats:
    settings = request.app.state.settings
    total, labeled, remaining = queue_stats(
        images_dir=settings.images_dir,
        csv_path=settings.csv_path,
        label_filter=label_filter,
        low_confidence_threshold=settings.low_confidence_threshold,
    )
    return LabelQueueStats(
        total=total,
        labeled=labeled,
        remaining=remaining,
        filter=label_filter,
    )


def _resolve_image_path(request: Request, image_name: str):
    settings = request.app.state.settings
    if "/" in image_name or "\\" in image_name or image_name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid image name")
    image_path = settings.images_dir / image_name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return image_path


def _process_image(request: Request, image_name: str):
    image_path = _resolve_image_path(request, image_name)
    pipeline = request.app.state.pipeline
    result = run_pipeline(pipeline, image_path)
    request.app.state.preview_cache[image_name] = result
    return result, image_path


def _next_item(
    request: Request,
    label_filter: str,
    *,
    skip: set[str] | None = None,
):
    settings = request.app.state.settings
    skip = skip or set()
    queue = build_queue(
        images_dir=settings.images_dir,
        csv_path=settings.csv_path,
        label_filter=label_filter,
        low_confidence_threshold=settings.low_confidence_threshold,
    )

    for image_name in queue:
        if image_name in skip:
            continue
        result, _image_path = _process_image(request, image_name)
        return build_label_item(
            image_name=image_name,
            result=result,
        )
    return None


@router.get("/classes")
def get_classes():
    return list_label_classes()


@router.get("/next", response_model=LabelNextResponse)
def get_next(
    request: Request,
    filter: str = Query(default="unlabeled", alias="filter"),
):
    if filter not in _VALID_FILTERS:
        raise HTTPException(status_code=400, detail=f"Invalid filter: {filter}")

    item = _next_item(request, filter)
    return LabelNextResponse(
        item=item,
        stats=_stats(request, filter),
        classes=list_label_classes(),
    )


@router.post("/submit", response_model=LabelSubmitResponse)
def submit_label(
    request: Request,
    body: LabelSubmitRequest,
    filter: str = Query(default="unlabeled", alias="filter"),
):
    settings = request.app.state.settings
    image_path = _resolve_image_path(request, body.image_name)

    label_filter = filter if filter in _VALID_FILTERS else "unlabeled"

    if body.skip:
        next_item = _next_item(
            request,
            label_filter,
            skip={body.image_name},
        )
        return LabelSubmitResponse(
            saved_ground_truth="",
            next_item=next_item,
            stats=_stats(request, label_filter),
        )

    cache = request.app.state.preview_cache
    result = cache.get(body.image_name)
    if result is None:
        result = run_pipeline(request.app.state.pipeline, image_path)

    if result is None:
        raise HTTPException(status_code=422, detail="No object detected in image")

    predicted = result.decision.category
    if body.confirm_prediction:
        ground_truth = predicted
    else:
        choice = body.ground_truth.strip()
        if choice == predicted or choice == "":
            ground_truth = resolve_ground_truth(predicted, "")
        else:
            # Match by label value or hotkey
            ground_truth = choice
            for option in list_label_classes():
                if choice in {option.value, option.label, option.hotkey}:
                    ground_truth = option.value
                    break

    csv_path = resolve_csv_path(str(settings.csv_path))
    exporter = DataExporter(csv_path)
    export_result(
        exporter=exporter,
        result=result,
        image_name=body.image_name,
        ground_truth=ground_truth,
    )

    next_item = _next_item(
        request,
        label_filter,
        skip={body.image_name},
    )

    return LabelSubmitResponse(
        saved_ground_truth=ground_truth,
        next_item=next_item,
        stats=_stats(request, label_filter),
    )


@router.get("/images/{image_name}")
def get_image(request: Request, image_name: str):
    image_path = _resolve_image_path(request, image_name)
    image = read_image(image_path)
    if image is None:
        raise HTTPException(status_code=404, detail="Could not read image")
    return Response(content=encode_jpeg(image), media_type="image/jpeg")


@router.get("/images/{image_name}/preview")
def get_preview(request: Request, image_name: str, overlay: bool = True):
    image_path = _resolve_image_path(request, image_name)
    image = read_image(image_path)
    if image is None:
        raise HTTPException(status_code=404, detail="Could not read image")

    result = request.app.state.preview_cache.get(image_name)
    if result is None:
        result = run_pipeline(request.app.state.pipeline, image_path)

    preview = render_preview(image, result, overlay=overlay)
    return Response(content=encode_jpeg(preview), media_type="image/jpeg")
