"""
AI Service — Construction Violation Detection
=============================================
Street-view uploads are analyzed with the configured YOLO checkpoint.
The provided `best_floor.pt` model is treated as a floor detector, so
street-view reports can be screened automatically while aerial reports
are routed into manual review until a dedicated aerial model exists.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from PIL import Image

from core.config import settings
from services.rule_engine import check_floor_violation

_STREET_MODEL = None
_STREET_MODEL_PATH: str | None = None


def _normalize_device() -> str | None:
    value = settings.AI_DEVICE.strip()
    if not value or value.lower() == "auto":
        return None
    return value


def _load_street_model():
    global _STREET_MODEL, _STREET_MODEL_PATH

    model_path = settings.resolved_ai_street_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Street model not found: {model_path}")

    current = str(model_path)
    if _STREET_MODEL is not None and _STREET_MODEL_PATH == current:
        return _STREET_MODEL

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run `pip install -r backend/requirements.txt`.",
        ) from exc

    _STREET_MODEL = YOLO(current)
    _STREET_MODEL_PATH = current
    return _STREET_MODEL


def _annotated_output_path() -> tuple[Path, str]:
    out_dir = Path(settings.UPLOAD_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"annotated_{uuid4().hex}.jpg"
    return out_dir / name, f"/uploads/{name}"


def _save_annotated_result(result, image_path: str) -> str:
    out_path, rel_path = _annotated_output_path()
    plotted = result.plot()
    if plotted is None:
        return image_path

    # Ultralytics returns BGR ndarray; convert to RGB before saving with Pillow.
    rgb = plotted[:, :, ::-1]
    Image.fromarray(rgb).save(out_path, format="JPEG", quality=92)
    return rel_path


def _count_detected_floors(result) -> int:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return 0
    names = getattr(result, "names", {}) or {}
    cls_values = getattr(boxes, "cls", None)
    try:
        total_boxes = int(len(boxes))
    except TypeError:
        return 0
    if cls_values is None:
        return total_boxes

    floor_like = {"floor", "ground", "storey", "story", "level"}
    matched = 0
    for raw_idx in cls_values.tolist():
        label = str(names.get(int(raw_idx), "")).strip().lower()
        if label in floor_like or "floor" in label or "storey" in label or "story" in label:
            matched += 1
    return matched or total_boxes


def _run_street_model(image_path: str, district: str) -> dict:
    model = _load_street_model()
    results = model.predict(
        source=image_path,
        conf=settings.AI_STREET_MODEL_CONFIDENCE,
        iou=settings.AI_STREET_MODEL_IOU,
        verbose=False,
        device=_normalize_device(),
    )
    if not results:
        return {
            "violation_flag": False,
            "violation_type": None,
            "detected_floors": 0,
            "setback_error": None,
            "image_evidence_path": image_path,
            "notes": "The model did not return any detections for this image.",
        }

    result = results[0]
    detected_floors = _count_detected_floors(result)
    rules = check_floor_violation(detected_floors, district)
    evidence_path = _save_annotated_result(result, image_path)

    return {
        "violation_flag": bool(rules.get("violation_flag")),
        "violation_type": rules.get("violation_type"),
        "detected_floors": detected_floors,
        "setback_error": None,
        "image_evidence_path": evidence_path,
        "notes": rules.get("detail"),
    }


async def process_street_view_image(image_path: str, district: str) -> dict:
    """
    Analyze a street-view image using the configured floor-detection YOLO model.
    """
    return await asyncio.to_thread(_run_street_model, image_path, district)


async def process_aerial_image(image_path: str, district: str) -> dict:
    """
    Aerial images need a separate model for setback / encroachment analysis.
    Until that exists, route the report to manual review without claiming compliance.
    """
    return {
        "violation_flag": False,
        "violation_type": "Manual_Review",
        "detected_floors": None,
        "setback_error": None,
        "image_evidence_path": image_path,
        "workflow_status": "Under_Review",
        "notes": (
            "Aerial image received. The configured model detects floors only, "
            "so this report has been routed for manual setback/encroachment review."
        ),
    }
