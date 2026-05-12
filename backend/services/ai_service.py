"""
AI Service — Construction Violation Detection
=============================================
Street-view uploads are analyzed with the configured YOLO checkpoint.
The provided `best_floor.pt` model is treated as a floor detector for
street-view reports, while aerial uploads are screened by the encroachment
package (OSM building / road footprints vs the SBCA setback rule).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from core.config import settings
from services.encroachment import analyze_aerial_encroachment
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


_FLOOR_LABEL_TOKENS = ("floor", "storey", "story", "level", "ground")


def _is_floor_label(label: str) -> bool:
    label = label.strip().lower()
    if not label:
        return False
    if label in {"floor", "ground", "storey", "story", "level"}:
        return True
    return any(token in label for token in _FLOOR_LABEL_TOKENS)


def _as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _gather_floor_boxes(result) -> np.ndarray:
    """
    Return the subset of YOLO boxes that are floor-class and above the
    post-inference confidence floor. Shape: (N, 5) — [x1, y1, x2, y2, conf].
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return np.zeros((0, 5), dtype=float)

    xyxy = _as_numpy(getattr(boxes, "xyxy", None))
    if xyxy is None or xyxy.size == 0:
        return np.zeros((0, 5), dtype=float)

    conf = _as_numpy(getattr(boxes, "conf", None))
    if conf is None:
        conf = np.ones(len(xyxy), dtype=float)

    cls = _as_numpy(getattr(boxes, "cls", None))
    names = getattr(result, "names", {}) or {}

    if cls is None:
        # Single-class model: treat every box as a floor candidate.
        keep_mask = np.ones(len(xyxy), dtype=bool)
    else:
        keep_mask = np.array(
            [_is_floor_label(str(names.get(int(c), ""))) for c in cls],
            dtype=bool,
        )

    keep_mask &= conf >= settings.AI_STREET_MODEL_MIN_FLOOR_CONFIDENCE
    if not keep_mask.any():
        return np.zeros((0, 5), dtype=float)

    selected = np.column_stack([xyxy[keep_mask], conf[keep_mask]]).astype(float)
    return selected


def _drop_narrow_outliers(floor_boxes: np.ndarray) -> np.ndarray:
    """Discard boxes much narrower than the median (windows / AC units / billboard text)."""
    if floor_boxes.shape[0] == 0:
        return floor_boxes
    widths = floor_boxes[:, 2] - floor_boxes[:, 0]
    if widths.size == 0:
        return floor_boxes
    median_w = float(np.median(widths))
    if median_w <= 0:
        return floor_boxes
    keep = widths >= median_w * settings.AI_STREET_MODEL_MIN_WIDTH_RATIO
    if not keep.any():
        return floor_boxes
    return floor_boxes[keep]


def _cluster_floor_levels(floor_boxes: np.ndarray) -> int:
    """
    Group boxes by their vertical center; storeys whose centers fall within
    `gap_ratio × median_box_height` of each other count as the same floor.
    """
    if floor_boxes.shape[0] == 0:
        return 0
    y_centers = (floor_boxes[:, 1] + floor_boxes[:, 3]) / 2.0
    heights = floor_boxes[:, 3] - floor_boxes[:, 1]
    median_h = float(np.median(heights)) if heights.size else 1.0
    if median_h <= 0:
        return int(floor_boxes.shape[0])

    gap_threshold = max(median_h * settings.AI_STREET_MODEL_FLOOR_GAP_RATIO, 1.0)
    sorted_y = np.sort(y_centers)
    cluster_count = 1
    last_y = float(sorted_y[0])
    for y in sorted_y[1:]:
        y_val = float(y)
        if y_val - last_y > gap_threshold:
            cluster_count += 1
        last_y = y_val
    return cluster_count


def _count_detected_floors(result) -> int:
    floor_boxes = _gather_floor_boxes(result)
    if floor_boxes.shape[0] == 0:
        return 0
    floor_boxes = _drop_narrow_outliers(floor_boxes)
    if floor_boxes.shape[0] == 0:
        return 0
    return _cluster_floor_levels(floor_boxes)


def _run_street_model(image_path: str, district: str) -> dict:
    model = _load_street_model()
    results = model.predict(
        source=image_path,
        conf=settings.AI_STREET_MODEL_CONFIDENCE,
        iou=settings.AI_STREET_MODEL_IOU,
        imgsz=settings.AI_STREET_MODEL_IMGSZ,
        augment=settings.AI_STREET_MODEL_AUGMENT,
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


async def process_aerial_image(
    image_path: str,
    district: str,
    gps_coords: str | None = None,
) -> dict:
    """
    Aerial submissions are run through the YOLO building segmenter and the OSM
    context layer (roads / public-space / water / mapped buildings). The result
    carries a per-category area breakdown and a color-coded evidence overlay.
    """
    result = await analyze_aerial_encroachment(image_path, district, gps_coords)
    payload: dict = {
        "violation_flag": result.violation_flag,
        "violation_type": result.violation_type,
        "detected_floors": None,
        "setback_error": None,
        "image_evidence_path": result.image_evidence_path,
        "notes": result.notes,
        "encroachment_total_m2": result.total_area_m2,
        "encroachment_breakdown": result.breakdown_json(),
    }
    if result.workflow_status:
        payload["workflow_status"] = result.workflow_status
    return payload
