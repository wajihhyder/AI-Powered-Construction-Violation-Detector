#!/usr/bin/env python3
"""
preview_floor_detection.py
==========================
Run the floor detector on a folder of street-view images and save annotated
previews into a new directory. Useful for picking demo pictures.

Reuses the production post-processing (vertical clustering + confidence /
width filtering) by importing the counter from `backend.services.ai_service`,
so what you see here is what the FastAPI flow will report.

Examples
--------
    # default: read every image under ./test_images, write to ./demo_floor_previews
    python scripts/preview_floor_detection.py --input-dir ./test_images

    # override model + output and disable test-time augmentation for speed
    python scripts/preview_floor_detection.py \\
        --input-dir ./samples \\
        --output-dir ./previews \\
        --model ./best_floor.pt \\
        --no-augment \\
        --device cpu
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"

# Allow `from core.config import settings` and friends without standing up the
# whole FastAPI app — set the dev secret so settings validation passes.
os.environ.setdefault("VIOSCAN_DEV_DEFAULT_SECRET", "1")
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from core.config import settings  # noqa: E402
from services.ai_service import _count_detected_floors  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_INPUT = REPO_ROOT / "test_images"
DEFAULT_OUTPUT = REPO_ROOT / "demo_floor_previews"
DEFAULT_MODEL = REPO_ROOT / "best_floor.pt"
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Folder containing source images (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Folder to write annotated images into (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to YOLO weights (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--conf", type=float, default=settings.AI_STREET_MODEL_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=settings.AI_STREET_MODEL_IOU)
    parser.add_argument("--imgsz", type=int, default=settings.AI_STREET_MODEL_IMGSZ)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable test-time augmentation (faster, slightly lower recall)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='YOLO device — "cpu", "cuda", "0", "auto" (default: cpu)',
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories of --input-dir",
    )
    return parser.parse_args()


def gather_images(folder: Path, recursive: bool) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f"Input folder not found: {folder}")
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(
    image_path: Path,
    result,
    floor_count: int,
    raw_boxes: int,
    out_dir: Path,
) -> Path:
    plotted = result.plot()
    if plotted is None:
        # Fall back to the original image so we still write something.
        img = Image.open(image_path).convert("RGB")
    else:
        img = Image.fromarray(plotted[:, :, ::-1])

    draw = ImageDraw.Draw(img)
    width = img.size[0]
    font_size = max(20, int(width * 0.035))
    font = _load_font(font_size)
    label = f"Floors: {floor_count}    raw boxes: {raw_boxes}"
    pad = max(8, font_size // 3)
    bbox = draw.textbbox((pad, pad), label, font=font)
    background = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(background, fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)

    out_name = f"floors_{floor_count:02d}_{image_path.stem}.jpg"
    out_path = out_dir / out_name
    img.save(out_path, format="JPEG", quality=92)
    return out_path


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model weights not found: {args.model}")
    images = gather_images(args.input_dir, args.recursive)
    if not images:
        raise SystemExit(
            f"No images found under {args.input_dir}. Supported suffixes: "
            f"{sorted(IMAGE_SUFFIXES)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run "
            "`pip install -r backend/requirements.txt` first."
        ) from exc

    print(f"Loading model: {args.model}")
    print(
        f"Inference settings — conf={args.conf}, iou={args.iou}, "
        f"imgsz={args.imgsz}, augment={not args.no_augment}, device={args.device}"
    )
    model = YOLO(str(args.model))

    rows: list[dict] = []
    for img_path in images:
        try:
            preds = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                augment=not args.no_augment,
                verbose=False,
                device=args.device,
            )
        except Exception as e:
            print(f"  ! {img_path.name}: predict failed — {e}")
            continue
        if not preds:
            print(f"  - {img_path.name}: no result returned")
            continue
        result = preds[0]
        floor_count = _count_detected_floors(result)
        raw_boxes = int(len(result.boxes)) if result.boxes is not None else 0
        out_path = annotate(img_path, result, floor_count, raw_boxes, args.output_dir)
        rows.append(
            {
                "image": str(img_path.relative_to(args.input_dir)),
                "annotated": out_path.name,
                "floors": floor_count,
                "raw_boxes": raw_boxes,
            }
        )
        print(
            f"  ✓ {img_path.name:50s} floors={floor_count:2d}  "
            f"raw_boxes={raw_boxes:3d}  →  {out_path.name}"
        )

    if rows:
        summary = args.output_dir / "floor_preview_summary.csv"
        with summary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image", "annotated", "floors", "raw_boxes"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nProcessed {len(rows)} images.")
        print(f"Annotated previews: {args.output_dir}")
        print(f"Summary CSV:        {summary}")
    else:
        print("\nNo images were successfully processed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
