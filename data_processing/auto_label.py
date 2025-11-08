"""Automatically annotate fish bounding boxes with a pretrained YOLO model.

This utility runs object detection for every image in the configured dataset
splits and emits Label Studio compatible JSON exports.  The generated files can
then be fed directly into :mod:`data_processing.convert_labels` to produce YOLO
TXT annotations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from uuid import uuid4

try:  # Import lazily so unit tests can exercise helpers without ultralytics.
    from ultralytics import YOLO  # type: ignore
except Exception as exc:  # pragma: no cover - handled at runtime
    YOLO = None  # type: ignore
    _IMPORT_ERROR = exc
else:  # pragma: no cover - runtime path, covered via integration usage
    _IMPORT_ERROR = None


DATA_SPLITS = ("train", "val", "test")


@dataclass
class Detection:
    """Lightweight representation of a single bounding box prediction."""

    cls: int
    conf: float
    x_center: float
    y_center: float
    width: float
    height: float


def _clamp(value: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def detection_to_label_studio_result(
    detection: Detection, *, class_names: Sequence[str], original_width: int, original_height: int
) -> Dict:
    """Convert a :class:`Detection` to the structure expected by Label Studio."""

    if detection.cls >= len(class_names):  # pragma: no cover - defensive guard
        raise IndexError(f"Class index {detection.cls} is not present in the provided class names")

    # Label Studio stores bounding boxes as percentages of the image size.
    x_center = _clamp(detection.x_center)
    y_center = _clamp(detection.y_center)
    width = _clamp(detection.width)
    height = _clamp(detection.height)

    x_pct = (x_center - width / 2) * 100
    y_pct = (y_center - height / 2) * 100
    width_pct = width * 100
    height_pct = height * 100

    # Guard against floating point drift that might push values outside the image.
    x_pct = max(0.0, min(100.0, x_pct))
    y_pct = max(0.0, min(100.0, y_pct))
    width_pct = max(0.0, min(100.0, width_pct))
    height_pct = max(0.0, min(100.0, height_pct))

    return {
        "id": str(uuid4()),
        "type": "rectanglelabels",
        "value": {
            "x": x_pct,
            "y": y_pct,
            "width": width_pct,
            "height": height_pct,
            "rotation": 0,
            "rectanglelabels": [class_names[detection.cls]],
        },
        "score": detection.conf,
        "from_name": "bbox",
        "to_name": "image",
        "image_rotation": 0,
        "original_width": original_width,
        "original_height": original_height,
    }


def build_label_studio_task(
    image_name: str,
    detections: Iterable[Detection],
    *,
    class_names: Sequence[str],
    original_width: int,
    original_height: int,
    min_confidence: float,
) -> Dict:
    """Construct a Label Studio task payload for an image."""

    results = [
        detection_to_label_studio_result(
            detection,
            class_names=class_names,
            original_width=original_width,
            original_height=original_height,
        )
        for detection in detections
        if detection.conf >= min_confidence
    ]

    return {
        "id": str(uuid4()),
        "data": {"image": image_name},
        "annotations": [
            {
                "id": str(uuid4()),
                "completed_by": 1,
                "result": results,
            }
        ],
    }


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_species(names_path: Path | None) -> List[str]:
    if names_path is None:
        return []
    return [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_class_names(user_provided: Sequence[str], model_names: Dict[int, str]) -> List[str]:
    if user_provided:
        return list(user_provided)
    return [model_names[idx] for idx in sorted(model_names)]


def collect_images(split_dir: Path) -> List[Path]:
    return sorted(
        [p for p in split_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    )


def run_inference(
    weights: Path | str,
    image_paths: Sequence[Path],
    *,
    device: str | None,
    conf: float,
    iou: float,
):
    if YOLO is None:  # pragma: no cover - runtime safeguard
        raise RuntimeError(
            "Ultralytics could not be imported. Install dependencies from requirements.txt before running this script"
        ) from _IMPORT_ERROR

    model = YOLO(str(weights))
    predictions = model.predict(image_paths, conf=conf, iou=iou, verbose=False, device=device, stream=False)
    return list(predictions)


def yolo_result_to_detections(result) -> List[Detection]:  # pragma: no cover - exercised via integration usage
    detections: List[Detection] = []
    if not hasattr(result, "boxes"):
        return detections

    boxes = result.boxes
    if boxes is None:
        return detections

    xywhn = boxes.xywhn
    classes = boxes.cls
    confidences = boxes.conf

    if xywhn is None or classes is None or confidences is None:
        return detections

    for idx in range(len(xywhn)):
        x_center, y_center, width, height = map(float, xywhn[idx].tolist())
        cls = int(float(classes[idx]))
        conf = float(confidences[idx])
        detections.append(Detection(cls=cls, conf=conf, x_center=x_center, y_center=y_center, width=width, height=height))
    return detections


def generate_annotations_for_split(
    split: str,
    images_dir: Path,
    output_dir: Path,
    *,
    weights: Path | str,
    device: str | None,
    conf: float,
    iou: float,
    class_names: Sequence[str],
    min_confidence: float,
) -> Path:
    image_paths = collect_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    predictions = run_inference(weights, image_paths, device=device, conf=conf, iou=iou)
    model_class_names = infer_class_names(class_names, predictions[0].names)

    tasks: List[Dict] = []
    for image_path, result in zip(image_paths, predictions):
        detections = yolo_result_to_detections(result)
        height, width = result.orig_shape[:2]
        task = build_label_studio_task(
            image_path.name,
            detections,
            class_names=model_class_names,
            original_width=width,
            original_height=height,
            min_confidence=min_confidence,
        )
        tasks.append(task)

    output_path = output_dir / f"label-studio-{split}.json"
    ensure_output_dir(output_path)
    output_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatically label fish bounding boxes using a YOLO detector")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("data/processed/images"),
        help="Directory containing split subfolders (train/val/test by default)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DATA_SPLITS),
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations/auto"),
        help="Where to store the generated Label Studio JSON files",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to pretrained YOLO weights for fish detection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device identifier recognised by Ultralytics (e.g. 'cpu', '0', '0,1')",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold passed to the YOLO predictor",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum confidence for retaining detections in the exported annotations",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Intersection-over-union threshold for non-maximum suppression",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Optional ordered list of class names to override the model's defaults",
    )
    parser.add_argument(
        "--class-names-file",
        type=Path,
        default=None,
        help="Path to a newline separated list of class names (used if --class-names is omitted)",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - exercised via integration usage
    args = parse_args()

    split_outputs: Dict[str, Path] = {}
    provided_class_names: Sequence[str] = args.class_names or load_species(args.class_names_file)

    for split in args.splits:
        images_dir = args.images_root / split
        if not images_dir.exists():
            raise FileNotFoundError(f"Split directory {images_dir} does not exist")
        output_path = generate_annotations_for_split(
            split,
            images_dir,
            args.output_dir,
            weights=args.weights,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            class_names=provided_class_names,
            min_confidence=args.min_confidence,
        )
        split_outputs[split] = output_path

    summary_lines = ["Generated annotation exports:"]
    for split, path in split_outputs.items():
        summary_lines.append(f"  {split}: {path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
