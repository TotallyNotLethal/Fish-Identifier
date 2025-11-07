"""Converters for turning annotation exports into YOLO-compatible labels.

The module currently supports Label Studio JSON and CVAT XML exports. It can be
used as a command line utility or imported from tests / other scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from xml.etree import ElementTree

YOLO_EXT = ".txt"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_yolo_annotations(output_path: Path, annotations: Iterable[Tuple[int, float, float, float, float]]) -> None:
    ensure_dir(output_path.parent)
    lines = ["{} {:.6f} {:.6f} {:.6f} {:.6f}".format(cls, cx, cy, w, h) for cls, cx, cy, w, h in annotations]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_label_studio(json_path: Path) -> List[Dict]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    tasks = payload if isinstance(payload, list) else payload.get("tasks") or payload
    if not isinstance(tasks, list):  # pragma: no cover - defensive
        raise ValueError("Label Studio export must be a list or contain a 'tasks' list")
    return tasks


def _label_studio_annotation_to_yolo(result: Dict, *, class_map: Dict[str, int]) -> Tuple[int, float, float, float, float]:
    value = result["value"]
    label = value["rectanglelabels"][0]
    cls = class_map[label]
    # Label Studio stores coordinates as percentages (0-100).
    x_pct = float(value["x"]) / 100
    y_pct = float(value["y"]) / 100
    w_pct = float(value["width"]) / 100
    h_pct = float(value["height"]) / 100
    cx = x_pct + w_pct / 2
    cy = y_pct + h_pct / 2
    return cls, cx, cy, w_pct, h_pct


def convert_label_studio(json_path: Path, images_dir: Path, output_dir: Path, classes: Sequence[str]) -> Dict[str, Path]:
    class_map = {label: idx for idx, label in enumerate(classes)}
    tasks = parse_label_studio(json_path)
    outputs: Dict[str, Path] = {}

    for task in tasks:
        data = task.get("data", {})
        image_name = Path(data.get("image", "")).name
        if not image_name:
            continue
        results = []
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                if result.get("type") not in {"rectanglelabels", "labels"}:
                    continue
                if "rectanglelabels" not in result.get("value", {}):
                    continue
                results.append(_label_studio_annotation_to_yolo(result, class_map=class_map))
        image_path = images_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image {image_path} referenced in annotations but not found")
        output_path = output_dir / f"{image_path.stem}{YOLO_EXT}"
        save_yolo_annotations(output_path, results)
        outputs[image_name] = output_path
    return outputs


def parse_cvat(xml_path: Path) -> ElementTree.Element:
    tree = ElementTree.parse(xml_path)
    return tree.getroot()


def convert_cvat(xml_path: Path, images_dir: Path, output_dir: Path, classes: Sequence[str]) -> Dict[str, Path]:
    class_map = {label: idx for idx, label in enumerate(classes)}
    root = parse_cvat(xml_path)
    outputs: Dict[str, Path] = {}

    for image_node in root.findall(".//image"):
        image_name = image_node.attrib.get("name")
        if not image_name:
            continue
        width = float(image_node.attrib.get("width", 1))
        height = float(image_node.attrib.get("height", 1))
        annotations = []
        for box in image_node.findall("box"):
            label = box.attrib.get("label")
            if label not in class_map:
                continue
            cls = class_map[label]
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            w = (xbr - xtl) / width
            h = (ybr - ytl) / height
            cx = (xtl + xbr) / 2 / width
            cy = (ytl + ybr) / 2 / height
            annotations.append((cls, cx, cy, w, h))
        image_path = images_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image {image_path} referenced in annotations but not found")
        output_path = output_dir / f"{image_path.stem}{YOLO_EXT}"
        save_yolo_annotations(output_path, annotations)
        outputs[image_name] = output_path
    return outputs


def _relative_to_root(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root)
    except ValueError:  # pragma: no cover - fallback for paths outside the root
        return path.resolve()


def write_dataset_manifest(manifest_path: Path, *, dataset_root: Path, train_dir: Path, val_dir: Path, test_dir: Path, names: Sequence[str]) -> None:
    ensure_dir(manifest_path.parent)
    dataset_root = dataset_root.resolve()
    content_lines = [
        f"path: {dataset_root}",
        f"train: {_relative_to_root(Path(train_dir), dataset_root)}",
        f"val: {_relative_to_root(Path(val_dir), dataset_root)}",
        f"test: {_relative_to_root(Path(test_dir), dataset_root)}",
        "names:",
    ]
    content_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    manifest_path.write_text("\n".join(content_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert annotations to YOLO format and emit dataset manifest")
    subparsers = parser.add_subparsers(dest="command")

    convert_parser = subparsers.add_parser("convert", help="Convert annotations into YOLO TXT files")
    convert_parser.add_argument("--format", choices=["label-studio", "cvat"], required=True, help="Annotation export format")
    convert_parser.add_argument("--input", type=Path, required=True, help="Path to the exported annotations")
    convert_parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing the referenced images")
    convert_parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write YOLO TXT files")
    convert_parser.add_argument("--classes", nargs="+", required=True, help="Ordered list of class labels")

    manifest_parser = subparsers.add_parser("manifest", help="Generate a dataset manifest YAML")
    manifest_parser.add_argument("--dataset-root", type=Path, default=Path("data/processed"), help="Root directory for dataset paths")
    manifest_parser.add_argument("--train-dir", type=Path, required=True, help="Path to train images directory")
    manifest_parser.add_argument("--val-dir", type=Path, required=True, help="Path to validation images directory")
    manifest_parser.add_argument("--test-dir", type=Path, required=True, help="Path to test images directory")
    manifest_parser.add_argument("--names", nargs="+", required=True, help="Class names in order")
    manifest_parser.add_argument("--output", type=Path, default=Path("data/datasets/fish.yaml"), help="Where to store the manifest")

    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    if args.command == "convert":
        converter = convert_label_studio if args.format == "label-studio" else convert_cvat
        converter(args.input, args.images_dir, args.output_dir, args.classes)
    elif args.command == "manifest":
        write_dataset_manifest(
            args.output,
            dataset_root=args.dataset_root,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            names=args.names,
        )
    else:
        raise SystemExit("A subcommand is required. Use --help for details.")


if __name__ == "__main__":
    cli()
