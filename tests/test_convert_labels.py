from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing.convert_labels import convert_cvat, convert_label_studio, write_dataset_manifest


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image-bytes")


def test_convert_label_studio_creates_yolo_labels(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    image_path = images_dir / "fish1.jpg"
    create_image(image_path)

    export_payload = [
        {
            "data": {"image": str(image_path)},
            "annotations": [
                {
                    "result": [
                        {
                            "type": "rectanglelabels",
                            "value": {
                                "x": 10,
                                "y": 20,
                                "width": 40,
                                "height": 50,
                                "rectanglelabels": ["Salmon"],
                            },
                        }
                    ]
                }
            ],
        }
    ]
    json_path = tmp_path / "label_studio.json"
    json_path.write_text(json.dumps(export_payload), encoding="utf-8")

    outputs = convert_label_studio(json_path, images_dir, labels_dir, ["Salmon", "Tuna"])
    output_path = outputs["fish1.jpg"]
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8").strip()
    assert content == "0 0.300000 0.450000 0.400000 0.500000"


def test_convert_cvat_creates_yolo_labels(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    image_path = images_dir / "fish2.jpg"
    create_image(image_path)

    xml_content = """
    <annotations>
        <image name=\"fish2.jpg\" width=\"1000\" height=\"500\">
            <box label=\"Salmon\" xtl=\"10\" ytl=\"20\" xbr=\"110\" ybr=\"220\" />
        </image>
    </annotations>
    """
    xml_path = tmp_path / "annotations.xml"
    xml_path.write_text(xml_content, encoding="utf-8")

    outputs = convert_cvat(xml_path, images_dir, labels_dir, ["Salmon", "Tuna"])
    output_path = outputs["fish2.jpg"]
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8").strip()
    assert content == "0 0.060000 0.240000 0.100000 0.400000"


def test_manifest_points_to_expected_directories(tmp_path: Path) -> None:
    dataset_root = tmp_path / "processed"
    train_dir = dataset_root / "images" / "train"
    val_dir = dataset_root / "images" / "val"
    test_dir = dataset_root / "images" / "test"

    for directory in (train_dir, val_dir, test_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = tmp_path / "fish.yaml"
    write_dataset_manifest(
        manifest_path,
        dataset_root=dataset_root,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        names=["Salmon", "Tuna"],
    )

    content = manifest_path.read_text(encoding="utf-8")
    assert f"path: {dataset_root.resolve()}" in content
    assert "train: images/train" in content
    assert "val: images/val" in content
    assert "test: images/test" in content
    assert "0: Salmon" in content
    assert "1: Tuna" in content
