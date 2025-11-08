from __future__ import annotations

from pathlib import Path

import pytest

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing.auto_label import (
    Detection,
    build_label_studio_task,
    detection_to_label_studio_result,
    ensure_output_dir,
)


def test_detection_to_label_studio_result_converts_percentages() -> None:
    detection = Detection(cls=0, conf=0.9, x_center=0.5, y_center=0.5, width=0.4, height=0.2)
    result = detection_to_label_studio_result(
        detection,
        class_names=["Salmon"],
        original_width=640,
        original_height=480,
    )

    assert result["value"]["rectanglelabels"] == ["Salmon"]
    assert pytest.approx(result["value"]["x"], rel=1e-3) == 30.0
    assert pytest.approx(result["value"]["y"], rel=1e-3) == 40.0
    assert pytest.approx(result["value"]["width"], rel=1e-3) == 40.0
    assert pytest.approx(result["value"]["height"], rel=1e-3) == 20.0


def test_build_label_studio_task_filters_by_confidence() -> None:
    detections = [
        Detection(cls=0, conf=0.95, x_center=0.5, y_center=0.5, width=0.2, height=0.2),
        Detection(cls=0, conf=0.1, x_center=0.5, y_center=0.5, width=0.5, height=0.5),
    ]
    task = build_label_studio_task(
        "image.jpg",
        detections,
        class_names=["Tuna"],
        original_width=640,
        original_height=480,
        min_confidence=0.5,
    )

    assert task["data"]["image"] == "image.jpg"
    assert len(task["annotations"][0]["result"]) == 1
    assert task["annotations"][0]["result"][0]["score"] == pytest.approx(0.95)


def test_ensure_output_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "file.json"
    ensure_output_dir(target)

    assert target.parent.exists()
