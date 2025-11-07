"""Command line utility for training and evaluating Ultralytics YOLO12 models."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from ultralytics import YOLO


LOGGER = logging.getLogger("train_yolo12")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).with_name("yolo12_config.yaml")

    parser = argparse.ArgumentParser(
        description="Train and evaluate YOLO12 models with Ultralytics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to the training configuration YAML file.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "val", "predict"),
        default="train",
        help="Execution mode: training, validation, or prediction.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override the dataset YAML file path defined in the config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override the training image size.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights to initialize or evaluate.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume training from a checkpoint path. Pass 'true' to resume the last run "
            "saved by Ultralytics."
        ),
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Directory where training artifacts (checkpoints, metrics) are stored.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the training run. Determines the subdirectory inside the project path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g. '0' for first CUDA device, 'cpu').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of dataloader workers. Overrides config value if provided.",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=None,
        help="Save a checkpoint every N epochs. Disabled by default.",
    )
    parser.add_argument(
        "--predict-source",
        type=str,
        default=None,
        help="Source path or glob for prediction when --mode predict is used.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold override for validation or prediction.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold override for validation or prediction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to use for training reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )

    return parser.parse_args()


def to_bool_or_str(value: str | None) -> bool | str | None:
    """Convert string representations of booleans into bools while keeping paths intact."""
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return value


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    config = load_config(args.config)
    model_cfg: Dict[str, Any] = config.get("model", {})
    data_cfg: Dict[str, Any] = config.get("data", {})
    train_cfg: Dict[str, Any] = config.get("train", {}).copy()
    val_cfg: Dict[str, Any] = config.get("val", {}).copy()
    predict_cfg: Dict[str, Any] = config.get("predict", {}).copy()
    logging_cfg: Dict[str, Any] = config.get("logging", {})

    weights = args.weights or model_cfg.get("weights", "yolo12n.pt")
    imgsz = args.imgsz or model_cfg.get("imgsz")

    if args.batch_size is not None:
        train_cfg["batch"] = args.batch_size
        val_cfg["batch"] = args.batch_size
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if imgsz is not None:
        train_cfg.setdefault("imgsz", imgsz)
        val_cfg.setdefault("imgsz", imgsz)
        predict_cfg.setdefault("imgsz", imgsz)
    if args.save_period is not None:
        train_cfg["save_period"] = args.save_period
    if args.device is not None:
        train_cfg["device"] = args.device
        val_cfg["device"] = args.device
        predict_cfg["device"] = args.device
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.conf is not None:
        val_cfg["conf"] = args.conf
        predict_cfg["conf"] = args.conf
    if args.iou is not None:
        val_cfg["iou"] = args.iou
        predict_cfg["iou"] = args.iou
    if args.workers is not None:
        data_cfg["workers"] = args.workers

    resume = to_bool_or_str(args.resume)
    if resume is None:
        resume = train_cfg.get("resume", False)

    data_yaml = args.data or data_cfg.get("yaml")
    if not data_yaml:
        raise ValueError(
            "A dataset YAML path is required. Provide it in the config or via --data."
        )

    project_dir = args.project or logging_cfg.get("project_dir") or Path("artifacts/checkpoints")
    project_path = Path(project_dir)
    project_path.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or logging_cfg.get("run_name") or "yolo12"

    LOGGER.info("Loading YOLO12 model from %s", weights)
    model = YOLO(weights)

    # Prepare shared kwargs for Ultralytics calls.
    dataset_kwargs = {k: v for k, v in data_cfg.items() if k != "yaml"}

    if args.mode == "train":
        train_kwargs: Dict[str, Any] = {**dataset_kwargs, **train_cfg}
        train_kwargs.setdefault("imgsz", imgsz or model_cfg.get("imgsz", 640))
        train_kwargs["data"] = data_yaml
        LOGGER.info(
            "Starting training for %s epochs (batch=%s) -> %s/%s",
            train_kwargs.get("epochs"),
            train_kwargs.get("batch"),
            project_path,
            run_name,
        )
        model.train(
            project=str(project_path),
            name=run_name,
            resume=resume,
            **train_kwargs,
        )
    elif args.mode == "val":
        val_kwargs: Dict[str, Any] = {**dataset_kwargs, **val_cfg}
        val_kwargs["data"] = data_yaml
        LOGGER.info("Running validation on %s", data_yaml)
        model.val(
            project=str(project_path),
            name=run_name,
            **val_kwargs,
        )
    else:
        predict_source = args.predict_source
        if not predict_source:
            raise ValueError("--predict-source is required when --mode predict is used.")
        predict_kwargs: Dict[str, Any] = {**dataset_kwargs, **predict_cfg}
        predict_kwargs.setdefault("conf", 0.25)
        LOGGER.info("Running prediction on %s", predict_source)
        model.predict(
            source=predict_source,
            project=str(project_path),
            name=run_name,
            **predict_kwargs,
        )


if __name__ == "__main__":
    main()
