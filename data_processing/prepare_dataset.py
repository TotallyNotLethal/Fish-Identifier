"""Utilities for cleaning raw image data and producing train/val/test splits.

The script scans ``data/raw`` for image files, removes duplicates (based on
SHA256 hashes), filters out low-resolution assets, and then copies the remaining
images into ``data/processed/images/{train,val,test}`` folders.

Usage
-----
```
python data_processing/prepare_dataset.py \
    --raw-dir data/raw \
    --output-dir data/processed/images \
    --min-width 256 --min-height 256 \
    --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
```

The script prints a summary of actions performed, making it easy to wire into
an automated data preparation workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    """Represents a unique image discovered during dataset preparation."""

    path: Path
    hash: str
    width: int
    height: int

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def iter_image_paths(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            yield path


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_image_metadata(path: Path) -> ImageRecord:
    with Image.open(path) as img:
        width, height = img.size
    return ImageRecord(path=path, hash=compute_sha256(path), width=width, height=height)


def deduplicate_records(records: Iterable[ImageRecord]) -> Dict[str, ImageRecord]:
    unique: Dict[str, ImageRecord] = {}
    for record in records:
        if record.hash not in unique:
            unique[record.hash] = record
    return unique


def filter_by_resolution(records: Iterable[ImageRecord], *, min_width: int, min_height: int) -> List[ImageRecord]:
    return [r for r in records if r.width >= min_width and r.height >= min_height]


def split_records(records: Sequence[ImageRecord], *, train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[ImageRecord]]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    records = list(records)
    random.Random(seed).shuffle(records)
    total = len(records)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }


def copy_records_to_split(split_map: Dict[str, List[ImageRecord]], output_dir: Path) -> None:
    for split_name, records in split_map.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            destination = split_dir / record.path.name
            if destination.exists():
                # If a collision occurs we append a counter to retain both files.
                stem = destination.stem
                suffix = destination.suffix
                counter = 1
                while True:
                    alternative = split_dir / f"{stem}_{counter}{suffix}"
                    if not alternative.exists():
                        destination = alternative
                        break
                    counter += 1
            shutil.copy2(record.path, destination)


def summarize(records: Sequence[ImageRecord], filtered: Sequence[ImageRecord], split_map: Dict[str, List[ImageRecord]]) -> str:
    lines = ["Dataset preparation summary:"]
    lines.append(f"  Raw images discovered: {len(records)}")
    lines.append(f"  Unique images kept: {len(filtered)}")
    lines.append("  Split distribution:")
    for split_name, split_records in split_map.items():
        lines.append(f"    - {split_name}: {len(split_records)}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare image dataset by deduplicating, filtering, and splitting")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing raw images")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/images"), help="Directory to write split images")
    parser.add_argument("--min-width", type=int, default=256, help="Minimum width for images to keep")
    parser.add_argument("--min-height", type=int, default=256, help="Minimum height for images to keep")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir: Path = args.raw_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory {raw_dir} does not exist")

    image_paths = list(iter_image_paths(raw_dir))
    records = [load_image_metadata(path) for path in image_paths]
    unique_records = deduplicate_records(records)
    filtered_records = filter_by_resolution(unique_records.values(), min_width=args.min_width, min_height=args.min_height)

    split_map = split_records(filtered_records, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_records_to_split(split_map, args.output_dir)

    print(summarize(records, filtered_records, split_map))


if __name__ == "__main__":
    main()
