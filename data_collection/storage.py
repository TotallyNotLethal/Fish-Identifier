"""Helpers for writing image binaries and metadata to disk."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Dict, Optional, Tuple

from .models import ImageResult


def guess_extension(url: str, headers: Dict[str, str]) -> str:
    content_type = headers.get("Content-Type", "").split(";")[0].strip().lower()
    if content_type:
        ext = mimetypes.guess_extension(content_type)
        if ext:
            return ext.lstrip(".")
    ext = Path(url).suffix
    if ext:
        return ext.lstrip(".")
    return "jpg"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_image(output_dir: Path, filename: str, data: bytes) -> Path:
    ensure_directory(output_dir)
    file_path = output_dir / filename
    file_path.write_bytes(data)
    return file_path


def append_metadata(metadata_path: Path, payload: Dict[str, object]) -> None:
    ensure_directory(metadata_path.parent)
    with metadata_path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")


def get_metadata_path(base_dir: Path) -> Path:
    return base_dir / "metadata.jsonl"


def get_audit_paths(log_dir: Path, provider: str) -> Tuple[Path, Path]:
    ensure_directory(log_dir)
    jsonl = log_dir / f"{provider}_downloads.jsonl"
    csv = log_dir / f"{provider}_downloads.csv"
    return jsonl, csv


def append_csv_row(csv_path: Path, row: Dict[str, object], header: Optional[Tuple[str, ...]] = None) -> None:
    ensure_directory(csv_path.parent)
    write_header = not csv_path.exists() and header is not None
    with csv_path.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write(",".join(header) + "\n")
        values = [str(row.get(col, "")) for col in header or row.keys()]
        handle.write(",".join(values) + "\n")


def metadata_payload(result: ImageResult, saved_path: Path, species: str, provider: str) -> Dict[str, object]:
    payload = {
        "species": species,
        "provider": provider,
        "file": str(saved_path),
        "source_url": result.url,
        "source_page": result.source_page,
        "license": result.license,
        "title": result.title,
        "keywords": result.keywords,
        "width": result.width,
        "height": result.height,
        "thumbnail_url": result.thumbnail_url,
        "provider_payload": result.provider_payload,
    }
    return payload
