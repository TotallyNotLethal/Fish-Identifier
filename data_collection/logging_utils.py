"""Audit logging helpers for data collection runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

from .models import DownloadRecord
from .storage import append_csv_row, ensure_directory


class AuditLogger:
    """Writes download results to JSONL and CSV for later auditing."""

    def __init__(self, jsonl_path: Path, csv_path: Path) -> None:
        self._jsonl_path = jsonl_path
        self._csv_path = csv_path
        ensure_directory(self._jsonl_path.parent)

    def log(self, record: DownloadRecord) -> None:
        payload = record.to_dict()
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
        header = (
            "timestamp",
            "provider",
            "species",
            "url",
            "saved_path",
            "success",
            "status",
            "license",
            "source_page",
        )
        append_csv_row(self._csv_path, payload, header)

    def log_many(self, records: Iterable[DownloadRecord]) -> None:
        for record in records:
            self.log(record)


def build_audit_logger(log_dir: Path, provider: str) -> Tuple[AuditLogger, Path, Path]:
    ensure_directory(log_dir)
    jsonl_path = log_dir / f"{provider}_downloads.jsonl"
    csv_path = log_dir / f"{provider}_downloads.csv"
    return AuditLogger(jsonl_path, csv_path), jsonl_path, csv_path
