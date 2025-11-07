"""Dataclasses used across data collection modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class ImageResult:
    """Metadata about an image returned by a provider search."""

    url: str
    source_page: Optional[str] = None
    title: Optional[str] = None
    license: Optional[str] = None
    thumbnail_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    keywords: List[str] = field(default_factory=list)
    provider_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DownloadRecord:
    """Represents the outcome of an image download attempt."""

    species: str
    provider: str
    url: str
    saved_path: Optional[Path]
    success: bool
    status: str
    license: Optional[str]
    source_page: Optional[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a JSON serialisable dictionary."""

        data = {
            "species": self.species,
            "provider": self.provider,
            "url": self.url,
            "saved_path": str(self.saved_path) if self.saved_path else None,
            "success": self.success,
            "status": self.status,
            "license": self.license,
            "source_page": self.source_page,
            "timestamp": self.timestamp.isoformat() + "Z",
        }
        if self.extra:
            data.update(self.extra)
        return data
