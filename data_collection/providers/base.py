"""Base classes for image providers."""

from __future__ import annotations

import abc
from typing import Iterable, List, Tuple, Dict

from ..http import AsyncHTTPClient
from ..models import ImageResult


class ImageProvider(abc.ABC):
    """Abstract interface for provider-specific search clients."""

    name: str

    def __init__(self, http: AsyncHTTPClient) -> None:
        self.http = http

    @abc.abstractmethod
    async def search(self, species: str, max_results: int) -> List[ImageResult]:
        """Return image candidates for the provided species."""

    async def enrich_keywords(self, species: str) -> Iterable[str]:
        """Hook for provider-specific keyword enrichment."""

        return (species,)

    async def download(self, result: ImageResult) -> Tuple[bytes, Dict[str, str]]:
        """Download the image binary for the given result."""

        return await self.http.fetch_bytes(result.url)
