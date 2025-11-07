"""Bing image search provider implementation."""

from __future__ import annotations

import os
from typing import List
from urllib.parse import urlencode

from ..http import AsyncHTTPClient
from ..models import ImageResult
from .base import ImageProvider

BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/search"


class BingImageProvider(ImageProvider):
    name = "bing"

    def __init__(self, http: AsyncHTTPClient, api_key: str | None = None) -> None:
        super().__init__(http)
        self._api_key = api_key or os.getenv("BING_IMAGE_SEARCH_KEY")
        if not self._api_key:
            raise RuntimeError(
                "BingImageProvider requires the BING_IMAGE_SEARCH_KEY environment variable"
            )

    async def search(self, species: str, max_results: int) -> List[ImageResult]:
        params = {
            "q": species,
            "count": max_results,
            "safeSearch": "Strict",
            "imageType": "Photo",
        }
        headers = {"Ocp-Apim-Subscription-Key": self._api_key}
        query = f"{BING_ENDPOINT}?{urlencode(params)}"
        payload = await self.http.fetch_json(query, headers=headers)
        results: List[ImageResult] = []
        for item in payload.get("value", [])[:max_results]:
            results.append(
                ImageResult(
                    url=item.get("contentUrl"),
                    source_page=item.get("hostPageUrl"),
                    title=item.get("name"),
                    license=item.get("contentLicense"),
                    thumbnail_url=item.get("thumbnailUrl"),
                    width=item.get("width"),
                    height=item.get("height"),
                    keywords=[species],
                    provider_payload=item,
                )
            )
        return results
