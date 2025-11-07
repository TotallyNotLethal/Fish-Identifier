"""Google Custom Search image provider."""

from __future__ import annotations

import os
from typing import List
from urllib.parse import urlencode

from ..models import ImageResult
from ..http import AsyncHTTPClient
from .base import ImageProvider

_GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


class GoogleImageProvider(ImageProvider):
    name = "google"

    def __init__(
        self,
        http: AsyncHTTPClient,
        api_key: str | None = None,
        cse_id: str | None = None,
    ) -> None:
        super().__init__(http)
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._cse_id = cse_id or os.getenv("GOOGLE_CSE_ID")
        if not self._api_key or not self._cse_id:
            raise RuntimeError(
                "GoogleImageProvider requires GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables"
            )

    async def search(self, species: str, max_results: int) -> List[ImageResult]:
        params = {
            "q": species,
            "searchType": "image",
            "num": min(max_results, 10),
            "cx": self._cse_id,
            "key": self._api_key,
            "safe": "high",
        }
        query = f"{_GOOGLE_ENDPOINT}?{urlencode(params)}"
        payload = await self.http.fetch_json(query)
        items = payload.get("items", [])
        results: List[ImageResult] = []
        for item in items[:max_results]:
            image = item.get("image", {})
            results.append(
                ImageResult(
                    url=item.get("link"),
                    source_page=item.get("image", {}).get("contextLink"),
                    title=item.get("title"),
                    license=image.get("license"),
                    thumbnail_url=image.get("thumbnailLink"),
                    width=image.get("width"),
                    height=image.get("height"),
                    keywords=[species],
                    provider_payload=item,
                )
            )
        return results
