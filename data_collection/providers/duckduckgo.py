"""DuckDuckGo image search provider implemented via the public interface."""

from __future__ import annotations

import asyncio
import json
import re
from typing import List
from urllib.parse import urlencode

from ..http import AsyncHTTPClient
from ..models import ImageResult
from .base import ImageProvider

_DDG_PAGE = "https://duckduckgo.com/"
_DDG_IMAGE_API = "https://duckduckgo.com/i.js"
_VQD_RE = re.compile(r"vqd=(?P<quote>['\"])(?P<vqd>[-\w]+)(?P=quote)")


class DuckDuckGoImageProvider(ImageProvider):
    name = "duckduckgo"

    async def search(self, species: str, max_results: int) -> List[ImageResult]:
        vqd = await self._fetch_vqd(species)
        params = {
            "l": "us-en",
            "o": "json",
            "q": species,
            "vqd": vqd,
            "f": ",type:photo",
            "p": "1",
        }
        results: List[ImageResult] = []
        next_url: str | None = f"{_DDG_IMAGE_API}?{urlencode(params)}"
        while next_url and len(results) < max_results:
            payload = await self.http.fetch_text(
                next_url,
                headers={"Referer": f"{_DDG_PAGE}?{urlencode({'q': species})}"},
            )
            data = json.loads(payload)
            for item in data.get("results", []):
                results.append(
                    ImageResult(
                        url=item.get("image"),
                        source_page=item.get("url"),
                        title=item.get("title"),
                        license=item.get("license"),
                        thumbnail_url=item.get("thumbnail"),
                        width=item.get("width"),
                        height=item.get("height"),
                        keywords=[species],
                        provider_payload=item,
                    )
                )
                if len(results) >= max_results:
                    break
            next_url = data.get("next")
            if next_url:
                next_url = f"https://duckduckgo.com{next_url}"
            await asyncio.sleep(0.2)
        return results

    async def _fetch_vqd(self, query: str) -> str:
        referer = f"{_DDG_PAGE}?{urlencode({'q': query})}"
        page = await self.http.fetch_text(
            f"{_DDG_PAGE}?{urlencode({'q': query, 'iax': 'images', 'ia': 'images'})}",
            headers={
                "Referer": referer,
                "User-Agent": self.http.user_agent,
            },
        )
        match = _VQD_RE.search(page)
        if not match:
            raise RuntimeError("Could not extract DuckDuckGo vqd token")
        return match.group("vqd")
