"""DuckDuckGo image search provider implemented via the public interface."""

from __future__ import annotations

import asyncio
import json
import re
from typing import List
from urllib.parse import urlencode

import aiohttp

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
        seen_urls: set[str | None] = set()
        next_url: str | None = f"{_DDG_IMAGE_API}?{urlencode(params)}"
        referer_url = f"{_DDG_PAGE}?{urlencode({'q': species})}"
        headers = {
            "Referer": referer_url,
            "User-Agent": self.http.user_agent,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9",
        }
        token_resets = 0
        max_token_resets = 3
        while next_url and len(results) < max_results:
            try:
                payload = await self.http.fetch_text(next_url, headers=headers)
            except aiohttp.ClientResponseError as exc:
                if exc.status == 403 and token_resets < max_token_resets:
                    token_resets += 1
                    await asyncio.sleep(1.0 * token_resets)
                    vqd = await self._fetch_vqd(species)
                    params["vqd"] = vqd
                    params.pop("s", None)
                    next_url = f"{_DDG_IMAGE_API}?{urlencode(params)}"
                    headers["Referer"] = referer_url
                    continue
                raise

            data = json.loads(payload)
            token_resets = 0
            for item in data.get("results", []):
                image_url = item.get("image")
                if image_url in seen_urls:
                    continue
                seen_urls.add(image_url)
                results.append(
                    ImageResult(
                        url=image_url,
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
                params["s"] = len(results)
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
