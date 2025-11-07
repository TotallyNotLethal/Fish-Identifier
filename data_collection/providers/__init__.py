"""Provider factory utilities."""

from __future__ import annotations

from typing import Dict, Type

from ..http import AsyncHTTPClient
from .base import ImageProvider
from .bing import BingImageProvider
from .duckduckgo import DuckDuckGoImageProvider
from .google import GoogleImageProvider

PROVIDERS: Dict[str, Type[ImageProvider]] = {
    BingImageProvider.name: BingImageProvider,
    DuckDuckGoImageProvider.name: DuckDuckGoImageProvider,
    GoogleImageProvider.name: GoogleImageProvider,
}


def create_provider(name: str, http: AsyncHTTPClient, **kwargs) -> ImageProvider:
    try:
        provider_cls = PROVIDERS[name.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown provider '{name}'. Available: {', '.join(PROVIDERS)}") from exc
    return provider_cls(http, **kwargs)


__all__ = ["ImageProvider", "create_provider", "PROVIDERS"]
