"""Asynchronous HTTP utilities with throttling, retry, and robots.txt support."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import aiohttp


@dataclass(frozen=True)
class ThrottleConfig:
    """Configuration for request throttling."""

    requests: int = 5
    period: float = 1.0
    concurrency: int = 5
    max_retries: int = 3
    backoff_factor: float = 1.5
    user_agent: str = "FishIdentifierBot/0.1"


class RobotsCache:
    """Caches robots.txt rules for hosts."""

    def __init__(self, session: aiohttp.ClientSession, user_agent: str) -> None:
        self._session = session
        self._user_agent = user_agent
        self._lock = asyncio.Lock()
        self._parsers: Dict[str, RobotFileParser] = {}
        self._fetching: Dict[str, asyncio.Future[RobotFileParser]] = {}

    async def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return True
        host = parsed.netloc
        async with self._lock:
            if host in self._parsers:
                parser = self._parsers[host]
                return parser.can_fetch(self._user_agent, url)
            if host in self._fetching:
                fut = self._fetching[host]
            else:
                fut = asyncio.get_event_loop().create_future()
                self._fetching[host] = fut
                asyncio.create_task(self._load_parser(host, parsed.scheme, fut))
            try:
                parser = await fut
            finally:
                if fut.done():
                    self._parsers[host] = fut.result()
                    self._fetching.pop(host, None)
        return parser.can_fetch(self._user_agent, url)

    async def _load_parser(
        self, host: str, scheme: str, fut: asyncio.Future[RobotFileParser]
    ) -> None:
        parser = RobotFileParser()
        robots_url = f"{scheme}://{host}/robots.txt"
        try:
            async with self._session.get(robots_url, timeout=10) as resp:
                if resp.status >= 400:
                    parser.parse([])
                else:
                    text = await resp.text()
                    parser.parse(text.splitlines())
        except Exception:
            parser.parse([])
        fut.set_result(parser)


class AsyncHTTPClient:
    """Wrapper around :class:`aiohttp.ClientSession` with throttling and retries."""

    def __init__(self, throttle: Optional[ThrottleConfig] = None) -> None:
        self._throttle = throttle or ThrottleConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._sem = asyncio.Semaphore(self._throttle.concurrency)
        self._timestamps: Deque[float] = deque()
        self._robots: Optional[RobotsCache] = None

    async def __aenter__(self) -> "AsyncHTTPClient":
        headers = {"User-Agent": self._throttle.user_agent}
        timeout = aiohttp.ClientTimeout(total=30)
        self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        self._robots = RobotsCache(self._session, self._throttle.user_agent)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()
        self._session = None
        self._robots = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            raise RuntimeError("AsyncHTTPClient must be used as an async context manager")
        return self._session

    async def _throttle_wait(self) -> None:
        now = time.monotonic()
        window = self._throttle.period
        while self._timestamps and now - self._timestamps[0] > window:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._throttle.requests:
            sleep_for = window - (now - self._timestamps[0])
            await asyncio.sleep(max(sleep_for, 0))
        self._timestamps.append(time.monotonic())

    async def _fetch_with_retries(
        self, method: str, url: str, **kwargs: Any
    ) -> aiohttp.ClientResponse:
        retries = 0
        backoff = self._throttle.backoff_factor
        while True:
            try:
                await self._throttle_wait()
                async with self._sem:
                    if self._robots and not await self._robots.can_fetch(url):
                        raise PermissionError(f"Blocked by robots.txt: {url}")
                    response = await self.session.request(method, url, **kwargs)
                    if response.status >= 500 and retries < self._throttle.max_retries:
                        await response.release()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message="server error",
                            headers=response.headers,
                        )
                    return response
            except (aiohttp.ClientError, asyncio.TimeoutError, PermissionError) as exc:
                retries += 1
                if retries > self._throttle.max_retries or isinstance(exc, PermissionError):
                    raise
                await asyncio.sleep(backoff ** retries)

    async def fetch_bytes(self, url: str, **kwargs: Any) -> Tuple[bytes, Dict[str, str]]:
        async with await self._fetch_with_retries("GET", url, **kwargs) as resp:
            resp.raise_for_status()
            return await resp.read(), dict(resp.headers)

    async def fetch_json(self, url: str, **kwargs: Any) -> Any:
        async with await self._fetch_with_retries("GET", url, **kwargs) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def fetch_text(self, url: str, **kwargs: Any) -> str:
        async with await self._fetch_with_retries("GET", url, **kwargs) as resp:
            resp.raise_for_status()
            return await resp.text()

    async def head(self, url: str, **kwargs: Any) -> Dict[str, str]:
        async with await self._fetch_with_retries("HEAD", url, **kwargs) as resp:
            resp.raise_for_status()
            return dict(resp.headers)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
            self._robots = None
