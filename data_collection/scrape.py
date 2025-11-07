"""Command line interface for refreshing fish image datasets."""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .http import AsyncHTTPClient, ThrottleConfig
from .logging_utils import build_audit_logger
from .models import DownloadRecord, ImageResult
from .providers import PROVIDERS, create_provider
from .providers.base import ImageProvider
from .storage import (
    append_metadata,
    ensure_directory,
    get_metadata_path,
    guess_extension,
    metadata_payload,
    save_image,
)

LOGGER = logging.getLogger("data_collection.scrape")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species-list",
        type=Path,
        default=Path("data/species.txt"),
        help="Path to a newline-delimited file of species names.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS.keys()),
        default="duckduckgo",
        help="Image provider to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save downloaded images and metadata.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("data/logs"),
        help="Directory to write audit logs.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of images to download per species.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Concurrent HTTP requests allowed.",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=5,
        help="Maximum requests per period (see --period).",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=1.0,
        help="Rate limiting window in seconds.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default="FishIdentifierBot/0.1",
        help="Custom user agent for HTTP requests.",
    )
    return parser.parse_args(argv)


def read_species(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Species list not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def slugify(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in value.lower())
    safe = "-".join(filter(None, safe.split("-")))
    return safe or "item"


@dataclass(slots=True)
class DownloadContext:
    provider: ImageProvider
    species: str
    species_dir: Path
    provider_name: str


async def download_single(result: ImageResult, idx: int, ctx: DownloadContext) -> DownloadRecord:
    filename = f"{idx:04d}"
    try:
        data, headers = await ctx.provider.download(result)
        extension = guess_extension(result.url, headers)
        target = save_image(ctx.species_dir, f"{filename}.{extension}", data)
        append_metadata(
            get_metadata_path(ctx.species_dir),
            metadata_payload(result, target, ctx.species, ctx.provider_name),
        )
        status = headers.get("Content-Type", "unknown") if isinstance(headers, dict) else "ok"
        return DownloadRecord(
            species=ctx.species,
            provider=ctx.provider_name,
            url=result.url,
            saved_path=target,
            success=True,
            status=status,
            license=result.license,
            source_page=result.source_page,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to download %s from %s", ctx.species, result.url)
        return DownloadRecord(
            species=ctx.species,
            provider=ctx.provider_name,
            url=result.url,
            saved_path=None,
            success=False,
            status=str(exc),
            license=result.license,
            source_page=result.source_page,
        )


async def process_species(
    provider: ImageProvider,
    species: str,
    max_results: int,
    output_root: Path,
) -> Sequence[DownloadRecord]:
    species_slug = slugify(species)
    species_dir = output_root / provider.name / species_slug
    ensure_directory(species_dir)
    LOGGER.info("Searching for %s (%s)", species, provider.name)
    results = await provider.search(species, max_results)
    ctx = DownloadContext(
        provider=provider,
        species=species,
        species_dir=species_dir,
        provider_name=provider.name,
    )
    tasks = [download_single(result, idx, ctx) for idx, result in enumerate(results)]
    if not tasks:
        return []
    records = await asyncio.gather(*tasks)
    return records


async def run(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    throttle = ThrottleConfig(
        requests=args.rate,
        period=args.period,
        concurrency=args.concurrency,
        user_agent=args.user_agent,
    )
    species_list = read_species(args.species_list)
    audit_logger, jsonl_path, csv_path = build_audit_logger(args.log_dir, args.provider)
    LOGGER.info("Audit log JSONL: %s", jsonl_path)
    LOGGER.info("Audit log CSV: %s", csv_path)
    async with AsyncHTTPClient(throttle) as http:
        try:
            provider = create_provider(args.provider, http)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to initialise provider '%s': %s", args.provider, exc)
            return 2
        all_records: List[DownloadRecord] = []
        for species in species_list:
            records = await process_species(provider, species, args.max_results, args.output_dir)
            audit_logger.log_many(records)
            all_records.extend(records)
    successes = sum(1 for record in all_records if record.success)
    failures = len(all_records) - successes
    LOGGER.info("Completed run with %d successes and %d failures", successes, failures)
    return 0 if failures == 0 else 1


def main(argv: Iterable[str] | None = None) -> int:
    with suppress(KeyboardInterrupt):
        return asyncio.run(run(argv))
    return 130


if __name__ == "__main__":
    raise SystemExit(main())
