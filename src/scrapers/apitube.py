"""APITube.io scraper.

Pulls recent news articles from the APITube.io ``everything`` endpoint
(https://api.apitube.io/v1/news/everything) for a configured query and maps
each article into a ContentItem so the rest of the Horizon pipeline
(deduplication, AI scoring, enrichment, summarization) treats them the same
way as RSS, Hacker News, GDELT, or NewsAPI items.

Design notes:

* An API key is required. It is read from the environment variable named by
  ``config.api_key_env`` (default ``APITUBE_API_KEY``). APITube accepts the
  key either as an ``api_key`` query parameter or via the ``X-API-Key``
  request header; we use the ``X-API-Key`` header so the secret never lands
  in URLs/logs. A missing/empty key yields an empty list and a warning rather
  than a network call or a crash.
* APITube uses ``title`` for keyword search, so the configured query is sent
  as the ``title`` parameter.
* The desired time window is expressed via ``published_at.start`` (derived
  from ``since``, ISO8601) and ``published_at.end`` (now), mirroring how the
  other news scrapers bound their result set off the ``since`` argument.
* The result array is normally under the ``results`` key, but some APITube
  responses place it under ``data``; both are read defensively, preferring
  ``results``.
* APITube can return non-JSON or error bodies, so the ``.json()`` call is
  guarded and a missing/empty result array yields an empty list rather than
  raising.
* A single malformed article is skipped, not allowed to abort the batch.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, List, Optional

import httpx

from .base import BaseScraper
from ..models import APITubeConfig, ContentItem, SourceType

logger = logging.getLogger(__name__)


class APITubeScraper(BaseScraper):
    """Scraper backed by the APITube.io ``everything`` endpoint."""

    SOURCE_TYPE = SourceType.APITUBE
    BASE_URL = "https://api.apitube.io/v1/news/everything"

    def __init__(self, config: APITubeConfig, http_client: httpx.AsyncClient):
        """Initialize the scraper.

        Args:
            config: APITube source configuration.
            http_client: Shared async HTTP client.
        """
        super().__init__({"apitube": config}, http_client)
        self.apitube_config = config

    async def fetch(self, since: datetime) -> List[ContentItem]:
        """Fetch articles from the APITube.io ``everything`` endpoint.

        Args:
            since: Only fetch items published after this time (used to derive
                the APITube ``published_at.start`` parameter).

        Returns:
            List[ContentItem]: Fetched content items.
        """
        if not self.apitube_config.enabled:
            return []

        query = (self.apitube_config.query or "").strip()
        if not query:
            return []

        api_key = os.getenv(self.apitube_config.api_key_env)
        if not api_key:
            logger.warning(
                "APITube key missing (env %s); skipping APITube fetch",
                self.apitube_config.api_key_env,
            )
            return []

        since_utc = self._ensure_utc(since)
        now_utc = datetime.now(timezone.utc)
        params: dict[str, Any] = {
            # APITube uses `title` for keyword search.
            "title": query,
            "language": self.apitube_config.language,
            "per_page": self.apitube_config.per_page,
            "published_at.start": since_utc.isoformat(),
            "published_at.end": now_utc.isoformat(),
            "sort.by": "published_at",
            "sort.order": "desc",
        }
        if self.apitube_config.category:
            params["category.name"] = self.apitube_config.category

        # Auth via header: APITube also accepts an `api_key` query param, but
        # we send the key in the X-API-Key header to keep it out of URLs/logs.
        headers = {"X-API-Key": api_key}

        try:
            response = await self.client.get(
                self.BASE_URL,
                params=params,
                headers=headers,
                follow_redirects=True,
            )
            response.raise_for_status()

            try:
                payload = response.json()
            except Exception as exc:
                logger.warning("APITube returned a non-JSON body: %s", exc)
                return []

            if not isinstance(payload, dict):
                return []

            # Results normally live under `results`; fall back to `data`.
            records = payload.get("results")
            if not records:
                records = payload.get("data")
            if not records:
                return []

            items: List[ContentItem] = []
            for raw in records:
                item = self._raw_to_item(raw)
                if item is not None:
                    items.append(item)
            return items

        except httpx.HTTPError as exc:
            logger.warning("Error fetching APITube articles: %s", exc)
            return []
        except Exception as exc:
            logger.warning("Error parsing APITube response: %s", exc)
            return []

    def _raw_to_item(self, raw: Any) -> Optional[ContentItem]:
        """Map one APITube article record into a ContentItem.

        Returns None when the record has no URL/title or an unparseable
        ``published_at`` (published_at is required), so a single bad article is
        skipped rather than aborting the batch.
        """
        if not isinstance(raw, dict):
            return None

        url = (raw.get("href") or raw.get("url") or "").strip()
        title = (raw.get("title") or "").strip()
        if not url or not title:
            return None

        published = self._parse_published(raw.get("published_at"))
        if published is None:
            return None

        source = raw.get("source")
        source_name = None
        source_domain = None
        if isinstance(source, dict):
            source_name = source.get("name")
            source_domain = source.get("domain")

        content = raw.get("description") or raw.get("content")

        meta = {
            "source_name": source_name,
            "domain": source_domain,
            "language": raw.get("language") or self.apitube_config.language,
            "query": self.apitube_config.query,
            "category": self.apitube_config.category,
        }

        try:
            return ContentItem(
                id=self._generate_id("apitube", "article", url),
                source_type=self.SOURCE_TYPE,
                title=title,
                url=url,
                content=content,
                author=source_name or raw.get("author"),
                published_at=published,
                metadata={k: v for k, v in meta.items() if v is not None},
            )
        except Exception as exc:
            logger.warning("Skipping invalid APITube article %s: %s", url, exc)
            return None

    @staticmethod
    def _parse_published(value: Any) -> Optional[datetime]:
        """Parse an APITube ``published_at`` ISO8601 string into aware UTC."""
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _ensure_utc(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)
