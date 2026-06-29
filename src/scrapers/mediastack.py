"""Mediastack scraper.

Pulls recent news articles from the Mediastack news API
(http://api.mediastack.com/v1/news) for the configured keywords and maps
each article into a ContentItem so the rest of the Horizon pipeline
(deduplication, AI scoring, enrichment, summarization) treats them the same
way as RSS, Hacker News, GDELT, or NewsAPI items.

Design notes:

* An API key is required. It is read from the environment variable named by
  ``config.api_key_env`` (default ``MEDIASTACK_API_KEY``) and sent as the
  ``access_key`` query parameter. A missing/empty key yields an empty list
  and a warning rather than a network call or a crash.
* The free Mediastack tier is HTTP-only (no HTTPS), so the base URL uses
  plain ``http://`` deliberately.
* The desired time window is expressed via the ``date`` request parameter as
  a comma-separated ``YYYY-MM-DD,YYYY-MM-DD`` range built from ``since`` (the
  start) and now (the end), mirroring how the other news scrapers bound their
  result set off the ``since`` argument.
* Mediastack signals failures with an HTTP 200 response whose JSON body holds
  an ``error`` key (rather than a non-2xx status), so the body is inspected
  defensively: an ``error`` key or a missing/empty ``data`` list yields an
  empty list rather than raising.
* A single malformed article is skipped, not allowed to abort the batch.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, List, Optional

import httpx

from .base import BaseScraper
from ..models import ContentItem, MediastackConfig, SourceType

logger = logging.getLogger(__name__)


class MediastackScraper(BaseScraper):
    """Scraper backed by the Mediastack news API."""

    SOURCE_TYPE = SourceType.MEDIASTACK
    # Free-tier Mediastack is HTTP-only; HTTPS is a paid feature.
    BASE_URL = "http://api.mediastack.com/v1/news"

    def __init__(self, config: MediastackConfig, http_client: httpx.AsyncClient):
        """Initialize the scraper.

        Args:
            config: Mediastack source configuration.
            http_client: Shared async HTTP client.
        """
        super().__init__({"mediastack": config}, http_client)
        self.mediastack_config = config

    async def fetch(self, since: datetime) -> List[ContentItem]:
        """Fetch articles from the Mediastack news API.

        Args:
            since: Only fetch items published after this time (used to derive
                the start of the Mediastack ``date`` range parameter).

        Returns:
            List[ContentItem]: Fetched content items.
        """
        if not self.mediastack_config.enabled:
            return []

        keywords = (self.mediastack_config.keywords or "").strip()
        if not keywords:
            return []

        api_key = os.getenv(self.mediastack_config.api_key_env)
        if not api_key:
            logger.warning(
                "Mediastack key missing (env %s); skipping Mediastack fetch",
                self.mediastack_config.api_key_env,
            )
            return []

        since_utc = self._ensure_utc(since)
        now_utc = datetime.now(timezone.utc)
        # Mediastack expects a comma-separated YYYY-MM-DD,YYYY-MM-DD range.
        date_range = (
            f"{since_utc.strftime('%Y-%m-%d')},{now_utc.strftime('%Y-%m-%d')}"
        )

        params: dict[str, Any] = {
            "access_key": api_key,
            "keywords": keywords,
            "languages": self.mediastack_config.languages,
            "limit": self.mediastack_config.limit,
            "sort": "published_desc",
            "date": date_range,
        }

        try:
            response = await self.client.get(
                self.BASE_URL, params=params, follow_redirects=True
            )
            response.raise_for_status()

            try:
                payload = response.json()
            except Exception as exc:
                logger.warning("Mediastack returned a non-JSON body: %s", exc)
                return []

            if not isinstance(payload, dict):
                return []

            # Mediastack reports errors as a JSON body (HTTP 200) with an
            # ``error`` key; treat that as an empty result.
            if payload.get("error"):
                logger.warning(
                    "Mediastack returned an error body: %s", payload.get("error")
                )
                return []

            articles = payload.get("data")
            if not articles:
                return []

            items: List[ContentItem] = []
            for raw in articles:
                item = self._raw_to_item(raw)
                if item is not None:
                    items.append(item)
            return items

        except httpx.HTTPError as exc:
            logger.warning("Error fetching Mediastack articles: %s", exc)
            return []
        except Exception as exc:
            logger.warning("Error parsing Mediastack response: %s", exc)
            return []

    def _raw_to_item(self, raw: Any) -> Optional[ContentItem]:
        """Map one Mediastack article record into a ContentItem.

        Returns None when the record has no URL/title or an unparseable
        ``published_at`` (published_at is required), so a single bad article is
        skipped rather than aborting the batch.
        """
        if not isinstance(raw, dict):
            return None

        url = (raw.get("url") or "").strip()
        title = (raw.get("title") or "").strip()
        if not url or not title:
            return None

        published = self._parse_published(raw.get("published_at"))
        if published is None:
            return None

        meta = {
            "source": raw.get("source"),
            "keywords": self.mediastack_config.keywords,
            "category": self.mediastack_config.category,
        }

        try:
            return ContentItem(
                id=self._generate_id("mediastack", "article", url),
                source_type=self.SOURCE_TYPE,
                title=title,
                url=url,
                content=raw.get("description"),
                author=raw.get("author"),
                published_at=published,
                metadata={k: v for k, v in meta.items() if v is not None},
            )
        except Exception as exc:
            logger.warning("Skipping invalid Mediastack article %s: %s", url, exc)
            return None

    @staticmethod
    def _parse_published(value: Any) -> Optional[datetime]:
        """Parse a Mediastack ``published_at`` ISO8601 string into aware UTC."""
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
