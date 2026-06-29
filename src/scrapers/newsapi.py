"""NewsAPI.org scraper.

Pulls recent news articles from the NewsAPI.org ``everything`` endpoint
(https://newsapi.org/v2/everything) for a configured query and maps each
article into a ContentItem so the rest of the Horizon pipeline
(deduplication, AI scoring, enrichment, summarization) treats them the same
way as RSS, Hacker News, or GDELT items.

Design notes:

* An API key is required. It is read from the environment variable named by
  ``config.api_key_env`` (default ``NEWSAPI_API_KEY``) and sent via the
  ``X-Api-Key`` request header. A missing/empty key yields an empty list and
  a warning rather than a network call or a crash.
* The desired time window is expressed via the ``from`` request parameter
  derived from ``since`` (ISO8601), with ``to`` set to now, mirroring how the
  other news scrapers bound their result set off the ``since`` argument.
* NewsAPI can return non-JSON or error bodies, so the ``.json()`` call is
  guarded and a missing/empty ``articles`` key yields an empty list rather
  than raising.
* A single malformed article is skipped, not allowed to abort the batch.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, List, Optional

import httpx

from .base import BaseScraper
from ..models import ContentItem, NewsAPIConfig, SourceType

logger = logging.getLogger(__name__)


class NewsAPIScraper(BaseScraper):
    """Scraper backed by the NewsAPI.org ``everything`` endpoint."""

    SOURCE_TYPE = SourceType.NEWSAPI
    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, config: NewsAPIConfig, http_client: httpx.AsyncClient):
        """Initialize the scraper.

        Args:
            config: NewsAPI source configuration.
            http_client: Shared async HTTP client.
        """
        super().__init__({"newsapi": config}, http_client)
        self.newsapi_config = config

    async def fetch(self, since: datetime) -> List[ContentItem]:
        """Fetch articles from the NewsAPI.org ``everything`` endpoint.

        Args:
            since: Only fetch items published after this time (used to derive
                the NewsAPI ``from`` parameter).

        Returns:
            List[ContentItem]: Fetched content items.
        """
        if not self.newsapi_config.enabled:
            return []

        query = (self.newsapi_config.query or "").strip()
        if not query:
            return []

        api_key = os.getenv(self.newsapi_config.api_key_env)
        if not api_key:
            logger.warning(
                "NewsAPI key missing (env %s); skipping NewsAPI fetch",
                self.newsapi_config.api_key_env,
            )
            return []

        since_utc = self._ensure_utc(since)
        now_utc = datetime.now(timezone.utc)
        params: dict[str, Any] = {
            "q": query,
            "language": self.newsapi_config.language,
            "pageSize": self.newsapi_config.page_size,
            "sortBy": "publishedAt",
            "from": since_utc.isoformat(),
            "to": now_utc.isoformat(),
        }
        headers = {"X-Api-Key": api_key}

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
                logger.warning("NewsAPI returned a non-JSON body: %s", exc)
                return []

            if not isinstance(payload, dict):
                return []

            articles = payload.get("articles")
            if not articles:
                return []

            items: List[ContentItem] = []
            for raw in articles:
                item = self._raw_to_item(raw)
                if item is not None:
                    items.append(item)
            return items

        except httpx.HTTPError as exc:
            logger.warning("Error fetching NewsAPI articles: %s", exc)
            return []
        except Exception as exc:
            logger.warning("Error parsing NewsAPI response: %s", exc)
            return []

    def _raw_to_item(self, raw: Any) -> Optional[ContentItem]:
        """Map one NewsAPI article record into a ContentItem.

        Returns None when the record has no URL/title or an unparseable
        ``publishedAt`` (published_at is required), so a single bad article is
        skipped rather than aborting the batch.
        """
        if not isinstance(raw, dict):
            return None

        url = (raw.get("url") or "").strip()
        title = (raw.get("title") or "").strip()
        if not url or not title:
            return None

        published = self._parse_published(raw.get("publishedAt"))
        if published is None:
            return None

        source = raw.get("source")
        source_name = None
        if isinstance(source, dict):
            source_name = source.get("name")

        content = raw.get("description") or raw.get("content")

        meta = {
            "source_name": source_name,
            "query": self.newsapi_config.query,
            "category": self.newsapi_config.category,
        }

        try:
            return ContentItem(
                id=self._generate_id("newsapi", "article", url),
                source_type=self.SOURCE_TYPE,
                title=title,
                url=url,
                content=content,
                author=raw.get("author"),
                published_at=published,
                metadata={k: v for k, v in meta.items() if v is not None},
            )
        except Exception as exc:
            logger.warning("Skipping invalid NewsAPI article %s: %s", url, exc)
            return None

    @staticmethod
    def _parse_published(value: Any) -> Optional[datetime]:
        """Parse a NewsAPI ``publishedAt`` ISO8601 string into aware UTC."""
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
