"""NewsData.io scraper.

Pulls recent news articles from the NewsData.io free ``latest`` endpoint
(https://newsdata.io/api/1/latest) for a configured query and maps each
article into a ContentItem so the rest of the Horizon pipeline
(deduplication, AI scoring, enrichment, summarization) treats them the same
way as RSS, Hacker News, GDELT, or NewsAPI items.

Design notes:

* An API key is required. It is read from the environment variable named by
  ``config.api_key_env`` (default ``NEWSDATA_API_KEY``) and sent via the
  ``apikey`` query parameter. A missing/empty key yields an empty list and a
  warning rather than a network call or a crash.
* Only the free ``latest`` endpoint is used. The paid Archive/date-range
  endpoint is intentionally NOT used. The free ``latest`` endpoint has NO
  server-side date filtering, so results are fetched and then filtered
  CLIENT-SIDE to keep only records whose ``pubDate`` is >= the ``since`` arg.
  Note: NewsData's free tier typically lags real time by ~12 hours, so very
  recent articles may not appear until that delay has elapsed.
* NewsData can return non-JSON or error bodies, so the ``.json()`` call is
  guarded and a missing/empty ``results`` key yields an empty list rather
  than raising.
* A single malformed record is skipped, not allowed to abort the batch.
  Records with an unparseable ``pubDate`` are skipped (published_at is
  required and the client-side date filter cannot be applied without it).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, List, Optional

import httpx

from .base import BaseScraper
from ..models import ContentItem, NewsDataConfig, SourceType

logger = logging.getLogger(__name__)


class NewsDataScraper(BaseScraper):
    """Scraper backed by the NewsData.io free ``latest`` endpoint."""

    SOURCE_TYPE = SourceType.NEWSDATA
    BASE_URL = "https://newsdata.io/api/1/latest"

    def __init__(self, config: NewsDataConfig, http_client: httpx.AsyncClient):
        """Initialize the scraper.

        Args:
            config: NewsData.io source configuration.
            http_client: Shared async HTTP client.
        """
        super().__init__({"newsdata": config}, http_client)
        self.newsdata_config = config

    async def fetch(self, since: datetime) -> List[ContentItem]:
        """Fetch articles from the NewsData.io free ``latest`` endpoint.

        The free ``latest`` endpoint offers no server-side date filtering, so
        records are filtered CLIENT-SIDE: only those whose ``pubDate`` is
        >= ``since`` are returned.

        Args:
            since: Only keep items published at/after this time.

        Returns:
            List[ContentItem]: Fetched content items.
        """
        if not self.newsdata_config.enabled:
            return []

        query = (self.newsdata_config.query or "").strip()
        if not query:
            return []

        api_key = os.getenv(self.newsdata_config.api_key_env)
        if not api_key:
            logger.warning(
                "NewsData key missing (env %s); skipping NewsData fetch",
                self.newsdata_config.api_key_env,
            )
            return []

        since_utc = self._ensure_utc(since)
        params: dict[str, Any] = {
            "apikey": api_key,
            "q": query,
            "language": self.newsdata_config.language,
        }

        try:
            response = await self.client.get(
                self.BASE_URL, params=params, follow_redirects=True
            )
            response.raise_for_status()

            try:
                payload = response.json()
            except Exception as exc:
                logger.warning("NewsData returned a non-JSON body: %s", exc)
                return []

            if not isinstance(payload, dict):
                return []

            results = payload.get("results")
            if not results:
                return []

            items: List[ContentItem] = []
            for raw in results:
                item = self._raw_to_item(raw, since_utc)
                if item is not None:
                    items.append(item)
            return items

        except httpx.HTTPError as exc:
            logger.warning("Error fetching NewsData articles: %s", exc)
            return []
        except Exception as exc:
            logger.warning("Error parsing NewsData response: %s", exc)
            return []

    def _raw_to_item(self, raw: Any, since_utc: datetime) -> Optional[ContentItem]:
        """Map one NewsData record into a ContentItem.

        Returns None when the record has no URL/title, has an unparseable
        ``pubDate`` (published_at is required and the client-side date filter
        needs it), or its ``pubDate`` is older than ``since_utc`` (CLIENT-SIDE
        date filter), so a single bad/old record is skipped rather than
        aborting the batch.
        """
        if not isinstance(raw, dict):
            return None

        url = (raw.get("link") or "").strip()
        title = (raw.get("title") or "").strip()
        if not url or not title:
            return None

        published = self._parse_pubdate(raw.get("pubDate"))
        if published is None:
            # Unparseable pubDate -> skip (cannot apply the date filter and
            # published_at is required).
            return None

        # CLIENT-SIDE date filter: the free `latest` endpoint has no
        # server-side date filtering, so drop anything older than `since`.
        if published < since_utc:
            return None

        # creator is an optional author list; use the first element if present.
        author = None
        creator = raw.get("creator")
        if isinstance(creator, list) and creator:
            author = creator[0]
        elif isinstance(creator, str):
            author = creator

        meta = {
            "source_id": raw.get("source_id"),
            "query": self.newsdata_config.query,
            "category": self.newsdata_config.category,
            "language": self.newsdata_config.language,
        }

        native_id = raw.get("article_id") or url

        try:
            return ContentItem(
                id=self._generate_id("newsdata", "article", str(native_id)),
                source_type=self.SOURCE_TYPE,
                title=title,
                url=url,
                content=raw.get("description"),
                author=author,
                published_at=published,
                metadata={k: v for k, v in meta.items() if v is not None},
            )
        except Exception as exc:
            logger.warning("Skipping invalid NewsData record %s: %s", url, exc)
            return None

    @staticmethod
    def _parse_pubdate(value: Any) -> Optional[datetime]:
        """Parse a NewsData ``pubDate`` ("YYYY-MM-DD HH:MM:SS") into aware UTC.

        NewsData returns ``pubDate`` in UTC; values without timezone info are
        assumed UTC. ISO8601 (with ``T``/``Z``) is also tolerated as a
        fallback.
        """
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None

        dt: Optional[datetime] = None
        try:
            dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
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
