"""Exa (exa.ai) neural search scraper.

Pulls recent web results from the Exa search API
(https://api.exa.ai/search) for a configured query and maps each result
into a ContentItem so the rest of the Horizon pipeline (deduplication, AI
scoring, enrichment, summarization) treats them the same way as RSS,
Hacker News, or GDELT items.

Design notes:

* An API key is required. It is read from the environment variable named
  by ``config.api_key_env`` (default ``EXA_API_KEY``) and sent as the
  ``x-api-key`` request header. A missing/empty key short-circuits to an
  empty list without making a network call.
* The desired time window is expressed via the ``startPublishedDate`` /
  ``endPublishedDate`` JSON body fields, derived from the ``since`` arg and
  "now" respectively (same window concept as the GDELT scraper).
* Exa's plain search returns no article body by default, so ``content`` is
  left as ``None`` (mirroring the GDELT scraper).
* A single malformed result is skipped, not allowed to abort the batch.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, List, Optional
from urllib.parse import urlparse

import httpx

from .base import BaseScraper
from ..models import ContentItem, ExaConfig, SourceType

logger = logging.getLogger(__name__)


class ExaScraper(BaseScraper):
    """Scraper backed by the Exa search API."""

    SOURCE_TYPE = SourceType.EXA
    BASE_URL = "https://api.exa.ai/search"

    def __init__(self, config: ExaConfig, http_client: httpx.AsyncClient):
        """Initialize the scraper.

        Args:
            config: Exa source configuration.
            http_client: Shared async HTTP client.
        """
        super().__init__({"exa": config}, http_client)
        self.exa_config = config

    async def fetch(self, since: datetime) -> List[ContentItem]:
        """Fetch results from the Exa search API.

        Args:
            since: Only fetch items published after this time (used to derive
                the ``startPublishedDate`` body field).

        Returns:
            List[ContentItem]: Fetched content items.
        """
        if not self.exa_config.enabled:
            return []

        query = (self.exa_config.query or "").strip()
        if not query:
            return []

        api_key = (os.getenv(self.exa_config.api_key_env) or "").strip()
        if not api_key:
            logger.warning(
                "Exa API key (%s) not set; skipping Exa scraper.",
                self.exa_config.api_key_env,
            )
            return []

        since_utc = self._ensure_utc(since)
        now_utc = datetime.now(timezone.utc)
        body: dict[str, Any] = {
            "query": query,
            "startPublishedDate": since_utc.isoformat(),
            "endPublishedDate": now_utc.isoformat(),
            "numResults": self.exa_config.num_results,
            "type": "auto",
        }
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }

        try:
            response = await self.client.post(
                self.BASE_URL, json=body, headers=headers, follow_redirects=True
            )
            response.raise_for_status()

            try:
                payload = response.json()
            except Exception as exc:
                logger.warning("Exa returned a non-JSON body: %s", exc)
                return []

            if not isinstance(payload, dict):
                return []

            results = payload.get("results")
            if not results:
                return []

            items: List[ContentItem] = []
            for raw in results:
                item = self._raw_to_item(raw)
                if item is not None:
                    items.append(item)
            return items

        except httpx.HTTPError as exc:
            logger.warning("Error fetching Exa results: %s", exc)
            return []
        except Exception as exc:
            logger.warning("Error parsing Exa response: %s", exc)
            return []

    def _raw_to_item(self, raw: Any) -> Optional[ContentItem]:
        """Map one Exa result record into a ContentItem.

        Returns None when the record has no URL/title or an unparseable
        ``publishedDate`` (published_at is required), so a single bad result
        is skipped rather than aborting the batch.
        """
        if not isinstance(raw, dict):
            return None

        url = (raw.get("url") or "").strip()
        title = (raw.get("title") or "").strip()
        if not url or not title:
            return None

        published = self._parse_published(raw.get("publishedDate"))
        if published is None:
            return None

        author = (raw.get("author") or "").strip() or None
        domain = self._domain(url)
        native_id = str(raw.get("id") or url)

        meta = {
            "domain": domain,
            "author": author,
            "query": self.exa_config.query,
            "category": self.exa_config.category,
        }

        try:
            return ContentItem(
                id=self._generate_id(self.SOURCE_TYPE.value, "search", native_id),
                source_type=self.SOURCE_TYPE,
                title=title,
                url=url,
                content=None,
                author=author,
                published_at=published,
                metadata={k: v for k, v in meta.items() if v is not None},
            )
        except Exception as exc:
            logger.warning("Skipping invalid Exa result %s: %s", url, exc)
            return None

    @staticmethod
    def _parse_published(value: Any) -> Optional[datetime]:
        """Parse an Exa ``publishedDate`` (ISO8601) into aware UTC."""
        if not value or not isinstance(value, str):
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _domain(url: str) -> Optional[str]:
        """Extract the host portion of a URL for a sub-source label."""
        try:
            host = urlparse(url).netloc
        except (ValueError, TypeError):
            return None
        return host or None

    @staticmethod
    def _ensure_utc(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)
