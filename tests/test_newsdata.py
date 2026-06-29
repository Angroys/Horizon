from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.models import NewsDataConfig
from src.scrapers.newsdata import NewsDataScraper


SINCE = datetime(2026, 6, 27, 8, 30, 0, tzinfo=timezone.utc)


def _mock_client(payload: dict | None = None) -> AsyncMock:
    response = MagicMock()
    response.json.return_value = payload if payload is not None else {}
    response.raise_for_status.return_value = None
    client = AsyncMock()
    client.get.return_value = response
    return client


def _results_payload() -> dict:
    return {
        "status": "success",
        "results": [
            {
                "article_id": "abc123",
                "link": "https://example.com/ai-1",
                "title": "AI breakthrough one",
                "pubDate": "2026-06-28 12:00:00",
                "source_id": "example",
                "description": "A description of the first article.",
                "creator": ["Jane Reporter"],
            },
            {
                "article_id": "def456",
                "link": "https://news.test/ai-2",
                "title": "AI breakthrough two",
                "pubDate": "2026-06-28 13:30:00",
                "source_id": "newstest",
                "description": "A description of the second article.",
            },
        ],
    }


def test_missing_key_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEWSDATA_API_KEY", raising=False)
    client = _mock_client(_results_payload())
    config = NewsDataConfig(enabled=True, query="ai")
    scraper = NewsDataScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []
    client.get.assert_not_called()  # no network when key is missing


def test_parses_results_into_content_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWSDATA_API_KEY", "secret")
    client = _mock_client(_results_payload())
    config = NewsDataConfig(enabled=True, query="ai")
    scraper = NewsDataScraper(config, client)

    items = asyncio.run(scraper.fetch(SINCE))

    assert len(items) == 2
    first = items[0]
    assert str(first.url) == "https://example.com/ai-1"
    assert first.title == "AI breakthrough one"
    assert first.published_at == datetime(
        2026, 6, 28, 12, 0, 0, tzinfo=timezone.utc
    )
    assert first.author == "Jane Reporter"
    assert first.content == "A description of the first article."
    assert first.metadata["source_id"] == "example"
    assert first.id.startswith("newsdata:article:")

    # auth + query params sent correctly
    params = client.get.call_args.kwargs["params"]
    assert params["apikey"] == "secret"
    assert params["q"] == "ai"
    assert params["language"] == "en"


def test_client_side_date_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEWSDATA_API_KEY", "secret")
    payload = {
        "results": [
            {  # after since -> kept
                "article_id": "keep-1",
                "link": "https://example.com/new",
                "title": "Newer article",
                "pubDate": "2026-06-28 09:00:00",
                "source_id": "example",
            },
            {  # before since -> dropped
                "article_id": "drop-1",
                "link": "https://example.com/old",
                "title": "Older article",
                "pubDate": "2026-06-20 09:00:00",
                "source_id": "example",
            },
            {  # exactly at since -> kept (>= since)
                "article_id": "keep-2",
                "link": "https://example.com/edge",
                "title": "Edge article",
                "pubDate": "2026-06-27 08:30:00",
                "source_id": "example",
            },
        ]
    }
    client = _mock_client(payload)
    config = NewsDataConfig(enabled=True, query="ai")
    scraper = NewsDataScraper(config, client)

    items = asyncio.run(scraper.fetch(SINCE))

    urls = {str(item.url) for item in items}
    assert urls == {
        "https://example.com/new",
        "https://example.com/edge",
    }
    assert "https://example.com/old" not in urls


def test_disabled_config_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWSDATA_API_KEY", "secret")
    client = _mock_client(_results_payload())
    config = NewsDataConfig(enabled=False, query="ai")
    scraper = NewsDataScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_http_error_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEWSDATA_API_KEY", "secret")
    client = AsyncMock()
    client.get.side_effect = httpx.HTTPError("boom")
    config = NewsDataConfig(enabled=True, query="ai")
    scraper = NewsDataScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_missing_results_key_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWSDATA_API_KEY", "secret")
    client = _mock_client({"status": "success"})
    config = NewsDataConfig(enabled=True, query="ai")
    scraper = NewsDataScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_empty_query_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEWSDATA_API_KEY", "secret")
    client = _mock_client(_results_payload())
    config = NewsDataConfig(enabled=True, query="   ")
    scraper = NewsDataScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_bad_record_is_skipped_not_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWSDATA_API_KEY", "secret")
    payload = {
        "results": [
            {  # missing link -> skipped
                "title": "No URL",
                "pubDate": "2026-06-28 12:00:00",
                "source_id": "example",
            },
            {  # unparseable pubDate -> skipped
                "link": "https://example.com/bad-date",
                "title": "Bad date",
                "pubDate": "not-a-date",
                "source_id": "example",
            },
            {  # valid -> kept
                "link": "https://example.com/good",
                "title": "Good one",
                "pubDate": "2026-06-28 14:00:00",
                "source_id": "example",
            },
        ]
    }
    client = _mock_client(payload)
    config = NewsDataConfig(enabled=True, query="ai")
    scraper = NewsDataScraper(config, client)

    items = asyncio.run(scraper.fetch(SINCE))

    assert len(items) == 1
    assert str(items[0].url) == "https://example.com/good"


def test_non_json_body_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWSDATA_API_KEY", "secret")
    response = MagicMock()
    response.json.side_effect = ValueError("no json")
    response.raise_for_status.return_value = None
    client = AsyncMock()
    client.get.return_value = response
    config = NewsDataConfig(enabled=True, query="ai")
    scraper = NewsDataScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []
