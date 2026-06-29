from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx

from src.models import NewsAPIConfig
from src.scrapers.newsapi import NewsAPIScraper


SINCE = datetime(2026, 6, 27, 8, 30, 0, tzinfo=timezone.utc)
API_KEY_ENV = "NEWSAPI_API_KEY"


def _mock_client(payload: dict | None = None) -> AsyncMock:
    response = MagicMock()
    response.json.return_value = payload if payload is not None else {}
    response.raise_for_status.return_value = None
    client = AsyncMock()
    client.get.return_value = response
    return client


def _articles_payload() -> dict:
    return {
        "status": "ok",
        "articles": [
            {
                "url": "https://example.com/ai-1",
                "title": "AI breakthrough one",
                "publishedAt": "2026-06-28T12:00:00Z",
                "source": {"id": "the-verge", "name": "The Verge"},
                "description": "A short description.",
                "content": "Full content one.",
                "author": "Jane Doe",
            },
            {
                "url": "https://news.test/ai-2",
                "title": "AI breakthrough two",
                "publishedAt": "2026-06-28T13:30:00Z",
                "source": {"id": None, "name": "News Test"},
                "description": None,
                "content": "Full content two.",
                "author": "John Roe",
            },
        ],
    }


def test_missing_api_key_returns_empty_without_network(monkeypatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    client = _mock_client(_articles_payload())
    config = NewsAPIConfig(enabled=True, query="ai")
    scraper = NewsAPIScraper(config, client)

    result = asyncio.run(scraper.fetch(SINCE))

    assert result == []
    client.get.assert_not_called()


def test_empty_api_key_returns_empty_without_network(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "")
    client = _mock_client(_articles_payload())
    config = NewsAPIConfig(enabled=True, query="ai")
    scraper = NewsAPIScraper(config, client)

    result = asyncio.run(scraper.fetch(SINCE))

    assert result == []
    client.get.assert_not_called()


def test_since_maps_to_from_param(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = _mock_client(_articles_payload())
    config = NewsAPIConfig(enabled=True, query="ai")
    scraper = NewsAPIScraper(config, client)

    asyncio.run(scraper.fetch(SINCE))

    params = client.get.call_args.kwargs["params"]
    assert params["from"] == SINCE.isoformat()
    assert "to" in params
    assert params["sortBy"] == "publishedAt"
    assert params["q"] == "ai"
    assert params["language"] == "en"
    assert params["pageSize"] == 100


def test_api_key_sent_via_header(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = _mock_client(_articles_payload())
    config = NewsAPIConfig(enabled=True, query="ai")
    scraper = NewsAPIScraper(config, client)

    asyncio.run(scraper.fetch(SINCE))

    headers = client.get.call_args.kwargs["headers"]
    assert headers["X-Api-Key"] == "secret-key"


def test_parses_articles_into_content_items(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = _mock_client(_articles_payload())
    config = NewsAPIConfig(enabled=True, query="ai")
    scraper = NewsAPIScraper(config, client)

    items = asyncio.run(scraper.fetch(SINCE))

    assert len(items) == 2
    first = items[0]
    assert str(first.url) == "https://example.com/ai-1"
    assert first.title == "AI breakthrough one"
    assert first.published_at == datetime(
        2026, 6, 28, 12, 0, 0, tzinfo=timezone.utc
    )
    assert first.author == "Jane Doe"
    assert first.content == "A short description."
    assert first.metadata["source_name"] == "The Verge"
    assert first.id == "newsapi:article:https://example.com/ai-1"

    # Falls back to `content` when `description` is missing.
    second = items[1]
    assert second.content == "Full content two."


def test_http_error_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = AsyncMock()
    client.get.side_effect = httpx.HTTPError("boom")
    config = NewsAPIConfig(enabled=True, query="ai")
    scraper = NewsAPIScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_empty_articles_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = _mock_client({"status": "ok", "articles": []})
    config = NewsAPIConfig(enabled=True, query="ai")
    scraper = NewsAPIScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_missing_articles_key_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = _mock_client({"status": "error"})
    config = NewsAPIConfig(enabled=True, query="ai")
    scraper = NewsAPIScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_disabled_config_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = _mock_client(_articles_payload())
    config = NewsAPIConfig(enabled=False, query="ai")
    scraper = NewsAPIScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []
    client.get.assert_not_called()


def test_empty_query_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = _mock_client(_articles_payload())
    config = NewsAPIConfig(enabled=True, query="   ")
    scraper = NewsAPIScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []
    client.get.assert_not_called()


def test_bad_article_is_skipped_not_crashing(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    payload = {
        "articles": [
            {  # missing url -> skipped
                "title": "No URL",
                "publishedAt": "2026-06-28T12:00:00Z",
                "source": {"name": "Example"},
            },
            {  # unparseable publishedAt -> skipped
                "url": "https://example.com/bad-date",
                "title": "Bad date",
                "publishedAt": "not-a-date",
                "source": {"name": "Example"},
            },
            {  # valid -> kept
                "url": "https://example.com/good",
                "title": "Good one",
                "publishedAt": "2026-06-28T14:00:00Z",
                "source": {"name": "Example"},
                "description": "ok",
                "author": "A",
            },
        ]
    }
    client = _mock_client(payload)
    config = NewsAPIConfig(enabled=True, query="ai")
    scraper = NewsAPIScraper(config, client)

    items = asyncio.run(scraper.fetch(SINCE))

    assert len(items) == 1
    assert str(items[0].url) == "https://example.com/good"


def test_non_json_body_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    response = MagicMock()
    response.json.side_effect = ValueError("no json")
    response.raise_for_status.return_value = None
    client = AsyncMock()
    client.get.return_value = response
    config = NewsAPIConfig(enabled=True, query="ai")
    scraper = NewsAPIScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []
