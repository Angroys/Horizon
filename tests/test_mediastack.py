from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx

from src.models import MediastackConfig
from src.scrapers.mediastack import MediastackScraper


SINCE = datetime(2026, 6, 27, 8, 30, 0, tzinfo=timezone.utc)
API_KEY_ENV = "MEDIASTACK_API_KEY"


def _mock_client(payload: dict | None = None) -> AsyncMock:
    response = MagicMock()
    response.json.return_value = payload if payload is not None else {}
    response.raise_for_status.return_value = None
    client = AsyncMock()
    client.get.return_value = response
    return client


def _data_payload() -> dict:
    return {
        "pagination": {"limit": 100, "offset": 0, "count": 2, "total": 2},
        "data": [
            {
                "url": "https://example.com/ai-1",
                "title": "AI breakthrough one",
                "published_at": "2026-06-28T12:00:00+00:00",
                "source": "Example News",
                "description": "A description of the first story.",
                "author": "Jane Reporter",
            },
            {
                "url": "https://news.test/ai-2",
                "title": "AI breakthrough two",
                "published_at": "2026-06-28T13:00:00+00:00",
                "source": "Test Wire",
                "description": "A description of the second story.",
                "author": "John Writer",
            },
        ],
    }


def test_missing_key_returns_empty_no_network(monkeypatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    client = _mock_client(_data_payload())
    config = MediastackConfig(enabled=True, keywords="ai")
    scraper = MediastackScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []
    client.get.assert_not_called()


def test_date_range_param_derived_from_since(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = _mock_client(_data_payload())
    config = MediastackConfig(enabled=True, keywords="ai")
    scraper = MediastackScraper(config, client)

    asyncio.run(scraper.fetch(SINCE))

    params = client.get.call_args.kwargs["params"]
    start = SINCE.strftime("%Y-%m-%d")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert params["date"] == f"{start},{today}"
    assert params["access_key"] == "secret-key"
    assert params["keywords"] == "ai"
    assert params["sort"] == "published_desc"


def test_parses_data_into_content_items(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = _mock_client(_data_payload())
    config = MediastackConfig(enabled=True, keywords="ai")
    scraper = MediastackScraper(config, client)

    items = asyncio.run(scraper.fetch(SINCE))

    assert len(items) == 2
    first = items[0]
    assert str(first.url) == "https://example.com/ai-1"
    assert first.title == "AI breakthrough one"
    assert first.published_at == datetime(2026, 6, 28, 12, 0, 0, tzinfo=timezone.utc)
    assert first.author == "Jane Reporter"
    assert first.content == "A description of the first story."
    assert first.metadata["source"] == "Example News"
    assert first.id.startswith("mediastack:article:")


def test_disabled_config_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = _mock_client(_data_payload())
    config = MediastackConfig(enabled=False, keywords="ai")
    scraper = MediastackScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_empty_keywords_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = _mock_client(_data_payload())
    config = MediastackConfig(enabled=True, keywords="   ")
    scraper = MediastackScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_http_error_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = AsyncMock()
    client.get.side_effect = httpx.HTTPError("boom")
    config = MediastackConfig(enabled=True, keywords="ai")
    scraper = MediastackScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_empty_data_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = _mock_client({"data": []})
    config = MediastackConfig(enabled=True, keywords="ai")
    scraper = MediastackScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_missing_data_key_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = _mock_client({"pagination": {}})
    config = MediastackConfig(enabled=True, keywords="ai")
    scraper = MediastackScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_error_body_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    error_body = {
        "error": {
            "code": "usage_limit_reached",
            "message": "Your monthly usage limit has been reached.",
        }
    }
    client = _mock_client(error_body)
    config = MediastackConfig(enabled=True, keywords="ai")
    scraper = MediastackScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_bad_article_is_skipped_not_crashing(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    payload = {
        "data": [
            {  # missing url -> skipped
                "title": "No URL",
                "published_at": "2026-06-28T12:00:00+00:00",
                "source": "Example News",
            },
            {  # unparseable published_at -> skipped
                "url": "https://example.com/bad-date",
                "title": "Bad date",
                "published_at": "not-a-date",
                "source": "Example News",
            },
            {  # valid -> kept
                "url": "https://example.com/good",
                "title": "Good one",
                "published_at": "2026-06-28T14:00:00+00:00",
                "source": "Example News",
            },
        ]
    }
    client = _mock_client(payload)
    config = MediastackConfig(enabled=True, keywords="ai")
    scraper = MediastackScraper(config, client)

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
    config = MediastackConfig(enabled=True, keywords="ai")
    scraper = MediastackScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []
