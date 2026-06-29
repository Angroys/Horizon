from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx

from src.models import ExaConfig
from src.scrapers.exa import ExaScraper


SINCE = datetime(2026, 6, 27, 8, 30, 0, tzinfo=timezone.utc)


def _mock_client(payload: dict | None = None) -> AsyncMock:
    response = MagicMock()
    response.json.return_value = payload if payload is not None else {}
    response.raise_for_status.return_value = None
    client = AsyncMock()
    client.post.return_value = response
    return client


def _results_payload() -> dict:
    return {
        "results": [
            {
                "id": "exa-1",
                "url": "https://example.com/ai-1",
                "title": "AI breakthrough one",
                "publishedDate": "2026-06-28T12:00:00.000Z",
                "author": "Jane Doe",
            },
            {
                "url": "https://news.test/ai-2",
                "title": "AI breakthrough two",
                "publishedDate": "2026-06-28T13:00:00Z",
                "author": "John Roe",
            },
        ]
    }


def test_missing_key_returns_empty_without_network(monkeypatch) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    client = _mock_client(_results_payload())
    config = ExaConfig(enabled=True, query="ai")
    scraper = ExaScraper(config, client)

    items = asyncio.run(scraper.fetch(SINCE))

    assert items == []
    client.post.assert_not_called()


def test_body_contains_expected_window_and_params(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "secret-key")
    client = _mock_client(_results_payload())
    config = ExaConfig(enabled=True, query="ai", num_results=42)
    scraper = ExaScraper(config, client)

    asyncio.run(scraper.fetch(SINCE))

    body = client.post.call_args.kwargs["json"]
    assert body["query"] == "ai"
    assert body["startPublishedDate"] == SINCE.isoformat()
    assert "endPublishedDate" in body
    assert body["numResults"] == 42
    assert body["type"] == "auto"

    headers = client.post.call_args.kwargs["headers"]
    assert headers["x-api-key"] == "secret-key"


def test_parses_results_into_content_items(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "secret-key")
    client = _mock_client(_results_payload())
    config = ExaConfig(enabled=True, query="ai")
    scraper = ExaScraper(config, client)

    items = asyncio.run(scraper.fetch(SINCE))

    assert len(items) == 2
    first = items[0]
    assert str(first.url) == "https://example.com/ai-1"
    assert first.title == "AI breakthrough one"
    assert first.published_at == datetime(2026, 6, 28, 12, 0, 0, tzinfo=timezone.utc)
    assert first.author == "Jane Doe"
    assert first.metadata["domain"] == "example.com"
    assert first.id.startswith("exa:search:")


def test_disabled_config_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "secret-key")
    client = _mock_client(_results_payload())
    config = ExaConfig(enabled=False, query="ai")
    scraper = ExaScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_http_error_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "secret-key")
    client = AsyncMock()
    client.post.side_effect = httpx.HTTPError("boom")
    config = ExaConfig(enabled=True, query="ai")
    scraper = ExaScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_missing_results_key_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "secret-key")
    client = _mock_client({"foo": "bar"})
    config = ExaConfig(enabled=True, query="ai")
    scraper = ExaScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_empty_query_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "secret-key")
    client = _mock_client(_results_payload())
    config = ExaConfig(enabled=True, query="   ")
    scraper = ExaScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []
    client.post.assert_not_called()


def test_bad_result_is_skipped_not_crashing(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "secret-key")
    payload = {
        "results": [
            {  # missing url -> skipped
                "title": "No URL",
                "publishedDate": "2026-06-28T12:00:00Z",
            },
            {  # unparseable publishedDate -> skipped
                "url": "https://example.com/bad-date",
                "title": "Bad date",
                "publishedDate": "not-a-date",
            },
            {  # valid -> kept
                "url": "https://example.com/good",
                "title": "Good one",
                "publishedDate": "2026-06-28T14:00:00Z",
                "author": "Someone",
            },
        ]
    }
    client = _mock_client(payload)
    config = ExaConfig(enabled=True, query="ai")
    scraper = ExaScraper(config, client)

    items = asyncio.run(scraper.fetch(SINCE))

    assert len(items) == 1
    assert str(items[0].url) == "https://example.com/good"


def test_non_json_body_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "secret-key")
    response = MagicMock()
    response.json.side_effect = ValueError("no json")
    response.raise_for_status.return_value = None
    client = AsyncMock()
    client.post.return_value = response
    config = ExaConfig(enabled=True, query="ai")
    scraper = ExaScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []
