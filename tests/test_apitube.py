from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx

from src.models import APITubeConfig
from src.scrapers.apitube import APITubeScraper


SINCE = datetime(2026, 6, 27, 8, 30, 0, tzinfo=timezone.utc)
ENV_KEY = "APITUBE_API_KEY"


def _mock_client(payload: dict | None = None) -> AsyncMock:
    response = MagicMock()
    response.json.return_value = payload if payload is not None else {}
    response.raise_for_status.return_value = None
    client = AsyncMock()
    client.get.return_value = response
    return client


def _results_payload() -> dict:
    return {
        "results": [
            {
                "href": "https://example.com/ai-1",
                "title": "AI breakthrough one",
                "published_at": "2026-06-28T12:00:00Z",
                "description": "First article body.",
                "language": "en",
                "source": {"name": "Example News", "domain": "example.com"},
            },
            {
                # uses `url` instead of `href` to prove the fallback
                "url": "https://news.test/ai-2",
                "title": "AI breakthrough two",
                "published_at": "2026-06-28T13:00:00+00:00",
                "description": "Second article body.",
                "source": {"name": "News Test", "domain": "news.test"},
            },
        ]
    }


def _data_payload() -> dict:
    """Same records but under the `data` key (defensive fallback)."""
    return {
        "data": [
            {
                "href": "https://example.com/under-data",
                "title": "Lives under data",
                "published_at": "2026-06-28T15:00:00Z",
                "description": "Body.",
                "source": {"name": "Data Source", "domain": "data.test"},
            }
        ]
    }


def test_missing_key_returns_empty_no_network(monkeypatch) -> None:
    monkeypatch.delenv(ENV_KEY, raising=False)
    client = _mock_client(_results_payload())
    config = APITubeConfig(enabled=True, query="ai")
    scraper = APITubeScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []
    client.get.assert_not_called()


def test_since_maps_to_published_at_start(monkeypatch) -> None:
    monkeypatch.setenv(ENV_KEY, "secret")
    client = _mock_client(_results_payload())
    config = APITubeConfig(enabled=True, query="ai")
    scraper = APITubeScraper(config, client)

    asyncio.run(scraper.fetch(SINCE))

    params = client.get.call_args.kwargs["params"]
    assert params["published_at.start"] == SINCE.isoformat()
    assert "published_at.end" in params
    assert params["title"] == "ai"


def test_auth_sent_via_x_api_key_header(monkeypatch) -> None:
    monkeypatch.setenv(ENV_KEY, "secret-token")
    client = _mock_client(_results_payload())
    config = APITubeConfig(enabled=True, query="ai")
    scraper = APITubeScraper(config, client)

    asyncio.run(scraper.fetch(SINCE))

    headers = client.get.call_args.kwargs["headers"]
    assert headers["X-API-Key"] == "secret-token"


def test_parses_results_into_content_items(monkeypatch) -> None:
    monkeypatch.setenv(ENV_KEY, "secret")
    client = _mock_client(_results_payload())
    config = APITubeConfig(enabled=True, query="ai")
    scraper = APITubeScraper(config, client)

    items = asyncio.run(scraper.fetch(SINCE))

    assert len(items) == 2
    first = items[0]
    assert str(first.url) == "https://example.com/ai-1"
    assert first.title == "AI breakthrough one"
    assert first.published_at == datetime(2026, 6, 28, 12, 0, 0, tzinfo=timezone.utc)
    assert first.content == "First article body."
    assert first.metadata["source_name"] == "Example News"
    assert first.metadata["domain"] == "example.com"
    assert first.id.startswith("apitube:article:")
    # `url` fallback when `href` is absent
    assert str(items[1].url) == "https://news.test/ai-2"


def test_defensive_fallback_to_data_key(monkeypatch) -> None:
    monkeypatch.setenv(ENV_KEY, "secret")
    client = _mock_client(_data_payload())
    config = APITubeConfig(enabled=True, query="ai")
    scraper = APITubeScraper(config, client)

    items = asyncio.run(scraper.fetch(SINCE))

    assert len(items) == 1
    assert str(items[0].url) == "https://example.com/under-data"
    assert items[0].title == "Lives under data"


def test_disabled_config_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(ENV_KEY, "secret")
    client = _mock_client(_results_payload())
    config = APITubeConfig(enabled=False, query="ai")
    scraper = APITubeScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_empty_query_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(ENV_KEY, "secret")
    client = _mock_client(_results_payload())
    config = APITubeConfig(enabled=True, query="   ")
    scraper = APITubeScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_http_error_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(ENV_KEY, "secret")
    client = AsyncMock()
    client.get.side_effect = httpx.HTTPError("boom")
    config = APITubeConfig(enabled=True, query="ai")
    scraper = APITubeScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_missing_results_key_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(ENV_KEY, "secret")
    client = _mock_client({"foo": "bar"})
    config = APITubeConfig(enabled=True, query="ai")
    scraper = APITubeScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []


def test_bad_article_is_skipped_not_crashing(monkeypatch) -> None:
    monkeypatch.setenv(ENV_KEY, "secret")
    payload = {
        "results": [
            {  # missing url/href -> skipped
                "title": "No URL",
                "published_at": "2026-06-28T12:00:00Z",
            },
            {  # unparseable published_at -> skipped
                "href": "https://example.com/bad-date",
                "title": "Bad date",
                "published_at": "not-a-date",
            },
            {  # valid -> kept
                "href": "https://example.com/good",
                "title": "Good one",
                "published_at": "2026-06-28T14:00:00Z",
            },
        ]
    }
    client = _mock_client(payload)
    config = APITubeConfig(enabled=True, query="ai")
    scraper = APITubeScraper(config, client)

    items = asyncio.run(scraper.fetch(SINCE))

    assert len(items) == 1
    assert str(items[0].url) == "https://example.com/good"


def test_non_json_body_returns_empty(monkeypatch) -> None:
    monkeypatch.setenv(ENV_KEY, "secret")
    response = MagicMock()
    response.json.side_effect = ValueError("no json")
    response.raise_for_status.return_value = None
    client = AsyncMock()
    client.get.return_value = response
    config = APITubeConfig(enabled=True, query="ai")
    scraper = APITubeScraper(config, client)

    assert asyncio.run(scraper.fetch(SINCE)) == []
