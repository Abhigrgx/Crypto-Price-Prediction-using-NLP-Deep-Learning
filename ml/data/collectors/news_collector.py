"""
News article collector from NewsAPI and CryptoPanic.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from newsapi import NewsApiClient
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class NewsCollector:
    """Fetches crypto-related news articles with timestamps."""

    CRYPTO_QUERIES: dict[str, str] = {
        "BTC": "bitcoin OR BTC",
        "ETH": "ethereum OR ETH",
        "SOL": "solana OR SOL",
        "BNB": "binance coin OR BNB",
        "XRP": "ripple OR XRP",
    }

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        cryptopanic_key: Optional[str] = None,
    ) -> None:
        self._newsapi_key = newsapi_key or os.getenv("NEWS_API_KEY", "")
        self._cryptopanic_key = cryptopanic_key or os.getenv("CRYPTOPANIC_API_KEY", "")
        if self._newsapi_key:
            self._newsapi = NewsApiClient(api_key=self._newsapi_key)
        else:
            self._newsapi = None
        logger.info("NewsCollector initialised.")

    # ── NewsAPI ───────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20), reraise=True)
    def fetch_newsapi(
        self,
        symbol: str = "BTC",
        days_back: int = 7,
        page_size: int = 100,
    ) -> list[dict]:
        """Return list of article dicts [{title, description, content, published_at, source}]."""
        if not self._newsapi:
            logger.warning("NewsAPI key not configured.")
            return []

        query = self.CRYPTO_QUERIES.get(symbol.upper(), symbol)
        from_date = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).isoformat()

        response = self._newsapi.get_everything(
            q=query,
            from_param=from_date,
            language="en",
            sort_by="publishedAt",
            page_size=page_size,
        )
        articles = response.get("articles", [])
        parsed = []
        for art in articles:
            parsed.append(
                {
                    "title": art.get("title", ""),
                    "description": art.get("description", ""),
                    "content": art.get("content", ""),
                    "published_at": art.get("publishedAt", ""),
                    "source": art.get("source", {}).get("name", ""),
                    "url": art.get("url", ""),
                    "symbol": symbol.upper(),
                }
            )
        logger.info(f"NewsAPI: {len(parsed)} articles for {symbol}.")
        return parsed

    # ── CryptoPanic ───────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20), reraise=True)
    def fetch_cryptopanic(
        self,
        currencies: str = "BTC,ETH",
        filter_: str = "hot",
        pages: int = 3,
    ) -> list[dict]:
        """
        Fetch posts from CryptoPanic (news + media).
        filter_: hot | rising | bullish | bearish | important | lol
        """
        if not self._cryptopanic_key:
            logger.warning("CryptoPanic key not configured.")
            return []

        base_url = "https://cryptopanic.com/api/v1/posts/"
        results: list[dict] = []

        for page in range(1, pages + 1):
            params = {
                "auth_token": self._cryptopanic_key,
                "currencies": currencies,
                "filter": filter_,
                "public": "true",
                "page": page,
            }
            resp = requests.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for post in data.get("results", []):
                results.append(
                    {
                        "title": post.get("title", ""),
                        "published_at": post.get("published_at", ""),
                        "source": post.get("source", {}).get("domain", ""),
                        "url": post.get("url", ""),
                        "votes_positive": post.get("votes", {}).get("positive", 0),
                        "votes_negative": post.get("votes", {}).get("negative", 0),
                        "currencies": [
                            c.get("code") for c in post.get("currencies", [])
                        ],
                    }
                )

        logger.info(f"CryptoPanic: {len(results)} posts.")
        return results
