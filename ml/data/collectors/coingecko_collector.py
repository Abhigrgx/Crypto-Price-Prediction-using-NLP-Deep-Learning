"""
CoinGecko historical market data collector (free & pro tier).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from pycoingecko import CoinGeckoAPI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class CoinGeckoCollector:
    """Fetches OHLCV, market cap and metadata from CoinGecko."""

    SYMBOL_TO_ID: dict[str, str] = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binancecoin",
        "SOL": "solana",
        "XRP": "ripple",
        "ADA": "cardano",
        "DOGE": "dogecoin",
        "AVAX": "avalanche-2",
        "DOT": "polkadot",
        "MATIC": "matic-network",
    }

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("COINGECKO_API_KEY", "")
        self.cg = CoinGeckoAPI(api_key=key if key else None)
        logger.info("CoinGeckoCollector initialised.")

    def _resolve_id(self, symbol: str) -> str:
        return self.SYMBOL_TO_ID.get(symbol.upper(), symbol.lower())

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30), reraise=True)
    def fetch_market_chart(
        self,
        symbol: str = "BTC",
        vs_currency: str = "usd",
        days: int = 365,
    ) -> pd.DataFrame:
        """
        Returns DataFrame with [timestamp, price, market_cap, volume].
        Granularity is automatic: hourly ≤90 days, daily >90 days.
        """
        coin_id = self._resolve_id(symbol)
        data = self.cg.get_coin_market_chart_by_id(
            id=coin_id, vs_currency=vs_currency, days=days
        )
        price_df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        mcap_df = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
        vol_df = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])

        df = price_df.merge(mcap_df, on="timestamp").merge(vol_df, on="timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        logger.info(f"CoinGecko: fetched {len(df)} rows for {symbol}.")
        return df

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30), reraise=True)
    def fetch_ohlc(
        self,
        symbol: str = "BTC",
        vs_currency: str = "usd",
        days: int = 30,
    ) -> pd.DataFrame:
        """Returns OHLC DataFrame for the given number of days."""
        coin_id = self._resolve_id(symbol)
        raw = self.cg.get_coin_ohlc_by_id(
            id=coin_id, vs_currency=vs_currency, days=days
        )
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        logger.info(f"CoinGecko OHLC: {len(df)} rows for {symbol}.")
        return df

    def get_top_coins(self, limit: int = 50) -> list[dict]:
        """Return top-N coins by market cap."""
        return self.cg.get_coins_markets(
            vs_currency="usd",
            order="market_cap_desc",
            per_page=limit,
            page=1,
            sparkline=False,
        )
