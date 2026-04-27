"""
Binance historical OHLCV data collector.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class BinanceCollector:
    """Fetches historical and live OHLCV data from Binance."""

    INTERVAL_MAP = {
        "1m": Client.KLINE_INTERVAL_1MINUTE,
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        key = api_key or os.getenv("BINANCE_API_KEY", "")
        secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        self.client = Client(key, secret)
        logger.info("BinanceCollector initialised.")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def fetch_historical_ohlcv(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1d",
        start: str = "1 Jan, 2020",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
        [open_time, open, high, low, close, volume,
         close_time, quote_volume, trades,
         taker_buy_base, taker_buy_quote].
        """
        kline_interval = self.INTERVAL_MAP.get(interval, Client.KLINE_INTERVAL_1DAY)
        try:
            raw = self.client.get_historical_klines(
                symbol=symbol.upper(),
                interval=kline_interval,
                start_str=start,
                end_str=end,
            )
        except BinanceAPIException as exc:
            logger.error(f"Binance API error: {exc}")
            raise

        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ]
        df = pd.DataFrame(raw, columns=columns)
        df.drop(columns=["ignore"], inplace=True)

        # Type coercions
        numeric_cols = [
            "open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        df.sort_index(inplace=True)

        logger.info(f"Binance: fetched {len(df)} rows for {symbol} ({interval}).")
        return df

    def fetch_current_price(self, symbol: str = "BTCUSDT") -> float:
        """Return the latest price as a float."""
        ticker = self.client.get_symbol_ticker(symbol=symbol.upper())
        return float(ticker["price"])
