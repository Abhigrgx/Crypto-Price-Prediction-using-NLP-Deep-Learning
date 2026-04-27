"""
Feature engineering pipeline:
  1. Market OHLCV + technical indicators
  2. Sentiment scores (aggregated from NLP module)
  3. Time-based features
  4. Final merged feature matrix
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from ml.features.technical_indicators import add_all_indicators, add_time_features


class FeatureEngineer:
    """Merge all feature sources into a single aligned DataFrame."""

    # Columns that form the final feature set (order matters for model input)
    MARKET_COLS = [
        "open", "high", "low", "close", "volume",
    ]
    INDICATOR_COLS = [
        "sma_7", "sma_21", "sma_50", "ema_7", "ema_21", "ema_50",
        "rsi_14", "rsi_7",
        "macd", "macd_signal", "macd_diff",
        "bb_upper", "bb_lower", "bb_width", "bb_pct_b",
        "atr_14",
        "stoch_k", "stoch_d",
        "obv",
        "cci_20",
        "williams_r",
        "roc_10",
        "volume_ratio",
        "body_size", "upper_wick", "lower_wick",
    ]
    TIME_COLS = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_weekend", "month", "quarter",
    ]
    SENTIMENT_COLS = [
        "sentiment_score",
        "sentiment_positive",
        "sentiment_negative",
        "news_count",
        "social_score",
    ]

    def build_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply indicators + time features to raw OHLCV frame."""
        df = add_all_indicators(df)
        df = add_time_features(df)
        return df

    def merge_sentiment(
        self,
        market_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """
        Resample sentiment_df to `freq` and left-join onto market_df.
        sentiment_df must have a DatetimeIndex and the SENTIMENT_COLS columns.
        Missing sentiment values are forward-filled then zero-filled.
        """
        if sentiment_df is None or sentiment_df.empty:
            for col in self.SENTIMENT_COLS:
                market_df[col] = 0.0
            return market_df

        resampled = sentiment_df[
            [c for c in self.SENTIMENT_COLS if c in sentiment_df.columns]
        ].resample(freq).mean()

        merged = market_df.join(resampled, how="left")
        for col in self.SENTIMENT_COLS:
            if col in merged.columns:
                merged[col] = merged[col].ffill().fillna(0.0)
        return merged

    def select_features(
        self,
        df: pd.DataFrame,
        include_sentiment: bool = True,
    ) -> pd.DataFrame:
        """Select and order the final feature columns, dropping NaN-heavy rows."""
        cols = self.MARKET_COLS + self.INDICATOR_COLS + self.TIME_COLS
        if include_sentiment:
            cols += self.SENTIMENT_COLS
        cols = [c for c in cols if c in df.columns]
        out = df[cols].copy()
        # Drop rows where >50% of values are NaN (indicator warm-up)
        threshold = int(len(cols) * 0.5)
        out.dropna(thresh=threshold, inplace=True)
        out.ffill(inplace=True)
        out.dropna(inplace=True)
        logger.info(f"Feature matrix shape: {out.shape}")
        return out

    def compute_rolling_volatility(
        self, df: pd.DataFrame, window: int = 24
    ) -> pd.Series:
        """Annualised rolling volatility of close returns."""
        log_ret = np.log(df["close"] / df["close"].shift(1))
        return log_ret.rolling(window).std() * np.sqrt(365 * 24)
