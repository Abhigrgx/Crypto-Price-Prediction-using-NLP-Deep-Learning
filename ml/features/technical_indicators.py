"""
Technical indicators module using the `ta` library.
Adds RSI, MACD, Bollinger Bands, EMA, SMA, ATR, OBV, and Stochastic.
"""
from __future__ import annotations

import pandas as pd
import ta.trend as _ta_trend  # type: ignore[import-untyped]
import ta.momentum as _ta_momentum  # type: ignore[import-untyped]
import ta.volatility as _ta_volatility  # type: ignore[import-untyped]
import ta.volume as _ta_volume  # type: ignore[import-untyped]
from loguru import logger

# Alias so existing `ta.xxx` references still work
class _ta:  # noqa: N801
    trend = _ta_trend
    momentum = _ta_momentum
    volatility = _ta_volatility
    volume = _ta_volume

ta = _ta()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and append all standard technical indicators to an OHLCV DataFrame.

    Expected columns: open, high, low, close, volume
    Returns the DataFrame with indicator columns appended.
    """
    df = df.copy()

    # ── Moving Averages ───────────────────────────────────────────────────
    for window in (7, 14, 21, 50, 100, 200):
        df[f"sma_{window}"] = ta.trend.sma_indicator(df["close"], window=window)
        df[f"ema_{window}"] = ta.trend.ema_indicator(df["close"], window=window)

    # ── RSI ───────────────────────────────────────────────────────────────
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["rsi_7"] = ta.momentum.RSIIndicator(df["close"], window=7).rsi()

    # ── MACD ──────────────────────────────────────────────────────────────
    macd_obj = ta.trend.MACD(df["close"])
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"] = macd_obj.macd_diff()

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_mavg"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct_b"] = bb.bollinger_pband()

    # ── Average True Range (ATR) ──────────────────────────────────────────
    df["atr_14"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    # ── Stochastic Oscillator ─────────────────────────────────────────────
    stoch = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"], window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ── On-Balance Volume (OBV) ───────────────────────────────────────────
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        df["close"], df["volume"]
    ).on_balance_volume()

    # ── Commodity Channel Index (CCI) ─────────────────────────────────────
    df["cci_20"] = ta.trend.CCIIndicator(
        df["high"], df["low"], df["close"], window=20
    ).cci()

    # ── Williams %R ───────────────────────────────────────────────────────
    df["williams_r"] = ta.momentum.WilliamsRIndicator(
        df["high"], df["low"], df["close"], lbp=14
    ).williams_r()

    # ── Price Rate of Change ──────────────────────────────────────────────
    df["roc_10"] = ta.momentum.ROCIndicator(df["close"], window=10).roc()

    # ── Volume features ───────────────────────────────────────────────────
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, 1)

    # ── Candlestick features ──────────────────────────────────────────────
    df["body_size"] = abs(df["close"] - df["open"])
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    logger.info(f"Added {len(df.columns)} columns after technical indicators.")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar / cyclical time-based features."""
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["day_of_month"] = idx.day
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Cyclical encoding for periodic features
    df["hour_sin"] = (2 * 3.14159 * df["hour"] / 24).apply(__import__("math").sin)
    df["hour_cos"] = (2 * 3.14159 * df["hour"] / 24).apply(__import__("math").cos)
    df["dow_sin"] = (2 * 3.14159 * df["day_of_week"] / 7).apply(__import__("math").sin)
    df["dow_cos"] = (2 * 3.14159 * df["day_of_week"] / 7).apply(__import__("math").cos)

    return df
