"""
Market data preprocessor: cleaning, scaling, and sequence creation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from loguru import logger


class MarketPreprocessor:
    """
    Cleans raw OHLCV DataFrames and prepares windowed sequences
    suitable for deep learning models.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        scaler_type: str = "minmax",
    ) -> None:
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = (
            MinMaxScaler(feature_range=(0, 1))
            if scaler_type == "minmax"
            else RobustScaler()
        )
        self._fitted = False

    # ── Cleaning ─────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, forward-fill gaps, drop residual NaNs."""
        df = df.copy()
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        # Forward-fill small gaps (≤3 consecutive), then drop remaining NaN rows
        df.ffill(limit=3, inplace=True)
        before = len(df)
        df.dropna(inplace=True)
        dropped = before - len(df)
        if dropped:
            logger.warning(f"Dropped {dropped} rows with NaN values after ffill.")

        return df

    # ── Scaling ───────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit scaler and transform. Returns 2-D array."""
        scaled = self.scaler.fit_transform(df.values)
        self._fitted = True
        logger.info(f"Scaler fitted on {df.shape} data.")
        return scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform first.")
        return self.scaler.transform(df.values)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse scale a 2-D array."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted.")
        return self.scaler.inverse_transform(data)

    def inverse_transform_price(
        self, scaled_prices: np.ndarray, price_col_idx: int = 3
    ) -> np.ndarray:
        """
        Inverse-transform only the close-price column.
        Expects scaled_prices shape (N,) or (N, 1).
        """
        n_features = self.scaler.n_features_in_
        dummy = np.zeros((len(scaled_prices.ravel()), n_features))
        dummy[:, price_col_idx] = scaled_prices.ravel()
        return self.scaler.inverse_transform(dummy)[:, price_col_idx]

    # ── Sequence Builder ──────────────────────────────────────────────────

    def create_sequences(
        self,
        data: np.ndarray,
        target_col: int = 3,  # close price index
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slide a window over `data` to create (X, y) pairs.

        Returns
        -------
        X : shape (samples, sequence_length, n_features)
        y : shape (samples,)  – target is price `prediction_horizon` steps ahead
        """
        X, y = [], []
        total = len(data) - self.sequence_length - self.prediction_horizon + 1
        for i in range(total):
            X.append(data[i : i + self.sequence_length])
            y.append(data[i + self.sequence_length + self.prediction_horizon - 1, target_col])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train_test_split_timeseries(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_ratio: float = 0.15,
        val_ratio: float = 0.10,
    ) -> Tuple[np.ndarray, ...]:
        """
        Chronological split – no shuffling.
        Returns (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        n = len(X)
        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)
        train_size = n - val_size - test_size

        X_train = X[:train_size]
        X_val = X[train_size : train_size + val_size]
        X_test = X[train_size + val_size :]

        y_train = y[:train_size]
        y_val = y[train_size : train_size + val_size]
        y_test = y[train_size + val_size :]

        logger.info(
            f"Split → train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ── Classification Labels ─────────────────────────────────────────────

    @staticmethod
    def create_trend_labels(prices: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Binary trend labels: 1 = price up, 0 = price down/flat.
        `prices` should be raw (unscaled) close prices.
        """
        shifts = np.roll(prices, -1)
        pct_change = (shifts - prices) / prices
        labels = (pct_change > threshold).astype(np.int32)
        return labels[:-1]  # last element is invalid (wrapped)
