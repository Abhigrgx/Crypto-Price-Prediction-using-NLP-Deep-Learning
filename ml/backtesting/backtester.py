"""
Backtesting engine: simulates a long-only trading strategy
driven by model predictions and evaluates trading performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class BacktestResult:
    total_return: float
    annualised_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    equity_curve: pd.Series = field(repr=False)
    trade_log: pd.DataFrame = field(repr=False)


class Backtester:
    """
    Simple event-driven backtester.
    - Enter long when model predicts price UP (signal=1).
    - Exit (go to cash) when model predicts price DOWN (signal=0).
    - Includes proportional transaction costs.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
        risk_free_rate: float = 0.02,
    ) -> None:
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        prices: np.ndarray,         # actual close prices (N,)
        signals: np.ndarray,        # binary: 1=long, 0=cash (N,)
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> BacktestResult:
        """
        Run the backtest and return a BacktestResult.

        Parameters
        ----------
        prices    : array of actual close prices aligned with signals
        signals   : binary signal array (1 = hold long, 0 = stay in cash)
        timestamps: optional DatetimeIndex for the equity curve
        """
        n = min(len(prices), len(signals))
        prices = prices[:n]
        signals = signals[:n]

        capital = self.initial_capital
        position = 0.0          # units held
        equity_curve = np.zeros(n)
        trades: list[dict] = []
        in_position = False
        entry_price = 0.0

        for i in range(n):
            price = float(prices[i])
            signal = int(signals[i])

            if signal == 1 and not in_position:
                # BUY
                cost = capital * self.transaction_cost
                units = (capital - cost) / price
                position = units
                capital = 0.0
                in_position = True
                entry_price = price
                trades.append({"type": "BUY", "price": price, "idx": i})

            elif signal == 0 and in_position:
                # SELL
                proceeds = position * price
                cost = proceeds * self.transaction_cost
                capital = proceeds - cost
                position = 0.0
                in_position = False
                trades.append(
                    {
                        "type": "SELL",
                        "price": price,
                        "idx": i,
                        "pnl": capital - self.initial_capital,
                        "return_pct": (price - entry_price) / entry_price * 100,
                    }
                )

            equity_curve[i] = capital + position * price

        # Liquidate if still in position at end
        if in_position:
            capital = position * float(prices[-1]) * (1 - self.transaction_cost)
            equity_curve[-1] = capital

        # ── Metrics ──────────────────────────────────────────────────────
        final_equity = equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        days = n / 365.0
        annualised_return = (1 + total_return) ** (1 / max(days, 1)) - 1

        returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
        excess = returns - self.risk_free_rate / 252
        sharpe = (
            float(np.mean(excess) / (np.std(excess) + 1e-9) * np.sqrt(252))
            if len(excess) > 0
            else 0.0
        )

        rolling_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - rolling_max) / (rolling_max + 1e-9)
        max_drawdown = float(np.min(drawdowns))

        sell_trades = [t for t in trades if t["type"] == "SELL"]
        win_rate = (
            float(
                sum(1 for t in sell_trades if t.get("return_pct", 0) > 0)
                / len(sell_trades)
            )
            if sell_trades
            else 0.0
        )

        idx = timestamps if timestamps is not None else pd.RangeIndex(n)
        eq_series = pd.Series(equity_curve, index=idx[:n], name="equity")

        logger.info(
            f"Backtest → total_return={total_return:.2%} | "
            f"sharpe={sharpe:.2f} | max_dd={max_drawdown:.2%} | "
            f"win_rate={win_rate:.2%} | trades={len(sell_trades)}"
        )

        return BacktestResult(
            total_return=total_return,
            annualised_return=annualised_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            num_trades=len(sell_trades),
            equity_curve=eq_series,
            trade_log=pd.DataFrame(trades),
        )
