"""
End-to-end training script.
Run:  python -m ml.training.train_pipeline --symbol BTC --model hybrid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ml.data.collectors.binance_collector import BinanceCollector
from ml.data.collectors.news_collector import NewsCollector
from ml.data.preprocessors.market_preprocessor import MarketPreprocessor
from ml.features.feature_engineering import FeatureEngineer
from ml.models import GRUModel, HybridModel, LSTMModel, TransformerModel
from ml.nlp.sentiment_analyzer import SentimentAnalyzer
from ml.training.evaluator import directional_accuracy, evaluate_regression
from ml.training.trainer import ModelTrainer
from ml.backtesting.backtester import Backtester


def build_model(model_type: str, input_size: int, task: str, **kwargs):
    registry = {
        "lstm": lambda: LSTMModel(input_size=input_size, task=task, **kwargs),
        "gru": lambda: GRUModel(input_size=input_size, task=task, **kwargs),
        "transformer": lambda: TransformerModel(input_size=input_size, task=task, **kwargs),
        "hybrid": lambda: HybridModel(
            market_input_size=input_size,
            sentiment_input_size=5,
            task=task,
            **kwargs,
        ),
    }
    if model_type not in registry:
        raise ValueError(f"Unknown model: {model_type}. Choose from {list(registry)}")
    return registry[model_type]()


def run_pipeline(
    symbol: str = "BTC",
    interval: str = "1d",
    model_type: str = "hybrid",
    task: str = "regression",
    seq_len: int = 60,
    horizon: int = 1,
    epochs: int = 150,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    start: str = "1 Jan, 2020",
    checkpoint_dir: str = "ml/saved_models",
) -> dict:
    logger.info(f"=== Pipeline START | {symbol} | {model_type} | {task} ===")

    # ── 1. Collect market data ────────────────────────────────────────────
    binance = BinanceCollector()
    raw = binance.fetch_historical_ohlcv(
        symbol=f"{symbol}USDT", interval=interval, start=start
    )

    # ── 2. Feature engineering ────────────────────────────────────────────
    fe = FeatureEngineer()
    market = fe.build_market_features(raw)

    # ── 3. Sentiment (optional, skip if no API key) ───────────────────────
    sentiment_df: pd.DataFrame = pd.DataFrame()
    try:
        news_col = NewsCollector()
        articles = news_col.fetch_newsapi(symbol=symbol, days_back=365)
        if articles:
            analyzer = SentimentAnalyzer()
            sentiment_df = analyzer.aggregate_to_timeseries(articles, freq=interval)
    except Exception as exc:
        logger.warning(f"Sentiment pipeline skipped: {exc}")

    include_sentiment = not sentiment_df.empty
    merged = fe.merge_sentiment(market, sentiment_df, freq=interval)
    feature_df = fe.select_features(merged, include_sentiment=include_sentiment)

    # ── 4. Preprocessing + sequences ─────────────────────────────────────
    proc = MarketPreprocessor(sequence_length=seq_len, prediction_horizon=horizon)
    proc.clean(feature_df)  # validate
    scaled = proc.fit_transform(feature_df)
    X, y = proc.create_sequences(scaled)

    splits = proc.train_test_split_timeseries(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    input_size = X_train.shape[-1]
    logger.info(f"Input size: {input_size} features")

    # ── 4b. Sentiment branch for hybrid ──────────────────────────────────
    sent_train = sent_val = sent_test = None
    if model_type == "hybrid" and include_sentiment:
        sent_cols = [
            "sentiment_score", "sentiment_positive", "sentiment_negative",
            "news_count", "social_score",
        ]
        sent_idx = [list(feature_df.columns).index(c) for c in sent_cols if c in feature_df.columns]
        if sent_idx:
            sent_train = X_train[:, -1, sent_idx]
            sent_val = X_val[:, -1, sent_idx]
            sent_test = X_test[:, -1, sent_idx]

    # ── 5. Build + train ──────────────────────────────────────────────────
    model = build_model(model_type, input_size, task)
    trainer = ModelTrainer(
        model=model,
        task=task,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        model_name=f"{symbol}_{model_type}_{task}",
    )
    history = trainer.fit(
        X_train, y_train, X_val, y_val,
        sentiment_train=sent_train, sentiment_val=sent_val,
    )

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    preds = trainer.predict(X_test, sentiment=sent_test)

    if task == "regression":
        y_test_inv = proc.inverse_transform_price(y_test)
        preds_inv = proc.inverse_transform_price(preds)
        metrics = evaluate_regression(y_test_inv, preds_inv, label=model_type)
        metrics["directional_accuracy"] = directional_accuracy(y_test_inv, preds_inv)
    else:
        from ml.training.evaluator import evaluate_classification
        metrics = evaluate_classification(y_test, preds, label=model_type)

    # ── 7. Backtest ───────────────────────────────────────────────────────
    if task == "regression":
        signals = (np.diff(preds_inv) > 0).astype(int)  # type: ignore[possibly-undefined]
        backtester = Backtester()
        bt_result = backtester.run(y_test_inv[1:], signals)  # type: ignore[possibly-undefined]
        metrics.update(
            {
                "backtest_total_return": bt_result.total_return,
                "backtest_sharpe": bt_result.sharpe_ratio,
                "backtest_max_drawdown": bt_result.max_drawdown,
                "backtest_win_rate": bt_result.win_rate,
            }
        )

    # ── 8. Save metrics ───────────────────────────────────────────────────
    out_path = Path(checkpoint_dir) / f"{symbol}_{model_type}_{task}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metrics saved to {out_path}")
    logger.info("=== Pipeline DONE ===")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crypto prediction model")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--model", default="hybrid", choices=["lstm", "gru", "transformer", "hybrid"])
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--start", default="1 Jan, 2020")
    args = parser.parse_args()

    run_pipeline(
        symbol=args.symbol,
        interval=args.interval,
        model_type=args.model,
        task=args.task,
        seq_len=args.seq_len,
        horizon=args.horizon,
        epochs=args.epochs,
        learning_rate=args.lr,
        start=args.start,
    )
