"""
Celery application and periodic task definitions.
"""
from __future__ import annotations

from celery import Celery
from celery.schedules import crontab
from loguru import logger

from app.config import settings

celery_app = Celery(
    "crypto_predict",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        # Refresh OHLCV every hour
        "collect-market-data": {
            "task": "app.tasks.collect_market_data",
            "schedule": crontab(minute=5),  # xx:05 every hour
        },
        # Run sentiment pipeline every 30 minutes
        "update-sentiment": {
            "task": "app.tasks.update_sentiment",
            "schedule": crontab(minute="*/30"),
        },
        # Generate predictions every hour
        "run-predictions": {
            "task": "app.tasks.run_predictions",
            "schedule": crontab(minute=15),
        },
    },
)


@celery_app.task(name="app.tasks.collect_market_data")
def collect_market_data():
    """Fetch and store latest OHLCV candles for all tracked symbols."""
    from ml.data.collectors.binance_collector import BinanceCollector
    collector = BinanceCollector()
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
    for sym in symbols:
        try:
            df = collector.fetch_historical_ohlcv(sym, interval="1h", start="7 days ago UTC")
            logger.info(f"Collected {len(df)} rows for {sym}")
        except Exception as exc:
            logger.error(f"Failed to collect {sym}: {exc}")


@celery_app.task(name="app.tasks.update_sentiment")
def update_sentiment():
    """Fetch news and update sentiment scores in MongoDB."""
    from ml.data.collectors.news_collector import NewsCollector
    from ml.nlp.sentiment_analyzer import SentimentAnalyzer
    news = NewsCollector()
    analyzer = SentimentAnalyzer()
    for symbol in ["BTC", "ETH", "SOL"]:
        try:
            articles = news.fetch_newsapi(symbol=symbol, days_back=1)
            if articles:
                agg = analyzer.aggregate_to_timeseries(articles)
                logger.info(f"Sentiment updated for {symbol}: {len(agg)} hourly records")
        except Exception as exc:
            logger.error(f"Sentiment update failed for {symbol}: {exc}")


@celery_app.task(name="app.tasks.run_predictions")
def run_predictions():
    """Generate predictions for all live models and store results."""
    logger.info("Prediction task triggered – see prediction service for implementation.")
