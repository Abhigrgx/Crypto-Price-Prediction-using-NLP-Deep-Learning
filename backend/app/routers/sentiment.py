"""
Sentiment router: latest scores and time-series aggregation.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


class SentimentPoint(BaseModel):
    timestamp: datetime
    sentiment_score: float
    sentiment_positive: int
    sentiment_negative: int
    news_count: int
    social_score: float


class SentimentSummary(BaseModel):
    symbol: str
    period_hours: int
    avg_sentiment: float
    dominant_label: str
    total_articles: int
    data: list[SentimentPoint]


@router.get("/{symbol}", response_model=SentimentSummary)
async def get_sentiment(
    symbol: str,
    hours: int = Query(24, ge=1, le=720, description="Look-back window in hours"),
):
    """
    Return aggregated sentiment for a symbol over the given window.
    Fetches fresh data from NewsAPI and runs FinBERT inference.
    """
    try:
        from ml.data.collectors.news_collector import NewsCollector
        from ml.nlp.sentiment_analyzer import SentimentAnalyzer

        collector = NewsCollector()
        articles = collector.fetch_newsapi(symbol=symbol.upper(), days_back=max(1, hours // 24 + 1))

        if not articles:
            raise HTTPException(status_code=404, detail="No articles found for symbol.")

        analyzer = SentimentAnalyzer()
        agg_df = analyzer.aggregate_to_timeseries(articles, freq="1h")

        if agg_df.empty:
            raise HTTPException(status_code=404, detail="No sentiment data aggregated.")

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        agg_df = agg_df[agg_df.index >= cutoff]

        points = [
            SentimentPoint(
                timestamp=idx if isinstance(idx, datetime) else datetime.fromisoformat(str(idx)),
                sentiment_score=float(row.get("sentiment_score", 0)),
                sentiment_positive=int(row.get("sentiment_positive", 0)),
                sentiment_negative=int(row.get("sentiment_negative", 0)),
                news_count=int(row.get("news_count", 0)),
                social_score=float(row.get("social_score", 0)),
            )
            for idx, row in agg_df.iterrows()
        ]

        avg = float(agg_df["sentiment_score"].mean()) if not agg_df.empty else 0.0
        dominant = "POSITIVE" if avg > 0.1 else "NEGATIVE" if avg < -0.1 else "NEUTRAL"

        return SentimentSummary(
            symbol=symbol.upper(),
            period_hours=hours,
            avg_sentiment=avg,
            dominant_label=dominant,
            total_articles=len(articles),
            data=points,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
