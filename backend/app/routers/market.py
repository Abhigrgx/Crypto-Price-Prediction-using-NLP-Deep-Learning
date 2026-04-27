"""
Market data router: live prices, OHLCV, and stored candles.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.crypto import CandleRecord

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────

class CandleOut(BaseModel):
    symbol: str
    interval: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    model_config = {"from_attributes": True}


class PriceOut(BaseModel):
    symbol: str
    price: float
    fetched_at: datetime


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/price/{symbol}", response_model=PriceOut)
async def get_current_price(symbol: str):
    """Fetch live price from Binance."""
    try:
        from ml.data.collectors.binance_collector import BinanceCollector
        collector = BinanceCollector()
        price = collector.fetch_current_price(f"{symbol.upper()}USDT")
        return PriceOut(symbol=symbol.upper(), price=price, fetched_at=datetime.utcnow())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/ohlcv/{symbol}", response_model=list[CandleOut])
async def get_ohlcv(
    symbol: str,
    interval: str = Query("1d", description="Candle interval: 1m, 5m, 1h, 4h, 1d"),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    """Return stored OHLCV candles from the database."""
    result = await db.execute(
        select(CandleRecord)
        .where(CandleRecord.symbol == symbol.upper())
        .where(CandleRecord.interval == interval)
        .order_by(CandleRecord.open_time.desc())
        .limit(limit)
    )
    candles = result.scalars().all()
    return [CandleOut.model_validate(c) for c in reversed(candles)]


@router.get("/supported-symbols")
async def get_supported_symbols():
    """Return the list of tracked symbols."""
    return {
        "symbols": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC"]
    }
