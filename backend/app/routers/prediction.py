"""
Prediction router: run inference and retrieve stored predictions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.crypto import PredictionRecord

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    symbol: str
    model_name: str = "hybrid"           # lstm | gru | transformer | hybrid
    task: str = "regression"             # regression | classification
    horizon: int = 1                     # steps ahead


class PredictionResponse(BaseModel):
    symbol: str
    model_name: str
    predicted_at: datetime
    predicted_price: Optional[float] = None
    predicted_trend: Optional[int] = None    # 1=up, 0=down
    confidence: Optional[float] = None
    signal: str                              # BUY | SELL | HOLD


class HistoricalPrediction(BaseModel):
    predicted_at: datetime
    predicted_price: Optional[float]
    actual_price: Optional[float]
    predicted_trend: Optional[int]

    model_config = {"from_attributes": True}


# ── Helpers ───────────────────────────────────────────────────────────────

def _load_model(symbol: str, model_name: str, task: str, input_size: int):
    """Load a saved PyTorch model checkpoint."""
    from ml.models import GRUModel, HybridModel, LSTMModel, TransformerModel

    registry = {
        "lstm": lambda: LSTMModel(input_size=input_size, task=task),
        "gru": lambda: GRUModel(input_size=input_size, task=task),
        "transformer": lambda: TransformerModel(input_size=input_size, task=task),
        "hybrid": lambda: HybridModel(market_input_size=input_size, task=task),
    }
    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}")

    model = registry[model_name]()
    model_dir = Path(settings.model_dir)

    # Primary convention (produced by trainer.py): best_{symbol}_{model}_{task}.pt
    ckpt_candidates = [
        model_dir / f"best_{symbol}_{model_name}_{task}.pt",
        # Backward-compatibility with older naming used in API code.
        model_dir / f"{symbol}_{model_name}_{task}_best.pt",
    ]
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt_path is None:
        expected = ", ".join(str(p) for p in ckpt_candidates)
        raise FileNotFoundError(f"Checkpoint not found. Tried: {expected}")

    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    req: PredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Run live inference:
    1. Fetch recent OHLCV from Binance
    2. Engineer features
    3. Run model forward pass
    4. Return prediction + trading signal
    """
    symbol = req.symbol.upper()

    try:
        from ml.data.collectors.binance_collector import BinanceCollector
        from ml.data.preprocessors.market_preprocessor import MarketPreprocessor
        from ml.features.feature_engineering import FeatureEngineer

        binance = BinanceCollector()
        raw = binance.fetch_historical_ohlcv(
            symbol=f"{symbol}USDT", interval="1d", start="1 Jan, 2022"
        )
        fe = FeatureEngineer()
        feature_df = fe.select_features(fe.build_market_features(raw), include_sentiment=False)

        proc = MarketPreprocessor(
            sequence_length=settings.sequence_length,
            prediction_horizon=req.horizon,
        )
        scaled = proc.fit_transform(feature_df)
        X, y = proc.create_sequences(scaled)

        if len(X) == 0:
            raise HTTPException(status_code=422, detail="Not enough data to form sequences.")

        input_size = X.shape[-1]
        model = _load_model(symbol, req.model_name, req.task, input_size)

        last_seq = torch.tensor(X[-1:], dtype=torch.float32)
        with torch.no_grad():
            if hasattr(model, "sentiment_mlp"):
                sentiment = torch.zeros((1, 5), dtype=torch.float32)
                output = model(last_seq, sentiment).item()
            else:
                output = model(last_seq).item()

        # Build response
        now = datetime.now(tz=timezone.utc)
        predicted_price = None
        predicted_trend = None
        confidence = None
        signal = "HOLD"

        if req.task == "regression":
            predicted_price = float(
                proc.inverse_transform_price(np.array([output]))[-1]
            )
            current_price = float(raw["close"].iloc[-1])
            if predicted_price > current_price * 1.005:
                signal = "BUY"
            elif predicted_price < current_price * 0.995:
                signal = "SELL"

        else:
            confidence = float(output)
            predicted_trend = 1 if output >= 0.5 else 0
            signal = "BUY" if predicted_trend == 1 else "SELL"

        # Persist prediction
        record = PredictionRecord(
            symbol=symbol,
            model_name=req.model_name,
            predicted_at=now,
            target_time=now,
            predicted_price=predicted_price,
            predicted_trend=predicted_trend,
            confidence=confidence,
        )
        db.add(record)
        await db.commit()

        return PredictionResponse(
            symbol=symbol,
            model_name=req.model_name,
            predicted_at=now,
            predicted_price=predicted_price,
            predicted_trend=predicted_trend,
            confidence=confidence,
            signal=signal,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/history/{symbol}", response_model=list[HistoricalPrediction])
async def get_prediction_history(
    symbol: str,
    model_name: str = Query("hybrid"),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Return the most recent stored predictions for a symbol."""
    result = await db.execute(
        select(PredictionRecord)
        .where(PredictionRecord.symbol == symbol.upper())
        .where(PredictionRecord.model_name == model_name)
        .order_by(PredictionRecord.predicted_at.desc())
        .limit(limit)
    )
    records = result.scalars().all()
    return [HistoricalPrediction.model_validate(r) for r in reversed(records)]
