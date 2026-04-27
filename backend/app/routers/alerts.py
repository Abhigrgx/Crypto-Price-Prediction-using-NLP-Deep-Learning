"""
Alerts router: manage price and sentiment threshold alerts.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

router = APIRouter()

# In-memory store (replace with DB in production)
_alerts: list[dict] = []
_alert_id_counter = 0


class AlertCreate(BaseModel):
    symbol: str
    alert_type: str           # price_above | price_below | sentiment_above | sentiment_below
    threshold: float
    notify_email: Optional[str] = None

    @field_validator("alert_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"price_above", "price_below", "sentiment_above", "sentiment_below"}
        if v not in allowed:
            raise ValueError(f"alert_type must be one of {allowed}")
        return v


class AlertOut(BaseModel):
    id: int
    symbol: str
    alert_type: str
    threshold: float
    notify_email: Optional[str]
    active: bool
    created_at: datetime
    triggered_at: Optional[datetime]


@router.post("/", response_model=AlertOut, status_code=201)
async def create_alert(payload: AlertCreate):
    global _alert_id_counter
    _alert_id_counter += 1
    alert = {
        "id": _alert_id_counter,
        "symbol": payload.symbol.upper(),
        "alert_type": payload.alert_type,
        "threshold": payload.threshold,
        "notify_email": payload.notify_email,
        "active": True,
        "created_at": datetime.now(tz=timezone.utc),
        "triggered_at": None,
    }
    _alerts.append(alert)
    return AlertOut(**alert)


@router.get("/", response_model=list[AlertOut])
async def list_alerts(symbol: Optional[str] = None):
    if symbol:
        return [AlertOut(**a) for a in _alerts if a["symbol"] == symbol.upper()]
    return [AlertOut(**a) for a in _alerts]


@router.delete("/{alert_id}", status_code=204)
async def delete_alert(alert_id: int):
    global _alerts
    before = len(_alerts)
    _alerts = [a for a in _alerts if a["id"] != alert_id]
    if len(_alerts) == before:
        raise HTTPException(status_code=404, detail="Alert not found.")
