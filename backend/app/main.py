"""
FastAPI application entry point.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.database import engine, Base
from app.routers import market, prediction, sentiment, alerts


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────
    logger.info("Starting up – creating database tables…")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready.")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────
    await engine.dispose()
    logger.info("Database connections closed.")


app = FastAPI(
    title="Crypto Price Prediction API",
    description="NLP + Deep Learning powered cryptocurrency forecasting platform.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────
app.include_router(market.router, prefix="/api/v1/market", tags=["Market Data"])
app.include_router(sentiment.router, prefix="/api/v1/sentiment", tags=["Sentiment"])
app.include_router(prediction.router, prefix="/api/v1/prediction", tags=["Prediction"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["Alerts"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Crypto Prediction API"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
