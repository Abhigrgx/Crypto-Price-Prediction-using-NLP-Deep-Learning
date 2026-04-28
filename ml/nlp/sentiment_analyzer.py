"""
Sentiment Analyzer using FinBERT (financial domain BERT).
Supports batch inference, time-aggregation and caching.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from loguru import logger


@dataclass
class SentimentResult:
    text: str
    label: str          # POSITIVE | NEUTRAL | NEGATIVE
    score: float        # confidence [0, 1]
    numeric: float      # +1 / 0 / -1 weighted by score


class SentimentAnalyzer:
    """
    Thin wrapper around any HuggingFace sequence-classification model.
    Defaults to ProsusAI/finbert (finance-tuned).
    """

    LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "HUGGINGFACE_MODEL", "ProsusAI/finbert"
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info(f"Loading sentiment model: {self.model_name} on {self.device}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self._pipeline = pipeline(
            "text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            device=0 if self.device == "cuda" else -1,
            truncation=True,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        logger.info("Sentiment model loaded.")

    def analyse(self, text: str) -> SentimentResult:
        """Classify a single text string."""
        if not text or not text.strip():
            return SentimentResult(text=text, label="NEUTRAL", score=1.0, numeric=0.0)
        result = self._pipeline(text)[0]  # type: ignore[index]
        label = result["label"].lower()  # type: ignore[index]
        score = float(result["score"])  # type: ignore[index]
        numeric = self.LABEL_MAP.get(label, 0.0) * score
        return SentimentResult(
            text=text,
            label=label.upper(),
            score=score,
            numeric=numeric,
        )

    def analyse_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Classify a list of texts efficiently in batches."""
        # Filter empties, keep index mapping
        valid_idx = [i for i, t in enumerate(texts) if t and t.strip()]
        valid_texts = [texts[i] for i in valid_idx]

        results: list[Optional[SentimentResult]] = [None] * len(texts)

        if valid_texts:
            raw = self._pipeline(valid_texts)  # type: ignore[index]
            for list_pos, orig_idx in enumerate(valid_idx):
                r = raw[list_pos]  # type: ignore[index]
                label = r["label"].lower()  # type: ignore[index]
                score = float(r["score"])  # type: ignore[index]
                numeric = self.LABEL_MAP.get(label, 0.0) * score
                results[orig_idx] = SentimentResult(
                    text=texts[orig_idx],
                    label=label.upper(),
                    score=score,
                    numeric=numeric,
                )

        # Fill None entries (empty texts)
        for i, r in enumerate(results):
            if r is None:
                results[i] = SentimentResult(
                    text=texts[i], label="NEUTRAL", score=1.0, numeric=0.0
                )

        return results  # type: ignore[return-value]

    # ── Aggregation ───────────────────────────────────────────────────────

    def aggregate_to_timeseries(
        self,
        records: list[dict],
        text_field: str = "title",
        time_field: str = "published_at",
        freq: str = "1h",
    ) -> pd.DataFrame:
        """
        Given a list of {text_field, time_field, ...} dicts,
        returns a DataFrame indexed by timestamp with columns:
          sentiment_score, sentiment_positive, sentiment_negative, news_count, social_score
        """
        if not records:
            return pd.DataFrame()

        texts = [r.get(text_field, "") for r in records]
        sentiments = self.analyse_batch(texts)

        rows = []
        for record, sent in zip(records, sentiments):
            ts_raw = record.get(time_field, "")
            try:
                ts = pd.to_datetime(ts_raw, utc=True)
            except Exception:
                continue
            rows.append(
                {
                    "timestamp": ts,
                    "numeric": sent.numeric,
                    "positive": 1 if sent.label == "POSITIVE" else 0,
                    "negative": 1 if sent.label == "NEGATIVE" else 0,
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        agg = df.resample(freq).agg(
            sentiment_score=("numeric", "mean"),
            sentiment_positive=("positive", "sum"),
            sentiment_negative=("negative", "sum"),
            news_count=("numeric", "count"),
        )
        # Social score: normalized weighted sentiment (−1 to +1)
        agg["social_score"] = agg["sentiment_score"].clip(-1, 1)
        agg.ffill(inplace=True)
        return agg
