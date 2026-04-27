"""
Hybrid model: LSTM encoder + Transformer attention + Sentiment fusion.

Architecture:
  ┌─────────────────────┐   ┌──────────────────────┐
  │  Market LSTM branch │   │  Sentiment MLP branch │
  └────────┬────────────┘   └──────────┬────────────┘
           │                           │
           └──────── Fusion layer ─────┘
                          │
                    Output head
"""
from __future__ import annotations

import torch
import torch.nn as nn
from ml.models.transformer_model import PositionalEncoding


class HybridModel(nn.Module):
    """
    Combines:
      - LSTM encoder for sequential market features
      - Multi-head self-attention layer
      - Dedicated MLP branch for aggregated sentiment features
      - Gated fusion of both branches
    """

    def __init__(
        self,
        market_input_size: int,
        sentiment_input_size: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        d_model: int = 128,
        nhead: int = 8,
        dropout: float = 0.2,
        task: str = "regression",
    ) -> None:
        super().__init__()
        self.task = task

        # ── Market branch ─────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=market_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.pos_enc = PositionalEncoding(lstm_hidden, dropout=dropout)
        attn_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden,
            nhead=nhead,
            dim_feedforward=lstm_hidden * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_encoder = nn.TransformerEncoder(attn_layer, num_layers=1)
        self.market_norm = nn.LayerNorm(lstm_hidden)

        # ── Sentiment branch ──────────────────────────────────────────────
        self.sentiment_mlp = nn.Sequential(
            nn.Linear(sentiment_input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # ── Gated fusion ──────────────────────────────────────────────────
        fusion_in = lstm_hidden + 32
        self.gate = nn.Sequential(
            nn.Linear(fusion_in, fusion_in),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(fusion_in)

        # ── Output head ───────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        market: torch.Tensor,
        sentiment: torch.Tensor,
    ) -> torch.Tensor:
        """
        market    : (batch, seq_len, market_input_size)
        sentiment : (batch, sentiment_input_size)   – per-sample aggregated features
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(market)              # (batch, seq, hidden)
        lstm_out = self.pos_enc(lstm_out)
        context = self.attn_encoder(lstm_out)        # (batch, seq, hidden)
        context = self.market_norm(context[:, -1, :])  # (batch, hidden)

        # Sentiment encoding
        sent_emb = self.sentiment_mlp(sentiment)      # (batch, 32)

        # Fusion with gating
        fused = torch.cat([context, sent_emb], dim=-1)  # (batch, hidden+32)
        gate = self.gate(fused)
        fused = self.fusion_norm(fused * gate)

        logit = self.head(fused)
        if self.task == "classification":
            return torch.sigmoid(logit)
        return logit
