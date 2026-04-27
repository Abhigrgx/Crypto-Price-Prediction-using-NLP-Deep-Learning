"""
GRU model for crypto price forecasting.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    Bidirectional GRU with layer normalisation and dual heads.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        task: str = "regression",
    ) -> None:
        super().__init__()
        self.task = task
        self.bidirectional = bidirectional
        direction_factor = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * direction_factor)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * direction_factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)                   # (batch, seq, hidden * dirs)
        out = self.layer_norm(out[:, -1, :])   # last step
        out = self.dropout(out)
        logit = self.fc(out)

        if self.task == "classification":
            return torch.sigmoid(logit)
        return logit
