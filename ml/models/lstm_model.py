"""
LSTM model for crypto price regression and trend classification.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Stacked LSTM with optional attention over the sequence,
    dropout regularisation, and dual output heads.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
        task: str = "regression",  # "regression" | "classification"
    ) -> None:
        super().__init__()
        self.task = task
        self.use_attention = use_attention
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: (batch, 1)
        """
        out, _ = self.lstm(x)  # (batch, seq, hidden)

        if self.use_attention:
            # Attention weighting over time steps
            attn_weights = torch.softmax(self.attention(out), dim=1)  # (batch, seq, 1)
            context = (attn_weights * out).sum(dim=1)                 # (batch, hidden)
        else:
            context = out[:, -1, :]                                   # last hidden state

        context = self.dropout(context)
        logit = self.fc(context)

        if self.task == "classification":
            return torch.sigmoid(logit)
        return logit
