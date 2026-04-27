"""
Universal model trainer with:
  - Early stopping
  - Learning-rate scheduling
  - Checkpoint saving
  - TensorBoard / W&B optional logging
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.triggered = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


class ModelTrainer:
    """
    Train any PyTorch model on windowed time-series data.
    Supports both regression (MSE) and classification (BCE) tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        task: str = "regression",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 15,
        device: Optional[str] = None,
        checkpoint_dir: str = "ml/saved_models",
        model_name: str = "model",
    ) -> None:
        self.model = model
        self.task = task
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.MSELoss() if task == "regression" else nn.BCELoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        self.early_stopping = EarlyStopping(patience=patience)
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _model_requires_sentiment(self) -> bool:
        """Detect hybrid-like models that expect (market, sentiment)."""
        return hasattr(self.model, "sentiment_mlp")

    def _zero_sentiment(self, batch_size: int) -> torch.Tensor:
        """Fallback sentiment tensor when sentiment data is unavailable."""
        return torch.zeros((batch_size, 5), dtype=torch.float32, device=self.device)

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
        sentiment: Optional[np.ndarray] = None,
    ) -> DataLoader:
        tensors = [torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)]
        if sentiment is not None:
            tensors.append(torch.tensor(sentiment, dtype=torch.float32))
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _step(self, batch: tuple, train: bool = True) -> float:
        has_sentiment = len(batch) == 3
        if has_sentiment:
            X_b, y_b, s_b = [t.to(self.device) for t in batch]
            preds = self.model(X_b, s_b).squeeze(-1)
        else:
            X_b, y_b = [t.to(self.device) for t in batch]
            if self._model_requires_sentiment():
                s_b = self._zero_sentiment(X_b.shape[0])
                preds = self.model(X_b, s_b).squeeze(-1)
            else:
                preds = self.model(X_b).squeeze(-1)

        loss = self.criterion(preds, y_b)
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        return loss.item()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sentiment_train: Optional[np.ndarray] = None,
        sentiment_val: Optional[np.ndarray] = None,
    ) -> dict[str, list[float]]:
        train_loader = self._make_loader(X_train, y_train, shuffle=True, sentiment=sentiment_train)
        val_loader = self._make_loader(X_val, y_val, shuffle=False, sentiment=sentiment_val)

        best_val_loss = float("inf")

        for epoch in range(1, self.max_epochs + 1):
            # ── Training ──────────────────────────────────────────────────
            self.model.train()
            train_losses = [self._step(batch, train=True) for batch in train_loader]

            # ── Validation ────────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_losses = [self._step(batch, train=False) for batch in val_loader]

            t_loss = float(np.mean(train_losses))
            v_loss = float(np.mean(val_losses))
            self.history["train_loss"].append(t_loss)
            self.history["val_loss"].append(v_loss)
            self.scheduler.step(v_loss)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:>4}/{self.max_epochs} | train={t_loss:.6f} | val={v_loss:.6f}")

            # Checkpoint best model
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                self._save(f"best_{self.model_name}.pt")

            if self.early_stopping(v_loss):
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        # Load best weights before returning
        self._load(f"best_{self.model_name}.pt")
        return self.history

    def predict(
        self,
        X: np.ndarray,
        sentiment: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.model.eval()
        loader = self._make_loader(
            X,
            np.zeros(len(X)),
            shuffle=False,
            sentiment=sentiment,
        )
        preds = []
        with torch.no_grad():
            for batch in loader:
                has_sentiment = len(batch) == 3
                if has_sentiment:
                    X_b, _, s_b = [t.to(self.device) for t in batch]
                    out = self.model(X_b, s_b).squeeze(-1)
                else:
                    X_b, _ = [t.to(self.device) for t in batch]
                    if self._model_requires_sentiment():
                        s_b = self._zero_sentiment(X_b.shape[0])
                        out = self.model(X_b, s_b).squeeze(-1)
                    else:
                        out = self.model(X_b).squeeze(-1)
                preds.extend(out.cpu().numpy().tolist())
        return np.array(preds)

    def _save(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)

    def _load(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location=self.device))
