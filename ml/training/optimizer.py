"""
Optuna-based hyperparameter optimisation for all model types.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import optuna
from loguru import logger

from ml.models import GRUModel, HybridModel, LSTMModel, TransformerModel
from ml.training.trainer import ModelTrainer


def _create_model(trial: optuna.Trial, model_type: str, input_size: int, task: str):
    if model_type == "lstm":
        return LSTMModel(
            input_size=input_size,
            hidden_size=trial.suggest_categorical("hidden_size", [64, 128, 256]),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            use_attention=trial.suggest_categorical("use_attention", [True, False]),
            task=task,
        )
    if model_type == "gru":
        return GRUModel(
            input_size=input_size,
            hidden_size=trial.suggest_categorical("hidden_size", [64, 128, 256]),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            bidirectional=trial.suggest_categorical("bidirectional", [True, False]),
            task=task,
        )
    if model_type == "transformer":
        return TransformerModel(
            input_size=input_size,
            d_model=trial.suggest_categorical("d_model", [64, 128, 256]),
            nhead=trial.suggest_categorical("nhead", [4, 8]),
            num_encoder_layers=trial.suggest_int("num_encoder_layers", 1, 4),
            dropout=trial.suggest_float("dropout", 0.05, 0.3),
            task=task,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def optimise(
    model_type: str,
    input_size: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "regression",
    n_trials: int = 30,
    timeout: Optional[int] = 3600,
) -> optuna.Study:
    """
    Run Optuna study and return the study object.
    Best params accessible via study.best_params.
    """

    def objective(trial: optuna.Trial) -> float:
        model = _create_model(trial, model_type, input_size, task)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch = trial.suggest_categorical("batch_size", [32, 64, 128])

        trainer = ModelTrainer(
            model=model,
            task=task,
            learning_rate=lr,
            batch_size=batch,
            max_epochs=50,
            patience=7,
            model_name=f"optuna_{trial.number}",
        )
        trainer.fit(X_train, y_train, X_val, y_val)
        return min(trainer.history["val_loss"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(f"Best val loss: {study.best_value:.6f}")
    logger.info(f"Best params: {study.best_params}")
    return study
