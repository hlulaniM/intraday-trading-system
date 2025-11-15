"""Dataset helpers and training utilities for baseline and hybrid models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import yaml

from models.baselines import (
    BaselineResult,
    regression_metrics,
    train_arima_baseline,
    train_lstm_baseline,
    train_random_forest_baseline,
)
from models.hybrid_model import HybridConfig, HybridForecastModel


@dataclass
class SequenceDataset:
    train_X: np.ndarray
    train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    feature_names: Tuple[str, ...] | None = None
    close_index: int = -1

    @classmethod
    def from_npz(cls, path: str | Path, close_index: int = -1) -> "SequenceDataset":
        data = np.load(path, allow_pickle=True)
        feature_names = data.get("feature_names")
        return cls(
            train_X=data["train_X"],
            train_y=data["train_y"],
            val_X=data["val_X"],
            val_y=data["val_y"],
            test_X=data["test_X"],
            test_y=data["test_y"],
            feature_names=tuple(feature_names.tolist()) if feature_names is not None else None,
            close_index=close_index,
        )

    def _direction_labels(self, split: str) -> np.ndarray:
        X = getattr(self, f"{split}_X")
        y = getattr(self, f"{split}_y")
        prev_close = X[:, -1, self.close_index]
        return (y > prev_close).astype(np.float32)

    def direction_labels(self) -> Dict[str, np.ndarray]:
        return {
            "train": self._direction_labels("train"),
            "val": self._direction_labels("val"),
            "test": self._direction_labels("test"),
        }


def load_config_section(path: str, section: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get(section, {})


def train_all_baselines(dataset: SequenceDataset) -> Dict[str, BaselineResult]:
    """Train ARIMA, random forest, and LSTM baselines."""
    results: Dict[str, BaselineResult] = {}

    arima_result = train_arima_baseline(
        dataset.train_y,
        dataset.test_y,
    )
    results["arima"] = arima_result

    rf_result = train_random_forest_baseline(
        dataset.train_X,
        dataset.train_y,
        dataset.test_X,
        dataset.test_y,
    )
    results["random_forest"] = rf_result

    lstm_result = train_lstm_baseline(
        dataset.train_X,
        dataset.train_y,
        dataset.val_X,
        dataset.val_y,
        dataset.test_X,
        dataset.test_y,
    )
    results["lstm"] = lstm_result
    return results


class HybridTrainer:
    def __init__(self, dataset: SequenceDataset, config: HybridConfig | None = None) -> None:
        self.dataset = dataset
        self.config = config or HybridConfig()
        self.model = HybridForecastModel(
            input_shape=(dataset.train_X.shape[1], dataset.train_X.shape[2]),
            config=self.config,
        )

    def train(self, epochs: int = 50, batch_size: int = 32) -> tf.keras.callbacks.History:
        y_dir = self.dataset._direction_labels("train")
        y_dir_val = self.dataset._direction_labels("val")
        history = self.model.model.fit(
            self.dataset.train_X,
            {"direction": y_dir, "level": self.dataset.train_y},
            validation_data=(
                self.dataset.val_X,
                {"direction": y_dir_val, "level": self.dataset.val_y},
            ),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        )
        return history

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        y_dir_test = self.dataset._direction_labels("test")
        dir_pred, level_pred = self.model.model.predict(self.dataset.test_X, verbose=0)
        dir_metrics = direction_metrics(y_dir_test, dir_pred.flatten())
        level_metrics = regression_metrics(self.dataset.test_y, level_pred.flatten())
        return {"direction": dir_metrics, "level": level_metrics}


def direction_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_hat = (y_pred >= threshold).astype(np.int32)
    accuracy = float((y_hat == y_true).mean())
    precision = float((np.sum((y_hat == 1) & (y_true == 1)) / max(np.sum(y_hat == 1), 1e-8)))
    recall = float((np.sum((y_hat == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1e-8)))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-8))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

