"""Hybrid LSTM + Transformer probabilistic model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers


@dataclass
class HybridConfig:
    lstm_units: Tuple[int, ...] = (128, 64)
    dropout: float = 0.3
    num_heads: int = 4
    transformer_dim: int = 128
    transformer_ff: int = 256
    learning_rate: float = 1e-3
    monte_carlo_samples: int = 50


class HybridForecastModel:
    """Encapsulates the hybrid architecture and Monte Carlo dropout inference."""

    def __init__(self, input_shape: Tuple[int, int], config: HybridConfig | None = None) -> None:
        self.input_shape = input_shape
        self.config = config or HybridConfig()
        self.model = self._build_model()

    def _build_model(self) -> Model:
        sequence_length, num_features = self.input_shape
        inputs = layers.Input(shape=(sequence_length, num_features))

        x = inputs
        for units in self.config.lstm_units:
            x = layers.LSTM(units, return_sequences=True)(x)
            x = layers.Dropout(self.config.dropout)(x)

        attn_output = layers.MultiHeadAttention(
            num_heads=self.config.num_heads,
            key_dim=self.config.transformer_dim,
            dropout=self.config.dropout,
        )(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)

        ff = layers.Dense(self.config.transformer_ff, activation="relu")(x)
        ff = layers.Dropout(self.config.dropout)(ff)
        ff = layers.Dense(x.shape[-1])(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)

        pooled = layers.GlobalAveragePooling1D()(x)
        shared = layers.Dense(128, activation="relu")(pooled)
        shared = layers.Dropout(self.config.dropout)(shared)

        direction_logits = layers.Dense(1, activation="sigmoid", name="direction")(shared)
        level_output = layers.Dense(1, activation="linear", name="level")(shared)

        model = Model(inputs=inputs, outputs=[direction_logits, level_output])
        optimizer = optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss={"direction": "binary_crossentropy", "level": "mse"},
            metrics={"direction": ["accuracy"], "level": ["mae"]},
        )
        return model

    def predict_with_uncertainty(self, X: np.ndarray, samples: int | None = None) -> Dict[str, np.ndarray]:
        """Return mean and std predictions using Monte Carlo dropout."""
        samples = samples or self.config.monte_carlo_samples
        direction_preds = []
        level_preds = []
        for _ in range(samples):
            dir_out, lvl_out = self.model(X, training=True)
            direction_preds.append(dir_out.numpy())
            level_preds.append(lvl_out.numpy())
        direction_stack = np.stack(direction_preds, axis=0)
        level_stack = np.stack(level_preds, axis=0)
        return {
            "direction_mean": direction_stack.mean(axis=0).flatten(),
            "direction_std": direction_stack.std(axis=0).flatten(),
            "level_mean": level_stack.mean(axis=0).flatten(),
            "level_std": level_stack.std(axis=0).flatten(),
        }

