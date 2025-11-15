"""Classical baseline models for comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers


@dataclass
class BaselineResult:
    name: str
    metrics: Dict[str, float]
    predictions: np.ndarray


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)))
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def train_arima_baseline(series: np.ndarray, test_series: np.ndarray, order: Tuple[int, int, int] = (5, 1, 0)) -> BaselineResult:
    """Fit an ARIMA model on a 1D series and forecast the test horizon."""
    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test_series))
    metrics = regression_metrics(test_series, forecast)
    return BaselineResult(name="arima", metrics=metrics, predictions=np.array(forecast))


def train_random_forest_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 200,
    max_depth: int | None = None,
) -> BaselineResult:
    """Random forest regression over flattened sequences."""
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_flat, y_train)
    preds = model.predict(X_test_flat)
    metrics = regression_metrics(y_test, preds)
    return BaselineResult(name="random_forest", metrics=metrics, predictions=preds)


def train_lstm_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> BaselineResult:
    """Simple LSTM regression baseline."""
    sequence_length = X_train.shape[1]
    num_features = X_train.shape[2]

    model = models.Sequential(
        [
            layers.Input(shape=(sequence_length, num_features)),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
    es = callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )
    preds = model.predict(X_test, verbose=0).flatten()
    metrics = regression_metrics(y_test, preds)
    return BaselineResult(name="lstm", metrics=metrics, predictions=preds)

