"""FastAPI server exposing hybrid model predictions."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf

from models.hybrid_model import HybridConfig, HybridForecastModel

app = FastAPI(title="Hybrid Intraday Forecast API")

MODEL_PATH = Path("models/hybrid_model.keras")
FEATURE_NAMES_PATH = Path("data/processed/sequences/feature_names.npy")


class PredictionRequest(BaseModel):
    sequence: List[List[float]]
    threshold: Optional[float] = 0.5


class PredictionResponse(BaseModel):
    direction_probability: float
    direction_decision: int
    level_mean: float
    level_variance: float


def _load_model() -> HybridForecastModel:
    if not MODEL_PATH.exists():
        raise RuntimeError("Saved model not found. Train and save the hybrid model first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    hybrid = HybridForecastModel(input_shape=model.input_shape[1:])  # type: ignore
    hybrid.model = model
    return hybrid


def _load_feature_names() -> List[str]:
    if not FEATURE_NAMES_PATH.exists():
        return []
    arr = np.load(FEATURE_NAMES_PATH, allow_pickle=True)
    return arr.tolist()


model_cache: HybridForecastModel | None = None
feature_names: List[str] = _load_feature_names()


@app.on_event("startup")
def load_model_on_startup() -> None:
    global model_cache
    try:
        model_cache = _load_model()
    except RuntimeError as exc:  # pragma: no cover - startup log
        print(f"WARNING: {exc}")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if model_cache is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    sequence = np.array(request.sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise HTTPException(status_code=400, detail="Sequence must be 2D [timesteps, features].")
    sequence = np.expand_dims(sequence, axis=0)
    dir_mean, lvl_mean = model_cache.model.predict(sequence, verbose=0)
    direction_prob = float(dir_mean.flatten()[0])
    decision = int(direction_prob >= request.threshold)
    mc = model_cache.predict_with_uncertainty(sequence)
    return PredictionResponse(
        direction_probability=direction_prob,
        direction_decision=decision,
        level_mean=float(mc["level_mean"][0]),
        level_variance=float(mc["level_std"][0] ** 2),
    )

