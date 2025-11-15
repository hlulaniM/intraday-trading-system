"""FastAPI server exposing hybrid model predictions, TradingView webhooks, and monitoring."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from models.hybrid_model import HybridConfig, HybridForecastModel
from models.trainer import SequenceDataset

app = FastAPI(title="Hybrid Intraday Forecast API")

MODEL_PATH = Path("models/hybrid_model.keras")
FEATURE_NAMES_PATH = Path("data/processed/sequences/feature_names.npy")
DATASET_DIR = Path("data/processed/sequences")
MANIFEST_PATH = Path("data/raw/manifest.json")
BASELINE_METRICS = Path("data/processed/sequences/aapl_baselines_latest.json")
HYBRID_METRICS = Path("data/processed/sequences/aapl_hybrid_latest_metrics.json")
TRADES_LOG = Path("logs/trades.jsonl")
DEFAULT_CLOSE_INDEX = -1

AUTO_TRADE_ENABLED = os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true"
AUTO_TRADE_QTY = int(os.getenv("AUTO_TRADE_QTY", "1"))
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


class PredictionRequest(BaseModel):
    sequence: Optional[List[List[float]]] = None
    symbol: Optional[str] = None
    threshold: float = 0.5


class WebhookPayload(BaseModel):
    symbol: Optional[str] = None
    sequence: Optional[List[List[float]]] = None
    alert_name: Optional[str] = None
    threshold: float = 0.5
    auto_trade: Optional[bool] = None


class PredictionResponse(BaseModel):
    direction_probability: float
    direction_decision: int
    level_mean: float
    level_variance: float


class WebhookResponse(BaseModel):
    alert: str
    decision: str
    confidence: float
    level_target: float
    order_id: Optional[str] = None


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
trading_client: TradingClient | None = None
metrics_state = {
    "predict_requests": 0,
    "webhook_requests": 0,
    "auto_trades": 0,
    "prediction_latency_ms": 0.0,
    "errors": 0,
}


def _load_sequence_from_dataset(symbol: str) -> np.ndarray:
    dataset_path = DATASET_DIR / f"{symbol.lower()}_intraday_latest.npz"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"No dataset found for symbol {symbol}")
    dataset = SequenceDataset.from_npz(dataset_path, close_index=DEFAULT_CLOSE_INDEX)
    return dataset.test_X[-1]


def _latest_manifest_entry() -> Optional[dict]:
    if not MANIFEST_PATH.exists():
        return None
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data[-1] if data else None


def _load_metrics(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _append_trade_log(entry: dict) -> None:
    TRADES_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TRADES_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry) + "\n")


def _load_trades(limit: int = 10) -> List[dict]:
    if not TRADES_LOG.exists():
        return []
    lines = TRADES_LOG.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines[-limit:]]


@app.on_event("startup")
def load_model_on_startup() -> None:
    global model_cache, trading_client
    try:
        model_cache = _load_model()
    except RuntimeError as exc:  # pragma: no cover - startup log
        print(f"WARNING: {exc}")
    if AUTO_TRADE_ENABLED and ALPACA_API_KEY and ALPACA_SECRET_KEY:
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)


def _run_prediction(sequence: np.ndarray, threshold: float) -> PredictionResponse:
    start_time = time.perf_counter()
    if model_cache is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if sequence.ndim != 2:
        raise HTTPException(status_code=400, detail="Sequence must be 2D [timesteps, features].")
    sequence = np.expand_dims(sequence, axis=0)
    dir_mean, lvl_mean = model_cache.model.predict(sequence, verbose=0)
    direction_prob = float(dir_mean.flatten()[0])
    decision = int(direction_prob >= threshold)
    mc = model_cache.predict_with_uncertainty(sequence)
    metrics_state["prediction_latency_ms"] = (time.perf_counter() - start_time) * 1000
    return PredictionResponse(
        direction_probability=direction_prob,
        direction_decision=decision,
        level_mean=float(mc["level_mean"][0]),
        level_variance=float(mc["level_std"][0] ** 2),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    metrics_state["predict_requests"] += 1
    if request.sequence is not None:
        sequence = np.array(request.sequence, dtype=np.float32)
    elif request.symbol:
        sequence = _load_sequence_from_dataset(request.symbol)
    else:
        raise HTTPException(status_code=400, detail="Provide either 'sequence' or 'symbol'.")
    return _run_prediction(sequence, request.threshold)


@app.post("/webhook", response_model=WebhookResponse)
def tradingview_webhook(payload: WebhookPayload) -> WebhookResponse:
    metrics_state["webhook_requests"] += 1
    if payload.sequence is not None:
        sequence = np.array(payload.sequence, dtype=np.float32)
    elif payload.symbol:
        sequence = _load_sequence_from_dataset(payload.symbol)
    else:
        raise HTTPException(status_code=400, detail="Webhook requires 'symbol' or 'sequence'.")
    prediction = _run_prediction(sequence, payload.threshold)
    decision = "long" if prediction.direction_decision == 1 else "short"
    alert_name = payload.alert_name or payload.symbol or "unknown"
    auto_trade_flag = payload.auto_trade if payload.auto_trade is not None else AUTO_TRADE_ENABLED
    order_id = None
    if auto_trade_flag and trading_client and payload.symbol:
        try:
            order = trading_client.submit_order(
                order_data=MarketOrderRequest(
                    symbol=payload.symbol,
                    qty=AUTO_TRADE_QTY,
                    side=OrderSide.BUY if decision == "long" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
            order_id = order.id
            metrics_state["auto_trades"] += 1
            _append_trade_log(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": payload.symbol,
                    "decision": decision,
                    "confidence": prediction.direction_probability,
                    "order_id": order.id,
                    "qty": AUTO_TRADE_QTY,
                }
            )
        except Exception as exc:  # pragma: no cover
            metrics_state["errors"] += 1
            raise HTTPException(status_code=500, detail=f"Auto-trade failed: {exc}")
    return WebhookResponse(
        alert=alert_name,
        decision=decision,
        confidence=prediction.direction_probability,
        level_target=prediction.level_mean,
        order_id=order_id,
    )


@app.get("/metrics")
def metrics() -> dict:
    manifest_entry = _latest_manifest_entry()
    baseline = _load_metrics(BASELINE_METRICS)
    hybrid = _load_metrics(HYBRID_METRICS)
    return {
        "model_loaded": model_cache is not None,
        "last_ingestion": manifest_entry,
        "baseline_metrics": baseline,
        "hybrid_metrics": hybrid,
        "feature_names": feature_names,
        "metrics_state": metrics_state,
        "recent_trades": _load_trades(),
    }


@app.get("/metrics/prom")
def prom_metrics() -> str:
    lines = [
        f"predict_requests_total {metrics_state['predict_requests']}",
        f"webhook_requests_total {metrics_state['webhook_requests']}",
        f"auto_trades_total {metrics_state['auto_trades']}",
        f"prediction_latency_ms {metrics_state['prediction_latency_ms']}",
        f"errors_total {metrics_state['errors']}",
    ]
    manifest = _latest_manifest_entry()
    if manifest:
        lines.append(f"ingestion_rows_last {manifest.get('rows', 0)}")
    return "\n".join(lines)

