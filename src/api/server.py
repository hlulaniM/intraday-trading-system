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
import logging
import sys
import tensorflow as tf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from models.hybrid_model import HybridConfig, HybridForecastModel
from models.trainer import SequenceDataset
from data.alpaca_client import AlpacaService
from utils.patterns import get_latest_pattern
import pandas as pd

app = FastAPI(title="Hybrid Intraday Forecast API")

# basic structured logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
    stream=sys.stdout,
)
logger = logging.getLogger("api")

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

# Phase 6 safeguards (env-configurable)
MAX_DAILY_LOSS_USD = float(os.getenv("MAX_DAILY_LOSS_USD", "0"))  # 0 disables
MAX_POSITIONS_PER_SYMBOL = int(os.getenv("MAX_POSITIONS_PER_SYMBOL", "0"))  # 0 disables
TRADE_COOLDOWN_SECONDS = int(os.getenv("TRADE_COOLDOWN_SECONDS", "0"))  # 0 disables
MAX_NOTIONAL_PER_TRADE_USD = float(os.getenv("MAX_NOTIONAL_PER_TRADE_USD", "0"))  # 0 disables
MARKET_OPEN_UTC = os.getenv("MARKET_OPEN_UTC", "13:30")  # NYSE 09:30 ET -> 13:30 UTC (standard)
MARKET_CLOSE_UTC = os.getenv("MARKET_CLOSE_UTC", "20:00")  # NYSE 16:00 ET -> 20:00 UTC (standard)
SYMBOL_ALLOWLIST = {s.strip().upper() for s in os.getenv("SYMBOL_ALLOWLIST", "").split(",") if s.strip()}


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
    require_engulfing: bool = True
    min_body_ratio: float = 1.5


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


class SignalRequest(BaseModel):
    symbol: str
    threshold: float = 0.5
    require_engulfing: bool = True
    min_body_ratio: float = 1.5


class SignalResponse(BaseModel):
    symbol: str
    direction_probability: float
    direction_decision: int
    level_mean: float
    level_variance: float
    engulfing_pattern: Optional[str] = None
    signal_confirmed: bool
    recommendation: str


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
alpaca_service: AlpacaService | None = None
metrics_state = {
    "predict_requests": 0,
    "webhook_requests": 0,
    "auto_trades": 0,
    "prediction_latency_ms": 0.0,
    "errors": 0,
    "last_trade_ts_per_symbol": {},  # str -> iso timestamp
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

def _notify_slack(text: str) -> None:
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return
    try:
        import requests
        requests.post(url, json={"text": text}, timeout=3)
    except Exception:
        pass


def _load_trades(limit: int = 10) -> List[dict]:
    if not TRADES_LOG.exists():
        return []
    lines = TRADES_LOG.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines[-limit:]]


def _risk_check_daily_loss() -> None:
    if not (MAX_DAILY_LOSS_USD and trading_client):
        return
    try:
        acct = trading_client.get_account()
        equity = float(acct.equity)
        last_equity = float(getattr(acct, "last_equity", equity))
        day_pl = equity - last_equity
        if day_pl <= -abs(MAX_DAILY_LOSS_USD):
            raise HTTPException(status_code=403, detail=f"Blocked by risk: daily P/L {day_pl:.2f} <= -{abs(MAX_DAILY_LOSS_USD):.2f}")
    except HTTPException:
        raise
    except Exception as exc:
        # If risk query fails, be conservative and block trading
        raise HTTPException(status_code=503, detail=f"Risk check failed (daily loss): {exc}")


def _risk_check_position_limit(symbol: str) -> None:
    if not (MAX_POSITIONS_PER_SYMBOL and trading_client):
        return
    try:
        # Count open position quantity via Alpaca
        pos = None
        try:
            pos = trading_client.get_open_position(symbol)
        except Exception:
            pos = None
        open_qty = float(getattr(pos, "qty", 0) or 0)
        if open_qty >= MAX_POSITIONS_PER_SYMBOL:
            raise HTTPException(status_code=403, detail=f"Blocked by risk: open qty {open_qty} >= {MAX_POSITIONS_PER_SYMBOL} for {symbol}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Risk check failed (positions): {exc}")


def _risk_check_cooldown(symbol: str) -> None:
    if not TRADE_COOLDOWN_SECONDS:
        return
    last_map: dict = metrics_state.get("last_trade_ts_per_symbol", {})  # type: ignore
    ts = last_map.get(symbol)
    if not ts:
        return
    try:
        last_dt = datetime.fromisoformat(ts)
        delta = datetime.utcnow() - last_dt
        if delta.total_seconds() < TRADE_COOLDOWN_SECONDS:
            raise HTTPException(status_code=429, detail=f"Blocked by cooldown: {int(TRADE_COOLDOWN_SECONDS - delta.total_seconds())}s remaining for {symbol}")
    except HTTPException:
        raise
    except Exception:
        # If parsing fails, ignore cooldown (do not block)
        return


def _risk_check_symbol(symbol: str) -> None:
    if SYMBOL_ALLOWLIST and symbol.upper() not in SYMBOL_ALLOWLIST:
        raise HTTPException(status_code=403, detail=f"Blocked by allowlist: {symbol} not permitted")


def _within_market_hours(now: Optional[datetime] = None) -> bool:
    now = now or datetime.utcnow()
    try:
        open_h, open_m = [int(x) for x in MARKET_OPEN_UTC.split(":")]
        close_h, close_m = [int(x) for x in MARKET_CLOSE_UTC.split(":")]
        start = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
        end = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
        return start <= now <= end
    except Exception:
        return True  # if misconfigured, do not block


def _risk_check_market_hours() -> None:
    if not _within_market_hours():
        raise HTTPException(status_code=403, detail="Blocked: outside configured market hours (UTC)")


@app.on_event("startup")
def load_model_on_startup() -> None:
    global model_cache, trading_client, alpaca_service
    try:
        model_cache = _load_model()
    except RuntimeError as exc:  # pragma: no cover - startup log
        print(f"WARNING: {exc}")
    if AUTO_TRADE_ENABLED and ALPACA_API_KEY and ALPACA_SECRET_KEY:
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        try:
            alpaca_service = AlpacaService()
        except Exception as exc:
            logger.warning(f"Failed to initialize AlpacaService: {exc}")


def _run_prediction(sequence: np.ndarray, threshold: float) -> PredictionResponse:
    start_time = time.perf_counter()
    if sequence.ndim != 2:
        raise HTTPException(status_code=400, detail="Sequence must be 2D [timesteps, features].")
    sequence = np.expand_dims(sequence, axis=0)
    # Fallback when model is not loaded: naive baseline from last close
    if model_cache is None:
        last_close = float(sequence[0, -1, DEFAULT_CLOSE_INDEX]) if DEFAULT_CLOSE_INDEX is not None else float(np.mean(sequence))
        metrics_state["prediction_latency_ms"] = (time.perf_counter() - start_time) * 1000
        return PredictionResponse(
            direction_probability=0.5,  # neutral
            direction_decision=int(0.5 >= threshold),
            level_mean=last_close,
            level_variance=0.0,
        )
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
        try:
            sequence = _load_sequence_from_dataset(request.symbol)
        except HTTPException:
            # Fallback synthetic sequence when dataset missing
            sequence = np.zeros((60, 1), dtype=np.float32)
    else:
        # Final fallback to synthetic sequence
        sequence = np.zeros((60, 1), dtype=np.float32)
    return _run_prediction(sequence, request.threshold)


@app.post("/webhook", response_model=WebhookResponse)
def tradingview_webhook(payload: WebhookPayload) -> WebhookResponse:
    """
    TradingView webhook endpoint with optional engulfing candle confirmation.
    
    If require_engulfing=True, only executes trades when:
    - Bullish prediction + bullish engulfing pattern = BUY
    - Bearish prediction + bearish engulfing pattern = SELL
    """
    metrics_state["webhook_requests"] += 1
    
    if not payload.symbol:
        raise HTTPException(status_code=400, detail="Webhook requires 'symbol' for engulfing confirmation.")
    
    # Use /signal endpoint logic for engulfing confirmation
    signal_request = SignalRequest(
        symbol=payload.symbol,
        threshold=payload.threshold,
        require_engulfing=payload.require_engulfing,
        min_body_ratio=payload.min_body_ratio,
    )
    signal = get_trading_signal(signal_request)
    
    # Only proceed if signal is confirmed
    if payload.require_engulfing and not signal.signal_confirmed:
        return WebhookResponse(
            alert=payload.alert_name or payload.symbol or "unknown",
            decision="wait",
            confidence=signal.direction_probability,
            level_target=signal.level_mean,
            order_id=None,
        )
    
    decision = "long" if signal.direction_decision == 1 else "short"
    alert_name = payload.alert_name or payload.symbol or "unknown"
    auto_trade_flag = payload.auto_trade if payload.auto_trade is not None else AUTO_TRADE_ENABLED
    order_id = None
    
    if auto_trade_flag and trading_client and payload.symbol:
        # Safeguards
        _risk_check_daily_loss()
        _risk_check_position_limit(payload.symbol)
        _risk_check_cooldown(payload.symbol)
        _risk_check_symbol(payload.symbol)
        _risk_check_market_hours()
        if MAX_NOTIONAL_PER_TRADE_USD:
            est_notional = float(AUTO_TRADE_QTY) * float(prediction.level_mean)
            if est_notional > MAX_NOTIONAL_PER_TRADE_USD:
                raise HTTPException(status_code=403, detail=f"Blocked by notional cap: est ${est_notional:.2f} > ${MAX_NOTIONAL_PER_TRADE_USD:.2f}")
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
                    "confidence": signal.direction_probability,
                    "engulfing_pattern": signal.engulfing_pattern,
                    "order_id": order.id,
                    "qty": AUTO_TRADE_QTY,
                }
            )
            # update cooldown map
            last_map: dict = metrics_state.get("last_trade_ts_per_symbol", {})  # type: ignore
            last_map[payload.symbol] = datetime.utcnow().isoformat()
            metrics_state["last_trade_ts_per_symbol"] = last_map
        except Exception as exc:  # pragma: no cover
            metrics_state["errors"] += 1
            _notify_slack(f"Auto-trade failed for {payload.symbol}: {exc}")
            raise HTTPException(status_code=500, detail=f"Auto-trade failed: {exc}")
    return WebhookResponse(
        alert=alert_name,
        decision=decision,
        confidence=signal.direction_probability,
        level_target=signal.level_mean,
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


@app.post("/signal", response_model=SignalResponse)
def get_trading_signal(request: SignalRequest) -> SignalResponse:
    """
    Get a trading signal combining model prediction with engulfing candle confirmation.
    
    This endpoint:
    1. Gets the model's direction prediction and level forecast
    2. Fetches recent OHLCV data from Alpaca
    3. Detects bullish/bearish engulfing patterns
    4. Confirms signal only if prediction aligns with pattern (if require_engulfing=True)
    """
    metrics_state["predict_requests"] += 1
    
    # Get prediction
    try:
        sequence = _load_sequence_from_dataset(request.symbol)
    except HTTPException:
        sequence = np.zeros((60, 1), dtype=np.float32)
    
    prediction = _run_prediction(sequence, request.threshold)
    
    # Get recent bars for pattern detection
    engulfing_pattern = None
    signal_confirmed = False
    recommendation = "HOLD"
    
    if alpaca_service:
        try:
            from datetime import timedelta, timezone
            bars = alpaca_service.get_recent_bars(
                symbol=request.symbol,
                lookback_minutes=10,  # Just need last few bars for pattern
            )
            if bars.data and request.symbol in bars.data:
                bar_list = bars.data[request.symbol]
                if len(bar_list) >= 2:
                    # Convert to DataFrame
                    df_data = []
                    for bar in bar_list[-10:]:  # Last 10 bars
                        df_data.append({
                            "timestamp": bar.timestamp,
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": float(bar.volume),
                        })
                    df = pd.DataFrame(df_data)
                    df.set_index("timestamp", inplace=True)
                    
                    # Detect pattern
                    pattern = get_latest_pattern(df, min_body_ratio=request.min_body_ratio)
                    engulfing_pattern = pattern["type"]
                    
                    # Confirm signal
                    if request.require_engulfing:
                        if prediction.direction_decision == 1 and engulfing_pattern == "bullish":
                            signal_confirmed = True
                            recommendation = "BUY"
                        elif prediction.direction_decision == 0 and engulfing_pattern == "bearish":
                            signal_confirmed = True
                            recommendation = "SELL"
                        else:
                            recommendation = "WAIT_FOR_PATTERN"
                    else:
                        # No engulfing required, use prediction directly
                        signal_confirmed = True
                        recommendation = "BUY" if prediction.direction_decision == 1 else "SELL"
        except Exception as exc:
            logger.warning(f"Pattern detection failed for {request.symbol}: {exc}")
            # Fallback: use prediction without pattern confirmation
            signal_confirmed = not request.require_engulfing
            recommendation = "BUY" if prediction.direction_decision == 1 else "SELL"
    else:
        # No Alpaca service, use prediction only
        signal_confirmed = not request.require_engulfing
        recommendation = "BUY" if prediction.direction_decision == 1 else "SELL"
    
    return SignalResponse(
        symbol=request.symbol,
        direction_probability=prediction.direction_probability,
        direction_decision=prediction.direction_decision,
        level_mean=prediction.level_mean,
        level_variance=prediction.level_variance,
        engulfing_pattern=engulfing_pattern,
        signal_confirmed=signal_confirmed,
        recommendation=recommendation,
    )


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model_loaded": model_cache is not None,
        "auto_trade_enabled": AUTO_TRADE_ENABLED,
        "cooldown_s": TRADE_COOLDOWN_SECONDS,
        "max_daily_loss_usd": MAX_DAILY_LOSS_USD,
        "max_positions_per_symbol": MAX_POSITIONS_PER_SYMBOL,
        "max_notional_per_trade_usd": MAX_NOTIONAL_PER_TRADE_USD,
        "market_open_utc": MARKET_OPEN_UTC,
        "market_close_utc": MARKET_CLOSE_UTC,
        "symbol_allowlist_enabled": bool(SYMBOL_ALLOWLIST),
    }

