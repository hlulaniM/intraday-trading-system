# Quick Start: TradingView Integration

## 5-Minute Setup

### 1. Start API Server

```bash
cd /path/to/intraday-trading-system-1
source .venv/bin/activate
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### 2. Install Pine Script

1. Open TradingView → Pine Editor
2. Copy contents of `tradingview/hybrid_forecast_strategy.pine`
3. Paste and click "Save" → "Add to Chart"

### 3. Create Alert

1. Right-click chart → "Add Alert"
2. Condition: Your strategy name
3. Webhook URL: `http://your-server-ip:8000/webhook`
4. Message:
```json
{"symbol":"{{ticker}}","threshold":0.6,"require_engulfing":true,"auto_trade":true}
```
5. Frequency: "Once Per Bar Close"
6. Create

### 4. Enable Auto-Trading (Optional)

```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
export AUTO_TRADE_ENABLED=true
```

Restart API server.

### 5. Test Signal Endpoint

```bash
curl -X POST http://localhost:8000/signal \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","threshold":0.6,"require_engulfing":true}'
```

## How It Works

1. **Model Prediction**: System predicts price direction (UP/DOWN) and target level
2. **Pattern Detection**: Fetches recent candles and detects engulfing patterns
3. **Signal Confirmation**: Only confirms when prediction + pattern align
4. **Trade Execution**: Executes via Alpaca when signal confirmed

## Signal Logic

- **BUY**: Prediction = UP (≥threshold) + Bullish Engulfing
- **SELL**: Prediction = DOWN (<1-threshold) + Bearish Engulfing
- **WAIT**: Prediction exists but pattern doesn't match

## 30-Day Live Testing

```bash
export LIVE_TRADING_SYMBOLS=AAPL,TSLA
export REQUIRE_ENGULFING=true
export AUTO_TRADE_ENABLED=true
export TRADING_DAYS=30

python scripts/live_trading_30days.py
```

## Troubleshooting

- **No signals**: Check model is loaded (`curl http://localhost:8000/health`)
- **Webhook not working**: Verify alert URL and message format
- **Trades not executing**: Check `AUTO_TRADE_ENABLED` and Alpaca credentials

See [Full Integration Guide](tradingview_integration.md) for details.

