# TradingView Integration Guide

This guide explains how to integrate the Hybrid Forecast system with TradingView for live trading with engulfing candle confirmation.

## Overview

The system combines:
1. **Probabilistic Deep Learning Predictions**: Direction (up/down) and level forecasts
2. **Engulfing Candle Confirmation**: Waits for bullish/bearish engulfing patterns before executing trades
3. **30-Day Live Testing**: Automated monitoring and execution framework

## Architecture

```
TradingView Chart
    ↓ (Pine Script detects pattern)
TradingView Alert
    ↓ (Webhook)
FastAPI /webhook endpoint
    ↓ (Checks prediction + pattern)
Alpaca Paper Trading API
    ↓
Trade Execution
```

## Setup Steps

### 1. Deploy the API Server

Start the FastAPI server:

```bash
cd /path/to/intraday-trading-system-1
source .venv/bin/activate
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

Or use Docker:

```bash
docker compose up api
```

### 2. Install Pine Script in TradingView

1. Open TradingView and go to the Pine Editor
2. Copy the contents of `tradingview/hybrid_forecast_strategy.pine`
3. Paste into the Pine Editor
4. Click "Save" and "Add to Chart"

### 3. Configure TradingView Alert

1. Right-click on the chart → "Add Alert"
2. Condition: Select your strategy (e.g., "Hybrid Forecast with Engulfing Confirmation")
3. Webhook URL: `http://your-server:8000/webhook`
4. Alert Message (JSON format):

```json
{
  "symbol": "{{ticker}}",
  "alert_name": "{{strategy.order.comment}}",
  "threshold": 0.6,
  "require_engulfing": true,
  "min_body_ratio": 1.5,
  "auto_trade": true
}
```

5. Frequency: "Once Per Bar Close"
6. Click "Create"

### 4. Configure Environment Variables

Set these in your `.env` file or environment:

```bash
# Alpaca API (required for auto-trading)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret

# Auto-trading settings
AUTO_TRADE_ENABLED=true
AUTO_TRADE_QTY=1

# Risk management
MAX_DAILY_LOSS_USD=500
MAX_POSITIONS_PER_SYMBOL=5
TRADE_COOLDOWN_SECONDS=300
MAX_NOTIONAL_PER_TRADE_USD=10000
SYMBOL_ALLOWLIST=AAPL,TSLA,BTC/USD,ETH/USD

# Market hours (UTC)
MARKET_OPEN_UTC=13:30
MARKET_CLOSE_UTC=20:00
```

## API Endpoints

### POST /signal

Get a trading signal with engulfing confirmation.

**Request:**
```json
{
  "symbol": "AAPL",
  "threshold": 0.6,
  "require_engulfing": true,
  "min_body_ratio": 1.5
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "direction_probability": 0.75,
  "direction_decision": 1,
  "level_mean": 150.25,
  "level_variance": 2.5,
  "engulfing_pattern": "bullish",
  "signal_confirmed": true,
  "recommendation": "BUY"
}
```

### POST /webhook

TradingView webhook endpoint (same as `/signal` but optimized for alerts).

**Request:** (from TradingView alert)
```json
{
  "symbol": "AAPL",
  "alert_name": "AAPL Bullish Signal",
  "threshold": 0.6,
  "require_engulfing": true,
  "auto_trade": true
}
```

**Response:**
```json
{
  "alert": "AAPL Bullish Signal",
  "decision": "long",
  "confidence": 0.75,
  "level_target": 150.25,
  "order_id": "abc123"
}
```

## 30-Day Live Trading Script

Run the automated 30-day trading script:

```bash
export API_BASE_URL=http://127.0.0.1:8000
export LIVE_TRADING_SYMBOLS=AAPL,TSLA
export LIVE_TRADING_THRESHOLD=0.6
export REQUIRE_ENGULFING=true
export AUTO_TRADE_ENABLED=true
export LIVE_TRADE_QTY=1
export CHECK_INTERVAL_SECONDS=60
export TRADING_DAYS=30

python scripts/live_trading_30days.py
```

The script will:
- Monitor symbols every minute during market hours
- Check for confirmed signals (prediction + engulfing pattern)
- Execute trades automatically (if enabled)
- Log all activity to `logs/live_trading_YYYYMMDD.jsonl`
- Generate performance report at the end

## Engulfing Pattern Detection

### Bullish Engulfing

A bullish engulfing pattern occurs when:
1. Previous candle is bearish (close < open)
2. Current candle is bullish (close > open)
3. Current candle's body completely engulfs previous candle's body
4. Current body is at least `min_body_ratio` times larger than previous body

### Bearish Engulfing

A bearish engulfing pattern occurs when:
1. Previous candle is bullish (close > open)
2. Current candle is bearish (close < open)
3. Current candle's body completely engulfs previous candle's body
4. Current body is at least `min_body_ratio` times larger than previous body

## Signal Confirmation Logic

When `require_engulfing=true`:

- **BUY Signal**: Model predicts UP (direction_probability >= threshold) **AND** bullish engulfing pattern detected
- **SELL Signal**: Model predicts DOWN (direction_probability < (1 - threshold)) **AND** bearish engulfing pattern detected
- **WAIT**: Prediction exists but pattern doesn't match → wait for confirmation

When `require_engulfing=false`:

- Uses model prediction directly without pattern confirmation

## Monitoring

### View Live Trades

```bash
# Tail the live trading log
tail -f logs/live_trading_*.jsonl

# View recent signals
grep "signal_check" logs/live_trading_*.jsonl | tail -20

# View executed trades
grep "trade_executed" logs/live_trading_*.jsonl | tail -20
```

### Dashboard

Access the dashboard at `http://localhost:8050` to view:
- Real-time predictions
- Recent trades
- Performance metrics
- Backtest results

### API Metrics

```bash
curl http://localhost:8000/metrics
```

## Troubleshooting

### No Signals Confirmed

- Check that model is trained and loaded: `curl http://localhost:8000/health`
- Verify dataset exists for symbol: `ls data/processed/sequences/{symbol}_intraday_latest.npz`
- Lower `threshold` if predictions are too conservative
- Set `require_engulfing=false` to test without pattern confirmation

### Webhook Not Receiving Alerts

- Verify TradingView alert is configured with correct webhook URL
- Check API server logs for incoming requests
- Ensure alert frequency is "Once Per Bar Close"
- Test webhook manually: `curl -X POST http://localhost:8000/webhook -H "Content-Type: application/json" -d '{"symbol":"AAPL","threshold":0.6}'`

### Trades Not Executing

- Verify `AUTO_TRADE_ENABLED=true`
- Check Alpaca API credentials are set
- Review risk management settings (daily loss, position limits, cooldown)
- Check market hours configuration
- Review `logs/trades.jsonl` for error messages

## Best Practices

1. **Start with Paper Trading**: Always test with `paper=true` in Alpaca before live trading
2. **Monitor Risk Limits**: Set conservative `MAX_DAILY_LOSS_USD` and `MAX_POSITIONS_PER_SYMBOL`
3. **Use Cooldowns**: Set `TRADE_COOLDOWN_SECONDS` to avoid overtrading
4. **Symbol Allowlist**: Restrict trading to known symbols via `SYMBOL_ALLOWLIST`
5. **Log Everything**: Review logs regularly to understand system behavior
6. **Gradual Rollout**: Start with 1-2 symbols, then expand after validation

## Performance Evaluation

After 30 days, review:

1. **Confirmation Rate**: `signals_confirmed / signals_checked`
2. **Win Rate**: Trades with positive P/L / Total trades
3. **Sharpe Ratio**: Risk-adjusted returns
4. **Max Drawdown**: Largest peak-to-trough decline
5. **Profit Factor**: Gross profit / Gross loss

Compare against baseline (prediction-only, no engulfing confirmation) to measure the value of pattern confirmation.

