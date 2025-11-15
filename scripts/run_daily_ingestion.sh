#!/bin/bash
set -euo pipefail
source /Users/hlulaninobela/intraday-trading-system-1/.venv/bin/activate
cd /Users/hlulaninobela/intraday-trading-system-1
PYTHONPATH=src ALPACA_API_KEY="$ALPACA_API_KEY" ALPACA_SECRET_KEY="$ALPACA_SECRET_KEY" ALPACA_BASE_URL="https://paper-api.alpaca.markets/v2" python scripts/collect_intraday_data.py --symbols AAPL TSLA BTC/USD ETH/USD --lookback-days 1 --timeframe 1Min --format parquet
