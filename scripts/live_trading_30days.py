"""30-day live trading script with engulfing candle confirmation.

This script:
1. Monitors specified symbols for trading signals
2. Uses /signal endpoint to get predictions + engulfing confirmation
3. Executes trades only when signal is confirmed
4. Logs all activity for 30 trading days
5. Generates performance report at the end
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import get_settings

settings = get_settings()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
SYMBOLS = os.getenv("LIVE_TRADING_SYMBOLS", "AAPL,TSLA").split(",")
THRESHOLD = float(os.getenv("LIVE_TRADING_THRESHOLD", "0.6"))
REQUIRE_ENGULFING = os.getenv("REQUIRE_ENGULFING", "true").lower() == "true"
MIN_BODY_RATIO = float(os.getenv("MIN_BODY_RATIO", "1.5"))
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "60"))  # Check every minute
TRADING_DAYS = int(os.getenv("TRADING_DAYS", "30"))
TRADE_QTY = int(os.getenv("LIVE_TRADE_QTY", "1"))

# Alpaca setup
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
AUTO_TRADE = os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true"

# Logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LIVE_TRADING_LOG = LOG_DIR / f"live_trading_{datetime.now().strftime('%Y%m%d')}.jsonl"
PERFORMANCE_LOG = LOG_DIR / f"live_trading_performance_{datetime.now().strftime('%Y%m%d')}.json"


def log_entry(entry: dict) -> None:
    """Append entry to live trading log."""
    with LIVE_TRADING_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_signal(symbol: str) -> Optional[dict]:
    """Get trading signal from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/signal",
            json={
                "symbol": symbol,
                "threshold": THRESHOLD,
                "require_engulfing": REQUIRE_ENGULFING,
                "min_body_ratio": MIN_BODY_RATIO,
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting signal for {symbol}: {e}")
        return None


def execute_trade(symbol: str, side: str, trading_client: TradingClient) -> Optional[str]:
    """Execute trade via Alpaca."""
    try:
        order = trading_client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                qty=TRADE_QTY,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        )
        return order.id
    except Exception as e:
        print(f"Error executing trade for {symbol}: {e}")
        return None


def is_market_open() -> bool:
    """Check if market is currently open (simplified - assumes US market hours)."""
    now = datetime.now(timezone.utc)
    # NYSE: 13:30-20:00 UTC (09:30-16:00 ET)
    market_open = now.replace(hour=13, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=20, minute=0, second=0, microsecond=0)
    # Skip weekends (simplified)
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    return market_open <= now <= market_close


def run_live_trading() -> None:
    """Main live trading loop for 30 trading days."""
    print(f"Starting 30-day live trading session")
    print(f"Symbols: {SYMBOLS}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Require Engulfing: {REQUIRE_ENGULFING}")
    print(f"Auto Trade: {AUTO_TRADE}")
    print(f"Log file: {LIVE_TRADING_LOG}")
    
    trading_client = None
    if AUTO_TRADE and ALPACA_API_KEY and ALPACA_SECRET_KEY:
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        print("Alpaca trading client initialized (paper trading)")
    
    start_date = datetime.now(timezone.utc)
    end_date = start_date + timedelta(days=TRADING_DAYS * 7 // 5)  # Approximate calendar days
    trading_days_elapsed = 0
    last_trade_per_symbol: dict[str, datetime] = {}
    signals_checked = 0
    signals_confirmed = 0
    trades_executed = 0
    
    log_entry({
        "event": "session_start",
        "timestamp": start_date.isoformat(),
        "config": {
            "symbols": SYMBOLS,
            "threshold": THRESHOLD,
            "require_engulfing": REQUIRE_ENGULFING,
            "auto_trade": AUTO_TRADE,
        },
    })
    
    try:
        while datetime.now(timezone.utc) < end_date and trading_days_elapsed < TRADING_DAYS:
            if not is_market_open():
                print(f"Market closed. Waiting {CHECK_INTERVAL_SECONDS}s...")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue
            
            for symbol in SYMBOLS:
                symbol = symbol.strip().upper()
                
                # Get signal
                signal = get_signal(symbol)
                if not signal:
                    continue
                
                signals_checked += 1
                timestamp = datetime.now(timezone.utc)
                
                log_entry({
                    "event": "signal_check",
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "signal": signal,
                })
                
                # Check if signal is confirmed
                if signal.get("signal_confirmed") and signal.get("recommendation") in ["BUY", "SELL"]:
                    signals_confirmed += 1
                    recommendation = signal["recommendation"]
                    
                    # Cooldown check (avoid trading too frequently)
                    last_trade = last_trade_per_symbol.get(symbol)
                    if last_trade and (timestamp - last_trade).total_seconds() < 300:  # 5 min cooldown
                        print(f"{symbol}: Signal confirmed but in cooldown")
                        continue
                    
                    print(f"{symbol}: {recommendation} signal confirmed!")
                    print(f"  Direction Prob: {signal['direction_probability']:.2f}")
                    print(f"  Engulfing Pattern: {signal.get('engulfing_pattern', 'N/A')}")
                    print(f"  Target Level: {signal['level_mean']:.2f}")
                    
                    # Execute trade if auto-trade enabled
                    order_id = None
                    if AUTO_TRADE and trading_client:
                        order_id = execute_trade(symbol, recommendation, trading_client)
                        if order_id:
                            trades_executed += 1
                            last_trade_per_symbol[symbol] = timestamp
                            print(f"  Order ID: {order_id}")
                    
                    log_entry({
                        "event": "trade_executed",
                        "timestamp": timestamp.isoformat(),
                        "symbol": symbol,
                        "recommendation": recommendation,
                        "signal": signal,
                        "order_id": order_id,
                    })
                else:
                    print(f"{symbol}: Signal not confirmed (recommendation: {signal.get('recommendation', 'N/A')})")
            
            # Sleep before next check
            time.sleep(CHECK_INTERVAL_SECONDS)
            
            # Update trading days (simplified - assumes 1 day per 24 hours of market hours)
            if datetime.now(timezone.utc).date() != start_date.date():
                trading_days_elapsed += 1
                start_date = datetime.now(timezone.utc)
                print(f"\n=== Trading Day {trading_days_elapsed}/{TRADING_DAYS} ===\n")
    
    except KeyboardInterrupt:
        print("\nStopping live trading (interrupted by user)")
    finally:
        # Generate performance report
        end_timestamp = datetime.now(timezone.utc)
        performance = {
            "start_date": start_date.isoformat(),
            "end_date": end_timestamp.isoformat(),
            "trading_days_elapsed": trading_days_elapsed,
            "signals_checked": signals_checked,
            "signals_confirmed": signals_confirmed,
            "trades_executed": trades_executed,
            "confirmation_rate": signals_confirmed / signals_checked if signals_checked > 0 else 0,
        }
        
        with PERFORMANCE_LOG.open("w", encoding="utf-8") as f:
            json.dump(performance, f, indent=2)
        
        log_entry({
            "event": "session_end",
            "timestamp": end_timestamp.isoformat(),
            "performance": performance,
        })
        
        print("\n=== Live Trading Session Complete ===")
        print(f"Trading Days: {trading_days_elapsed}/{TRADING_DAYS}")
        print(f"Signals Checked: {signals_checked}")
        print(f"Signals Confirmed: {signals_confirmed}")
        print(f"Trades Executed: {trades_executed}")
        print(f"Confirmation Rate: {performance['confirmation_rate']:.2%}")
        print(f"\nPerformance log: {PERFORMANCE_LOG}")


if __name__ == "__main__":
    run_live_trading()

