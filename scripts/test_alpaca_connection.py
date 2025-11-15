"""Smoke test script to verify Alpaca credentials and data access."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import get_settings  # noqa: E402
from data.alpaca_client import AlpacaService  # noqa: E402


def main() -> None:
    settings = get_settings()
    print("Project root:", PROJECT_ROOT)
    print("Data directories:", settings.data_raw_dir, settings.data_processed_dir)

    client = AlpacaService()
    status = client.get_account_status()
    print(f"Alpaca account status: {status}")

    symbols = client.list_active_assets(limit=3)
    print("Sample active symbols:", symbols)

    target_symbol = symbols[0] if symbols else "AAPL"
    bars = client.get_recent_bars(symbol=target_symbol)
    payload = bars.data
    if not payload:
        print(f"No bar data for {target_symbol}, retrying with AAPL.")
        bars = client.get_recent_bars(symbol="AAPL")
        payload = bars.data
    print("Fetched bar payload (truncated):")
    print(json.dumps({k: v[:3] for k, v in payload.items()}, default=str, indent=2))


if __name__ == "__main__":
    main()

