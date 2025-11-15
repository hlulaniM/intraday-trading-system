"""Command-line entry point for collecting intraday data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.collector import CollectorConfig, IntradayCollector, parse_timeframe  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger("collect_intraday_script")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect intraday market data from Alpaca.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config.")
    parser.add_argument("--symbols", nargs="*", help="Override symbol list.")
    parser.add_argument("--lookback-days", type=int, help="Override lookback window.")
    parser.add_argument("--timeframe", help="Override timeframe (e.g., 1Min, 5Min).")
    parser.add_argument("--format", choices=["parquet", "csv"], help="Storage format.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        logger.warning("Config file not found, using defaults", extra={"path": path})
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_collector_config(args: argparse.Namespace, base_config: dict) -> CollectorConfig:
    cfg = CollectorConfig.from_dict(base_config)
    if args.symbols:
        cfg.symbols = args.symbols
    if args.lookback_days:
        cfg.lookback_days = args.lookback_days
    if args.timeframe:
        cfg.timeframe = parse_timeframe(args.timeframe)
    if args.format:
        cfg.storage_format = args.format
    return cfg


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    collector_config = build_collector_config(args, base_config)
    collector = IntradayCollector(collector_config)
    stored_files = collector.collect_all()
    logger.info("Collection run completed", extra={"files": [str(f) for f in stored_files]})
    print(json.dumps({"stored_files": [str(f) for f in stored_files]}, indent=2))


if __name__ == "__main__":
    main()

