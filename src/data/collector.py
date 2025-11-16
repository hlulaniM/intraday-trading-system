"""Market data collection utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Sequence
import json

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame

from config import get_settings
from data.alpaca_client import AlpacaService
from utils.logger import setup_logger
from utils.symbols import sanitize_symbol

logger = setup_logger("data_collector")


def parse_timeframe(value: str) -> TimeFrame:
    """Convert a string timeframe (e.g., '1Min') into Alpaca's TimeFrame enum."""
    normalized = value.strip().lower()
    mapping = {
        "1min": TimeFrame.Minute,
        "5min": TimeFrame.Minute,  # fallback; adjust if API exposes more enums
        "15min": TimeFrame.Minute,
        "30min": TimeFrame.Minute,
        "1hour": TimeFrame.Hour,
        "1day": TimeFrame.Day,
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported timeframe '{value}'. Supported values: {', '.join(mapping.keys())}"
        )
    return mapping[normalized]


@dataclass
class CollectorConfig:
    """Configuration for minute-level collection."""

    symbols: Sequence[str]
    timeframe: TimeFrame = TimeFrame.Minute
    lookback_days: int = 7
    storage_format: str = "parquet"
    feed: DataFeed = DataFeed.IEX
    raw_data_path: Path = field(default_factory=lambda: get_settings().data_raw_dir)
    manifest_path: Path | None = None
    max_symbols_per_run: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "CollectorConfig":
        collection = data.get("data_collection", {})
        timeframe_str = collection.get("timeframe", "1Min")
        return cls(
            symbols=collection.get("symbols", ["AAPL"]),
            timeframe=parse_timeframe(timeframe_str),
            lookback_days=int(collection.get("lookback_days", 7)),
            storage_format=collection.get("storage", {}).get("format", "parquet"),
            raw_data_path=Path(collection.get("storage", {}).get("raw_data_path", "./data/raw")),
            manifest_path=Path(collection.get("manifest_path", "./data/raw/manifest.json")),
            max_symbols_per_run=collection.get("max_symbols_per_run"),
        )


class IntradayCollector:
    """Handles retrieving and persisting intraday data."""

    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self.service = AlpacaService()
        self.raw_path = config.raw_data_path
        self.raw_path.mkdir(parents=True, exist_ok=True)

    def collect_all(self) -> List[Path]:
        """Collect data for every configured symbol."""
        stored_files: List[Path] = []
        symbols = list(self.config.symbols)
        if self.config.max_symbols_per_run:
            symbols = symbols[: self.config.max_symbols_per_run]
        for symbol in symbols:
            try:
                stored_files.extend(self.collect_symbol(symbol))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(
                    "Failed to collect data",
                    extra={"symbol": symbol, "error": str(exc)},
                )
        return stored_files

    def collect_symbol(self, symbol: str) -> List[Path]:
        """Collect data for a single symbol and persist to disk."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.config.lookback_days)
        logger.info(
            "Collecting intraday data",
            extra={
                "symbol": symbol,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "timeframe": self.config.timeframe.value,
            },
        )
        df = self._fetch_dataframe(symbol, start, end)
        if df.empty:
            logger.warning("No data returned for symbol", extra={"symbol": symbol})
            return []

        safe_symbol = sanitize_symbol(symbol)
        symbol_dir = (self.raw_path / safe_symbol).resolve()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        start_tag = start.strftime("%Y%m%d")
        end_tag = end.strftime("%Y%m%d")
        file_name = f"{safe_symbol}_{start_tag}_{end_tag}.{self._extension}"
        file_path = symbol_dir / file_name

        if self.config.storage_format.lower() == "csv":
            df.to_csv(file_path, index=True)
        else:
            df.to_parquet(file_path)

        logger.info("Stored intraday data", extra={"file": str(file_path), "rows": len(df)})
        self._log_manifest(symbol, start, end, file_path, len(df))
        return [file_path]

    def _fetch_dataframe(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        response = self.service.get_bars_dataframe(
            symbols=[symbol],
            start=start,
            end=end,
            timeframe=self.config.timeframe,
            feed=self.config.feed,
        )
        if response.empty:
            return pd.DataFrame()

        try:
            df = response.xs(symbol, level="symbol")
        except Exception:
            df = response.copy()

        df = df.sort_index()
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "trade_count": "trade_count",
                "vwap": "vwap",
            }
        )
        df["symbol"] = symbol.upper()
        return df

    @property
    def _extension(self) -> str:
        if self.config.storage_format.lower() == "csv":
            return "csv"
        return "parquet"

    def _log_manifest(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        file_path: Path,
        row_count: int,
    ) -> None:
        if not self.config.manifest_path:
            return
        manifest_path = self.config.manifest_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "symbol": symbol,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "file": str(file_path),
            "rows": row_count,
            "timeframe": self.config.timeframe.value,
            "collected_at": datetime.utcnow().isoformat(),
        }
        try:
            if manifest_path.exists():
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                data = []
        except json.JSONDecodeError:
            data = []
        data.append(record)
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

