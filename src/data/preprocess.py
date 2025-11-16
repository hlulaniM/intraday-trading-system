"""Data cleaning and synchronization helpers for intraday bars."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from config import get_settings
from utils.logger import setup_logger
from utils.symbols import sanitize_symbol

logger = setup_logger("data_preprocess")


def clean_price_frame(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Apply standard cleaning steps: sort index, fill timestamps, drop obvious outliers.
    """
    if df.empty:
        return df

    df = df.sort_index()
    df = _ensure_datetime_index(df)
    df = df[~df.index.duplicated(keep="last")]

    df = _fill_missing_timestamps(df, freq=freq)
    df = _remove_outliers(df, columns=["open", "high", "low", "close"])
    df = df.ffill()
    df["volume"] = df["volume"].fillna(0)
    return df


def synchronize_frames(frames: Dict[str, pd.DataFrame], freq: str = "1min") -> Dict[str, pd.DataFrame]:
    """
    Align multiple symbol dataframes on a shared timestamp grid.
    """
    synced: Dict[str, pd.DataFrame] = {}
    all_indices = []
    for df in frames.values():
        if not df.empty:
            all_indices.append(_ensure_datetime_index(df).index)

    if not all_indices:
        return synced

    unified_index = all_indices[0]
    for idx in all_indices[1:]:
        unified_index = unified_index.union(idx)

    unified_index = unified_index.sort_values()

    for symbol, df in frames.items():
        if df.empty:
            continue
        df = _ensure_datetime_index(df).reindex(unified_index)
        df = df.interpolate(method="time").ffill().bfill()
        df["symbol"] = symbol
        synced[symbol] = df
    return synced


def save_processed_frame(df: pd.DataFrame, symbol: str, suffix: str = "cleaned") -> Path:
    """Persist cleaned data into the processed directory."""
    settings = get_settings()
    safe_symbol = sanitize_symbol(symbol)
    processed_dir = settings.data_processed_dir / safe_symbol
    processed_dir.mkdir(parents=True, exist_ok=True)
    timestamp_label = df.index.min().strftime("%Y%m%d") if not df.empty else "empty"
    path = processed_dir / f"{safe_symbol}_{suffix}_{timestamp_label}.parquet"
    df.to_parquet(path)
    logger.info("Saved processed dataframe", extra={"symbol": symbol, "path": str(path)})
    return path


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    return df


def _fill_missing_timestamps(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty:
        return df
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    return df.reindex(full_index)


def _remove_outliers(df: pd.DataFrame, columns: Iterable[str], z_threshold: float = 3.0) -> pd.DataFrame:
    cleaned = df.copy()
    for column in columns:
        if column not in cleaned.columns:
            continue
        series = cleaned[column]
        if series.std(ddof=0) == 0 or series.empty:
            continue
        z_scores = (series - series.mean()) / series.std(ddof=0)
        cleaned = cleaned[z_scores.abs() <= z_threshold]
    return cleaned

