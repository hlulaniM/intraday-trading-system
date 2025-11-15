"""Sentiment feature utilities (placeholder for FinBERT integration)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd

from utils.logger import setup_logger

logger = setup_logger("sentiment_features")


@dataclass
class SentimentRecord:
    timestamp: datetime
    score: float
    source: str = "news"


def sentiment_frame(records: Iterable[SentimentRecord]) -> pd.DataFrame:
    """Convert sentiment records to a pandas DataFrame."""
    rows = [
        {"timestamp": record.timestamp, "sentiment_score": record.score, "source": record.source}
        for record in records
    ]
    if not rows:
        return pd.DataFrame(columns=["sentiment_score"])
    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def merge_sentiment(
    price_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame],
    smoothing_window: int = 5,
) -> pd.DataFrame:
    """Join sentiment scores with price data, applying optional smoothing."""
    if sentiment_df is None or sentiment_df.empty:
        price_df["sentiment_score"] = 0.0
        price_df["sentiment_anomaly"] = 0
        return price_df

    sentiment_df = sentiment_df.copy()
    sentiment_df["sentiment_score"] = sentiment_df["sentiment_score"].rolling(
        smoothing_window, min_periods=1
    ).mean()
    sentiment_df["sentiment_anomaly"] = (
        (sentiment_df["sentiment_score"] - sentiment_df["sentiment_score"].mean()).abs()
        > sentiment_df["sentiment_score"].std()
    ).astype(int)

    merged = price_df.join(sentiment_df[["sentiment_score", "sentiment_anomaly"]], how="left")
    merged[["sentiment_score", "sentiment_anomaly"]] = merged[
        ["sentiment_score", "sentiment_anomaly"]
    ].fillna({"sentiment_score": 0.0, "sentiment_anomaly": 0})
    return merged

