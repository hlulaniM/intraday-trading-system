"""Candlestick pattern detection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_bullish_engulfing(df: pd.DataFrame, min_body_ratio: float = 1.5) -> pd.Series:
    """
    Detect bullish engulfing patterns.
    
    A bullish engulfing pattern occurs when:
    1. Previous candle is bearish (close < open)
    2. Current candle is bullish (close > open)
    3. Current candle's body completely engulfs previous candle's body
    4. Current body is at least min_body_ratio times larger than previous body
    
    Args:
        df: DataFrame with OHLC columns (open, high, low, close)
        min_body_ratio: Minimum ratio of current body to previous body (default 1.5)
    
    Returns:
        Boolean Series indicating bullish engulfing patterns
    """
    if not all(col in df.columns for col in ["open", "high", "low", "close"]):
        raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close' columns")
    
    prev_bearish = df["close"].shift(1) < df["open"].shift(1)
    curr_bullish = df["close"] > df["open"]
    
    prev_body = (df["close"].shift(1) - df["open"].shift(1)).abs()
    curr_body = (df["close"] - df["open"]).abs()
    
    engulfs = (df["open"] < df["close"].shift(1)) & (df["close"] > df["open"].shift(1))
    size_ok = curr_body >= prev_body * min_body_ratio
    
    return prev_bearish & curr_bullish & engulfs & size_ok


def detect_bearish_engulfing(df: pd.DataFrame, min_body_ratio: float = 1.5) -> pd.Series:
    """
    Detect bearish engulfing patterns.
    
    A bearish engulfing pattern occurs when:
    1. Previous candle is bullish (close > open)
    2. Current candle is bearish (close < open)
    3. Current candle's body completely engulfs previous candle's body
    4. Current body is at least min_body_ratio times larger than previous body
    
    Args:
        df: DataFrame with OHLC columns (open, high, low, close)
        min_body_ratio: Minimum ratio of current body to previous body (default 1.5)
    
    Returns:
        Boolean Series indicating bearish engulfing patterns
    """
    if not all(col in df.columns for col in ["open", "high", "low", "close"]):
        raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close' columns")
    
    prev_bullish = df["close"].shift(1) > df["open"].shift(1)
    curr_bearish = df["close"] < df["open"]
    
    prev_body = (df["close"].shift(1) - df["open"].shift(1)).abs()
    curr_body = (df["open"] - df["close"]).abs()
    
    engulfs = (df["open"] > df["close"].shift(1)) & (df["close"] < df["open"].shift(1))
    size_ok = curr_body >= prev_body * min_body_ratio
    
    return prev_bullish & curr_bearish & engulfs & size_ok


def get_latest_pattern(df: pd.DataFrame, min_body_ratio: float = 1.5) -> dict:
    """
    Get the latest engulfing pattern from a DataFrame.
    
    Args:
        df: DataFrame with OHLC columns
        min_body_ratio: Minimum body ratio for engulfing patterns
    
    Returns:
        Dictionary with 'type' ('bullish', 'bearish', or None) and 'timestamp'
    """
    bullish = detect_bullish_engulfing(df, min_body_ratio)
    bearish = detect_bearish_engulfing(df, min_body_ratio)
    
    if bullish.iloc[-1]:
        return {"type": "bullish", "timestamp": df.index[-1]}
    elif bearish.iloc[-1]:
        return {"type": "bearish", "timestamp": df.index[-1]}
    else:
        return {"type": None, "timestamp": df.index[-1]}

