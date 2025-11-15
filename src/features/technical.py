"""Technical indicator engineering."""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from utils.logger import setup_logger

logger = setup_logger("technical_features")


def add_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Compute the technical indicators specified in the config dictionary.
    """
    if df.empty:
        return df

    result = df.copy()
    trend_cfg = config.get("features", {}).get("technical_indicators", config)  # allow direct dict

    if trend_cfg.get("macd", {}).get("enabled", True):
        macd_params = trend_cfg.get("macd", {})
        macd = MACD(
            close=result["close"],
            window_fast=macd_params.get("fast_period", 12),
            window_slow=macd_params.get("slow_period", 26),
            window_sign=macd_params.get("signal_period", 9),
        )
        result["macd"] = macd.macd()
        result["macd_signal"] = macd.macd_signal()
        result["macd_diff"] = macd.macd_diff()

    if trend_cfg.get("rsi", {}).get("enabled", True):
        period = trend_cfg.get("rsi", {}).get("period", 14)
        result["rsi"] = RSIIndicator(close=result["close"], window=period).rsi()

    if trend_cfg.get("bollinger_bands", {}).get("enabled", True):
        bb_cfg = trend_cfg.get("bollinger_bands", {})
        bands = BollingerBands(
            close=result["close"],
            window=bb_cfg.get("period", 20),
            window_dev=bb_cfg.get("std_dev", 2),
        )
        result["bb_high"] = bands.bollinger_hband()
        result["bb_low"] = bands.bollinger_lband()
        result["bb_mavg"] = bands.bollinger_mavg()

    if trend_cfg.get("atr", {}).get("enabled", True):
        atr_period = trend_cfg.get("atr", {}).get("period", 14)
        result["atr"] = AverageTrueRange(
            high=result["high"], low=result["low"], close=result["close"], window=atr_period
        ).average_true_range()

    if trend_cfg.get("ema", {}).get("enabled", True):
        for period in trend_cfg.get("ema", {}).get("periods", [9, 21, 50]):
            result[f"ema_{period}"] = EMAIndicator(close=result["close"], window=period).ema_indicator()

    if trend_cfg.get("sma", {}).get("enabled", True):
        for period in trend_cfg.get("sma", {}).get("periods", [20, 50, 200]):
            result[f"sma_{period}"] = SMAIndicator(close=result["close"], window=period).sma_indicator()

    if trend_cfg.get("stochastic", {}).get("enabled", True):
        stoch_cfg = trend_cfg.get("stochastic", {})
        stoch = StochasticOscillator(
            high=result["high"],
            low=result["low"],
            close=result["close"],
            window=stoch_cfg.get("k_period", 14),
            smooth_window=stoch_cfg.get("d_period", 3),
        )
        result["stoch_k"] = stoch.stoch()
        result["stoch_d"] = stoch.stoch_signal()

    return result

