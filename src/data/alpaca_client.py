"""Convenience wrapper for Alpaca trading and market data clients."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Sequence, Union

import pandas as pd

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

from config import get_settings
from utils.logger import setup_logger

settings = get_settings()
logger = setup_logger("alpaca_client")


class AlpacaService:
    """High-level helper around Alpaca's trading and data APIs."""

    def __init__(self) -> None:
        self._trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper="paper" in settings.alpaca_base_url,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

    def get_account_status(self) -> str:
        """Return current account status string."""
        account = self._trading_client.get_account()
        logger.info("Fetched Alpaca account info", extra={"account_status": account.status})
        return account.status

    def list_active_assets(self, limit: int = 5) -> List[str]:
        """Fetch a subset of active tradable US equity symbols."""
        request = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status="active")
        assets = self._trading_client.get_all_assets(request)
        symbols = [asset.symbol for asset in assets[:limit]]
        logger.info("Retrieved Alpaca assets", extra={"count": len(symbols)})
        return symbols

    def get_recent_bars(
        self,
        symbol: str = "AAPL",
        lookback_minutes: int = 390,
        timeframe: TimeFrame = TimeFrame.Minute,
    ):
        """Fetch intraday bars for a symbol."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=lookback_minutes)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=lookback_minutes,
            feed=DataFeed.IEX,
        )
        bars = self._data_client.get_stock_bars(request)
        logger.info(
            "Fetched Alpaca bars",
            extra={
                "symbol": symbol,
                "bar_count": len(bars.data.get(symbol, [])),
                "timeframe": timeframe.value,
            },
        )
        return bars

    def get_bars_dataframe(
        self,
        symbols: Union[str, Sequence[str]],
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Minute,
        feed: DataFeed = DataFeed.IEX,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch bars for one or multiple symbols and return a tidy DataFrame.
        """
        if isinstance(symbols, str):
            symbol_list = [symbols]
        else:
            symbol_list = list(symbols)

        request = StockBarsRequest(
            symbol_or_symbols=symbol_list,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=feed,
            limit=limit,
        )
        response = self._data_client.get_stock_bars(request)

        if hasattr(response, "df") and response.df is not None:
            df = response.df.copy()
            if not df.empty:
                # Ensure timezone aware index for downstream alignment
                df.index = df.index.set_levels(
                    [df.index.levels[0], df.index.levels[1].tz_convert("UTC")],
                    level=[0, 1],
                )
            return df

        logger.warning(
            "No bar data returned",
            extra={"symbols": symbol_list, "timeframe": timeframe.value},
        )
        return pd.DataFrame()

