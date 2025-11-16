"""Convenience wrapper for Alpaca trading and market data clients."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Sequence, Union

import pandas as pd

from alpaca.data.enums import CryptoFeed, DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
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
        self._stock_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        self._crypto_client = CryptoHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

    @staticmethod
    def _is_crypto(symbol: str) -> bool:
        return "/" in symbol or symbol.upper().startswith(("BTC", "ETH", "SOL", "ADA"))

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
        if self._is_crypto(symbol):
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                feed=CryptoFeed.US,
            )
            bars = self._crypto_client.get_crypto_bars(request)
        else:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=lookback_minutes,
                feed=DataFeed.IEX,
            )
            bars = self._stock_client.get_stock_bars(request)
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

        stock_symbols = [sym for sym in symbol_list if not self._is_crypto(sym)]
        crypto_symbols = [sym for sym in symbol_list if self._is_crypto(sym)]

        frames: List[pd.DataFrame] = []

        if stock_symbols:
            stock_request = StockBarsRequest(
                symbol_or_symbols=stock_symbols,
                timeframe=timeframe,
                start=start,
                end=end,
                feed=feed,
                limit=limit,
            )
            stock_response = self._stock_client.get_stock_bars(stock_request)
            if getattr(stock_response, "df", None) is not None:
                frames.append(stock_response.df.copy())

        if crypto_symbols:
            crypto_request = CryptoBarsRequest(
                symbol_or_symbols=crypto_symbols,
                timeframe=timeframe,
                start=start,
                end=end,
                feed=CryptoFeed.US,
            )
            crypto_response = self._crypto_client.get_crypto_bars(crypto_request)
            if getattr(crypto_response, "df", None) is not None:
                frames.append(crypto_response.df.copy())

        if not frames:
            logger.warning(
                "No bar data returned",
                extra={"symbols": symbol_list, "timeframe": timeframe.value},
            )
            return pd.DataFrame()

        df = pd.concat(frames)
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.set_levels(
                [df.index.levels[0], df.index.levels[1].tz_convert("UTC")],
                level=[0, 1],
            )
        else:
            df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
        return df

