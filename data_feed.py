"""
Data Feed Module
Fetches OHLCV data for futures instruments.

- Historical data via yfinance (for backtesting / strategy dev)
- Live data should come from NinjaTrader via file export or websocket feed
"""

import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


FUTURES_SYMBOLS = {
    'ES':  'ES=F',    # S&P 500 E-mini
    'NQ':  'NQ=F',    # Nasdaq E-mini
    'CL':  'CL=F',    # Crude Oil WTI
    'GC':  'GC=F',    # Gold
    'ZB':  'ZB=F',    # 30-Year T-Bond
    'RTY': 'RTY=F',   # Russell 2000 E-mini
    'YM':  'YM=F',    # Dow Jones E-mini
    'SI':  'SI=F',    # Silver
    'NG':  'NG=F',    # Natural Gas
    'ZC':  'ZC=F',    # Corn
    'ZS':  'ZS=F',    # Soybeans
    'ZW':  'ZW=F',    # Wheat
}


class DataFeed:
    """
    Fetches historical OHLCV data for futures symbols.

    For live trading, NinjaTrader should push bar data to the bot
    via file polling or a custom NT8 indicator/script that writes bars
    to a shared CSV/SQLite file that this class reads.
    """

    def get_historical(
        self,
        symbol: str,
        period: str = '2y',
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from Yahoo Finance.

        Parameters
        ----------
        symbol  : Futures symbol (e.g. 'ES', 'NQ', 'CL')
        period  : Lookback period ('1y', '2y', '5y', 'max')
        interval: Bar interval ('1d', '1h', '15m', '5m')

        Returns
        -------
        DataFrame with lowercase columns: open, high, low, close, volume
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        ticker = FUTURES_SYMBOLS.get(symbol.upper(), symbol)

        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)

            if df.empty:
                logger.warning(f"No data returned for {symbol} ({ticker})")
                return None

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            df.dropna(inplace=True)
            logger.info(f"Fetched {len(df)} bars for {symbol} ({interval}, {period})")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def get_intraday(
        self,
        symbol: str,
        interval: str = '15m',
        period: str = '60d'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch intraday bar data from Yahoo Finance.

        Parameters
        ----------
        symbol   : Futures symbol (e.g. 'ES', 'NQ', 'CL')
        interval : Bar interval ('1m', '5m', '15m', '60m')
        period   : Lookback period — yfinance limits 1m to 7 days max, use '5d'

        Returns
        -------
        DataFrame with lowercase columns: open, high, low, close, volume
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        ticker = FUTURES_SYMBOLS.get(symbol.upper(), symbol)

        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)

            if df.empty:
                logger.warning(f"No intraday data returned for {symbol} ({ticker}, {interval})")
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            df.dropna(inplace=True)
            logger.info(f"Fetched {len(df)} intraday bars for {symbol} ({interval}, {period})")
            return df

        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return None

    def get_live_bars_from_file(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Read live bar data written by a NinjaTrader indicator to a CSV file.

        Set up a NinjaScript indicator in NT8 to export bars to:
          C:\\NinjaTrader\\LiveBars\\<symbol>.csv

        Expected CSV columns: datetime,open,high,low,close,volume
        """
        try:
            df = pd.read_csv(
                filepath,
                parse_dates=['datetime'],
                index_col='datetime'
            )
            df.columns = [c.lower() for c in df.columns]
            return df
        except FileNotFoundError:
            logger.error(f"Live bar file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error reading live bar file: {e}")
            return None
