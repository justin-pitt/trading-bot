"""
Trend-Following Strategy with ATR-Based Position Sizing
Based on the Turtle Trading / CTA dual moving average crossover framework.

Most proven futures strategy:
- Dual EMA crossover (20/55 period — Turtle system defaults)
- ATR-based stop placement (2x ATR)
- Risk-per-trade position sizing (1% account risk)
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    action: str           # 'BUY', 'SELL', 'HOLD'
    symbol: str
    stop_price: float
    atr: float
    ema_fast: float
    ema_slow: float
    suggested_quantity: int = 0
    reason: str = ''


class TrendFollowingStrategy:
    """
    Dual EMA crossover with ATR stops and risk-based position sizing.

    Parameters
    ----------
    fast_period : int
        Fast EMA period (default 20)
    slow_period : int
        Slow EMA period (default 55 — Turtle system)
    atr_period : int
        ATR lookback period (default 14)
    atr_stop_multiplier : float
        Stop distance = ATR * multiplier (default 2.0)
    risk_per_trade : float
        Fraction of account to risk per trade (default 0.01 = 1%)
    """

    TICK_VALUES = {
        'ES': 12.50,    # S&P 500 E-mini  ($12.50/tick, 0.25 tick size)
        'NQ': 5.00,     # Nasdaq E-mini   ($5.00/tick,  0.25 tick size)
        'CL': 10.00,    # Crude Oil       ($10.00/tick, 0.01 tick size)
        'GC': 10.00,    # Gold            ($10.00/tick, 0.10 tick size)
        'ZB': 31.25,    # 30yr T-Bond     ($31.25/tick)
        'RTY': 5.00,    # Russell 2000    ($5.00/tick)
        'YM': 5.00,     # Dow E-mini      ($5.00/tick)
    }

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 55,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        risk_per_trade: float = 0.01
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.risk_per_trade = risk_per_trade

    # ── Indicator Calculation ────────────────────────────────────────────────

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all required indicators to the DataFrame.
        Input df must have lowercase columns: open, high, low, close, volume
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()

        # True Range & ATR
        prev_close = df['close'].shift(1)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                (df['high'] - prev_close).abs(),
                (df['low'] - prev_close).abs()
            )
        )
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()

        # Raw signal: 1 = bullish, -1 = bearish
        df['signal'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)

        # Crossover: change in signal direction
        df['crossover'] = df['signal'].diff()

        # Stop distance
        df['stop_distance'] = df['atr'] * self.atr_stop_multiplier

        return df

    # ── Signal Generation ────────────────────────────────────────────────────

    def get_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        account_value: float = 100_000
    ) -> TradeSignal:
        """
        Evaluate the latest bar and return a TradeSignal.
        Only fires on a fresh crossover, not on every bar.
        """
        df = self.calculate_indicators(df)

        if len(df) < self.slow_period + 5:
            return TradeSignal('HOLD', symbol, 0, 0, 0, 0, reason='Insufficient data')

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        ema_f = latest['ema_fast']
        ema_s = latest['ema_slow']
        atr = latest['atr']
        close = latest['close']

        # Bullish crossover
        if ema_f > ema_s and prev['ema_fast'] <= prev['ema_slow']:
            stop = close - (atr * self.atr_stop_multiplier)
            qty = self.calculate_position_size(account_value, atr, symbol)
            return TradeSignal(
                action='BUY',
                symbol=symbol,
                stop_price=round(stop, 2),
                atr=round(atr, 4),
                ema_fast=round(ema_f, 2),
                ema_slow=round(ema_s, 2),
                suggested_quantity=qty,
                reason=f'EMA{self.fast_period} crossed above EMA{self.slow_period}'
            )

        # Bearish crossover
        if ema_f < ema_s and prev['ema_fast'] >= prev['ema_slow']:
            stop = close + (atr * self.atr_stop_multiplier)
            qty = self.calculate_position_size(account_value, atr, symbol)
            return TradeSignal(
                action='SELL',
                symbol=symbol,
                stop_price=round(stop, 2),
                atr=round(atr, 4),
                ema_fast=round(ema_f, 2),
                ema_slow=round(ema_s, 2),
                suggested_quantity=qty,
                reason=f'EMA{self.fast_period} crossed below EMA{self.slow_period}'
            )

        return TradeSignal(
            action='HOLD',
            symbol=symbol,
            stop_price=0,
            atr=round(atr, 4),
            ema_fast=round(ema_f, 2),
            ema_slow=round(ema_s, 2),
            reason='No crossover'
        )

    # ── Position Sizing ──────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        account_value: float,
        atr: float,
        symbol: str
    ) -> int:
        """
        Risk-based N-unit sizing from the Turtle Trading system.
        Contracts = (Account * Risk%) / (ATR * TickValue)
        """
        tick_value = self.TICK_VALUES.get(symbol.upper(), 10.0)
        dollar_risk = account_value * self.risk_per_trade

        if atr <= 0:
            return 1

        contracts = dollar_risk / (atr * tick_value)
        return max(1, int(contracts))
