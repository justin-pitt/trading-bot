"""
Riley Coleman Price Action Strategy
Multi-timeframe S/R zones, retest reversals, and failed breakout reversals.
No indicators — pure price action on 15m (structure) and 1m (entry).
"""

import datetime
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from datetime import time

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    action: str           # 'BUY', 'SELL', 'HOLD'
    symbol: str
    setup_type: str       # 'RETEST', 'FAILED_BREAKOUT', '7AM_REVERSAL', 'NONE'
    entry_price: float
    stop_price: float
    target_price: float
    risk_reward: float
    level: float
    atr: float
    suggested_quantity: int = 1
    reason: str = ''


class PriceActionStrategy:
    """
    Riley Coleman-style price action strategy.
    Detects S/R zones on 15m chart, confirms entries on 1m chart.
    Three setups: Retest Reversal, Failed Breakout, 7 AM Reversal.
    """

    TICK_VALUES = {
        'ES':  12.50,
        'NQ':   5.00,
        'CL':  10.00,
        'GC':  10.00,
        'ZB':  31.25,
        'RTY':  5.00,
        'YM':   5.00,
    }

    def __init__(
        self,
        sr_lookback: int = 50,
        sr_touch_min: int = 2,
        sr_zone_buffer: float = 0.3,
        min_risk_reward: float = 2.0,
        retest_tolerance: float = 0.5,
        rejection_wick_min: float = 1.5,
        open_window_start: datetime.time = time(9, 30),
        open_window_end: datetime.time = time(10, 0),
        risk_per_trade: float = 0.01,
        atr_period: int = 14,
    ):
        self.sr_lookback = sr_lookback
        self.sr_touch_min = sr_touch_min
        self.sr_zone_buffer = sr_zone_buffer
        self.min_risk_reward = min_risk_reward
        self.retest_tolerance = retest_tolerance
        self.rejection_wick_min = rejection_wick_min
        self.open_window_start = open_window_start
        self.open_window_end = open_window_end
        self.risk_per_trade = risk_per_trade
        self.atr_period = atr_period

    def _calc_atr(self, df: pd.DataFrame) -> pd.Series:
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def find_sr_levels(self, df_15m: pd.DataFrame) -> list:
        df = df_15m.iloc[-self.sr_lookback:].copy()
        atr_val = self._calc_atr(df).iloc[-1]
        zone = atr_val * self.sr_zone_buffer

        candidates = []
        for i in range(2, len(df) - 2):
            row = df.iloc[i]
            # Swing high: greater than 2 bars on each side
            if (row['high'] > df.iloc[i - 1]['high'] and
                    row['high'] > df.iloc[i - 2]['high'] and
                    row['high'] > df.iloc[i + 1]['high'] and
                    row['high'] > df.iloc[i + 2]['high']):
                candidates.append(row['high'])
            # Swing low: less than 2 bars on each side
            if (row['low'] < df.iloc[i - 1]['low'] and
                    row['low'] < df.iloc[i - 2]['low'] and
                    row['low'] < df.iloc[i + 1]['low'] and
                    row['low'] < df.iloc[i + 2]['low']):
                candidates.append(row['low'])

        if not candidates:
            return []

        candidates.sort()
        clusters = []
        current = [candidates[0]]
        for price in candidates[1:]:
            if price - current[0] <= zone * 2:
                current.append(price)
            else:
                clusters.append(current)
                current = [price]
        clusters.append(current)

        return [float(np.mean(c)) for c in clusters if len(c) >= self.sr_touch_min]

    def _is_rejection_candle(self, row: pd.Series, direction: str) -> bool:
        body = abs(row['close'] - row['open'])
        total_range = row['high'] - row['low']
        if total_range == 0:
            return False
        lower_wick = min(row['open'], row['close']) - row['low']
        upper_wick = row['high'] - max(row['open'], row['close'])
        if direction == 'bullish':
            return (lower_wick >= body * self.rejection_wick_min and
                    lower_wick >= total_range * 0.4)
        if direction == 'bearish':
            return (upper_wick >= body * self.rejection_wick_min and
                    upper_wick >= total_range * 0.4)
        return False

    def _in_morning_window(self, df: pd.DataFrame) -> bool:
        last_idx = df.index[-1]
        if not hasattr(last_idx, 'time'):
            return False
        t = last_idx.time()
        return self.open_window_start <= t <= self.open_window_end

    def _position_size(self, account_value: float, atr: float, symbol: str) -> int:
        dollar_risk = account_value * self.risk_per_trade
        tick_value = self.TICK_VALUES.get(symbol.upper(), 10.0)
        if atr <= 0:
            return 1
        contracts = dollar_risk / (atr * 1.5 * tick_value)
        return max(1, int(contracts))

    def get_signal(
        self,
        df_15m: pd.DataFrame,
        df_1m: pd.DataFrame,
        symbol: str,
        account_value: float = 100_000,
    ) -> TradeSignal:

        def hold(reason=''):
            return TradeSignal('HOLD', symbol, 'NONE', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, reason=reason)

        if len(df_15m) < self.sr_lookback or len(df_1m) < 20:
            return hold('Insufficient data')

        df_15m = df_15m.copy()
        df_1m = df_1m.copy()
        df_15m.columns = [c.lower() for c in df_15m.columns]
        df_1m.columns = [c.lower() for c in df_1m.columns]

        atr = float(self._calc_atr(df_1m).iloc[-1])
        levels = self.find_sr_levels(df_15m)
        if not levels:
            return hold('No S/R levels found')

        latest = df_1m.iloc[-1]
        prev = df_1m.iloc[-2]
        close = float(latest['close'])
        zone = atr * self.sr_zone_buffer
        in_window = self._in_morning_window(df_1m)
        qty = self._position_size(account_value, atr, symbol)
        recent_15m = df_15m.iloc[-10:]

        for level in levels:
            price_at_level = abs(close - level) <= zone * (1 + self.retest_tolerance)

            # Setup A — Retest Reversal BUY
            recently_broke_above = any(
                recent_15m.iloc[j]['close'] > level and recent_15m.iloc[j - 1]['close'] < level
                for j in range(1, len(recent_15m))
            )
            if recently_broke_above and price_at_level and self._is_rejection_candle(latest, 'bullish'):
                stop = level - atr * 1.5
                target = close + (close - stop) * self.min_risk_reward
                rr = (target - close) / (close - stop) if (close - stop) != 0 else 0
                if rr >= self.min_risk_reward:
                    setup = '7AM_REVERSAL' if in_window else 'RETEST'
                    return TradeSignal(
                        action='BUY', symbol=symbol, setup_type=setup,
                        entry_price=round(close, 2), stop_price=round(stop, 2),
                        target_price=round(target, 2), risk_reward=round(rr, 2),
                        level=round(level, 2), atr=round(atr, 4),
                        suggested_quantity=qty,
                        reason=f'{setup}: broke above {level:.2f}, retesting',
                    )

            # Setup B — Retest Reversal SELL
            recently_broke_below = any(
                recent_15m.iloc[j]['close'] < level and recent_15m.iloc[j - 1]['close'] > level
                for j in range(1, len(recent_15m))
            )
            if recently_broke_below and price_at_level and self._is_rejection_candle(latest, 'bearish'):
                stop = level + atr * 1.5
                target = close - (stop - close) * self.min_risk_reward
                rr = (close - target) / (stop - close) if (stop - close) != 0 else 0
                if rr >= self.min_risk_reward:
                    setup = '7AM_REVERSAL' if in_window else 'RETEST'
                    return TradeSignal(
                        action='SELL', symbol=symbol, setup_type=setup,
                        entry_price=round(close, 2), stop_price=round(stop, 2),
                        target_price=round(target, 2), risk_reward=round(rr, 2),
                        level=round(level, 2), atr=round(atr, 4),
                        suggested_quantity=qty,
                        reason=f'{setup}: broke below {level:.2f}, retesting',
                    )

            # Setup C — Failed Breakout BUY
            fakeout_bull = (
                float(prev['low']) < level - zone and
                close > level and
                float(latest['close']) > float(latest['open'])
            )
            if fakeout_bull:
                stop = float(prev['low']) - atr * 0.5
                target = close + (close - stop) * self.min_risk_reward
                rr = (target - close) / (close - stop) if (close - stop) != 0 else 0
                if rr >= self.min_risk_reward:
                    return TradeSignal(
                        action='BUY', symbol=symbol, setup_type='FAILED_BREAKOUT',
                        entry_price=round(close, 2), stop_price=round(stop, 2),
                        target_price=round(target, 2), risk_reward=round(rr, 2),
                        level=round(level, 2), atr=round(atr, 4),
                        suggested_quantity=qty,
                        reason=f'FAILED_BREAKOUT: fakeout below {level:.2f}, closed back above',
                    )

            # Setup D — Failed Breakout SELL
            fakeout_bear = (
                float(prev['high']) > level + zone and
                close < level and
                float(latest['close']) < float(latest['open'])
            )
            if fakeout_bear:
                stop = float(prev['high']) + atr * 0.5
                target = close - (stop - close) * self.min_risk_reward
                rr = (close - target) / (stop - close) if (stop - close) != 0 else 0
                if rr >= self.min_risk_reward:
                    return TradeSignal(
                        action='SELL', symbol=symbol, setup_type='FAILED_BREAKOUT',
                        entry_price=round(close, 2), stop_price=round(stop, 2),
                        target_price=round(target, 2), risk_reward=round(rr, 2),
                        level=round(level, 2), atr=round(atr, 4),
                        suggested_quantity=qty,
                        reason=f'FAILED_BREAKOUT: fakeout above {level:.2f}, closed back below',
                    )

        return hold('No setup')
