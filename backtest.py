"""
Backtester
Run and evaluate the trend-following strategy on historical data.

Usage:
    python backtest.py
    python backtest.py --symbol NQ --fast 15 --slow 50
"""

import argparse
import pandas as pd
import numpy as np
import logging
from data_feed import DataFeed
from strategy import TrendFollowingStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('Backtest')


def run_backtest(
    symbol: str = 'ES',
    fast_period: int = 20,
    slow_period: int = 55,
    atr_period: int = 14,
    atr_stop_multiplier: float = 2.0,
    risk_per_trade: float = 0.01,
    starting_capital: float = 100_000,
    period: str = '5y'
) -> dict:
    feed = DataFeed()
    strategy = TrendFollowingStrategy(fast_period, slow_period, atr_period, atr_stop_multiplier, risk_per_trade)

    logger.info(f"Fetching {period} of data for {symbol}...")
    df = feed.get_historical(symbol, period=period, interval='1d')

    if df is None or df.empty:
        logger.error(f"No data for {symbol}")
        return {}

    df = strategy.calculate_indicators(df)
    df = df.dropna()

    # ── Simulate trades ───────────────────────────────────────────────────────
    capital = starting_capital
    position = 0       # 1 = long, -1 = short, 0 = flat
    entry_price = 0.0
    stop_price = 0.0
    quantity = 1

    tick_value = strategy.TICK_VALUES.get(symbol, 10.0)
    trades = []
    equity_curve = [capital]

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Check stop loss
        if position == 1 and row['low'] <= stop_price:
            pnl = (stop_price - entry_price) * quantity * tick_value
            capital += pnl
            trades.append({'type': 'LONG', 'entry': entry_price, 'exit': stop_price, 'pnl': pnl, 'exit_reason': 'STOP'})
            position = 0

        elif position == -1 and row['high'] >= stop_price:
            pnl = (entry_price - stop_price) * quantity * tick_value
            capital += pnl
            trades.append({'type': 'SHORT', 'entry': entry_price, 'exit': stop_price, 'pnl': pnl, 'exit_reason': 'STOP'})
            position = 0

        # Check crossover signals
        bullish_cross = row['ema_fast'] > row['ema_slow'] and prev['ema_fast'] <= prev['ema_slow']
        bearish_cross = row['ema_fast'] < row['ema_slow'] and prev['ema_fast'] >= prev['ema_slow']

        if bullish_cross:
            if position == -1:  # Close short
                pnl = (entry_price - row['close']) * quantity * tick_value
                capital += pnl
                trades.append({'type': 'SHORT', 'entry': entry_price, 'exit': row['close'], 'pnl': pnl, 'exit_reason': 'SIGNAL'})

            quantity = strategy.calculate_position_size(capital, row['atr'], symbol)
            entry_price = row['close']
            stop_price = row['close'] - (row['atr'] * atr_stop_multiplier)
            position = 1

        elif bearish_cross:
            if position == 1:  # Close long
                pnl = (row['close'] - entry_price) * quantity * tick_value
                capital += pnl
                trades.append({'type': 'LONG', 'entry': entry_price, 'exit': row['close'], 'pnl': pnl, 'exit_reason': 'SIGNAL'})

            quantity = strategy.calculate_position_size(capital, row['atr'], symbol)
            entry_price = row['close']
            stop_price = row['close'] + (row['atr'] * atr_stop_multiplier)
            position = -1

        equity_curve.append(capital)

    # ── Metrics ───────────────────────────────────────────────────────────────
    if not trades:
        logger.warning("No trades generated.")
        return {}

    trade_df = pd.DataFrame(trades)
    wins = trade_df[trade_df['pnl'] > 0]
    losses = trade_df[trade_df['pnl'] <= 0]

    total_return = (capital - starting_capital) / starting_capital
    win_rate = len(wins) / len(trade_df) if len(trade_df) > 0 else 0
    profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf')

    equity = pd.Series(equity_curve)
    returns = equity.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    results = {
        'symbol': symbol,
        'params': {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'atr_period': atr_period,
            'atr_stop_multiplier': atr_stop_multiplier,
            'risk_per_trade': risk_per_trade
        },
        'total_trades': len(trade_df),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 2),
        'total_return': round(total_return, 4),
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(max_drawdown, 4),
        'final_capital': round(capital, 2)
    }

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info(f"  BACKTEST RESULTS — {symbol}")
    logger.info("=" * 50)
    logger.info(f"  Period:         {period}")
    logger.info(f"  Total Trades:   {results['total_trades']}")
    logger.info(f"  Win Rate:       {results['win_rate']:.1%}")
    logger.info(f"  Profit Factor:  {results['profit_factor']:.2f}")
    logger.info(f"  Total Return:   {results['total_return']:.1%}")
    logger.info(f"  Sharpe Ratio:   {results['sharpe']:.2f}")
    logger.info(f"  Max Drawdown:   {results['max_drawdown']:.1%}")
    logger.info(f"  Final Capital:  ${results['final_capital']:,.2f}")
    logger.info("=" * 50)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest the trend-following strategy')
    parser.add_argument('--symbol', default='ES', help='Futures symbol (ES, NQ, CL, etc.)')
    parser.add_argument('--fast', type=int, default=20, help='Fast EMA period')
    parser.add_argument('--slow', type=int, default=55, help='Slow EMA period')
    parser.add_argument('--atr', type=int, default=14, help='ATR period')
    parser.add_argument('--stop-mult', type=float, default=2.0, help='ATR stop multiplier')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade (0.01 = 1%%)')
    parser.add_argument('--capital', type=float, default=100_000, help='Starting capital')
    parser.add_argument('--period', default='5y', help='Data period (1y, 2y, 5y)')
    args = parser.parse_args()

    run_backtest(
        symbol=args.symbol,
        fast_period=args.fast,
        slow_period=args.slow,
        atr_period=args.atr,
        atr_stop_multiplier=args.stop_mult,
        risk_per_trade=args.risk,
        starting_capital=args.capital,
        period=args.period
    )
