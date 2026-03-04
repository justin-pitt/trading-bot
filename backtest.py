"""
Backtester — Riley Coleman Price Action Strategy
Bar-by-bar simulation on 1m data using 15m S/R zones.

Usage:
    python backtest.py
    python backtest.py --symbol NQ --capital 50000 --min-rr 2.5
"""

import argparse
import pandas as pd
import numpy as np
import logging
from data_feed import DataFeed
from strategy import PriceActionStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('Backtest')


def run_backtest(
    symbol: str = 'ES',
    period: str = '60d',
    starting_capital: float = 100_000,
    min_rr: float = 2.0,
) -> dict:
    feed = DataFeed()
    strategy = PriceActionStrategy(min_risk_reward=min_rr)
    tick_value = strategy.TICK_VALUES.get(symbol.upper(), 10.0)

    logger.info(f"Fetching 15m data ({period}) and 1m data (5d) for {symbol}...")
    df_15m = feed.get_intraday(symbol, interval='15m', period=period)
    df_1m  = feed.get_intraday(symbol, interval='1m',  period='5d')

    if df_15m is None or df_15m.empty:
        logger.error(f"No 15m data for {symbol}")
        return {}
    if df_1m is None or df_1m.empty:
        logger.error(f"No 1m data for {symbol}")
        return {}

    logger.info(f"Simulating {len(df_1m)} 1m bars...")

    capital = starting_capital
    position = None   # dict with: action, entry, stop, target, qty, setup_type
    trades = []
    equity_curve = [capital]

    for i in range(20, len(df_1m)):
        df_1m_so_far = df_1m.iloc[:i + 1]
        bar = df_1m.iloc[i]

        # Check open position exit first
        if position is not None:
            if position['action'] == 'BUY':
                stop_hit   = bar['low']  <= position['stop']
                target_hit = bar['high'] >= position['target']
            else:
                stop_hit   = bar['high'] >= position['stop']
                target_hit = bar['low']  <= position['target']

            if stop_hit or target_hit:
                # If both hit same bar, conservatively assume stop
                if stop_hit and target_hit:
                    exit_price  = position['stop']
                    exit_reason = 'STOP'
                elif target_hit:
                    exit_price  = position['target']
                    exit_reason = 'TARGET'
                else:
                    exit_price  = position['stop']
                    exit_reason = 'STOP'

                qty = position['qty']
                if position['action'] == 'BUY':
                    pnl = (exit_price - position['entry']) * qty * tick_value
                else:
                    pnl = (position['entry'] - exit_price) * qty * tick_value

                capital += pnl
                trades.append({
                    'setup_type': position['setup_type'],
                    'action':     position['action'],
                    'entry':      position['entry'],
                    'exit':       exit_price,
                    'stop':       position['stop'],
                    'target':     position['target'],
                    'pnl':        pnl,
                    'exit_reason': exit_reason,
                })
                position = None
                equity_curve.append(capital)
                continue

        # No open position — check for signal
        if position is None:
            sig = strategy.get_signal(df_15m, df_1m_so_far, symbol, capital)
            if sig.action in ('BUY', 'SELL'):
                position = {
                    'action':     sig.action,
                    'entry':      sig.entry_price,
                    'stop':       sig.stop_price,
                    'target':     sig.target_price,
                    'qty':        sig.suggested_quantity,
                    'setup_type': sig.setup_type,
                }

        equity_curve.append(capital)

    # ── Metrics ───────────────────────────────────────────────────────────────
    if not trades:
        logger.warning("No trades generated.")
        return {}

    trade_df = pd.DataFrame(trades)
    wins   = trade_df[trade_df['pnl'] > 0]
    losses = trade_df[trade_df['pnl'] <= 0]

    total_return  = (capital - starting_capital) / starting_capital
    win_rate      = len(wins) / len(trade_df)
    profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf')

    pnl_series = trade_df['pnl']
    sharpe = (pnl_series.mean() / pnl_series.std() * np.sqrt(252 * 390)) if pnl_series.std() > 0 else 0

    equity = pd.Series(equity_curve)
    rolling_max  = equity.cummax()
    drawdown     = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Breakdown by setup type
    setup_stats = {}
    for setup in ('RETEST', 'FAILED_BREAKOUT', '7AM_REVERSAL'):
        subset = trade_df[trade_df['setup_type'] == setup]
        if len(subset) > 0:
            setup_wins = subset[subset['pnl'] > 0]
            setup_stats[setup] = {
                'count':    len(subset),
                'win_rate': len(setup_wins) / len(subset),
            }

    results = {
        'symbol':        symbol,
        'total_trades':  len(trade_df),
        'win_rate':      round(win_rate, 4),
        'profit_factor': round(profit_factor, 2),
        'total_return':  round(total_return, 4),
        'sharpe':        round(sharpe, 2),
        'max_drawdown':  round(max_drawdown, 4),
        'final_capital': round(capital, 2),
        'setup_stats':   setup_stats,
    }

    # Print summary
    logger.info("\n" + "=" * 55)
    logger.info(f"  BACKTEST RESULTS — {symbol}  (Riley Coleman PA)")
    logger.info("=" * 55)
    logger.info(f"  Total Trades:   {results['total_trades']}")
    logger.info(f"  Win Rate:       {results['win_rate']:.1%}")
    logger.info(f"  Profit Factor:  {results['profit_factor']:.2f}")
    logger.info(f"  Total Return:   {results['total_return']:.1%}")
    logger.info(f"  Sharpe Ratio:   {results['sharpe']:.2f}")
    logger.info(f"  Max Drawdown:   {results['max_drawdown']:.1%}")
    logger.info(f"  Final Capital:  ${results['final_capital']:,.2f}")
    if setup_stats:
        logger.info("  --- By Setup ---")
        for setup, stats in setup_stats.items():
            logger.info(f"  {setup:<20} count={stats['count']}  win={stats['win_rate']:.1%}")
    logger.info("=" * 55)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest the Riley Coleman price action strategy')
    parser.add_argument('--symbol',  default='ES',      help='Futures symbol (ES, NQ, CL, etc.)')
    parser.add_argument('--period',  default='60d',     help='15m data lookback period (e.g. 60d)')
    parser.add_argument('--capital', type=float, default=100_000, help='Starting capital')
    parser.add_argument('--min-rr',  type=float, default=2.0,     help='Minimum risk:reward ratio')
    args = parser.parse_args()

    run_backtest(
        symbol=args.symbol,
        period=args.period,
        starting_capital=args.capital,
        min_rr=args.min_rr,
    )
