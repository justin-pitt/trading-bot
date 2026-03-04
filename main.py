"""
Main Trading Bot Orchestrator
Ties together NinjaTrader bridge, strategy, data feed, risk manager,
and optional LangChain AI layer.

Run:
    python main.py

Environment variables (set in .env):
    NT_HOST           NinjaTrader host (default 127.0.0.1)
    NT_PORT           NinjaTrader AT port (default 36973)
    NT_ACCOUNT        NinjaTrader account name (e.g. Sim101)
    ACCOUNT_VALUE     Starting account value in dollars
    ANTHROPIC_API_KEY Required only if USE_LANGCHAIN=true
    USE_LANGCHAIN     Enable LangChain AI layer (true/false, default false)
    SYMBOLS           Comma-separated futures symbols (default ES,NQ,CL)
    SCAN_INTERVAL     Seconds between strategy scans (default 60 for intraday)
    LIVE_DATA_DIR     Path to directory where LiveBarExporter writes CSV files.
                      If set, live NT8 bar data is used instead of Yahoo Finance.
                      Example: C:\\NinjaTrader\\LiveBars
    LIVE_INTERVAL     Bar interval suffix matching the NinjaScript file names
                      (default 1D for daily). Examples: 1D, 5M, 15M, 60M
"""

import os
import time
import logging
import signal
import sys
from dotenv import load_dotenv

from ninjatrader_bridge import NinjaTraderBridge
from strategy import PriceActionStrategy
from data_feed import DataFeed
from risk_manager import RiskManager

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger('TradingBot')


class TradingBot:
    def __init__(self):
        # Config from environment
        self.host          = os.getenv('NT_HOST', '127.0.0.1')
        self.port          = int(os.getenv('NT_PORT', 36973))
        self.account       = os.getenv('NT_ACCOUNT', 'Sim101')
        self.account_value = float(os.getenv('ACCOUNT_VALUE', 100_000))
        self.symbols       = os.getenv('SYMBOLS', 'ES,NQ,CL').split(',')
        self.scan_interval = int(os.getenv('SCAN_INTERVAL', 60))
        self.use_langchain = os.getenv('USE_LANGCHAIN', 'false').lower() == 'true'

        # Live data feed config (set LIVE_DATA_DIR to switch from Yahoo Finance to NT8 bars)
        self.live_data_dir      = os.getenv('LIVE_DATA_DIR', '').strip() or None
        self.live_interval      = os.getenv('LIVE_INTERVAL', '1D').strip()

        # Core components
        self.bridge   = NinjaTraderBridge(self.host, self.port)
        self.strategy = PriceActionStrategy()
        self.feed     = DataFeed()
        self.risk     = RiskManager()

        # Optional LangChain layer
        self.sentiment = None
        if self.use_langchain:
            try:
                from langchain_layer import SentimentFilter
                self.sentiment = SentimentFilter()
                logger.info("LangChain SentimentFilter enabled.")
            except ImportError:
                logger.warning("LangChain not installed. Disabling AI layer.")

        self._running = False

    # ── Startup / Shutdown ───────────────────────────────────────────────────

    def start(self):
        logger.info("=" * 60)
        logger.info("  NinjaTrader Futures Trading Bot  ")
        logger.info("=" * 60)
        logger.info(f"Account:   {self.account}")
        logger.info(f"Symbols:   {self.symbols}")
        logger.info(f"Interval:  {self.scan_interval}s")
        logger.info(f"Data:      {'NT8 live files (' + self.live_data_dir + ')' if self.live_data_dir else 'Yahoo Finance (daily bars)'}")
        logger.info(f"Strategy: Riley Coleman Price Action (Retest + Failed Breakout)")
        logger.info(f"Claude AI layer: {'enabled' if self.use_langchain else 'disabled'}")

        if not self.bridge.connect():
            logger.error("Cannot start: NinjaTrader connection failed.")
            logger.error("Make sure NT8 is running with AT Interface enabled.")
            sys.exit(1)

        self.risk.initialize(self.account_value)
        self._running = True

        # Register graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        logger.info("Bot started. Press Ctrl+C to stop.")
        self._run_loop()

    def _shutdown(self, *args):
        logger.info("Shutdown signal received. Closing...")
        self._running = False
        logger.warning(f"Flattening all positions on {self.account}...")
        try:
            self.bridge.flatten_everything(self.account)
        except Exception as e:
            logger.error(f"Error flattening on shutdown: {e}")
        self.bridge.disconnect()
        sys.exit(0)

    # ── Main Loop ────────────────────────────────────────────────────────────

    def _run_loop(self):
        while self._running:
            try:
                self._scan_all_symbols()
                self._log_status()
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)

            logger.info(f"Sleeping {self.scan_interval}s until next scan...")
            time.sleep(self.scan_interval)

    def _scan_all_symbols(self):
        logger.info(f"--- Scanning {len(self.symbols)} symbols ---")

        for symbol in self.symbols:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    def _get_bars(self, symbol: str):
        """
        Load bar data for a symbol.

        If LIVE_DATA_DIR is set, reads the CSV written by the LiveBarExporter
        NinjaScript indicator. Falls back to Yahoo Finance if the file is
        missing or doesn't have enough bars.
        """
        if self.live_data_dir:
            filepath = os.path.join(
                self.live_data_dir,
                f"{symbol}_{self.live_interval}.csv"
            )
            df = self.feed.get_live_bars_from_file(filepath)
            if df is not None and len(df) >= 60:
                logger.info(f"{symbol}: loaded {len(df)} bars from NT8 file ({filepath})")
                return df, 'live'
            logger.warning(
                f"{symbol}: NT8 file missing or insufficient data ({filepath}), "
                "falling back to Yahoo Finance."
            )

        df = self.feed.get_historical(symbol, period='1y', interval='1d')
        return df, 'yahoo'

    def _process_symbol(self, symbol: str):
        # 1. Fetch both timeframes
        df_15m = self.feed.get_intraday(symbol, interval='15m', period='60d')
        df_1m  = self.feed.get_intraday(symbol, interval='1m',  period='5d')

        if df_15m is None or len(df_15m) < 50:
            logger.warning(f"{symbol}: insufficient 15m data, skipping.")
            return
        if df_1m is None or len(df_1m) < 20:
            logger.warning(f"{symbol}: insufficient 1m data, skipping.")
            return

        # 2. Get strategy signal
        signal = self.strategy.get_signal(df_15m, df_1m, symbol, self.account_value)
        logger.info(
            f"{symbol}: action={signal.action} | setup={signal.setup_type} | "
            f"level={signal.level} | R:R={signal.risk_reward} | reason={signal.reason}"
        )

        if signal.action == 'HOLD':
            return

        # 3. Risk check
        approved, reason = self.risk.check_order(symbol, signal.suggested_quantity, signal.action)
        if not approved:
            logger.warning(f"{symbol}: Order blocked by RiskManager — {reason}")
            return

        # 4. Optional LangChain sentiment filter
        quantity = signal.suggested_quantity
        if self.sentiment:
            decision = self.sentiment.should_trade(symbol, signal.action)
            logger.info(f"{symbol}: Claude decision={decision.decision} | {decision.reason}")
            if decision.decision == 'SKIP':
                logger.info(f"{symbol}: Trade skipped by Claude.")
                return
            if decision.decision == 'REDUCE':
                quantity = max(1, int(quantity * decision.size_multiplier))
                logger.info(f"{symbol}: Position size reduced to {quantity} contracts.")

        # 5. Execute
        self._execute_signal(symbol, signal.action, quantity, signal.stop_price, signal.target_price)

    def _execute_signal(self, symbol: str, action: str, quantity: int, stop_price: float, target_price: float):
        logger.info(f"EXECUTING: {action} {quantity}x {symbol} | stop={stop_price} | target={target_price}")

        # Place main market order
        self.bridge.place_market_order(self.account, action, quantity, symbol)

        # Place protective stop loss
        stop_action = 'SELL' if action == 'BUY' else 'BUY'
        self.bridge.place_stop_order(
            account=self.account,
            action=stop_action,
            quantity=quantity,
            symbol=symbol,
            stop_price=stop_price
        )

        # Place profit target as a limit order
        target_action = 'SELL' if action == 'BUY' else 'BUY'
        self.bridge.place_limit_order(
            account=self.account,
            action=target_action,
            quantity=quantity,
            symbol=symbol,
            limit_price=target_price
        )

        # Update risk state
        self.risk.record_open_position(symbol, action, quantity, 0)

    def _log_status(self):
        status = self.risk.get_status()
        logger.info(
            f"STATUS | Account: ${status.get('account_value', 0):,.2f} | "
            f"Daily P&L: ${status.get('daily_pnl', 0):,.2f} | "
            f"Drawdown: {status.get('drawdown_pct', 0):.1%} | "
            f"Open positions: {len(status.get('open_positions', {}))} | "
            f"Total trades: {status.get('total_trades', 0)}"
        )


if __name__ == '__main__':
    bot = TradingBot()
    bot.start()
