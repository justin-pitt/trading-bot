"""
NinjaTrader 8 AT (Automated Trading) Interface Bridge
Uses Option A: TCP socket connection to NT8's built-in AT server.

To enable in NinjaTrader 8:
  Tools > Options > Automated Trading Interface > Enable AT Interface
  Default port: 36973
"""

import socket
import threading
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class NinjaTraderBridge:
    def __init__(self, host: str = '127.0.0.1', port: int = 36973, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._connected = False

    # ── Connection Management ────────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host, self.port))
            self._connected = True
            logger.info(f"Connected to NinjaTrader AT Interface at {self.host}:{self.port}")
            return True
        except (ConnectionRefusedError, OSError) as e:
            logger.error(f"Failed to connect to NinjaTrader: {e}")
            logger.error("Ensure NT8 is running with AT Interface enabled (Tools > Options > AT Interface)")
            self._connected = False
            return False

    def disconnect(self):
        if self.sock:
            self.sock.close()
            self.sock = None
        self._connected = False
        logger.info("Disconnected from NinjaTrader")

    def is_connected(self) -> bool:
        return self._connected

    def reconnect(self, retries: int = 5, delay: float = 5.0) -> bool:
        for attempt in range(1, retries + 1):
            logger.info(f"Reconnection attempt {attempt}/{retries}...")
            if self.connect():
                return True
            time.sleep(delay)
        logger.error("All reconnection attempts failed.")
        return False

    # ── Core Send ────────────────────────────────────────────────────────────

    def _send(self, command: str):
        """Thread-safe command sender."""
        with self._lock:
            if not self._connected or self.sock is None:
                logger.warning("Not connected. Attempting reconnect...")
                if not self.reconnect():
                    raise ConnectionError("Unable to send command: not connected to NinjaTrader.")
            try:
                full_cmd = command.strip() + '\n'
                self.sock.sendall(full_cmd.encode('utf-8'))
                logger.debug(f"Sent: {full_cmd.strip()}")
            except (BrokenPipeError, OSError) as e:
                self._connected = False
                logger.error(f"Send error: {e}. Connection lost.")
                raise

    # ── Order Management ─────────────────────────────────────────────────────

    def place_market_order(
        self,
        account: str,
        action: str,       # 'BUY' or 'SELL'
        quantity: int,
        symbol: str,
        tif: str = 'GTC'   # Time in force: GTC, DAY
    ):
        """Place a market order."""
        cmd = f"PLACE;{account};{action};{quantity};{symbol};MARKET;0;0;{tif};;"
        self._send(cmd)
        logger.info(f"Market order sent: {action} {quantity}x {symbol} [{account}]")

    def place_limit_order(
        self,
        account: str,
        action: str,
        quantity: int,
        symbol: str,
        limit_price: float,
        tif: str = 'GTC'
    ):
        """Place a limit order."""
        cmd = f"PLACE;{account};{action};{quantity};{symbol};LIMIT;{limit_price};0;{tif};;"
        self._send(cmd)
        logger.info(f"Limit order sent: {action} {quantity}x {symbol} @ {limit_price} [{account}]")

    def place_stop_order(
        self,
        account: str,
        action: str,
        quantity: int,
        symbol: str,
        stop_price: float,
        tif: str = 'GTC'
    ):
        """Place a stop order."""
        cmd = f"PLACE;{account};{action};{quantity};{symbol};STOP;0;{stop_price};{tif};;"
        self._send(cmd)
        logger.info(f"Stop order sent: {action} {quantity}x {symbol} stop@{stop_price} [{account}]")

    def place_stop_limit_order(
        self,
        account: str,
        action: str,
        quantity: int,
        symbol: str,
        limit_price: float,
        stop_price: float,
        tif: str = 'GTC'
    ):
        """Place a stop-limit order."""
        cmd = f"PLACE;{account};{action};{quantity};{symbol};STOPLIMIT;{limit_price};{stop_price};{tif};;"
        self._send(cmd)
        logger.info(f"Stop-limit order sent: {action} {quantity}x {symbol} lim@{limit_price} stop@{stop_price}")

    # ── Position Management ──────────────────────────────────────────────────

    def close_position(self, account: str, symbol: str):
        """Close all open positions for a symbol."""
        cmd = f"CLOSEPOSITION;{account};{symbol};;"
        self._send(cmd)
        logger.info(f"Close position sent: {symbol} [{account}]")

    def cancel_order(self, order_id: str):
        """Cancel a specific order by ID."""
        cmd = f"CANCEL;{order_id};;"
        self._send(cmd)
        logger.info(f"Cancel order sent: {order_id}")

    def cancel_all_orders(self, account: str):
        """Cancel all open orders for an account."""
        cmd = f"CANCELALLORDERS;{account};;"
        self._send(cmd)
        logger.info(f"Cancel all orders sent [{account}]")

    def flatten_everything(self, account: str):
        """Emergency flatten: cancel all orders + close all positions."""
        cmd = f"FLATTENEVERYTHING;{account};;"
        self._send(cmd)
        logger.warning(f"FLATTEN EVERYTHING sent [{account}]")

    def change_order(self, order_id: str, quantity: int, limit_price: float, stop_price: float):
        """Modify an existing order."""
        cmd = f"CHANGE;{order_id};{quantity};{limit_price};{stop_price};;"
        self._send(cmd)
        logger.info(f"Change order sent: {order_id} qty={quantity}")

    # ── Context Manager ──────────────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
