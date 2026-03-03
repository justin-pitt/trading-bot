"""
Risk Manager
Enforces account-level risk rules before any order is placed.

Rules enforced:
- Max daily loss limit (default 3%)
- Max drawdown limit (default 10%)
- Max concurrent positions
- Cooldown period after hitting limits
"""

import logging
from datetime import date
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    account_value: float
    starting_account_value: float
    daily_pnl: float = 0.0
    open_positions: dict = field(default_factory=dict)
    trade_log: list = field(default_factory=list)
    last_reset: date = field(default_factory=date.today)
    trading_halted: bool = False


class RiskManager:
    """
    Stateful risk manager. Call check_order() before placing any order.
    Call update_pnl() after each fill or mark-to-market update.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,    # 3% daily loss limit
        max_drawdown_pct: float = 0.10,       # 10% max drawdown
        max_open_positions: int = 5,
        max_position_size: int = 10           # max contracts per symbol
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_open_positions = max_open_positions
        self.max_position_size = max_position_size
        self.state: RiskState = None

    def initialize(self, account_value: float):
        self.state = RiskState(
            account_value=account_value,
            starting_account_value=account_value
        )
        logger.info(f"RiskManager initialized. Account: ${account_value:,.2f}")

    def reset_daily(self):
        """Call at the start of each trading day."""
        if self.state:
            self.state.daily_pnl = 0.0
            self.state.trading_halted = False
            self.state.last_reset = date.today()
            logger.info("Daily risk limits reset.")

    def check_order(self, symbol: str, quantity: int, action: str) -> tuple[bool, str]:
        """
        Returns (approved: bool, reason: str)
        Call this before every order submission.
        """
        if self.state is None:
            return False, "RiskManager not initialized. Call initialize() first."

        # Auto-reset daily limits at start of new trading day
        if date.today() > self.state.last_reset:
            self.reset_daily()

        # Trading halt check
        if self.state.trading_halted:
            return False, "Trading halted: risk limit breached. Resume tomorrow or manually reset."

        # Daily loss limit
        daily_loss_pct = abs(min(self.state.daily_pnl, 0)) / self.state.starting_account_value
        if daily_loss_pct >= self.max_daily_loss_pct:
            self.state.trading_halted = True
            return False, (
                f"Daily loss limit reached: {daily_loss_pct:.1%} >= {self.max_daily_loss_pct:.1%}. "
                "Trading halted for today."
            )

        # Drawdown limit
        drawdown = (self.state.starting_account_value - self.state.account_value) / self.state.starting_account_value
        if drawdown >= self.max_drawdown_pct:
            self.state.trading_halted = True
            return False, (
                f"Max drawdown reached: {drawdown:.1%} >= {self.max_drawdown_pct:.1%}. "
                "Trading halted."
            )

        # Max open positions
        if len(self.state.open_positions) >= self.max_open_positions:
            if symbol not in self.state.open_positions:
                return False, (
                    f"Max open positions ({self.max_open_positions}) reached. "
                    "Close a position before opening new ones."
                )

        # Max position size
        if quantity > self.max_position_size:
            return False, (
                f"Position size {quantity} exceeds max allowed {self.max_position_size} contracts."
            )

        return True, "Approved"

    def update_pnl(self, pnl_change: float):
        """Update running P&L. Call after each fill or MTM update."""
        if self.state:
            self.state.daily_pnl += pnl_change
            self.state.account_value += pnl_change

    def record_open_position(self, symbol: str, action: str, quantity: int, entry_price: float):
        if self.state:
            self.state.open_positions[symbol] = {
                'action': action,
                'quantity': quantity,
                'entry_price': entry_price
            }

    def record_closed_position(self, symbol: str, exit_price: float):
        if self.state and symbol in self.state.open_positions:
            pos = self.state.open_positions.pop(symbol)
            pnl = (exit_price - pos['entry_price']) * pos['quantity']
            if pos['action'] == 'SELL':
                pnl = -pnl
            self.state.trade_log.append({
                'symbol': symbol,
                'action': pos['action'],
                'quantity': pos['quantity'],
                'entry': pos['entry_price'],
                'exit': exit_price,
                'pnl': round(pnl, 2)
            })
            self.update_pnl(pnl)
            return pnl
        return 0.0

    def get_status(self) -> dict:
        if not self.state:
            return {}
        drawdown = (self.state.starting_account_value - self.state.account_value) / self.state.starting_account_value
        daily_loss_pct = abs(min(self.state.daily_pnl, 0)) / self.state.starting_account_value
        return {
            'account_value': round(self.state.account_value, 2),
            'daily_pnl': round(self.state.daily_pnl, 2),
            'daily_loss_pct': round(daily_loss_pct, 4),
            'drawdown_pct': round(drawdown, 4),
            'open_positions': self.state.open_positions,
            'trading_halted': self.state.trading_halted,
            'total_trades': len(self.state.trade_log)
        }
