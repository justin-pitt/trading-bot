"""
LangChain AI Layer
Four practical integrations for the trading bot:

1. TradeAnalyst    — Post-trade journal analysis
2. SentimentFilter — News sentiment check before entering trades
3. RiskCopilot     — Natural language portfolio management
4. StrategyOptimizer — Parameter suggestions from backtest results

NOTE: LangChain is used as a co-pilot/analysis layer only.
      Never put LangChain in the critical execution path (latency risk).

Requires: pip install langchain langchain-anthropic
Set ANTHROPIC_API_KEY in your .env file.
"""

import json
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SentimentDecision:
    decision: str           # 'TRADE', 'REDUCE', 'SKIP'
    reason: str
    size_multiplier: float  # 0.0 – 1.0


def _get_llm(temperature: float = 0):
    """Lazy-load the LLM to avoid import errors if langchain-anthropic isn't installed."""
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-6", temperature=temperature)
    except ImportError:
        raise ImportError(
            "LangChain Anthropic not installed. Run: pip install langchain langchain-anthropic"
        )


# ── 1. Trade Journal Analyst ─────────────────────────────────────────────────

class TradeAnalyst:
    """
    Analyzes completed trades and surfaces behavioral/strategy patterns.

    Usage:
        analyst = TradeAnalyst()
        insight = analyst.analyze_trade_log(trades)
    """

    def analyze_trade_log(self, trades: list[dict]) -> str:
        """
        trades: list of dicts with keys like:
          symbol, action, entry_price, exit_price, pnl, duration_bars, reason
        """
        llm = _get_llm()
        try:
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError:
            raise ImportError("Run: pip install langchain-core")

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert futures trading coach. Analyze trade logs and identify: "
             "1) behavioral patterns in losses, 2) strategy weaknesses, "
             "3) time-of-day or market-regime effects, 4) concrete improvement actions."),
            ("user", "Analyze these trades and give me actionable insights:\n\n{trades}")
        ])
        chain = prompt | llm
        response = chain.invoke({"trades": json.dumps(trades, indent=2)})
        return response.content


# ── 2. Sentiment Filter ───────────────────────────────────────────────────────

class SentimentFilter:
    """
    Before entering a trade, checks if news sentiment conflicts with the signal.

    Usage:
        sf = SentimentFilter()
        decision = sf.should_trade('ES', 'BUY')
        if decision.decision != 'SKIP':
            place_order(...)
    """

    def should_trade(
        self,
        symbol: str,
        signal: str,
        news_context: Optional[str] = None
    ) -> SentimentDecision:
        """
        symbol       : e.g. 'ES', 'NQ', 'CL'
        signal       : 'BUY' or 'SELL'
        news_context : Optional string of recent news headlines (from your news API)
        """
        llm = _get_llm()

        news_section = (
            f"\nRecent news context:\n{news_context}"
            if news_context
            else "\n(No live news context provided — use general market knowledge.)"
        )

        prompt = (
            f"Our trend-following system generated a {signal} signal for {symbol} futures.{news_section}\n\n"
            "Based on current market conditions, decide:\n"
            "- TRADE: proceed normally\n"
            "- REDUCE: proceed with smaller size (specify multiplier)\n"
            "- SKIP: do not trade this signal\n\n"
            "Respond ONLY with valid JSON, no extra text:\n"
            '{"decision": "TRADE|REDUCE|SKIP", "reason": "...", "size_multiplier": 0.5}'
        )

        try:
            response = llm.invoke(prompt)
            raw = response.content.strip()
            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            return SentimentDecision(
                decision=data.get('decision', 'TRADE'),
                reason=data.get('reason', ''),
                size_multiplier=float(data.get('size_multiplier', 1.0))
            )
        except Exception as e:
            logger.warning(f"SentimentFilter failed ({e}), defaulting to TRADE")
            return SentimentDecision('TRADE', 'LangChain unavailable, proceeding', 1.0)


# ── 3. Risk Copilot ───────────────────────────────────────────────────────────

class RiskCopilot:
    """
    Natural language interface to monitor the bot.

    Examples:
        copilot.chat("How are my open positions doing?")
        copilot.chat("Reduce all sizes by 50% today")
        copilot.chat("What's my max drawdown this week?")
    """

    def __init__(self):
        self._history = []

    def chat(self, message: str, portfolio_state: dict) -> str:
        """
        message        : Natural language query from the user
        portfolio_state: Dict with keys like open_positions, daily_pnl, account_value, etc.
        """
        llm = _get_llm()

        system_msg = (
            "You are a risk management assistant for a futures trading bot. "
            f"Current portfolio state:\n{json.dumps(portfolio_state, indent=2)}\n\n"
            "Be concise and practical. If the user asks to take an action (e.g. reduce size, "
            "flatten positions), confirm clearly and describe what should happen."
        )

        messages = [{"role": "system", "content": system_msg}]
        for turn in self._history[-10:]:  # last 5 exchanges
            messages.append(turn)
        messages.append({"role": "user", "content": message})

        response = llm.invoke(messages)
        reply = response.content

        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": reply})

        return reply

    def clear_history(self):
        self._history = []


# ── 4. Strategy Optimizer ─────────────────────────────────────────────────────

class StrategyOptimizer:
    """
    Reviews backtest results and suggests parameter adjustments.

    Usage:
        optimizer = StrategyOptimizer()
        suggestions = optimizer.suggest_parameters(backtest_results, 'trending')
    """

    def suggest_parameters(
        self,
        backtest_results: dict,
        market_regime: str = 'unknown'
    ) -> dict:
        """
        backtest_results: dict with keys:
          params (dict), sharpe, max_drawdown, win_rate, profit_factor, total_trades
        market_regime: 'trending', 'ranging', 'volatile', 'low_volatility'
        """
        llm = _get_llm()

        prompt = (
            f"Current strategy parameters:\n{json.dumps(backtest_results.get('params', {}), indent=2)}\n\n"
            f"Backtest performance:\n"
            f"  Sharpe Ratio:    {backtest_results.get('sharpe', 'N/A')}\n"
            f"  Max Drawdown:    {backtest_results.get('max_drawdown', 'N/A')}\n"
            f"  Win Rate:        {backtest_results.get('win_rate', 'N/A')}\n"
            f"  Profit Factor:   {backtest_results.get('profit_factor', 'N/A')}\n"
            f"  Total Trades:    {backtest_results.get('total_trades', 'N/A')}\n\n"
            f"Current market regime: {market_regime}\n\n"
            "Suggest optimized parameters and explain your reasoning. "
            "Respond ONLY with valid JSON:\n"
            '{"fast_period": 20, "slow_period": 55, "atr_period": 14, '
            '"atr_stop_multiplier": 2.0, "risk_per_trade": 0.01, "reasoning": "..."}'
        )

        try:
            response = llm.invoke(prompt)
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            logger.error(f"StrategyOptimizer error: {e}")
            return {}
