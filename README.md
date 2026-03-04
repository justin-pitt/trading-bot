# NinjaTrader Futures Trading Bot

A Python-based futures trading bot using NinjaTrader 8's AT Interface (TCP),
a proven trend-following strategy, and an optional LangChain AI layer.

---

## Project Structure

```
trading-bot/
├── main.py                # Bot orchestrator — run this
├── ninjatrader_bridge.py  # NT8 AT TCP bridge
├── strategy.py            # Dual EMA + ATR trend-following strategy
├── data_feed.py           # Historical & live data fetching
├── risk_manager.py        # Account-level risk rules
├── langchain_layer.py     # AI co-pilot (optional)
├── backtest.py            # Standalone backtester
├── requirements.txt
├── .env.example           # Copy to .env and fill in your values
└── .env                   # Your config (never commit this)
```

---

## Quick Start

**Prerequisites:** Python 3.10+

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure your environment
```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
NT_HOST=127.0.0.1              # NinjaTrader host (127.0.0.1 for local)
NT_PORT=36973                   # AT Interface port (default 36973)
NT_ACCOUNT=Sim101               # Your NT8 account name
ACCOUNT_VALUE=100000            # Must match your actual account balance — used for position sizing
SYMBOLS=ES,NQ,CL                # Comma-separated futures symbols to scan
SCAN_INTERVAL=30                # Seconds between strategy scans
USE_LANGCHAIN=false             # Set to true to enable AI sentiment filter
ANTHROPIC_API_KEY=              # Required only if USE_LANGCHAIN=true

# Live data from NinjaTrader (optional — leave blank to use Yahoo Finance)
LIVE_DATA_DIR=C:\NinjaTrader\LiveBars   # Must match Export Directory in LiveBarExporter
LIVE_INTERVAL=1D                         # Match the bar type on your NT8 chart (1D, 5M, 15M, 60M)
```

> **Important:** `ACCOUNT_VALUE` drives the risk-based position sizing formula. Set it to your actual sim or live account balance. If it's too small (e.g. $250), one stop-loss hit can exceed the account value and put the risk manager into a permanent halt.

### 3. Enable the AT Interface in NinjaTrader 8
```
Tools > Options > Automated Trading Interface
✓ Enable AT Interface
Port: 36973 (default)
```

### 4. Run the bot
```bash
python main.py
```

### 5. Run a backtest (no NinjaTrader required)
```bash
python backtest.py --symbol ES --period 5y
python backtest.py --symbol NQ --fast 15 --slow 50
```

---

## Data Source

The bot supports two data modes, selected via `.env`:

### Mode 1: Yahoo Finance (default)
Historical daily bars fetched from Yahoo Finance via `yfinance`. No extra setup needed.
Good for testing and daily-bar strategies. Not real-time.

### Mode 2: Live NT8 bars (set `LIVE_DATA_DIR`)
Reads CSV files written directly by the `LiveBarExporter` NinjaScript indicator running on your NT8 charts. Updated on every bar close. Works for any timeframe (daily, 5-min, 15-min, etc.).

**Setup:**
1. In NinjaTrader 8: `Tools > Import > NinjaScript Add-On`, select [LiveBarExporter.cs](LiveBarExporter.cs)
2. NT8 will compile the file automatically — check the Output window for errors
3. Add the indicator to each chart you want the bot to read (right-click chart → Indicators → LiveBarExporter)
4. Set **Export Directory** in the indicator properties (e.g. `C:\NinjaTrader\LiveBars`)
5. Set `LIVE_DATA_DIR` and `LIVE_INTERVAL` in your `.env` to match

**File naming convention:**

| Chart in NT8         | File written              | `.env` setting        |
|----------------------|---------------------------|-----------------------|
| ES daily             | `ES_1D.csv`               | `LIVE_INTERVAL=1D`    |
| NQ 5-minute          | `NQ_5M.csv`               | `LIVE_INTERVAL=5M`    |
| CL 15-minute         | `CL_15M.csv`              | `LIVE_INTERVAL=15M`   |
| ES 60-minute         | `ES_60M.csv`              | `LIVE_INTERVAL=60M`   |

The bot falls back to Yahoo Finance automatically if a live file is missing or has insufficient bars.

---

## Strategy: Riley Coleman Price Action

No indicators — pure price action on two timeframes:

| Timeframe | Purpose                                      |
|-----------|----------------------------------------------|
| 15-minute | Identify support & resistance zones          |
| 1-minute  | Confirm entry (rejection candles, breakouts) |

**Three setups:**

**1. Retest Reversal** — Price breaks a key S/R level, pulls back to retest it, then shows a rejection candle. Enter in the direction of the original break. Stop goes just beyond the S/R zone (1.5x ATR). Target is set at 2x risk (minimum R:R = 2.0).

**2. Failed Breakout** — Price briefly pierces through a level but closes back on the other side with a strong candle. Enter in the fade direction. Stop goes beyond the failed wick.

**3. 7 AM Reversal** — The same retest setup, but detected between 9:30–10:00 AM ET (the NY open window). Tagged separately in logs because this window has the highest probability of sharp reversals.

**S/R Level Detection:**
- Scans the last 50 bars of the 15m chart for swing highs and lows
- Clusters nearby pivots into zones
- Requires at least 2 touches to qualify a level
- Zone width scales with ATR (adapts to volatility)

**Entry Confirmation:**
- Rejection candle: lower wick ≥ 1.5x body for bullish, upper wick ≥ 1.5x body for bearish
- Wick must be ≥ 40% of total bar range
- Price must be within 1.5x ATR of the level

**Risk Management:**
- Minimum 2:1 R:R required — trades below this are skipped
- Position size = (Account × 1%) / (ATR × 1.5 × Tick Value)
- Profit target placed as a limit order automatically
- Stop loss placed as a stop order automatically

---

## Supported Futures Symbols

| Symbol | Instrument          | Tick Value | Notes                        |
|--------|---------------------|------------|------------------------------|
| ES     | S&P 500 E-mini      | $12.50     |                              |
| NQ     | Nasdaq E-mini       | $5.00      |                              |
| CL     | Crude Oil WTI       | $10.00     |                              |
| GC     | Gold                | $10.00     |                              |
| ZB     | 30-Year T-Bond      | $31.25     |                              |
| RTY    | Russell 2000 E-mini | $5.00      |                              |
| YM     | Dow Jones E-mini    | $5.00      |                              |
| SI     | Silver              | $10.00     | Default tick value           |
| NG     | Natural Gas         | $10.00     | Default tick value           |
| ZC     | Corn                | $10.00     | Default tick value           |
| ZS     | Soybeans            | $10.00     | Default tick value           |
| ZW     | Wheat               | $10.00     | Default tick value           |

---

## LangChain AI Layer (Optional)

Enable with `USE_LANGCHAIN=true` in `.env` and set `ANTHROPIC_API_KEY`.

Four integrations available in `langchain_layer.py`:

1. **TradeAnalyst** — Post-trade journal analysis, identifies behavioral patterns
2. **SentimentFilter** — Checks news sentiment before each trade signal (used in the main loop)
3. **RiskCopilot** — Natural language portfolio management chat interface
4. **StrategyOptimizer** — Suggests parameter adjustments based on backtest results

Powered by **Claude (`claude-sonnet-4-6`)** via `langchain-anthropic`.

Only `SentimentFilter` is wired into the main trading loop. The other three are standalone utilities you call manually.

---

## Risk Management

Hard limits enforced before every order:

| Rule                | Default | Behavior on breach         |
|---------------------|---------|----------------------------|
| Daily loss limit    | 3%      | Halt trading for the day   |
| Max drawdown        | 10%     | Halt trading permanently   |
| Max open positions  | 5       | Block new position opens   |
| Max position size   | 10 lots | Block oversized orders     |

Daily limits reset automatically at the start of each new trading day.
Emergency flatten (`FLATTENEVERYTHING`) is sent on Ctrl+C shutdown.

---

## NinjaTrader AT Interface Commands Used

| Command              | Description                        |
|----------------------|------------------------------------|
| `PLACE`              | Submit market/limit/stop orders    |
| `CANCEL`             | Cancel specific order              |
| `CANCELALLORDERS`    | Cancel all orders for account      |
| `CLOSEPOSITION`      | Close all contracts for a symbol   |
| `FLATTENEVERYTHING`  | Emergency: cancel all + close all  |
| `CHANGE`             | Modify existing order              |

Full AT Interface documentation: NinjaTrader 8 > Help > AT Interface

---

## Disclaimer

This software is for educational purposes only. Trading futures involves
substantial risk of loss and is not appropriate for all investors. Always
paper trade (use `Sim101` account) before going live. Never risk money
you cannot afford to lose.
