# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent cryptocurrency trading bot using LangChain agents to make trading decisions based on market data and technical indicators. Built for the WEEX AI Trading Hackathon. Currently targets WEEX exchange for order execution while using Binance as the primary data source (due to its market dominance and pricing authority).

**Key Features:**
- LLM-powered trading decisions (MiniMax-M2.1 via Anthropic-compatible API)
- Multi-timeframe analysis (15m, 1h, 4h)
- Multi-factor signal aggregation
- Auto breakeven stop loss protection
- Dry run mode for safe testing
- Telegram notifications for position updates
- Complete AI decision audit logging

## Commands

```bash
# Install dependencies (including dev tools)
uv sync --all-extras

# Run the application
uv run python main.py

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_file.py::test_function_name

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Textual dev console (run in separate terminal for debugging)
uv run textual console

# Run app with dev console connection
uv run textual run --dev main.py

# Serve app in browser
uv run textual serve "python main.py"
```

## Environment Setup

Copy `env.example` to `.env` and configure:

**Required:**
- `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` - MiniMax API (Anthropic-compatible)
- `WEEX_API_KEY` / `WEEX_API_SECRET` / `WEEX_PASSPHRASE` - WEEX exchange credentials

**Trading Parameters:**
- `TRADING_SYMBOL` - Symbol to trade (e.g., DOGEUSDT)
- `TRADING_EXCHANGE` - Target exchange (weex or binance)
- `TRADING_LEVERAGE` - Leverage multiplier (1-20, capped at 20x)
- `TRADING_MAX_POSITION_PERCENT` - Max position size as % of balance (default: 75%)
- `DRY_RUN` - Set to true to simulate trades without execution

**Optional:**
- `BINANCE_API_KEY` / `BINANCE_API_SECRET` - Required if trading on Binance
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` / `TELEGRAM_ACCOUNT_LABEL` - Notifications

## Project Structure

```
src/ai_trading_team/
├── config.py              # Configuration management (loads from .env)
├── logging.py             # Logging setup (outputs to logs/)
├── core/                  # Core infrastructure
│   ├── types.py           # Global enums (Side, OrderType, etc.)
│   ├── events.py          # Event system for inter-module communication
│   ├── data_pool.py       # Thread-safe real-time data storage
│   ├── signal_queue.py    # Timestamped signal queue
│   └── session.py         # Trading session management
├── data/                  # Market data (Binance source)
│   ├── models.py          # Data models (Kline, Ticker, OrderBook)
│   ├── binance/           # Binance REST + WebSocket
│   │   ├── rest.py        # REST API client
│   │   └── stream.py      # WebSocket streams
│   └── manager.py         # Data manager
├── indicators/            # Technical indicators (talipp wrapper)
│   ├── base.py            # Indicator base class
│   ├── registry.py        # Indicator registry
│   └── technical.py       # RSI, MACD, BB, ATR implementations
├── strategy/              # Trading strategies
│   ├── base.py            # Strategy base class
│   ├── conditions.py      # Condition definitions
│   ├── orchestrator.py    # Strategy orchestration
│   ├── state_machine.py   # Position state machine
│   ├── signals/           # Signal generators
│   │   ├── base.py        # Signal base class
│   │   ├── types.py       # Signal type definitions
│   │   ├── aggregator.py  # Multi-factor signal aggregation
│   │   ├── rsi_extreme.py # RSI overbought/oversold
│   │   ├── macd_crossover.py
│   │   ├── ma_crossover.py
│   │   ├── bollinger_breakout.py
│   │   ├── funding_rate.py
│   │   ├── ls_ratio.py    # Long/short ratio
│   │   ├── open_interest.py
│   │   ├── liquidation.py
│   │   ├── pnl_change.py  # Profit/loss monitoring
│   │   ├── risk_signal.py
│   │   └── order_status.py
│   └── factors/           # Single-factor strategies (legacy)
│       ├── rsi_oversold.py
│       ├── macd_cross.py
│       ├── ma_crossover.py
│       ├── ma_position.py
│       ├── price_level.py
│       ├── volatility.py
│       ├── funding_rate.py
│       └── long_short_ratio.py
├── agent/                 # LangChain trading agent
│   ├── llm.py             # LLM client configuration
│   ├── prompts.py         # Main prompt templates
│   ├── prompts_neutral.py # Neutral/balanced prompts (default)
│   ├── prompts_long.py    # Long-biased prompts
│   ├── schemas.py         # Agent decision schemas
│   ├── commands.py        # Command definitions (AgentAction, AgentCommand)
│   └── trader.py          # Trading agent logic (LangChainTradingAgent)
├── execution/             # Order execution
│   ├── models.py          # Position, Order, Account models
│   ├── base.py            # Exchange abstract interface
│   ├── manager.py         # Execution manager
│   ├── dry_run.py         # Dry run executor (simulation)
│   ├── mock_executor.py   # Mock executor for testing
│   ├── weex/              # WEEX implementation
│   │   ├── executor.py    # WEEX executor
│   │   ├── rest.py        # WEEX REST client
│   │   └── stream.py      # WEEX WebSocket
│   └── binance/           # Binance implementation
│       └── executor.py    # Binance executor
├── risk/                  # Independent risk control
│   ├── rules.py           # Risk rules (StopLoss, TakeProfit, MaxDrawdown)
│   ├── actions.py         # Risk actions
│   └── monitor.py         # Risk monitor
├── audit/                 # AI decision logging
│   ├── models.py          # Log models (AgentLog, OrderLog)
│   ├── writer.py          # Local file writer
│   ├── uploaders/         # Pluggable uploaders
│   │   ├── base.py        # Uploader interface
│   │   └── weex.py        # WEEX API uploader
│   └── manager.py         # Audit manager
├── notifications/         # Notification services
│   └── telegram.py        # Telegram bot notifications
└── ui/                    # Textual TUI
    ├── app.py             # Main application
    ├── screens/           # Dashboard, Logs screens
    │   ├── dashboard.py
    │   └── logs.py
    └── widgets/           # UI components
        ├── ticker.py
        ├── positions.py
        ├── orders.py
        ├── orderbook.py
        ├── indicators.py
        ├── price_chart.py
        ├── activity_log.py
        └── risk.py

tests/                     # Test files
logs/                      # Runtime logs (gitignored)
docs/                      # Reference materials (do not import)
```

## Architecture

The system is designed with the following modular layers:

### UI Layer
- **Textual TUI**: Terminal-based user interface built with Textual framework
  - Async-native, integrates with WebSocket data streams
  - Supports running in terminal or browser via `textual serve`
  - Debug with `textual console` in a separate terminal
  - Command palette available via `Ctrl+P`

### Data Layer
- **Data Module**: Fetches real-time market data from Binance (REST + WebSocket)
  - Klines (15m, 1h, 4h timeframes)
  - Ticker, orderbook, trades
  - Long/short ratio, funding rate, mark price
  - Open interest, liquidation events
- **Shared Data Pool**: Central real-time data repository that all modules read from

### Analysis Layer
- **Indicator Module**: Computes technical indicators using `talipp` library
  - RSI, MACD, Bollinger Bands, ATR (multi-timeframe)
- **Signal Module**: Triggers signals when predefined conditions are met
  - Multi-factor signal aggregation
  - Weighted scoring across timeframes

### Decision Layer
- **Agent Module**: LangChain-based agent receives signals with full market context
  - Uses MiniMax-M2.1 via Anthropic-compatible API
  - Returns structured JSON commands (OPEN, CLOSE, ADJUST_SL, OBSERVE)
  - Multiple prompt variants: neutral, long-biased
- **State Machine**: Manages position lifecycle and state transitions
- **Risk Control Module**: Independent mechanical risk management
  - Auto breakeven stop loss when profit exceeds threshold
  - Maximum drawdown protection

### Execution Layer
- **Execution Module**: Manages positions, orders, and executes trades
  - Supports WEEX, Binance, DryRun, and Mock executors
  - Real-time sync via REST + WebSocket
  - Automatic reconnection handling
  - Leverage capped at 20x (competition requirement)

### Logging Layer
- Complete AI decision logs (input, output, reasoning) for WEEX AI competition compliance
- Local file persistence for analysis and auditing
- Telegram notifications for position updates

## Key Design Principles

1. **Signal Queue**: Strategy signals are timestamped and queued to prevent signal overlap
2. **Data Snapshot**: When signals trigger, capture a point-in-time snapshot from the data pool
3. **Dual Data Sources**: Binance for market data, target exchange for account/order data
4. **Structured Agent Output**: Agent responses must be JSON-formatted commands
5. **Position Protection**: Auto breakeven stop loss moves SL to entry when profit threshold reached
6. **Conservative Risk**: Single position margin = min(available/20, 750 - used_margin)

## Agent Commands

The agent outputs one of these actions:

| Action | Description |
|--------|-------------|
| `open` | Open a new position with side, size, stop_loss_price, take_profit_price |
| `close` | Close current position |
| `adjust_sl` | Adjust stop loss price only |
| `observe` | Take no action, continue monitoring |

## Important Notes

- Requires Python 3.12+
- Files in `docs/` are reference materials only - do not import or use code from this directory
- Use `talipp` library for all technical indicator calculations
- The main entry point (`main.py`) contains the full TradingBot implementation
- Leverage is capped at 20x regardless of config value (competition limit)
- DRY_RUN mode simulates all trading operations without actual execution
