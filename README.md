# AI Trading Team

Multi-agent cryptocurrency trading bot that combines event-driven signals, an AI
decision layer (LangChain), and exchange execution with independent risk controls.

## Highlights

- Event-driven signal system (MA crossover, RSI extremes, funding rate, long/short ratio)
- Strategy state machine + queueing to prevent overlapping decisions
- LangChain agent produces structured JSON commands for execution
- Binance for market data; WEEX or Binance for execution; MockExecutor for DRY_RUN
- Textual TUI for real-time monitoring (optional)

## Requirements

- Python 3.12+
- uv (recommended)

## Quick Start

```bash
# Install dependencies
uv sync

# Copy environment configuration
cp env.example .env
# Edit .env with your API keys

# Run the application
uv run python main.py
```

## Running Modes

```bash
# Dry run (simulated execution)
DRY_RUN=true uv run python main.py

# TUI mode
uv run python main.py --tui
```

## Configuration

See `env.example` for all options. Common variables:

| Variable | Purpose | Notes |
| --- | --- | --- |
| ANTHROPIC_BASE_URL | LLM API base URL | Anthropic-compatible provider |
| ANTHROPIC_API_KEY | LLM API key | Required |
| BINANCE_API_KEY | Binance API key | Required for private endpoints or Binance execution |
| BINANCE_API_SECRET | Binance API secret | Required for private endpoints or Binance execution |
| TRADING_SYMBOL | Trading symbol | Example: DOGEUSDT |
| TRADING_EXCHANGE | Execution exchange | weex or binance |
| TRADING_LEVERAGE | Leverage | Clamped to 1-20 |
| TRADING_MAX_POSITION_PERCENT | Position sizing limit | Percent of available balance |
| TELEGRAM_BOT_TOKEN | Telegram bot token | Optional |
| TELEGRAM_CHAT_ID | Telegram chat ID | Optional |
| TELEGRAM_ACCOUNT_LABEL | Notification label | Optional |
| DRY_RUN | Simulated execution | true or false |

Notes:
- Market data always comes from Binance, regardless of execution exchange.
- For live execution, make sure the selected exchange credentials are configured.

## Development

```bash
# Install dev extras
uv sync --all-extras

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy src/

# Textual dev console (in separate terminal)
uv run textual console
```

## Project Structure

```
src/ai_trading_team/   # Main package
  agent/               # LangChain trading agent
  audit/               # Decision logging
  core/                # Shared types, events, data pool
  data/                # Binance data ingestion
  execution/           # Exchange executors
  indicators/          # Technical indicators (talipp)
  risk/                # Independent risk controls
  strategy/            # Signals and state machine
  ui/                  # Textual TUI
tests/                 # Test files
logs/                  # Runtime logs (gitignored)
docs/                  # Reference materials (do not import)
```

## Docs

- `docs/CONCEPTS.md` for system concepts and design constraints
- `docs/STRATEGY.md` for trading strategy notes
- `docs/STRATEGY_LIFECYCLE.md` for lifecycle and signal details

## Safety

This project can place real trades. Use `DRY_RUN=true` while testing and review
your exchange settings before running live.
