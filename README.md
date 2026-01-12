# AI Trading Team

Multi-agent cryptocurrency trading bot using LangChain agents for decision making.

## Quick Start

```bash
# Install dependencies
uv sync

# Copy environment configuration
cp env.example .env
# Edit .env with your API keys
# Set TRADING_EXCHANGE=weex or binance for live execution

# Run the application
uv run python main.py
```

## Development

```bash
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
tests/                 # Test files
logs/                  # Runtime logs (gitignored)
docs/                  # Reference materials (do not import)
```

## Architecture

See [CONCEPTS.md](CONCEPTS.md) for detailed architecture design.
