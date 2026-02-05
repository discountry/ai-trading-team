"""Grid Trading Strategy for Weex SDK

A dynamic single-direction grid trading strategy implementation for Weex exchange.

## Features

- **Dynamic Grid Range**: Automatically adjusts grid range based on current price (default ±3%)
- **Single Direction**: Supports both long and short directions
- **Equal Percentage Distribution**: Grid levels distributed by equal percentage intervals
- **Automatic Order Management**: Automatic opening/closing orders when grid levels are filled
- **Stop Loss**: Set absolute loss amount for automatic position closure
- **State Persistence**: Save and restore strategy state for restarts
- **WebSocket Integration**: Real-time ticker, order, and position updates
- **Grid Rebuilding**: Automatically rebuilds grid when price moves outside range

## Quick Start

### Long Direction Strategy

```python
import os
import time
from dotenv import load_dotenv
from weex_sdk import WeexClient, WeexWebSocket
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import strategies

load_dotenv()

# Initialize client and WebSocket
client = WeexClient(api_key=os.getenv("API_KEY"), secret_key=os.getenv("SECRET_KEY"), passphrase=os.getenv("PASSPHRASE"))
ws = WeexWebSocket(api_key=os.getenv("API_KEY"), secret_key=os.getenv("SECRET_KEY"), passphrase=os.getenv("PASSPHRASE"), is_private=True)

# Create strategy
strategy = strategies.GridStrategy(
    client=client,
    ws=ws,
    symbol="cmt_btcusdt",
    direction="long",
    grid_count=10,
    size_per_grid=0.01,
    price_range_percent=0.03,
    stop_loss_amount=50.0,
    margin_mode=1
)

# Start strategy
strategy.start()

# Keep running
while strategy.is_running:
    time.sleep(1)

# Stop strategy
strategy.stop()
```

### Short Direction Strategy

```python
strategy = strategies.GridStrategy(
    client=client,
    ws=ws,
    symbol="cmt_btcusdt",
    direction="short",
    grid_count=10,
    size_per_grid=0.01,
    price_range_percent=0.03,
    stop_loss_amount=50.0,
    margin_mode=1
)

strategy.start()
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | str | - | Trading pair symbol (e.g., "cmt_btcusdt") |
| `direction` | str | - | Direction: "long" or "short" |
| `grid_count` | int | - | Number of grid levels |
| `size_per_grid` | float | - | Size per grid level (in contract units) |
| `price_range_percent` | float | 0.03 | Price range percentage (e.g., 0.03 = 3%) |
| `stop_loss_amount` | float | 0.0 | Absolute stop loss amount in USDT (0 = no stop loss) |
| `margin_mode` | int | 1 | Margin mode: 1 (Cross) or 3 (Isolated) |
| `state_file` | str | - | Path to state file for persistence |
| `log_level` | int | - | Logging level (e.g., logging.INFO) |

## How It Works

### Long Direction

1. **Initial Setup**:
   - Calculate price range: [P × (1-3%), P × (1+3%)]
   - Create grid levels using equal percentage distribution
   - Place buy orders at all grid levels

2. **Order Execution**:
   - When a buy order fills → Place sell order at next higher grid price
   - When a sell order fills → Place new buy order at same grid price

3. **Grid Rebuilding**:
   - When price moves outside current range → Cancel unfilled orders
   - Preserve filled position closing orders at their original prices
   - Recalculate grid range around new price
   - Place new orders at empty grid levels

4. **Stop Loss**:
   - Monitor floating PnL from position updates
   - If floating loss ≥ stop_loss_amount → Close all positions, cancel all orders

### Short Direction

Same logic as long, but with sell orders first and buy orders for closing.

## State Persistence

The strategy automatically saves its state to a JSON file. This includes:

- Current price and grid bounds
- All grid levels and their states
- Order IDs and fill information
- Realized PnL
- Start time

When strategy is restarted, it will:
1. Load saved state
2. Verify order status with exchange
3. Resume operation from where it left off

## Requirements

- Python >= 3.8
- weex-sdk >= 1.0.0
- python-dotenv (for examples)

## Running Examples

Make sure to set your API credentials in a `.env` file:

```
API_KEY=your_api_key
SECRET_KEY=your_secret_key
PASSPHRASE=your_passphrase
```

Then run:

```bash
# Long strategy
cd strategies/examples
python run_long_strategy.py

# Short strategy
python run_short_strategy.py
```

## Risk Warning

Grid trading involves significant risk. Past performance does not guarantee future results. Please:

1. Start with small amounts for testing
2. Set appropriate stop losses
3. Monitor your strategy regularly
4. Ensure adequate margin is available

## License

MIT License

## Support

For issues or questions, please visit: https://github.com/discountry/weex-sdk-python/issues
