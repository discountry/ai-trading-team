# Weex SDK for Python

![PyPI - Version](https://img.shields.io/pypi/v/weex-sdk)
![PyPI - License](https://img.shields.io/pypi/l/weex-sdk)
![PyPI - Status](https://img.shields.io/pypi/status/weex-sdk)

A comprehensive Python SDK for Weex exchange API, supporting both HTTP REST API and WebSocket connections with automatic reconnection and heartbeat mechanisms.

## Features

- **Complete API Coverage**: Full support for Account, Market, Trade, and AI log upload APIs
- **Sync & Async Support**: Both synchronous and asynchronous clients
- **WebSocket Support**: Real-time market data and account updates with auto-reconnect
- **Robust Error Handling**: Comprehensive exception handling with detailed error messages
- **Type Safety**: Full type hints with strict typing (no `Any` types)
- **Logging**: Built-in logging with configurable levels
- **Auto Reconnection**: WebSocket automatic reconnection with exponential backoff
- **Heartbeat**: Automatic ping/pong handling for WebSocket connections
- **Grid Trading Strategy**: Dynamic single-direction grid trading strategy with automatic range adjustment

## Installation

```bash
pip install weex-sdk
```

Or install from source:

```bash
git clone https://github.com/discountry/weex-sdk-python.git
cd weex-sdk-python
pip install -e .
```

## Quick Start

### Synchronous Usage

```python
from weex_sdk import WeexClient

# Initialize client
client = WeexClient(
    api_key="your_api_key",
    secret_key="your_secret_key",
    passphrase="your_passphrase"
)

# Get account information
account = client.account.get_accounts()
print(account)

# Get market ticker
ticker = client.market.get_ticker("cmt_btcusdt")
print(f"Current price: {ticker['last']}")

# Place an order
order = client.trade.place_order(
    symbol="cmt_btcusdt",
    client_oid="unique_order_id_123",
    size="0.01",
    order_type="0",  # Normal order
    match_price="0",  # Limit price
    price="50000",
    type="1"  # Open long
)
print(f"Order placed: {order['order_id']}")

# Upload AI log
client.ai.upload_ai_log(
    stage="Decision Making",
    model="GPT-5-mini",
    input_data={"prompt": "Analyze BTC trend"},
    output={"signal": "Buy", "confidence": 0.82},
    explanation="AI analyzed market data and generated buy signal",
    order_id=int(order['order_id'])
)
```

### Asynchronous Usage

```python
import asyncio
from weex_sdk import AsyncWeexClient

async def main():
    async with AsyncWeexClient(
        api_key="your_api_key",
        secret_key="your_secret_key",
        passphrase="your_passphrase"
    ) as client:
        # Get account information
        account = await client.account.get_accounts()
        print(account)

        # Get market ticker
        ticker = await client.market.get_ticker("cmt_btcusdt")
        print(f"Current price: {ticker['last']}")

        # Place an order
        order = await client.trade.place_order(
            symbol="cmt_btcusdt",
            client_oid="unique_order_id_123",
            size="0.01",
            order_type="0",
            match_price="0",
            price="50000",
            type="1"
        )
        print(f"Order placed: {order['order_id']}")

asyncio.run(main())
```

### WebSocket Usage

#### Public Channels (Market Data)

```python
from weex_sdk import WeexWebSocket
import json

def on_ticker(data):
    """Handle ticker updates."""
    print(f"Ticker update: {json.dumps(data, indent=2)}")

# Connect to public WebSocket
ws = WeexWebSocket(is_private=False)

# Subscribe to ticker
ws.subscribe_ticker("cmt_btcusdt", callback=on_ticker)

# Connect (runs in background thread)
ws.connect()

# Keep running
import time
time.sleep(60)

# Close connection
ws.close()
```

#### Private Channels (Account Updates)

```python
from weex_sdk import WeexWebSocket

def on_account_update(data):
    """Handle account updates."""
    print(f"Account update: {data}")

def on_order_update(data):
    """Handle order updates."""
    print(f"Order update: {data}")

# Connect to private WebSocket
ws = WeexWebSocket(
    api_key="your_api_key",
    secret_key="your_secret_key",
    passphrase="your_passphrase",
    is_private=True
)

# Subscribe to account and order updates
ws.subscribe_account(callback=on_account_update)
ws.subscribe_order(callback=on_order_update)

# Connect
ws.connect()

# Keep running
import time
time.sleep(60)

ws.close()
```

#### Async WebSocket

```python
import asyncio
from weex_sdk import AsyncWeexWebSocket

async def on_ticker(data):
    """Handle ticker updates."""
    print(f"Ticker: {data}")

async def main():
    ws = AsyncWeexWebSocket(is_private=False)
    await ws.connect()
    await ws.subscribe_ticker("cmt_btcusdt", callback=on_ticker)
    
    # Keep running
    await asyncio.sleep(60)
    await ws.close()

asyncio.run(main())
```

## API Reference

### Account API

```python
# Get all accounts
accounts = client.account.get_accounts()

# Get account for specific coin
account = client.account.get_account("USDT")

# Get assets
assets = client.account.get_assets()

# Get bills
bills = client.account.get_bills(
    coin="USDT",
    symbol="cmt_btcusdt",
    limit=10
)

# Get settings
settings = client.account.get_settings(symbol="cmt_btcusdt")

# Set leverage
client.account.set_leverage(
    symbol="cmt_btcusdt",
    margin_mode=1,  # 1: Cross, 3: Isolated
    long_leverage="10",
    short_leverage="10"
)

# Get positions
positions = client.account.get_all_positions()
position = client.account.get_single_position("cmt_btcusdt")
```

### Market API

```python
# Get server time
time_info = client.market.get_server_time()

# Get contracts
contracts = client.market.get_contracts(symbol="cmt_btcusdt")

# Get market depth
depth = client.market.get_depth("cmt_btcusdt", limit=15)

# Get tickers
tickers = client.market.get_tickers()
ticker = client.market.get_ticker("cmt_btcusdt")

# Get trades
trades = client.market.get_trades("cmt_btcusdt", limit=100)

# Get K-line data
candles = client.market.get_candles(
    symbol="cmt_btcusdt",
    granularity="1m",
    limit=100
)

# Get funding rate
fund_rate = client.market.get_current_fund_rate("cmt_btcusdt")
```

### Trade API

```python
# Place order
order = client.trade.place_order(
    symbol="cmt_btcusdt",
    client_oid="unique_id",
    size="0.01",
    order_type="0",  # 0: Normal, 1: Post-Only, 2: FOK, 3: IOC
    match_price="0",  # 0: Limit, 1: Market
    price="50000",
    type="1"  # 1: Open long, 2: Open short, 3: Close long, 4: Close short
)

# Cancel order
result = client.trade.cancel_order(order_id=order['order_id'])

# Get order details
order_detail = client.trade.get_order_detail(order['order_id'])

# Get current orders
current_orders = client.trade.get_current_orders(symbol="cmt_btcusdt")

# Get order fills
fills = client.trade.get_order_fills(order_id=order['order_id'])

# Close all positions
close_result = client.trade.close_positions(symbol="cmt_btcusdt")

# Cancel all orders
cancel_result = client.trade.cancel_all_orders(
    cancel_order_type="normal",
    symbol="cmt_btcusdt"
)
```

### AI API

```python
# Upload AI log
result = client.ai.upload_ai_log(
    stage="Decision Making",
    model="GPT-5-mini",
    input_data={
        "prompt": "Analyze BTC/USDT price trend for the next 3 hours",
        "data": {
            "RSI_14": 36.8,
            "EMA_20": 68950.4,
            "FundingRate": -0.0021,
            "OpenInterest": 512.3
        }
    },
    output={
        "signal": "Buy",
        "confidence": 0.82,
        "target_price": 69300,
        "reason": "Negative funding + rising open interest implies short squeeze potential."
    },
    explanation="Low RSI and price near the EMA20 suggest weakening downside momentum. Negative funding with rising open interest points to short-side pressure and potential squeeze risk, indicating a bullish bias for BTC over the next three hours.",
    order_id=123456789
)
```

## Error Handling

The SDK provides comprehensive error handling with custom exceptions:

```python
from weex_sdk import (
    WeexAPIException,
    WeexAuthenticationError,
    WeexRateLimitError,
    WeexNetworkError,
    WeexWebSocketError,
    WeexValidationError,
)

try:
    order = client.trade.place_order(...)
except WeexAuthenticationError as e:
    print(f"Authentication failed: {e}")
except WeexRateLimitError as e:
    print(f"Rate limit exceeded. Retry after {e.retry_after} seconds")
except WeexValidationError as e:
    print(f"Invalid parameters: {e}")
except WeexAPIException as e:
    print(f"API error: {e}")
```

## Logging

Configure logging level:

```python
from weex_sdk.logger import setup_logger
import logging

# Set logging level
setup_logger(level=logging.DEBUG)
```

## WebSocket Features

### Auto Reconnection

WebSocket connections automatically reconnect on disconnect with exponential backoff:

```python
ws = WeexWebSocket(
    is_private=False,
    reconnect_attempts=10,  # Max reconnection attempts
    reconnect_delay=1,  # Initial delay (seconds)
    max_reconnect_delay=60  # Max delay (seconds)
)
```

### Heartbeat

The SDK automatically handles ping/pong messages to keep connections alive.

### Subscription Management

Subscriptions are automatically restored after reconnection:

```python
ws.subscribe_ticker("cmt_btcusdt", callback=on_ticker)
ws.subscribe_depth("cmt_btcusdt", limit=15, callback=on_depth)

# After reconnection, all subscriptions are automatically restored
```

## Requirements

- Python >= 3.8
- requests >= 2.31.0
- aiohttp >= 3.9.0
- websocket-client >= 1.6.0
- websockets >= 12.0

## License

MIT License

## Support

For issues and questions:
- GitHub Issues: https://github.com/discountry/weex-sdk-python/issues
- Documentation: https://www.weex.com/api-doc

## Grid Trading Strategy

The SDK includes a comprehensive grid trading strategy implementation with dynamic range adjustment.

### Features

- Dynamic grid range (default ±3%)
- Single direction (long/short)
- Automatic order management
- Stop loss support
- State persistence
- WebSocket integration

### Quick Example

```python
from weex_sdk import WeexClient, WeexWebSocket
import strategies

# Initialize
client = WeexClient(api_key="...", secret_key="...", passphrase="...")
ws = WeexWebSocket(api_key="...", secret_key="...", passphrase="...", is_private=True)

# Create strategy
strategy = strategies.GridStrategy(
    client=client,
    ws=ws,
    symbol="cmt_btcusdt",
    direction="long",
    grid_count=10,
    size_per_grid=0.01,
    price_range_percent=0.03,
    stop_loss_amount=50.0
)

# Start strategy
strategy.start()
```

### Documentation

See [strategies/README.md](strategies/README.md) for detailed documentation and examples.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
