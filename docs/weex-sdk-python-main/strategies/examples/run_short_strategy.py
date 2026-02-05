"""Example: Short direction grid strategy."""

import os
import time
import sys
from dotenv import load_dotenv
from weex_sdk import WeexClient, WeexWebSocket
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from strategies import GridStrategy

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    SYMBOL = "cmt_btcusdt"
    DIRECTION = "short"
    GRID_COUNT = 10
    SIZE_PER_GRID = 0.01
    PRICE_RANGE_PERCENT = 0.03
    STOP_LOSS_AMOUNT = 50.0
    MARGIN_MODE = 1

    client = WeexClient(
        api_key=os.getenv("API_KEY"),
        secret_key=os.getenv("SECRET_KEY"),
        passphrase=os.getenv("PASSPHRASE"),
    )

    ws = WeexWebSocket(
        api_key=os.getenv("API_KEY"),
        secret_key=os.getenv("SECRET_KEY"),
        passphrase=os.getenv("PASSPHRASE"),
        is_private=True,
    )

    strategy = GridStrategy(
        client=client,
        ws=ws,
        symbol=SYMBOL,
        direction=DIRECTION,
        grid_count=GRID_COUNT,
        size_per_grid=SIZE_PER_GRID,
        price_range_percent=PRICE_RANGE_PERCENT,
        stop_loss_amount=STOP_LOSS_AMOUNT,
        margin_mode=MARGIN_MODE,
        state_file=f"short_grid_{SYMBOL}.json",
        log_level=logging.INFO,
    )

    try:
        strategy.start()

        print("\n" + "=" * 60)
        print("Strategy is running. Press Ctrl+C to stop.")
        print("=" * 60 + "\n")

        while strategy.is_running:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received, stopping strategy...")
        strategy.stop()
    except Exception as e:
        print(f"\n\nError: {e}")
        strategy.stop()
    finally:
        print("\nStrategy stopped.")


if __name__ == "__main__":
    main()
