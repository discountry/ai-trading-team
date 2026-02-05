"""Minimal orderbook watcher showing top 5 levels and live spread."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Dict, Iterable, Optional, Tuple

from weex_sdk import WeexWebSocket


def _format_number(value: Optional[float | str | int]) -> str:
    """Format a number (which may be string, int, float, or None) to a clean string."""
    if value is None:
        return ""

    # Convert to float if it's a string
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value

    # Format the number
    try:
        text = f"{value:.8f}".rstrip("0").rstrip(".")
        return text if text else "0"
    except (TypeError, ValueError):
        return str(value)


def _parse_level(entry: object) -> Optional[Tuple[float, float]]:
    if isinstance(entry, dict):
        price = entry.get("price")
        size = entry.get("size")
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
        price, size = entry[0], entry[1]
    else:
        return None

    try:
        return float(price), float(size)
    except (TypeError, ValueError):
        return None


class OrderBook:
    def __init__(self, levels: int = 5) -> None:
        self.levels = levels
        self.asks: Dict[float, float] = {}
        self.bids: Dict[float, float] = {}
        self.ready = False
        self.last_render = 0.0

    def apply_snapshot(self, asks: Iterable[object], bids: Iterable[object]) -> None:
        self.asks = self._build_side(asks)
        self.bids = self._build_side(bids)
        self.ready = True

    def apply_changes(self, asks: Iterable[object], bids: Iterable[object]) -> None:
        self._update_side(self.asks, asks)
        self._update_side(self.bids, bids)

    def best_prices(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.asks or not self.bids:
            return None, None
        return min(self.asks), max(self.bids)

    def top_asks(self) -> Iterable[Tuple[float, float]]:
        return sorted(self.asks.items())[: self.levels]

    def top_bids(self) -> Iterable[Tuple[float, float]]:
        return sorted(self.bids.items(), key=lambda item: item[0], reverse=True)[: self.levels]

    def render(self, symbol: str, trades: Optional[list[dict]] = None) -> None:
        if not self.ready:
            return
        now = time.monotonic()
        if now - self.last_render < 0.2:
            return
        self.last_render = now

        best_ask, best_bid = self.best_prices()
        spread_text = (
            _format_number(best_ask - best_bid) if best_ask is not None and best_bid is not None else "N/A"
        )

        asks = list(self.top_asks())
        bids = list(self.top_bids())

        lines = [
            f"Symbol: {symbol}  Spread: {spread_text}",
            f"{'ASK':>14} {'SIZE':>14} | {'BID':>14} {'SIZE':>14}",
        ]
        for idx in range(self.levels):
            ask = asks[idx] if idx < len(asks) else None
            bid = bids[idx] if idx < len(bids) else None
            ask_price = _format_number(ask[0]) if ask else ""
            ask_size = _format_number(ask[1]) if ask else ""
            bid_price = _format_number(bid[0]) if bid else ""
            bid_size = _format_number(bid[1]) if bid else ""
            lines.append(f"{ask_price:>14} {ask_size:>14} | {bid_price:>14} {bid_size:>14}")

        # Display recent trades on the right
        if trades:
            lines[0] += f"{'':20}Recent Trades:"
            lines[1] += f"{'':23} {'TIME':>8} {'PRICE':>12} {'SIZE':>12} {'SIDE':>6}"

            trade_lines = []
            for idx, trade in enumerate(trades[:5]):
                trade_time = trade.get("time", "")
                if trade_time:
                    # Format timestamp to HH:MM:SS
                    try:
                        ts = int(trade_time) / 1000
                        trade_time = time.strftime("%H:%M:%S", time.localtime(ts))
                    except (ValueError, TypeError):
                        trade_time = str(trade_time)

                price = _format_number(trade.get("price"))
                size = _format_number(trade.get("size"))

                buyer_maker = trade.get("buyerMaker", False)
                side_marker = "BUY" if not buyer_maker else "SELL"

                trade_line = f"{' '*23} {trade_time:>8} {price:>12} {size:>12} {side_marker:>6}"
                trade_lines.append(trade_line)

            # Combine order book and trade lines
            max_lines = max(len(lines), len(trade_lines) + 2)  # +2 for header
            output_lines = []
            for i in range(max_lines):
                if i < len(lines):
                    output_line = lines[i]
                else:
                    output_line = ""

                if i >= 2 and i - 2 < len(trade_lines):
                    output_line += trade_lines[i - 2]
                elif i >= len(lines):
                    output_line += ""

                output_lines.append(output_line)

            sys.stdout.write("\033[H\033[J" + "\n".join(output_lines) + "\n")
        else:
            sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n")

        sys.stdout.flush()

    def _build_side(self, levels: Iterable[object]) -> Dict[float, float]:
        side: Dict[float, float] = {}
        for entry in levels:
            parsed = _parse_level(entry)
            if not parsed:
                continue
            price, size = parsed
            if size > 0:
                side[price] = size
        return side

    def _update_side(self, side: Dict[float, float], updates: Iterable[object]) -> None:
        for entry in updates:
            parsed = _parse_level(entry)
            if not parsed:
                continue
            price, size = parsed
            if size <= 0:
                side.pop(price, None)
            else:
                side[price] = size


class TradeHistory:
    """Manages recent trade history."""

    def __init__(self, max_trades: int = 10) -> None:
        self.max_trades = max_trades
        self.trades: list[dict] = []
        self.lock = threading.Lock()

    def add_trade(self, trade: dict) -> None:
        """Add a new trade to the history."""
        with self.lock:
            self.trades.insert(0, trade)
            if len(self.trades) > self.max_trades:
                self.trades = self.trades[: self.max_trades]

    def get_trades(self) -> list[dict]:
        """Get current trades."""
        with self.lock:
            return list(self.trades)


def _run_self_test() -> None:
    book = OrderBook(levels=5)
    book.apply_snapshot(
        asks=[{"price": "101", "size": "1"}, {"price": "102", "size": "2"}],
        bids=[{"price": "99", "size": "3"}, {"price": "98", "size": "4"}],
    )
    best_ask, best_bid = book.best_prices()
    assert best_ask == 101.0
    assert best_bid == 99.0

    book.apply_changes(
        asks=[{"price": "101", "size": "0"}, {"price": "103", "size": "5"}],
        bids=[{"price": "98", "size": "0"}, {"price": "100", "size": "6"}],
    )
    asks = dict(book.top_asks())
    bids = dict(book.top_bids())
    assert 101.0 not in asks
    assert asks[103.0] == 5.0
    assert 98.0 not in bids
    assert bids[100.0] == 6.0
    print("self-test ok")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal orderbook watcher")
    parser.add_argument("--symbol", default="cmt_btcusdt", help="Trading pair symbol")
    parser.add_argument("--levels", type=int, default=5, help="Levels to display")
    parser.add_argument("--self-test", action="store_true", help="Run built-in checks and exit")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _run_self_test()
        return

    book = OrderBook(levels=max(1, args.levels))
    trade_history = TradeHistory(max_trades=5)
    ws = WeexWebSocket(is_private=False)

    def on_depth(message: Dict[str, object]) -> None:
        data_list = message.get("data") if isinstance(message, dict) else None
        if not data_list:
            return
        for payload in data_list:
            if not isinstance(payload, dict):
                continue
            depth_type = payload.get("depthType")
            asks = payload.get("asks", [])
            bids = payload.get("bids", [])
            if depth_type == "SNAPSHOT" or not book.ready:
                book.apply_snapshot(asks, bids)
            else:
                book.apply_changes(asks, bids)
        book.render(args.symbol, trade_history.get_trades())

    def on_trade(message: Dict[str, object]) -> None:
        data_list = message.get("data") if isinstance(message, dict) else None
        if not data_list:
            return
        for trade_data in data_list:
            if isinstance(trade_data, dict):
                trade_history.add_trade(trade_data)
        # Trigger a render update with new trades
        if book.ready:
            book.render(args.symbol, trade_history.get_trades())

    thread = threading.Thread(target=ws.connect, daemon=True)
    thread.start()

    start_time = time.monotonic()
    while not ws.connected:
        if time.monotonic() - start_time > 10:
            ws.close()
            raise RuntimeError("WebSocket connection timeout")
        time.sleep(0.1)

    ws.subscribe_depth(args.symbol, limit=15, callback=on_depth)
    ws.subscribe_trades(args.symbol, callback=on_trade)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        ws.close()


if __name__ == "__main__":
    main()
