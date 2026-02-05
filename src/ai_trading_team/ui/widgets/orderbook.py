"""Order book display widget."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static


class OrderBookWidget(Static):
    """Real-time order book display widget."""

    DEFAULT_CSS = """
    OrderBookWidget {
        height: 100%;
        width: 100%;
    }

    .orderbook-container {
        height: 100%;
        width: 100%;
    }

    .orderbook-side {
        width: 50%;
        height: 100%;
        padding: 0 1;
    }

    .orderbook-header {
        text-style: bold;
        text-align: center;
        height: 1;
        margin-bottom: 1;
    }

    .bid-header {
        color: $success;
    }

    .ask-header {
        color: $error;
    }

    .orderbook-row {
        height: 1;
    }

    .bid-row {
        color: $success;
    }

    .ask-row {
        color: $error;
    }

    .price-col {
        width: 50%;
        text-align: right;
    }

    .qty-col {
        width: 50%;
        text-align: right;
    }

    .spread-row {
        text-align: center;
        color: $warning;
        text-style: bold;
        margin: 1 0;
    }
    """

    def __init__(self, max_levels: int = 8, **kwargs) -> None:
        """Initialize order book widget.

        Args:
            max_levels: Maximum price levels to display per side
        """
        super().__init__(**kwargs)
        self._max_levels = max_levels
        self._bids: list[tuple[float, float]] = []
        self._asks: list[tuple[float, float]] = []
        self._spread = 0.0
        self._price_precision = 4  # Default precision
        self._qty_precision = 0  # Default quantity precision

    def set_precision(self, price_precision: int, qty_precision: int = 0) -> None:
        """Set display precision for prices and quantities.

        Args:
            price_precision: Number of decimal places for prices
            qty_precision: Number of decimal places for quantities
        """
        self._price_precision = price_precision
        self._qty_precision = qty_precision

    def compose(self) -> ComposeResult:
        """Compose the order book layout."""
        with Vertical(classes="orderbook-container"):
            yield Static("Spread: --", id="spread-display", classes="spread-row")
            with Horizontal():
                with Vertical(classes="orderbook-side"):
                    yield Static("BID (Buy)", classes="orderbook-header bid-header")
                    yield Static(id="bids-content")
                with Vertical(classes="orderbook-side"):
                    yield Static("ASK (Sell)", classes="orderbook-header ask-header")
                    yield Static(id="asks-content")

    def update_orderbook(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> None:
        """Update order book data.

        Args:
            bids: List of (price, quantity) tuples for bids
            asks: List of (price, quantity) tuples for asks
        """
        self._bids = sorted(bids, key=lambda x: x[0], reverse=True)[: self._max_levels]
        self._asks = sorted(asks, key=lambda x: x[0])[: self._max_levels]

        pp = self._price_precision
        qp = self._qty_precision

        # Calculate spread
        if self._bids and self._asks:
            best_bid = self._bids[0][0]
            best_ask = self._asks[0][0]
            self._spread = best_ask - best_bid
            spread_pct = (self._spread / best_bid) * 100 if best_bid > 0 else 0
            spread_text = f"Spread: {self._spread:.{pp}f} ({spread_pct:.3f}%)"
        else:
            spread_text = "Spread: --"

        # Update spread display
        spread_display = self.query_one("#spread-display", Static)
        spread_display.update(spread_text)

        # Format bids
        bids_lines = []
        for price, qty in self._bids:
            bids_lines.append(f"{price:>12.{pp}f} │ {qty:>12.{qp}f}")
        bids_content = self.query_one("#bids-content", Static)
        bids_content.update("\n".join(bids_lines) if bids_lines else "No bids")

        # Format asks
        asks_lines = []
        for price, qty in self._asks:
            asks_lines.append(f"{price:>12.{pp}f} │ {qty:>12.{qp}f}")
        asks_content = self.query_one("#asks-content", Static)
        asks_content.update("\n".join(asks_lines) if asks_lines else "No asks")

    def clear(self) -> None:
        """Clear order book data."""
        self._bids.clear()
        self._asks.clear()
        self.query_one("#spread-display", Static).update("Spread: --")
        self.query_one("#bids-content", Static).update("No data")
        self.query_one("#asks-content", Static).update("No data")
