"""Ticker display widget."""

from textual.app import ComposeResult
from textual.widgets import DataTable, Static


class TickerWidget(Static):
    """Real-time ticker display widget."""

    DEFAULT_CSS = """
    TickerWidget {
        height: auto;
    }

    DataTable {
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the ticker widget."""
        yield DataTable(id="ticker-table")

    def on_mount(self) -> None:
        """Initialize the ticker table."""
        table = self.query_one("#ticker-table", DataTable)
        table.add_columns("Symbol", "Price", "24h Change", "24h High", "24h Low", "Volume")
        # Placeholder data - will be updated via data pool subscription
        table.add_row("BTCUSDT", "--", "--", "--", "--", "--")

    def update_ticker(
        self,
        symbol: str,
        price: str,
        change: str,
        high: str,
        low: str,
        volume: str,
    ) -> None:
        """Update ticker data."""
        table = self.query_one("#ticker-table", DataTable)
        # Clear and re-add (simple approach for single symbol)
        table.clear()
        table.add_row(symbol, price, change, high, low, volume)
