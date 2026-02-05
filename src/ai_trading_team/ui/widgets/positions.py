"""Positions display widget."""

from textual.app import ComposeResult
from textual.widgets import DataTable, Static


class PositionsWidget(Static):
    """Positions display widget."""

    DEFAULT_CSS = """
    PositionsWidget {
        height: auto;
    }

    DataTable {
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the positions widget."""
        yield DataTable(id="positions-table")

    def on_mount(self) -> None:
        """Initialize the positions table."""
        table = self.query_one("#positions-table", DataTable)
        table.add_columns("Symbol", "Side", "Size", "Entry", "PnL", "Liq. Price")
        # Will be updated via execution module subscription

    def update_position(
        self,
        symbol: str,
        side: str,
        size: str,
        entry_price: str,
        pnl: str,
        liquidation_price: str,
    ) -> None:
        """Update or add position data."""
        table = self.query_one("#positions-table", DataTable)
        table.clear()
        table.add_row(symbol, side, size, entry_price, pnl, liquidation_price)

    def clear_positions(self) -> None:
        """Clear all positions."""
        table = self.query_one("#positions-table", DataTable)
        table.clear()
