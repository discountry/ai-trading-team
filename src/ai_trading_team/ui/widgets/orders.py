"""Open orders display widget."""

from textual.app import ComposeResult
from textual.widgets import DataTable, Static


class OrdersWidget(Static):
    """Open orders display widget."""

    DEFAULT_CSS = """
    OrdersWidget {
        height: 100%;
        width: 100%;
    }

    DataTable {
        height: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the orders widget."""
        yield DataTable(id="orders-table")

    def on_mount(self) -> None:
        """Initialize the orders table."""
        table = self.query_one("#orders-table", DataTable)
        table.add_columns("ID", "Side", "Type", "Price", "Size", "Filled", "Status")

    def update_orders(self, orders: list[dict]) -> None:
        """Update orders display.

        Args:
            orders: List of order dictionaries
        """
        table = self.query_one("#orders-table", DataTable)
        table.clear()

        for order in orders:
            order_id = str(order.get("orderId", order.get("order_id", "--")))[-8:]
            side = order.get("side", "--")
            order_type = order.get("type", order.get("order_type", "--"))
            price = order.get("price", 0)
            price_str = f"{float(price):.4f}" if price else "Market"
            size = order.get("origQty", order.get("size", order.get("quantity", "--")))
            filled = order.get("executedQty", order.get("filled", 0))
            status = order.get("status", "--")

            # Color based on side
            side_display = f"[green]{side}[/]" if side.upper() == "BUY" else f"[red]{side}[/]"

            table.add_row(
                order_id,
                side_display,
                order_type,
                price_str,
                str(size),
                str(filled),
                status,
            )

    def clear_orders(self) -> None:
        """Clear all orders."""
        table = self.query_one("#orders-table", DataTable)
        table.clear()
