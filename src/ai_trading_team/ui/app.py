"""Main Textual application with real-time data integration."""

import asyncio
import contextlib
from typing import Any

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header

from ai_trading_team.core.data_pool import DataPool
from ai_trading_team.ui.screens.dashboard import DashboardScreen


class TradingApp(App[None]):
    """AI Trading Team TUI Application with real-time data updates."""

    TITLE = "AI Trading Team"
    SUB_TITLE = "Real-time Trading Dashboard"

    CSS = """
    Screen {
        background: $surface;
    }

    Header {
        dock: top;
    }

    Footer {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("d", "switch_screen('dashboard')", "Dashboard"),
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    SCREENS = {
        "dashboard": DashboardScreen,
    }

    def __init__(
        self,
        data_pool: DataPool | None = None,
        symbol: str = "BTCUSDT",
        **kwargs: Any,
    ) -> None:
        """Initialize the trading app.

        Args:
            data_pool: Shared data pool for real-time data
            symbol: Trading symbol
        """
        super().__init__(**kwargs)
        self._data_pool = data_pool
        self._symbol = symbol
        self._update_task: asyncio.Task | None = None
        self._running = False
        self._precision_set = False  # Track if orderbook precision has been set

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.push_screen("dashboard")

        # Start update loop if data pool is available
        if self._data_pool:
            self._running = True
            self._update_task = asyncio.create_task(self._update_loop())

    async def on_unmount(self) -> None:
        """Called when app is unmounted."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._update_task

    async def _update_loop(self) -> None:
        """Periodically update UI with data from data pool."""
        while self._running:
            try:
                await self._update_ui()
            except Exception as e:
                self.log.error(f"UI update error: {e}")
                import traceback

                self.log.error(traceback.format_exc())

            await asyncio.sleep(1.0)  # Update every second

    async def _update_ui(self) -> None:
        """Update all UI widgets with current data."""
        if not self._data_pool:
            return

        snapshot = self._data_pool.get_snapshot()
        dashboard = self.get_screen("dashboard")

        if not isinstance(dashboard, DashboardScreen):
            return

        # Update ticker
        if snapshot.ticker:
            ticker = snapshot.ticker
            # Use symbol precision if available
            pp = snapshot.symbol_info.get("pricePrecision", 4) if snapshot.symbol_info else 4
            dashboard.ticker_widget.update_ticker(
                symbol=self._symbol,
                price=f"{float(ticker.get('last_price', 0)):.{pp}f}",
                change=f"{float(ticker.get('price_change_percent', 0)):+.2f}%",
                high=f"{float(ticker.get('high_price', 0)):.{pp}f}",
                low=f"{float(ticker.get('low_price', 0)):.{pp}f}",
                volume=f"{float(ticker.get('volume', 0)):,.0f}",
            )

        # Update price chart
        if snapshot.klines:
            klines = snapshot.klines.get("1m", [])
            if klines:
                dashboard.chart_widget.set_klines(klines, self._symbol)

        # Set orderbook precision from symbol info (once)
        if not self._precision_set and snapshot.symbol_info:
            price_precision = snapshot.symbol_info.get("pricePrecision", 4)
            qty_precision = snapshot.symbol_info.get("quantityPrecision", 0)
            dashboard.orderbook_widget.set_precision(price_precision, qty_precision)
            self._precision_set = True

        # Update orderbook
        if snapshot.orderbook:
            bids = snapshot.orderbook.get("bids", [])
            asks = snapshot.orderbook.get("asks", [])
            if bids or asks:
                # Convert to tuples
                bid_tuples = [(float(b[0]), float(b[1])) for b in bids[:10] if len(b) >= 2]
                ask_tuples = [(float(a[0]), float(a[1])) for a in asks[:10] if len(a) >= 2]
                dashboard.orderbook_widget.update_orderbook(bid_tuples, ask_tuples)

        # Update indicators
        indicators_data = self._build_indicators_data(snapshot)
        dashboard.indicators_widget.update_indicators(indicators_data)

        # Update positions
        if snapshot.position:
            pos = snapshot.position
            dashboard.positions_widget.update_position(
                symbol=str(pos.get("symbol", self._symbol)),
                side=str(pos.get("side", "--")),
                size=f"{float(pos.get('size', 0)):.4f}",
                entry_price=f"{float(pos.get('entry_price', 0)):.4f}",
                pnl=f"{float(pos.get('unrealized_pnl', 0)):+.2f}",
                liquidation_price=f"{float(pos.get('liquidation_price', 0)):.4f}",
            )
        else:
            dashboard.positions_widget.clear_positions()

        # Update orders
        if snapshot.orders:
            dashboard.orders_widget.update_orders(snapshot.orders)
        else:
            dashboard.orders_widget.clear_orders()

        # Update risk widget
        if snapshot.position:
            pos = snapshot.position
            pnl = float(pos.get("unrealized_pnl", 0))
            margin = float(pos.get("margin", 1))
            pnl_pct = (pnl / margin * 100) if margin > 0 else 0
            dashboard.risk_widget.update_status(
                pnl_percent=pnl_pct,
                drawdown_percent=None,  # Would need to track
                stop_loss_triggered=False,
                trailing_stop_active=False,
            )

    def _build_indicators_data(self, snapshot: Any) -> dict:
        """Build indicators data dict from snapshot.

        Args:
            snapshot: Data snapshot

        Returns:
            Dictionary with indicator values
        """
        data: dict[str, Any] = {}

        # Get current price from ticker
        if snapshot.ticker:
            data["price"] = float(snapshot.ticker.get("last_price", 0))

        # Get klines for calculation
        if snapshot.klines:
            klines = snapshot.klines.get("1h", [])
            closes = [float(k.get("close", 0)) for k in klines] if klines else []

            # MA60
            if len(closes) >= 60:
                data["ma60"] = sum(closes[-60:]) / 60

            # RSI calculation (14 period)
            if len(closes) >= 15:
                recent_closes = closes[-15:]
                changes = [
                    recent_closes[i] - recent_closes[i - 1] for i in range(1, len(recent_closes))
                ]
                gains = [max(0, c) for c in changes]
                losses = [abs(min(0, c)) for c in changes]
                avg_gain = sum(gains) / 14 if gains else 0
                avg_loss = sum(losses) / 14 if losses else 1
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    data["rsi"] = 100 - (100 / (1 + rs))

            # MACD calculation (12, 26, 9)
            if len(closes) >= 26:
                # EMA calculation helper
                def ema(values: list[float], period: int) -> float:
                    if len(values) < period:
                        return sum(values) / len(values) if values else 0
                    multiplier = 2 / (period + 1)
                    ema_val = sum(values[:period]) / period
                    for val in values[period:]:
                        ema_val = (val - ema_val) * multiplier + ema_val
                    return ema_val

                ema12 = ema(closes, 12)
                ema26 = ema(closes, 26)
                macd_line = ema12 - ema26

                # Calculate MACD line history for signal line
                macd_history = []
                for i in range(26, len(closes) + 1):
                    subset = closes[:i]
                    e12 = ema(subset, 12)
                    e26 = ema(subset, 26)
                    macd_history.append(e12 - e26)

                if len(macd_history) >= 9:
                    signal_line = ema(macd_history, 9)
                    data["macd"] = macd_line
                    data["macd_signal"] = signal_line

            # Bollinger Bands (20 period, 2 std)
            if len(closes) >= 20:
                period = 20
                recent = closes[-period:]
                sma = sum(recent) / period
                variance = sum((x - sma) ** 2 for x in recent) / period
                std = variance**0.5
                data["bb_upper"] = sma + 2 * std
                data["bb_lower"] = sma - 2 * std
                data["bb_middle"] = sma

        # Get funding rate
        if snapshot.funding_rate:
            data["funding_rate"] = snapshot.funding_rate.get(
                "funding_rate", snapshot.funding_rate.get("lastFundingRate", 0)
            )

        # Get L/S ratio
        if snapshot.long_short_ratio:
            data["ls_ratio"] = snapshot.long_short_ratio.get("long_short_ratio")

        # Get open interest
        if snapshot.open_interest:
            data["open_interest"] = snapshot.open_interest.get("open_interest")

        return data

    async def action_switch_screen(self, screen_name: str) -> None:
        """Switch to a different screen."""
        if screen_name in self.SCREENS:
            self.switch_screen(screen_name)

    def action_refresh(self) -> None:
        """Force refresh UI."""
        if self._data_pool:
            asyncio.create_task(self._update_ui())

    # Public methods for external updates
    def add_signal(self, signal_type: str, details: str) -> None:
        """Add a signal to the activity log.

        Args:
            signal_type: Type of signal
            details: Signal details
        """
        try:
            dashboard = self.get_screen("dashboard")
            if isinstance(dashboard, DashboardScreen):
                dashboard.activity_widget.add_signal(signal_type, details)
        except Exception:
            pass

    def add_agent_log(self, action: str, details: str, result: str = "") -> None:
        """Add an agent log entry.

        Args:
            action: Action taken
            details: Action details
            result: Result of action
        """
        try:
            dashboard = self.get_screen("dashboard")
            if isinstance(dashboard, DashboardScreen):
                dashboard.activity_widget.add_agent_action(action, details, result)
        except Exception:
            pass

    def add_risk_event(self, event_type: str, message: str) -> None:
        """Add a risk event.

        Args:
            event_type: Type of risk event
            message: Event message
        """
        try:
            dashboard = self.get_screen("dashboard")
            if isinstance(dashboard, DashboardScreen):
                dashboard.activity_widget.add_risk_event(event_type, message)
        except Exception:
            pass


def run_tui(data_pool: DataPool | None = None, symbol: str = "BTCUSDT") -> None:
    """Run the TUI application.

    Args:
        data_pool: Optional data pool for real-time data
        symbol: Trading symbol
    """
    app = TradingApp(data_pool=data_pool, symbol=symbol)
    app.run()


if __name__ == "__main__":
    # Demo mode without data pool
    run_tui()
