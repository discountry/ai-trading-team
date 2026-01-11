"""Main trading dashboard screen with comprehensive real-time display."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Static

from ai_trading_team.ui.widgets import (
    ActivityLogWidget,
    IndicatorsWidget,
    OrderBookWidget,
    OrdersWidget,
    PositionsWidget,
    PriceChartWidget,
    RiskWidget,
    TickerWidget,
)


class DashboardScreen(Screen[None]):
    """Main trading dashboard screen with real-time data display.

    Layout (3-column grid):
    ┌─────────────────────────────────────────────────────────────┐
    │                      TICKER BAR (3/3)                        │
    ├──────────────────────────────────┬──────────────────────────┤
    │     PRICE CHART (2/3)            │    ORDERBOOK (1/3)       │
    │                                  │                          │
    ├──────────────┬───────────────────┴──────────────────────────┤
    │ INDICATORS   │         POSITIONS / ORDERS (2/3)             │
    │    (1/3)     │                                              │
    ├──────────────┴──────────────────────────┬───────────────────┤
    │     ACTIVITY LOG (2/3)                  │  RISK (1/3)       │
    └─────────────────────────────────────────┴───────────────────┘
    """

    CSS = """
    DashboardScreen {
        layout: grid;
        grid-size: 3 4;
        grid-gutter: 1;
        grid-rows: 3 1fr 1fr 1fr;
        padding: 0 1;
    }

    .panel {
        border: solid $primary-darken-2;
        padding: 0 1;
    }

    .panel-title {
        text-style: bold;
        color: $primary;
        background: $surface-darken-1;
        padding: 0 1;
        height: 1;
    }

    /* Ticker spans full width */
    #ticker-panel {
        column-span: 3;
        height: 3;
    }

    /* Chart panel - 2/3 width */
    #chart-panel {
        column-span: 2;
        row-span: 1;
        min-height: 12;
    }

    /* Orderbook panel - 1/3 width */
    #orderbook-panel {
        column-span: 1;
        row-span: 1;
        min-height: 12;
    }

    /* Indicators panel - 1/3 width */
    #indicators-panel {
        column-span: 1;
        row-span: 1;
        min-height: 10;
    }

    /* Positions and Orders panel - 2/3 width */
    #positions-panel {
        column-span: 2;
        row-span: 1;
        min-height: 10;
    }

    /* Activity log - 2/3 width */
    #activity-panel {
        column-span: 2;
        row-span: 1;
        min-height: 12;
    }

    /* Risk - 1/3 width */
    #risk-panel {
        column-span: 1;
        row-span: 1;
        min-height: 12;
    }

    /* Panel content sizing */
    .panel-content {
        height: 100%;
        width: 100%;
        overflow: auto;
    }

    /* Position/Orders split */
    .split-container {
        height: 100%;
    }

    .split-half {
        height: 50%;
        border-bottom: dashed $primary-darken-3;
    }

    .split-half:last-child {
        border-bottom: none;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        # Row 1: Ticker bar
        with Container(id="ticker-panel", classes="panel"):
            yield Static("Market Data", classes="panel-title")
            yield TickerWidget(id="ticker-widget")

        # Row 2: Price Chart + Orderbook
        with Container(id="chart-panel", classes="panel"):
            yield Static("Price Chart (1m)", classes="panel-title")
            yield PriceChartWidget(id="chart-widget", max_points=60)

        with Container(id="orderbook-panel", classes="panel"):
            yield Static("Order Book", classes="panel-title")
            yield OrderBookWidget(id="orderbook-widget", max_levels=10)

        # Row 3: Indicators + Positions/Orders
        with Container(id="indicators-panel", classes="panel"):
            yield Static("Indicators", classes="panel-title")
            yield IndicatorsWidget(id="indicators-widget")

        with Container(id="positions-panel", classes="panel"), Vertical(classes="split-container"):
            with Vertical(classes="split-half"):
                yield Static("Positions", classes="panel-title")
                yield PositionsWidget(id="positions-widget")
            with Vertical(classes="split-half"):
                yield Static("Open Orders", classes="panel-title")
                yield OrdersWidget(id="orders-widget")

        # Row 4: Activity Log, Risk
        with Container(id="activity-panel", classes="panel"):
            yield Static("Activity Log", classes="panel-title")
            yield ActivityLogWidget(id="activity-widget")

        with Container(id="risk-panel", classes="panel"):
            yield Static("Risk Control", classes="panel-title")
            yield RiskWidget(id="risk-widget")

    # Widget accessor methods for easy updates
    @property
    def ticker_widget(self) -> TickerWidget:
        """Get ticker widget."""
        return self.query_one("#ticker-widget", TickerWidget)

    @property
    def chart_widget(self) -> PriceChartWidget:
        """Get price chart widget."""
        return self.query_one("#chart-widget", PriceChartWidget)

    @property
    def orderbook_widget(self) -> OrderBookWidget:
        """Get orderbook widget."""
        return self.query_one("#orderbook-widget", OrderBookWidget)

    @property
    def indicators_widget(self) -> IndicatorsWidget:
        """Get indicators widget."""
        return self.query_one("#indicators-widget", IndicatorsWidget)

    @property
    def positions_widget(self) -> PositionsWidget:
        """Get positions widget."""
        return self.query_one("#positions-widget", PositionsWidget)

    @property
    def orders_widget(self) -> OrdersWidget:
        """Get orders widget."""
        return self.query_one("#orders-widget", OrdersWidget)

    @property
    def activity_widget(self) -> ActivityLogWidget:
        """Get activity log widget."""
        return self.query_one("#activity-widget", ActivityLogWidget)

    @property
    def risk_widget(self) -> RiskWidget:
        """Get risk widget."""
        return self.query_one("#risk-widget", RiskWidget)
