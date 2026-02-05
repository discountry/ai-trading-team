"""Price chart widget using textual-plotext."""

from collections import deque
from datetime import datetime

from textual.app import ComposeResult
from textual.widgets import Static
from textual_plotext import PlotextPlot


class PriceChartWidget(Static):
    """Real-time price line chart widget."""

    DEFAULT_CSS = """
    PriceChartWidget {
        height: 100%;
        width: 100%;
    }

    PlotextPlot {
        height: 100%;
        width: 100%;
    }
    """

    def __init__(
        self,
        max_points: int = 60,
        **kwargs,
    ) -> None:
        """Initialize price chart.

        Args:
            max_points: Maximum data points to display
        """
        super().__init__(**kwargs)
        self._max_points = max_points
        self._prices: deque[float] = deque(maxlen=max_points)
        self._times: deque[str] = deque(maxlen=max_points)
        self._symbol = "---"

    def compose(self) -> ComposeResult:
        """Compose the chart widget."""
        yield PlotextPlot(id="price-plot")

    def on_mount(self) -> None:
        """Initialize chart on mount."""
        self._render_chart()

    def _render_chart(self) -> None:
        """Render the price chart."""
        plot = self.query_one("#price-plot", PlotextPlot)
        plt = plot.plt

        plt.clear_figure()
        plt.theme("dark")
        plt.title(f"{self._symbol} Price")

        if len(self._prices) > 1:
            x_vals = list(range(len(self._prices)))
            y_vals = list(self._prices)
            plt.plot(x_vals, y_vals, marker="braille")

            # Show price range
            min_price = min(self._prices)
            max_price = max(self._prices)
            if min_price != max_price:
                plt.ylim(min_price * 0.999, max_price * 1.001)

            # Set x-axis labels with time
            if self._times:
                times_list = list(self._times)
                num_labels = min(5, len(times_list))  # Show at most 5 labels
                if num_labels > 1:
                    step = max(1, len(times_list) // (num_labels - 1))
                    x_ticks = [i for i in range(0, len(times_list), step)]
                    x_labels = [times_list[i] for i in x_ticks if i < len(times_list)]
                    # Ensure we include the last point
                    if x_ticks[-1] != len(times_list) - 1:
                        x_ticks.append(len(times_list) - 1)
                        x_labels.append(times_list[-1])
                    plt.xticks(x_ticks, x_labels)

            plt.ylabel("Price")
        else:
            plt.title(f"{self._symbol} - Waiting for data...")

        plot.refresh()

    def update_price(self, price: float, symbol: str = "") -> None:
        """Add a new price point.

        Args:
            price: Current price
            symbol: Trading symbol
        """
        if symbol:
            self._symbol = symbol

        self._prices.append(price)
        self._times.append(datetime.now().strftime("%H:%M:%S"))
        self._render_chart()

    def set_klines(self, klines: list[dict], symbol: str = "") -> None:
        """Set price data from klines.

        Args:
            klines: List of kline dictionaries
            symbol: Trading symbol
        """
        if symbol:
            self._symbol = symbol

        self._prices.clear()
        self._times.clear()

        for kline in klines[-self._max_points :]:
            close = float(kline.get("close", 0))
            if close > 0:
                self._prices.append(close)
                # Use open_time if available
                open_time = kline.get("open_time")
                if open_time:
                    if isinstance(open_time, int):
                        ts = datetime.fromtimestamp(open_time / 1000)
                        self._times.append(ts.strftime("%H:%M"))
                    else:
                        self._times.append(str(open_time)[-8:-3])
                else:
                    self._times.append("")

        self._render_chart()

    def clear(self) -> None:
        """Clear all price data."""
        self._prices.clear()
        self._times.clear()
        self._render_chart()
