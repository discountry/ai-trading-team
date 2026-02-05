"""Technical indicators display widget."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class IndicatorsWidget(Static):
    """Technical indicators display widget."""

    DEFAULT_CSS = """
    IndicatorsWidget {
        height: 100%;
        width: 100%;
    }

    .indicator-row {
        height: 1;
        margin-bottom: 0;
    }

    .indicator-label {
        width: 12;
        text-style: bold;
    }

    .indicator-value {
        width: auto;
    }

    .bullish {
        color: $success;
    }

    .bearish {
        color: $error;
    }

    .neutral {
        color: $warning;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize indicators widget."""
        super().__init__(**kwargs)
        self._indicators: dict[str, dict] = {}

    def compose(self) -> ComposeResult:
        """Compose the indicators layout."""
        with Vertical(id="indicators-container"):
            yield Static("MA60:     --", id="ind-ma60", classes="indicator-row")
            yield Static("RSI(14):  --", id="ind-rsi", classes="indicator-row")
            yield Static("MACD:     --", id="ind-macd", classes="indicator-row")
            yield Static("BB:       --", id="ind-bb", classes="indicator-row")
            yield Static("Funding:  --", id="ind-funding", classes="indicator-row")
            yield Static("L/S:      --", id="ind-ls", classes="indicator-row")
            yield Static("OI:       --", id="ind-oi", classes="indicator-row")

    def update_indicators(self, data: dict) -> None:
        """Update all indicators from data.

        Args:
            data: Dictionary containing indicator values
        """
        self._indicators = data

        # MA60
        ma60 = data.get("ma60")
        price = data.get("price", 0)
        if ma60:
            ma_class = "bullish" if price > ma60 else "bearish"
            diff_pct = ((price - ma60) / ma60 * 100) if ma60 > 0 else 0
            sign = "+" if diff_pct > 0 else ""
            self._update_indicator(
                "ind-ma60", f"MA60:     {ma60:.4f} ({sign}{diff_pct:.2f}%)", ma_class
            )
        else:
            self._update_indicator("ind-ma60", "MA60:     --", "neutral")

        # RSI
        rsi = data.get("rsi")
        if rsi is not None:
            if rsi < 30:
                rsi_class = "bullish"
                rsi_status = "Oversold"
            elif rsi > 70:
                rsi_class = "bearish"
                rsi_status = "Overbought"
            else:
                rsi_class = "neutral"
                rsi_status = "Neutral"
            self._update_indicator("ind-rsi", f"RSI(14):  {rsi:.1f} ({rsi_status})", rsi_class)
        else:
            self._update_indicator("ind-rsi", "RSI(14):  --", "neutral")

        # MACD
        macd = data.get("macd")
        macd_signal = data.get("macd_signal")
        if macd is not None and macd_signal is not None:
            macd_class = "bullish" if macd > macd_signal else "bearish"
            hist = macd - macd_signal
            self._update_indicator(
                "ind-macd", f"MACD:     {macd:.6f} (Hist: {hist:+.6f})", macd_class
            )
        else:
            self._update_indicator("ind-macd", "MACD:     --", "neutral")

        # Bollinger Bands
        bb_upper = data.get("bb_upper")
        bb_lower = data.get("bb_lower")
        if bb_upper and bb_lower and price:
            if price > bb_upper:
                bb_class = "bearish"
                bb_status = "Above Upper"
            elif price < bb_lower:
                bb_class = "bullish"
                bb_status = "Below Lower"
            else:
                bb_class = "neutral"
                bb_status = "Inside"
            self._update_indicator(
                "ind-bb", f"BB:       {bb_status} [{bb_lower:.2f}-{bb_upper:.2f}]", bb_class
            )
        else:
            self._update_indicator("ind-bb", "BB:       --", "neutral")

        # Funding Rate (API returns decimal, e.g., -0.00001154 = -0.001154%)
        funding = data.get("funding_rate")
        if funding is not None:
            # Convert to percentage (multiply by 100)
            funding_pct = float(funding) * 100
            # High positive = bearish (longs pay shorts)
            # Negative = bullish (shorts pay longs)
            if funding_pct > 0.03:
                f_class = "bearish"
            elif funding_pct < -0.01:
                f_class = "bullish"
            else:
                f_class = "neutral"
            self._update_indicator("ind-funding", f"Funding:  {funding_pct:+.4f}%", f_class)
        else:
            self._update_indicator("ind-funding", "Funding:  --", "neutral")

        # Long/Short Ratio
        ls_ratio = data.get("ls_ratio")
        if ls_ratio is not None:
            if ls_ratio > 1.5:
                ls_class = "bearish"
            elif ls_ratio < 0.7:
                ls_class = "bullish"
            else:
                ls_class = "neutral"
            self._update_indicator("ind-ls", f"L/S:      {ls_ratio:.4f}", ls_class)
        else:
            self._update_indicator("ind-ls", "L/S:      --", "neutral")

        # Open Interest
        oi = data.get("open_interest")
        if oi is not None:
            oi_display = f"{oi / 1e6:.2f}M" if oi > 1e6 else f"{oi:,.0f}"
            self._update_indicator("ind-oi", f"OI:       {oi_display}", "neutral")
        else:
            self._update_indicator("ind-oi", "OI:       --", "neutral")

    def _update_indicator(self, widget_id: str, text: str, css_class: str) -> None:
        """Update a single indicator display."""
        try:
            widget = self.query_one(f"#{widget_id}", Static)
            widget.update(text)
            widget.remove_class("bullish", "bearish", "neutral")
            widget.add_class(css_class)
        except Exception:
            pass

    def clear(self) -> None:
        """Clear all indicators."""
        self._indicators.clear()
        for ind_id in [
            "ind-ma60",
            "ind-rsi",
            "ind-macd",
            "ind-bb",
            "ind-funding",
            "ind-ls",
            "ind-oi",
        ]:
            self._update_indicator(ind_id, f"{ind_id[4:].upper()}:  --", "neutral")
