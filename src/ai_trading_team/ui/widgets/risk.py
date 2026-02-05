"""Risk control status widget."""

from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class RiskWidget(Static):
    """Risk control status and logs widget."""

    DEFAULT_CSS = """
    RiskWidget {
        height: 100%;
        width: 100%;
    }

    .risk-status {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $primary-darken-2;
    }

    .risk-label {
        text-style: bold;
        width: 16;
    }

    .risk-value {
        width: auto;
    }

    .risk-ok {
        color: $success;
    }

    .risk-warning {
        color: $warning;
    }

    .risk-danger {
        color: $error;
    }

    .risk-logs {
        height: 100%;
        overflow-y: auto;
    }

    .risk-log-entry {
        height: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize risk widget."""
        super().__init__(**kwargs)
        self._risk_status: dict = {}
        self._risk_logs: list[dict] = []

    def compose(self) -> ComposeResult:
        """Compose the risk widget."""
        with Vertical():
            # Status section
            with Vertical(classes="risk-status"):
                yield Static("PnL:       --", id="risk-pnl")
                yield Static("Drawdown:  --", id="risk-drawdown")
                yield Static("Stop Loss: --", id="risk-stoploss")
                yield Static("Trailing:  --", id="risk-trailing")

            # Logs section
            yield Static("", id="risk-logs", classes="risk-logs")

    def update_status(
        self,
        pnl_percent: float | None = None,
        drawdown_percent: float | None = None,
        stop_loss_triggered: bool = False,
        trailing_stop_active: bool = False,
        trailing_stop_price: float | None = None,
    ) -> None:
        """Update risk status display.

        Args:
            pnl_percent: Current P&L as percentage
            drawdown_percent: Current drawdown as percentage
            stop_loss_triggered: Whether stop loss is triggered
            trailing_stop_active: Whether trailing stop is active
            trailing_stop_price: Trailing stop trigger price
        """
        # PnL
        if pnl_percent is not None:
            if pnl_percent >= 10:
                pnl_class = "risk-ok"
            elif pnl_percent >= 0:
                pnl_class = "risk-warning"
            else:
                pnl_class = "risk-danger"
            self._update_status_line("risk-pnl", f"PnL:       {pnl_percent:+.2f}%", pnl_class)
        else:
            self._update_status_line("risk-pnl", "PnL:       --", "")

        # Drawdown
        if drawdown_percent is not None:
            if drawdown_percent < 10:
                dd_class = "risk-ok"
            elif drawdown_percent < 20:
                dd_class = "risk-warning"
            else:
                dd_class = "risk-danger"
            self._update_status_line(
                "risk-drawdown", f"Drawdown:  {drawdown_percent:.2f}%", dd_class
            )
        else:
            self._update_status_line("risk-drawdown", "Drawdown:  --", "")

        # Stop Loss
        if stop_loss_triggered:
            self._update_status_line("risk-stoploss", "Stop Loss: TRIGGERED!", "risk-danger")
        else:
            self._update_status_line("risk-stoploss", "Stop Loss: OK", "risk-ok")

        # Trailing Stop
        if trailing_stop_active and trailing_stop_price:
            self._update_status_line(
                "risk-trailing", f"Trailing:  ACTIVE @ {trailing_stop_price:.4f}", "risk-warning"
            )
        elif trailing_stop_active:
            self._update_status_line("risk-trailing", "Trailing:  ACTIVE", "risk-warning")
        else:
            self._update_status_line("risk-trailing", "Trailing:  Inactive", "")

    def _update_status_line(self, widget_id: str, text: str, css_class: str) -> None:
        """Update a status line."""
        try:
            widget = self.query_one(f"#{widget_id}", Static)
            widget.update(text)
            widget.remove_class("risk-ok", "risk-warning", "risk-danger")
            if css_class:
                widget.add_class(css_class)
        except Exception:
            pass

    def add_risk_event(
        self,
        event_type: str,
        message: str,
        timestamp: datetime | None = None,
    ) -> None:
        """Add a risk event to the log.

        Args:
            event_type: Type of risk event
            message: Event message
            timestamp: Optional timestamp
        """
        ts = timestamp or datetime.now()

        event_colors = {
            "STOP_LOSS": "[red]",
            "TRAILING": "[yellow]",
            "TAKE_PROFIT": "[green]",
            "WARNING": "[yellow]",
            "INFO": "[cyan]",
        }

        self._risk_logs.append(
            {
                "timestamp": ts,
                "type": event_type,
                "message": message,
            }
        )

        # Keep only last 20 entries
        if len(self._risk_logs) > 20:
            self._risk_logs = self._risk_logs[-20:]

        # Update display
        logs_widget = self.query_one("#risk-logs", Static)
        log_lines = []
        for log in reversed(self._risk_logs[-10:]):
            log_ts = log["timestamp"].strftime("%H:%M:%S")
            log_type = log["type"]
            log_msg = log["message"][:40]
            c = event_colors.get(log_type.upper(), "[white]")
            log_lines.append(f"[dim]{log_ts}[/] {c}{log_type}[/] {log_msg}")

        logs_widget.update("\n".join(log_lines))

    def clear(self) -> None:
        """Clear all risk data."""
        self._risk_status.clear()
        self._risk_logs.clear()
        self._update_status_line("risk-pnl", "PnL:       --", "")
        self._update_status_line("risk-drawdown", "Drawdown:  --", "")
        self._update_status_line("risk-stoploss", "Stop Loss: --", "")
        self._update_status_line("risk-trailing", "Trailing:  --", "")
        self.query_one("#risk-logs", Static).update("")
