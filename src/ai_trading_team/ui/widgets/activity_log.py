"""Activity log widget - combines signals and agent logs."""

from collections import deque
from datetime import datetime

from textual.app import ComposeResult
from textual.widgets import RichLog, Static


class ActivityLogWidget(Static):
    """Combined activity log for signals and agent operations."""

    DEFAULT_CSS = """
    ActivityLogWidget {
        height: 100%;
        width: 100%;
    }

    RichLog {
        height: 100%;
        background: $surface;
        border: solid $primary-darken-2;
        scrollbar-gutter: stable;
    }
    """

    def __init__(self, max_logs: int = 100, **kwargs) -> None:
        """Initialize activity log widget.

        Args:
            max_logs: Maximum logs to keep in memory
        """
        super().__init__(**kwargs)
        self._max_logs = max_logs
        self._logs: deque[dict] = deque(maxlen=max_logs)

    def compose(self) -> ComposeResult:
        """Compose the logs widget."""
        yield RichLog(id="activity-log", highlight=True, markup=True)

    def add_signal(
        self,
        signal_type: str,
        details: str,
        timestamp: datetime | None = None,
    ) -> None:
        """Add a strategy signal.

        Args:
            signal_type: Type of signal (RSI, MACD, etc.)
            details: Signal details
            timestamp: Optional timestamp
        """
        ts = timestamp or datetime.now()
        ts_str = ts.strftime("%H:%M:%S")

        self._logs.append(
            {
                "timestamp": ts,
                "type": "signal",
                "signal_type": signal_type,
                "details": details,
            }
        )

        log_widget = self.query_one("#activity-log", RichLog)
        log_widget.write(f"[dim]{ts_str}[/] [magenta]SIGNAL[/]  {signal_type}: {details}")

    def add_agent_action(
        self,
        action: str,
        details: str,
        result: str = "",
        timestamp: datetime | None = None,
    ) -> None:
        """Add an agent action log.

        Args:
            action: Action taken (OPEN, CLOSE, OBSERVE, etc.)
            details: Action details
            result: Result of the action
            timestamp: Optional timestamp
        """
        ts = timestamp or datetime.now()
        ts_str = ts.strftime("%H:%M:%S")

        # Color based on action
        action_colors = {
            "OPEN": "green",
            "ADD": "green",
            "CLOSE": "red",
            "REDUCE": "yellow",
            "OBSERVE": "cyan",
            "ERROR": "red bold",
            "REJECT": "yellow",
        }
        color = action_colors.get(action.upper(), "white")

        self._logs.append(
            {
                "timestamp": ts,
                "type": "action",
                "action": action,
                "details": details,
                "result": result,
            }
        )

        log_line = f"[dim]{ts_str}[/] [{color}]{action:8}[/] {details}"
        if result:
            log_line += f" [dim]→ {result}[/]"

        log_widget = self.query_one("#activity-log", RichLog)
        log_widget.write(log_line)

    def add_decision(
        self,
        signal_type: str,
        decision: str,
        reason: str,
        timestamp: datetime | None = None,
    ) -> None:
        """Add an agent decision with reasoning.

        Args:
            signal_type: Type of signal that triggered decision
            decision: Decision made (action)
            reason: Reasoning for decision
            timestamp: Optional timestamp
        """
        ts = timestamp or datetime.now()
        ts_str = ts.strftime("%H:%M:%S")

        decision_colors = {
            "OPEN": "green",
            "CLOSE": "red",
            "OBSERVE": "cyan",
            "ADD": "green",
            "REDUCE": "yellow",
        }
        color = decision_colors.get(decision.upper(), "white")

        self._logs.append(
            {
                "timestamp": ts,
                "type": "decision",
                "signal_type": signal_type,
                "decision": decision,
                "reason": reason,
            }
        )

        log_widget = self.query_one("#activity-log", RichLog)
        log_widget.write(f"[dim]{ts_str}[/] [magenta]SIGNAL[/]  {signal_type}")
        log_widget.write(f"[dim]{ts_str}[/] [{color}]DECIDE[/]  {decision}")
        # Truncate long reasons
        reason_display = reason[:100] + "..." if len(reason) > 100 else reason
        log_widget.write(f"[dim]{ts_str}[/] [dim]REASON[/]  {reason_display}")

    def add_risk_event(
        self,
        event_type: str,
        message: str,
        timestamp: datetime | None = None,
    ) -> None:
        """Add a risk control event.

        Args:
            event_type: Type of risk event
            message: Event message
            timestamp: Optional timestamp
        """
        ts = timestamp or datetime.now()
        ts_str = ts.strftime("%H:%M:%S")

        risk_colors = {
            "STOP_LOSS": "red bold",
            "TAKE_PROFIT": "green",
            "MAX_DRAWDOWN": "red bold",
            "WARNING": "yellow",
        }
        color = risk_colors.get(event_type.upper(), "yellow")

        self._logs.append(
            {
                "timestamp": ts,
                "type": "risk",
                "event_type": event_type,
                "message": message,
            }
        )

        log_widget = self.query_one("#activity-log", RichLog)
        log_widget.write(f"[dim]{ts_str}[/] [{color}]RISK[/]    {event_type}: {message}")

    def clear(self) -> None:
        """Clear all logs."""
        self._logs.clear()
        log_widget = self.query_one("#activity-log", RichLog)
        log_widget.clear()
