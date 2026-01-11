"""AI Agent operation logs widget."""

from collections import deque
from datetime import datetime

from textual.app import ComposeResult
from textual.widgets import RichLog, Static


class AgentLogsWidget(Static):
    """AI Agent operation logs display widget."""

    DEFAULT_CSS = """
    AgentLogsWidget {
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
        """Initialize agent logs widget.

        Args:
            max_logs: Maximum logs to keep in memory
        """
        super().__init__(**kwargs)
        self._max_logs = max_logs
        self._logs: deque[dict] = deque(maxlen=max_logs)

    def compose(self) -> ComposeResult:
        """Compose the logs widget."""
        yield RichLog(id="agent-log", highlight=True, markup=True)

    def add_log(
        self,
        action: str,
        details: str,
        result: str = "",
        timestamp: datetime | None = None,
    ) -> None:
        """Add a new agent log entry.

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
            "SIGNAL": "magenta",
        }
        color = action_colors.get(action.upper(), "white")

        log_entry = {
            "timestamp": ts,
            "action": action,
            "details": details,
            "result": result,
        }
        self._logs.append(log_entry)

        # Format log line
        log_line = f"[dim]{ts_str}[/] [{color}]{action:8}[/] {details}"
        if result:
            log_line += f" [dim]→ {result}[/]"

        log_widget = self.query_one("#agent-log", RichLog)
        log_widget.write(log_line)

    def add_decision(
        self,
        signal_type: str,
        decision: str,
        reason: str,
        timestamp: datetime | None = None,
    ) -> None:
        """Add an agent decision log.

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

        log_widget = self.query_one("#agent-log", RichLog)
        log_widget.write(f"[dim]{ts_str}[/] [magenta]SIGNAL[/]   {signal_type}")
        log_widget.write(f"[dim]{ts_str}[/] [{color}]DECIDE[/]   {decision}")
        log_widget.write(f"[dim]{ts_str}[/] [dim]REASON[/]   {reason[:80]}...")

    def clear(self) -> None:
        """Clear all logs."""
        self._logs.clear()
        log_widget = self.query_one("#agent-log", RichLog)
        log_widget.clear()
