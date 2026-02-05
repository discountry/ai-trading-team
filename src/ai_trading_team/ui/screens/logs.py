"""Logs viewer screen."""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Log, Static


class LogsScreen(Screen[None]):
    """Log viewer screen."""

    CSS = """
    LogsScreen {
        layout: vertical;
        padding: 1;
    }

    .panel-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    Log {
        border: solid $primary;
        height: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the logs screen."""
        yield Static("Agent Decision Logs", classes="panel-title")
        yield Log(id="agent-logs")

    def on_mount(self) -> None:
        """Load initial logs."""
        log_widget = self.query_one("#agent-logs", Log)
        log_widget.write_line("Log viewer initialized...")
