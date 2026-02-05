"""Audit manager - coordinates local storage and uploaders."""

import contextlib

from ai_trading_team.audit.models import AgentLog, OrderLog
from ai_trading_team.audit.uploaders.base import LogUploader
from ai_trading_team.audit.writer import LocalLogWriter


class AuditManager:
    """Manages audit logging with local storage and optional uploaders.

    All logs are stored locally. Agent decision logs can be
    uploaded to trading platforms via pluggable uploaders.
    """

    def __init__(self, writer: LocalLogWriter | None = None) -> None:
        self._writer = writer or LocalLogWriter()
        self._uploaders: list[LogUploader] = []

    def add_uploader(self, uploader: LogUploader) -> None:
        """Add a log uploader."""
        self._uploaders.append(uploader)

    def remove_uploader(self, platform_name: str) -> None:
        """Remove uploader by platform name."""
        self._uploaders = [u for u in self._uploaders if u.platform_name != platform_name]

    async def log_agent_decision(self, log: AgentLog) -> None:
        """Log an agent decision.

        Writes to local storage and uploads to all registered uploaders.
        """
        # Always write locally first
        self._writer.write_agent_log(log)

        # Upload to registered platforms
        for uploader in self._uploaders:
            with contextlib.suppress(Exception):
                await uploader.upload(log)

    def log_order_execution(self, log: OrderLog) -> None:
        """Log an order execution.

        Only written locally (not uploaded to platforms).
        """
        self._writer.write_order_log(log)

    @property
    def uploaders(self) -> list[LogUploader]:
        """Get registered uploaders."""
        return list(self._uploaders)
