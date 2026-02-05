"""Abstract log uploader interface."""

from abc import ABC, abstractmethod

from ai_trading_team.audit.models import AgentLog


class LogUploader(ABC):
    """Abstract interface for log uploaders.

    Implement this interface to upload agent decision logs
    to different trading platforms.
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Platform name this uploader targets."""
        ...

    @abstractmethod
    async def upload(self, log: AgentLog) -> bool:
        """Upload an agent decision log.

        Args:
            log: Agent decision log to upload

        Returns:
            True if upload successful
        """
        ...

    @abstractmethod
    async def upload_batch(self, logs: list[AgentLog]) -> int:
        """Upload multiple logs in batch.

        Args:
            logs: List of agent logs to upload

        Returns:
            Number of successfully uploaded logs
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if uploader connection is healthy."""
        ...
