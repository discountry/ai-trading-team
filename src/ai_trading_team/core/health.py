"""Health check module - periodic status reporting."""

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_trading_team.core.data_pool import DataPool
    from ai_trading_team.data.manager import BinanceDataManager
    from ai_trading_team.strategy.state_machine import StrategyStateMachine

logger = logging.getLogger(__name__)


@dataclass
class ModuleStatus:
    """Status of a single module."""

    name: str
    connected: bool
    details: str = ""


class HealthMonitor:
    """Monitors and reports health status of all modules.

    Periodically logs connection status of:
    - Binance WebSocket (market data)
    - WEEX/Exchange WebSocket (account data)
    - Data pool freshness
    - Strategy state machine
    """

    def __init__(
        self,
        data_manager: "BinanceDataManager",
        executor: object,
        data_pool: "DataPool",
        state_machine: "StrategyStateMachine | None" = None,
        interval_seconds: int = 60,
    ) -> None:
        """Initialize health monitor.

        Args:
            data_manager: Binance data manager
            executor: Exchange executor (WEEX/Binance/Mock)
            data_pool: Shared data pool
            state_machine: Trading state machine (optional)
            interval_seconds: Health check interval in seconds (default: 60)
        """
        self._data_manager = data_manager
        self._executor = executor
        self._data_pool = data_pool
        self._state_machine = state_machine
        self._interval = interval_seconds
        self._running = False
        self._task: asyncio.Task | None = None
        self._start_time: datetime | None = None
        self._check_count = 0

    def set_state_machine(self, state_machine: "StrategyStateMachine") -> None:
        """Set the state machine reference (can be set after init)."""
        self._state_machine = state_machine

    async def start(self) -> None:
        """Start periodic health checks."""
        self._running = True
        self._start_time = datetime.now()
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Health monitor started (interval: {self._interval}s)")

    async def stop(self) -> None:
        """Stop health checks."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Health monitor stopped")

    async def _run_loop(self) -> None:
        """Main health check loop."""
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                if self._running:
                    self._check_count += 1
                    self._log_health_status()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    def _log_health_status(self) -> None:
        """Log current health status of all modules."""
        statuses = self._collect_statuses()
        uptime = self._format_uptime()

        # Build status line
        status_parts = []
        all_ok = True

        for status in statuses:
            icon = "✅" if status.connected else "❌"
            if not status.connected:
                all_ok = False
            part = f"{status.name}:{icon}"
            if status.details:
                part += f"({status.details})"
            status_parts.append(part)

        overall = "🟢" if all_ok else "🔴"
        status_line = " | ".join(status_parts)

        logger.info(f"{overall} Health #{self._check_count} [{uptime}] {status_line}")

    def _collect_statuses(self) -> list[ModuleStatus]:
        """Collect status from all modules."""
        statuses = []

        # 1. Binance WebSocket (market data)
        binance_connected = self._data_manager.is_connected
        binance_ready = self._data_manager.is_data_ready
        binance_details = "ready" if binance_ready else "loading"
        statuses.append(
            ModuleStatus(
                name="Binance",
                connected=binance_connected,
                details=binance_details if binance_connected else "disconnected",
            )
        )

        # 2. Exchange executor
        executor_name = type(self._executor).__name__.replace("Executor", "")
        executor_connected = getattr(self._executor, "is_connected", True)
        statuses.append(
            ModuleStatus(
                name=executor_name,
                connected=executor_connected,
            )
        )

        # 3. Data pool freshness
        ticker = self._data_pool.ticker
        if ticker:
            ticker_age = self._get_ticker_age(ticker)
            data_fresh = ticker_age < 30  # Less than 30 seconds old
            statuses.append(
                ModuleStatus(
                    name="Data",
                    connected=data_fresh,
                    details=f"{ticker_age:.0f}s" if ticker_age < 999 else "stale",
                )
            )
        else:
            statuses.append(ModuleStatus(name="Data", connected=False, details="no ticker"))

        # 4. State machine
        if self._state_machine:
            state = self._state_machine.state.value
            statuses.append(
                ModuleStatus(
                    name="State",
                    connected=True,
                    details=state,
                )
            )

        # 5. Position info
        snapshot = self._data_pool.get_snapshot()
        if snapshot.position:
            side = snapshot.position.get("side", "?")
            size = snapshot.position.get("size", 0)
            pnl = snapshot.position.get("unrealized_pnl", 0)
            statuses.append(
                ModuleStatus(
                    name="Pos",
                    connected=True,
                    details=f"{side}:{size:.2f}/PnL:{pnl:.2f}",
                )
            )
        else:
            statuses.append(ModuleStatus(name="Pos", connected=True, details="none"))

        return statuses

    def _get_ticker_age(self, ticker: dict) -> float:
        """Get age of ticker data in seconds."""
        try:
            timestamp = ticker.get("timestamp")
            if timestamp:
                if isinstance(timestamp, datetime):
                    return (datetime.now() - timestamp).total_seconds()
                elif isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp)
                    return (datetime.now() - dt).total_seconds()
            return 999.0
        except Exception:
            return 999.0

    def _format_uptime(self) -> str:
        """Format uptime as human-readable string."""
        if not self._start_time:
            return "0s"

        delta = datetime.now() - self._start_time
        total_seconds = int(delta.total_seconds())

        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h{minutes}m"
        elif minutes > 0:
            return f"{minutes}m{seconds}s"
        else:
            return f"{seconds}s"

    def get_status_summary(self) -> dict:
        """Get status summary as a dictionary (for API/UI use)."""
        statuses = self._collect_statuses()
        return {
            "uptime": self._format_uptime(),
            "check_count": self._check_count,
            "modules": [
                {
                    "name": s.name,
                    "connected": s.connected,
                    "details": s.details,
                }
                for s in statuses
            ],
            "all_healthy": all(s.connected for s in statuses),
        }
