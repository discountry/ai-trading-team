"""Session management for state persistence and recovery.

Provides mechanisms to:
1. Persist trading state to disk
2. Recover state after interruptions
3. Reconcile local state with exchange data
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from ai_trading_team.core.types import Side

logger = logging.getLogger(__name__)

# Default session directory
DEFAULT_SESSION_DIR = Path("sessions")


@dataclass
class PositionState:
    """Persisted position state."""

    symbol: str = ""
    side: str = ""  # "long" or "short"
    size: str = "0"  # Decimal as string
    entry_price: str = "0"
    margin: str = "0"
    leverage: int = 1
    entry_time: str | None = None


@dataclass
class SessionState:
    """Complete session state for persistence."""

    # Session metadata
    session_id: str = ""
    symbol: str = ""
    started_at: str = ""
    last_updated: str = ""

    # Strategy state
    strategy_state: str = "idle"
    previous_state: str | None = None

    # Position state
    position: PositionState | None = None

    # Signal state
    last_signal_type: str | None = None
    last_signal_time: str | None = None

    # Trade statistics
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0

    # Risk state
    highest_pnl_percent: str = "0"
    last_profit_threshold: str = "0"

    # Recent operations (last 20)
    operations: list[dict[str, Any]] = field(default_factory=list)

    # Pending signals
    pending_signals: list[dict[str, Any]] = field(default_factory=list)


class SessionManager:
    """Manages session state persistence and recovery.

    Features:
    - Auto-save state periodically and on key events
    - Recovery from disk on startup
    - Reconciliation with exchange data
    """

    def __init__(
        self,
        symbol: str,
        session_dir: Path | str = DEFAULT_SESSION_DIR,
        auto_save_interval: int = 30,
    ) -> None:
        """Initialize session manager.

        Args:
            symbol: Trading symbol
            session_dir: Directory for session files
            auto_save_interval: Seconds between auto-saves
        """
        self._symbol = symbol
        self._session_dir = Path(session_dir)
        self._auto_save_interval = auto_save_interval
        self._session_file = self._session_dir / f"{symbol.lower()}_session.json"
        self._state: SessionState | None = None
        self._dirty = False

        # Ensure session directory exists
        self._session_dir.mkdir(parents=True, exist_ok=True)

    @property
    def session_file(self) -> Path:
        """Get session file path."""
        return self._session_file

    @property
    def has_saved_session(self) -> bool:
        """Check if a saved session exists."""
        return self._session_file.exists()

    def create_session(self, session_id: str | None = None) -> SessionState:
        """Create a new session.

        Args:
            session_id: Optional session ID (auto-generated if not provided)

        Returns:
            New session state
        """
        now = datetime.now().isoformat()
        self._state = SessionState(
            session_id=session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            symbol=self._symbol,
            started_at=now,
            last_updated=now,
        )
        self.save()
        logger.info(f"Created new session: {self._state.session_id}")
        return self._state

    def load_session(self) -> SessionState | None:
        """Load session from disk.

        Returns:
            Loaded session state or None if not found
        """
        if not self._session_file.exists():
            logger.info("No saved session found")
            return None

        try:
            with open(self._session_file) as f:
                data = json.load(f)

            # Reconstruct position if present
            position = None
            if data.get("position"):
                position = PositionState(**data["position"])

            self._state = SessionState(
                session_id=data.get("session_id", ""),
                symbol=data.get("symbol", ""),
                started_at=data.get("started_at", ""),
                last_updated=data.get("last_updated", ""),
                strategy_state=data.get("strategy_state", "idle"),
                previous_state=data.get("previous_state"),
                position=position,
                last_signal_type=data.get("last_signal_type"),
                last_signal_time=data.get("last_signal_time"),
                trades_today=data.get("trades_today", 0),
                wins_today=data.get("wins_today", 0),
                losses_today=data.get("losses_today", 0),
                highest_pnl_percent=data.get("highest_pnl_percent", "0"),
                last_profit_threshold=data.get("last_profit_threshold", "0"),
                operations=data.get("operations", []),
                pending_signals=data.get("pending_signals", []),
            )

            logger.info(
                f"Loaded session: {self._state.session_id}, "
                f"state={self._state.strategy_state}, "
                f"has_position={self._state.position is not None}"
            )
            return self._state

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None

    def save(self) -> bool:
        """Save session to disk.

        Returns:
            True if saved successfully
        """
        if not self._state:
            return False

        try:
            self._state.last_updated = datetime.now().isoformat()

            # Convert to dict for JSON serialization
            data = {
                "session_id": self._state.session_id,
                "symbol": self._state.symbol,
                "started_at": self._state.started_at,
                "last_updated": self._state.last_updated,
                "strategy_state": self._state.strategy_state,
                "previous_state": self._state.previous_state,
                "position": asdict(self._state.position) if self._state.position else None,
                "last_signal_type": self._state.last_signal_type,
                "last_signal_time": self._state.last_signal_time,
                "trades_today": self._state.trades_today,
                "wins_today": self._state.wins_today,
                "losses_today": self._state.losses_today,
                "highest_pnl_percent": self._state.highest_pnl_percent,
                "last_profit_threshold": self._state.last_profit_threshold,
                "operations": self._state.operations[-20:],  # Keep last 20
                "pending_signals": self._state.pending_signals,
            }

            # Write atomically
            temp_file = self._session_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.rename(self._session_file)

            self._dirty = False
            return True

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def update_strategy_state(self, state: str, previous: str | None = None) -> None:
        """Update strategy state.

        Args:
            state: New state
            previous: Previous state
        """
        if self._state:
            self._state.strategy_state = state
            self._state.previous_state = previous
            self._dirty = True

    def update_position(
        self,
        symbol: str,
        side: Side | None,
        size: Decimal,
        entry_price: Decimal,
        margin: Decimal,
        leverage: int,
    ) -> None:
        """Update position state.

        Args:
            symbol: Trading symbol
            side: Position side
            size: Position size
            entry_price: Entry price
            margin: Margin used
            leverage: Leverage
        """
        if not self._state:
            return

        if side is None or size == 0:
            self._state.position = None
        else:
            self._state.position = PositionState(
                symbol=symbol,
                side=side.value,
                size=str(size),
                entry_price=str(entry_price),
                margin=str(margin),
                leverage=leverage,
                entry_time=datetime.now().isoformat(),
            )
        self._dirty = True

    def clear_position(self) -> None:
        """Clear position state."""
        if self._state:
            self._state.position = None
            self._dirty = True

    def add_operation(self, operation: dict[str, Any]) -> None:
        """Add an operation to history.

        Args:
            operation: Operation record
        """
        if self._state:
            self._state.operations.append(operation)
            # Keep only last 20
            if len(self._state.operations) > 20:
                self._state.operations = self._state.operations[-20:]
            self._dirty = True

    def update_trade_stats(self, is_win: bool) -> None:
        """Update trade statistics.

        Args:
            is_win: Whether the trade was profitable
        """
        if self._state:
            self._state.trades_today += 1
            if is_win:
                self._state.wins_today += 1
            else:
                self._state.losses_today += 1
            self._dirty = True

    def update_signal(self, signal_type: str) -> None:
        """Update last signal info.

        Args:
            signal_type: Type of signal
        """
        if self._state:
            self._state.last_signal_type = signal_type
            self._state.last_signal_time = datetime.now().isoformat()
            self._dirty = True

    def update_pnl_tracking(
        self,
        highest_pnl_percent: Decimal,
        last_profit_threshold: Decimal,
    ) -> None:
        """Update P&L tracking state.

        Args:
            highest_pnl_percent: Highest P&L percentage
            last_profit_threshold: Last profit threshold crossed
        """
        if self._state:
            self._state.highest_pnl_percent = str(highest_pnl_percent)
            self._state.last_profit_threshold = str(last_profit_threshold)
            self._dirty = True

    def add_pending_signal(self, signal_data: dict[str, Any]) -> None:
        """Add a pending signal.

        Args:
            signal_data: Signal data to persist
        """
        if self._state:
            self._state.pending_signals.append(signal_data)
            self._dirty = True

    def clear_pending_signals(self) -> None:
        """Clear all pending signals."""
        if self._state:
            self._state.pending_signals = []
            self._dirty = True

    def save_if_dirty(self) -> bool:
        """Save only if state has changed.

        Returns:
            True if saved
        """
        if self._dirty:
            return self.save()
        return False

    def clear_session(self) -> None:
        """Clear the session file."""
        if self._session_file.exists():
            self._session_file.unlink()
            logger.info("Session file cleared")
        self._state = None

    def get_recovery_info(self) -> dict[str, Any]:
        """Get recovery information for logging.

        Returns:
            Recovery info dictionary
        """
        if not self._state:
            return {"has_session": False}

        return {
            "has_session": True,
            "session_id": self._state.session_id,
            "strategy_state": self._state.strategy_state,
            "has_position": self._state.position is not None,
            "position_side": self._state.position.side if self._state.position else None,
            "trades_today": self._state.trades_today,
            "pending_signals": len(self._state.pending_signals),
            "last_updated": self._state.last_updated,
        }

    def get_add_count(self, side: str) -> int:
        """Get the number of ADD operations for a position side.

        Counts "add" operations from the current session's operations list
        for the given side. Resets when position is closed.

        Args:
            side: Position side ("long" or "short")

        Returns:
            Number of add operations for the given side
        """
        if not self._state:
            return 0

        count = 0
        for op in self._state.operations:
            if op.get("action") == "add" and op.get("side") == side:
                count += 1
        return count

    def reset_add_count(self) -> None:
        """Reset add count by clearing add operations from history.

        Called when a position is closed to reset the add counter.
        """
        if not self._state:
            return

        # Remove add operations from history when position closes
        self._state.operations = [op for op in self._state.operations if op.get("action") != "add"]
        self._dirty = True
