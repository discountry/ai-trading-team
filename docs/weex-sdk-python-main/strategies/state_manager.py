"""State persistence management for grid strategy."""

import json
import os
from typing import Optional, Dict, List
from .grid_level import GridLevel
from weex_sdk.logger import get_logger

logger = get_logger("grid_strategy.state_manager")


class StateManager:
    """Manages strategy state persistence."""

    def __init__(self, state_file: str):
        """Initialize StateManager.

        Args:
            state_file: Path to state file
        """
        self.state_file = state_file
        logger.info(f"StateManager initialized with state file: {state_file}")

    def save(
        self,
        version: str,
        symbol: str,
        direction: str,
        grid_count: int,
        size_per_grid: float,
        price_range_percent: float,
        stop_loss_amount: float,
        current_price: float,
        upper_bound: float,
        lower_bound: float,
        grid_levels: List[GridLevel],
        realized_pnl: float,
        start_time: int,
        margin_mode: int,
    ) -> bool:
        """Save strategy state to file.

        Args:
            version: State version
            symbol: Trading pair symbol
            direction: 'long' or 'short'
            grid_count: Number of grid levels
            size_per_grid: Size per grid level
            price_range_percent: Price range percentage
            stop_loss_amount: Stop loss amount
            current_price: Current market price
            upper_bound: Grid upper bound
            lower_bound: Grid lower bound
            grid_levels: List of GridLevel objects
            realized_pnl: Total realized PnL
            start_time: Strategy start time
            margin_mode: Margin mode

        Returns:
            True if successful
        """
        try:
            state = {
                "version": version,
                "symbol": symbol,
                "direction": direction,
                "grid_count": grid_count,
                "size_per_grid": size_per_grid,
                "price_range_percent": price_range_percent,
                "stop_loss_amount": stop_loss_amount,
                "current_price": current_price,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "realized_pnl": realized_pnl,
                "grid_levels": [grid.to_dict() for grid in grid_levels],
                "start_time": start_time,
                "last_update": int(__import__("time").time() * 1000),
                "margin_mode": margin_mode,
            }

            temp_file = self.state_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(state, f, indent=2)

            os.rename(temp_file, self.state_file)

            logger.info(f"State saved to {self.state_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def load(self) -> Optional[Dict]:
        """Load strategy state from file.

        Returns:
            Dictionary containing state data or None if not found
        """
        try:
            if not os.path.exists(self.state_file):
                logger.info(f"State file not found: {self.state_file}")
                return None

            with open(self.state_file, "r") as f:
                state = json.load(f)

            grid_levels = [
                GridLevel.from_dict(level_data) for level_data in state.get("grid_levels", [])
            ]
            state["grid_levels"] = grid_levels

            logger.info(f"State loaded from {self.state_file}")
            return state

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def remove(self) -> bool:
        """Remove state file.

        Returns:
            True if successful
        """
        try:
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
                logger.info(f"State file removed: {self.state_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove state file: {e}")
            return False
