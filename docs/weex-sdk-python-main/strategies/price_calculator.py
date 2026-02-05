"""Price calculation utilities for grid strategy."""

import math
from typing import List, Tuple, Optional
from weex_sdk.exceptions import WeexValidationError


class PriceCalculator:
    """Calculate grid prices and handle price-related operations."""

    @staticmethod
    def calculate_grid_prices(
        current_price: float,
        direction: str,
        grid_count: int,
        price_range_percent: float,
        tick_size: float,
    ) -> Tuple[float, float, List[float]]:
        """Calculate grid prices based on current price and range.

        Args:
            current_price: Current market price
            direction: 'long' or 'short'
            grid_count: Number of grid levels
            price_range_percent: Price range percentage (e.g., 0.03 for 3%)
            tick_size: Minimum price increment from contract info

        Returns:
            Tuple of (lower_bound, upper_bound, grid_prices_list)
        """
        if direction not in ["long", "short"]:
            raise WeexValidationError(f"Invalid direction: {direction}")

        lower_bound = current_price * (1 - price_range_percent)
        upper_bound = current_price * (1 + price_range_percent)

        lower_bound = PriceCalculator.adjust_price_to_step_size(lower_bound, tick_size)
        upper_bound = PriceCalculator.adjust_price_to_step_size(upper_bound, tick_size)

        grid_prices = PriceCalculator._generate_percentage_grid(
            lower_bound, upper_bound, grid_count, tick_size
        )

        if direction == "short":
            grid_prices.sort(reverse=True)
        else:
            grid_prices.sort()

        return lower_bound, upper_bound, grid_prices

    @staticmethod
    def _generate_percentage_grid(
        lower_bound: float, upper_bound: float, grid_count: int, tick_size: float
    ) -> List[float]:
        """Generate grid prices with equal percentage distribution."""
        if grid_count < 2:
            return [lower_bound]

        total_percent_change = (upper_bound - lower_bound) / lower_bound
        percent_step = total_percent_change / (grid_count - 1)

        grid_prices = []
        for i in range(grid_count):
            price = lower_bound * (1 + i * percent_step)
            adjusted_price = PriceCalculator.adjust_price_to_step_size(price, tick_size)

            tolerance = tick_size / 2
            is_duplicate = any(abs(adjusted_price - p) < tolerance for p in grid_prices)
            if not is_duplicate:
                grid_prices.append(adjusted_price)

        return grid_prices

    @staticmethod
    def should_rebuild_grid(current_price: float, upper_bound: float, lower_bound: float) -> bool:
        """Check if grid needs to be rebuilt.

        Args:
            current_price: Current market price
            upper_bound: Current grid upper bound
            lower_bound: Current grid lower bound

        Returns:
            True if grid should be rebuilt
        """
        return current_price > upper_bound or current_price < lower_bound

    @staticmethod
    def adjust_price_to_step_size(price: float, tick_size: float) -> float:
        """Adjust price to match tick size requirements.

        Args:
            price: Original price
            tick_size: Minimum price increment

        Returns:
            Adjusted price
        """
        adjusted = math.floor(price / tick_size) * tick_size
        decimal_places = PriceCalculator._get_decimal_places(tick_size)
        return round(adjusted, decimal_places)

    @staticmethod
    def _get_decimal_places(tick_size: float) -> int:
        """Get number of decimal places from tick size."""
        if tick_size >= 1:
            return 0
        str_tick = str(tick_size).rstrip("0").rstrip(".")
        if "." in str_tick:
            return len(str_tick.split(".")[1])
        return 0

    @staticmethod
    def find_next_grid_price(
        current_price: float, grid_prices: List[float], direction: str
    ) -> Optional[float]:
        """Find next grid price for closing position.

        For long: find next higher price
        For short: find next lower price

        Args:
            current_price: Current filled price
            grid_prices: List of all grid prices
            direction: 'long' or 'short'

        Returns:
            Next grid price or None if not found
        """
        sorted_prices = sorted(grid_prices)

        if direction == "long":
            for price in sorted_prices:
                if price > current_price:
                    return price
        else:
            for price in reversed(sorted_prices):
                if price < current_price:
                    return price

        return None
