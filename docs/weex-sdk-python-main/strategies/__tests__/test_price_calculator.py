"""Unit tests for PriceCalculator."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from strategies.price_calculator import PriceCalculator


class TestPriceCalculator:
    """Test PriceCalculator methods."""

    def test_calculate_grid_prices_long(self):
        """Test grid price calculation for long direction."""
        current_price = 50000.0
        direction = "long"
        grid_count = 5
        price_range_percent = 0.03
        tick_size = 0.1

        lower, upper, prices = PriceCalculator.calculate_grid_prices(
            current_price, direction, grid_count, price_range_percent, tick_size
        )

        assert lower == pytest.approx(48500.0, rel=0.01)
        assert upper == pytest.approx(51500.0, rel=0.01)
        assert len(prices) == 5
        assert prices == sorted(prices)

    def test_calculate_grid_prices_short(self):
        """Test grid price calculation for short direction."""
        current_price = 50000.0
        direction = "short"
        grid_count = 5
        price_range_percent = 0.03
        tick_size = 0.1

        lower, upper, prices = PriceCalculator.calculate_grid_prices(
            current_price, direction, grid_count, price_range_percent, tick_size
        )

        assert lower == pytest.approx(48500.0, rel=0.01)
        assert upper == pytest.approx(51500.0, rel=0.01)
        assert len(prices) == 5
        assert prices == sorted(prices, reverse=True)

    def test_should_rebuild_grid(self):
        """Test grid rebuild condition."""
        assert PriceCalculator.should_rebuild_grid(52000.0, 51500.0, 48500.0) == True
        assert PriceCalculator.should_rebuild_grid(48000.0, 51500.0, 48500.0) == True
        assert PriceCalculator.should_rebuild_grid(50000.0, 51500.0, 48500.0) == False

    def test_adjust_price_to_step_size(self):
        """Test price adjustment to tick size."""
        assert PriceCalculator.adjust_price_to_step_size(50000.12345, 0.1) == 50000.1
        assert PriceCalculator.adjust_price_to_step_size(50000.16789, 0.01) == 50000.16

    def test_find_next_grid_price_long(self):
        """Test finding next grid price for long direction."""
        grid_prices = [48500.0, 49125.0, 49750.0, 50375.0, 51000.0]

        next_price = PriceCalculator.find_next_grid_price(49125.0, grid_prices, "long")
        assert next_price == 49750.0

        next_price = PriceCalculator.find_next_grid_price(50375.0, grid_prices, "long")
        assert next_price == 51000.0

    def test_find_next_grid_price_short(self):
        """Test finding next grid price for short direction."""
        grid_prices = [48500.0, 49125.0, 49750.0, 50375.0, 51000.0]

        next_price = PriceCalculator.find_next_grid_price(49750.0, grid_prices, "short")
        assert next_price == 49125.0

        next_price = PriceCalculator.find_next_grid_price(49125.0, grid_prices, "short")
        assert next_price == 48500.0
